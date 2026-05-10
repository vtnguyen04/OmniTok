"""TokenizerTrainer — Stage 1 training loop using Accelerate."""

import datetime
import logging
import os
import time
from collections import deque
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from ..losses.gan import GANLoss
from ..losses.reconstruction import ReconstructionLoss
from ..models.tokenizer import Tokenizer
from ..teachers.multi_teacher import MultiTeacher
from ..teachers.sparse_router import SparseTeacherRouter
from ..utils.logger import OmniTokLogger
from ..utils.metrics import MetricsTracker
from .utils import load_checkpoint, save_checkpoint, update_ema

logger = logging.getLogger(__name__)


class TokenizerTrainer:
    """Stage 1 Tokenizer Trainer with Accelerate."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        teachers: Optional[MultiTeacher],
        alignment_loss: Optional[nn.Module],
        recon_loss: ReconstructionLoss,
        gan_loss: Optional[GANLoss],
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        teacher_router: Optional[SparseTeacherRouter] = None,
        kl_loss: Optional[nn.Module] = None,
        latent_norm_loss: Optional[nn.Module] = None,
        gaussianity_loss: Optional[nn.Module] = None,
        understanding_loss: Optional[nn.Module] = None,
        alignment_weight: float = 1.0,
        kl_weight: float = 1.0,
        understanding_weight: float = 1.0,
        scheduler: Optional[Any] = None,
        disc_optimizer: Optional[torch.optim.Optimizer] = None,
        disc_scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or {}

        self.max_steps = self.config.get("max_steps", 100_000)
        self.log_every = self.config.get("log_every", 100)
        self.save_every = self.config.get("save_every", 10_000)
        self.ema_decay = self.config.get("ema_decay", 0.9999)
        self.alignment_weight = self.config.get("alignment_weight", alignment_weight)
        self.kl_weight = kl_weight
        self.understanding_weight = understanding_weight
        self.use_adaptive_weighting = self.config.get("use_adaptive_weighting", False)
        self.grad_clip = self.config.get("grad_clip", 1.0)
        self.output_dir = self.config.get("output_dir", "outputs")
        self.use_wandb = self.config.get("use_wandb", True)
        self.seed = self.config.get("seed", 42)
        self.log_dir = self.config.get("log_dir", os.path.join(self.output_dir, "logs"))

        self.accelerator = Accelerator(
            mixed_precision=self.config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=self.config.get("grad_accum_steps", 1),
            log_with="wandb" if self.use_wandb else None,
        )

        set_seed(self.seed)

        self.tokenizer = tokenizer
        self.teachers = teachers
        self.ema_tokenizer = deepcopy(tokenizer)
        self.ema_tokenizer.eval()
        for p in self.ema_tokenizer.parameters():
            p.requires_grad = False

        self.alignment_loss = alignment_loss
        self.recon_loss = recon_loss
        self.gan_loss = gan_loss
        self.kl_loss = kl_loss
        self.latent_norm_loss = latent_norm_loss
        self.gaussianity_loss = gaussianity_loss
        self.understanding_loss = understanding_loss
        self.teacher_router = teacher_router

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.disc_optimizer = disc_optimizer
        self.disc_scheduler = disc_scheduler
        self.train_dataloader = train_dataloader

        self.omni_logger = OmniTokLogger(
            name=self.config.get("exp_name", "omnitok"),
            rank=0,
            log_dir=self.log_dir,
            verbose=self.config.get("verbose", False),
        )
        self.metrics = MetricsTracker(window_size=self.config.get("metrics_window", 100))

        self._prepare()

        self.global_step = 0
        self.epoch = 0

        # Timing state
        self._train_start_time: Optional[float] = None
        self._step_times: deque = deque(maxlen=50)
        self._last_step_time: Optional[float] = None

        # Last batch cache for artifact generation
        self._last_images: Optional[torch.Tensor] = None
        self._last_labels: Optional[torch.Tensor] = None

    def _prepare(self) -> None:
        prepared = self.accelerator.prepare(self.tokenizer, self.optimizer, self.train_dataloader)
        self.tokenizer, self.optimizer, self.train_dataloader = prepared

        if self.gan_loss is not None and self.disc_optimizer is not None:
            self.gan_loss, self.disc_optimizer = self.accelerator.prepare(self.gan_loss, self.disc_optimizer)

        if self.alignment_loss is not None:
            self.alignment_loss = self.accelerator.prepare(self.alignment_loss)

        if self.recon_loss is not None:
            self.recon_loss = self.recon_loss.to(self.accelerator.device)

        self.ema_tokenizer = self.ema_tokenizer.to(self.accelerator.device)

        if self.teachers is not None:
            self.teachers = self.teachers.to(self.accelerator.device)
            self.teachers.eval()

        if self.teacher_router is not None:
            self.teacher_router = self.teacher_router.to(self.accelerator.device)

        self.omni_logger.rank = 0 if self.accelerator.is_main_process else 1

    def train(self) -> None:
        """Main training loop."""
        if self.accelerator.is_main_process:
            self.omni_logger.info(f"Starting training for {self.max_steps:,} steps")
            if self.use_wandb:
                exp_name = self.config.get("exp_name", "omnitok")
                stage = self.config.get("stage", "tokenizer")
                notes = self.config.get("notes", f"Running experiment {exp_name}")
                init_kwargs = {
                    "wandb": {
                        "name": exp_name,
                        "tags": ["omnitok", stage, exp_name],
                        "notes": notes,
                    }
                }
                self.accelerator.init_trackers("omnitok", config=self.config, init_kwargs=init_kwargs)

        self.tokenizer.train()

        self.freeze_encoder = self.config.get("freeze_encoder_backbone", False)
        self.unfreeze_step = self.config.get("unfreeze_at_step", -1)

        if self.freeze_encoder and self.global_step < self.unfreeze_step:
            logger.info(f"Freezing encoder backbone until step {self.unfreeze_step} (bottleneck remains trainable)")
            unwrapped_model = self.accelerator.unwrap_model(self.tokenizer)
            if hasattr(unwrapped_model, "freeze_backbone"):
                unwrapped_model.freeze_backbone()
            else:
                for p in unwrapped_model.parameters():
                    p.requires_grad = False

        self._train_start_time = time.time()
        self._last_step_time = time.time()

        data_iter = iter(self.train_dataloader)

        with self.omni_logger.training_progress(self.max_steps, description="Training Tokenizer") as pbar:
            task_id = pbar.add_task("Training Tokenizer", total=self.max_steps, completed=self.global_step)

            while self.global_step < self.max_steps:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    self.epoch += 1
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                images = batch[0].to(self.accelerator.device)
                texts = batch[1] if len(batch) > 2 else None  # ImageTextDataset returns img, text, label
                labels = (
                    batch[-1].to(self.accelerator.device)
                    if len(batch) > 1 and isinstance(batch[-1], torch.Tensor)
                    else None
                )

                if self.accelerator.is_main_process and self.accelerator.sync_gradients:
                    self._last_images = images.detach().cpu()
                    self._last_labels = labels.detach().cpu() if labels is not None else None

                if (
                    hasattr(self.gan_loss, "disc_start")
                    and self.global_step == self.gan_loss.disc_start
                    and self.global_step > 0
                ):
                    if self.accelerator.is_main_process:
                        logger.info(
                            f"GAN Phase starting at step {self.global_step}! "
                            "Resetting optimizer states to prevent momentum shock."
                        )
                    self.optimizer.state.clear()

                loss_dict = self._generator_step(images, texts, labels)

                if self.gan_loss is not None:
                    d_loss_dict = self._discriminator_step(images)
                    loss_dict.update(d_loss_dict)

                if self.accelerator.sync_gradients:
                    # Track step timing
                    now = time.time()
                    self._step_times.append(now - self._last_step_time)
                    self._last_step_time = now

                    update_ema(
                        self.ema_tokenizer,
                        self.accelerator.unwrap_model(self.tokenizer),
                        decay=self.ema_decay,
                    )
                    if self.scheduler is not None:
                        self.scheduler.step()
                    if self.disc_scheduler is not None:
                        self.disc_scheduler.step()

                    self.global_step += 1

                    # Dynamic unfreezing
                    if self.freeze_encoder and self.global_step == self.unfreeze_step:
                        logger.info(f"Unfreezing encoder backbone at step {self.global_step}!")
                        unwrapped_model = self.accelerator.unwrap_model(self.tokenizer)
                        if hasattr(unwrapped_model, "unfreeze_backbone"):
                            unwrapped_model.unfreeze_backbone()
                        else:
                            for p in unwrapped_model.parameters():
                                p.requires_grad = True

                    if self.accelerator.is_main_process:
                        self.metrics.update_dict(loss_dict, step=self.global_step)
                        pbar.update(task_id, advance=1)

                    if self.global_step % self.log_every == 0:
                        self._log(loss_dict)

                    if self.global_step == 1 or self.global_step % self.save_every == 0:
                        self._save()

        self._save()

        if self.use_wandb and self.accelerator.is_main_process:
            self.accelerator.end_training()

    def _log(self, loss_dict: Dict[str, float]) -> None:
        """Print professional one-liner + optional wandb."""
        if not self.accelerator.is_main_process:
            return

        lr = self.optimizer.param_groups[0]["lr"]

        # Step rate (smooth over recent steps)
        if self._step_times:
            avg_step_sec = sum(self._step_times) / len(self._step_times)
            rate = 1.0 / avg_step_sec if avg_step_sec > 0 else 0.0
        else:
            rate = 0.0

        # ETA
        remaining = self.max_steps - self.global_step
        eta_sec = int(remaining / rate) if rate > 0 else 0
        eta_str = str(datetime.timedelta(seconds=eta_sec))

        # GPU memory
        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated(self.accelerator.device) / 1e9
            mem_str = f"  mem [bold]{mem_gb:.1f}[/bold]GB"
        else:
            mem_str = ""

        # Core metrics — always present
        total = loss_dict.get("loss/total", 0.0)
        recon = loss_dict.get("loss/recon_pixel", 0.0)
        lpips = loss_dict.get("loss/recon_perceptual", 0.0)

        # Optional metrics — only show if they exist
        extras = []
        if "loss/align_total" in loss_dict:
            extras.append(f"align=[bold]{loss_dict['loss/align_total']:.4f}[/bold]")
        if "loss/gaussianity" in loss_dict:
            extras.append(f"gauss=[bold]{loss_dict['loss/gaussianity']:.4f}[/bold]")
        if "loss/gan_g" in loss_dict:
            extras.append(f"gan=[bold]{loss_dict['loss/gan_g']:.4f}[/bold]")
        if "router/balance_score" in loss_dict:
            bal = loss_dict["router/balance_score"]
            ent = loss_dict.get("router/entropy_ratio", 0.0)
            bal_color = "green" if bal > 0.8 else ("yellow" if bal > 0.5 else "red")
            # Build compact per-teacher usage string
            usage_parts = []
            for key, val in loss_dict.items():
                if key.startswith("router/usage_"):
                    t_short = key.replace("router/usage_", "")
                    usage_parts.append(f"{t_short}={val:.0%}")
            usage_str = "/".join(usage_parts) if usage_parts else ""
            extras.append(
                f"[{bal_color}]router[/{bal_color}]="
                f"[bold]{bal:.2f}[/bold]"
                f"[dim](ent:{ent:.2f})({usage_str})[/dim]"
            )
        extras_str = "  " + "  ".join(extras) if extras else ""

        # Elapsed
        elapsed = time.time() - self._train_start_time
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))

        # Smoothed total for trend arrow
        smooth = self.metrics.get_smooth("loss/total")
        trend = "↓" if smooth <= total * 1.05 else "↑"

        msg = (
            f"Step {self.global_step}/{self.max_steps} | "
            f"loss={total:.4f} | l1={recon:.4f} | lpips={lpips:.4f} | "
            f"lr={lr:.2e} | rate={rate:.1f}it/s"
        )
        self.omni_logger.info(msg, phase="train")

        self.omni_logger.console.print(
            f"[dim]│[/dim] "
            f"[bold cyan]{self.global_step:>6}[/bold cyan][dim]/{self.max_steps}[/dim]"
            f"  [bold yellow]loss[/bold yellow]=[bold]{total:.4f}[/bold][dim]{trend}[/dim]"
            f"  [cyan]l1[/cyan]=[bold]{recon:.4f}[/bold]"
            f"  [magenta]lpips[/magenta]=[bold]{lpips:.4f}[/bold]"
            f"{extras_str}"
            f"  [green]lr[/green]=[bold]{lr:.2e}[/bold]"
            f"  [dim]{rate:.1f}it/s  elapsed {elapsed_str}  eta {eta_str}{mem_str}[/dim]"
        )

        if self.use_wandb:
            # Include LR in metrics for plotting
            loss_dict["train/lr"] = lr
            self.accelerator.log(
                {**loss_dict, "train/epoch": self.epoch},
                step=self.global_step,
            )

            # Periodic router distribution chart (every 100 steps to avoid WandB overhead)
            if (
                self.teacher_router is not None
                and "router/balance_score" in loss_dict
                and self.global_step % 100 == 0
                and hasattr(self, "wandb_logger")
                and self.wandb_logger is not None
            ):
                teacher_names = self.teachers.teacher_names
                usage = [loss_dict.get(f"router/usage_{t}", 0.0) for t in teacher_names]
                weights = [loss_dict.get(f"router/weight_{t}", 0.0) for t in teacher_names]
                self.wandb_logger.log_router_distribution(
                    teacher_names, usage, weights, self.global_step
                )

    def _calculate_adaptive_weight(
        self, loss_base: torch.Tensor,
        loss_target: torch.Tensor,
        last_layer: nn.Parameter,
    ) -> torch.Tensor:
        """Calculate adaptive weight to balance gradients (VA-VAE / VQGAN style)."""
        if not loss_target.requires_grad:
            return torch.tensor(1.0, device=loss_base.device)

        grad_base_tuple = torch.autograd.grad(loss_base, last_layer, retain_graph=True, allow_unused=True)
        grad_base = grad_base_tuple[0]
        norm_base = torch.norm(grad_base) if grad_base is not None else torch.tensor(1.0, device=loss_base.device)

        grad_target_tuple = torch.autograd.grad(loss_target, last_layer, retain_graph=True, allow_unused=True)
        grad_target = grad_target_tuple[0]

        if grad_target is not None:
            norm_target = torch.norm(grad_target)
            weight = norm_base / (norm_target + 1e-4)
            return torch.clamp(weight, 0.0, 10.0).detach()
        return torch.tensor(1.0, device=loss_base.device)

    def _generator_step(
        self, images: torch.Tensor,
        texts: Optional[list] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        with self.accelerator.accumulate(self.tokenizer):
            mask_ratio = self.config.get("mask_ratio", 0.0)
            output = self.tokenizer(images, return_features=True, mask_ratio=mask_ratio, return_mask=mask_ratio > 0.0)
            recon = output["reconstruction"]
            features = output.get("features", {})

            recon_result = self.recon_loss(images, recon)
            total_loss = recon_result["total"]
            loss_dict = {
                "loss/recon_pixel": recon_result["pixel"].item(),
                "loss/recon_perceptual": recon_result["perceptual"].item(),
                "loss/channel_balance": recon_result.get("channel_balance", torch.tensor(0.0)).item(),
            }

            if self.teachers is not None and self.alignment_loss is not None:
                # Use pre or post bottleneck features for alignment depending on config.
                align_from = self.config.get("alignment", {}).get("align_from", "pre_bottleneck")
                if align_from == "pre_bottleneck":
                    # REPA-E style: 768-dim → projector → teacher_dim gives much richer gradient signal
                    student_tokens = features.get("x_norm_patchtokens_raw", features["x_norm_patchtokens"])
                else:
                    # MAETok style: Align from compressed latents.
                    # Shape of output["latent"] is (B, C, h, w). We need (B, N, C).
                    latents = output["latent"]
                    student_tokens = latents.flatten(2).transpose(1, 2)

                align_total = 0.0
                mask = output.get("mask", None)

                if self.teacher_router is not None:
                    # Sparse Teacher Routing: select top-k teachers per sample
                    routing = self.teacher_router(student_tokens)
                    teacher_features = self.teachers.extract_selected(images, routing.selected_indices)

                    # Compute alignment loss only for selected teachers
                    for t_name, t_feat in teacher_features.items():
                        t_idx = self.teachers.teacher_names.index(t_name)
                        # Per-sample mask: which samples selected this teacher
                        sample_mask = (routing.selected_indices == t_idx).any(dim=-1)  # (B,)
                        if not sample_mask.any():
                            continue

                        # Compute gating weight for this teacher across the batch
                        # For samples that selected this teacher, get their gating weight
                        weight_mask = (routing.selected_indices == t_idx).float()  # (B, top_k)
                        gate_weight = (routing.gating_weights * weight_mask).sum(dim=-1).mean()  # scalar

                        if isinstance(self.alignment_loss, torch.nn.ModuleDict):
                            a_loss = self.alignment_loss[t_name](student_tokens, t_feat, mask=mask)
                        else:
                            a_loss = self.alignment_loss(student_tokens, t_feat, mask=mask)
                        align_total = align_total + gate_weight * a_loss
                        loss_dict[f"loss/align_{t_name}"] = a_loss.item()

                    # Add load balance loss and log router metrics
                    align_total = align_total + routing.load_balance_loss
                    loss_dict["loss/router_balance"] = routing.load_balance_loss.item()

                    router_metrics = self.teacher_router.get_routing_metrics(
                        routing, self.teachers.teacher_names
                    )
                    loss_dict.update(router_metrics)
                else:
                    # Dense alignment: extract all teachers, compute all losses
                    teacher_features = self.teachers.extract_all(images)
                    weights = self.teachers.get_loss_weights()
                    for t_name, t_feat in teacher_features.items():
                        if isinstance(self.alignment_loss, torch.nn.ModuleDict):
                            a_loss = self.alignment_loss[t_name](student_tokens, t_feat, mask=mask)
                        else:
                            a_loss = self.alignment_loss(student_tokens, t_feat, mask=mask)
                        align_total = align_total + weights[t_name] * a_loss
                        loss_dict[f"loss/align_{t_name}"] = a_loss.item()

                    align_total = align_total + self.teachers.get_regularization()

                # ---- ADAPTIVE GRADIENT BALANCING (VA-VAE Style) ----
                # We want to balance the gradients of L_rec and L_align arriving at the encoder backbone.
                last_layer = None
                if hasattr(self.tokenizer, "get_last_shared_layer"):
                    last_layer = self.tokenizer.get_last_shared_layer()

                if last_layer is not None and last_layer.requires_grad and self.use_adaptive_weighting:
                    adaptive_weight = self._calculate_adaptive_weight(total_loss, align_total, last_layer)
                    # Clamp to prevent explosion (max 5.0 to prevent overpowering L1)
                    adaptive_weight = torch.clamp(adaptive_weight, 0.0, 5.0)
                else:
                    adaptive_weight = torch.tensor(1.0, device=total_loss.device)

                # Apply adaptive weight and base weight
                final_align_weight = adaptive_weight * self.alignment_weight
                total_loss = total_loss + final_align_weight * align_total

                loss_dict["loss/align_total"] = align_total.item()
                loss_dict["meta/adaptive_weight"] = (
                    adaptive_weight.item()
                    if isinstance(adaptive_weight, torch.Tensor)
                    else adaptive_weight
                )

            # Understanding Loss (Contrastive/Generative)
            if self.understanding_loss is not None and texts is not None:
                # Assuming understanding_loss handles tokenization of text internally or via a wrapper
                # student_feat could be the global CLS token
                cls_token = features.get("x_norm_clstoken")
                if cls_token is not None:
                    u_loss = self.understanding_loss(cls_token, texts)
                    total_loss = total_loss + self.understanding_weight * u_loss
                    loss_dict["loss/understanding"] = u_loss.item()

            latent = output["latent"]

            if self.kl_loss is not None and "posterior" in output:
                kl_result = self.kl_loss(posterior=output["posterior"])
                total_loss = total_loss + kl_result["total"]
                loss_dict["loss/kl"] = kl_result["total"].item()

            if self.latent_norm_loss is not None:
                if "posterior" in output:
                    ln_result = self.latent_norm_loss(posterior=output["posterior"])
                    if isinstance(ln_result, dict):
                        ln_loss = ln_result["total"]
                    else:
                        ln_loss = ln_result
                else:
                    ln_result = self.latent_norm_loss(latent)
                    if isinstance(ln_result, dict):
                        ln_loss = ln_result["total"]
                    else:
                        ln_loss = ln_result

                total_loss = total_loss + ln_loss
                loss_dict["loss/latent_norm"] = ln_loss.item()

            if self.gaussianity_loss is not None:
                g_loss = self.gaussianity_loss(latent)
                if isinstance(g_loss, dict):
                    loss_val = g_loss.get("total", g_loss.get("loss", next(iter(g_loss.values()))))
                else:
                    loss_val = g_loss
                total_loss = total_loss + loss_val
                loss_dict["loss/gaussianity"] = loss_val.item()

            if self.gan_loss is not None:
                g_result = self.gan_loss.generator_loss(recon, self.global_step)

                # Adaptive GAN Weight (VQGAN style)
                last_layer = None
                if hasattr(self.tokenizer, "get_decoder_last_layer"):
                    last_layer = self.tokenizer.get_decoder_last_layer()

                gan_loss_val = g_result["total"]
                if last_layer is not None and last_layer.requires_grad and self.use_adaptive_weighting:
                    d_weight = self._calculate_adaptive_weight(recon_result["total"], gan_loss_val, last_layer)
                    # The value is clamped to max_gan_weight (default 10.0 instead of 10000) for safety.
                    max_gan_weight = self.config.get("max_gan_weight", 10.0)
                    d_weight = torch.clamp(d_weight, 0.0, max_gan_weight)
                else:
                    d_weight = torch.tensor(1.0, device=total_loss.device)

                # Fade-in the discriminator gradient over 1000 steps to prevent shock collapse
                if hasattr(self.gan_loss, "disc_start"):
                    steps_since_start = max(0, self.global_step - self.gan_loss.disc_start)
                    fade_factor = min(1.0, steps_since_start / 1000.0)
                else:
                    fade_factor = 1.0

                d_weight = d_weight * fade_factor

                # final_gan_weight = d_weight * disc_weight (g_result["total"] is now raw GAN loss)
                disc_weight = getattr(self.gan_loss, "disc_weight", 0.5)
                total_loss = total_loss + d_weight * disc_weight * gan_loss_val
                loss_dict["loss/gan_g"] = g_result["gan"].item()
                loss_dict["meta/adaptive_gan_weight"] = d_weight.item()

            loss_dict["loss/total"] = total_loss.item()

            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients:
                if self.grad_clip > 0:
                    grad_norm = self.accelerator.clip_grad_norm_(self.tokenizer.parameters(), self.grad_clip)
                else:
                    # Calculate without clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), float("inf"))
                loss_dict["loss/grad_norm"] = grad_norm.item()

            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_dict

    def _discriminator_step(self, images: torch.Tensor) -> Dict[str, float]:
        with self.accelerator.accumulate(self.gan_loss):
            with torch.no_grad():
                output = self.tokenizer(images)
                recon = output["reconstruction"]

            d_result = self.gan_loss.discriminator_loss(images, recon, self.global_step)
            if d_result["total"].requires_grad:
                self.accelerator.backward(d_result["total"])
                self.disc_optimizer.step()
            self.disc_optimizer.zero_grad()

        return {
            "loss/disc": d_result["d_loss"].item(),
            "disc/real": d_result["logits_real"].item(),
            "disc/fake": d_result["logits_fake"].item(),
        }

    def _save(self) -> None:
        if not self.accelerator.is_main_process:
            return

        save_dir = os.path.join(self.output_dir, "checkpoints")
        unwrapped = self.accelerator.unwrap_model(self.tokenizer)
        current_loss = self.metrics.get_smooth("loss/total")
        save_checkpoint(
            model=unwrapped,
            ema_model=self.ema_tokenizer,
            optimizer=self.optimizer,
            step=self.global_step,
            epoch=self.epoch,
            save_dir=save_dir,
            loss=current_loss,
        )
        self.metrics.export_json(os.path.join(self.output_dir, "metrics.json"))

        try:
            if hasattr(self, "plot_generator") and self.plot_generator is not None:
                plot_path = self.plot_generator.plot_loss_curves(
                    self.metrics._history,
                    title=f"Training Loss (Step {self.global_step})",
                    filename="loss_curves.png",
                )
            else:
                from omnitok.utils.plots import PlotGenerator
                plotter = PlotGenerator(os.path.join(self.output_dir, "plots"))
                plot_path = plotter.plot_loss_curves(
                    self.metrics._history,
                    title=f"Training Loss (Step {self.global_step})",
                    filename="loss_curves.png",
                )

            if hasattr(self, "wandb_logger") and self.wandb_logger is not None and self.use_wandb:
                self.wandb_logger.log_plot_file(plot_path, "viz/loss_curves", self.global_step)
        except Exception as e:
            logger.warning(f"Failed to plot loss curves: {e}")

        self.omni_logger.success(f"Checkpoint saved  step={self.global_step}  loss={current_loss:.4f}  → {save_dir}")

        if hasattr(self, "artifact_manager") and self._last_images is not None:
            self._save_artifacts()

    def _save_artifacts(self) -> None:
        device = self.accelerator.device
        n_vis = min(8, self._last_images.shape[0])
        images = self._last_images[:n_vis].to(device)

        # Use training model (not EMA) for artifacts: EMA only converges after τ=1/(1-decay) steps.
        # For short runs (< τ steps), EMA is biased toward early (poor) model states.
        vis_model = self.accelerator.unwrap_model(self.tokenizer)
        vis_model.eval()
        try:
            with torch.no_grad():
                out = vis_model(images)
                recon = out["reconstruction"]
            imgs_vis = (images.cpu() * 0.5 + 0.5).clamp(0, 1)
            recon_vis = (recon.cpu() * 0.5 + 0.5).clamp(0, 1)
            self.artifact_manager.save_recon_grid(imgs_vis, recon_vis, step=self.global_step, nrow=min(4, n_vis))
        except Exception as e:
            logger.warning(f"Recon grid failed at step {self.global_step}: {e}")
        finally:
            vis_model.train()

        if self._last_labels is not None:
            try:
                all_imgs = self._last_images.to(device)
                with torch.no_grad():
                    z_all = vis_model.encode(all_imgs)
                z_pooled = z_all.mean(dim=[-2, -1]).cpu()
                self.artifact_manager.save_tsne(z_pooled, self._last_labels, step=self.global_step)
            except Exception as e:
                logger.warning(f"t-SNE failed at step {self.global_step}: {e}")

        try:
            # Save attention map from the first image
            with torch.no_grad():
                out = vis_model(images[0:1], return_features=True)
                # DINOv2 might not expose attentions easily, so we skip if not available
                # or log placeholder
                pass
        except Exception:
            pass

        # Log weight histograms
        if hasattr(self, "wandb_logger") and self.wandb_logger is not None and self.use_wandb:
            try:
                self.wandb_logger.log_weight_histograms(vis_model, self.global_step, prefix="weights")
            except Exception as e:
                logger.warning(f"Weight histograms failed at step {self.global_step}: {e}")

        try:
            attn_map = self._get_encoder_attn_map(images[:1])
            if attn_map is not None:
                self.artifact_manager.save_attn_map(attn_map, step=self.global_step)
        except Exception as e:
            logger.warning(f"Attention map failed at step {self.global_step}: {e}")

        # Log saved artifacts to WandB
        if hasattr(self, "wandb_logger") and self.wandb_logger is not None and self.use_wandb:
            recon_path = os.path.join(
                self.artifact_manager.output_dir, "recon", f"recon_step{self.global_step:07d}.png"
            )
            tsne_path = os.path.join(self.artifact_manager.output_dir, "tsne", f"tsne_step{self.global_step:07d}.png")
            attn_path = os.path.join(self.artifact_manager.output_dir, "attn", f"attn_step{self.global_step:07d}.png")

            if os.path.exists(recon_path):
                self.wandb_logger.log_image_file(
                    recon_path,
                    "viz/recon_grid",
                    self.global_step,
                    caption=f"Reconstruction at Step {self.global_step}",
                )
            if os.path.exists(tsne_path):
                self.wandb_logger.log_image_file(
                    tsne_path, "viz/tsne", self.global_step, caption=f"Latent t-SNE at Step {self.global_step}"
                )
            if os.path.exists(attn_path):
                self.wandb_logger.log_image_file(
                    attn_path,
                    "viz/attention",
                    self.global_step,
                    caption=f"Attention Map at Step {self.global_step}",
                )

    def _get_encoder_attn_map(self, images: torch.Tensor) -> Optional[torch.Tensor]:
        encoder = self.ema_tokenizer.encoder
        if not hasattr(encoder, "blocks") or len(encoder.blocks) == 0:
            return None

        captured: dict = {}

        def qkv_hook(module, inp, out):
            captured["qkv"] = out.detach().float()

        last_attn = encoder.blocks[-1].attn
        handle = last_attn.qkv.register_forward_hook(qkv_hook)
        try:
            with torch.no_grad():
                encoder.forward_features(images)
        finally:
            handle.remove()

        if "qkv" not in captured:
            return None

        qkv = captured["qkv"]
        B, N, _ = qkv.shape
        n_heads = last_attn.num_heads
        head_dim = qkv.shape[-1] // (3 * n_heads)

        qkv = qkv.reshape(B, N, 3, n_heads, head_dim)
        q, k, _ = torch.unbind(qkv, 2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        scale = head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)

        cls_attn = attn[0, :, 0, 1:].cpu()
        n_patches = cls_attn.shape[1]
        h = w = int(n_patches**0.5)
        if h * w != n_patches:
            return None

        return cls_attn.reshape(n_heads, h, w)

    def resume(self, ckpt_path: str) -> None:
        resume_weights_only = self.config.get("resume_weights_only", False)
        state = load_checkpoint(
            ckpt_path,
            model=self.accelerator.unwrap_model(self.tokenizer),
            ema_model=self.ema_tokenizer,
            optimizer=self.optimizer,
            resume_weights_only=resume_weights_only,
        )
        if not resume_weights_only:
            self.global_step = state["step"]
            self.epoch = state["epoch"]
            logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")
        else:
            logger.info("Resumed weights only, starting at step 0")
