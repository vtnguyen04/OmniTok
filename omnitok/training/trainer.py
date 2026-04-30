"""TokenizerTrainer — Stage 1 training loop using Accelerate.

Handles:
- Tokenizer (encoder+decoder) training with reconstruction + alignment losses
- Multi-teacher feature extraction with PHI-S balancing
- GAN discriminator training (optional, after warmup)
- EMA model updates
- WandB logging + checkpoint management
- Multi-GPU via HuggingFace Accelerate

Ported training patterns from REPA-E and continuous_tokenizer.
"""

import logging
import math
import os
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..models.tokenizer import Tokenizer
from ..teachers.multi_teacher import MultiTeacher
from ..losses.reconstruction import ReconstructionLoss
from ..losses.gan import GANLoss
from .utils import update_ema, count_params, save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class TokenizerTrainer:
    """Stage 1 Tokenizer Trainer with Accelerate.

    Orchestrates the full training loop:
    1. Forward pass: tokenizer(images) → reconstruction + features
    2. Reconstruction loss (L1/L2 + LPIPS)
    3. Alignment loss (cosine/relational/prediction with frozen teachers)
    4. GAN loss (optional, after disc_start steps)
    5. EMA update + logging + checkpointing

    Args:
        tokenizer: Tokenizer model (encoder+decoder).
        teachers: MultiTeacher with frozen VFMs.
        alignment_loss: Alignment loss function.
        recon_loss: Reconstruction loss function.
        gan_loss: Optional GAN loss (None to disable).
        train_dataloader: Training data loader.
        optimizer: Optimizer for tokenizer.
        disc_optimizer: Optional optimizer for discriminator.
        config: Training config dict.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        teachers: Optional[MultiTeacher],
        alignment_loss: Optional[nn.Module],
        recon_loss: ReconstructionLoss,
        gan_loss: Optional[GANLoss],
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        disc_optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config or {}

        # Training params
        self.max_steps = self.config.get("max_steps", 100_000)
        self.log_every = self.config.get("log_every", 100)
        self.save_every = self.config.get("save_every", 10_000)
        self.ema_decay = self.config.get("ema_decay", 0.9999)
        self.alignment_weight = self.config.get("alignment_weight", 1.0)
        self.grad_clip = self.config.get("grad_clip", 1.0)
        self.output_dir = self.config.get("output_dir", "outputs")
        self.use_wandb = self.config.get("use_wandb", True)
        self.seed = self.config.get("seed", 42)

        # Setup Accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=self.config.get("grad_accum_steps", 1),
            log_with="wandb" if self.use_wandb else None,
        )

        set_seed(self.seed)

        # Models
        self.tokenizer = tokenizer
        self.teachers = teachers
        self.ema_tokenizer = deepcopy(tokenizer)
        self.ema_tokenizer.eval()
        for p in self.ema_tokenizer.parameters():
            p.requires_grad = False

        # Losses
        self.alignment_loss = alignment_loss
        self.recon_loss = recon_loss
        self.gan_loss = gan_loss

        # Optimizers
        self.optimizer = optimizer
        self.disc_optimizer = disc_optimizer

        # DataLoader
        self.train_dataloader = train_dataloader

        # Prepare with Accelerate
        self._prepare()

        # State
        self.global_step = 0
        self.epoch = 0

    def _prepare(self) -> None:
        """Prepare models and data with Accelerator."""
        prepared = self.accelerator.prepare(
            self.tokenizer, self.optimizer, self.train_dataloader
        )
        self.tokenizer, self.optimizer, self.train_dataloader = prepared

        if self.gan_loss is not None and self.disc_optimizer is not None:
            self.gan_loss, self.disc_optimizer = self.accelerator.prepare(
                self.gan_loss, self.disc_optimizer
            )

        if self.alignment_loss is not None:
            self.alignment_loss = self.accelerator.prepare(self.alignment_loss)

        # EMA stays on device but not wrapped
        self.ema_tokenizer = self.ema_tokenizer.to(self.accelerator.device)

        # Teachers on device (frozen, not wrapped)
        if self.teachers is not None:
            self.teachers = self.teachers.to(self.accelerator.device)
            self.teachers.eval()

    def train(self) -> None:
        """Main training loop."""
        if self.accelerator.is_main_process:
            logger.info(f"Starting training for {self.max_steps} steps")
            params = count_params(self.accelerator.unwrap_model(self.tokenizer))
            logger.info(f"Tokenizer params: {params['trainable_M']:.1f}M trainable")

            if self.use_wandb:
                self.accelerator.init_trackers(
                    "omnitok-tokenizer",
                    config=self.config,
                )

        self.tokenizer.train()
        data_iter = iter(self.train_dataloader)

        pbar = tqdm(
            range(self.global_step, self.max_steps),
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )

        for step in pbar:
            # Get batch (loop over epochs)
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            images = batch[0]  # (B, 3, H, W) in [0, 1]

            # --- Generator step ---
            loss_dict = self._generator_step(images)

            # --- Discriminator step ---
            if self.gan_loss is not None:
                d_loss_dict = self._discriminator_step(images)
                loss_dict.update(d_loss_dict)

            # --- EMA update ---
            update_ema(
                self.ema_tokenizer,
                self.accelerator.unwrap_model(self.tokenizer),
                decay=self.ema_decay,
            )

            self.global_step = step + 1

            # --- Logging ---
            if self.global_step % self.log_every == 0:
                self._log(loss_dict, pbar)

            # --- Checkpointing ---
            if self.global_step % self.save_every == 0:
                self._save()

        # Final save
        self._save()

        if self.use_wandb and self.accelerator.is_main_process:
            self.accelerator.end_training()

    def _generator_step(self, images: torch.Tensor) -> Dict[str, float]:
        """Single generator training step."""
        with self.accelerator.accumulate(self.tokenizer):
            # Forward through tokenizer
            output = self.tokenizer(images, return_features=True)
            recon = output["reconstruction"]
            features = output["features"]

            # Reconstruction loss
            recon_result = self.recon_loss(images, recon)
            total_loss = recon_result["total"]
            loss_dict = {
                "loss/recon_pixel": recon_result["pixel"].item(),
                "loss/recon_perceptual": recon_result["perceptual"].item(),
            }

            # Alignment loss (with teachers)
            if self.teachers is not None and self.alignment_loss is not None:
                teacher_features = self.teachers.extract_all(images)
                weights = self.teachers.get_loss_weights()

                # Student features = encoder patch tokens (before bottleneck)
                student_tokens = features["x_norm_patchtokens"]

                align_total = torch.zeros(1, device=images.device)
                for t_name, t_feat in teacher_features.items():
                    a_loss = self.alignment_loss(student_tokens, t_feat)
                    align_total += weights[t_name] * a_loss
                    loss_dict[f"loss/align_{t_name}"] = a_loss.item()

                # Add PHI-S regularization
                align_total += self.teachers.get_regularization()
                total_loss += self.alignment_weight * align_total

                loss_dict["loss/align_total"] = align_total.item()

            # GAN generator loss
            if self.gan_loss is not None:
                g_result = self.gan_loss.generator_loss(recon, self.global_step)
                total_loss += g_result["total"]
                loss_dict["loss/gan_g"] = g_result["gan"].item()

            loss_dict["loss/total"] = total_loss.item()

            # Backward
            self.accelerator.backward(total_loss)
            if self.grad_clip > 0:
                self.accelerator.clip_grad_norm_(self.tokenizer.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_dict

    def _discriminator_step(self, images: torch.Tensor) -> Dict[str, float]:
        """Single discriminator training step."""
        with self.accelerator.accumulate(self.gan_loss):
            with torch.no_grad():
                output = self.tokenizer(images)
                recon = output["reconstruction"]

            d_result = self.gan_loss.discriminator_loss(images, recon, self.global_step)

            self.accelerator.backward(d_result["total"])
            self.disc_optimizer.step()
            self.disc_optimizer.zero_grad()

        return {
            "loss/disc": d_result["d_loss"].item(),
            "disc/real": d_result["logits_real"].item(),
            "disc/fake": d_result["logits_fake"].item(),
        }

    def _log(self, loss_dict: Dict[str, float], pbar: tqdm) -> None:
        """Log metrics to wandb and progress bar."""
        if self.accelerator.is_main_process:
            # Progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict.get('loss/total', 0):.4f}",
                "recon": f"{loss_dict.get('loss/recon_pixel', 0):.4f}",
                "step": self.global_step,
                "epoch": self.epoch,
            })

            # WandB
            if self.use_wandb:
                self.accelerator.log(
                    {**loss_dict, "train/epoch": self.epoch, "train/lr": self.optimizer.param_groups[0]["lr"]},
                    step=self.global_step,
                )

    def _save(self) -> None:
        """Save checkpoint (main process only)."""
        if self.accelerator.is_main_process:
            save_dir = os.path.join(self.output_dir, "checkpoints")
            unwrapped = self.accelerator.unwrap_model(self.tokenizer)
            save_checkpoint(
                model=unwrapped,
                ema_model=self.ema_tokenizer,
                optimizer=self.optimizer,
                step=self.global_step,
                epoch=self.epoch,
                save_dir=save_dir,
            )

    def resume(self, ckpt_path: str) -> None:
        """Resume training from checkpoint.

        Args:
            ckpt_path: Path to checkpoint file.
        """
        state = load_checkpoint(
            ckpt_path,
            model=self.accelerator.unwrap_model(self.tokenizer),
            ema_model=self.ema_tokenizer,
            optimizer=self.optimizer,
        )
        self.global_step = state["step"]
        self.epoch = state["epoch"]
        logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")
