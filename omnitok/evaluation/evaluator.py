"""Unified TokenizerEvaluator — runs all evaluation metrics in one call.

Combines:
- rFID: Reconstruction quality (generation proxy)
- PSNR: Pixel-level fidelity
- Linear Probe Accuracy: Semantic quality of encoder features
- Gaussianity Score: UNE hypothesis validation

Reference:
    LightningDiT/evaluate_tokenizer.py — PSNR + LPIPS pattern
    VTP/tools/test_linear_probing_hf.py — feature extraction pattern
"""

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .gaussianity import GaussianityEvaluator
from .linear_probe import LinearProbeEvaluator
from .rfid import RFIDEvaluator

logger = logging.getLogger(__name__)


def _compute_psnr(real: Tensor, recon: Tensor) -> float:
    """MSE-based PSNR in dB. Images should be in [-1, 1]."""
    mse = torch.mean((real.float() - recon.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    # PSNR relative to the full range [0, 255] convention
    return 20 * math.log10(255.0) - 10 * math.log10(mse * (127.5**2))


class TokenizerEvaluator:
    """All-in-one evaluator for OmniTok tokenizer ablations.

    Runs rFID, PSNR, optional LinearProbe, and optional Gaussianity Score
    in a single `evaluate()` call. Each metric can be individually disabled
    to save compute during quick ablation checks.

    Args:
        run_rfid: Compute Reconstruction FID.
        run_psnr: Compute PSNR (cheap, always on by default).
        run_linear_probe: Compute linear probe accuracy (needs val labels).
        run_gaussianity: Compute Gaussianity Score on encoder latents.
        rfid_max_images: Max images for rFID (None = all).
        gaussianity_max_samples: Max samples for Gaussianity test.
        linear_probe_max_train: Max train samples for linear probe.
        device: Compute device (auto-detects if None).
    """

    def __init__(
        self,
        run_rfid: bool = True,
        run_psnr: bool = True,
        run_linear_probe: bool = False,
        run_gaussianity: bool = True,
        rfid_max_images: Optional[int] = 5000,
        gaussianity_max_samples: int = 50000,
        linear_probe_epochs: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        self.run_rfid = run_rfid
        self.run_psnr = run_psnr
        self.run_linear_probe = run_linear_probe
        self.run_gaussianity = run_gaussianity
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if run_rfid:
            self.rfid_eval = RFIDEvaluator(device=self.device, max_images=rfid_max_images)
        if run_gaussianity:
            self.gauss_eval = GaussianityEvaluator(max_samples=gaussianity_max_samples)
        if run_linear_probe:
            self.probe_eval = LinearProbeEvaluator(epochs=linear_probe_epochs)

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        train_loader: Optional[DataLoader] = None,
        n_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run all enabled metrics.

        Args:
            model: Tokenizer. Must have .encode(x) → z and .decode(z) → recon.
                   For LinearProbe, encode output is used directly as features.
            val_loader: Validation DataLoader yielding (images, labels) or (images,).
                        images ∈ [-1, 1].
            train_loader: Required only for LinearProbe (provides labeled train data).
            n_batches: Limit batches (useful for quick sanity checks).

        Returns:
            Merged dict of all metric results.
        """
        model.eval()
        metrics: Dict[str, float] = {}

        # --- Collect real/recon pairs for rFID + PSNR ---
        if self.run_rfid or self.run_psnr or self.run_gaussianity:
            real_imgs, recon_imgs, latents = self._collect_recon(model, val_loader, n_batches)

            if self.run_psnr:
                psnr = _compute_psnr(real_imgs, recon_imgs)
                metrics["psnr"] = psnr
                logger.info(f"PSNR: {psnr:.2f} dB")

            if self.run_rfid:
                rfid_metrics = self.rfid_eval.compute(real_imgs, recon_imgs)
                metrics.update(rfid_metrics)

            if self.run_gaussianity:
                gauss_metrics = self.gauss_eval.compute(latents)
                metrics.update(gauss_metrics)

        # --- Linear Probe ---
        if self.run_linear_probe:
            if train_loader is None:
                logger.warning("LinearProbe requested but train_loader is None — skipping.")
            else:
                probe_metrics = self.probe_eval.compute_from_model(
                    encoder=_EncoderWrapper(model),
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=self.device,
                )
                metrics.update(probe_metrics)

        return metrics

    def _collect_recon(
        self,
        model: nn.Module,
        loader: DataLoader,
        n_batches: Optional[int],
    ):
        """Returns (real_imgs, recon_imgs, latents) as CPU tensors."""
        real_list, recon_list, latent_list = [], [], []
        total = 0

        for i, batch in enumerate(loader):
            if n_batches is not None and i >= n_batches:
                break

            images = batch[0].to(self.device)
            z = model.encode(images)
            recon = model.decode(z)

            # Flatten spatial latent for Gaussianity: (B, C, h, w) → (B*h*w, C)
            if z.ndim == 4:
                B, C, h, w = z.shape
                flat_z = z.permute(0, 2, 3, 1).reshape(-1, C)
            else:
                flat_z = z.reshape(z.shape[0], -1)

            real_list.append(images.cpu())
            recon_list.append(recon.cpu())
            latent_list.append(flat_z.cpu())
            total += images.shape[0]

        if total == 0:
            raise RuntimeError("No samples collected — empty dataloader?")

        return (
            torch.cat(real_list, dim=0),
            torch.cat(recon_list, dim=0),
            torch.cat(latent_list, dim=0),
        )


class _EncoderWrapper(nn.Module):
    """Wraps tokenizer.encode() as a standalone module for LinearProbe."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model.encode(x)
