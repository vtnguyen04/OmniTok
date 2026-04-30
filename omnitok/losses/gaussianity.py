"""Gaussianity Regularization Loss (L_gauss) — from UNE paper.

Encourages the latent space covariance to approximate the identity matrix,
pushing the latent distribution toward a standard Gaussian N(0, I).

    L_gauss = MSE( Cov(z), I )

where z is the post-bottleneck latent, centered per-batch.

This is a TRAINING LOSS, not an evaluation metric.
For the Gaussianity Score evaluation metric (Anderson-Darling),
see omnitok.evaluation.gaussianity.

Reference:
    UNE paper: "The Universal Normal Embedding"
    Proposed method: 04_proposed_method.md §Loss 4
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..registry import LOSS_REGISTRY

logger = logging.getLogger(__name__)


@LOSS_REGISTRY.register("gaussianity")
class GaussianityLoss(nn.Module):
    """Gaussianity regularization — push Cov(z) → Identity.

    Forces the batch covariance of latent vectors toward I, encouraging
    a whitened, Gaussian-compatible latent space. From UNE theory: good
    latent spaces for generation should be approximately Gaussian, and
    the mapping between any two such spaces is approximately linear.

    Args:
        weight: Loss weight (λ_gauss in total loss).
        mean_penalty: If True, also penalize deviation of batch mean from 0.
    """

    def __init__(self, weight: float = 1e-4, mean_penalty: bool = True) -> None:
        super().__init__()
        self.weight = weight
        self.mean_penalty = mean_penalty

    def forward(self, latents: Tensor) -> dict:
        """Compute Gaussianity regularization loss.

        Args:
            latents: Post-bottleneck latent vectors.
                     Supports:
                     - (B, D): flattened latent vectors
                     - (B, C, h, w): spatial latents → flatten to (B*h*w, C)

        Returns:
            Dict with:
                - 'total': weighted total loss (scalar).
                - 'cov_loss': MSE(Cov(z), I) before weighting.
                - 'mean_loss': MSE(mean(z), 0) before weighting (if enabled).
        """
        # Flatten spatial latents to (N, D)
        if latents.ndim == 4:
            B, C, h, w = latents.shape
            z = latents.permute(0, 2, 3, 1).reshape(-1, C)
        elif latents.ndim == 3:
            B, L, D = latents.shape
            z = latents.reshape(-1, D)
        elif latents.ndim == 2:
            z = latents
        else:
            raise ValueError(f"Expected 2D/3D/4D latent, got shape {latents.shape}")

        N, D = z.shape

        # Center
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Covariance: (D, D) = (z^T z) / N
        cov = (z_centered.T @ z_centered) / max(N - 1, 1)

        # Target: identity matrix
        eye = torch.eye(D, device=z.device, dtype=z.dtype)

        # MSE(Cov, I)
        cov_loss = F.mse_loss(cov, eye)

        result = {
            "cov_loss": cov_loss,
        }

        total = cov_loss

        # Optional: penalize non-zero mean
        if self.mean_penalty:
            mean_loss = (z.mean(dim=0) ** 2).mean()
            total = total + mean_loss
            result["mean_loss"] = mean_loss

        result["total"] = self.weight * total

        return result
