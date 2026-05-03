"""Latent normalization loss — ported from AIOTok.

Soft constraint that keeps latent z diffusion-friendly (mean≈0, std≈1)
WITHOUT using KL divergence, which can crush semantic content early in training.

"""

import torch.nn as nn
from torch import Tensor

from omnitok.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("latent_norm")
class LatentNormLoss(nn.Module):
    """Soft constraint: mean(z) ≈ 0, std(z) ≈ 1.

    Keeps latent distribution diffusion-friendly without KL divergence.
    Much gentler than KL — does not crush semantic content early in training.

    L = |mean(z)| + |std(z) - 1|

    Args:
        weight: Scaling weight for the loss.
    """

    def __init__(self, weight: float = 0.01) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, z: Tensor = None, posterior=None) -> Tensor:
        """Compute latent norm loss.

        Args:
            z: Latent tensor, any shape (B, C, H, W) or (B, N, D).
            posterior: Optional DiagonalGaussianDistribution object.

        Returns:
            Weighted scalar loss.
        """
        if posterior is not None:
            z = posterior.sample()

        raw = z.mean().abs() + (z.std() - 1.0).abs()
        return self.weight * raw
