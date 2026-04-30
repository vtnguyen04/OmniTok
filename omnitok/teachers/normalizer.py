"""Feature normalization utilities for multi-teacher alignment.

Ported from RADIO's feature whitening approach. Different VFMs produce features
with different statistics — normalizing them is essential for stable alignment.
"""

import torch
import torch.nn as nn
from torch import Tensor


class FeatureNormalizer(nn.Module):
    """Running mean/variance normalizer for teacher features.

    Whitens features to zero-mean, unit-variance using exponential moving
    average statistics. Essential when aligning with multiple teachers that
    have different feature scales.

    Args:
        feature_dim: Dimension of input features.
        momentum: EMA momentum for updating running stats (default: 0.1).
        eps: Small constant for numerical stability.
    """

    def __init__(self, feature_dim: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(feature_dim))
        self.register_buffer("running_var", torch.ones(feature_dim))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize features using running statistics.

        Args:
            x: Input features (B, N, D) or (B, D).

        Returns:
            Normalized features with same shape.
        """
        if self.training:
            self._update_stats(x)

        return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)

    @torch.no_grad()
    def _update_stats(self, x: Tensor) -> None:
        """Update running mean and variance with EMA."""
        # Flatten to (total_tokens, D) for stats computation
        if x.ndim == 3:
            flat = x.reshape(-1, x.shape[-1])
        else:
            flat = x

        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)

        self.num_batches_tracked += 1

        if self.num_batches_tracked == 1:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var)
        else:
            self.running_mean.mul_(1 - self.momentum).add_(batch_mean, alpha=self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(batch_var, alpha=self.momentum)


class ProjectedNormalizer(nn.Module):
    """Feature normalizer with learnable linear projection.

    Projects teacher features to a common dimension before normalization.
    Useful when teachers have different feature dimensions.

    Args:
        in_dim: Input feature dimension (teacher-specific).
        out_dim: Output feature dimension (common alignment space).
        momentum: EMA momentum for normalizer.
    """

    def __init__(self, in_dim: int, out_dim: int, momentum: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.normalizer = FeatureNormalizer(out_dim, momentum=momentum)

    def forward(self, x: Tensor) -> Tensor:
        """Project and normalize features.

        Args:
            x: Input features (B, N, D_in).

        Returns:
            Projected and normalized features (B, N, D_out).
        """
        return self.normalizer(self.proj(x))
