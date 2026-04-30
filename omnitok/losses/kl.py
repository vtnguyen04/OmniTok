"""KL divergence loss for VAE-style tokenizers.

Ported from REPA-E's KL loss computation.
"""

import torch
import torch.nn as nn
from torch import Tensor

from ..registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("kl")
class KLLoss(nn.Module):
    """KL divergence loss for Gaussian VAE latents.

    Computes KL(q(z|x) || N(0,1)) for diagonal Gaussian posterior.

    Args:
        weight: Scaling weight for KL loss.
    """

    def __init__(self, weight: float = 1e-6) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, mean: Tensor, logvar: Tensor) -> dict[str, Tensor]:
        """Compute KL divergence.

        Args:
            mean: Posterior mean (B, D) or (B, N, D).
            logvar: Posterior log-variance, same shape as mean.

        Returns:
            Dict with 'total' and 'kl_raw' losses.
        """
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl = kl.sum() / mean.shape[0]  # Sum over dims, mean over batch
        total = self.weight * kl

        return {"total": total, "kl_raw": kl.detach()}
