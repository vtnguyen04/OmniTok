from typing import Any, Dict, Tuple

import torch.nn as nn
from torch import Tensor

from omnitok.registry import BOTTLENECK_REGISTRY


class BaseBottleneck(nn.Module):
    """Base class for all bottlenecks."""
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        raise NotImplementedError

@BOTTLENECK_REGISTRY.register("linear")
class LinearBottleneck(BaseBottleneck):
    """Standard linear bottleneck (AutoEncoder style).

    Args:
        in_dim: Input feature dimension.
        latent_dim: Target bottleneck dimension.
    """
    def __init__(self, in_dim: int, latent_dim: int, **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(in_dim, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        # x: (B, N, C) or (B, C, H, W)
        if x.ndim == 4:
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, c)
            z = self.proj(x)
            z = z.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        else:
            z = self.proj(x)
        return z, {}

@BOTTLENECK_REGISTRY.register("variational")
class VariationalBottleneck(BaseBottleneck):
    """Variational bottleneck (VAE style).

    Predicts mu and logvar, then samples using reparameterization trick.

    Args:
        in_dim: Input feature dimension.
        latent_dim: Latent dimension (mu/logvar each have this dim).
    """
    def __init__(self, in_dim: int, latent_dim: int, **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(in_dim, 2 * latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        # x: (B, N, C) or (B, C, H, W)
        if x.ndim == 4:
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, c)
            moments = self.proj(x)
            moments = moments.reshape(b, h, w, 2 * self.latent_dim).permute(0, 3, 1, 2)
        else:
            moments = self.proj(x)

        from omnitok.models.distributions import DiagonalGaussianDistribution
        posterior = DiagonalGaussianDistribution(moments)
        z = posterior.sample() if self.training else posterior.mode()

        return z, {"posterior": posterior}

@BOTTLENECK_REGISTRY.register("identity")
class IdentityBottleneck(BaseBottleneck):
    """Identity bottleneck (No dimensionality reduction)."""
    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        return x, {}
