"""Projector heads — map student features into teacher feature space.

Ablation study (from UNE paper):
    P1: LinearProjector   — UNE hypothesis: linear map is sufficient
    P2: MLPProjector(2)   — middle ground
    P3: MLPProjector(3)   — REPA-E default (may be over-parameterized per UNE)

All projectors are registered in PROJECTOR_REGISTRY.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from omnitok.registry import PROJECTOR_REGISTRY


class BaseProjector(nn.Module):
    """Abstract base for all projector heads.

    Args:
        in_dim: Input (student) feature dimension.
        out_dim: Output (teacher) feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        """Project student features toward teacher space.

        Args:
            x: Student features (..., in_dim).

        Returns:
            Projected features (..., out_dim).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(in={self.in_dim}, out={self.out_dim})"


@PROJECTOR_REGISTRY.register("linear")
class LinearProjector(BaseProjector):
    """Single linear projection without bias.

    UNE paper argues linear maps are sufficient because all good latent spaces
    are related by a linear transformation of a shared Gaussian.
    Equivalent to VA-VAE's Conv1×1 projection.

    Args:
        in_dim: Student feature dimension.
        out_dim: Teacher feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(in_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


@PROJECTOR_REGISTRY.register("mlp2")
class MLP2Projector(BaseProjector):
    """2-layer MLP projector with SiLU activation.

    Intermediate between Linear and REPA-E's 3-layer MLP.

    Args:
        in_dim: Student feature dimension.
        out_dim: Teacher feature dimension.
        hidden_dim: Hidden layer dimension. Defaults to max(in_dim, out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0) -> None:
        super().__init__(in_dim, out_dim)
        hidden = hidden_dim or max(in_dim, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, out_dim, bias=False),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


@PROJECTOR_REGISTRY.register("mlp3")
class MLP3Projector(BaseProjector):
    """3-layer MLP projector with SiLU, ported from REPA-E.

    Reference:
        REPA-E models/sit.py — ProjectionModel class (lines ~330-360)

    Args:
        in_dim: Student feature dimension.
        out_dim: Teacher feature dimension.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0) -> None:
        super().__init__(in_dim, out_dim)
        hidden = hidden_dim or max(in_dim, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, out_dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


@PROJECTOR_REGISTRY.register("identity")
class IdentityProjector(BaseProjector):
    """No-op projector — used when in_dim == out_dim and no projection needed."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(in_dim, out_dim)
        if in_dim != out_dim:
            raise ValueError(
                f"IdentityProjector requires in_dim == out_dim, got {in_dim} != {out_dim}"
            )

    def forward(self, x: Tensor) -> Tensor:
        return x


def build_projector(
    proj_type: str,
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 0,
) -> BaseProjector:
    """Factory function for projectors.

    Args:
        proj_type: One of "linear", "mlp2", "mlp3", "identity".
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        hidden_dim: Hidden dim for MLP variants (0 = auto).

    Returns:
        Projector instance.
    """
    if proj_type == "identity":
        return PROJECTOR_REGISTRY.build("identity", in_dim=in_dim, out_dim=out_dim)
    if proj_type == "linear":
        return PROJECTOR_REGISTRY.build("linear", in_dim=in_dim, out_dim=out_dim)
    if proj_type in ("mlp2", "mlp3"):
        return PROJECTOR_REGISTRY.build(
            proj_type, in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim
        )
    raise ValueError(
        f"Unknown projector '{proj_type}'. "
        f"Available: {PROJECTOR_REGISTRY.available()}"
    )
