"""Projector heads — map student features into teacher feature space.

Ablation study (from UNE paper):
    P1: LinearProjector   — UNE hypothesis: linear map is sufficient
    P2: MLPProjector(2)   — middle ground
    P3: MLPProjector(3)   — REPA-E default (may be over-parameterized per UNE)

All projectors are registered in PROJECTOR_REGISTRY.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from omnitok.registry import PROJECTOR_REGISTRY


class BaseProjector(nn.Module):
    """Abstract base for all projector heads.

    Args:
        in_dim: Input (student) feature dimension.
        out_dim: Output (teacher) feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Project student features toward teacher space.

        Args:
            x: Student features (..., in_dim).
            mask: Optional token mask.
            kwargs: Additional arguments like teacher_cond.

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

    def __init__(self, in_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__(in_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
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

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0, **kwargs) -> None:
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

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
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

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0, **kwargs) -> None:
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

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        return self.proj(x)


@PROJECTOR_REGISTRY.register("identity")
class IdentityProjector(BaseProjector):
    """No-op projector — used when in_dim == out_dim and no projection needed."""

    def __init__(self, in_dim: int, out_dim: int, **kwargs) -> None:
        super().__init__(in_dim, out_dim)
        if in_dim != out_dim:
            raise ValueError(f"IdentityProjector requires in_dim == out_dim, got {in_dim} != {out_dim}")

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
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
    if proj_type in ("mlp2", "mlp3", "moe"):
        return PROJECTOR_REGISTRY.build(proj_type, in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown projector '{proj_type}'. Available: {PROJECTOR_REGISTRY.available()}")


@PROJECTOR_REGISTRY.register("moe")
class MoEProjector(BaseProjector):
    """MoE Alignment Projector for filtering multi-modal teacher gradients.

    Acts as a massive 'gradient router' during training. Uses teacher conditioning
    to dynamically route student features into specific MLP experts before projecting
    them to the teacher's feature space.

    Args:
        in_dim: Student feature dimension.
        out_dim: Teacher feature dimension.
        hidden_dim: Hidden layer dimension for experts.
        num_experts: Number of routing experts.
        top_k: Number of experts to activate per token.
        teacher_dim: Dimension of teacher conditioning signal.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 0,
        num_experts: int = 4,
        top_k: int = 2,
        teacher_dim: int = 0,
        **kwargs
    ) -> None:
        super().__init__(in_dim, out_dim)
        self.num_experts = num_experts
        self.top_k = top_k
        self.teacher_dim = teacher_dim

        hidden = hidden_dim or max(in_dim, out_dim)

        from omnitok.models.bottleneck.moe_crossattn import MoEGating
        self.gating = MoEGating(in_dim, num_experts, top_k, teacher_dim)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden, bias=True),
                nn.SiLU(),
                nn.Linear(hidden, hidden, bias=True),
                nn.SiLU(),
                nn.Linear(hidden, out_dim, bias=True),
            ) for _ in range(num_experts)
        ])

        self.norm = nn.LayerNorm(in_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for expert in self.experts:
            for m in expert:
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        teacher_cond = kwargs.get("teacher_cond", None)
        x_normed = self.norm(x)

        weights, indices = self.gating(x_normed, teacher_cond)

        # Dense expert execution
        expert_outputs = torch.stack(
            [expert(x_normed) for expert in self.experts], dim=2
        )  # (B, N, num_experts, out_dim)

        out_d = self.out_dim
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, out_d)
        selected = torch.gather(expert_outputs, dim=2, index=indices_expanded)
        output = (selected * weights.unsqueeze(-1)).sum(dim=2)

        return output


@PROJECTOR_REGISTRY.register("vit_decoder")
class ViTDecoderProjector(BaseProjector):
    """ViT-based projector head.

    Used in MAETok prediction alignment. Passes features through a lightweight
    ViT decoder block stack instead of an MLP.

    Args:
        in_dim: Student feature dimension.
        out_dim: Teacher feature dimension.
        embed_dim: Decoder embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        num_patches: Sequence length (N) for positional embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        embed_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        num_patches: int = 256,
        **kwargs
    ) -> None:
        super().__init__(in_dim, out_dim)
        from omnitok.models.decoder.aux_decoder import AuxiliaryViTDecoder
        self.decoder = AuxiliaryViTDecoder(
            latent_dim=in_dim,
            out_dim=out_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_patches=num_patches,
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        return self.decoder(x, mask=mask)


@PROJECTOR_REGISTRY.register("pixel_shuffle")
class PixelShuffleProjector(BaseProjector):
    """UniLIP-style upsampling projector.

    Rearranges (B, N, C) into (B, H, W, C), applies pixel shuffle to increase
    spatial resolution while decreasing channel dimension, then applies MLP.

    Args:
        in_dim: Student feature dimension.
        out_dim: Teacher feature dimension.
        scale_factor: Upsampling factor (e.g., 0.5 for 2x resolution increase).
        hidden_dim: Hidden dimension for MLP.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        scale_factor: float = 0.5,
        hidden_dim: int = 0,
        **kwargs
    ) -> None:
        super().__init__(in_dim, out_dim)
        self.scale_factor = scale_factor
        # After pixel_shuffle(x, scale_factor=0.5), channels = C // (scale_factor**2)
        # For scale_factor=0.5, channels = C * 4
        new_in_dim = int(in_dim / (scale_factor * scale_factor))

        hidden = hidden_dim or max(new_in_dim, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(new_in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, out_dim, bias=True)
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        # x: (B, N, C)
        B, N, C = x.size()
        H = W = int(N ** 0.5)
        if H * W != N:
            # If there's a cls token, we assume it's stripped before projector.
            raise ValueError(f"PixelShuffleProjector requires spatial tokens only (N={N} is not a perfect square).")

        x = x.view(B, W, H, C) # N, W, H, C
        # N, W, H * scale, C // scale
        x = x.view(B, W, int(H * self.scale_factor), int(C / self.scale_factor))
        # N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(B, int(H * self.scale_factor), int(W * self.scale_factor), int(C / (self.scale_factor * self.scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()

        # Flatten back to sequence
        x = x.view(B, -1, x.size(-1))

        return self.proj(x)
