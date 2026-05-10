"""MoE Cross-Attention Bottleneck — Learnable Query Bottleneck with Mixture of Experts.

Implements a Perceiver-style bidirectional read-write bottleneck where:
  1. Learnable queries cross-attend to the feature map (Read)
  2. Queries self-attend to consolidate global context (Process)
  3. Feature map cross-attends back to queries (Write)

Each cross-attention layer uses a sparse Mixture of Experts (MoE) gating
mechanism. When multi-teacher features are provided, the gating network
conditions on teacher features to specialize experts per teacher modality
(e.g., Expert 1 → DINOv2 spatial, Expert 2 → SigLIP semantic, Expert 3 → SAM boundary).

Architecture diagram:
    Feature Map ──(K,V)──► MoE Cross-Attention ◄──(Q)── Queries
                                    │
                              Self-Attention (reused from layers/)
                                    │
                              MoE-FFN (experts built from layers/ffn.py)
                                    │
    Feature Map ──(Q)───► MoE Cross-Attention ◄──(K,V)── Queries
                                    │
                                   FFN (reused from layers/)
                                    │
                            Output Feature Map

Reference:
    - Perceiver (Jaegle et al., 2021) — latent workspace pattern
    - Mixture of Experts (Shazeer et al., 2017) — sparse gating
    - DETR (Carion et al., 2020) — learned queries for detection
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from omnitok.models.bottleneck.standard import BaseBottleneck
from omnitok.models.layers.ffn import Mlp
from omnitok.registry import BOTTLENECK_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MoE Gating
# ---------------------------------------------------------------------------

class MoEGating(nn.Module):
    """Top-k sparse gating for Mixture of Experts.

    Routes each token to the top-k experts. When teacher features are provided,
    gating is conditioned on the concatenation of input + teacher features,
    enabling teacher-aware expert specialization.

    Args:
        dim: Input dimension.
        num_experts: Total number of experts.
        top_k: Number of experts to activate per token.
        teacher_dim: If > 0, enables teacher-conditioned gating.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        teacher_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        gate_in_dim = dim + teacher_dim if teacher_dim > 0 else dim
        self.gate = nn.Linear(gate_in_dim, num_experts, bias=False)
        self.teacher_dim = teacher_dim

    def forward(
        self,
        x: Tensor,
        teacher_cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute gating weights with optional teacher conditioning.

        Args:
            x: Input tensor (B, N, D).
            teacher_cond: Optional teacher features (B, N, D_teacher) for
                         conditioned gating. When provided, experts specialize
                         per teacher modality.

        Returns:
            Tuple of:
                - weights: Gating weights (B, N, top_k), sums to 1.
                - indices: Expert indices (B, N, top_k).
        """
        if self.teacher_dim > 0:
            if teacher_cond is None:
                teacher_cond = torch.zeros(x.shape[0], 1, self.teacher_dim, device=x.device, dtype=x.dtype)

            if teacher_cond.shape[1] != x.shape[1]:
                teacher_cond = teacher_cond.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)

            gate_input = torch.cat([x, teacher_cond], dim=-1)
        else:
            gate_input = x

        logits = self.gate(gate_input)
        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)
        return weights, indices


# ---------------------------------------------------------------------------
# Cross-Attention Expert (lightweight, no DRY duplication)
# ---------------------------------------------------------------------------

class CrossAttentionExpert(nn.Module):
    """Single cross-attention expert with Q/K/V projections.

    Kept as a dedicated module because cross-attention (separate Q and KV sources)
    is not covered by the existing SelfAttention in layers/attention.py.

    Args:
        dim: Model dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        """Compute cross-attention from q attending to kv.

        Args:
            q: Query tensor (B, Nq, D).
            kv: Key-Value tensor (B, Nkv, D).

        Returns:
            Attended output (B, Nq, D).
        """
        b, nq, d = q.shape
        nkv = kv.shape[1]
        head_dim = d // self.num_heads

        queries = self.q_proj(q).reshape(b, nq, self.num_heads, head_dim).transpose(1, 2)
        keys = self.k_proj(kv).reshape(b, nkv, self.num_heads, head_dim).transpose(1, 2)
        values = self.v_proj(kv).reshape(b, nkv, self.num_heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(queries, keys, values)
        out = out.transpose(1, 2).reshape(b, nq, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# MoE Cross-Attention Layer
# ---------------------------------------------------------------------------

class MoECrossAttention(nn.Module):
    """Cross-Attention layer with Mixture of Experts routing.

    Each token is routed to top-k experts, and outputs are aggregated
    by gating weights. Supports optional teacher-conditioned gating.

    Args:
        dim: Model dimension.
        num_experts: Number of attention experts.
        top_k: Number of experts activated per token.
        num_heads_per_expert: Attention heads per expert.
        teacher_dim: Dimension of teacher conditioning features.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        num_heads_per_expert: int = 4,
        teacher_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gating = MoEGating(dim, num_experts, top_k, teacher_dim)
        self.experts = nn.ModuleList([
            CrossAttentionExpert(dim, num_heads_per_expert)
            for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        teacher_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """MoE Cross-Attention with residual connection.

        Args:
            q: Query tensor (B, Nq, D).
            kv: Key-Value tensor (B, Nkv, D).
            teacher_cond: Optional teacher features for conditioned gating.

        Returns:
            Output tensor (B, Nq, D) with residual.
        """
        residual = q
        q_normed = self.norm(q)

        weights, indices = self.gating(q_normed, teacher_cond)

        # Compute all expert outputs
        expert_outputs = torch.stack(
            [expert(q_normed, kv) for expert in self.experts], dim=2
        )  # (B, Nq, num_experts, D)

        # Gather top-k expert outputs and aggregate
        b, nq, d = q_normed.shape
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, d)
        selected = torch.gather(expert_outputs, dim=2, index=indices_expanded)
        output = (selected * weights.unsqueeze(-1)).sum(dim=2)

        return residual + output


# ---------------------------------------------------------------------------
# MoE FFN — wraps existing Mlp from layers/ffn.py as experts (DRY compliant)
# ---------------------------------------------------------------------------

class MoEFFN(nn.Module):
    """MoE Feed-Forward Network using layers/ffn.Mlp as expert backbone.

    Reuses the existing Mlp class from omnitok.models.layers.ffn for each expert,
    ensuring DRY compliance. Supports teacher-conditioned gating.

    Args:
        dim: Model dimension.
        num_experts: Number of FFN experts.
        top_k: Experts activated per token.
        ffn_ratio: Hidden dimension multiplier.
        teacher_dim: Dimension of teacher conditioning features.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        ffn_ratio: float = 4.0,
        teacher_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = int(dim * ffn_ratio)

        self.gating = MoEGating(dim, num_experts, top_k, teacher_dim)
        # Reuse Mlp from layers/ffn.py — DRY compliant
        self.experts = nn.ModuleList([
            Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=nn.SiLU)
            for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        teacher_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """MoE FFN with residual connection.

        Args:
            x: Input tensor (B, N, D).
            teacher_cond: Optional teacher conditioning for gating.

        Returns:
            Output tensor (B, N, D) with residual.
        """
        residual = x
        x_normed = self.norm(x)

        weights, indices = self.gating(x_normed, teacher_cond)

        expert_outputs = torch.stack(
            [expert(x_normed) for expert in self.experts], dim=2
        )

        b, n, d = x_normed.shape
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, d)
        selected = torch.gather(expert_outputs, dim=2, index=indices_expanded)
        output = (selected * weights.unsqueeze(-1)).sum(dim=2)

        return residual + output


# ---------------------------------------------------------------------------
# MoE Cross-Attention Block (one L× iteration)
# ---------------------------------------------------------------------------

class MoECrossAttentionBlock(nn.Module):
    """One block of the MoE Cross-Attention Bottleneck.

    Flow per block:
        1. MoE Cross-Attention (Queries attend to Feature Map — Read)
        2. Self-Attention (Queries self-attend — Process, reuses layers/attention.py)
        3. MoE-FFN (Non-linear processing — Transform)
        4. MoE Cross-Attention (Feature Map attends to Queries — Write-back)
        5. FFN (Final feature map processing, reuses layers/ffn.Mlp)

    All teacher_cond signals are passed through to MoE gating for
    teacher-aware expert specialization.

    Args:
        dim: Model dimension.
        num_experts: Number of MoE experts per layer.
        top_k: Active experts per token.
        num_heads: Self-attention heads.
        num_heads_per_expert: Attention heads per expert.
        ffn_ratio: FFN hidden dimension ratio.
        teacher_dim: Dimension of teacher conditioning features.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        num_heads: int = 8,
        num_heads_per_expert: int = 4,
        ffn_ratio: float = 4.0,
        teacher_dim: int = 0,
    ) -> None:
        super().__init__()
        # Step 1: Read — Queries cross-attend to Feature Map
        self.read_crossattn = MoECrossAttention(
            dim, num_experts, top_k, num_heads_per_expert, teacher_dim
        )
        # Step 2: Process — Queries self-attend
        # Reuses SelfAttention from layers/attention.py — DRY compliant
        from omnitok.models.layers.attention import SelfAttention
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads)

        # Step 3: Transform — MoE FFN on Queries
        self.moe_ffn = MoEFFN(dim, num_experts, top_k, ffn_ratio, teacher_dim)

        # Step 4: Write — Feature Map cross-attends to Queries
        self.write_crossattn = MoECrossAttention(
            dim, num_experts, top_k, num_heads_per_expert, teacher_dim
        )
        # Step 5: Final FFN on Feature Map — Reuses Mlp from layers/ffn.py
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim * ffn_ratio), act_layer=nn.SiLU)

    def forward(
        self,
        feature_map: Tensor,
        queries: Tensor,
        teacher_cond: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Process one block of bidirectional read-write.

        Args:
            feature_map: Encoder feature tokens (B, N_feat, D).
            queries: Learnable query tokens (B, N_query, D).
            teacher_cond: Optional aggregated teacher features (B, N_feat, D_teacher)
                         for conditioned MoE gating. Enables expert specialization
                         per teacher modality (DINOv2/SigLIP/SAM).

        Returns:
            Tuple of:
                - Updated feature_map (B, N_feat, D).
                - Updated queries (B, N_query, D).
        """
        # Read: Queries absorb information from Feature Map
        queries = self.read_crossattn(q=queries, kv=feature_map, teacher_cond=teacher_cond)

        # Process: Queries consolidate context internally (layers/attention.SelfAttention)
        queries = queries + self.self_attn(self.self_attn_norm(queries))

        # Transform: Non-linear processing via MoE FFN
        queries = self.moe_ffn(queries, teacher_cond=teacher_cond)

        # Write: Feature Map updates from processed Queries
        feature_map = self.write_crossattn(q=feature_map, kv=queries, teacher_cond=teacher_cond)

        # Final FFN on feature map (layers/ffn.Mlp)
        feature_map = feature_map + self.ffn(self.ffn_norm(feature_map))

        return feature_map, queries


# ---------------------------------------------------------------------------
# Main Bottleneck Module
# ---------------------------------------------------------------------------

@BOTTLENECK_REGISTRY.register("moe_crossattn")
class MoECrossAttnBottleneck(BaseBottleneck):
    """MoE Cross-Attention Bottleneck — Perceiver-style latent workspace with MoE.

    Replaces simple linear projection with a learned bidirectional read-write
    mechanism using Mixture of Experts. Learnable queries act as an information
    bottleneck, compressing the feature map through cross-attention.

    Multi-Teacher Conditioning:
        When teacher_dims is provided (e.g., {"dinov2": 768, "siglip": 1152}),
        the bottleneck learns per-teacher projection layers and concatenates
        them into a conditioning signal for MoE gating. This enables experts
        to automatically specialize on different teacher modalities:
            - Expert 1 → Spatial features (DINOv2)
            - Expert 2 → Semantic features (SigLIP)
            - Expert 3 → Boundary features (SAM)

    Advantages over LinearBottleneck:
        - Preserves spatial relationships through attention (vs. point-wise projection)
        - MoE enables specialization (texture/shape/semantic experts)
        - Learnable queries adapt bottleneck capacity to content
        - Bidirectional read-write avoids information loss from one-shot projection
        - Teacher-conditioned gating for multi-teacher alignment

    Args:
        in_dim: Input feature dimension from encoder.
        latent_dim: Output latent dimension (controls capacity).
        num_queries: Number of learnable query tokens.
        num_blocks: Number of read-write blocks (L in the diagram).
        num_experts: Number of MoE experts per attention layer.
        top_k: Number of active experts per token.
        num_heads: Number of self-attention heads.
        num_heads_per_expert: Number of attention heads per expert.
        ffn_ratio: FFN hidden dimension ratio.
        teacher_dims: Dict mapping teacher names to their feature dims.
                      If provided, enables teacher-conditioned MoE gating.
        teacher_cond_dim: Projected dimension for teacher conditioning signal.

    Example:
        >>> # Without teacher conditioning
        >>> bottleneck = MoECrossAttnBottleneck(in_dim=768, latent_dim=64)
        >>> z, info = bottleneck(torch.randn(2, 256, 768))
        >>> z.shape  # (2, 256, 64)

        >>> # With multi-teacher conditioning
        >>> bottleneck = MoECrossAttnBottleneck(
        ...     in_dim=768, latent_dim=64,
        ...     teacher_dims={"dinov2": 768, "siglip": 1152},
        ...     teacher_cond_dim=128,
        ... )
        >>> teacher_feats = {"dinov2": torch.randn(2, 256, 768),
        ...                   "siglip": torch.randn(2, 256, 1152)}
        >>> z, info = bottleneck(torch.randn(2, 256, 768), teacher_features=teacher_feats)
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        num_queries: int = 256,
        num_blocks: int = 2,
        num_experts: int = 4,
        top_k: int = 2,
        num_heads: int = 8,
        num_heads_per_expert: int = 4,
        ffn_ratio: float = 4.0,
        teacher_dims: Optional[Dict[str, int]] = None,
        teacher_cond_dim: int = 128,
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.in_dim = in_dim
        self.num_queries = num_queries

        # Teacher-conditioned gating
        self.teacher_dims = teacher_dims or {}
        self.teacher_cond_dim = teacher_cond_dim if self.teacher_dims else 0

        if self.teacher_dims:
            # Per-teacher projection to common conditioning dimension
            self.teacher_projectors = nn.ModuleDict({
                name: nn.Sequential(
                    nn.LayerNorm(feat_dim),
                    nn.Linear(feat_dim, teacher_cond_dim),
                )
                for name, feat_dim in self.teacher_dims.items()
            })
            logger.info(
                f"MoE teacher-conditioned gating enabled: "
                f"{list(self.teacher_dims.keys())} → {teacher_cond_dim}d conditioning"
            )

        # Learnable queries (the latent workspace)
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, in_dim) * 0.02
        )

        # Stacked read-write blocks
        self.blocks = nn.ModuleList([
            MoECrossAttentionBlock(
                dim=in_dim,
                num_experts=num_experts,
                top_k=top_k,
                num_heads=num_heads,
                num_heads_per_expert=num_heads_per_expert,
                ffn_ratio=ffn_ratio,
                teacher_dim=self.teacher_cond_dim,
            )
            for _ in range(num_blocks)
        ])

        # Final projection to latent dimension
        self.output_norm = nn.LayerNorm(in_dim)
        self.output_proj = nn.Linear(in_dim, latent_dim)

        self._init_weights()
        self._log_params()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _log_params(self) -> None:
        """Log parameter count for debugging."""
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MoE CrossAttn Bottleneck: {total:,} params "
            f"(queries={self.num_queries}, blocks={len(self.blocks)}, "
            f"in_dim={self.in_dim}, latent_dim={self.latent_dim}, "
            f"teachers={list(self.teacher_dims.keys()) or 'none'})"
        )

    def _build_teacher_conditioning(
        self,
        teacher_features: Optional[Dict[str, Tensor]],
        n_tokens: int,
    ) -> Optional[Tensor]:
        """Build aggregated teacher conditioning signal for MoE gating.

        Projects each teacher's features to a common dimension and sums them
        to create a single conditioning vector per token.

        Args:
            teacher_features: Dict of {teacher_name: features (B, N, D_teacher)}.
            n_tokens: Expected number of tokens for spatial alignment.

        Returns:
            Conditioning tensor (B, N, teacher_cond_dim) or None.
        """
        if not self.teacher_dims or teacher_features is None:
            return None

        cond_parts = []
        for name, proj in self.teacher_projectors.items():
            if name in teacher_features:
                feat = teacher_features[name]
                # Handle spatial dimension mismatch via interpolation
                if feat.shape[1] != n_tokens:
                    h = w = int(math.sqrt(n_tokens))
                    h_f = w_f = int(math.sqrt(feat.shape[1]))
                    feat = feat.reshape(feat.shape[0], h_f, w_f, -1).permute(0, 3, 1, 2)
                    feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
                    feat = feat.permute(0, 2, 3, 1).reshape(feat.shape[0], n_tokens, -1)
                cond_parts.append(proj(feat))

        if not cond_parts:
            return None

        # Sum teacher conditionings (enables variable number of teachers at inference)
        return sum(cond_parts)

    def forward(
        self,
        x: Tensor,
        teacher_features: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Forward pass through MoE Cross-Attention Bottleneck.

        Handles both 3D (B, N, C) and 4D (B, C, H, W) inputs.
        When teacher_features are provided, MoE gating is conditioned on
        aggregated teacher signals for expert specialization.

        Args:
            x: Input features from encoder, either (B, N, C) or (B, C, H, W).
            teacher_features: Optional dict of {teacher_name: features}.
                             Enables teacher-conditioned expert routing.

        Returns:
            Tuple of:
                - z: Latent features, same spatial layout as input.
                - info: Dict with 'queries' and 'teacher_conditioned' flag.
        """
        is_4d = x.ndim == 4
        if is_4d:
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        else:
            b = x.shape[0]

        # Build teacher conditioning (returns None if no teachers)
        teacher_cond = self._build_teacher_conditioning(teacher_features, x.shape[1])

        # Expand learnable queries for the batch
        queries = self.queries.expand(b, -1, -1)

        # Process through bidirectional read-write blocks
        feature_map = x
        for block in self.blocks:
            feature_map, queries = block(feature_map, queries, teacher_cond)

        # Project the enriched feature map to latent dimension
        z = self.output_proj(self.output_norm(feature_map))

        if is_4d:
            z = z.reshape(b, h, w, self.latent_dim).permute(0, 3, 1, 2)

        return z, {
            "queries": queries,
            "teacher_conditioned": teacher_cond is not None,
        }
