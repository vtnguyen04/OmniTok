"""Tests for MoE Cross-Attention Bottleneck.

Verifies:
  - Output shapes for 3D and 4D inputs
  - Gradient flow through all components
  - MoE gating produces valid weights
  - Multi-teacher conditioned gating
  - Parameter count increases with experts
  - Integration with registry
"""

import pytest
import torch
import torch.nn as nn

from omnitok.models.bottleneck.moe_crossattn import (
    CrossAttentionExpert,
    MoECrossAttnBottleneck,
    MoECrossAttention,
    MoEFFN,
    MoEGating,
)
from omnitok.registry import BOTTLENECK_REGISTRY


class TestCrossAttentionExpert:
    """Test individual cross-attention expert."""

    def test_output_shape(self) -> None:
        """Expert output matches query shape."""
        expert = CrossAttentionExpert(dim=64, num_heads=4)
        q = torch.randn(2, 8, 64)
        kv = torch.randn(2, 16, 64)
        out = expert(q, kv)
        assert out.shape == (2, 8, 64)

    def test_gradient_flow(self) -> None:
        """Gradients flow through expert attention."""
        expert = CrossAttentionExpert(dim=64, num_heads=4)
        q = torch.randn(2, 8, 64, requires_grad=True)
        kv = torch.randn(2, 16, 64)
        out = expert(q, kv)
        out.sum().backward()
        assert q.grad is not None


class TestMoEGating:
    """Test MoE gating mechanism."""

    def test_output_shapes(self) -> None:
        """Gating produces correct weight and index shapes."""
        gating = MoEGating(dim=64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        weights, indices = gating(x)
        assert weights.shape == (2, 8, 2)
        assert indices.shape == (2, 8, 2)

    def test_weights_sum_to_one(self) -> None:
        """Gating weights sum to 1 across selected experts."""
        gating = MoEGating(dim=64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        weights, _ = gating(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_indices_in_range(self) -> None:
        """Expert indices are within valid range."""
        gating = MoEGating(dim=64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        _, indices = gating(x)
        assert indices.min() >= 0
        assert indices.max() < 4

    def test_teacher_conditioned_gating(self) -> None:
        """Gating accepts teacher conditioning signal."""
        gating = MoEGating(dim=64, num_experts=4, top_k=2, teacher_dim=32)
        x = torch.randn(2, 8, 64)
        teacher_cond = torch.randn(2, 8, 32)
        weights, indices = gating(x, teacher_cond)
        assert weights.shape == (2, 8, 2)

    def test_teacher_cond_changes_routing(self) -> None:
        """Different teacher conditioning produces different routing."""
        gating = MoEGating(dim=64, num_experts=4, top_k=2, teacher_dim=32)
        x = torch.randn(2, 8, 64)
        cond_a = torch.randn(2, 8, 32)
        cond_b = torch.randn(2, 8, 32) * 10  # very different
        _, idx_a = gating(x, cond_a)
        _, idx_b = gating(x, cond_b)
        # With very different conditioning, routing should differ
        assert not torch.equal(idx_a, idx_b)


class TestMoECrossAttention:
    """Test MoE Cross-Attention layer."""

    def test_output_shape(self) -> None:
        """MoE Cross-Attention preserves query shape with residual."""
        layer = MoECrossAttention(dim=64, num_experts=4, top_k=2)
        q = torch.randn(2, 8, 64)
        kv = torch.randn(2, 16, 64)
        out = layer(q, kv)
        assert out.shape == (2, 8, 64)

    def test_with_teacher_cond(self) -> None:
        """MoE Cross-Attention works with teacher conditioning."""
        layer = MoECrossAttention(dim=64, num_experts=4, top_k=2, teacher_dim=32)
        q = torch.randn(2, 8, 64)
        kv = torch.randn(2, 16, 64)
        cond = torch.randn(2, 8, 32)
        out = layer(q, kv, teacher_cond=cond)
        assert out.shape == (2, 8, 64)


class TestMoEFFN:
    """Test MoE Feed-Forward Network (uses layers/ffn.Mlp internally)."""

    def test_output_shape(self) -> None:
        """MoE FFN preserves input shape."""
        ffn = MoEFFN(dim=64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        out = ffn(x)
        assert out.shape == (2, 8, 64)

    def test_with_teacher_cond(self) -> None:
        """MoE FFN works with teacher conditioning."""
        ffn = MoEFFN(dim=64, num_experts=4, top_k=2, teacher_dim=32)
        x = torch.randn(2, 8, 64)
        cond = torch.randn(2, 8, 32)
        out = ffn(x, teacher_cond=cond)
        assert out.shape == (2, 8, 64)


class TestMoECrossAttnBottleneck:
    """Test the full MoE Cross-Attention Bottleneck."""

    @pytest.fixture
    def bottleneck(self) -> MoECrossAttnBottleneck:
        """Create a small bottleneck for testing (no teacher conditioning)."""
        return MoECrossAttnBottleneck(
            in_dim=64,
            latent_dim=16,
            num_queries=8,
            num_blocks=1,
            num_experts=2,
            top_k=1,
            num_heads=4,
            num_heads_per_expert=2,
            ffn_ratio=2.0,
        )

    @pytest.fixture
    def bottleneck_with_teachers(self) -> MoECrossAttnBottleneck:
        """Create a bottleneck with multi-teacher conditioning."""
        return MoECrossAttnBottleneck(
            in_dim=64,
            latent_dim=16,
            num_queries=8,
            num_blocks=1,
            num_experts=4,
            top_k=2,
            num_heads=4,
            num_heads_per_expert=2,
            ffn_ratio=2.0,
            teacher_dims={"dinov2": 32, "siglip": 48},
            teacher_cond_dim=16,
        )

    def test_3d_input_output_shape(self, bottleneck: MoECrossAttnBottleneck) -> None:
        """3D input (B, N, C) produces correct latent shape."""
        x = torch.randn(2, 8, 64)
        z, info = bottleneck(x)
        assert z.shape == (2, 8, 16)
        assert "queries" in info
        assert info["teacher_conditioned"] is False

    def test_4d_input_output_shape(self, bottleneck: MoECrossAttnBottleneck) -> None:
        """4D input (B, C, H, W) produces correct spatial latent shape."""
        x = torch.randn(2, 64, 4, 4)
        z, info = bottleneck(x)
        assert z.shape == (2, 16, 4, 4)

    def test_gradient_flow_all_params(self, bottleneck: MoECrossAttnBottleneck) -> None:
        """Gradients exist for all trainable parameters.

        Note: MoE gating weights may have zero gradient because torch.topk
        index selection is non-differentiable — only softmax weights propagate.
        """
        x = torch.randn(2, 8, 64)
        z, _ = bottleneck(x)
        loss = z.sum()
        loss.backward()

        for name, param in bottleneck.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_nan_output(self, bottleneck: MoECrossAttnBottleneck) -> None:
        """Output contains no NaN values."""
        x = torch.randn(2, 8, 64)
        z, _ = bottleneck(x)
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()

    def test_zero_input(self, bottleneck: MoECrossAttnBottleneck) -> None:
        """Handles zero input gracefully."""
        x = torch.zeros(2, 8, 64)
        z, _ = bottleneck(x)
        assert not torch.isnan(z).any()

    def test_latent_dim_property(self, bottleneck: MoECrossAttnBottleneck) -> None:
        """latent_dim property returns correct value."""
        assert bottleneck.latent_dim == 16

    def test_more_params_than_linear(self) -> None:
        """MoE bottleneck has more parameters than equivalent linear."""
        from omnitok.models.bottleneck.standard import LinearBottleneck

        linear = LinearBottleneck(in_dim=768, latent_dim=64)
        moe = MoECrossAttnBottleneck(
            in_dim=768, latent_dim=64, num_queries=256,
            num_blocks=2, num_experts=4, top_k=2,
        )

        linear_params = sum(p.numel() for p in linear.parameters())
        moe_params = sum(p.numel() for p in moe.parameters())
        assert moe_params > linear_params

    def test_registry_registered(self) -> None:
        """MoE bottleneck is registered in BOTTLENECK_REGISTRY."""
        assert "moe_crossattn" in BOTTLENECK_REGISTRY

    def test_registry_build(self) -> None:
        """Can build MoE bottleneck from registry."""
        bottleneck = BOTTLENECK_REGISTRY.build(
            "moe_crossattn",
            in_dim=64, latent_dim=16, num_queries=8,
            num_blocks=1, num_experts=2, top_k=1,
        )
        x = torch.randn(2, 8, 64)
        z, _ = bottleneck(x)
        assert z.shape == (2, 8, 16)


class TestMultiTeacherConditioning:
    """Test multi-teacher conditioned MoE gating."""

    @pytest.fixture
    def bottleneck(self) -> MoECrossAttnBottleneck:
        """Bottleneck with DINOv2 + SigLIP teacher conditioning."""
        return MoECrossAttnBottleneck(
            in_dim=64,
            latent_dim=16,
            num_queries=8,
            num_blocks=1,
            num_experts=4,
            top_k=2,
            num_heads=4,
            num_heads_per_expert=2,
            ffn_ratio=2.0,
            teacher_dims={"dinov2": 32, "siglip": 48},
            teacher_cond_dim=16,
        )

    def test_forward_with_teachers(self, bottleneck: MoECrossAttnBottleneck) -> None:
        """Forward pass with teacher features activates conditioning."""
        x = torch.randn(2, 8, 64)
        teacher_feats = {
            "dinov2": torch.randn(2, 8, 32),
            "siglip": torch.randn(2, 8, 48),
        }
        z, info = bottleneck(x, teacher_features=teacher_feats)
        assert z.shape == (2, 8, 16)
        assert info["teacher_conditioned"] is True

    def test_forward_without_teachers(self, bottleneck: MoECrossAttnBottleneck) -> None:
        """Forward pass without teacher features still works (graceful fallback)."""
        x = torch.randn(2, 8, 64)
        z, info = bottleneck(x)
        assert z.shape == (2, 8, 16)
        assert info["teacher_conditioned"] is False

    def test_partial_teachers(self, bottleneck: MoECrossAttnBottleneck) -> None:
        """Forward pass with only some teachers (e.g., only DINOv2)."""
        x = torch.randn(2, 8, 64)
        teacher_feats = {"dinov2": torch.randn(2, 8, 32)}
        z, info = bottleneck(x, teacher_features=teacher_feats)
        assert z.shape == (2, 8, 16)
        assert info["teacher_conditioned"] is True

    def test_gradient_flows_through_teacher_projectors(
        self, bottleneck: MoECrossAttnBottleneck
    ) -> None:
        """Gradients flow through teacher projector layers."""
        x = torch.randn(2, 8, 64)
        teacher_feats = {
            "dinov2": torch.randn(2, 8, 32, requires_grad=True),
            "siglip": torch.randn(2, 8, 48, requires_grad=True),
        }
        z, _ = bottleneck(x, teacher_features=teacher_feats)
        z.sum().backward()

        for name in ["dinov2", "siglip"]:
            proj = bottleneck.teacher_projectors[name]
            for pname, param in proj.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"No grad for teacher_projectors.{name}.{pname}"

    def test_has_teacher_projectors(self, bottleneck: MoECrossAttnBottleneck) -> None:
        """Teacher projectors are created for each teacher."""
        assert "dinov2" in bottleneck.teacher_projectors
        assert "siglip" in bottleneck.teacher_projectors

    def test_teacher_spatial_mismatch_handled(
        self, bottleneck: MoECrossAttnBottleneck
    ) -> None:
        """Teacher features with different spatial size are interpolated."""
        x = torch.randn(2, 16, 64)  # 16 tokens = 4x4
        teacher_feats = {
            "dinov2": torch.randn(2, 4, 32),  # 4 tokens = 2x2 (mismatch)
        }
        z, info = bottleneck(x, teacher_features=teacher_feats)
        assert z.shape == (2, 16, 16)
        assert info["teacher_conditioned"] is True
