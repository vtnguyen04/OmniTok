"""Tests for new architecture components.

Covers:
- CNNDecoder (from REPA-E/VA-VAE)
- CNNEncoder (from REPA-E)
- LatentNormLoss (from AIOTok)
- MLP Bottleneck option (from AIOTok)
- get_last_layer() fix (VA-VAE convention)
"""

import torch
import torch.nn as nn

import omnitok.losses  # noqa: F401

# Trigger registry decorators
import omnitok.models.decoder  # noqa: F401
import omnitok.models.encoder  # noqa: F401
from omnitok.registry import DECODER_REGISTRY, ENCODER_REGISTRY, LOSS_REGISTRY

# =============================================================================
# CNN Decoder Tests
# =============================================================================

class TestCNNDecoder:
    """Test CNN decoder ported from REPA-E/VA-VAE."""

    def test_registry_registered(self):
        """cnn_decoder must be in DECODER_REGISTRY."""
        assert "cnn_decoder" in DECODER_REGISTRY

    def test_forward_shape(self):
        """CNN decoder: (B, z_ch, 16, 16) → (B, 3, 256, 256)."""
        decoder = DECODER_REGISTRY.build(
            "cnn_decoder", ch=64, ch_mult=[1, 1, 2, 2, 4],
            z_channels=32, resolution=256, num_res_blocks=1,
        )
        z = torch.randn(2, 32, 16, 16)
        out = decoder(z)
        assert out.shape == (2, 3, 256, 256)

    def test_forward_f8d4(self):
        """CNN decoder with f8d4 config: (B, 4, 32, 32) → (B, 3, 256, 256)."""
        decoder = DECODER_REGISTRY.build(
            "cnn_decoder", ch=64, ch_mult=[1, 2, 4, 4],
            z_channels=4, resolution=256, num_res_blocks=1,
        )
        z = torch.randn(2, 4, 32, 32)
        out = decoder(z)
        assert out.shape == (2, 3, 256, 256)

    def test_backward(self):
        """CNN decoder must support backward pass."""
        decoder = DECODER_REGISTRY.build(
            "cnn_decoder", ch=32, ch_mult=[1, 1, 2],
            z_channels=16, resolution=64, num_res_blocks=1,
        )
        z = torch.randn(2, 16, 16, 16, requires_grad=True)
        out = decoder(z)
        loss = out.mean()
        loss.backward()
        assert z.grad is not None

    def test_get_last_layer(self):
        """CNN decoder get_last_layer must return conv_out.weight."""
        decoder = DECODER_REGISTRY.build(
            "cnn_decoder", ch=32, ch_mult=[1, 1, 2],
            z_channels=16, resolution=64, num_res_blocks=1,
        )
        last = decoder.get_last_layer()
        assert isinstance(last, nn.Parameter)
        assert last is decoder.conv_out.weight


# =============================================================================
# CNN Encoder Tests
# =============================================================================

class TestCNNEncoder:
    """Test CNN encoder ported from REPA-E."""

    def test_registry_registered(self):
        """cnn_encoder must be in ENCODER_REGISTRY."""
        assert "cnn_encoder" in ENCODER_REGISTRY

    def test_forward_shape(self):
        """CNN encoder: (B, 3, 256, 256) → (B, 2*z_ch, 16, 16) with double_z=True."""
        encoder = ENCODER_REGISTRY.build(
            "cnn_encoder", ch=64, ch_mult=[1, 1, 2, 2, 4],
            z_channels=32, resolution=256, num_res_blocks=1, double_z=True,
        )
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, 64, 16, 16)  # 2*32 = 64

    def test_forward_no_double_z(self):
        """CNN encoder: (B, 3, 256, 256) → (B, z_ch, 16, 16) with double_z=False."""
        encoder = ENCODER_REGISTRY.build(
            "cnn_encoder", ch=64, ch_mult=[1, 1, 2, 2, 4],
            z_channels=32, resolution=256, num_res_blocks=1, double_z=False,
        )
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, 32, 16, 16)

    def test_backward(self):
        """CNN encoder must support backward pass."""
        encoder = ENCODER_REGISTRY.build(
            "cnn_encoder", ch=32, ch_mult=[1, 1, 2],
            z_channels=8, resolution=64, num_res_blocks=1, double_z=False,
        )
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        out = encoder(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None


# =============================================================================
# CNN Encoder + CNN Decoder Pipeline
# =============================================================================

class TestCNNPipeline:
    """Test full CNN encoder→decoder pipeline (REPA-E/VA-VAE style)."""

    def test_encode_decode(self):
        """CNN encode→decode: (B,3,64,64) → latent → (B,3,64,64)."""
        enc = ENCODER_REGISTRY.build(
            "cnn_encoder", ch=32, ch_mult=[1, 1, 2],
            z_channels=8, resolution=64, num_res_blocks=1, double_z=False,
        )
        dec = DECODER_REGISTRY.build(
            "cnn_decoder", ch=32, ch_mult=[1, 1, 2],
            z_channels=8, resolution=64, num_res_blocks=1,
        )
        x = torch.randn(2, 3, 64, 64)
        z = enc(x)
        recon = dec(z)
        assert recon.shape == x.shape


# =============================================================================
# LatentNormLoss Tests
# =============================================================================

class TestLatentNormLoss:
    """Test LatentNormLoss ported from AIOTok."""

    def test_registry_registered(self):
        """latent_norm must be in LOSS_REGISTRY."""
        assert "latent_norm" in LOSS_REGISTRY

    def test_output_scalar(self):
        """LatentNormLoss must output a scalar tensor."""
        from omnitok.losses.latent_norm import LatentNormLoss
        loss_fn = LatentNormLoss(weight=0.01)
        z = torch.randn(4, 32, 16, 16, requires_grad=True)
        loss = loss_fn(z)
        assert loss.shape == ()
        assert loss.requires_grad

    def test_zero_for_standard_normal(self):
        """Loss ≈ 0 for z ~ N(0,1)."""
        from omnitok.losses.latent_norm import LatentNormLoss
        loss_fn = LatentNormLoss(weight=1.0)
        z = torch.randn(1000, 32)  # Large sample → mean≈0, std≈1
        loss = loss_fn(z)
        assert loss.item() < 0.5  # Should be close to 0

    def test_high_for_offset(self):
        """Loss >> 0 for z with mean=5."""
        from omnitok.losses.latent_norm import LatentNormLoss
        loss_fn = LatentNormLoss(weight=1.0)
        z = torch.randn(100, 32) + 5.0  # Large mean offset
        loss = loss_fn(z)
        assert loss.item() > 4.0  # mean.abs() ≈ 5

    def test_weight_scaling(self):
        """Weight scales the loss correctly."""
        from omnitok.losses.latent_norm import LatentNormLoss
        loss_01 = LatentNormLoss(weight=0.1)
        loss_10 = LatentNormLoss(weight=1.0)
        z = torch.randn(4, 32) + 3.0
        l1 = loss_01(z)
        l2 = loss_10(z)
        assert abs(l2.item() / l1.item() - 10.0) < 0.5

    def test_works_with_4d(self):
        """LatentNormLoss must work with spatial (B,C,H,W) tensors."""
        from omnitok.losses.latent_norm import LatentNormLoss
        loss_fn = LatentNormLoss(weight=0.01)
        z = torch.randn(4, 32, 16, 16)
        loss = loss_fn(z)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# =============================================================================
# MLP Bottleneck Tests
# =============================================================================

class TestMLPBottleneck:
    """Test MLP bottleneck option in ViT encoder."""

    def test_mlp_bottleneck_creates_sequential(self):
        """bottleneck_type='mlp' creates nn.Sequential with LayerNorm+Linear+GELU+Linear."""
        from omnitok.models.encoder.vision_transformer_bottleneck import DinoVisionTransformerWithBottleneck
        model = DinoVisionTransformerWithBottleneck(
            patch_size=16, embed_dim=192, depth=2, num_heads=3,
            vit_feature_bottleneck=64, bottleneck_type="mlp",
        )
        assert isinstance(model.feature_bottleneck, nn.Sequential)
        assert len(model.feature_bottleneck) == 4  # LN, Linear, GELU, Linear

    def test_linear_bottleneck_creates_linear(self):
        """bottleneck_type='linear' creates nn.Linear."""
        from omnitok.models.encoder.vision_transformer_bottleneck import DinoVisionTransformerWithBottleneck
        model = DinoVisionTransformerWithBottleneck(
            patch_size=16, embed_dim=192, depth=2, num_heads=3,
            vit_feature_bottleneck=64, bottleneck_type="linear",
        )
        assert isinstance(model.feature_bottleneck, nn.Linear)


# =============================================================================
# get_last_layer Tests
# =============================================================================

class TestGetLastLayer:
    """Test that tokenizer.get_last_shared_layer() delegates to decoder."""

    def test_cnn_decoder_returns_conv_out(self):
        """With CNN decoder, should return decoder.conv_out.weight."""
        # Minimal ViT encoder
        from omnitok.models.encoder.vision_transformer_bottleneck import DinoVisionTransformerWithBottleneck
        from omnitok.models.tokenizer import Tokenizer
        enc = DinoVisionTransformerWithBottleneck(
            patch_size=16, embed_dim=192, depth=2, num_heads=3,
            vit_feature_bottleneck=32,
        )
        dec = DECODER_REGISTRY.build(
            "cnn_decoder", ch=32, ch_mult=[1, 1, 2],
            z_channels=32, resolution=64, num_res_blocks=1,
        )
        tok = Tokenizer(encoder=enc, decoder=dec)
        last = tok.get_last_shared_layer()
        assert last is dec.conv_out.weight

    def test_pixel_decoder_returns_refine_weight(self):
        """With ViT pixel decoder, should return final_refine last conv weight."""
        from omnitok.models.encoder.vision_transformer_bottleneck import DinoVisionTransformerWithBottleneck
        from omnitok.models.tokenizer import Tokenizer
        enc = DinoVisionTransformerWithBottleneck(
            patch_size=16, embed_dim=192, depth=2, num_heads=3,
            vit_feature_bottleneck=32,
        )
        dec = DECODER_REGISTRY.build(
            "pixel_decoder", in_chans=32, embed_dim=192, depth=2,
            num_heads=3, upscale_factor=16,
        )
        tok = Tokenizer(encoder=enc, decoder=dec)
        last = tok.get_last_shared_layer()
        assert last is dec.final_refine[-1].weight
