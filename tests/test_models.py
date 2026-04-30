"""Smoke tests for ported VTP models — encoder, bottleneck, decoder."""

import pytest
import torch

from omnitok.models.encoder.vision_transformer import DinoVisionTransformer
from omnitok.models.encoder.vision_transformer_bottleneck import DinoVisionTransformerWithBottleneck
from omnitok.models.decoder.pixel_decoder import DinoV3PixelDecoder


@pytest.fixture
def small_encoder_kwargs():
    """Minimal ViT config for fast testing."""
    return dict(
        img_size=64,
        patch_size=8,
        embed_dim=128,
        depth=2,
        num_heads=4,
        ffn_layer="mlp",
        norm_layer="layernorm",
    )


class TestDinoVisionTransformer:
    """Tests for the base ViT encoder."""

    def test_forward_features_shape(self, small_encoder_kwargs):
        """forward_features returns dict with correct shapes."""
        model = DinoVisionTransformer(**small_encoder_kwargs)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model.forward_features(x)
        # 64/8 = 8 patches per side -> 64 patches
        assert out["x_norm_clstoken"].shape == (2, 128)
        assert out["x_norm_patchtokens"].shape == (2, 64, 128)

    def test_forward_training_returns_dict(self, small_encoder_kwargs):
        """forward with is_training=True returns feature dict."""
        model = DinoVisionTransformer(**small_encoder_kwargs)
        model.train()
        x = torch.randn(2, 3, 64, 64)
        out = model(x, is_training=True)
        assert "x_norm_clstoken" in out
        assert "x_norm_patchtokens" in out

    def test_gradient_flows(self, small_encoder_kwargs):
        """Gradients flow through the encoder."""
        model = DinoVisionTransformer(**small_encoder_kwargs)
        model.train()
        x = torch.randn(2, 3, 64, 64)
        out = model(x, is_training=True)
        loss = out["x_norm_patchtokens"].mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad


class TestBottleneckEncoder:
    """Tests for encoder with bottleneck projection."""

    def test_bottleneck_reduces_dim(self, small_encoder_kwargs):
        """Bottleneck reduces patch token dimension."""
        kwargs = {**small_encoder_kwargs, "vit_feature_bottleneck": 32}
        model = DinoVisionTransformerWithBottleneck(**kwargs)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model.forward_features(x)
        # Bottleneck projects 128 -> 32
        assert out["x_norm_patchtokens"].shape == (2, 64, 32)
        assert out["x_norm_clstoken"].shape == (2, 32)

    def test_encode_returns_spatial(self, small_encoder_kwargs):
        """encode() returns spatial 4D features (B, C, H, W)."""
        kwargs = {**small_encoder_kwargs, "vit_feature_bottleneck": 32}
        model = DinoVisionTransformerWithBottleneck(**kwargs)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            z = model.encode(x)
        # patch_size=8, img=64 -> 8x8 spatial, bottleneck_dim=32
        assert z.shape == (2, 32, 8, 8)


class TestPixelDecoder:
    """Tests for the pixel decoder."""

    def test_forward_reconstructs(self):
        """Decoder reconstructs image from spatial latent features."""
        # Decoder expects 4D input: (B, C, H, W)
        # upscale_factor=8 means 8x8 spatial -> 64x64 output
        decoder = DinoV3PixelDecoder(
            in_chans=32,
            out_chans=3,
            upscale_factor=8,
            embed_dim=128,
            depth=2,
            num_heads=4,
            ffn_layer="mlp",
            norm_layer="layernorm",
        )
        decoder.eval()
        # Input: (B, in_chans, H, W) — spatial latent
        z = torch.randn(2, 32, 8, 8)
        with torch.no_grad():
            recon = decoder(z)
        # Output: (B, 3, H*upscale, W*upscale) = (2, 3, 64, 64)
        assert recon.shape == (2, 3, 64, 64)

    def test_gradient_flows(self):
        """Gradients flow through the decoder."""
        decoder = DinoV3PixelDecoder(
            in_chans=32,
            out_chans=3,
            upscale_factor=8,
            embed_dim=128,
            depth=2,
            num_heads=4,
            ffn_layer="mlp",
            norm_layer="layernorm",
        )
        decoder.train()
        z = torch.randn(2, 32, 8, 8)
        recon = decoder(z)
        loss = recon.mean()
        loss.backward()
        has_grad = any(p.grad is not None for p in decoder.parameters() if p.requires_grad)
        assert has_grad

    def test_encoder_decoder_pipeline(self):
        """Full encode -> decode pipeline produces correct output shape."""
        encoder = DinoVisionTransformerWithBottleneck(
            img_size=64,
            patch_size=8,
            embed_dim=128,
            depth=2,
            num_heads=4,
            ffn_layer="mlp",
            norm_layer="layernorm",
            vit_feature_bottleneck=32,
        )
        decoder = DinoV3PixelDecoder(
            in_chans=32,
            out_chans=3,
            upscale_factor=8,
            embed_dim=128,
            depth=2,
            num_heads=4,
            ffn_layer="mlp",
            norm_layer="layernorm",
        )
        encoder.eval()
        decoder.eval()

        img = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            z = encoder.encode(img)           # (2, 32, 8, 8)
            recon = decoder(z)                 # (2, 3, 64, 64)
        assert recon.shape == img.shape
