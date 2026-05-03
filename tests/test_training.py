"""Tests for training utilities and tokenizer model."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from omnitok.models.decoder.pixel_decoder import DinoV3PixelDecoder
from omnitok.models.encoder.vision_transformer_bottleneck import DinoVisionTransformerWithBottleneck
from omnitok.models.tokenizer import Tokenizer
from omnitok.training.utils import (
    count_params,
    fix_seeds,
    load_checkpoint,
    requires_grad,
    save_checkpoint,
    update_ema,
)


@pytest.fixture
def small_tokenizer():
    """Create a minimal tokenizer for testing."""
    encoder = DinoVisionTransformerWithBottleneck(
        img_size=64, patch_size=8, embed_dim=128, depth=2,
        num_heads=4, ffn_layer="mlp", norm_layer="layernorm",
        vit_feature_bottleneck=32,
    )
    decoder = DinoV3PixelDecoder(
        in_chans=32, out_chans=3, upscale_factor=8,
        embed_dim=128, depth=2, num_heads=4,
        ffn_layer="mlp", norm_layer="layernorm",
    )
    return Tokenizer(encoder=encoder, decoder=decoder)


class TestTokenizerModel:
    """Tests for the Tokenizer composition model."""

    def test_forward_returns_dict(self, small_tokenizer):
        """Forward produces reconstruction and latent."""
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = small_tokenizer(x)
        assert "reconstruction" in out
        assert "latent" in out
        assert out["reconstruction"].shape == (2, 3, 64, 64)
        assert out["latent"].shape == (2, 32, 8, 8)

    def test_forward_with_features(self, small_tokenizer):
        """Forward with return_features=True includes features dict."""
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = small_tokenizer(x, return_features=True)
        assert "features" in out
        assert "x_norm_patchtokens" in out["features"]

    def test_encode_decode(self, small_tokenizer):
        """Separate encode/decode produces correct shapes."""
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            z = small_tokenizer.encode(x)
            recon = small_tokenizer.decode(z)
        assert z.shape == (2, 32, 8, 8)
        assert recon.shape == (2, 3, 64, 64)

    def test_gradient_flows(self, small_tokenizer):
        """Gradients flow through full tokenizer."""
        small_tokenizer.train()
        x = torch.randn(2, 3, 64, 64)
        out = small_tokenizer(x)
        loss = out["reconstruction"].mean()
        loss.backward()
        # Check encoder has grads
        enc_grad = any(p.grad is not None for p in small_tokenizer.encoder.parameters() if p.requires_grad)
        dec_grad = any(p.grad is not None for p in small_tokenizer.decoder.parameters() if p.requires_grad)
        assert enc_grad and dec_grad


class TestEMA:
    """Tests for EMA utility."""

    def test_ema_moves_toward_model(self):
        """EMA params move toward model params."""
        model = nn.Linear(10, 10)
        ema = nn.Linear(10, 10)
        # Set different weights
        with torch.no_grad():
            ema.weight.fill_(0.0)
            model.weight.fill_(1.0)

        update_ema(ema, model, decay=0.5)

        # EMA should now be 0.5 * 0 + 0.5 * 1 = 0.5
        assert torch.allclose(ema.weight, torch.full_like(ema.weight, 0.5))


class TestRequiresGrad:
    """Tests for requires_grad utility."""

    def test_freeze(self):
        model = nn.Linear(10, 10)
        requires_grad(model, False)
        assert not any(p.requires_grad for p in model.parameters())

    def test_unfreeze(self):
        model = nn.Linear(10, 10)
        requires_grad(model, False)
        requires_grad(model, True)
        assert all(p.requires_grad for p in model.parameters())


class TestCheckpoint:
    """Tests for save/load checkpoint."""

    def test_save_and_load(self):
        """Save then load preserves model state."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # Do a forward+backward to create optimizer state
        x = torch.randn(2, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(model, None, optimizer, step=100, epoch=5, save_dir=tmpdir)

            # Create fresh model + optimizer
            model2 = nn.Linear(10, 5)
            opt2 = torch.optim.Adam(model2.parameters())

            state = load_checkpoint(
                os.path.join(tmpdir, "last.pt"),
                model2, optimizer=opt2,
            )

            assert state["step"] == 100
            assert state["epoch"] == 5
            assert torch.equal(model.weight.data, model2.weight.data)


class TestCountParams:
    """Tests for count_params."""

    def test_counts(self):
        model = nn.Linear(10, 5, bias=False)
        info = count_params(model)
        assert info["total"] == 50
        assert info["trainable"] == 50
        assert info["frozen"] == 0

    def test_frozen_params(self):
        model = nn.Linear(10, 5, bias=False)
        for p in model.parameters():
            p.requires_grad = False
        info = count_params(model)
        assert info["trainable"] == 0
        assert info["frozen"] == 50


class TestFixSeeds:
    """Tests for fix_seeds."""

    def test_deterministic(self):
        fix_seeds(42)
        a = torch.randn(3)
        fix_seeds(42)
        b = torch.randn(3)
        assert torch.equal(a, b)
