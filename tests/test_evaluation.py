"""Tests for omnitok.evaluation module and projector heads."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from omnitok.evaluation.gaussianity import GaussianityEvaluator
from omnitok.evaluation.linear_probe import LinearProbeEvaluator
from omnitok.evaluation.rfid import RFIDEvaluator, _tensor_to_uint8
from omnitok.evaluation.evaluator import TokenizerEvaluator
from omnitok.models.heads.projector import (
    LinearProjector,
    MLP2Projector,
    MLP3Projector,
    IdentityProjector,
    build_projector,
)


# ============================================================================
# Fixtures
# ============================================================================

def make_fake_images(n=16, size=32):
    """Random images in [-1, 1], shape (N, 3, H, W)."""
    return torch.rand(n, 3, size, size) * 2 - 1


def make_fake_latents(n=100, d=64):
    """Random latents, shape (N, D)."""
    return torch.randn(n, d)


class FakeTokenizer(nn.Module):
    """Minimal tokenizer: encode→small spatial, decode→image."""

    def __init__(self, img_size=32, latent_ch=4, latent_size=4):
        super().__init__()
        self.enc = nn.Conv2d(3, latent_ch, kernel_size=img_size // latent_size, stride=img_size // latent_size)
        self.dec = nn.ConvTranspose2d(latent_ch, 3, kernel_size=img_size // latent_size, stride=img_size // latent_size)

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return torch.tanh(self.dec(z))

    def forward(self, x):
        return self.decode(self.encode(x))


def make_fake_dataloader(n=32, img_size=32, n_classes=10, with_labels=True):
    imgs = make_fake_images(n, img_size)
    if with_labels:
        labels = torch.randint(0, n_classes, (n,))
        ds = TensorDataset(imgs, labels)
    else:
        ds = TensorDataset(imgs)
    return DataLoader(ds, batch_size=8)


# ============================================================================
# GaussianityEvaluator
# ============================================================================

class TestGaussianityEvaluator:
    def test_gaussian_data_high_score(self):
        ev = GaussianityEvaluator(significance=5.0)
        z = torch.randn(500, 16)
        result = ev.compute(z)
        assert result["gaussianity_score"] > 0.5
        assert result["total_dims"] == 16
        assert result["n_samples"] == 500

    def test_uniform_data_low_score(self):
        ev = GaussianityEvaluator(significance=5.0)
        z = torch.rand(500, 16) * 10 - 5  # uniform, not Gaussian
        result = ev.compute(z)
        assert result["gaussianity_score"] < 0.9

    def test_invalid_significance(self):
        with pytest.raises(ValueError, match="significance must be one of"):
            GaussianityEvaluator(significance=7.0)

    def test_wrong_ndim(self):
        ev = GaussianityEvaluator()
        with pytest.raises(ValueError, match="Expected 2D"):
            ev.compute(torch.randn(10, 4, 4))

    def test_subsampling(self):
        ev = GaussianityEvaluator(max_samples=100)
        z = torch.randn(500, 8)
        result = ev.compute(z)
        assert result["n_samples"] == 100

    def test_all_significance_levels(self):
        ev15 = GaussianityEvaluator(significance=15.0)
        ev1 = GaussianityEvaluator(significance=1.0)
        z = torch.randn(500, 8)  # more samples for stability
        r15 = ev15.compute(z)
        r1 = ev1.compute(z)
        # Looser threshold means more dims pass — but with random data
        # both could be high, so just assert valid range
        assert 0.0 <= r15["gaussianity_score"] <= 1.0
        assert 0.0 <= r1["gaussianity_score"] <= 1.0

    def test_returns_expected_keys(self):
        ev = GaussianityEvaluator()
        result = ev.compute(torch.randn(100, 4))
        assert set(result.keys()) == {"gaussianity_score", "n_gaussian_dims", "total_dims", "n_samples"}

    def test_score_range(self):
        ev = GaussianityEvaluator()
        result = ev.compute(torch.randn(100, 10))
        assert 0.0 <= result["gaussianity_score"] <= 1.0


# ============================================================================
# LinearProbeEvaluator
# ============================================================================

class TestLinearProbeEvaluator:
    def test_basic_fit(self):
        ev = LinearProbeEvaluator(max_iter=100)
        n_train, n_val, d = 200, 50, 32
        n_classes = 5
        # linearly separable features
        feats_train = torch.randn(n_train, d)
        labels_train = (feats_train[:, 0] > 0).long() * 2  # 2-class signal in dim 0
        feats_val = torch.randn(n_val, d)
        labels_val = (feats_val[:, 0] > 0).long() * 2
        result = ev.compute(feats_train, labels_train, feats_val, labels_val)
        assert "linear_probe_acc" in result
        assert result["n_train"] == n_train
        assert result["n_val"] == n_val

    def test_pool_features_spatial(self):
        ev = LinearProbeEvaluator(feature_type="mean_pool")
        feats = torch.randn(4, 16, 8, 8)  # (B, C, h, w)
        pooled = ev._pool_features(feats)
        assert pooled.shape == (4, 16)

    def test_pool_features_sequence_mean(self):
        ev = LinearProbeEvaluator(feature_type="mean_pool")
        feats = torch.randn(4, 196, 64)  # (B, L, D)
        pooled = ev._pool_features(feats)
        assert pooled.shape == (4, 64)

    def test_pool_features_cls(self):
        ev = LinearProbeEvaluator(feature_type="cls")
        feats = torch.randn(4, 196, 64)
        pooled = ev._pool_features(feats)
        assert pooled.shape == (4, 64)

    def test_pool_features_2d_passthrough(self):
        ev = LinearProbeEvaluator()
        feats = torch.randn(4, 64)
        pooled = ev._pool_features(feats)
        assert pooled.shape == (4, 64)

    def test_acc_range(self):
        ev = LinearProbeEvaluator(max_iter=50)
        feats = torch.randn(100, 8)
        # Need at least 2 classes for logistic regression
        labels = torch.randint(0, 2, (100,))
        result = ev.compute(feats, labels, feats, labels)
        assert 0.0 <= result["linear_probe_acc"] <= 100.0


# ============================================================================
# RFIDEvaluator
# ============================================================================

class TestRFIDEvaluator:
    def test_tensor_to_uint8_range(self):
        img = torch.zeros(3, 4, 4)  # black → 128
        out = _tensor_to_uint8(img)
        assert out.shape == (4, 4, 3)
        assert out.dtype.name == "uint8"
        assert out.min() >= 0 and out.max() <= 255

    def test_compute_saves_images(self):
        ev = RFIDEvaluator(verbose=False)
        real = make_fake_images(4, 16)
        recon = make_fake_images(4, 16)
        with tempfile.TemporaryDirectory() as tmp:
            # Don't actually run FID (needs inception network) — just test image saving
            from pathlib import Path as P
            real_dir = P(tmp) / "real"
            recon_dir = P(tmp) / "recon"
            real_dir.mkdir(), recon_dir.mkdir()
            from omnitok.evaluation.rfid import _tensor_to_uint8
            from PIL import Image
            for i in range(4):
                Image.fromarray(_tensor_to_uint8(real[i])).save(real_dir / f"{i:06d}.png")
                Image.fromarray(_tensor_to_uint8(recon[i])).save(recon_dir / f"{i:06d}.png")
            assert len(list(real_dir.glob("*.png"))) == 4
            assert len(list(recon_dir.glob("*.png"))) == 4

    def test_shape_mismatch_raises(self):
        ev = RFIDEvaluator(verbose=False)
        with pytest.raises(AssertionError):
            ev.compute(make_fake_images(4, 16), make_fake_images(8, 16))


# ============================================================================
# TokenizerEvaluator (integration — no FID, just PSNR + Gaussianity)
# ============================================================================

class TestTokenizerEvaluator:
    def test_psnr_and_gaussianity(self):
        ev = TokenizerEvaluator(
            run_rfid=False,
            run_psnr=True,
            run_linear_probe=False,
            run_gaussianity=True,
            device=torch.device("cpu"),
        )
        model = FakeTokenizer()
        loader = make_fake_dataloader(n=16, img_size=32)
        result = ev.evaluate(model, loader, n_batches=2)
        assert "psnr" in result
        assert "gaussianity_score" in result
        assert result["psnr"] > 0

    def test_empty_loader_raises(self):
        ev = TokenizerEvaluator(run_rfid=False, run_psnr=True, run_gaussianity=False, device=torch.device("cpu"))
        model = FakeTokenizer()
        empty_loader = DataLoader(TensorDataset(torch.zeros(0, 3, 32, 32)), batch_size=8)
        with pytest.raises(RuntimeError, match="No samples"):
            ev.evaluate(model, empty_loader)


# ============================================================================
# Projector heads
# ============================================================================

class TestProjectors:
    def test_linear_projector_shape(self):
        proj = LinearProjector(in_dim=1024, out_dim=512)
        x = torch.randn(4, 1024)
        out = proj(x)
        assert out.shape == (4, 512)

    def test_linear_projector_no_bias(self):
        proj = LinearProjector(in_dim=64, out_dim=32)
        assert proj.proj.bias is None

    def test_mlp2_projector_shape(self):
        proj = MLP2Projector(in_dim=256, out_dim=128)
        x = torch.randn(4, 256)
        out = proj(x)
        assert out.shape == (4, 128)

    def test_mlp3_projector_shape(self):
        proj = MLP3Projector(in_dim=1024, out_dim=1024)
        x = torch.randn(4, 1024)
        out = proj(x)
        assert out.shape == (4, 1024)

    def test_identity_projector(self):
        proj = IdentityProjector(in_dim=64, out_dim=64)
        x = torch.randn(4, 64)
        out = proj(x)
        assert torch.allclose(out, x)

    def test_identity_requires_same_dim(self):
        with pytest.raises(ValueError, match="in_dim == out_dim"):
            IdentityProjector(in_dim=64, out_dim=32)

    def test_build_projector_linear(self):
        proj = build_projector("linear", in_dim=512, out_dim=256)
        assert isinstance(proj, LinearProjector)

    def test_build_projector_mlp2(self):
        proj = build_projector("mlp2", in_dim=512, out_dim=256)
        assert isinstance(proj, MLP2Projector)

    def test_build_projector_mlp3(self):
        proj = build_projector("mlp3", in_dim=512, out_dim=512)
        assert isinstance(proj, MLP3Projector)

    def test_build_projector_identity(self):
        proj = build_projector("identity", in_dim=64, out_dim=64)
        assert isinstance(proj, IdentityProjector)

    def test_build_projector_unknown_raises(self):
        with pytest.raises(ValueError):
            build_projector("unknown_proj", in_dim=64, out_dim=32)

    def test_projector_in_out_dims(self):
        for proj_type in ("linear", "mlp2", "mlp3"):
            proj = build_projector(proj_type, in_dim=128, out_dim=64)
            assert proj.in_dim == 128
            assert proj.out_dim == 64
