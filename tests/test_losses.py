"""Tests for loss modules — alignment, reconstruction, KL, GAN."""

import pytest
import torch

from omnitok.registry import ALIGNMENT_REGISTRY, LOSS_REGISTRY
from omnitok.losses.alignment.cosine import CosineAlignmentLoss
from omnitok.losses.alignment.relational import RelationalKDLoss
from omnitok.losses.alignment.prediction import PredictionAlignmentLoss
from omnitok.losses.kl import KLLoss
from omnitok.losses.gaussianity import GaussianityLoss


class TestCosineAlignmentLoss:
    """Tests for cosine alignment loss (from REPA-E)."""

    def test_output_is_scalar(self):
        loss_fn = CosineAlignmentLoss()
        s = torch.randn(2, 16, 64)
        t = torch.randn(2, 16, 64)
        loss = loss_fn(s, t)
        assert loss.ndim == 0

    def test_identical_features_low_loss(self):
        """Identical features should have very low loss."""
        loss_fn = CosineAlignmentLoss()
        x = torch.randn(2, 16, 64)
        loss = loss_fn(x, x)
        assert loss.item() < -0.9  # Negative cosine, close to -1

    def test_gradient_flows_to_student(self):
        """Gradient flows to student but not teacher."""
        loss_fn = CosineAlignmentLoss()
        s = torch.randn(2, 16, 64, requires_grad=True)
        t = torch.randn(2, 16, 64)
        loss = loss_fn(s, t)
        loss.backward()
        assert s.grad is not None


class TestRelationalKDLoss:
    """Tests for relational KD alignment loss (from VA-VAE)."""

    def test_output_is_scalar(self):
        loss_fn = RelationalKDLoss()
        s = torch.randn(2, 16, 32)
        t = torch.randn(2, 16, 64)
        loss = loss_fn(s, t)
        assert loss.ndim == 0

    def test_different_feature_dims(self):
        """Should work with different student/teacher dims."""
        loss_fn = RelationalKDLoss()
        s = torch.randn(2, 8, 32)
        t = torch.randn(2, 8, 128)
        loss = loss_fn(s, t)
        assert loss.item() >= 0

    def test_identical_structure_low_loss(self):
        """Same features should have near-zero relational loss."""
        loss_fn = RelationalKDLoss()
        x = torch.randn(4, 8, 64)
        loss = loss_fn(x, x)
        assert loss.item() < 0.1


class TestPredictionAlignmentLoss:
    """Tests for prediction alignment loss (from MAETok)."""

    def test_output_is_scalar(self):
        loss_fn = PredictionAlignmentLoss(student_dim=32, teacher_dim=64, hidden_dim=48)
        s = torch.randn(2, 16, 32)
        t = torch.randn(2, 16, 64)
        loss = loss_fn(s, t)
        assert loss.ndim == 0

    def test_gradient_flows_through_predictor(self):
        """Gradient flows through the predictor MLP."""
        loss_fn = PredictionAlignmentLoss(student_dim=32, teacher_dim=64)
        s = torch.randn(2, 16, 32)
        t = torch.randn(2, 16, 64)
        loss = loss_fn(s, t)
        loss.backward()
        has_grad = any(p.grad is not None for p in loss_fn.predictor.parameters())
        assert has_grad


class TestKLLoss:
    """Tests for KL divergence loss."""

    def test_output_dict(self):
        loss_fn = KLLoss(weight=1e-4)
        mean = torch.randn(4, 32)
        logvar = torch.randn(4, 32)
        result = loss_fn(mean, logvar)
        assert "total" in result and "kl_raw" in result

    def test_zero_mean_unit_var_low_loss(self):
        """N(0,1) posterior should have near-zero KL."""
        loss_fn = KLLoss(weight=1.0)
        mean = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)
        result = loss_fn(mean, logvar)
        assert result["kl_raw"].item() < 0.01


class TestAlignmentRegistry:
    """Tests for alignment loss registration."""

    def test_cosine_registered(self):
        assert "cosine" in ALIGNMENT_REGISTRY

    def test_relational_kd_registered(self):
        assert "relational_kd" in ALIGNMENT_REGISTRY

    def test_prediction_registered(self):
        assert "prediction" in ALIGNMENT_REGISTRY

    def test_mse_registered(self):
        assert "mse" in ALIGNMENT_REGISTRY

    def test_smooth_l1_registered(self):
        assert "smooth_l1" in ALIGNMENT_REGISTRY

    def test_build_cosine(self):
        loss = ALIGNMENT_REGISTRY.build("cosine")
        assert isinstance(loss, CosineAlignmentLoss)


class TestLossRegistry:
    """Tests for main loss registration."""

    def test_reconstruction_registered(self):
        assert "reconstruction" in LOSS_REGISTRY

    def test_kl_registered(self):
        assert "kl" in LOSS_REGISTRY

    def test_gan_registered(self):
        assert "gan" in LOSS_REGISTRY

    def test_gaussianity_registered(self):
        assert "gaussianity" in LOSS_REGISTRY


class TestGaussianityLoss:
    """Tests for Gaussianity regularization loss (L_gauss from UNE)."""

    def test_output_dict(self):
        loss_fn = GaussianityLoss(weight=1e-4)
        z = torch.randn(8, 64)
        result = loss_fn(z)
        assert "total" in result
        assert "cov_loss" in result
        assert "mean_loss" in result

    def test_identity_cov_low_loss(self):
        """Whitened data should have low cov loss."""
        loss_fn = GaussianityLoss(weight=1.0)
        # Generate whitened data: mean=0, cov≈I
        z = torch.randn(1000, 16)  # large N for good cov estimate
        result = loss_fn(z)
        assert result["cov_loss"].item() < 0.5

    def test_correlated_data_high_loss(self):
        """Correlated data should have higher loss than whitened data."""
        loss_fn = GaussianityLoss(weight=1.0)
        # Whitened baseline
        z_white = torch.randn(200, 16)
        result_white = loss_fn(z_white)
        # Correlated
        base = torch.randn(200, 1)
        z_corr = base.expand(-1, 16) + torch.randn(200, 16) * 0.01
        result_corr = loss_fn(z_corr)
        assert result_corr["cov_loss"].item() > result_white["cov_loss"].item()

    def test_4d_input(self):
        """Should handle (B, C, h, w) spatial latents."""
        loss_fn = GaussianityLoss()
        z = torch.randn(4, 32, 4, 4)
        result = loss_fn(z)
        assert result["total"].ndim == 0

    def test_3d_input(self):
        """Should handle (B, L, D) token sequences."""
        loss_fn = GaussianityLoss()
        z = torch.randn(4, 16, 64)
        result = loss_fn(z)
        assert result["total"].ndim == 0

    def test_gradient_flows(self):
        loss_fn = GaussianityLoss(weight=1.0)
        z = torch.randn(16, 32, requires_grad=True)
        result = loss_fn(z)
        result["total"].backward()
        assert z.grad is not None

    def test_no_mean_penalty(self):
        loss_fn = GaussianityLoss(weight=1.0, mean_penalty=False)
        z = torch.randn(16, 32)
        result = loss_fn(z)
        assert "mean_loss" not in result
