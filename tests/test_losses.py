"""Tests for loss modules — alignment, reconstruction, KL, GAN."""

import pytest
import torch

from omnitok.registry import ALIGNMENT_REGISTRY, LOSS_REGISTRY
from omnitok.losses.alignment.cosine import CosineAlignmentLoss
from omnitok.losses.alignment.relational import RelationalKDLoss
from omnitok.losses.alignment.prediction import PredictionAlignmentLoss
from omnitok.losses.kl import KLLoss


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
