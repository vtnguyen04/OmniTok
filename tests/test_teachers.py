"""Tests for teacher modules — normalizer, multi-teacher, registry."""

import torch

from omnitok.registry import TEACHER_REGISTRY
from omnitok.teachers.base import BaseTeacher
from omnitok.teachers.multi_teacher import MultiTeacher
from omnitok.teachers.normalizer import FeatureNormalizer, ProjectedNormalizer

# --- Fake teacher for testing (no model download) ---

class FakeTeacher(BaseTeacher):
    """Minimal teacher for unit testing — returns random features."""

    def __init__(self, feat_dim: int = 64, p_size: int = 16) -> None:
        super().__init__(model_name="fake", device=None)
        self._feat_dim = feat_dim
        self._p_size = p_size
        self._model = torch.nn.Identity()  # Skip setup

    def _build_model(self):
        return torch.nn.Identity()

    def _extract_features(self, x):
        B = x.shape[0]
        N = (x.shape[-1] // self._p_size) ** 2
        return torch.randn(B, N, self._feat_dim)

    @property
    def feature_dim(self) -> int:
        return self._feat_dim

    @property
    def patch_size(self) -> int:
        return self._p_size


class TestFeatureNormalizer:
    """Tests for FeatureNormalizer."""

    def test_output_shape(self):
        """Normalizer preserves shape."""
        norm = FeatureNormalizer(64)
        x = torch.randn(4, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_running_stats_update(self):
        """Running stats change after forward pass in training mode."""
        norm = FeatureNormalizer(32)
        norm.train()
        x = torch.randn(4, 16, 32) * 5 + 3  # Non-zero mean/var
        norm(x)
        assert norm.num_batches_tracked.item() == 1
        assert not torch.allclose(norm.running_mean, torch.zeros(32), atol=1e-3)

    def test_eval_no_update(self):
        """Running stats don't change in eval mode."""
        norm = FeatureNormalizer(32)
        norm.eval()
        mean_before = norm.running_mean.clone()
        x = torch.randn(4, 16, 32) * 10
        norm(x)
        assert torch.equal(norm.running_mean, mean_before)


class TestProjectedNormalizer:
    """Tests for ProjectedNormalizer."""

    def test_projects_dimension(self):
        """Projector changes feature dim."""
        proj = ProjectedNormalizer(in_dim=128, out_dim=64)
        x = torch.randn(2, 16, 128)
        out = proj(x)
        assert out.shape == (2, 16, 64)


class TestMultiTeacher:
    """Tests for MultiTeacher."""

    def test_extract_all(self):
        """MultiTeacher extracts features from all teachers."""
        teachers = {
            "t1": FakeTeacher(feat_dim=64),
            "t2": FakeTeacher(feat_dim=128),
        }
        multi = MultiTeacher(teachers, common_dim=32)
        x = torch.randn(2, 3, 64, 64)
        feats = multi.extract_all(x)
        assert "t1" in feats and "t2" in feats
        # Both projected to common_dim=32
        assert feats["t1"].shape[-1] == 32
        assert feats["t2"].shape[-1] == 32

    def test_loss_weights(self):
        """PHI-S loss weights are positive scalars."""
        teachers = {"a": FakeTeacher(), "b": FakeTeacher()}
        multi = MultiTeacher(teachers)
        weights = multi.get_loss_weights()
        assert len(weights) == 2
        for w in weights.values():
            assert w.item() > 0

    def test_regularization(self):
        """Regularization term is scalar."""
        teachers = {"a": FakeTeacher()}
        multi = MultiTeacher(teachers)
        reg = multi.get_regularization()
        assert reg.ndim == 0 or (hasattr(reg, 'shape') and reg.shape == torch.Size([1]))

    def test_teachers_frozen_in_train_mode(self):
        """Teachers stay frozen even when MultiTeacher is in train mode."""
        teachers = {"t": FakeTeacher()}
        multi = MultiTeacher(teachers)
        multi.train()
        assert not multi.teachers["t"].training

    def test_num_teachers(self):
        """num_teachers property works."""
        teachers = {"a": FakeTeacher(), "b": FakeTeacher(), "c": FakeTeacher()}
        multi = MultiTeacher(teachers)
        assert multi.num_teachers == 3


class TestTeacherRegistry:
    """Tests for teacher registration."""

    def test_dinov2_registered(self):
        """DINOv2 teacher is registered."""
        assert "dinov2" in TEACHER_REGISTRY

    def test_siglip_registered(self):
        """SigLIP teacher is registered."""
        assert "siglip" in TEACHER_REGISTRY

    def test_sam_registered(self):
        """SAM teacher is registered."""
        assert "sam" in TEACHER_REGISTRY

    def test_depth_anything_registered(self):
        """Depth Anything teacher is registered."""
        assert "depth_anything" in TEACHER_REGISTRY

    def test_hog_registered(self):
        """HOG teacher is registered."""
        assert "hog" in TEACHER_REGISTRY

    def test_available_teachers(self):
        """At least 4 teachers registered."""
        assert len(TEACHER_REGISTRY) >= 4
