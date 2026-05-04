"""Tests for SparseTeacherRouter — sparse multi-teacher routing."""

import torch
import pytest

from omnitok.teachers.sparse_router import SparseTeacherRouter, TeacherRoutingResult


class TestSparseTeacherRouter:
    """Unit tests for SparseTeacherRouter."""

    def test_basic_forward(self):
        """Router produces valid routing result with correct shapes."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=1)
        x = torch.randn(4, 256, 768)
        result = router(x)

        assert isinstance(result, TeacherRoutingResult)
        assert result.selected_indices.shape == (4, 1)
        assert result.gating_weights.shape == (4, 1)
        assert result.load_balance_loss.shape == (1,) or result.load_balance_loss.ndim == 0
        assert result.router_logits.shape == (4, 3)

    def test_top_k_2(self):
        """Router selects 2 teachers per sample."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=2)
        x = torch.randn(4, 256, 768)
        result = router(x)

        assert result.selected_indices.shape == (4, 2)
        assert result.gating_weights.shape == (4, 2)

    def test_top_k_equals_num_teachers(self):
        """When top_k == num_teachers, all teachers are selected (dense mode)."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=3)
        x = torch.randn(4, 256, 768)
        result = router(x)

        assert result.selected_indices.shape == (4, 3)
        # All indices should be present per sample
        for i in range(4):
            assert set(result.selected_indices[i].tolist()) == {0, 1, 2}

    def test_top_k_exceeds_num_teachers_raises(self):
        """top_k > num_teachers should raise ValueError."""
        with pytest.raises(ValueError, match="top_k.*cannot exceed"):
            SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=5)

    def test_gating_weights_sum_to_one(self):
        """Gating weights should sum to 1.0 per sample."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=2)
        x = torch.randn(4, 256, 768)
        result = router(x)

        weight_sums = result.gating_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones(4), atol=1e-5)

    def test_selected_indices_valid_range(self):
        """Selected indices should be in [0, num_teachers)."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=5, top_k=2)
        x = torch.randn(8, 256, 768)
        result = router(x)

        assert (result.selected_indices >= 0).all()
        assert (result.selected_indices < 5).all()

    def test_load_balance_loss_is_scalar(self):
        """Load balance loss should be a scalar tensor with gradient."""
        router = SparseTeacherRouter(
            student_dim=768, num_teachers=3, top_k=1, load_balance_weight=0.01
        )
        x = torch.randn(4, 256, 768)
        result = router(x)

        assert result.load_balance_loss.ndim <= 1
        assert not torch.isnan(result.load_balance_loss)
        assert not torch.isinf(result.load_balance_loss)

    def test_load_balance_disabled(self):
        """Load balance loss should be zero when both aux weights are 0."""
        router = SparseTeacherRouter(
            student_dim=768, num_teachers=3, top_k=1,
            load_balance_weight=0.0, z_loss_weight=0.0,
        )
        x = torch.randn(4, 256, 768)
        result = router(x)

        assert result.load_balance_loss.item() == 0.0

    def test_gradient_flow(self):
        """Router gate should have gradients after backward."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=2)
        x = torch.randn(4, 256, 768, requires_grad=True)
        result = router(x)

        # Simulate loss from gating weights
        loss = result.gating_weights.sum() + result.load_balance_loss
        loss.backward()

        assert router.gate.weight.grad is not None

    def test_2d_input(self):
        """Router should accept 2D input (B, D) without pooling."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=1)
        x = torch.randn(4, 768)
        result = router(x)

        assert result.selected_indices.shape == (4, 1)

    def test_teacher_usage(self):
        """Teacher usage should sum to top_k / num_teachers when balanced."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=1)
        x = torch.randn(100, 256, 768)
        result = router(x)

        usage = router.get_teacher_usage(result.selected_indices)
        assert usage.shape == (3,)
        # Usage should sum to 1.0 (fraction, not count)
        assert torch.allclose(usage.sum(), torch.tensor(1.0), atol=1e-5)

    def test_temperature_effect(self):
        """Lower temperature should produce more peaked (less uniform) distributions."""
        x = torch.randn(100, 256, 768)

        router_high_temp = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=2, temperature=10.0)
        router_low_temp = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=2, temperature=0.1)
        # Copy same weights
        router_low_temp.gate.weight.data.copy_(router_high_temp.gate.weight.data)

        result_high = router_high_temp(x)
        result_low = router_low_temp(x)

        # Low temperature → more peaked gating weights (max weight closer to 1.0)
        max_high = result_high.gating_weights.max(dim=-1).values.mean()
        max_low = result_low.gating_weights.max(dim=-1).values.mean()
        assert max_low > max_high

    def test_deterministic(self):
        """Router should be deterministic in eval mode."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=1)
        router.eval()
        x = torch.randn(4, 256, 768)

        result1 = router(x)
        result2 = router(x)

        assert torch.equal(result1.selected_indices, result2.selected_indices)
        assert torch.allclose(result1.gating_weights, result2.gating_weights)

    def test_z_loss_penalizes_large_logits(self):
        """Z-loss should increase with larger router logits."""
        router = SparseTeacherRouter(
            student_dim=768, num_teachers=3, top_k=1,
            load_balance_weight=0.0, z_loss_weight=0.01,
        )
        x_small = torch.randn(4, 256, 768) * 0.01
        x_large = torch.randn(4, 256, 768) * 10.0

        result_small = router(x_small)
        result_large = router(x_large)

        assert result_large.load_balance_loss > result_small.load_balance_loss

    def test_routing_metrics_complete(self):
        """get_routing_metrics should return all expected keys."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=2)
        x = torch.randn(8, 256, 768)
        result = router(x)

        metrics = router.get_routing_metrics(result, ["dinov2", "siglip", "sam"])

        expected_keys = [
            "router/usage_dinov2", "router/usage_siglip", "router/usage_sam",
            "router/entropy", "router/entropy_ratio",
            "router/max_prob",
            "router/weight_dinov2", "router/weight_siglip", "router/weight_sam",
            "router/balance_score",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
            assert isinstance(metrics[key], float)

    def test_balance_score_range(self):
        """Balance score should be between 0.0 and 1.0."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=1)
        x = torch.randn(32, 256, 768)
        result = router(x)

        metrics = router.get_routing_metrics(result, ["t0", "t1", "t2"])
        assert 0.0 <= metrics["router/balance_score"] <= 1.0
        assert 0.0 <= metrics["router/entropy_ratio"] <= 1.0
        assert 0.0 <= metrics["router/max_prob"] <= 1.0

    def test_router_probs_in_result(self):
        """TeacherRoutingResult should include router_probs for logging."""
        router = SparseTeacherRouter(student_dim=768, num_teachers=3, top_k=1)
        x = torch.randn(4, 256, 768)
        result = router(x)

        assert result.router_probs.shape == (4, 3)
        assert torch.allclose(result.router_probs.sum(dim=-1), torch.ones(4), atol=1e-5)


class TestMultiTeacherExtractSelected:
    """Tests for MultiTeacher.extract_selected integration."""

    def test_extract_selected_basic(self):
        """extract_selected should only return features for selected teachers."""
        from omnitok.teachers.base import BaseTeacher
        from omnitok.teachers.multi_teacher import MultiTeacher

        class DummyTeacher(BaseTeacher):
            def __init__(self, dim: int):
                super().__init__(model_name="dummy")
                self._dim = dim

            @property
            def feature_dim(self) -> int:
                return self._dim

            @property
            def patch_size(self) -> int:
                return 16

            def _build_model(self):
                return torch.nn.Identity()

            def _extract_features(self, x):
                return torch.randn(x.shape[0], 16, self._dim)

            def forward(self, x):
                return self._extract_features(x)

        teachers = MultiTeacher(
            {
                "t0": DummyTeacher(768),
                "t1": DummyTeacher(1152),
                "t2": DummyTeacher(256),
            },
            normalize=False,
            phi_s_balancing=False,
        )

        images = torch.randn(4, 3, 64, 64)
        # Select only teacher 0 and 2
        selected = torch.tensor([[0, 2], [0, 2], [1, 2], [0, 1]])
        features = teachers.extract_selected(images, selected)

        # All 3 should be extracted because unique indices = {0, 1, 2}
        assert len(features) == 3

    def test_extract_selected_subset(self):
        """extract_selected with only 1 unique teacher should extract 1."""
        from omnitok.teachers.base import BaseTeacher
        from omnitok.teachers.multi_teacher import MultiTeacher

        class DummyTeacher(BaseTeacher):
            def __init__(self, dim: int):
                super().__init__(model_name="dummy")
                self._dim = dim
                self.call_count = 0

            @property
            def feature_dim(self) -> int:
                return self._dim

            @property
            def patch_size(self) -> int:
                return 16

            def _build_model(self):
                return torch.nn.Identity()

            def _extract_features(self, x):
                self.call_count += 1
                return torch.randn(x.shape[0], 16, self._dim)

            def forward(self, x):
                return self._extract_features(x)

        t0 = DummyTeacher(768)
        t1 = DummyTeacher(1152)
        t2 = DummyTeacher(256)

        teachers = MultiTeacher(
            {"t0": t0, "t1": t1, "t2": t2},
            normalize=False,
            phi_s_balancing=False,
        )

        images = torch.randn(4, 3, 64, 64)
        # All samples select only teacher 0
        selected = torch.tensor([[0], [0], [0], [0]])
        features = teachers.extract_selected(images, selected)

        assert len(features) == 1
        assert "t0" in features
        assert t0.call_count == 1
        assert t1.call_count == 0
        assert t2.call_count == 0
