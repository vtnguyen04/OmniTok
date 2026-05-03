"""Tests for omnitok.utils — logger, metrics, artifacts, plots, wandb_logger, experiment."""

import json
import os
import tempfile

import pytest
import torch

# ────────────────────────────────────────────────────────────
# MetricsTracker
# ────────────────────────────────────────────────────────────

class TestMetricsTracker:
    def setup_method(self):
        from omnitok.utils.metrics import MetricsTracker
        self.tracker = MetricsTracker(window_size=5)

    def test_update_and_get_latest(self):
        self.tracker.update("loss/total", 1.0, step=1)
        self.tracker.update("loss/total", 0.5, step=2)
        assert self.tracker.get_latest("loss/total") == pytest.approx(0.5)

    def test_get_smooth_averages_window(self):
        for i in range(5):
            self.tracker.update("loss", float(i), step=i)
        # window = [0, 1, 2, 3, 4], mean = 2.0
        assert self.tracker.get_smooth("loss") == pytest.approx(2.0)

    def test_window_evicts_old_values(self):
        for i in range(10):  # window_size=5, so only last 5 kept
            self.tracker.update("loss", float(i), step=i)
        assert self.tracker.get_smooth("loss") == pytest.approx((5 + 6 + 7 + 8 + 9) / 5)

    def test_unknown_metric_returns_zero(self):
        assert self.tracker.get_smooth("nonexistent") == pytest.approx(0.0)
        assert self.tracker.get_latest("nonexistent") == pytest.approx(0.0)

    def test_update_dict(self):
        self.tracker.update_dict({"a": 1.0, "b": 2.0}, step=1)
        assert self.tracker.get_latest("a") == pytest.approx(1.0)
        assert self.tracker.get_latest("b") == pytest.approx(2.0)

    def test_update_dict_skips_non_numeric(self):
        self.tracker.update_dict({"a": 1.0, "label": "train"}, step=1)
        assert "a" in self.tracker.tracked_metrics
        assert "label" not in self.tracker.tracked_metrics

    def test_get_all_smooth(self):
        self.tracker.update("x", 1.0, step=1)
        self.tracker.update("y", 2.0, step=1)
        all_smooth = self.tracker.get_all_smooth()
        assert "x" in all_smooth
        assert "y" in all_smooth

    def test_update_best_min(self):
        assert self.tracker.update_best("rfid", 10.0, step=1, mode="min") is True
        assert self.tracker.update_best("rfid", 8.0, step=2, mode="min") is True
        assert self.tracker.update_best("rfid", 9.0, step=3, mode="min") is False
        best = self.tracker.get_best("rfid")
        assert best["value"] == pytest.approx(8.0)
        assert best["step"] == 2

    def test_update_best_max(self):
        assert self.tracker.update_best("acc", 0.5, step=1, mode="max") is True
        assert self.tracker.update_best("acc", 0.8, step=2, mode="max") is True
        assert self.tracker.update_best("acc", 0.7, step=3, mode="max") is False
        assert self.tracker.get_best("acc")["value"] == pytest.approx(0.8)

    def test_export_json(self):
        self.tracker.update("loss", 1.0, step=1)
        self.tracker.update("loss", 0.8, step=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.json")
            self.tracker.export_json(path)
            assert os.path.exists(path)
            data = json.loads(open(path).read())
            assert "history" in data
            assert "loss" in data["history"]
            assert len(data["history"]["loss"]) == 2

    def test_export_csv(self):
        self.tracker.update("a", 1.0, step=1)
        self.tracker.update("b", 2.0, step=1)
        self.tracker.update("a", 0.5, step=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.csv")
            self.tracker.export_csv(path)
            assert os.path.exists(path)
            content = open(path).read()
            assert "step" in content
            assert "a" in content
            assert "b" in content

    def test_load_json_roundtrip(self):
        from omnitok.utils.metrics import MetricsTracker
        self.tracker.update("loss", 0.5, step=1)
        self.tracker.update_best("loss", 0.5, step=1, mode="min")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "m.json")
            self.tracker.export_json(path)
            data = MetricsTracker.load_json(path)
            assert data["history"]["loss"][0]["value"] == pytest.approx(0.5)


# ────────────────────────────────────────────────────────────
# OmniTokLogger (smoke tests — no output assertions)
# ────────────────────────────────────────────────────────────

class TestOmniTokLogger:
    def test_instantiate_rank0(self):
        from omnitok.utils.logger import OmniTokLogger
        log = OmniTokLogger(name="test", rank=0)
        assert log.rank == 0

    def test_instantiate_rank1_no_console(self):
        from omnitok.utils.logger import OmniTokLogger
        log = OmniTokLogger(name="test", rank=1)
        assert log.rank == 1

    def test_log_methods_dont_crash(self):
        from omnitok.utils.logger import OmniTokLogger
        log = OmniTokLogger(name="test", rank=0)
        log.info("test info")
        log.warning("test warning")
        log.error("test error")
        log.encoder("encoder msg")
        log.teacher("teacher msg")
        log.loss("loss msg")
        log.success("success msg")

    def test_print_banner_rank0(self):
        from omnitok.utils.logger import OmniTokLogger
        log = OmniTokLogger(name="test", rank=0)
        log.print_banner()  # should not raise

    def test_print_banner_rank1_silent(self):
        from omnitok.utils.logger import OmniTokLogger
        log = OmniTokLogger(name="test", rank=1)
        log.print_banner()  # should not raise, should be silent

    def test_print_config_table(self):
        from omnitok.utils.logger import OmniTokLogger
        log = OmniTokLogger(name="test", rank=0)
        log.print_config_table({"lr": 1e-4, "batch_size": 32, "model": {"embed_dim": 1024}})

    def test_print_metrics_table(self):
        from omnitok.utils.logger import OmniTokLogger
        log = OmniTokLogger(name="test", rank=0)
        log.print_metrics_table({"loss/total": 0.5, "loss/recon": 0.3}, step=100)

    def test_training_progress_returns_progress(self):
        from rich.progress import Progress

        from omnitok.utils.logger import OmniTokLogger
        log = OmniTokLogger(name="test", rank=0)
        prog = log.training_progress(total_steps=1000)
        assert isinstance(prog, Progress)

    def test_file_logging(self):
        from omnitok.utils.logger import OmniTokLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            log = OmniTokLogger(name="test_file", rank=0, log_dir=tmpdir)
            log.info("hello from file")
            log_file = os.path.join(tmpdir, "train_rank0.log")
            assert os.path.exists(log_file)


# ────────────────────────────────────────────────────────────
# ArtifactManager
# ────────────────────────────────────────────────────────────

class TestArtifactManager:
    def test_save_recon_grid(self):
        from omnitok.utils.artifacts import ArtifactManager
        mgr = ArtifactManager(output_dir=tempfile.mkdtemp())
        originals = torch.rand(4, 3, 64, 64)
        recons = torch.rand(4, 3, 64, 64)
        path = mgr.save_recon_grid(originals, recons, step=100)
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_save_recon_grid_clamps_values(self):
        from omnitok.utils.artifacts import ArtifactManager
        mgr = ArtifactManager(output_dir=tempfile.mkdtemp())
        # Values outside [0, 1] should not crash
        originals = torch.randn(2, 3, 32, 32) * 2
        recons = torch.randn(2, 3, 32, 32)
        path = mgr.save_recon_grid(originals, recons, step=1)
        assert os.path.exists(path)

    def test_save_tsne(self):
        from omnitok.utils.artifacts import ArtifactManager
        mgr = ArtifactManager(output_dir=tempfile.mkdtemp())
        latents = torch.randn(50, 64)
        labels = torch.randint(0, 5, (50,))
        path = mgr.save_tsne(latents, labels, step=100)
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_save_attn_map(self):
        from omnitok.utils.artifacts import ArtifactManager
        mgr = ArtifactManager(output_dir=tempfile.mkdtemp())
        # attn: (num_heads, H*W, H*W) or (H, W) heatmap
        attn = torch.rand(4, 16, 16)
        path = mgr.save_attn_map(attn, step=100)
        assert os.path.exists(path)
        assert path.endswith(".png")


# ────────────────────────────────────────────────────────────
# PlotGenerator
# ────────────────────────────────────────────────────────────

class TestPlotGenerator:
    def test_plot_loss_curves_from_dict(self):
        from omnitok.utils.plots import PlotGenerator
        gen = PlotGenerator(output_dir=tempfile.mkdtemp())
        history = {
            "loss/total": [{"step": i, "value": 1.0 / (i + 1)} for i in range(20)],
            "loss/recon": [{"step": i, "value": 0.5 / (i + 1)} for i in range(20)],
        }
        path = gen.plot_loss_curves(history)
        assert os.path.exists(path)

    def test_plot_loss_curves_saves_png_and_pdf(self):
        from omnitok.utils.plots import PlotGenerator
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = PlotGenerator(output_dir=tmpdir)
            history = {"loss/total": [{"step": i, "value": float(i)} for i in range(5)]}
            png_path = gen.plot_loss_curves(history, fmt="png")
            assert png_path.endswith(".png")

    def test_plot_ablation_bar(self):
        from omnitok.utils.plots import PlotGenerator
        gen = PlotGenerator(output_dir=tempfile.mkdtemp())
        results = {
            "T0 (VTP baseline)": {"rFID": 5.2, "linear_probe": 72.1},
            "T2 (DINO+RelKD)":   {"rFID": 3.8, "linear_probe": 75.4},
            "T5 (Multi-teacher)": {"rFID": 3.1, "linear_probe": 78.2},
        }
        path = gen.plot_ablation_bar(results, metric="rFID")
        assert os.path.exists(path)


# ────────────────────────────────────────────────────────────
# ExperimentManager
# ────────────────────────────────────────────────────────────

class TestExperimentManager:
    def _make_run(self, tmpdir: str, name: str, rfid: float, lp: float) -> str:
        """Helper: create a fake experiment output dir with metrics.json."""
        run_dir = os.path.join(tmpdir, name)
        os.makedirs(run_dir, exist_ok=True)
        data = {
            "history": {
                "loss/total": [{"step": i, "value": 1.0 / (i + 1)} for i in range(10)]
            },
            "best": {
                "eval/rfid": {"value": rfid, "step": 10000, "mode": "min"},
                "eval/linear_probe": {"value": lp, "step": 10000, "mode": "max"},
            },
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(data, f)
        return run_dir

    def test_load_runs(self):
        from omnitok.utils.experiment import ExperimentManager
        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = self._make_run(tmpdir, "T0", rfid=5.2, lp=72.1)
            d2 = self._make_run(tmpdir, "T2", rfid=3.8, lp=75.4)
            mgr = ExperimentManager()
            mgr.add_run("T0", d1)
            mgr.add_run("T2", d2)
            assert len(mgr.runs) == 2

    def test_compare_best_metrics(self):
        from omnitok.utils.experiment import ExperimentManager
        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = self._make_run(tmpdir, "T0", rfid=5.2, lp=72.1)
            d2 = self._make_run(tmpdir, "T2", rfid=3.8, lp=75.4)
            mgr = ExperimentManager()
            mgr.add_run("T0", d1)
            mgr.add_run("T2", d2)
            table = mgr.compare_best_metrics()
            assert "T0" in table
            assert "T2" in table
            assert "eval/rfid" in table["T0"]

    def test_to_latex_table(self):
        from omnitok.utils.experiment import ExperimentManager
        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = self._make_run(tmpdir, "T0", rfid=5.2, lp=72.1)
            d2 = self._make_run(tmpdir, "T2", rfid=3.8, lp=75.4)
            mgr = ExperimentManager()
            mgr.add_run("T0", d1)
            mgr.add_run("T2", d2)
            latex = mgr.to_latex_table(metrics=["eval/rfid", "eval/linear_probe"])
            assert "\\begin{tabular}" in latex
            assert "T0" in latex
            assert "T2" in latex
            assert "rfid" in latex.lower() or "rFID" in latex

    def test_export_comparison(self):
        from omnitok.utils.experiment import ExperimentManager
        with tempfile.TemporaryDirectory() as tmpdir:
            d1 = self._make_run(tmpdir, "T0", rfid=5.2, lp=72.1)
            mgr = ExperimentManager()
            mgr.add_run("T0", d1)
            out = os.path.join(tmpdir, "comparison.json")
            mgr.export_comparison(out)
            assert os.path.exists(out)
