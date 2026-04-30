"""PlotGenerator — Publication-quality plots for OmniTok experiments.

Produces loss curves, ablation bar charts, and metric comparison figures
suitable for inclusion in papers (PDF + PNG output).
"""

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Publication style
sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


class PlotGenerator:
    """Generates publication-quality plots from experiment data.

    Args:
        output_dir: Directory to save plots.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_loss_curves(
        self,
        history: Dict[str, List[dict]],
        title: str = "Training Loss",
        smooth_window: int = 50,
        fmt: str = "png",
        filename: Optional[str] = None,
    ) -> str:
        """Plot training loss curves from metrics history.

        Args:
            history: Dict of metric_name → [{"step": int, "value": float}, ...].
            title: Plot title.
            smooth_window: Rolling average window for smoothing.
            fmt: Output format ("png" or "pdf").
            filename: Override default filename.

        Returns:
            Saved file path.
        """
        import pandas as pd

        fig, ax = plt.subplots(figsize=(10, 5))

        for metric_name, entries in sorted(history.items()):
            if not entries:
                continue
            steps = [e["step"] for e in entries]
            values = [e["value"] for e in entries]

            # Raw (faint)
            ax.plot(steps, values, alpha=0.2, linewidth=0.8)

            # Smoothed
            if len(values) >= smooth_window:
                ser = pd.Series(values)
                smoothed = ser.rolling(window=smooth_window, min_periods=1).mean()
                ax.plot(steps, smoothed.tolist(), linewidth=1.8, label=metric_name)
            else:
                ax.plot(steps, values, linewidth=1.8, label=metric_name)

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
        fig.tight_layout()

        fname = filename or f"loss_curves.{fmt}"
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved loss curves to {path}")
        return path

    def plot_ablation_bar(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str,
        title: Optional[str] = None,
        lower_is_better: Optional[bool] = None,
        fmt: str = "png",
        filename: Optional[str] = None,
    ) -> str:
        """Bar chart comparing experiment variants on a single metric.

        Args:
            results: Dict of experiment_name → {metric_name: value, ...}.
            metric: Metric to plot (e.g., "rFID").
            title: Plot title (defaults to metric name).
            lower_is_better: If True, highlight lowest bar; if False, highest.
                             Auto-detected from "fid"/"loss" in metric name if None.
            fmt: Output format ("png" or "pdf").
            filename: Override default filename.

        Returns:
            Saved file path.
        """
        names = list(results.keys())
        values = [results[n].get(metric, float("nan")) for n in names]

        # Auto-detect direction
        if lower_is_better is None:
            lower_is_better = any(k in metric.lower() for k in ("fid", "loss", "error"))

        colors = []
        best_val = min(values) if lower_is_better else max(values)
        for v in values:
            colors.append("#2ecc71" if v == best_val else "#95a5a6")

        fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 5))
        bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)

        # Value labels
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

        ax.set_ylabel(metric)
        ax.set_title(title or f"Ablation: {metric}")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        fig.tight_layout()

        fname = filename or f"ablation_{metric.replace('/', '_')}.{fmt}"
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved ablation bar chart to {path}")
        return path

    def plot_multi_metric_radar(
        self,
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        fmt: str = "png",
        filename: Optional[str] = None,
    ) -> str:
        """Radar/spider chart comparing experiments across multiple metrics.

        Args:
            results: Dict of experiment_name → {metric: value}.
            metrics: List of metrics to include as axes.
            fmt: Output format.
            filename: Override default filename.

        Returns:
            Saved file path.
        """
        import numpy as np

        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close polygon

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

        for name, metric_dict in results.items():
            values = [metric_dict.get(m, 0.0) for m in metrics]
            # Normalize each metric to [0, 1] for radar display
            all_vals = [results[r].get(m, 0.0) for r in results for m in [m]]
            max_v = max(all_vals) if max(all_vals) != 0 else 1.0
            norm_vals = [v / max_v for v in values]
            norm_vals += norm_vals[:1]
            ax.plot(angles, norm_vals, linewidth=1.5, label=name)
            ax.fill(angles, norm_vals, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_title("Multi-metric Comparison", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
        fig.tight_layout()

        fname = filename or f"radar_comparison.{fmt}"
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved radar chart to {path}")
        return path
