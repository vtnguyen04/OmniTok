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
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


class PlotGenerator:
    """Generates publication-quality plots from experiment data.

    Args:
        output_dir: Directory to save plots.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # Distinct, colorblind-friendly palette — one color per metric
    _METRIC_COLORS: Dict[str, str] = {
        "loss/total": "#E74C3C",  # red
        "loss/recon_pixel": "#2980B9",  # blue
        "loss/recon_perceptual": "#8E44AD",  # purple
        "loss/channel_balance": "#E67E22",  # orange
        "loss/align_dinov2": "#27AE60",  # green
        "loss/align_total": "#16A085",  # teal
        "loss/gan_g": "#C0392B",  # dark red
        "loss/disc": "#7F8C8D",  # gray
        "loss/gaussianity": "#F1C40F",  # yellow
    }

    # Logical groupings → one subplot each
    _PANEL_GROUPS: Dict[str, List[str]] = {
        "Total Loss": ["loss/total"],
        "Reconstruction": ["loss/recon_pixel", "loss/recon_perceptual", "loss/channel_balance"],
        "Alignment": ["loss/align_dinov2", "loss/align_total"],
        "GAN / Regularization": ["loss/gan_g", "loss/disc", "loss/gaussianity"],
    }

    def plot_loss_curves(
        self,
        history: Dict[str, List[dict]],
        title: str = "Training Loss",
        smooth_window: int = 50,
        fmt: str = "png",
        filename: Optional[str] = None,
    ) -> str:
        """Plot training loss curves — one clean subplot per loss group.

        Args:
            history: Dict of metric_name → [{"step": int, "value": float}, ...].
            title: Overall figure title.
            smooth_window: Rolling average window for smoothing.
            fmt: Output format ("png" or "pdf").
            filename: Override default filename.

        Returns:
            Saved file path.
        """
        import numpy as np
        import pandas as pd

        existing = {k for k, v in history.items() if v}

        # Build active panels (skip panels with no data)
        panels: Dict[str, List[str]] = {}
        for panel_name, metrics in self._PANEL_GROUPS.items():
            active = [m for m in metrics if m in existing]
            if active:
                panels[panel_name] = active

        # Catch-all: metrics not assigned to any panel
        assigned = {m for ms in self._PANEL_GROUPS.values() for m in ms}
        extras = [m for m in sorted(existing) if m not in assigned]
        if extras:
            panels["Other"] = extras

        if not panels:
            return ""

        n = len(panels)
        BG = "#F8F9FA"
        PANEL_BG = "#FFFFFF"
        GRID_COLOR = "#E0E0E0"
        TEXT_COLOR = "#2C3E50"

        fig = plt.figure(figsize=(13, 4 * n), facecolor=BG)
        fig.suptitle(title, fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.0)

        axes = fig.subplots(n, 1)
        if n == 1:
            axes = [axes]

        for ax, (panel_name, metrics) in zip(axes, panels.items()):
            ax.set_facecolor(PANEL_BG)
            ax.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
            ax.spines[:].set_color(GRID_COLOR)

            for metric in metrics:
                entries = history.get(metric, [])
                if not entries:
                    continue
                steps = np.array([e["step"] for e in entries])
                vals = np.array([e["value"] for e in entries])
                color = self._METRIC_COLORS.get(metric, "#555555")

                # Faint raw trace
                ax.plot(steps, vals, color=color, alpha=0.15, linewidth=0.9, zorder=1)

                # Smoothed
                ser = pd.Series(vals)
                sm = ser.rolling(window=smooth_window, min_periods=1).mean().values
                label = metric.replace("loss/", "")
                ax.plot(steps, sm, color=color, linewidth=2.2, label=label, zorder=2, solid_capstyle="round")

                # Annotate final value
                ax.annotate(
                    f"{sm[-1]:.4f}",
                    xy=(steps[-1], sm[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    va="center",
                    fontweight="bold",
                )

            ax.set_title(panel_name, fontsize=11, fontweight="bold", color=TEXT_COLOR, loc="left", pad=8)
            ax.set_xlabel("Step", fontsize=9, color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR, labelsize=8)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.legend(
                loc="upper right",
                fontsize=9,
                framealpha=0.9,
                edgecolor=GRID_COLOR,
                fancybox=True,
                ncol=min(len(metrics), 3),
            )
            # Zero-line reference
            ax.axhline(0, color=GRID_COLOR, linewidth=1.0, zorder=0)

        fig.tight_layout(rect=[0, 0, 1, 0.98])

        fname = filename or f"loss_curves.{fmt}"
        if not os.path.isabs(fname):
            path = os.path.join(self.output_dir, fname)
        else:
            path = fname

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        fig.savefig(path, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        logger.info(f"Saved loss curves → {path}")
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
                ha="center",
                va="bottom",
                fontsize=9,
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
            norm_vals = []
            for m in metrics:
                val = metric_dict.get(m, 0.0)
                all_vals = [results[r].get(m, 0.0) for r in results]
                max_v = max(all_vals) if max(all_vals) != 0 else 1.0
                norm_vals.append(val / max_v)
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
