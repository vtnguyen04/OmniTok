"""ExperimentManager — Cross-run comparison and LaTeX table generation.

Loads metrics.json from multiple experiment output directories,
compares best metrics, and generates publication-ready LaTeX tables.
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Compare multiple experiment runs and generate result tables.

    Usage:
        mgr = ExperimentManager()
        mgr.add_run("T0", "/outputs/T0")
        mgr.add_run("T2", "/outputs/T2")
        table = mgr.compare_best_metrics()
        print(mgr.to_latex_table(metrics=["eval/rfid", "eval/linear_probe"]))
    """

    def __init__(self) -> None:
        # run_name → {"metrics": loaded metrics.json, "run_dir": str}
        self.runs: Dict[str, dict] = {}

    def add_run(self, name: str, run_dir: str) -> None:
        """Register an experiment run directory.

        Args:
            name: Display name (e.g., "T2-frozen-dino").
            run_dir: Path to run output dir containing metrics.json.

        Raises:
            FileNotFoundError: If metrics.json is not found in run_dir.
        """
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"metrics.json not found in {run_dir}")

        with open(metrics_path) as f:
            metrics = json.load(f)

        self.runs[name] = {"metrics": metrics, "run_dir": run_dir}
        logger.info(f"Registered run '{name}' from {run_dir}")

    def compare_best_metrics(self) -> Dict[str, Dict[str, float]]:
        """Extract best metric values for all registered runs.

        Returns:
            Dict of run_name → {metric_name: best_value}.
        """
        table: Dict[str, Dict[str, float]] = {}
        for run_name, run_data in self.runs.items():
            best = run_data["metrics"].get("best", {})
            table[run_name] = {k: v["value"] for k, v in best.items()}
        return table

    def to_latex_table(
        self,
        metrics: Optional[List[str]] = None,
        caption: str = "Ablation Study Results",
        label: str = "tab:ablation",
        bold_best: bool = True,
    ) -> str:
        """Generate a LaTeX table comparing best metrics across runs.

        Args:
            metrics: List of metric keys to include. If None, uses all found.
            caption: LaTeX table caption.
            label: LaTeX table label.
            bold_best: Bold the best value in each column.

        Returns:
            LaTeX table string.
        """
        comparison = self.compare_best_metrics()
        run_names = list(comparison.keys())

        # Determine metrics to include
        if metrics is None:
            all_keys: set = set()
            for v in comparison.values():
                all_keys.update(v.keys())
            metrics = sorted(all_keys)

        # Determine best per metric (lower_is_better by heuristic)
        best_vals: Dict[str, float] = {}
        for m in metrics:
            vals = [comparison[r].get(m, float("nan")) for r in run_names]
            valid = [v for v in vals if not _is_nan(v)]
            if not valid:
                continue
            lower = any(k in m.lower() for k in ("fid", "loss", "error"))
            best_vals[m] = min(valid) if lower else max(valid)

        # Build header
        col_labels = [_format_metric_name(m) for m in metrics]
        n_cols = len(metrics)
        col_fmt = "l" + "c" * n_cols

        lines = [
            "\\begin{table}[h]",
            "  \\centering",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            f"  \\begin{{tabular}}{{{col_fmt}}}",
            "    \\toprule",
            "    Method & " + " & ".join(col_labels) + " \\\\",
            "    \\midrule",
        ]

        for run_name in run_names:
            cells = []
            for m in metrics:
                val = comparison[run_name].get(m, None)
                if val is None or _is_nan(val):
                    cells.append("--")
                else:
                    formatted = f"{val:.2f}"
                    if bold_best and m in best_vals and abs(val - best_vals[m]) < 1e-9:
                        formatted = f"\\textbf{{{formatted}}}"
                    cells.append(formatted)
            lines.append(f"    {run_name} & " + " & ".join(cells) + " \\\\")

        lines += [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]

        return "\n".join(lines)

    def export_comparison(self, output_path: str) -> None:
        """Export cross-run comparison to JSON.

        Args:
            output_path: Path for output JSON file.
        """
        comparison = self.compare_best_metrics()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Exported comparison to {output_path}")

    def print_summary_table(self, metrics: Optional[List[str]] = None) -> None:
        """Print a Rich console summary table.

        Args:
            metrics: Metrics to display. If None, uses all.
        """
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            logger.warning("rich not installed — using plain print")
            print(self.compare_best_metrics())
            return

        comparison = self.compare_best_metrics()
        run_names = list(comparison.keys())

        if metrics is None:
            all_keys: set = set()
            for v in comparison.values():
                all_keys.update(v.keys())
            metrics = sorted(all_keys)

        console = Console()
        table = Table(title="Experiment Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Method", style="cyan")
        for m in metrics:
            table.add_column(_format_metric_name(m), justify="right")

        for run_name in run_names:
            cells = []
            for m in metrics:
                val = comparison[run_name].get(m, None)
                cells.append(f"{val:.2f}" if val is not None and not _is_nan(val) else "--")
            table.add_row(run_name, *cells)

        console.print(table)


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────


def _is_nan(v: float) -> bool:
    import math

    return math.isnan(v) or math.isinf(v)


def _format_metric_name(name: str) -> str:
    """Convert 'eval/linear_probe' → 'Lin. Probe' for table headers."""
    mapping = {
        "eval/rfid": "rFID",
        "eval/linear_probe": "Lin. Probe",
        "eval/zero_shot": "ZS Acc.",
        "eval/gaussianity": "Gauss. Score",
        "loss/total": "Total Loss",
    }
    if name in mapping:
        return mapping[name]
    parts = name.replace("/", " ").replace("_", " ").split()
    return " ".join(p.capitalize() for p in parts)
