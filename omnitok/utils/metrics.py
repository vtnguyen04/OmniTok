"""MetricsTracker — Accumulate, reduce, and export training metrics.

Features:
- Per-step metric accumulation with smoothing window
- Multi-GPU reduce (all_reduce)
- Automatic best-metric tracking (min/max)
- JSON + CSV export for post-hoc analysis
"""

import csv
import json
import logging
import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Training metrics accumulator with smoothing and export.

    Args:
        window_size: Smoothing window for running averages.
    """

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._history: Dict[str, List[dict]] = defaultdict(list)
        self._best: Dict[str, dict] = {}

    def update(self, name: str, value: float, step: int) -> None:
        """Record a metric value.

        Args:
            name: Metric name (e.g., "loss/total").
            value: Metric value.
            step: Current training step.
        """
        self._windows[name].append(value)
        self._history[name].append({"step": step, "value": value})

    def update_dict(self, metrics: Dict[str, float], step: int) -> None:
        """Record multiple metrics at once.

        Args:
            metrics: Dict of metric name → value.
            step: Current training step.
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.update(name, value, step)

    def get_smooth(self, name: str) -> float:
        """Get smoothed metric (moving average over window).

        Args:
            name: Metric name.

        Returns:
            Smoothed value, or 0 if no data.
        """
        window = self._windows.get(name)
        if not window:
            return 0.0
        return sum(window) / len(window)

    def get_latest(self, name: str) -> float:
        """Get latest metric value.

        Args:
            name: Metric name.

        Returns:
            Latest value, or 0 if no data.
        """
        window = self._windows.get(name)
        return window[-1] if window else 0.0

    def get_all_smooth(self) -> Dict[str, float]:
        """Get smoothed values for all tracked metrics."""
        return {name: self.get_smooth(name) for name in self._windows}

    def update_best(self, name: str, value: float, step: int, mode: str = "min") -> bool:
        """Track best metric value.

        Args:
            name: Metric name.
            value: Current value.
            step: Current step.
            mode: "min" (lower is better) or "max" (higher is better).

        Returns:
            True if this is a new best.
        """
        if name not in self._best:
            self._best[name] = {"value": value, "step": step, "mode": mode}
            return True

        prev = self._best[name]["value"]
        is_better = (mode == "min" and value < prev) or (mode == "max" and value > prev)
        if is_better:
            self._best[name] = {"value": value, "step": step, "mode": mode}
            return True
        return False

    def get_best(self, name: str) -> Optional[dict]:
        """Get best metric value and step.

        Returns:
            Dict with 'value', 'step', 'mode' or None.
        """
        return self._best.get(name)

    def export_json(self, path: str) -> None:
        """Export full history as JSON.

        Args:
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "history": dict(self._history),
            "best": self._best,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported metrics to {path}")

    def export_csv(self, path: str) -> None:
        """Export history as CSV (step, metric1, metric2, ...).

        Args:
            path: Output file path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Collect all steps and metrics
        all_steps = sorted(set(
            entry["step"]
            for entries in self._history.values()
            for entry in entries
        ))
        metric_names = sorted(self._history.keys())

        # Build step → {metric: value} lookup
        step_data: Dict[int, Dict[str, float]] = defaultdict(dict)
        for name, entries in self._history.items():
            for entry in entries:
                step_data[entry["step"]][name] = entry["value"]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + metric_names)
            for step in all_steps:
                row = [step] + [step_data[step].get(name, "") for name in metric_names]
                writer.writerow(row)

        logger.info(f"Exported CSV to {path}")

    @staticmethod
    def load_json(path: str) -> dict:
        """Load metrics from exported JSON.

        Args:
            path: Path to metrics.json.

        Returns:
            Dict with 'history' and 'best'.
        """
        with open(path) as f:
            return json.load(f)

    @property
    def tracked_metrics(self) -> List[str]:
        """List of all tracked metric names."""
        return sorted(self._windows.keys())
