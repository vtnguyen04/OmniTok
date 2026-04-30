"""OmniTok Rich Logger — Professional console output with ASCII banner.

Features:
- ASCII art banner on startup
- Colored, structured log output with phase markers
- Rich progress bars with GPU stats
- Auto file + console logging
- Rank-aware (only rank 0 prints)
"""

import logging
import os
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# OmniTok color theme
OMNITOK_THEME = Theme({
    "encoder": "bold cyan",
    "teacher": "bold green",
    "loss": "bold yellow",
    "gan": "bold red",
    "info": "bold blue",
    "metric": "bold magenta",
    "success": "bold green",
    "warning": "bold yellow",
})

BANNER = r"""
 ╔═══════════════════════════════════════════════════════╗
 ║    ___                  _ _____     _                 ║
 ║   / _ \ _ __ ___  _ __ (_)_   _|__ | | __             ║
 ║  | | | | '_ ` _ \| '_ \| | | |/ _ \| |/ /            ║
 ║  | |_| | | | | | | | | | | | | (_) |   <             ║
 ║   \___/|_| |_| |_|_| |_|_| |_|\___/|_|\_\            ║
 ║                                                       ║
 ║   Multi-Teacher Visual Tokenizer Framework            ║
 ╚═══════════════════════════════════════════════════════╝
"""


class OmniTokLogger:
    """Professional Rich-based logger for OmniTok.

    Args:
        name: Logger name (e.g., experiment name).
        rank: Process rank (only rank 0 prints to console).
        log_dir: Directory for log files.
        verbose: Enable debug-level logging.
    """

    def __init__(
        self,
        name: str = "omnitok",
        rank: int = 0,
        log_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.rank = rank
        self.console = Console(theme=OMNITOK_THEME)
        self.name = name

        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.logger.handlers.clear()

        if rank == 0:
            # Console handler with Rich
            rich_handler = RichHandler(
                console=self.console,
                show_path=False,
                show_time=True,
                rich_tracebacks=True,
            )
            rich_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
            self.logger.addHandler(rich_handler)

        # File handler (all ranks)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(log_dir, f"train_rank{rank}.log")
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            self.logger.addHandler(file_handler)

    def print_banner(self) -> None:
        """Print OmniTok ASCII art banner."""
        if self.rank == 0:
            self.console.print(Text(BANNER, style="bold cyan"))

    def print_config_table(self, config: Dict[str, Any], title: str = "Config Summary") -> None:
        """Print config as a Rich table.

        Args:
            config: Flat or nested config dict.
            title: Table title.
        """
        if self.rank != 0:
            return

        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Parameter", style="cyan", width=30)
        table.add_column("Value", style="white")

        def _flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, key)
                else:
                    table.add_row(key, str(v))

        if isinstance(config, dict):
            _flatten(config)
        else:
            # OmegaConf/DictConfig
            from omegaconf import OmegaConf
            _flatten(OmegaConf.to_container(config, resolve=True))

        self.console.print(table)

    def print_model_summary(self, model_info: Dict[str, Any]) -> None:
        """Print model parameter summary.

        Args:
            model_info: Dict from count_params().
        """
        if self.rank != 0:
            return

        table = Table(title="Model Summary", show_header=True, header_style="bold green")
        table.add_column("Component", style="cyan")
        table.add_column("Params", style="white", justify="right")

        for k, v in model_info.items():
            if k.endswith("_M"):
                table.add_row(k.replace("_M", ""), f"{v:.1f}M")
            elif isinstance(v, int):
                table.add_row(k, f"{v:,}")

        self.console.print(table)

    def print_metrics_table(
        self, metrics: Dict[str, float], step: int, title: str = "Metrics"
    ) -> None:
        """Print metrics as a formatted table.

        Args:
            metrics: Dict of metric name → value.
            step: Current step.
            title: Table title.
        """
        if self.rank != 0:
            return

        table = Table(title=f"{title} @ Step {step}", show_header=True, header_style="bold yellow")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white", justify="right")

        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                table.add_row(k, f"{v:.6f}")
            else:
                table.add_row(k, str(v))

        self.console.print(table)

    def training_progress(self, total_steps: int, description: str = "Training") -> Progress:
        """Create a Rich progress bar for training.

        Args:
            total_steps: Total training steps.
            description: Progress bar description.

        Returns:
            Rich Progress context manager.
        """
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            disable=(self.rank != 0),
        )

    # Convenience logging methods with phase tags
    def info(self, msg: str, phase: str = "info") -> None:
        self.logger.info(f"[{phase.upper()}] {msg}")

    def encoder(self, msg: str) -> None:
        self.logger.info(f"[ENCODER] {msg}")

    def teacher(self, msg: str) -> None:
        self.logger.info(f"[TEACHER] {msg}")

    def loss(self, msg: str) -> None:
        self.logger.info(f"[LOSS] {msg}")

    def gan(self, msg: str) -> None:
        self.logger.info(f"[GAN] {msg}")

    def metric(self, msg: str) -> None:
        self.logger.info(f"[METRIC] {msg}")

    def success(self, msg: str) -> None:
        self.logger.info(f"[SUCCESS] {msg}")

    def warning(self, msg: str) -> None:
        self.logger.warning(f"[WARNING] {msg}")

    def error(self, msg: str) -> None:
        self.logger.error(f"[ERROR] {msg}")
