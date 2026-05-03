"""OmniTokWandBLogger — Rich WandB logging for OmniTok experiments.

Extends the basic Accelerate WandB integration with:
- Structured image logging (reconstruction grids)
- Histogram logging (latent distributions, weight norms)
- Artifact uploads (checkpoints, plots)
- Experiment config summary table
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class OmniTokWandBLogger:
    """WandB logger with image/histogram/artifact support.

    Wraps the wandb API with OmniTok-specific helpers.
    Gracefully degrades if wandb is not installed or disabled.

    Args:
        project: WandB project name.
        name: Run name (experiment ID, e.g., "T2-frozen-dino").
        config: Experiment config dict to log.
        tags: List of run tags.
        enabled: Set False to disable all logging (dry-run / test mode).
    """

    def __init__(
        self,
        project: str = "omnitok",
        name: str = "experiment",
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
        run_id: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self._run = None

        if not enabled:
            return

        try:
            import wandb

            self._wandb = wandb

            # Auto-resume if run_id is provided, otherwise let WandB generate one or use name
            _id = run_id if run_id else wandb.util.generate_id()

            self._run = wandb.init(
                project=project,
                name=name,
                id=_id,
                resume="allow",
                config=config or {},
                tags=tags or [],
                reinit=True,
            )
            logger.info(f"WandB run initialized: {self._run.url}")
        except ImportError:
            logger.warning("wandb not installed — logging disabled")
            self.enabled = False
        except Exception as e:
            logger.warning(f"WandB init failed ({e}) — logging disabled")
            self.enabled = False

    # ──────────────────────────────────────────
    # Scalar logging
    # ──────────────────────────────────────────

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log scalar metrics.

        Args:
            metrics: Dict of metric_name → value.
            step: Global training step.
        """
        if not self.enabled or self._run is None:
            return
        self._wandb.log(metrics, step=step)

    # ──────────────────────────────────────────
    # Image logging
    # ──────────────────────────────────────────

    def log_recon_images(
        self,
        originals: torch.Tensor,
        recons: torch.Tensor,
        step: int,
        n: int = 8,
        key: str = "viz/recon",
    ) -> None:
        """Log reconstruction comparison images to WandB.

        Args:
            originals: Original images (B, 3, H, W), range [0, 1].
            recons: Reconstructed images (B, 3, H, W).
            step: Current step.
            n: Max number of image pairs to log.
            key: WandB log key.
        """
        if not self.enabled or self._run is None:
            return

        import numpy as np
        from torchvision.utils import make_grid

        orig = originals[:n].detach().cpu().clamp(0, 1)
        rec = recons[:n].detach().cpu().clamp(0, 1)
        pairs = torch.stack([orig, rec], dim=1).flatten(0, 1)
        grid = make_grid(pairs, nrow=n * 2, padding=2)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        self._wandb.log({key: self._wandb.Image(grid_np, caption=f"step {step}")}, step=step)

    def log_image_file(self, path: str, key: str, step: int, caption: str = "") -> None:
        """Log a saved image file to WandB.

        Args:
            path: Path to image file.
            key: WandB log key.
            step: Current step.
            caption: Image caption.
        """
        if not self.enabled or self._run is None or not os.path.exists(path):
            return
        self._wandb.log({key: self._wandb.Image(path, caption=caption)}, step=step)

    # ──────────────────────────────────────────
    # Histogram logging
    # ──────────────────────────────────────────

    def log_histogram(
        self,
        tensor: torch.Tensor,
        key: str,
        step: int,
        num_bins: int = 64,
    ) -> None:
        """Log a tensor as a histogram.

        Args:
            tensor: Any-shape tensor.
            key: WandB log key.
            step: Current step.
            num_bins: Number of histogram bins.
        """
        if not self.enabled or self._run is None:
            return
        flat = tensor.detach().cpu().float().numpy().flatten()
        self._wandb.log({key: self._wandb.Histogram(flat, num_bins=num_bins)}, step=step)

    def log_weight_histograms(
        self,
        model: torch.nn.Module,
        step: int,
        prefix: str = "weights",
    ) -> None:
        """Log histograms of all model parameter tensors.

        Args:
            model: PyTorch model.
            step: Current step.
            prefix: Key prefix for WandB.
        """
        if not self.enabled or self._run is None:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.log_histogram(param.data, f"{prefix}/{name}", step)
                self.log_histogram(param.grad, f"grads/{name}", step)

    # ──────────────────────────────────────────
    # Artifact logging
    # ──────────────────────────────────────────

    def log_checkpoint(self, path: str, name: str, step: int) -> None:
        """Upload a checkpoint file as a WandB artifact.

        Args:
            path: Path to checkpoint file (.pt).
            name: Artifact name (e.g., "tokenizer-T2").
            step: Current step (added as metadata).
        """
        if not self.enabled or self._run is None or not os.path.exists(path):
            return
        artifact = self._wandb.Artifact(
            name=name,
            type="model",
            metadata={"step": step},
        )
        artifact.add_file(path)
        self._run.log_artifact(artifact)
        logger.info(f"Uploaded checkpoint artifact: {name}")

    def log_plot_file(self, path: str, key: str, step: int) -> None:
        """Log a saved plot PNG/PDF as a WandB artifact.

        Args:
            path: Path to plot file.
            key: WandB media key.
            step: Current step.
        """
        if not self.enabled or self._run is None or not os.path.exists(path):
            return
        if path.endswith(".png"):
            self.log_image_file(path, key, step)
        else:
            artifact = self._wandb.Artifact(name=os.path.basename(path), type="plot")
            artifact.add_file(path)
            self._run.log_artifact(artifact)

    # ──────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────

    def summary(self, metrics: Dict[str, float]) -> None:
        """Update WandB run summary (final metrics).

        Args:
            metrics: Dict of metric_name → final value.
        """
        if not self.enabled or self._run is None:
            return
        for k, v in metrics.items():
            self._wandb.run.summary[k] = v

    def finish(self) -> None:
        """Finalize the WandB run."""
        if self.enabled and self._run is not None:
            self._wandb.finish()
            logger.info("WandB run finished")

    @property
    def run_url(self) -> Optional[str]:
        """URL of the current WandB run."""
        if self._run is not None:
            return self._run.url
        return None
