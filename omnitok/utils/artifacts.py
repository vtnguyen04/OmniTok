"""ArtifactManager — Save visual artifacts during training.

Handles:
- Reconstruction comparison grids (original vs. reconstructed)
- Latent space t-SNE visualizations
- Attention map heatmaps
"""

import logging
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torchvision.utils import make_grid

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Generates and saves visual training artifacts.

    Args:
        output_dir: Root directory for artifact output.
        dpi: DPI for saved figures.
    """

    def __init__(self, output_dir: str, dpi: int = 150) -> None:
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(os.path.join(output_dir, "recon"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tsne"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "attn"), exist_ok=True)

    def save_recon_grid(
        self,
        originals: torch.Tensor,
        recons: torch.Tensor,
        step: int,
        nrow: int = 4,
        filename: Optional[str] = None,
    ) -> str:
        """Save a side-by-side grid of original and reconstructed images.

        Args:
            originals: Original images tensor (B, 3, H, W), range [0, 1].
            recons: Reconstructed images tensor (B, 3, H, W).
            step: Current training step (used in filename).
            nrow: Number of images per row in the grid.
            filename: Override default filename.

        Returns:
            Saved file path.
        """
        originals = originals.detach().cpu().clamp(0, 1)
        recons = recons.detach().cpu().clamp(0, 1)

        # Interleave: [orig0, recon0, orig1, recon1, ...]
        pairs = torch.stack([originals, recons], dim=1).flatten(0, 1)
        grid = make_grid(pairs, nrow=nrow * 2, padding=2, normalize=False)

        fname = filename or f"recon_step{step:07d}.png"
        path = os.path.join(self.output_dir, "recon", fname)

        fig, ax = plt.subplots(figsize=(nrow * 4, originals.shape[0] // nrow * 2 + 1))
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.axis("off")
        ax.set_title(f"Left: Original | Right: Reconstruction  (step {step})", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved recon grid to {path}")
        return path

    def save_tsne(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        n_components: int = 2,
        perplexity: float = 30.0,
        filename: Optional[str] = None,
    ) -> str:
        """Save t-SNE visualization of the latent space.

        Args:
            latents: Latent vectors (N, D).
            labels: Class labels (N,) for color-coding.
            step: Current training step.
            n_components: t-SNE output dimensions (2).
            perplexity: t-SNE perplexity.
            filename: Override default filename.

        Returns:
            Saved file path.
        """
        z = latents.detach().cpu().float().numpy()
        y = labels.detach().cpu().numpy()

        # Subsample if too large for fast t-SNE
        if len(z) > 2000:
            idx = np.random.choice(len(z), 2000, replace=False)
            z, y = z[idx], y[idx]

        tsne = TSNE(n_components=n_components, perplexity=min(perplexity, len(z) - 1), random_state=42)
        z_2d = tsne.fit_transform(z)

        fname = filename or f"tsne_step{step:07d}.png"
        path = os.path.join(self.output_dir, "tsne", fname)

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y, cmap="tab20", alpha=0.6, s=8)
        plt.colorbar(scatter, ax=ax, label="Class")
        ax.set_title(f"Latent t-SNE (step {step})")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved t-SNE to {path}")
        return path

    def save_attn_map(
        self,
        attn: torch.Tensor,
        step: int,
        filename: Optional[str] = None,
    ) -> str:
        """Save attention map heatmap.

        Args:
            attn: Attention weights. Accepts:
                  - (num_heads, H, W) — averaged over heads
                  - (H, W) — single map
            step: Current training step.
            filename: Override default filename.

        Returns:
            Saved file path.
        """
        attn = attn.detach().cpu().float()

        if attn.ndim == 3:
            # Average over heads
            attn_map = attn.mean(dim=0).numpy()
        elif attn.ndim == 2:
            attn_map = attn.numpy()
        else:
            raise ValueError(f"Expected attn shape (H,W) or (heads,H,W), got {attn.shape}")

        fname = filename or f"attn_step{step:07d}.png"
        path = os.path.join(self.output_dir, "attn", fname)

        fig, axes = plt.subplots(1, min(attn.shape[0], 4) + 1 if attn.ndim == 3 else 1,
                                  figsize=(4 * (min(attn.shape[0], 4) + 1) if attn.ndim == 3 else 4, 4))

        if attn.ndim == 3:
            axes = axes if hasattr(axes, "__len__") else [axes]
            axes[0].imshow(attn_map, cmap="viridis", interpolation="nearest")
            axes[0].set_title("Mean")
            axes[0].axis("off")
            for i, ax in enumerate(axes[1:]):
                ax.imshow(attn[i].numpy(), cmap="viridis", interpolation="nearest")
                ax.set_title(f"Head {i}")
                ax.axis("off")
        else:
            ax = axes if not hasattr(axes, "__len__") else axes[0]
            ax.imshow(attn_map, cmap="viridis", interpolation="nearest")
            ax.set_title(f"Attention map (step {step})")
            ax.axis("off")

        fig.suptitle(f"Attention Maps — Step {step}", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved attention map to {path}")
        return path
