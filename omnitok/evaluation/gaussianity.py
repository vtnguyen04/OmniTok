"""Gaussianity Score evaluator — from UNE paper.

Measures how Gaussian each dimension of the latent space is using the
Anderson-Darling normality test. Score = fraction of dimensions that
pass the test at significance level p > 0.05.

This is an EVALUATION METRIC, not a loss. For the Gaussianity
regularization loss (L_gauss), see omnitok.losses.gaussianity.

Reference:
    UNE paper: "The Universal Normal Embedding"
    Test: scipy.stats.anderson (Anderson-Darling normality test)
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
from scipy.stats import anderson
from torch import Tensor

logger = logging.getLogger(__name__)


class GaussianityEvaluator:
    """Compute Gaussianity Score for tokenizer latent spaces.

    For each latent dimension, runs Anderson-Darling test.
    Score = fraction of dimensions where the null hypothesis (Gaussian)
    is NOT rejected at the given significance level.

    Args:
        significance: Significance level for the test. One of 15%, 10%, 5%, 2.5%, 1%.
                      Default 5% (p > 0.05 ↔ Gaussian).
        max_samples: Max number of samples to use (test is O(N log N), cap for speed).
    """

    # Anderson-Darling critical value index → significance levels
    # From scipy.stats.anderson: [15%, 10%, 5%, 2.5%, 1%]
    _SIG_LEVELS = {15: 0, 10: 1, 5: 2, 2.5: 3, 1: 4}

    def __init__(self, significance: float = 5.0, max_samples: int = 50000) -> None:
        if significance not in self._SIG_LEVELS:
            raise ValueError(
                f"significance must be one of {list(self._SIG_LEVELS)}, got {significance}"
            )
        self.significance = significance
        self._sig_idx = self._SIG_LEVELS[significance]
        self.max_samples = max_samples

    @torch.no_grad()
    def compute(self, latents: Tensor) -> Dict[str, float]:
        """Compute Gaussianity Score from latent samples.

        Args:
            latents: Latent tensor of shape (N, D) — one vector per sample.
                     Typically: flatten spatial latents (B, C, H, W) → (B*H*W, C)
                     or use CLS token / mean-pooled features.

        Returns:
            Dict with:
                - 'gaussianity_score': Fraction of Gaussian dimensions [0, 1].
                - 'n_gaussian_dims': Number of Gaussian dimensions.
                - 'total_dims': Total latent dimensions.
                - 'n_samples': Number of samples used.
        """
        z = latents.detach().cpu().float()

        # Flatten to 2D: (N, D)
        if z.ndim != 2:
            raise ValueError(f"Expected 2D latent (N, D), got shape {z.shape}")

        n_samples, n_dims = z.shape

        # Subsample for speed
        if n_samples > self.max_samples:
            idx = torch.randperm(n_samples)[: self.max_samples]
            z = z[idx]
            n_samples = self.max_samples

        z_np = z.numpy()
        n_gaussian = 0

        for d in range(n_dims):
            col = z_np[:, d]
            # Anderson-Darling test for normality
            result = anderson(col, dist="norm")
            # critical_values[sig_idx] is the threshold at our significance level
            if result.statistic < result.critical_values[self._sig_idx]:
                n_gaussian += 1

        score = n_gaussian / n_dims if n_dims > 0 else 0.0

        logger.info(
            f"Gaussianity: {n_gaussian}/{n_dims} dims pass at {self.significance}% "
            f"(score={score:.4f}, N={n_samples})"
        )

        return {
            "gaussianity_score": score,
            "n_gaussian_dims": n_gaussian,
            "total_dims": n_dims,
            "n_samples": n_samples,
        }

    @torch.no_grad()
    def compute_from_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        n_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Extract latents from model and compute Gaussianity Score.

        Args:
            model: Tokenizer with .encode(x) → (B, C, h, w) latents.
            dataloader: Image DataLoader.
            device: Compute device.
            n_batches: Max batches to process (None = full dataset).

        Returns:
            Gaussianity metrics dict.
        """
        all_latents = []
        model.eval()

        for i, batch in enumerate(dataloader):
            if n_batches is not None and i >= n_batches:
                break

            images = batch[0].to(device)
            latent = model.encode(images)  # (B, C, h, w)

            # Flatten spatial dims → (B*h*w, C)
            B, C, h, w = latent.shape
            flat = latent.permute(0, 2, 3, 1).reshape(-1, C)
            all_latents.append(flat.cpu())

            if sum(t.shape[0] for t in all_latents) >= self.max_samples:
                break

        if not all_latents:
            raise RuntimeError("No latents collected — empty dataloader?")

        latents = torch.cat(all_latents, dim=0)
        return self.compute(latents)
