"""Linear Probe evaluator for tokenizer semantic quality.

Extracts encoder features, fits a logistic regression classifier (sklearn),
and reports top-1 accuracy — a proxy for how well the latent space retains
semantic information.

Reference:
    VTP/tools/test_linear_probing_hf.py — feature extraction + linear head
    sklearn.linear_model.LogisticRegression — efficient solver, no GPU needed
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LinearProbeEvaluator:
    """Evaluate semantic quality of tokenizer encoder via linear probing.

    Extracts features from the encoder (frozen), then fits a logistic
    regression classifier on training features and evaluates on val features.

    Designed for quick evaluation during ablation — sklearn solver is fast
    and avoids the need for a full SGD training loop like VTP's DDP version.

    Args:
        max_train_samples: Max training samples to collect (memory limit).
        max_val_samples: Max validation samples to collect.
        solver: sklearn LogisticRegression solver.
        max_iter: Max iterations for logistic regression solver.
        C: Inverse regularization strength.
        feature_type: How to extract features from encoder output.
            - 'mean_pool': Mean-pool over spatial/sequence dim → (B, D).
            - 'cls': Use first token (CLS) → (B, D).
            - 'flatten': Flatten all tokens → (B, D*L) — only for small models.
    """

    def __init__(
        self,
        max_train_samples: int = 50000,
        max_val_samples: int = 10000,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        C: float = 1.0,
        feature_type: str = "mean_pool",
    ) -> None:
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.solver = solver
        self.max_iter = max_iter
        self.C = C
        assert feature_type in ("mean_pool", "cls", "flatten")
        self.feature_type = feature_type

    @torch.no_grad()
    def compute(
        self,
        train_features: Tensor,
        train_labels: Tensor,
        val_features: Tensor,
        val_labels: Tensor,
    ) -> Dict[str, float]:
        """Fit linear probe on precomputed features and return accuracy.

        Args:
            train_features: (N_train, D) float features.
            train_labels: (N_train,) int class labels.
            val_features: (N_val, D) float features.
            val_labels: (N_val,) int class labels.

        Returns:
            Dict with 'linear_probe_acc' (top-1 %) and sample counts.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X_train = train_features.cpu().float().numpy()
        y_train = train_labels.cpu().numpy()
        X_val = val_features.cpu().float().numpy()
        y_val = val_labels.cpu().numpy()

        # Normalize features — standard practice for linear probing
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        clf = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
            C=self.C,
            multi_class="multinomial" if self.solver != "liblinear" else "ovr",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        acc = 100.0 * clf.score(X_val, y_val)

        logger.info(
            f"LinearProbe: acc={acc:.2f}%  "
            f"(train={len(y_train)}, val={len(y_val)}, solver={self.solver})"
        )

        return {
            "linear_probe_acc": acc,
            "n_train": len(y_train),
            "n_val": len(y_val),
        }

    @torch.no_grad()
    def compute_from_model(
        self,
        encoder: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Extract encoder features and run linear probe.

        Args:
            encoder: Frozen encoder. Called as encoder(images) → features.
                     Features may be (B, D), (B, L, D), or (B, C, h, w).
            train_loader: Training DataLoader yielding (images, labels).
            val_loader: Validation DataLoader yielding (images, labels).
            device: Compute device.

        Returns:
            Linear probe accuracy dict.
        """
        encoder.eval()

        train_feats, train_labels = self._extract_features(
            encoder, train_loader, device, self.max_train_samples, split="train"
        )
        val_feats, val_labels = self._extract_features(
            encoder, val_loader, device, self.max_val_samples, split="val"
        )

        return self.compute(train_feats, train_labels, val_feats, val_labels)

    def _extract_features(
        self,
        encoder: nn.Module,
        loader: DataLoader,
        device: torch.device,
        max_samples: int,
        split: str,
    ) -> Tuple[Tensor, Tensor]:
        all_feats = []
        all_labels = []
        total = 0

        for images, labels in loader:
            if total >= max_samples:
                break

            images = images.to(device)
            feats = encoder(images)
            feats = self._pool_features(feats)  # → (B, D)

            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())
            total += images.shape[0]

            if total % 5000 == 0:
                logger.info(f"LinearProbe [{split}]: extracted {total} samples")

        if not all_feats:
            raise RuntimeError(f"No features extracted from {split} loader.")

        return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

    def _pool_features(self, feats: Tensor) -> Tensor:
        """Pool arbitrary encoder output to (B, D)."""
        if feats.ndim == 2:
            return feats  # already (B, D)

        if feats.ndim == 4:
            # (B, C, h, w) — spatial latents: flatten spatial → mean pool
            B, C, h, w = feats.shape
            return feats.reshape(B, C, -1).mean(dim=-1)  # (B, C)

        if feats.ndim == 3:
            # (B, L, D) — sequence of tokens
            if self.feature_type == "cls":
                return feats[:, 0, :]  # CLS token
            elif self.feature_type == "flatten":
                B, L, D = feats.shape
                return feats.reshape(B, L * D)
            else:  # mean_pool (default)
                return feats.mean(dim=1)

        raise ValueError(f"Unsupported feature shape: {feats.shape}")
