"""Cosine alignment loss — ported from REPA-E's projection alignment.

This is the core alignment loss from REPA/REPA-E: compute negative cosine
similarity between L2-normalized student and teacher features.

Reference:
    REPA-E: loss/_forward_generator_alignment (lines 411-422)
    continuous_tokenizer: losses/sit_loss.py (lines 91-110)
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ...registry import ALIGNMENT_REGISTRY
from .base import BaseAlignmentLoss


def mean_flat(x: Tensor) -> Tensor:
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


@ALIGNMENT_REGISTRY.register("cosine")
class CosineAlignmentLoss(BaseAlignmentLoss):
    """Cosine alignment loss from REPA-E.

    Computes: -mean(sum(normalize(student) * normalize(teacher), dim=-1))

    This is equivalent to 1 - cosine_similarity, but the REPA formulation
    uses negative dot product of L2-normalized vectors.
    """

    def compute(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        """Compute cosine alignment loss.

        Args:
            student_features: (B, N, D) student (tokenizer encoder) features.
            teacher_features: (B, N, D) teacher (VFM) features. Must be same N.

        Returns:
            Scalar loss (lower = better alignment).
        """
        student_norm = F.normalize(student_features, dim=-1)
        teacher_norm = F.normalize(teacher_features.detach(), dim=-1)
        loss = mean_flat(-(student_norm * teacher_norm).sum(dim=-1))
        return loss.mean()


@ALIGNMENT_REGISTRY.register("mse")
class MSEAlignmentLoss(BaseAlignmentLoss):
    """Simple MSE alignment loss between features."""

    def compute(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        return F.mse_loss(student_features, teacher_features.detach())


@ALIGNMENT_REGISTRY.register("smooth_l1")
class SmoothL1AlignmentLoss(BaseAlignmentLoss):
    """Smooth L1 alignment loss — more robust to outliers than MSE."""

    def compute(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        return F.smooth_l1_loss(student_features, teacher_features.detach())
