"""Cosine alignment loss — ported from REPA-E's projection alignment.

This is the core alignment loss from REPA/REPA-E: compute negative cosine
similarity between L2-normalized student and teacher features.

Reference:
    REPA-E: loss/_forward_generator_alignment (lines 411-422)
    continuous_tokenizer: losses/sit_loss.py (lines 91-110)
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ...registry import ALIGNMENT_REGISTRY, PROJECTOR_REGISTRY
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

    def __init__(
        self,
        projector: str = None,
        student_dim: int = 64,
        teacher_dim: int = 1024,
        projector_hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.projector_type = projector
        if projector and projector != "none":
            self.projector = PROJECTOR_REGISTRY.build(
                projector,
                in_dim=student_dim,
                out_dim=teacher_dim,
                hidden_dim=projector_hidden_dim if projector_hidden_dim > 0 else 0,
            )
        else:
            self.projector = None

    def compute(self, student_features: Tensor, teacher_features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute cosine alignment loss.

        Args:
            student_features: (B, N, D_s) student (tokenizer encoder) features.
            teacher_features: (B, N, D_t) teacher (VFM) features. Must be same N.
            mask: Optional binary mask (B, N).

        Returns:
            Scalar loss (lower = better alignment).
        """
        if self.projector is not None:
            student_features = self.projector(student_features)

        student_norm = F.normalize(student_features, dim=-1)
        teacher_norm = F.normalize(teacher_features.detach(), dim=-1)

        # (B, N)
        cos_sim_err = 1.0 - (student_norm * teacher_norm).sum(dim=-1)

        if mask is not None:
            # mask: (B, N)
            loss = (cos_sim_err * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = mean_flat(cos_sim_err).mean()
        return loss


@ALIGNMENT_REGISTRY.register("mse")
class MSEAlignmentLoss(BaseAlignmentLoss):
    """Simple MSE alignment loss between features."""

    def compute(self, student_features: Tensor, teacher_features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            return (F.mse_loss(student_features, teacher_features.detach(), reduction="none") * mask.unsqueeze(-1)).mean()
        return F.mse_loss(student_features, teacher_features.detach())


@ALIGNMENT_REGISTRY.register("smooth_l1")
class SmoothL1AlignmentLoss(BaseAlignmentLoss):
    """Smooth L1 alignment loss — more robust to outliers than MSE."""

    def compute(self, student_features: Tensor, teacher_features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            return (F.smooth_l1_loss(student_features, teacher_features.detach(), reduction="none") * mask.unsqueeze(-1)).mean()
        return F.smooth_l1_loss(student_features, teacher_features.detach())
