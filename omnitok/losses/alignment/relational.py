"""Relational Knowledge Distillation alignment loss.

Ported from VA-VAE: aligns pairwise token relationships rather than
individual features. More robust to representation gap between student/teacher.

Reference: Park et al. "Relational Knowledge Distillation" (CVPR 2019)
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from ...registry import ALIGNMENT_REGISTRY
from .base import BaseAlignmentLoss


@ALIGNMENT_REGISTRY.register("relational_kd")
class RelationalKDLoss(BaseAlignmentLoss):
    """Relational Knowledge Distillation alignment.

    Instead of aligning individual features, aligns the pairwise distance
    structure between tokens. This captures "relationships" between patches
    rather than absolute feature values.

    L = MSE(D_student, D_teacher) where D = pairwise cosine distance matrix.

    Args:
        distance_type: "cosine" or "l2" for pairwise distance computation.
    """

    def __init__(self, distance_type: str = "cosine") -> None:
        super().__init__()
        self.distance_type = distance_type

    def _pairwise_distance(self, x: Tensor) -> Tensor:
        """Compute pairwise distance matrix.

        Args:
            x: Features (B, N, D).

        Returns:
            Distance matrix (B, N, N).
        """
        if self.distance_type == "cosine":
            x_norm = F.normalize(x, dim=-1)
            return 1 - torch.bmm(x_norm, x_norm.transpose(1, 2))
        elif self.distance_type == "l2":
            diff = x.unsqueeze(2) - x.unsqueeze(1)
            return diff.pow(2).sum(-1).sqrt()
        else:
            raise ValueError(f"Unknown distance_type: {self.distance_type}")

    def compute(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        """Compute relational KD loss.

        Args:
            student_features: (B, N, D_s).
            teacher_features: (B, N, D_t).

        Returns:
            Scalar loss.
        """
        d_student = self._pairwise_distance(student_features)
        d_teacher = self._pairwise_distance(teacher_features.detach())

        # Normalize distances to mean=0, std=1 per batch for scale-invariance
        d_student = (d_student - d_student.mean()) / (d_student.std() + 1e-6)
        d_teacher = (d_teacher - d_teacher.mean()) / (d_teacher.std() + 1e-6)

        return F.mse_loss(d_student, d_teacher)
