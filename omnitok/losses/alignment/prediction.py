"""Prediction-based alignment loss — ported from MAETok (continuous_tokenizer).

Instead of direct feature matching, trains a small predictor head to
predict teacher features from student features. The loss is on the
predictor's output.

Reference: continuous_tokenizer/modelling/tokenizer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...registry import ALIGNMENT_REGISTRY
from .base import BaseAlignmentLoss


@ALIGNMENT_REGISTRY.register("prediction")
class PredictionAlignmentLoss(BaseAlignmentLoss):
    """Prediction-based alignment from MAETok.

    Uses a lightweight MLP predictor to project student features toward
    teacher features. The alignment signal is softer than direct matching.

    Args:
        student_dim: Student feature dimension.
        teacher_dim: Teacher feature dimension.
        hidden_dim: Hidden dimension of the predictor MLP.
    """

    def __init__(
        self,
        student_dim: int = 256,
        teacher_dim: int = 1024,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, teacher_dim),
        )

    def compute(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        """Compute prediction alignment loss.

        Args:
            student_features: (B, N, D_student).
            teacher_features: (B, N, D_teacher).

        Returns:
            Scalar loss (cosine distance between predicted and teacher).
        """
        predicted = self.predictor(student_features)
        pred_norm = F.normalize(predicted, dim=-1)
        teacher_norm = F.normalize(teacher_features.detach(), dim=-1)
        loss = -(pred_norm * teacher_norm).sum(dim=-1).mean()
        return loss
