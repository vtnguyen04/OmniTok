"""Prediction-based alignment loss — ported from MAETok (continuous_tokenizer).

Instead of direct feature matching, trains a small predictor head to
predict teacher features from student features. The loss is on the
predictor's output.

Reference: continuous_tokenizer/modelling/tokenizer.py
"""

import torch.nn.functional as F
from torch import Tensor

from ...models.decoder.aux_decoder import AuxiliaryViTDecoder
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
        student_dim: int = 64,
        teacher_dim: int = 1024,
        embed_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        num_patches: int = 256,
    ) -> None:
        super().__init__()
        self.predictor = AuxiliaryViTDecoder(
            latent_dim=student_dim,
            out_dim=teacher_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_patches=num_patches,
        )

    def compute(self, student_features: Tensor, teacher_features: Tensor) -> Tensor:
        """Compute prediction alignment loss.

        Args:
            student_features: (B, N, D_student).
            teacher_features: (B, N, D_teacher).

        Returns:
            Scalar loss (cosine distance between predicted and teacher).
        """
        # Pass mask if available in kwargs
        # But prediction.py currently only gets student_features and teacher_features
        # If Tokenizer passes mask, it should be in kwargs or tuple. For now, assume None.
        predicted = self.predictor(student_features)

        # Ensure predicted has same shape as teacher
        if predicted.shape[1] != teacher_features.shape[1]:
            # Spatial interpolation or fallback
            B, N_p, D_p = predicted.shape
            B, N_t, D_t = teacher_features.shape
            if N_p > N_t:
                # E.g. 256 patches but teacher has 257 (cls token)
                # Actually teacher usually has CLS token
                pass

        pred_norm = F.normalize(predicted, dim=-1)
        teacher_norm = F.normalize(teacher_features.detach(), dim=-1)

        # Match lengths if teacher has cls token and student doesn't (or vice versa)
        min_len = min(pred_norm.shape[1], teacher_norm.shape[1])
        pred_norm = pred_norm[:, :min_len]
        teacher_norm = teacher_norm[:, :min_len]

        loss = -(pred_norm * teacher_norm).sum(dim=-1).mean()
        return loss
