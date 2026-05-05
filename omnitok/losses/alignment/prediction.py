"""Prediction-based alignment loss — ported from MAETok (continuous_tokenizer).

Instead of direct feature matching, trains a small predictor head to
predict teacher features from student features. The loss is on the
predictor's output.

Reference: continuous_tokenizer/modelling/tokenizer.py
"""

from typing import Optional

import torch.nn.functional as F
from torch import Tensor

from omnitok.losses.alignment.base import BaseAlignmentLoss
from omnitok.registry import ALIGNMENT_REGISTRY, PROJECTOR_REGISTRY


@ALIGNMENT_REGISTRY.register("prediction")
class PredictionAlignmentLoss(BaseAlignmentLoss):
    """Prediction-based alignment from MAETok.

    Uses a lightweight MLP predictor to project student features toward
    teacher features. The alignment signal is softer than direct matching.

    Args:
        student_dim: Student feature dimension.
        teacher_dim: Teacher feature dimension.
        embed_dim: Hidden dimension of the predictor MLP.
        depth: Number of layers in the predictor.
        projector_type: Name of the projector/predictor to use.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        embed_dim: int = 512,
        depth: int = 2,
        projector_type: str = "mlp3",
        distance_type: str = "mse",
        align_only_masked: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.distance_type = distance_type
        self.align_only_masked = align_only_masked
        self.projector = PROJECTOR_REGISTRY.build(
            projector_type,
            in_dim=student_dim,
            out_dim=teacher_dim,
            embed_dim=embed_dim,
            depth=depth,
            **kwargs,
        )
        if hasattr(self.projector, "forward"):
            pass
        else:
            raise ValueError("PredictionAlignmentLoss requires a projector.")

    def compute(self, student_features: Tensor, teacher_features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute prediction loss.

        Args:
            student_features: Encoded tokens (B, N_s, D_s).
            teacher_features: Target tokens (B, N_t, D_t).
            mask: Optional binary mask of kept tokens.

        Returns:
            Scalar alignment loss.
        """
        # (B, N, D_t)
        pred_features = self.projector(student_features, mask=mask)
        target_features = teacher_features.detach()

        # Match lengths if teacher has cls token and student doesn't (or vice versa)
        if pred_features.shape[1] != target_features.shape[1]:
            # This logic should be handled by the projector or via pooling
            # For now, just slice or pad
            min_len = min(pred_features.shape[1], target_features.shape[1])
            pred_features = pred_features[:, :min_len]
            target_features = target_features[:, :min_len]

        if self.distance_type == "cosine":
            # 1 - cosine similarity
            pred_features = F.normalize(pred_features, dim=-1)
            target_features = F.normalize(target_features, dim=-1)
            loss_per_token = 1 - (pred_features * target_features).sum(dim=-1)
        elif self.distance_type == "l1":
            loss_per_token = F.l1_loss(pred_features, target_features, reduction="none").mean(dim=-1)
        elif self.distance_type == "smooth_l1":
            loss_per_token = F.smooth_l1_loss(pred_features, target_features, reduction="none").mean(dim=-1)
        elif self.distance_type == "mse":
            loss_per_token = F.mse_loss(pred_features, target_features, reduction="none").mean(dim=-1)
        else:
            raise ValueError(f"Unknown prediction alignment distance type: {self.distance_type}")

        if mask is not None and self.align_only_masked:
            # mask: 1 = KEPT, 0 = MASKED (in OmniTok). We want loss on MASKED tokens.
            # However, mask includes CLS token (index 0). pred_features may not.
            if mask.shape[1] > pred_features.shape[1]:
                mask = mask[:, -pred_features.shape[1]:]

            # The tokens to align are the ones that were MASKED (mask == 0)
            align_mask = (mask == 0).float()

            # If nothing was masked (e.g. at eval time), just take the mean
            if align_mask.sum() > 0:
                loss = (loss_per_token * align_mask).sum() / align_mask.sum()
            else:
                loss = loss_per_token.mean()
        else:
            loss = loss_per_token.mean()

        return loss
