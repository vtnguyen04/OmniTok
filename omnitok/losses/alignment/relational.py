"""Relational Knowledge Distillation alignment loss.

Supports multiple modes:
- 'mse': Standard Relational KD (CVPR 2019) using MSE between distance matrices.
- 'relu_margin': VA-VAE (LightningDiT) style using ReLU(diff - margin).
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ...registry import ALIGNMENT_REGISTRY, PROJECTOR_REGISTRY
from .base import BaseAlignmentLoss


@ALIGNMENT_REGISTRY.register("relational_kd")
class RelationalKDLoss(BaseAlignmentLoss):
    """Flexible Relational Knowledge Distillation alignment.

    Args:
        mode: "mse" (standard RKD) or "relu_margin" (VA-VAE style).
        distance_type: "cosine" or "l2" for pairwise distance computation.
        distmat_margin: Margin for "relu_margin" mode.
        distmat_weight: Weight for the relational component.
        cos_margin: Margin for the direct cosine component.
        cos_weight: Weight for the direct cosine component.
        projector: "linear", "mlp", or None.
        student_dim: Student feature dimension (for projector).
        teacher_dim: Teacher feature dimension.
        projector_hidden_dim: Hidden dim for MLP projector.
    """

    def __init__(
        self,
        mode: str = "relu_margin",
        distance_type: str = "cosine",
        distmat_margin: float = 0.2,
        distmat_weight: float = 1.0,
        cos_margin: float = 0.0,
        cos_weight: float = 0.0,
        projector: str = None,
        student_dim: int = 64,
        teacher_dim: int = 1024,
        projector_hidden_dim: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.distance_type = distance_type
        self.distmat_margin = distmat_margin
        self.distmat_weight = distmat_weight
        self.cos_margin = cos_margin
        self.cos_weight = cos_weight

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

    def _pairwise_distance(self, x: Tensor) -> Tensor:
        if self.distance_type == "cosine":
            x_norm = F.normalize(x, dim=-1)
            # Distance matrix: 1 - cosine similarity
            return 1 - torch.bmm(x_norm, x_norm.transpose(1, 2))
        elif self.distance_type == "l2":
            diff = x.unsqueeze(2) - x.unsqueeze(1)
            return diff.pow(2).sum(-1).sqrt()
        else:
            raise ValueError(f"Unknown distance_type: {self.distance_type}")

    def compute(self, student_features: Tensor, teacher_features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.projector is not None:
            z = self.projector(student_features, mask=mask, teacher_cond=teacher_features)
        else:
            z = student_features

        aux_feature = teacher_features.detach()

        # Direct cosine component
        vf_loss_2 = torch.tensor(0.0, device=z.device)
        if self.cos_weight > 0.0:
            z_norm = F.normalize(z, dim=-1)
            aux_norm = F.normalize(aux_feature, dim=-1)
            cos_sim = (z_norm * aux_norm).sum(dim=-1)

            err = F.relu(1 - self.cos_margin - cos_sim)
            if mask is not None:
                vf_loss_2 = (err * mask).sum() / (mask.sum() + 1e-6)
            else:
                vf_loss_2 = err.mean()

        # Relational component
        vf_loss_1 = torch.tensor(0.0, device=z.device)
        if self.distmat_weight > 0.0:
            d_student = self._pairwise_distance(z)
            d_teacher = self._pairwise_distance(aux_feature)

            if self.mode == "relu_margin":
                diff = torch.abs(d_student - d_teacher)
                if mask is not None:
                    # Mask the distance matrix: (B, N, N)
                    # A patch (i, j) is valid only if both i and j are kept
                    mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
                    err = F.relu(diff - self.distmat_margin)
                    vf_loss_1 = (err * mask_2d).sum() / (mask_2d.sum() + 1e-6)
                else:
                    vf_loss_1 = F.relu(diff - self.distmat_margin).mean()
            elif self.mode == "mse":
                if mask is not None:
                     mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
                     err = F.mse_loss(d_student, d_teacher, reduction="none")
                     vf_loss_1 = (err * mask_2d).sum() / (mask_2d.sum() + 1e-6)
                else:
                    # Standard RKD: normalize distance matrices first
                    d_student = (d_student - d_student.mean()) / (d_student.std() + 1e-6)
                    d_teacher = (d_teacher - d_teacher.mean()) / (d_teacher.std() + 1e-6)
                    vf_loss_1 = F.mse_loss(d_student, d_teacher)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        vf_loss = vf_loss_1 * self.distmat_weight + vf_loss_2 * self.cos_weight
        return vf_loss
