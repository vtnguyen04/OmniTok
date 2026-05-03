from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from omnitok.losses.alignment.base import BaseAlignmentLoss
from omnitok.registry import ALIGNMENT_REGISTRY


@ALIGNMENT_REGISTRY.register("contrastive")
class ContrastiveAlignmentLoss(BaseAlignmentLoss):
    """Bidirectional Contrastive Loss (CLIP/InfoNCE style).

    Used for Understanding Loss (Option A in proposed method).
    Aligns image features with text features (or other modalities) via cross-entropy
    on the similarity matrix.

    Args:
        temperature: Initial temperature for logit scaling.
        learnable_temp: Whether temperature can be optimized.
    """

    def __init__(self, temperature: float = 0.07, learnable_temp: bool = True) -> None:
        super().__init__()
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / temperature).log())
        else:
            self.register_buffer("logit_scale", torch.ones([]) * torch.tensor(1 / temperature).log())

    def compute(self, student_feat: Tensor, teacher_feat: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute contrastive loss.

        Args:
            student_feat: Flattened latent vector (B, D). Usually pooled or CLS token.
            teacher_feat: Target feature vector (B, D). E.g. Text embeddings.
            mask: Optional patch mask (usually ignored for global contrastive).

        Returns:
            Scalar loss.
        """
        # Ensure flat vectors (B, D)
        if student_feat.dim() > 2:
            student_feat = student_feat.flatten(1)
        if teacher_feat.dim() > 2:
            teacher_feat = teacher_feat.flatten(1)

        student_feat = F.normalize(student_feat, dim=-1)
        teacher_feat = F.normalize(teacher_feat, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100)

        # Cosine similarity matrix (B, B)
        logits_per_student = logit_scale * student_feat @ teacher_feat.T
        logits_per_teacher = logits_per_student.T

        labels = torch.arange(logits_per_student.shape[0], device=student_feat.device, dtype=torch.long)

        loss_s = F.cross_entropy(logits_per_student, labels)
        loss_t = F.cross_entropy(logits_per_teacher, labels)

        return (loss_s + loss_t) / 2

@ALIGNMENT_REGISTRY.register("siglip_contrastive")
class SigLIPContrastiveLoss(BaseAlignmentLoss):
    """Sigmoid Contrastive Loss (SigLIP style).

    Used for Understanding Loss (Option C in proposed method).
    Uses pairwise sigmoid loss instead of softmax, which is more memory efficient
    for large batch sizes.

    Args:
        temperature: Initial temperature for logit scaling.
        bias: Initial bias.
    """

    def __init__(self, temperature: float = 0.1, bias: float = -10.0) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / temperature).log())
        self.bias = nn.Parameter(torch.ones([]) * bias)

    def compute(self, student_feat: Tensor, teacher_feat: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute SigLIP contrastive loss."""
        if student_feat.dim() > 2:
            student_feat = student_feat.flatten(1)
        if teacher_feat.dim() > 2:
            teacher_feat = teacher_feat.flatten(1)

        student_feat = F.normalize(student_feat, dim=-1)
        teacher_feat = F.normalize(teacher_feat, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100)

        # Cosine similarity matrix
        logits = student_feat @ teacher_feat.T * logit_scale + self.bias

        labels = 2 * torch.eye(logits.shape[0], device=student_feat.device) - 1 # 1 for positive, -1 for negative

        # -log(sigmoid(labels * logits))
        loss = -F.logsigmoid(labels * logits).mean()

        return loss
