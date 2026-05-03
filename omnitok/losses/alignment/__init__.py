"""Alignment loss strategies for VFM feature alignment."""

# Import to trigger ALIGNMENT_REGISTRY registration
from ...models.heads import projector  # noqa: F401
from .base import BaseAlignmentLoss
from .contrastive import ContrastiveAlignmentLoss, SigLIPContrastiveLoss
from .cosine import CosineAlignmentLoss
from .distillation import DINOLoss, PatchContrastiveLoss
from .prediction import PredictionAlignmentLoss
from .relational import RelationalKDLoss
from .vicreg import BarlowTwinsAlignmentLoss, VICRegAlignmentLoss

__all__ = [
    "BaseAlignmentLoss",
    "RelationalKDLoss",
    "CosineAlignmentLoss",
    "PredictionAlignmentLoss",
    "ContrastiveAlignmentLoss",
    "SigLIPContrastiveLoss",
    "DINOLoss",
    "PatchContrastiveLoss",
    "VICRegAlignmentLoss",
    "BarlowTwinsAlignmentLoss",
]
