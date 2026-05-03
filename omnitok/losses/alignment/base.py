"""Base alignment loss interface.

All alignment losses follow Strategy pattern — registered in ALIGNMENT_REGISTRY
and selected by config. They compute alignment between student (tokenizer)
features and frozen teacher features.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch.nn as nn
from torch import Tensor


class BaseAlignmentLoss(ABC, nn.Module):
    """Abstract base class for all alignment losses.

    Subclasses must implement compute() which takes student and teacher
    features and returns a scalar loss.
    """

    @abstractmethod
    def compute(self, student_features: Tensor, teacher_features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute alignment loss between student and teacher features.

        Args:
            student_features: Tokenizer encoder features (B, N, D_s).
            teacher_features: Frozen teacher features (B, N, D_t).
            mask: Optional binary mask of kept tokens (1=kept, 0=masked).

        Returns:
            Scalar alignment loss.
        """
        ...

    def forward(self, student_features: Tensor, teacher_features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass — delegates to compute()."""
        return self.compute(student_features, teacher_features, mask)
