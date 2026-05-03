"""Multi-teacher wrapper with PHI-S adaptive loss balancing.

Manages multiple frozen teachers and balances their alignment losses
using the uncertainty-based PHI-S weighting from RADIO.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseTeacher
from .normalizer import FeatureNormalizer, ProjectedNormalizer

logger = logging.getLogger(__name__)


class MultiTeacher(nn.Module):
    """Manages multiple teacher models with optional feature normalization.

    Features from each teacher can be optionally projected to a common
    dimension and/or normalized before being used in alignment losses.

    Args:
        teachers: Dict mapping teacher names to BaseTeacher instances.
        common_dim: If set, project all teacher features to this dimension.
        normalize: Whether to apply running normalization to features.
    """

    def __init__(
        self,
        teachers: Dict[str, BaseTeacher],
        common_dim: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.teacher_names = list(teachers.keys())
        self.teachers = nn.ModuleDict(teachers)
        self.common_dim = common_dim

        # Build normalizers/projectors per teacher
        self.projectors = nn.ModuleDict()
        for name, teacher in teachers.items():
            if common_dim is not None and teacher.feature_dim != common_dim:
                self.projectors[name] = ProjectedNormalizer(
                    in_dim=teacher.feature_dim,
                    out_dim=common_dim,
                )
            elif normalize:
                self.projectors[name] = FeatureNormalizer(teacher.feature_dim)
            else:
                self.projectors[name] = nn.Identity()

        # PHI-S learnable log-variance for adaptive loss weighting
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in teachers})

    @property
    def feature_dim(self) -> int:
        """Get the output feature dimension of the teachers."""
        if self.common_dim is not None:
            return self.common_dim
        return next(iter(self.teachers.values())).feature_dim

    @torch.no_grad()
    def extract_all(self, x: Tensor) -> Dict[str, Tensor]:
        """Extract features from all teachers.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Dict mapping teacher names to their normalized features.
        """
        features = {}
        for name in self.teacher_names:
            raw = self.teachers[name](x)
            features[name] = self.projectors[name](raw)
        return features

    def get_loss_weights(self) -> Dict[str, Tensor]:
        """Compute adaptive loss weights using PHI-S (uncertainty weighting).

        Returns:
            Dict mapping teacher names to scalar weights.
            Weight_i = 1 / (2 * exp(log_var_i))
        """
        weights = {}
        for name in self.teacher_names:
            precision = torch.exp(-self.log_vars[name])
            weights[name] = 0.5 * precision
        return weights

    def get_regularization(self) -> Tensor:
        """PHI-S regularization term: sum of 0.5 * log_var_i.

        Returns:
            Scalar regularization loss.
        """
        return sum(0.5 * self.log_vars[name] for name in self.teacher_names)

    @property
    def num_teachers(self) -> int:
        return len(self.teacher_names)

    def train(self, mode: bool = True) -> "MultiTeacher":
        """Teachers always stay in eval, but projectors/log_vars can train."""
        super().train(mode)
        for teacher in self.teachers.values():
            teacher.eval()
        return self

    def extra_repr(self) -> str:
        return f"num_teachers={self.num_teachers}, teachers={self.teacher_names}"
