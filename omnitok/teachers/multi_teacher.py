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
from .normalizer import FeatureNormalizer

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
        normalize: bool = True,
        phi_s_balancing: bool = True,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.teacher_names = list(teachers.keys())
        self.teachers = nn.ModuleDict(teachers)
        self.phi_s_balancing = phi_s_balancing
        self.loss_weights = loss_weights or {name: 1.0 for name in self.teacher_names}

        # Teachers ONLY extract features. The student encoder must have multiple
        # projectors to map into these native spaces. We only keep the normalizer
        # to ensure the teacher features have zero mean and unit variance.
        self.normalizers = nn.ModuleDict()
        for name, teacher in teachers.items():
            if normalize:
                self.normalizers[name] = FeatureNormalizer(teacher.feature_dim)
            else:
                self.normalizers[name] = nn.Identity()

        if self.phi_s_balancing:
            # PHI-S learnable log-variance for adaptive loss weighting
            self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(1)) for name in teachers})

    @property
    def feature_dim(self) -> int:
        """Get the output feature dimension of the teachers (returns fallback first dim)."""
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
            features[name] = self.normalizers[name](raw)
        return features

    @torch.no_grad()
    def extract_selected(
        self,
        x: Tensor,
        selected_indices: Tensor,
    ) -> Dict[str, Tensor]:
        """Extract features ONLY from selected teachers.

        Only forwards the unique set of teachers selected by the router
        for the current batch. Saves compute when top_k < num_teachers.

        Args:
            x: Input images (B, 3, H, W).
            selected_indices: (B, top_k) indices of selected teachers.

        Returns:
            Dict mapping selected teacher names to their normalized features.
        """
        unique_indices = selected_indices.unique().tolist()
        features = {}
        for idx in unique_indices:
            name = self.teacher_names[idx]
            raw = self.teachers[name](x)
            features[name] = self.normalizers[name](raw)
        return features

    def get_loss_weights(self) -> Dict[str, Tensor]:
        """Compute adaptive loss weights using PHI-S (uncertainty weighting) or default to config weights."""
        weights = {}
        for name in self.teacher_names:
            if self.phi_s_balancing:
                precision = torch.exp(-self.log_vars[name])
                weights[name] = 0.5 * precision
            else:
                weights[name] = torch.tensor(self.loss_weights[name], device=next(self.parameters()).device)
        return weights

    def get_regularization(self) -> Tensor:
        """PHI-S regularization term: sum of 0.5 * log_var_i, or 0.0."""
        if self.phi_s_balancing:
            return sum(0.5 * self.log_vars[name] for name in self.teacher_names)
        return torch.tensor(0.0, device=next(self.parameters()).device)

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
