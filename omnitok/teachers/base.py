"""Base teacher interface and teacher wrapper for frozen VFMs.

Teachers wrap pretrained Vision Foundation Models (DINOv2, SigLIP, SAM, etc.)
and provide a unified interface for extracting features.
All teachers are frozen — no gradients flow through them.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class BaseTeacher(ABC, nn.Module):
    """Abstract base class for all teacher models.

    Teachers are frozen feature extractors from pre-trained Vision Foundation
    Models. They provide normalized features for alignment losses.

    Subclasses must implement:
        - _build_model(): Load/create the backbone model.
        - _extract_features(): Extract raw features from input.
        - feature_dim: Property returning the output feature dimension.
    """

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        super().__init__()
        self.model_name = model_name
        self._device = device
        self._model: Optional[nn.Module] = None

    def setup(self) -> None:
        """Build and freeze the model. Call this after moving to device."""
        self._model = self._build_model()
        self._freeze()

    def _freeze(self) -> None:
        """Freeze all parameters — teachers never train."""
        if self._model is not None:
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Load or create the pretrained backbone model.

        Returns:
            The pretrained model module.
        """
        ...

    @abstractmethod
    def _extract_features(self, x: Tensor) -> Tensor:
        """Extract raw features from input images.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Patch-level features (B, N, D) where N=num_patches, D=feature_dim.
        """
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Output feature dimension of this teacher."""
        ...

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Patch size used by this teacher's backbone."""
        ...

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Extract features (always in no_grad mode).

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Normalized patch features (B, N, D).
        """
        if self._model is None:
            self.setup()
            self._model.to(x.device)
        return self._extract_features(x)

    def train(self, mode: bool = True) -> "BaseTeacher":
        """Override train() — teachers are always in eval mode."""
        return super().train(False)

    def extra_repr(self) -> str:
        return f"model_name={self.model_name}, feature_dim={self.feature_dim}"
