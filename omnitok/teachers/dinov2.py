"""DINOv2 teacher — wraps DINOv2 ViT models from torch.hub.

Provides patch-level features (no CLS token) for alignment.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..registry import TEACHER_REGISTRY
from .base import BaseTeacher

logger = logging.getLogger(__name__)

# Map of model names to their dimensions and hub identifiers
_DINOV2_CONFIGS = {
    "dinov2_vits14": {"embed_dim": 384, "patch_size": 14, "hub": "dinov2_vits14"},
    "dinov2_vitb14": {"embed_dim": 768, "patch_size": 14, "hub": "dinov2_vitb14"},
    "dinov2_vitl14": {"embed_dim": 1024, "patch_size": 14, "hub": "dinov2_vitl14"},
    "dinov2_vitg14": {"embed_dim": 1536, "patch_size": 14, "hub": "dinov2_vitg14"},
}


@TEACHER_REGISTRY.register("dinov2")
class DINOv2Teacher(BaseTeacher):
    """DINOv2 teacher for extracting patch-level features.

    Uses torch.hub to load pretrained DINOv2 models from facebookresearch.

    Args:
        model_name: One of dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14.
        device: Optional device string.
    """

    def __init__(self, model_name: str = "dinov2_vitl14", device: Optional[str] = None) -> None:
        super().__init__(model_name=model_name, device=device)
        if model_name not in _DINOV2_CONFIGS:
            raise ValueError(
                f"Unknown DINOv2 model: {model_name}. "
                f"Available: {list(_DINOV2_CONFIGS.keys())}"
            )
        self._config = _DINOV2_CONFIGS[model_name]

    def _build_model(self) -> nn.Module:
        """Load DINOv2 from torch.hub."""
        logger.info(f"Loading DINOv2 teacher: {self.model_name}")
        model = torch.hub.load("facebookresearch/dinov2", self._config["hub"])
        logger.info(f"DINOv2 loaded: embed_dim={self._config['embed_dim']}")
        return model

    def _extract_features(self, x: Tensor) -> Tensor:
        """Extract patch features from DINOv2 (no CLS token).

        Args:
            x: Input images (B, 3, H, W). H, W must be divisible by patch_size.

        Returns:
            Patch features (B, N, D) where N = (H/14) * (W/14).
        """
        features = self._model.forward_features(x)
        # DINOv2 returns dict with 'x_norm_patchtokens'
        if isinstance(features, dict):
            return features["x_norm_patchtokens"]
        # Fallback: assume (B, 1+N, D), strip CLS
        return features[:, 1:, :]

    @property
    def feature_dim(self) -> int:
        return self._config["embed_dim"]

    @property
    def patch_size(self) -> int:
        return self._config["patch_size"]
