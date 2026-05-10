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

# Map of model names to their dimensions and hub/timm identifiers
_DINOV2_CONFIGS = {
    # DINOv2
    "dinov2_vits14": {"embed_dim": 384, "patch_size": 14, "hub": "dinov2_vits14", "source": "hub"},
    "dinov2_vitb14": {"embed_dim": 768, "patch_size": 14, "hub": "dinov2_vitb14", "source": "hub"},
    "dinov2_vitl14": {"embed_dim": 1024, "patch_size": 14, "hub": "dinov2_vitl14", "source": "hub"},
    "dinov2_vitg14": {"embed_dim": 1536, "patch_size": 14, "hub": "dinov2_vitg14", "source": "hub"},
    # DINOv3 (from timm)
    "dinov3_vits16": {"embed_dim": 384, "patch_size": 16, "timm": "vit_small_patch16_dinov3", "source": "timm"},
    "dinov3_vitb16": {"embed_dim": 768, "patch_size": 16, "timm": "vit_base_patch16_dinov3", "source": "timm"},
    "dinov3_vitl16": {"embed_dim": 1024, "patch_size": 16, "timm": "vit_large_patch16_dinov3", "source": "timm"},
}


@TEACHER_REGISTRY.register("dinov2")
@TEACHER_REGISTRY.register("dinov3")
class DINOv2Teacher(BaseTeacher):
    """DINO teacher for extracting patch-level features.

    Uses torch.hub to load DINOv2 or timm to load DINOv3.

    Args:
        model_name: Model name from _DINOV2_CONFIGS.
        device: Optional device string.
    """

    def __init__(self, model_name: str = "dinov3_vitl16", device: Optional[str] = None) -> None:
        super().__init__(model_name=model_name, device=device)
        if model_name not in _DINOV2_CONFIGS:
            raise ValueError(f"Unknown DINO model: {model_name}. Available: {list(_DINOV2_CONFIGS.keys())}")
        self._config = _DINOV2_CONFIGS[model_name]

    def _build_model(self) -> nn.Module:
        """Load DINO from hub or timm."""
        logger.info(f"Loading DINO teacher: {self.model_name}")

        if self._config["source"] == "hub":
            model = torch.hub.load("facebookresearch/dinov2", self._config["hub"])
        else:
            import timm
            model = timm.create_model(
                self._config["timm"],
                pretrained=True,
                num_classes=0,
                dynamic_img_size=True,
            )

        logger.info(f"DINO loaded: embed_dim={self._config['embed_dim']}")
        return model

    def _extract_features(self, x: Tensor) -> Tensor:
        """Extract patch features from DINOv2 (no CLS token).

        Args:
            x: Input images (B, 3, H, W). If not divisible by patch_size, it will be interpolated.

        Returns:
            Patch features (B, N, D).
        """
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            # Scale to match the same number of patches assuming student patch_size=16
            target_H = (H // 16) * self.patch_size
            target_W = (W // 16) * self.patch_size
            x = torch.nn.functional.interpolate(
                x, size=(target_H, target_W), mode="bicubic", align_corners=False, antialias=True
            )

        features = self._model.forward_features(x)
        # DINOv2 returns dict with 'x_norm_patchtokens'
        if isinstance(features, dict):
            return features["x_norm_patchtokens"]

        # Fallback for timm models (which may have CLS and/or Register tokens)
        num_patches = (x.shape[-2] // self.patch_size) * (x.shape[-1] // self.patch_size)
        if features.shape[1] > num_patches:
            # Patch tokens are always at the end in ViT implementations
            return features[:, -num_patches:, :]
        return features

    @property
    def feature_dim(self) -> int:
        return self._config["embed_dim"]

    @property
    def patch_size(self) -> int:
        return self._config["patch_size"]
