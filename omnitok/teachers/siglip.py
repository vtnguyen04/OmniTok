"""SigLIP teacher — wraps SigLIP vision encoders from timm/HuggingFace.

Provides patch-level features for semantic alignment.
"""

import logging
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..registry import TEACHER_REGISTRY
from .base import BaseTeacher

logger = logging.getLogger(__name__)

_SIGLIP_CONFIGS = {
    "siglip_vit_b16_256": {"embed_dim": 768, "patch_size": 16, "img_size": 256, "timm": "vit_base_patch16_siglip_256"},
    "siglip_vit_b16_384": {"embed_dim": 768, "patch_size": 16, "img_size": 384, "timm": "vit_base_patch16_siglip_384"},
    "siglip_vit_l16_256": {
        "embed_dim": 1024,
        "patch_size": 16,
        "img_size": 256,
        "timm": "vit_large_patch16_siglip_256",
    },
    "siglip_vit_so400m_14_384": {
        "embed_dim": 1152,
        "patch_size": 14,
        "img_size": 384,
        "timm": "vit_so400m_patch14_siglip_384",
    },
}


@TEACHER_REGISTRY.register("siglip")
class SigLIPTeacher(BaseTeacher):
    """SigLIP teacher for extracting semantic patch-level features.

    Uses timm to load pretrained SigLIP vision encoders.

    Args:
        model_name: Key from _SIGLIP_CONFIGS.
        device: Optional device string.
    """

    def __init__(self, model_name: str = "siglip_vit_l16_256", device: Optional[str] = None) -> None:
        super().__init__(model_name=model_name, device=device)
        if model_name not in _SIGLIP_CONFIGS:
            raise ValueError(f"Unknown SigLIP model: {model_name}. Available: {list(_SIGLIP_CONFIGS.keys())}")
        self._config = _SIGLIP_CONFIGS[model_name]

    def _build_model(self) -> nn.Module:
        """Load SigLIP from timm."""
        import timm

        logger.info(f"Loading SigLIP teacher: {self._config['timm']}")
        model = timm.create_model(
            self._config["timm"],
            pretrained=True,
            num_classes=0,  # Remove classification head
        )
        logger.info(f"SigLIP loaded: embed_dim={self._config['embed_dim']}")
        return model

    def _extract_features(self, x: Tensor) -> Tensor:
        """Extract patch features from SigLIP (no CLS token).

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Patch features (B, N, D).
        """
        # timm ViT forward_features returns (B, N+1, D) with CLS
        features = self._model.forward_features(x)
        # Strip CLS token if present
        if features.shape[1] > (x.shape[-1] // self._config["patch_size"]) ** 2:
            return features[:, 1:, :]
        return features

    @property
    def feature_dim(self) -> int:
        return self._config["embed_dim"]

    @property
    def patch_size(self) -> int:
        return self._config["patch_size"]

    @property
    def input_mean(self) -> tuple[float, float, float]:
        """SigLIP expects mean 0.5."""
        return (0.5, 0.5, 0.5)

    @property
    def input_std(self) -> tuple[float, float, float]:
        """SigLIP expects std 0.5."""
        return (0.5, 0.5, 0.5)
