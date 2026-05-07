"""SigLIP teacher — wraps SigLIP vision encoders from timm/HuggingFace.

Provides patch-level features for semantic alignment.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..registry import TEACHER_REGISTRY
from .base import BaseTeacher

logger = logging.getLogger(__name__)

_SIGLIP_CONFIGS = {
    # SigLIP (timm)
    "siglip_vit_b16_256": {
        "embed_dim": 768, "patch_size": 16, "img_size": 256,
        "timm": "vit_base_patch16_siglip_256", "source": "timm",
    },
    "siglip_vit_b16_384": {
        "embed_dim": 768, "patch_size": 16, "img_size": 384,
        "timm": "vit_base_patch16_siglip_384", "source": "timm",
    },
    "siglip_vit_l16_256": {
        "embed_dim": 1024,
        "patch_size": 16,
        "img_size": 256,
        "timm": "vit_large_patch16_siglip_256",
        "source": "timm"
    },
    "siglip_vit_so400m_14_384": {
        "embed_dim": 1152,
        "patch_size": 14,
        "img_size": 384,
        "timm": "vit_so400m_patch14_siglip_384",
        "source": "timm"
    },
    # SigLIP 2 (HuggingFace)
    "siglip2_vit_b16_256": {
        "embed_dim": 768, "patch_size": 16, "img_size": 256,
        "hf": "google/siglip2-base-patch16-256", "source": "hf",
    },
    "siglip2_vit_l16_256": {
        "embed_dim": 1024, "patch_size": 16, "img_size": 256,
        "hf": "google/siglip2-large-patch16-256", "source": "hf",
    },
    "siglip2_vit_so400m_14_384": {
        "embed_dim": 1152, "patch_size": 14, "img_size": 384,
        "hf": "google/siglip2-so400m-patch14-384", "source": "hf",
    },
}


@TEACHER_REGISTRY.register("siglip")
@TEACHER_REGISTRY.register("siglip2")
class SigLIPTeacher(BaseTeacher):
    """SigLIP teacher for extracting semantic patch-level features.

    Uses timm or Hugging Face Transformers to load pretrained SigLIP encoders.

    Args:
        model_name: Key from _SIGLIP_CONFIGS.
        device: Optional device string.
    """

    def __init__(self, model_name: str = "siglip2_vit_l16_256", device: Optional[str] = None) -> None:
        super().__init__(model_name=model_name, device=device)
        if model_name not in _SIGLIP_CONFIGS:
            raise ValueError(f"Unknown SigLIP model: {model_name}. Available: {list(_SIGLIP_CONFIGS.keys())}")
        self._config = _SIGLIP_CONFIGS[model_name]

    def _build_model(self) -> nn.Module:
        """Load SigLIP from timm or HF."""
        if self._config["source"] == "hf":
            from transformers import AutoModel
            logger.info(f"Loading SigLIP2 teacher from HF: {self._config['hf']}")
            model = AutoModel.from_pretrained(self._config["hf"]).vision_model
        else:
            import timm
            logger.info(f"Loading SigLIP teacher from timm: {self._config['timm']}")
            model = timm.create_model(
                self._config["timm"],
                pretrained=True,
                num_classes=0,  # Remove classification head
                dynamic_img_size=True,  # Fixes AssertionError for 256x256 input on 384x384 models
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
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            target_H = (H // 16) * self.patch_size
            target_W = (W // 16) * self.patch_size
            x = torch.nn.functional.interpolate(
                x, size=(target_H, target_W), mode="bicubic", align_corners=False, antialias=True
            )

        if self._config.get("source") == "hf":
            outputs = self._model(x)
            features = outputs.last_hidden_state
        else:
            # timm ViT forward_features returns (B, N+1, D) with CLS
            features = self._model.forward_features(x)

        # Strip CLS or other extra tokens if present
        num_patches = (x.shape[-2] // self._config["patch_size"]) * (x.shape[-1] // self._config["patch_size"])
        if features.shape[1] > num_patches:
            return features[:, -num_patches:, :]
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
