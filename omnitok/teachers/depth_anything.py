"""Depth Anything teacher — wraps Depth Anything V2 models from Hugging Face transformers.

Provides patch-level features for alignment.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForDepthEstimation

from ..registry import TEACHER_REGISTRY
from .base import BaseTeacher

logger = logging.getLogger(__name__)

# Map of model names to their dimensions and HF identifiers
_DEPTH_CONFIGS = {
    "depth_anything_v2_small": {
        "embed_dim": 384,
        "patch_size": 14,
        "hf_name": "depth-anything/Depth-Anything-V2-Small-hf",
    },
    "depth_anything_v2_base": {
        "embed_dim": 768,
        "patch_size": 14,
        "hf_name": "depth-anything/Depth-Anything-V2-Base-hf",
    },
    "depth_anything_v2_large": {
        "embed_dim": 1024,
        "patch_size": 14,
        "hf_name": "depth-anything/Depth-Anything-V2-Large-hf",
    },
}


@TEACHER_REGISTRY.register("depth_anything")
class DepthAnythingTeacher(BaseTeacher):
    """Depth Anything teacher for extracting dense semantic depth features.

    Uses Hugging Face transformers to load pretrained Depth Anything V2 models.
    We extract the hidden states from the backbone before the depth prediction head.

    Args:
        model_name: One of depth_anything_v2_small, depth_anything_v2_base, depth_anything_v2_large.
        device: Optional device string.
    """

    def __init__(self, model_name: str = "depth_anything_v2_base", device: Optional[str] = None) -> None:
        super().__init__(model_name=model_name, device=device)
        if model_name not in _DEPTH_CONFIGS:
            raise ValueError(f"Unknown Depth Anything model: {model_name}. Available: {list(_DEPTH_CONFIGS.keys())}")
        self._config = _DEPTH_CONFIGS[model_name]

    def _build_model(self) -> nn.Module:
        """Load Depth Anything from transformers."""
        logger.info(f"Loading Depth Anything teacher: {self.model_name}")
        model = AutoModelForDepthEstimation.from_pretrained(self._config["hf_name"])
        logger.info(f"Depth Anything loaded: embed_dim={self._config['embed_dim']}")
        return model

    def _extract_features(self, x: Tensor) -> Tensor:
        """Extract patch features from the Depth Anything backbone.

        Args:
            x: Input images (B, 3, H, W).

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

        # Output hidden states
        outputs = self._model(x, output_hidden_states=True)
        # Get the last hidden state from the backbone (shape: B, L, D)
        features = outputs.hidden_states[-1]

        # Strip the CLS token
        return features[:, 1:, :]

    @property
    def feature_dim(self) -> int:
        return self._config["embed_dim"]

    @property
    def patch_size(self) -> int:
        return self._config["patch_size"]
