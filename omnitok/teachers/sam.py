"""SAM teacher — wraps Segment Anything ViT models from timm.

Provides dense patch-level features for alignment.
"""

import logging
from typing import Optional

import timm
import torch.nn as nn
from torch import Tensor

from ..registry import TEACHER_REGISTRY
from .base import BaseTeacher

logger = logging.getLogger(__name__)

# Map of model names to their dimensions and timm identifiers
_SAM_CONFIGS = {
    "sam_vit_b": {"embed_dim": 256, "patch_size": 16, "timm_name": "samvit_base_patch16.sa1b"},
    "sam_vit_l": {"embed_dim": 256, "patch_size": 16, "timm_name": "samvit_large_patch16.sa1b"},
    "sam_vit_h": {"embed_dim": 256, "patch_size": 16, "timm_name": "samvit_huge_patch16.sa1b"},
}


@TEACHER_REGISTRY.register("sam")
class SAMTeacher(BaseTeacher):
    """Segment Anything (SAM) teacher for extracting dense patch features.

    Uses timm to load pretrained SAM ViT models. SAM outputs dense feature maps
    (e.g., 256-dim feature at 16x16 resolution for 256x256 input).

    Args:
        model_name: One of sam_vit_b, sam_vit_l, sam_vit_h.
        device: Optional device string.
    """

    def __init__(self, model_name: str = "sam_vit_b", device: Optional[str] = None) -> None:
        super().__init__(model_name=model_name, device=device)
        if model_name not in _SAM_CONFIGS:
            raise ValueError(
                f"Unknown SAM model: {model_name}. "
                f"Available: {list(_SAM_CONFIGS.keys())}"
            )
        self._config = _SAM_CONFIGS[model_name]

    def _build_model(self) -> nn.Module:
        """Load SAM from timm."""
        logger.info(f"Loading SAM teacher: {self.model_name}")
        # num_classes=0 gives us the raw features from the backbone neck
        model = timm.create_model(self._config["timm_name"], pretrained=True, num_classes=0)
        logger.info(f"SAM loaded: embed_dim={self._config['embed_dim']}")
        return model

    def _extract_features(self, x: Tensor) -> Tensor:
        """Extract patch features from SAM.

        Args:
            x: Input images (B, 3, H, W). 

        Returns:
            Patch features (B, N, D). For SAM, timm returns (B, C, H_out, W_out).
            We reshape it to (B, N, D).
        """
        # timm SAM forward_features returns shape (B, D, H_out, W_out)
        features = self._model.forward_features(x)

        # Reshape to (B, N, D)
        B, D, H_out, W_out = features.shape
        features = features.view(B, D, -1).transpose(1, 2)  # (B, H_out*W_out, D)

        return features

    @property
    def feature_dim(self) -> int:
        return self._config["embed_dim"]

    @property
    def patch_size(self) -> int:
        return self._config["patch_size"]
