"""Tokenizer model — composes encoder + bottleneck + decoder into a unified module.

This is the main trainable model for Stage 1 tokenizer training.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class Tokenizer(nn.Module):
    """Visual Tokenizer: Encoder → Bottleneck → Decoder.

    Composes VTP encoder (with bottleneck) and pixel decoder into a
    unified module for end-to-end tokenizer training.

    Args:
        encoder: ViT encoder with bottleneck (DinoVisionTransformerWithBottleneck).
        decoder: Pixel decoder (DinoV3PixelDecoder).
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: Tensor) -> Tensor:
        """Encode images to spatial latents.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Spatial latent features (B, C, h, w) where h=H/patch_size.
        """
        return self.encoder.encode(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode spatial latents to images.

        Args:
            z: Spatial latent features (B, C, h, w).

        Returns:
            Reconstructed images (B, 3, H, W) in [0, 1].
        """
        return self.decoder(z)

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Dict[str, Tensor]:
        """Full forward pass: encode → decode.

        Args:
            x: Input images (B, 3, H, W) in [0, 1].
            return_features: If True, also return encoder features dict.

        Returns:
            Dict with:
                - 'reconstruction': Reconstructed images (B, 3, H, W).
                - 'latent': Spatial latent (B, C, h, w).
                - 'features': (optional) Raw encoder features dict.
        """
        # Get encoder features (includes alignment-ready patch tokens)
        features = self.encoder.forward_features(x)

        # Get spatial latent from bottleneck encoder
        latent = self.encoder.encode(x)

        # Decode
        reconstruction = self.decoder(latent)

        result = {
            "reconstruction": reconstruction,
            "latent": latent,
        }

        if return_features:
            result["features"] = features

        return result

    @property
    def patch_size(self) -> int:
        return self.encoder.patch_size

    @property
    def latent_dim(self) -> int:
        return self.encoder.vit_feature_bottleneck
