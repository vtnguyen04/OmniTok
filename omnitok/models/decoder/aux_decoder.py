"""Auxiliary Decoder for Prediction Alignment (ported from MAETok).

This module implements a lightweight ViT decoder used to predict teacher
features from the tokenizer's latent representations.

Reference: continuous_tokenizer/modelling/modules/timm_vit/timm_vit_models.py
"""

from typing import Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class AuxiliaryViTDecoder(nn.Module):
    """Auxiliary ViT Decoder for predicting teacher features.

    Takes a compressed latent sequence (e.g., 64d), maps it to decoder embed dim,
    appends mask tokens to restore the full spatial sequence length, adds
    positional embeddings, and processes them through shallow ViT blocks.

    Args:
        latent_dim: Dimension of the tokenizer's latent space (e.g., 64).
        out_dim: Target feature dimension to predict (e.g., 1024 for DINOv2).
        embed_dim: Hidden dimension of the decoder blocks.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        num_patches: Total number of spatial patches expected by the teacher.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        out_dim: int = 1024,
        embed_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_patches: int = 256,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        # Project latent to decoder dimension
        self.embed = nn.Linear(latent_dim, embed_dim)

        # Mask token for MAE-style decoding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (1D standard)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Final projection to teacher space
        self.head = nn.Linear(embed_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode latents to predicted teacher features.

        Args:
            x: Latent features (B, N, latent_dim).
            mask: Optional binary mask of kept tokens (if using token masking).
                  1 = kept, 0 = masked.

        Returns:
            Predicted features (B, N, out_dim).
        """
        B, N, _ = x.shape
        x = self.embed(x)

        # If masking was used in encoder, we need to reconstruct full sequence
        if mask is not None:
            # mask is (B, num_patches)
            assert mask.shape[1] == self.num_patches
            full_x = self.mask_token.expand(B, self.num_patches, -1).clone()

            # Scatter the kept tokens into their original positions
            # x shape is (B, N_kept, embed_dim)
            # mask shape is (B, num_patches), boolean or float 1/0
            mask_bool = mask.bool()

            # For each batch element, place the N_kept tokens into the true positions
            for b in range(B):
                full_x[b, mask_bool[b]] = x[b]

            x = full_x
        else:
            # If no masking, assume N == num_patches
            if N != self.num_patches:
                # E.g. x is spatial (B, H, W, C)
                if x.ndim == 4:
                    x = x.flatten(1, 2)
                elif N < self.num_patches:
                    # In case it's compressed, pad with mask tokens
                    # MAETok style: concatenate mask tokens to match length
                    num_masks = self.num_patches - N
                    masks = self.mask_token.expand(B, num_masks, -1)
                    x = torch.cat([x, masks], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # Predict target features
        out = self.head(x)
        return out
