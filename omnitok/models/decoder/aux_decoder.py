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
        in_dim: Dimension of the tokenizer's latent space (e.g., 64).
        out_dim: Target feature dimension to predict (e.g., 1024 for DINOv2).
        embed_dim: Hidden dimension of the decoder blocks.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        num_patches: Total number of spatial patches expected by the teacher.
    """

    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 1024,
        embed_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_patches: int = 256,
        is_1d_latent: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.is_1d_latent = is_1d_latent

        # Project latent to decoder dimension
        self.embed = nn.Linear(in_dim, embed_dim)

        # Mask token for MAE-style decoding
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding (2D sincos)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        import math
        grid_size = int(math.sqrt(num_patches))
        from omnitok.models.layers.embeddings import get_2d_sincos_pos_embed
        sincos_emb = get_2d_sincos_pos_embed(embed_dim, grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(sincos_emb).float().unsqueeze(0))
        self.pos_embed.requires_grad = False

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
        # pos_embed is already initialized with sincos in __init__
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

        # If we have exactly num_patches, it's 2D spatial patches paradigm
        if N == self.num_patches:
            x = x + self.pos_embed
        # If it's explicitly marked as 1D Latents Paradigm (MAETok)
        elif self.is_1d_latent:
            # We have N latent tokens. We need to create num_patches mask tokens.
            mask_tokens = self.mask_token.expand(B, self.num_patches, -1)
            mask_tokens = mask_tokens + self.pos_embed
            x = torch.cat([mask_tokens, x], dim=1) # (B, num_patches + N, embed_dim)
        # If we have < num_patches but it's a masked sequence (REPA with dropping)
        elif mask is not None and mask.sum(dim=1)[0].int() == N:
            # Spatial dropping (REPA style)
            assert mask.shape[1] == self.num_patches
            full_x = self.mask_token.expand(B, self.num_patches, -1).clone()

            # Scatter the kept tokens into their original positions
            mask_bool = mask.bool()
            for b in range(B):
                full_x[b, mask_bool[b]] = x[b]

            x = full_x
            x = x + self.pos_embed
        else:
            if x.ndim == 4:
                x = x.flatten(1, 2)
                x = x + self.pos_embed
            else:
                x = x + self.pos_embed

        # Apply blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # If we appended latents, we only want to predict teacher features for the image patches (the first num_patches)
        if x.shape[1] > self.num_patches:
            x = x[:, :self.num_patches]

        # Predict target features
        out = self.head(x)
        return out
