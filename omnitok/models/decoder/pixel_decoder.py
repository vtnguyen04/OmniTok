import logging
from typing import Any, Literal, Optional

import torch
import torch.nn.init
from torch import Tensor, nn

from omnitok.registry import DECODER_REGISTRY

from ..encoder.vision_transformer import dtype_dict, ffn_layer_dict, init_weights_vit, norm_layer_dict
from ..layers import RopePositionEmbedding, SelfAttentionBlock
from ..layers._utils import named_apply

logger = logging.getLogger("dinov3_pixel_decoder")


def _icnr_init(weight: torch.Tensor, upscale_factor: int, std: float = 0.02) -> None:
    """ICNR initialization for the conv preceding PixelShuffle.

    Makes each upscale_factor² group of output channels identical, so that the
    PixelShuffle at initialization is equivalent to nearest-neighbor upsampling.
    This eliminates the checkerboard / ArUco-marker artifact that comes from
    independent random initialization of each sub-pixel channel.

    Args:
        weight: Conv2d weight (out_channels, in_channels, kH, kW).
        upscale_factor: PixelShuffle upscale factor r; out_channels must be divisible by r².
        std: Std for the base kernel initialization.
    """
    out_channels, in_channels, kH, kW = weight.shape
    n_unique = out_channels // (upscale_factor**2)
    subkernel = torch.empty(n_unique, in_channels, kH, kW, device=weight.device, dtype=weight.dtype)
    nn.init.trunc_normal_(subkernel, std=std)
    # Tile each unique kernel r² times along dim=0
    kernel = subkernel.repeat_interleave(upscale_factor**2, dim=0)
    with torch.no_grad():
        weight.copy_(kernel)


@DECODER_REGISTRY.register("pixel_decoder")
class DinoV3PixelDecoder(nn.Module):
    """DINOv3-based Pixel Decoder for image reconstruction.

    This decoder uses a transformer architecture with RoPE position embeddings
    and PixelShuffle for upsampling latent features back to pixel space.
    """

    def __init__(
        self,
        *,
        in_chans: int = 256,
        out_chans: int = 3,
        upscale_factor: int = 16,
        # ViT params
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: Optional[float] = None,
        pos_embed_rope_max_period: Optional[float] = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: Optional[float] = None,
        pos_embed_rope_jitter_coords: Optional[float] = None,
        pos_embed_rope_rescale_coords: Optional[float] = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "swiglu",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        mask_k_bias: bool = False,
        device: Optional[Any] = None,
        use_qk_norm: bool = False,
        **ignored_kwargs,
    ):
        super().__init__()
        self.num_latent_tokens = ignored_kwargs.get("num_latent_tokens", 0)
        self.num_patches_1d = ignored_kwargs.get("num_patches", 256)

        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        norm_layer_cls = norm_layer_dict[norm_layer]
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Input projection
        self.proj_in = nn.Conv2d(in_chans, embed_dim, kernel_size=1, bias=proj_bias)

        # 2. RoPE
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )

        # 3. Transformer blocks
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio_sequence[i],
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer_cls,
                    act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls,
                    init_values=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    device=device,
                    use_qk_norm=use_qk_norm,
                )
                for i in range(depth)
            ]
        )

        # 4. Final norm
        self.norm = norm_layer_cls(embed_dim)

        # self.num_latent_tokens and self.num_patches_1d already assigned at the top
        if self.num_latent_tokens > 0:
            self.mask_token = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, embed_dim))
            self.mask_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_1d, embed_dim))

        # 5. Progressive upsampling: log2(upscale_factor) stages of PixelShuffle(2).
        # Replaces single PixelShuffle(16) which requires learning 256 sub-pixel values at once.
        # Each 2× stage only needs 4 sub-pixel values — much easier to train → no Lego blocks.
        import math as _math

        self.upscale_factor = upscale_factor
        _n_up = int(round(_math.log2(upscale_factor)))
        # Channel schedule: embed_dim → 256 → 128 → 64 → 32 (for 4 stages)
        _chs = [embed_dim, 256, 128, 64, 32]
        _chs = _chs[: _n_up + 1]

        self.up_stages = nn.ModuleList()
        for _i in range(_n_up):
            _ic, _oc = _chs[_i], _chs[_i + 1]
            self.up_stages.append(
                nn.Sequential(
                    # PixelShuffle(2) needs 4× input channels
                    nn.Conv2d(_ic, _oc * 4, kernel_size=3, padding=1, bias=proj_bias),
                    nn.PixelShuffle(2),
                    nn.GELU(),
                    # Cross-boundary refinement at this resolution
                    nn.Conv2d(_oc, _oc, kernel_size=3, padding=1, bias=proj_bias),
                    nn.GELU(),
                    nn.Conv2d(_oc, _oc, kernel_size=3, padding=1, bias=proj_bias),
                    nn.GELU(),
                )
            )

        # Final projection to RGB — zero-init so output starts at tanh(0)=0 (mid-gray)
        self.proj_rgb = nn.Conv2d(_chs[-1], out_chans, kernel_size=3, padding=1, bias=proj_bias)

        # Final cross-patch blending: dilated convs, RF grows 3→7→15→31px.
        # Crosses 16px patch boundary → blends NN-block artifacts.
        # Zero-init last layer: starts as identity (no-op), learned incrementally.
        self.final_refine = nn.Sequential(
            nn.Conv2d(out_chans, 32, kernel_size=3, padding=1, dilation=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4, bias=True),
            nn.GELU(),
            nn.Conv2d(32, out_chans, kernel_size=3, padding=8, dilation=8, bias=True),
        )

        # Weight init
        self.init_weights()

    def init_weights(self):
        # Initialize transformer blocks first
        named_apply(init_weights_vit, self)

        # proj_in: standard trunc_normal
        torch.nn.init.trunc_normal_(self.proj_in.weight, std=0.02)
        if self.proj_in.bias is not None:
            nn.init.zeros_(self.proj_in.bias)

        if self.num_latent_tokens > 0:
            nn.init.zeros_(self.mask_token)
            nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)
            # MAE strictly requires 2D sincos pos embed for mask tokens
            import math
            grid_size = int(math.sqrt(self.num_patches_1d))
            from omnitok.models.layers.embeddings import get_2d_sincos_pos_embed
            sincos_emb = get_2d_sincos_pos_embed(self.embed_dim, grid_size)
            self.mask_pos_embed.data.copy_(torch.from_numpy(sincos_emb).float().unsqueeze(0))

        # Progressive upsampler: ICNR init on each PixelShuffle(2) conv (starts as 2× nearest-neighbor)
        for stage in self.up_stages:
            ps_conv = stage[0]  # Conv2d before PixelShuffle(2)
            _icnr_init(ps_conv.weight, upscale_factor=2, std=0.02)
            if ps_conv.bias is not None:
                nn.init.zeros_(ps_conv.bias)
            for mod in stage[2:]:  # refinement convs
                if isinstance(mod, nn.Conv2d):
                    nn.init.trunc_normal_(mod.weight, std=0.02)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)

        # proj_rgb: standard trunc_normal so gradients can flow to the decoder!
        # Do NOT zero-init the main branch, otherwise it blocks gradient flow at step 0.
        nn.init.trunc_normal_(self.proj_rgb.weight, std=0.02)
        if self.proj_rgb.bias is not None:
            nn.init.zeros_(self.proj_rgb.bias)

        # final_refine: zero-init last conv → starts as no-op, learned incrementally
        if hasattr(self, "final_refine"):
            nn.init.zeros_(self.final_refine[-1].weight)
            if self.final_refine[-1].bias is not None:
                nn.init.zeros_(self.final_refine[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.num_latent_tokens > 0 and x.ndim == 3:
            # 1D Latent Tokens Paradigm (MAETok)
            # x is (B, C, L)
            B, C, L = x.shape

            # 1. Project 1D latents to decoder embed_dim
            x = x.unsqueeze(-1) # (B, C, L, 1)
            x = self.proj_in(x) # (B, D, L, 1)
            x = x.squeeze(-1).transpose(1, 2) # (B, L, D)

            # 2. Add latent positional embedding
            x = x + self.latent_pos_embed

            # 3. Create mask tokens for the image patches
            num_patches = self.num_patches_1d
            mask_tokens = self.mask_token.expand(B, num_patches, -1)

            # Add spatial positional embedding to mask tokens!
            # (Crucial: without this, all mask tokens are identical zeros and RoPE on zero q/k does nothing!)
            mask_tokens = mask_tokens + self.mask_pos_embed

            # 4. Concatenate latent tokens and mask tokens (Latents first so they act as 'prefix' for RoPE)
            x = torch.cat([x, mask_tokens], dim=1) # (B, L + num_patches, D)

            import math
            H = W = int(math.sqrt(num_patches))
        else:
            # 2D Spatial Patches Paradigm (VTP/REPA)
            B, _, H, W = x.shape

            # 1. Project in: (B, C, H, W) -> (B, D, H, W)
            x = self.proj_in(x)

            # 2. Reshape for transformer: (B, D, H, W) -> (B, H*W, D)
            x = x.flatten(2).transpose(1, 2)
            num_patches = H * W

        # 3. Get RoPE (applied to the last num_patches tokens, leaving latents as prefix)
        rope_sincos = self.rope_embed(H=H, W=W)

        # 4. Transformer blocks
        for blk in self.blocks:
            x = blk(x, rope_sincos)

        # 5. Final Norm
        x = self.norm(x)

        if self.num_latent_tokens > 0 and x.shape[1] > num_patches:
            # Discard latent tokens (which are at the beginning), keep only the reconstructed patches
            x = x[:, self.num_latent_tokens:]

        # 6. Reshape back to image-like: (B, H*W, D) -> (B, D, H, W)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        # 7. Progressive upsampling: 16×16 → 32 → 64 → 128 → 256
        for up in self.up_stages:
            x = up(x)

        # 8. Final RGB projection
        x = self.proj_rgb(x)

        # 9. Cross-patch blending — dilated RF=31px blurs 16px patch boundaries
        x = x + self.final_refine(x)

        return torch.tanh(x)

    def get_last_layer(self) -> nn.Parameter:
        """Return last layer weight for adaptive gradient balancing (VA-VAE convention)."""
        return self.proj_rgb.weight


# Factory functions for different model sizes
def dinov3_pixel_decoder_small(in_chans=256, out_chans=3, upscale_factor=4, **kwargs):
    """Small DINOv3-based pixel decoder"""
    model = DinoV3PixelDecoder(
        in_chans=in_chans,
        out_chans=out_chans,
        upscale_factor=upscale_factor,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def dinov3_pixel_decoder_base(in_chans=256, out_chans=3, upscale_factor=4, **kwargs):
    """Base DINOv3-based pixel decoder"""
    model = DinoV3PixelDecoder(
        in_chans=in_chans,
        out_chans=out_chans,
        upscale_factor=upscale_factor,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def dinov3_pixel_decoder_large(in_chans=256, out_chans=3, upscale_factor=4, **kwargs):
    """Large DINOv3-based pixel decoder"""
    model = DinoV3PixelDecoder(
        in_chans=in_chans,
        out_chans=out_chans,
        upscale_factor=upscale_factor,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model
