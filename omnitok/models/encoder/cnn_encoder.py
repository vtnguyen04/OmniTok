"""CNN Encoder — ported from REPA-E / VA-VAE / LightningDiT.

Standard LDM-style CNN encoder with ResNet blocks, self-attention at
configurable resolutions, and strided convolution downsampling.

This is the encoder used by REPA-E (f16d32) and VA-VAE.
When combined with CNNDecoder, it forms the classic SD-VAE architecture.

Reference:
    REPA-E: models/autoencoder.py — class Encoder (lines 165-276)
    LightningDiT: vavae/ldm/modules/diffusionmodules/model.py — class Encoder (lines 368-459)

Args are fully configurable via YAML config.
"""

import logging
from typing import Sequence

import torch
import torch.nn as nn

from omnitok.models.decoder.cnn_decoder import _AttnBlock, _nonlinearity, _normalize, _ResnetBlock
from omnitok.registry import ENCODER_REGISTRY

logger = logging.getLogger(__name__)


class _Downsample(nn.Module):
    """2× downsample with optional strided 3×3 conv."""

    def __init__(self, in_channels: int, with_conv: bool = True) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


@ENCODER_REGISTRY.register("cnn_encoder")
class CNNEncoder(nn.Module):
    """LDM-style CNN encoder with ResNet blocks and self-attention.

    Encodes image (B, 3, H, W) → spatial latent (B, z_channels, h, w).

    Architecture:
        conv_in → N downsample stages (ResBlock × num_res_blocks + optional Attn + Downsample)
        → mid (ResBlock + Attn + ResBlock) → norm → swish → conv_out

    Default config (REPA-E f16d32):
        ch=128, ch_mult=(1,1,2,2,4), z_channels=32, resolution=256
        → 4 downsample stages: 256→128→64→32→16

    Reference:
        REPA-E: models/autoencoder.py Encoder class

    Args:
        ch: Base channel count.
        out_ch: Unused (kept for API compat with LDM).
        ch_mult: Channel multiplier per resolution level.
        num_res_blocks: Number of ResBlocks per level.
        attn_resolutions: Resolutions at which to apply self-attention.
        dropout: Dropout rate in ResBlocks.
        in_channels: Input image channels (3 for RGB).
        resolution: Input spatial resolution.
        z_channels: Output latent channels.
        double_z: If True, output 2×z_channels for VAE mean+logvar.
    """

    def __init__(
        self,
        *,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: Sequence[int] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        dropout: float = 0.0,
        in_channels: int = 3,
        resolution: int = 256,
        z_channels: int = 32,
        double_z: bool = True,
        **ignored_kwargs,
    ) -> None:
        super().__init__()
        if ignored_kwargs:
            logger.warning(f"CNNEncoder ignored kwargs: {ignored_kwargs}")

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # For compatibility with Tokenizer interface
        self.embed_dim = z_channels
        self.patch_size = 2 ** (self.num_resolutions - 1)

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, 3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(_ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = _Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(in_channels=block_in, dropout=dropout)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(in_channels=block_in, dropout=dropout)

        # end
        self.norm_out = _normalize(block_in)
        out_z = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, out_z, 3, stride=1, padding=1)

        logger.info(
            f"CNNEncoder: ch={ch}, ch_mult={ch_mult}, z_channels={z_channels}, "
            f"resolution={resolution}, double_z={double_z}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent.

        Args:
            x: Input image (B, 3, H, W).

        Returns:
            Latent tensor (B, z_channels or 2*z_channels, h, w).
        """

        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = _nonlinearity(h)
        h = self.conv_out(h)
        return h
