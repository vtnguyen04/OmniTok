"""CNN Decoder — ported from REPA-E / VA-VAE / LightningDiT.

Standard LDM-style CNN decoder with ResNet blocks, self-attention at
configurable resolutions, and nearest-neighbour upsampling.

This is the decoder used by REPA-E (f16d32), VA-VAE (LightningDiT),
and (simplified variant) AIOTok. It converges much faster than ViT
decoders on small datasets due to strong spatial inductive bias.

Reference:
    REPA-E: models/autoencoder.py — class Decoder (lines 279-400)
    LightningDiT: vavae/ldm/modules/diffusionmodules/model.py — class Decoder (lines 462-568)

All parameters are configurable via YAML config.
"""

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnitok.registry import DECODER_REGISTRY

logger = logging.getLogger(__name__)


def _nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish activation (SiLU)."""
    return x * torch.sigmoid(x)


def _normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with default 32 groups (LDM convention)."""
    return nn.GroupNorm(
        num_groups=min(num_groups, in_channels),
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


class _Upsample(nn.Module):
    """2× nearest-neighbour upsample + optional 3×3 conv."""

    def __init__(self, in_channels: int, with_conv: bool = True) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class _ResnetBlock(nn.Module):
    """ResNet block with optional channel change (LDM-style)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = _normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = _nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)

        return x + h


class _AttnBlock(nn.Module):
    """Vanilla self-attention block (LDM-style)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = _normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        h_ = self.proj_out(h_)

        return x + h_


@DECODER_REGISTRY.register("cnn_decoder")
class CNNDecoder(nn.Module):
    """LDM-style CNN decoder with ResNet blocks and self-attention.

    Decodes spatial latent z (B, z_channels, h, w) → image (B, 3, H, W).

    Architecture:
        conv_in → mid (ResBlock + Attn + ResBlock)
        → N upsample stages (ResBlock × (num_res_blocks+1) + optional Attn + Upsample)
        → norm → swish → conv_out

    Default config (REPA-E f16d32):
        ch=128, ch_mult=(1,1,2,2,4), z_channels=32, resolution=256
        → 4 upsample stages: 16→32→64→128→256

    Reference:
        REPA-E: models/autoencoder.py Decoder class
        LightningDiT: vavae/ldm/modules/diffusionmodules/model.py Decoder class

    Args:
        ch: Base channel count.
        out_ch: Output channels (3 for RGB).
        ch_mult: Channel multiplier per resolution level.
        num_res_blocks: Number of ResBlocks per level.
        attn_resolutions: Resolutions at which to apply self-attention.
        dropout: Dropout rate in ResBlocks.
        z_channels: Input latent channels.
        resolution: Target output resolution.
        give_pre_end: If True, return features before final conv (for get_last_layer).
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
        z_channels: int = 32,
        resolution: int = 256,
        give_pre_end: bool = False,
        **ignored_kwargs,
    ) -> None:
        super().__init__()
        if ignored_kwargs:
            logger.warning(f"CNNDecoder ignored kwargs: {ignored_kwargs}")

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(in_channels=block_in, dropout=dropout)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(in_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(_ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = _Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, stride=1, padding=1)

        logger.info(
            f"CNNDecoder: ch={ch}, ch_mult={ch_mult}, z_channels={z_channels}, "
            f"resolution={resolution}, attn@{attn_resolutions}"
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image.

        Args:
            z: Latent tensor (B, z_channels, h, w).

        Returns:
            Reconstructed image (B, out_ch, H, W).
        """
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, )
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, )

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = _nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self) -> nn.Parameter:
        """Return last conv weight for adaptive gradient balancing (VA-VAE convention)."""
        return self.conv_out.weight
