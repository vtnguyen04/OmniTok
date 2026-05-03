"""Reconstruction loss combining pixel-level and perceptual losses.

Ported from REPA-E with clean interface. Supports L1/L2 + LPIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..registry import LOSS_REGISTRY
from .lpips import LPIPS


@LOSS_REGISTRY.register("reconstruction")
class ReconstructionLoss(nn.Module):
    """Pixel reconstruction loss: L1/L2 + LPIPS perceptual.

    Args:
        recon_type: "l1" or "l2" pixel loss.
        recon_weight: Weight for pixel loss.
        perceptual_weight: Weight for LPIPS perceptual loss.
    """

    def __init__(
        self,
        recon_type: str = "l1",
        recon_weight: float = 1.0,
        perceptual_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.recon_type = recon_type
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight

        if perceptual_weight > 0:
            self.lpips = LPIPS().eval()
            for p in self.lpips.parameters():
                p.requires_grad = False
        else:
            self.lpips = None

    def forward(self, inputs: Tensor, reconstructions: Tensor) -> dict[str, Tensor]:
        """Compute reconstruction loss.

        Args:
            inputs: Original images (B, 3, H, W) in [-1, 1].
            reconstructions: Reconstructed images (B, 3, H, W) in [-1, 1].

        Returns:
            Dict with 'total', 'pixel', 'perceptual', 'channel_balance' loss values.
        """
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # Pixel loss
        if self.recon_type == "l1":
            pixel_loss = F.l1_loss(inputs, reconstructions)
        elif self.recon_type == "l2":
            pixel_loss = F.mse_loss(inputs, reconstructions)
        else:
            raise ValueError(f"Unsupported recon_type: {self.recon_type}")

        # Perceptual loss — LPIPS/VGG is luminance-biased (Y≈0.6G+0.3R+0.1B) and can
        # drive the decoder to learn the G channel first, causing a green tint early in training.
        if self.lpips is not None and self.perceptual_weight > 0:
            perceptual_loss = self.lpips(inputs, reconstructions.clamp(-1, 1)).mean()
        else:
            perceptual_loss = torch.zeros(1, device=inputs.device)

        # Channel balance: penalize per-image per-channel mean drift.
        # Forces R, G, B to track the same mean as the target, preventing color bias.
        # Weight is small (0.1×recon_weight) — just enough to kill the green drift.
        recon_ch_mean = reconstructions.mean(dim=[2, 3])  # (B, 3)
        input_ch_mean = inputs.mean(dim=[2, 3])  # (B, 3)
        channel_balance = F.mse_loss(recon_ch_mean, input_ch_mean)

        total = (
            self.recon_weight * pixel_loss
            + self.perceptual_weight * perceptual_loss
            + 0.1 * self.recon_weight * channel_balance
        )

        return {
            "total": total,
            "pixel": pixel_loss.detach(),
            "perceptual": perceptual_loss.detach(),
            "channel_balance": channel_balance.detach(),
        }
