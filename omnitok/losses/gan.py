"""GAN loss with hinge discriminator — ported from REPA-E.

Wraps NLayerDiscriminator with hinge loss for generator/discriminator training.
Includes optional LeCam regularization for training stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..registry import LOSS_REGISTRY
from .discriminator import NLayerDiscriminator, weights_init


def hinge_d_loss(logits_real: Tensor, logits_fake: Tensor) -> Tensor:
    """Hinge loss for discriminator. Ported from REPA-E."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


@LOSS_REGISTRY.register("gan")
class GANLoss(nn.Module):
    """GAN loss with NLayerDiscriminator and hinge loss.

    Ported from REPA-E's ReconstructionLoss_Stage2.

    Args:
        n_layers: Number of discriminator layers.
        disc_start: Global step to start discriminator training.
        disc_weight: Weight for discriminator loss.
        lecam_weight: LeCam regularization weight (0 = disabled).
    """

    def __init__(
        self,
        n_layers: int = 3,
        disc_start: int = 0,
        disc_weight: float = 0.5,
        lecam_weight: float = 0.0,
        lecam_ema_decay: float = 0.999,
    ) -> None:
        super().__init__()
        self.discriminator = NLayerDiscriminator(input_nc=3, n_layers=n_layers, use_actnorm=False).apply(weights_init)
        self.disc_start = disc_start
        self.disc_weight = disc_weight
        self.lecam_weight = lecam_weight
        self.lecam_ema_decay = lecam_ema_decay

        if lecam_weight > 0:
            self.register_buffer("ema_real_mean", torch.zeros(1))
            self.register_buffer("ema_fake_mean", torch.zeros(1))

    def should_train(self, global_step: int) -> bool:
        """Whether discriminator should be active at this step."""
        return global_step >= self.disc_start

    def generator_loss(self, reconstructions: Tensor, global_step: int) -> dict[str, Tensor]:
        """Generator loss: fool the discriminator.

        Args:
            reconstructions: Fake images from decoder (B, 3, H, W).
            global_step: Current training step.

        Returns:
            Dict with 'total' and 'gan' losses.
        """
        if not self.should_train(global_step):
            zero = torch.zeros(1, device=reconstructions.device)
            return {"total": zero, "gan": zero}

        for p in self.discriminator.parameters():
            p.requires_grad = False

        logits_fake = self.discriminator(reconstructions)
        gan_loss = -torch.mean(logits_fake)

        return {"total": gan_loss, "gan": gan_loss.detach()}

    def discriminator_loss(self, inputs: Tensor, reconstructions: Tensor, global_step: int) -> dict[str, Tensor]:
        """Discriminator loss: distinguish real from fake.

        Args:
            inputs: Real images (B, 3, H, W).
            reconstructions: Fake images (B, 3, H, W).
            global_step: Current training step.

        Returns:
            Dict with discriminator loss components.
        """
        if not self.should_train(global_step):
            zero = torch.zeros(1, device=inputs.device)
            return {
                "total": zero,
                "d_loss": zero,
                "logits_real": zero,
                "logits_fake": zero,
                "lecam": zero
            }

        for p in self.discriminator.parameters():
            p.requires_grad = True

        logits_real = self.discriminator(inputs.detach())
        logits_fake = self.discriminator(reconstructions.detach())
        d_loss = hinge_d_loss(logits_real, logits_fake)

        # LeCam regularization
        lecam_loss = torch.zeros(1, device=inputs.device)
        if self.lecam_weight > 0:
            real_mean = torch.mean(logits_real)
            fake_mean = torch.mean(logits_fake)
            lecam_loss = (
                torch.mean(F.relu(real_mean - self.ema_fake_mean) ** 2)
                + torch.mean(F.relu(self.ema_real_mean - fake_mean) ** 2)
            ) * self.lecam_weight
            # Update EMA
            self.ema_real_mean = self.ema_real_mean * self.lecam_ema_decay + real_mean.detach() * (
                1 - self.lecam_ema_decay
            )
            self.ema_fake_mean = self.ema_fake_mean * self.lecam_ema_decay + fake_mean.detach() * (
                1 - self.lecam_ema_decay
            )

        total = d_loss + lecam_loss

        return {
            "total": total,
            "d_loss": d_loss.detach(),
            "logits_real": logits_real.detach().mean(),
            "logits_fake": logits_fake.detach().mean(),
            "lecam": lecam_loss.detach(),
        }
