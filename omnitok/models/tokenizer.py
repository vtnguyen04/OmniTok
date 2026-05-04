"""Tokenizer model — composes encoder + bottleneck + decoder into a unified module.

This is the main trainable model for Stage 1 tokenizer training.
Ported from VTP (vtp/models/vtp.py) and continuous_tokenizer (modelling/tokenizer.py).
"""

import logging
from typing import Dict, Optional

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
        bottleneck: Optional external bottleneck module.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, bottleneck: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck

    def encode(self, x: Tensor) -> Tensor:
        """Encode images to spatial latents.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Spatial latent features (B, C, h, w) where h=H/patch_size.
        """
        # Pre-processing
        if hasattr(self.encoder, "input_mean") and hasattr(self.encoder, "input_std"):
            import torchvision.transforms.functional as TF
            x_0_1 = (x + 1.0) / 2.0
            x_norm = TF.normalize(
                x_0_1,
                mean=list(self.encoder.input_mean),
                std=list(self.encoder.input_std)
            )
        else:
            x_norm = x

        # This handles internal bottleneck (VTP style)
        if hasattr(self.encoder, "encode"):
             z = self.encoder.encode(x_norm)
        else:
             z = self.encoder(x_norm)

        # This handles external bottleneck (modular style)
        if self.bottleneck is not None:
             z, _ = self.bottleneck(z)

        return z

    def decode(self, z: Tensor) -> Tensor:
        """Decode spatial latents to images.

        Args:
            z: Spatial latent features (B, C, h, w).

        Returns:
            Reconstructed images (B, 3, H, W) in [-1, 1].
        """
        return self.decoder(z)

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Full forward pass: encode → decode (single encoder call).

        Args:
            x: Input images (B, 3, H, W) in [-1, 1].
            return_features: If True, also return encoder features dict.
            **kwargs: Extra arguments passed to encoder.

        Returns:
            Dict with:
                - 'reconstruction': Reconstructed images (B, 3, H, W).
                - 'latent': Spatial latent (B, C, h, w).
                - 'features': (optional) Raw encoder features dict.
                - 'mask': (optional) Mask tensor.
        """
        # Pre-processing
        if hasattr(self.encoder, "input_mean") and hasattr(self.encoder, "input_std"):
            import torchvision.transforms.functional as TF
            x_0_1 = (x + 1.0) / 2.0
            x_norm = TF.normalize(
                x_0_1,
                mean=list(self.encoder.input_mean),
                std=list(self.encoder.input_std)
            )
        else:
            x_norm = x

        # 1. Encode
        features = None
        if hasattr(self.encoder, "forward_features"):
            features = self.encoder.forward_features(x_norm, **kwargs)
            patch_tokens = features["x_norm_patchtokens"]
            b, n, c = patch_tokens.shape
            h = x.shape[-2] // self.encoder.patch_size
            w = x.shape[-1] // self.encoder.patch_size
            latent = patch_tokens.reshape(b, h, w, c).permute(0, 3, 1, 2)
        else:
            latent = self.encoder(x_norm)

        # 2. Bottleneck (Variational or Deterministic)
        posterior = None
        if self.bottleneck is not None:
            latent, bottleneck_info = self.bottleneck(latent)
            if "posterior" in bottleneck_info:
                posterior = bottleneck_info["posterior"]
        elif latent.shape[1] == self.latent_dim * 2:
            # Legacy/Implicit VAE detection (VTP style)
            from omnitok.models.distributions import DiagonalGaussianDistribution
            posterior = DiagonalGaussianDistribution(latent)
            latent = posterior.sample() if self.training else posterior.mode()

        # 3. Decode
        reconstruction = self.decoder(latent)

        result = {
            "reconstruction": reconstruction,
            "latent": latent,
        }

        if posterior is not None:
            result["posterior"] = posterior

        if return_features and features is not None:
            result["features"] = features

        if features is not None and "mask" in features:
            result["mask"] = features["mask"]

        return result

    @property
    def patch_size(self) -> int:
        return self.encoder.patch_size

    @property
    def latent_dim(self) -> int:
        if hasattr(self.encoder, "vit_feature_bottleneck"):
            return self.encoder.vit_feature_bottleneck
        if hasattr(self.encoder, "embed_dim"):
            return self.encoder.embed_dim
        return self.decoder.embed_dim

    def get_param_groups(self, base_lr: float, lr_multipliers: dict = None) -> list[dict]:
        """Return parameter groups with layer-wise learning rates.

        Respects the Open/Closed Principle: delegates internal structure knowledge
        to the Tokenizer rather than hardcoding in the training script.
        Supports arbitrary module matching via prefix matching.

        Args:
            base_lr: Base learning rate.
            lr_multipliers: Dict of {module_prefix: multiplier}.

        Returns:
            List of dictionaries for the optimizer.
        """
        if lr_multipliers is None:
            lr_multipliers = {}

        # Group parameters by their computed learning rate
        lr_to_params = {}

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            # Find longest matching prefix
            best_match = ""
            mult = 1.0
            for prefix, m in lr_multipliers.items():
                if name.startswith(prefix) and len(prefix) > len(best_match):
                    best_match = prefix
                    mult = m

            lr = base_lr * mult
            if lr not in lr_to_params:
                lr_to_params[lr] = []
            lr_to_params[lr].append(p)

        param_groups = [{"params": params, "lr": lr} for lr, params in lr_to_params.items()]
        return param_groups

    def freeze_backbone(self) -> None:
        """Freeze the pretrained encoder backbone, keeping bottleneck trainable."""
        for name, p in self.encoder.named_parameters():
            if "feature_bottleneck" not in name:
                p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the encoder backbone for fine-tuning."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    def get_last_shared_layer(self) -> nn.Parameter | None:
        """Return the weight of the last shared layer.

        Since alignment loss is computed on the encoder's output features,
        and reconstruction loss is computed on the decoder's output,
        the gradients meet at the encoder's final layer.
        """
        if hasattr(self.encoder, "norm") and hasattr(self.encoder.norm, "weight"):
            return self.encoder.norm.weight

        # Last resort: any encoder parameter
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        return params[-1] if params else None

    def get_decoder_last_layer(self) -> nn.Parameter | None:
        """Return the weight of the last layer of the decoder.

        Used for adaptive weighting of GAN loss vs Reconstruction loss,
        since both operate on the decoded image.
        """
        if hasattr(self.decoder, "get_last_layer"):
            return self.decoder.get_last_layer()
        if hasattr(self.decoder, "proj_rgb") and hasattr(self.decoder.proj_rgb, "weight"):
            return self.decoder.proj_rgb.weight
        if hasattr(self.decoder, "conv_out") and hasattr(self.decoder.conv_out, "weight"):
            return self.decoder.conv_out.weight

        params = [p for p in self.decoder.parameters() if p.requires_grad]
        return params[-1] if params else None




def build_tokenizer(config: dict) -> Tokenizer:
    """Build Tokenizer from configuration dict.

    Handles modular selection of Encoder, Bottleneck, and Decoder.
    Supports setting all dimensions (dim, bottle_dim, latent_dim).

    Args:
        config: Dictionary containing 'model' configuration.

    Returns:
        Instantiated Tokenizer.
    """
    from omnitok.registry import BOTTLENECK_REGISTRY, DECODER_REGISTRY, ENCODER_REGISTRY

    m_cfg = config.get("model", {})
    e_cfg = m_cfg.get("encoder", {})
    b_cfg = m_cfg.get("bottleneck", {})
    d_cfg = m_cfg.get("decoder", {})

    # 1. Build Encoder
    e_type = e_cfg.pop("type", "vit_encoder")
    encoder = ENCODER_REGISTRY.build(
        e_type,
        **e_cfg
    )

    # 2. Build Bottleneck (Optional)
    bottleneck = None
    b_latent_dim = b_cfg.get("latent_dim") or b_cfg.get("out_channels")
    if "type" in b_cfg:
        b_type = b_cfg.pop("type")
        bottleneck = BOTTLENECK_REGISTRY.build(
            b_type,
            **b_cfg
        )

    # 3. Build Decoder
    d_type = d_cfg.pop("type", "pixel_decoder")

    # Determine latent dimension connecting to decoder
    latent_dim = None
    if bottleneck is not None and hasattr(bottleneck, "latent_dim"):
        latent_dim = bottleneck.latent_dim
    elif bottleneck is not None and hasattr(bottleneck, "out_channels"):
        latent_dim = bottleneck.out_channels
    elif b_latent_dim is not None:
        latent_dim = b_latent_dim
    elif hasattr(encoder, "num_features"):
        latent_dim = encoder.num_features
    elif hasattr(encoder, "embed_dim"):
        latent_dim = encoder.embed_dim

    if latent_dim is not None:
        d_cfg["in_chans"] = latent_dim
        logger.info(f"Auto-setting decoder in_chans={latent_dim} to match upstream component")

    decoder = DECODER_REGISTRY.build(
        d_type,
        **d_cfg
    )

    return Tokenizer(encoder=encoder, decoder=decoder, bottleneck=bottleneck)
