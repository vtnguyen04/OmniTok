import logging
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from omnitok.registry import ENCODER_REGISTRY

from .vision_transformer import DinoVisionTransformer

logger = logging.getLogger(__name__)

# DINOv2 hub configs — matched by embed_dim for automatic selection
_DINOV2_HUB_CONFIGS: Dict[int, str] = {
    384: "dinov2_vits14",
    768: "dinov2_vitb14",
    1024: "dinov2_vitl14",
    1536: "dinov2_vitg14",
}


@ENCODER_REGISTRY.register("vit_small_bottleneck")
def build_vit_small_bottleneck(**kwargs):
    kwargs.setdefault("embed_dim", 384)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 6)
    return DinoVisionTransformerWithBottleneck(**kwargs)

@ENCODER_REGISTRY.register("vit_base_bottleneck")
def build_vit_base_bottleneck(**kwargs):
    kwargs.setdefault("embed_dim", 768)
    kwargs.setdefault("depth", 12)
    kwargs.setdefault("num_heads", 12)
    return DinoVisionTransformerWithBottleneck(**kwargs)

@ENCODER_REGISTRY.register("vit_large_bottleneck")
def build_vit_large_bottleneck(**kwargs):
    kwargs.setdefault("embed_dim", 1024)
    kwargs.setdefault("depth", 24)
    kwargs.setdefault("num_heads", 16)
    return DinoVisionTransformerWithBottleneck(**kwargs)

@ENCODER_REGISTRY.register("dinov3_bottleneck")
def build_dinov3_bottleneck(**kwargs):
    return DinoVisionTransformerWithBottleneck(**kwargs)


class DinoVisionTransformerWithBottleneck(DinoVisionTransformer):
    """DINOv3 Vision Transformer with feature bottleneck for dimensionality reduction.

    This variant adds a projection layer to reduce the feature dimension
    after the transformer blocks, which is useful for tokenization tasks where
    a lower-dimensional representation is preferred.

    Supports two bottleneck types (switchable via config):
        - 'linear': Single linear projection (VTP original).
        - 'mlp': LayerNorm → Linear(4C) → GELU → Linear(C) .
    """

    def __init__(
        self,
        *args,
        vit_feature_bottleneck: Optional[int] = None,
        bottleneck_type: str = "linear",
        double_z: bool = False,
        **kwargs,
    ):
        # Extract from kwargs if passed that way (Hydra often does this)
        if vit_feature_bottleneck is None:
            vit_feature_bottleneck = kwargs.pop("vit_feature_bottleneck", None)
        if bottleneck_type == "linear" and "bottleneck_type" in kwargs:
            bottleneck_type = kwargs.pop("bottleneck_type")

        super().__init__(*args, **kwargs)

        # Bottleneck setup
        self.original_embed_dim = self.embed_dim
        if vit_feature_bottleneck is None:
            vit_feature_bottleneck = self.embed_dim

        logger.info(f"Initializing bottleneck: embed_dim={self.original_embed_dim}, target={vit_feature_bottleneck}")

        self.vit_feature_bottleneck = vit_feature_bottleneck
        self.bottleneck_type = bottleneck_type
        self.double_z = double_z
        out_dim = vit_feature_bottleneck * 2 if double_z else vit_feature_bottleneck
        self.bottleneck_type = bottleneck_type

        if self.vit_feature_bottleneck != self.original_embed_dim or self.double_z:
            if bottleneck_type == "mlp":
                self.feature_bottleneck = nn.Sequential(
                    nn.LayerNorm(self.original_embed_dim),
                    nn.Linear(self.original_embed_dim, self.original_embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.original_embed_dim * 4, out_dim),
                )
                logger.info(f"Built MLP bottleneck: {self.original_embed_dim} -> {out_dim}")
            elif bottleneck_type == "linear":
                self.feature_bottleneck = nn.Linear(self.original_embed_dim, out_dim)
                logger.info(f"Built Linear bottleneck: {self.original_embed_dim} -> {out_dim}")
            else:
                raise ValueError(f"Unknown bottleneck_type: {bottleneck_type}")
        else:
            self.feature_bottleneck = nn.Identity()
            logger.info("Using Identity bottleneck (no dim reduction)")

        self.num_features = out_dim

        # use init weights from official DINOv3
        self.init_weights()

    def load_pretrained_dinov2(self, model_name: Optional[str] = None) -> None:
        """Transfer pretrained DINOv2 weights into this encoder.

        Copies all compatible weight tensors from the pretrained DINOv2 model.
        Interpolates `pos_embed` and `patch_embed.proj.weight` if spatial dimensions
        do not match (e.g., DINOv2 uses patch_size=14, we use 16).

        Args:
            model_name: DINOv2 hub name, e.g. "dinov2_vitb14". Auto-selected from
                        embed_dim if None.
        """
        if model_name is None:
            model_name = _DINOV2_HUB_CONFIGS.get(self.original_embed_dim)
            if model_name is None:
                raise ValueError(
                    f"No DINOv2 model for embed_dim={self.original_embed_dim}. "
                    f"Supported: {list(_DINOV2_HUB_CONFIGS.keys())}. "
                    "Pass model_name explicitly."
                )

        logger.info(f"Loading pretrained DINOv2: {model_name}")
        dino = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
        dino_state = dino.state_dict()
        our_state = self.state_dict()

        # Handle pos_embed interpolation
        if "pos_embed" in dino_state and "pos_embed" in our_state:
            pos_v = dino_state["pos_embed"]
            pos_our = our_state["pos_embed"]
            if pos_v.shape != pos_our.shape:
                num_extra_tokens = 1  # cls token
                orig_size = int((pos_v.shape[1] - num_extra_tokens) ** 0.5)
                new_size = int((pos_our.shape[1] - num_extra_tokens) ** 0.5)

                extra_tokens = pos_v[:, :num_extra_tokens]
                pos_tokens = pos_v[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
                )
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                dino_state["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)

        # Handle patch_embed.proj.weight interpolation
        if "patch_embed.proj.weight" in dino_state and "patch_embed.proj.weight" in our_state:
            patch_v = dino_state["patch_embed.proj.weight"]
            patch_our = our_state["patch_embed.proj.weight"]
            if patch_v.shape != patch_our.shape:
                patch_v_interp = torch.nn.functional.interpolate(
                    patch_v.float(), size=(patch_our.shape[2], patch_our.shape[3]), mode="bicubic", align_corners=False
                ).to(patch_v.dtype)
                dino_state["patch_embed.proj.weight"] = patch_v_interp

        compatible: Dict[str, torch.Tensor] = {}
        skipped = []
        for k, v in dino_state.items():
            if k in our_state and our_state[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append(k)

        missing, unexpected = self.load_state_dict(compatible, strict=False)
        del dino

        logger.info(
            f"DINOv2 weight transfer: {len(compatible)} transferred, "
            f"{len(skipped)} skipped (shape mismatch: {skipped[:5]}{'...' if len(skipped) > 5 else ''}), "
            f"{len(missing)} randomly initialized: {missing[:3]}"
        )

    def init_weights(self):
        super().init_weights()

    def _apply_feature_bottleneck(self, features):
        if self.feature_bottleneck is not None:
            return self.feature_bottleneck(features)
        return features

    def forward_features(
        self, x: Union[torch.Tensor, List[torch.Tensor]], masks: Optional[torch.Tensor] = None, **kwargs
    ):
        # Propagate arguments down to parent
        output = super().forward_features(
            x,
            masks,
            drop_ratio=kwargs.get("drop_ratio", None),
            return_mask=kwargs.get("return_mask", False),
            mask_ratio=kwargs.get("mask_ratio", 0.5),
        )

        use_bottleneck = kwargs.get("use_bottleneck", True)

        if self.feature_bottleneck is None or not use_bottleneck:
            return output

        if isinstance(output, list):
            processed_output = []
            for item in output:
                processed_item = self._process_output_dict(item)
                processed_output.append(processed_item)
            return processed_output
        else:
            return self._process_output_dict(output)

    def _process_output_dict(self, output_dict):
        # Preserve pre-bottleneck patch tokens for alignment loss (REPA-E style).
        output_dict["x_norm_patchtokens_raw"] = output_dict["x_norm_patchtokens"]

        # Apply bottleneck
        cls_token = self._apply_feature_bottleneck(output_dict["x_norm_clstoken"])

        # If we have 1D Latent Tokens (MAETok Paradigm 2), bottleneck them instead of patches
        if "x_norm_latenttokens" in output_dict:
            latent_tokens = output_dict["x_norm_latenttokens"]
            batch_size, num_latents, _ = latent_tokens.shape
            latent_tokens = self.feature_bottleneck(latent_tokens.reshape(-1, self.original_embed_dim))
            latent_tokens = latent_tokens.reshape(batch_size, num_latents, self.num_features)
            output_dict["x_norm_latenttokens"] = latent_tokens

            # The patches might still be passed down (e.g. for Dense Alignment or bypassing),
            # so we optionally bottleneck them too (or leave them as raw, but usually we just bottleneck the latents).
            # To be safe, we also bottleneck the patches if needed.
            patch_tokens = output_dict["x_norm_patchtokens"]
            batch_size, num_patches, _ = patch_tokens.shape
            patch_tokens = self.feature_bottleneck(patch_tokens.reshape(-1, self.original_embed_dim))
            patch_tokens = patch_tokens.reshape(batch_size, num_patches, self.num_features)
            output_dict["x_norm_patchtokens"] = patch_tokens
        else:
            patch_tokens = output_dict["x_norm_patchtokens"]
            batch_size, num_patches, _ = patch_tokens.shape
            patch_tokens = self.feature_bottleneck(patch_tokens.reshape(-1, self.original_embed_dim))
            patch_tokens = patch_tokens.reshape(batch_size, num_patches, self.num_features)
            output_dict["x_norm_patchtokens"] = patch_tokens

        output_dict["x_norm_clstoken"] = cls_token
        return output_dict

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        """
        Retrieves intermediate layer features from the model, bypassing the feature bottleneck.
        This method directly calls the parent class's implementation to return the
        original, high-dimensional features from the transformer blocks, which is
        typically desired for linear probing evaluations.
        """
        return super().get_intermediate_layers(
            x, n=n, reshape=reshape, return_class_token=return_class_token, norm=norm
        )

    def encode(self, x):
        """Encode images to latent features."""
        output = self.forward_features(x)
        b, n, c = output["x_norm_patchtokens"].shape
        h, w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        return output["x_norm_patchtokens"].reshape(b, h, w, c).permute(0, 3, 1, 2)
