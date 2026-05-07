"""Flow Matching Alignment Loss.

Implements the Semantic-aware End-to-End Alignment technique.
This loss adds diffusion noise to VAE latents using flow-matching interpolants
and uses a 1-layer ViT + MLP mapper to predict Vision Foundation Model features.
The alignment is enforced via Cosine Similarity loss.

Reference: Semantic-aware End-to-End VAE (Send-VAE)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from omnitok.registry import ALIGNMENT_REGISTRY
from omnitok.losses.alignment.base import BaseAlignmentLoss
from omnitok.models.layers.embeddings import get_2d_sincos_pos_embed

# Timm imports for the mapper architecture
from timm.models.vision_transformer import PatchEmbed, Block


class FlowMatchingMapper(nn.Module):
    """The 1-layer ViT + MLP projector used for flow-matching alignment."""

    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 1,
        in_channels: int = 32,
        hidden_size: int = 768,
        projector_dim: int = 2048,
        z_dims: int = 1024,
        bn_momentum: float = 0.1,
        heads: int = 12,
    ) -> None:
        super().__init__()
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        num_patches = self.x_embedder.num_patches
        self.bn = nn.BatchNorm2d(
            in_channels, eps=1e-4, momentum=bn_momentum, affine=False, track_running_stats=True
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.vit_mapper = nn.Sequential(
            Block(dim=hidden_size, num_heads=heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm)
        )

        self.projector = nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dims),
        )

        self.bn.reset_running_stats()
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Use the DRY function from omnitok.models.layers.embeddings
        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], grid_size, cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

    def interpolant(self, t: torch.Tensor, path_type: str = "linear") -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute flow matching interpolants."""
        if path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
        elif path_type == "cosine":
            alpha_t = torch.cos(t * torch.pi / 2)
            sigma_t = torch.sin(t * torch.pi / 2)
        else:
            raise ValueError(f"Unknown path_type: {path_type}")
        return alpha_t, sigma_t

    def forward(self, x: torch.Tensor, path_type: str = "linear") -> torch.Tensor:
        """Forward pass with noise addition and projection."""
        normalized_x = self.bn(x)

        # Add noise (uniform weighting)
        time_input = torch.rand((normalized_x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype)
        noises = torch.randn_like(normalized_x)
        alpha_t, sigma_t = self.interpolant(time_input, path_type=path_type)
        model_input = alpha_t * normalized_x + sigma_t * noises

        # Pass through mapper
        x_mapped = self.x_embedder(model_input) + self.pos_embed
        x_mapped = self.vit_mapper(x_mapped)
        proj_x = self.projector(x_mapped)
        return proj_x


@ALIGNMENT_REGISTRY.register("disentangled")
class DisentangledAlignmentLoss(BaseAlignmentLoss):
    """Disentangled Representation Alignment Loss.

    Extracts patches, adds diffusion noise using flow-matching interpolants,
    maps through 1 ViT block, and computes cosine similarity against teacher features.
    """

    def __init__(
        self,
        *,
        student_dim: int = 32,
        teacher_dim: int = 1024,
        spatial_size: int = 16,
        patch_size: int = 1,
        hidden_size: int = 768,
        projector_dim: int = 2048,
        num_heads: int = 12,
        path_type: str = "linear",
        **kwargs,
    ) -> None:
        super().__init__()
        self.path_type = path_type
        self.mapper = FlowMatchingMapper(
            input_size=spatial_size,
            patch_size=patch_size,
            in_channels=student_dim,
            hidden_size=hidden_size,
            projector_dim=projector_dim,
            z_dims=teacher_dim,
            heads=num_heads,
        )

    def compute(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the flow-matching projection loss.

        Args:
            student_features: VAE latents (B, C, H, W) or (B, N, C).
            teacher_features: Teacher features (B, N, D) or (B, D, H, W).
            mask: Unused in this alignment type.

        Returns:
            Scalar loss tensor.
        """
        # Ensure student is (B, C, H, W)
        if student_features.ndim == 3:
            B, N, C = student_features.shape
            H = int(math.sqrt(N))
            student_features = student_features.transpose(1, 2).view(B, C, H, H)

        # Ensure teacher is (B, N, D)
        if teacher_features.ndim == 4:
            B, C, H, W = teacher_features.shape
            teacher_features = teacher_features.view(B, C, -1).transpose(1, 2)

        # Forward mapper (adds noise and predicts teacher features)
        proj_x = self.mapper(student_features, path_type=self.path_type)

        # Compute cosine similarity loss
        proj_x = nn.functional.normalize(proj_x, dim=-1)
        teacher_features = nn.functional.normalize(teacher_features, dim=-1)

        # Loss = 1 - cosine_similarity
        cos_sim = (teacher_features * proj_x).sum(dim=-1)
        loss = (1 - cos_sim).mean()

        return loss
