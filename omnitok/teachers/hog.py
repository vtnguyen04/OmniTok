import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnitok.registry import TEACHER_REGISTRY
from omnitok.teachers.base import BaseTeacher


class HOGGenerator(nn.Module):
    """Generate HOG feature for images.
    Ported from MAETok / MaskFeat.
    """

    def __init__(self, nbins: int = 9, pool: int = 8, gaussian_window: int = 16) -> None:
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1).contiguous()
        weight_y = weight_x.transpose(2, 3).contiguous()
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gaussian_kernel = self.get_gaussian_kernel(gaussian_window, gaussian_window // 2)
            self.register_buffer('gaussian_kernel', gaussian_kernel)

    def get_gaussian_kernel(self, kernlen: int, std: int) -> torch.Tensor:
        def _gaussian_fn(kernlen: int, std: int) -> torch.Tensor:
            n = torch.arange(0, kernlen).float()
            n -= n.mean()
            n /= std
            w = torch.exp(-0.5 * n**2)
            return w

        kernel_1d = _gaussian_fn(kernlen, std)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d / kernel_2d.sum()

    def _reshape(self, hog_feat: torch.Tensor) -> torch.Tensor:
        """Reshape HOG Features for output. Returns (B, N, D)."""
        hog_feat = hog_feat.flatten(1, 2)
        # We assume 16x16 patch size usually, but here we just follow MAETok's logic
        unfold_size = hog_feat.shape[-1] // 16
        if unfold_size == 0:
            unfold_size = 1 # fallback
        hog_feat = hog_feat.permute(0, 2, 3, 1)
        hog_feat = hog_feat.unfold(1, unfold_size, unfold_size).unfold(2, unfold_size, unfold_size)
        hog_feat = hog_feat.flatten(1, 2).flatten(2)
        return hog_feat

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
        gx_rgb = F.conv2d(x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w), dtype=torch.float, device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)

        if self.gaussian_window:
            if h != self.gaussian_window:
                repeat_rate = h // self.gaussian_window
                temp_gaussian_kernel = self.gaussian_kernel.repeat([repeat_rate, repeat_rate])
            else:
                temp_gaussian_kernel = self.gaussian_kernel
            norm_rgb *= temp_gaussian_kernel

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = F.normalize(out, p=2, dim=2)
        return self._reshape(out)


@TEACHER_REGISTRY.register("hog")
class HOGTeacher(BaseTeacher):
    """HOG Feature Extractor as a Teacher (used in MAETok)."""

    def __init__(self, model_name: str = "hog_8", **kwargs) -> None:
        super().__init__(model_name=model_name, **kwargs)

    def _build_model(self) -> nn.Module:
        return HOGGenerator(nbins=9, pool=8, gaussian_window=16)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # HOG generator expects RGB images.
        # BaseTeacher.preprocess converts [-1, 1] -> [0, 1] then normalizes.
        # HOG usually works on [0, 1] raw or similar.
        return self._model(x)

    @property
    def feature_dim(self) -> int:
        return 108 # 9 bins * 3 channels * (16/pool)^2 = 9 * 3 * 4 = 108

    @property
    def patch_size(self) -> int:
        return 16 # HOGGenerator usually produces 16x16 patch equivs

