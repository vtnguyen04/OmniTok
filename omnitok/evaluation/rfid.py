"""Reconstruction FID (rFID) evaluator.

Saves real and reconstructed images to temp dirs, then computes FID via
cleanfid's clean-resize pipeline (matches common paper conventions).

Reference:
    LightningDiT/evaluate_tokenizer.py — save-then-fid pattern
    cleanfid: https://github.com/GaParmar/clean-fid
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _tensor_to_uint8(x: Tensor) -> np.ndarray:
    """Convert [-1, 1] image tensor (C, H, W) to uint8 HWC numpy array."""
    x = x.detach().cpu().float()
    x = torch.clamp(127.5 * x + 128.0, 0, 255)
    return x.permute(1, 2, 0).numpy().astype(np.uint8)


class RFIDEvaluator:
    """Compute Reconstruction FID (rFID) for a tokenizer.

    Saves real and reconstructed images to temp directories, then calls
    cleanfid.fid.compute_fid(real_dir, recon_dir) — exactly the approach
    used in LightningDiT and VA-VAE papers.

    Args:
        batch_size: Batch size for inception feature extraction.
        num_workers: DataLoader workers for cleanfid.
        device: Compute device.
        max_images: Max images to evaluate (None = all).
        verbose: Whether cleanfid prints progress.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        device: Optional[torch.device] = None,
        max_images: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_images = max_images
        self.verbose = verbose

    @torch.no_grad()
    def compute(
        self,
        real_images: Tensor,
        recon_images: Tensor,
        work_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute rFID from paired real/reconstructed image tensors.

        Args:
            real_images: (N, C, H, W) in [-1, 1].
            recon_images: (N, C, H, W) in [-1, 1], same order as real.
            work_dir: Optional persistent dir (temp dir used if None).

        Returns:
            Dict with 'rfid' and 'n_images'.
        """
        assert real_images.shape == recon_images.shape, (
            f"Shape mismatch: {real_images.shape} vs {recon_images.shape}"
        )
        n = real_images.shape[0]

        ctx = tempfile.TemporaryDirectory() if work_dir is None else _NullCtx(work_dir)
        with ctx as tmp:
            real_dir = Path(tmp) / "real"
            recon_dir = Path(tmp) / "recon"
            real_dir.mkdir(parents=True, exist_ok=True)
            recon_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n):
                Image.fromarray(_tensor_to_uint8(real_images[i])).save(real_dir / f"{i:06d}.png")
                Image.fromarray(_tensor_to_uint8(recon_images[i])).save(recon_dir / f"{i:06d}.png")

            score = self._compute_fid(str(real_dir), str(recon_dir))

        logger.info(f"rFID: {score:.4f}  (N={n})")
        return {"rfid": score, "n_images": n}

    @torch.no_grad()
    def compute_from_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        n_batches: Optional[int] = None,
        work_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """Extract real + reconstructed images from model and compute rFID.

        Args:
            model: Tokenizer with .encode(x) and .decode(z) (or .forward(x) → recon).
            dataloader: Image DataLoader yielding (images, *) where images ∈ [-1, 1].
            n_batches: Max batches to process (None = full dataset).
            work_dir: Optional persistent directory for saved images.

        Returns:
            Dict with 'rfid' and 'n_images'.
        """
        model.eval()
        ctx = tempfile.TemporaryDirectory() if work_dir is None else _NullCtx(work_dir)

        with ctx as tmp:
            real_dir = Path(tmp) / "real"
            recon_dir = Path(tmp) / "recon"
            real_dir.mkdir(parents=True, exist_ok=True)
            recon_dir.mkdir(parents=True, exist_ok=True)

            total = 0
            for i, batch in enumerate(dataloader):
                if n_batches is not None and i >= n_batches:
                    break
                if self.max_images is not None and total >= self.max_images:
                    break

                images = batch[0].to(self.device)
                B = images.shape[0]

                # Support both encode+decode and forward interfaces
                if hasattr(model, "encode") and hasattr(model, "decode"):
                    z = model.encode(images)
                    recon = model.decode(z)
                else:
                    recon = model(images)

                real_u8 = images.detach().cpu()
                recon_u8 = recon.detach().cpu()

                for j in range(B):
                    idx = total + j
                    Image.fromarray(_tensor_to_uint8(real_u8[j])).save(real_dir / f"{idx:06d}.png")
                    Image.fromarray(_tensor_to_uint8(recon_u8[j])).save(recon_dir / f"{idx:06d}.png")

                total += B
                if i % 10 == 0:
                    logger.info(f"rFID: saved {total} image pairs...")

            if total == 0:
                raise RuntimeError("No images collected — empty dataloader?")

            score = self._compute_fid(str(real_dir), str(recon_dir))

        logger.info(f"rFID: {score:.4f}  (N={total})")
        return {"rfid": score, "n_images": total}

    def _compute_fid(self, real_dir: str, recon_dir: str) -> float:
        from cleanfid import fid
        return fid.compute_fid(
            real_dir,
            recon_dir,
            mode="clean",
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            verbose=self.verbose,
        )


class _NullCtx:
    """Context manager that wraps an existing directory (no cleanup)."""

    def __init__(self, path: str) -> None:
        self._path = path

    def __enter__(self) -> str:
        Path(self._path).mkdir(parents=True, exist_ok=True)
        return self._path

    def __exit__(self, *args) -> None:
        pass
