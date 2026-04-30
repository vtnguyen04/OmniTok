"""Dataset implementations — ImageFolder + H5 cached datasets.

Supports:
- ImageFolder: Standard torchvision ImageFolder for training from raw images.
- H5Dataset: REPA-E style h5 dataset for pre-cached latents/images.
- CachedLatentFolder: MAETok style cached latents for DiT training.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from .transforms import build_eval_transform, build_train_transform

logger = logging.getLogger(__name__)


class ImageFolderDataset(datasets.ImageFolder):
    """ImageFolder dataset with configurable transforms.

    Standard torchvision ImageFolder with OmniTok's train/eval transforms.

    Args:
        root: Path to ImageNet-style directory (root/class_name/images).
        image_size: Target image size.
        split: "train" or "val" — determines transform.
    """

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        split: str = "train",
    ) -> None:
        transform = (
            build_train_transform(image_size)
            if split == "train"
            else build_eval_transform(image_size)
        )
        super().__init__(root, transform=transform)
        logger.info(f"ImageFolderDataset: {len(self)} images from {root} ({split})")


class CachedLatentFolder(datasets.DatasetFolder):
    """Cached latent dataset for DiT training — ported from MAETok.

    Loads pre-computed tokenizer latents from .npz files.
    Supports random horizontal flip via pre-cached flipped latents.

    Args:
        root: Path to directory with .npz files.
        img_root: Optional path to original images (for reconstruction eval).
        return_img: Whether to also return the original image.
        transform: Optional transform for images.
    """

    def __init__(
        self,
        root: str,
        img_root: Optional[str] = None,
        return_img: bool = False,
        transform=None,
    ) -> None:
        super().__init__(root, loader=None, extensions=(".npz",), transform=transform)
        self.img_root = img_root
        self.return_img = return_img
        logger.info(f"CachedLatentFolder: {len(self)} latents from {root}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load cached latent + class label.

        Returns:
            (latent_tensor, class_label) or (latent, label, image) if return_img.
        """
        path, target = self.samples[index]
        data = np.load(path)

        # Random horizontal flip using pre-cached flipped version
        if "zq_flip" in data and torch.rand(1) < 0.5:
            zq = data["zq_flip"]
        else:
            zq = data["zq"]

        if self.return_img and self.img_root:
            img_path = os.path.join(self.img_root, str(data["path"]))
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return torch.from_numpy(zq), target, img

        return torch.from_numpy(zq), target
