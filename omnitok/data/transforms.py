"""Image transforms — center/random crop from ADM, normalization.

Ported from continuous_tokenizer/utils/data.py.
"""

import math
import random

import numpy as np
from PIL import Image
from torchvision import transforms


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center cropping from ADM (guided-diffusion).

    Progressively downsamples with BOX filter first, then bicubic to target.

    Args:
        pil_image: Input PIL image.
        image_size: Target crop size.

    Returns:
        Center-cropped PIL image of size (image_size, image_size).
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def random_crop_arr(
    pil_image: Image.Image,
    image_size: int,
    min_crop_frac: float = 0.8,
    max_crop_frac: float = 1.0,
) -> Image.Image:
    """Random cropping from ADM with configurable crop fraction.

    Args:
        pil_image: Input PIL image.
        image_size: Target crop size.
        min_crop_frac: Minimum crop fraction.
        max_crop_frac: Maximum crop fraction.

    Returns:
        Random-cropped PIL image.
    """
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


_IMAGENET_MEAN = [0.5, 0.5, 0.5]
_IMAGENET_STD = [0.5, 0.5, 0.5]


def build_train_transform(image_size: int = 256, use_random_crop: bool = True) -> transforms.Compose:
    """Build training transform: crop + flip + normalize to [-1, 1].

    Matches the normalization convention used in continuous_tokenizer,
    LightningDiT, and REPA-E: mean=0.5, std=0.5 → output range [-1, 1].

    Args:
        image_size: Target image size.
        use_random_crop: Use random crop (True) or center crop (False).

    Returns:
        torchvision Compose transform.
    """
    crop_fn = random_crop_arr if use_random_crop else center_crop_arr
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: crop_fn(img, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 256) -> transforms.Compose:
    """Build evaluation transform: center crop + normalize to [-1, 1].

    Args:
        image_size: Target image size.

    Returns:
        torchvision Compose transform.
    """
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
