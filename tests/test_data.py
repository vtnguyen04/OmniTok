"""Tests for data pipeline — transforms, datasets."""

import os

import pytest
import torch
import numpy as np
from PIL import Image

from omnitok.data.transforms import (
    center_crop_arr,
    random_crop_arr,
    build_train_transform,
    build_eval_transform,
)


@pytest.fixture
def sample_pil_image():
    """Create a 512x384 RGB test image."""
    arr = np.random.randint(0, 255, (384, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestCenterCrop:
    """Tests for center_crop_arr."""

    def test_output_size(self, sample_pil_image):
        result = center_crop_arr(sample_pil_image, 256)
        assert result.size == (256, 256)

    def test_small_image(self):
        """Works with images smaller than target (upscale first)."""
        small = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        result = center_crop_arr(small, 128)
        assert result.size == (128, 128)


class TestRandomCrop:
    """Tests for random_crop_arr."""

    def test_output_size(self, sample_pil_image):
        result = random_crop_arr(sample_pil_image, 256)
        assert result.size == (256, 256)


class TestBuildTransforms:
    """Tests for transform builders."""

    def test_train_transform_output(self, sample_pil_image):
        """Train transform produces tensor in [0, 1]."""
        transform = build_train_transform(image_size=128)
        tensor = transform(sample_pil_image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 128, 128)
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0

    def test_eval_transform_output(self, sample_pil_image):
        """Eval transform produces tensor in [0, 1]."""
        transform = build_eval_transform(image_size=128)
        tensor = transform(sample_pil_image)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 128, 128)

    def test_eval_deterministic(self, sample_pil_image):
        """Eval transform is deterministic."""
        transform = build_eval_transform(image_size=128)
        t1 = transform(sample_pil_image)
        t2 = transform(sample_pil_image)
        assert torch.equal(t1, t2)
