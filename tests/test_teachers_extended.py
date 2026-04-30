"""Unit tests for SAM and Depth Anything teachers."""

import pytest
import torch
from unittest.mock import patch, MagicMock

from omnitok.teachers.sam import SAMTeacher
from omnitok.teachers.depth_anything import DepthAnythingTeacher

class TestSAMTeacher:
    """Test SAM teacher extraction logic."""

    @patch("omnitok.teachers.sam.timm.create_model")
    def test_sam_extract_features(self, mock_create_model):
        """Test SAM extracts and reshapes features correctly."""
        # Setup mock SAM model
        mock_model = MagicMock()
        # timm SAM forward_features returns (B, D, H, W)
        # For a 256x256 input, it returns 16x16 tokens of dimension 256
        mock_model.forward_features.return_value = torch.randn(2, 256, 16, 16)
        mock_create_model.return_value = mock_model

        teacher = SAMTeacher(model_name="sam_vit_b", device="cpu")
        x = torch.randn(2, 3, 256, 256)
        
        feats = teacher(x)  # calls _extract_features

        # Check mock was called
        mock_model.forward_features.assert_called_once()
        # Feature shape should be (B, N, D) -> (2, 256, 256)
        assert feats.shape == (2, 256, 256)
        assert teacher.feature_dim == 256
        assert teacher.patch_size == 16


class TestDepthAnythingTeacher:
    """Test Depth Anything teacher extraction logic."""

    @patch("omnitok.teachers.depth_anything.AutoModelForDepthEstimation.from_pretrained")
    def test_depth_anything_extract_features_no_resize(self, mock_from_pretrained):
        """Test extraction without resizing (input already divisible by patch_size)."""
        mock_model = MagicMock()
        # Depth Anything returns an object with hidden_states
        mock_output = MagicMock()
        # 1 CLS token + 16*16 patch tokens = 257 tokens
        mock_output.hidden_states = [torch.randn(2, 257, 384)]
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        teacher = DepthAnythingTeacher(model_name="depth_anything_v2_small", device="cpu")
        # 224x224 input, patch_size 14 -> 16x16 patches = 256 patches.
        x = torch.randn(2, 3, 224, 224)
        
        feats = teacher(x)

        # Should strip CLS token, returning (B, 256, 384)
        assert feats.shape == (2, 256, 384)
        assert teacher.feature_dim == 384
        assert teacher.patch_size == 14

    @patch("omnitok.teachers.depth_anything.AutoModelForDepthEstimation.from_pretrained")
    def test_depth_anything_extract_features_with_resize(self, mock_from_pretrained):
        """Test extraction with resizing (input not divisible by patch_size)."""
        mock_model = MagicMock()
        mock_output = MagicMock()
        # Expecting 16x16=256 patches + 1 CLS token
        mock_output.hidden_states = [torch.randn(2, 257, 384)]
        mock_model.return_value = mock_output
        mock_from_pretrained.return_value = mock_model

        teacher = DepthAnythingTeacher(model_name="depth_anything_v2_small", device="cpu")
        # 256x256 input is not divisible by 14. Should be interpolated to 224x224 inside _extract_features.
        x = torch.randn(2, 3, 256, 256)
        
        feats = teacher(x)

        # Model should have been called with interpolated input shape (2, 3, 224, 224)
        called_x = mock_model.call_args[0][0]
        assert called_x.shape == (2, 3, 224, 224)
        assert feats.shape == (2, 256, 384)

