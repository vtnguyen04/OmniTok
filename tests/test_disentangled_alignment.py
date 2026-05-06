"""Unit tests for Disentangled Representation Alignment loss."""

import pytest
import torch

from omnitok.losses.alignment.disentangled_alignment import DisentangledAlignmentLoss


def test_disentangled_alignment_loss_forward():
    """Test that Disentangled alignment loss computes correctly."""
    loss_fn = DisentangledAlignmentLoss(
        student_dim=32,
        teacher_dim=1024,
        spatial_size=16,
        patch_size=1,
        hidden_size=128,  # Small hidden size for fast testing
        projector_dim=256,
        num_heads=4,
    )
    
    # Batch = 2, Channels = 32, Spatial = 16x16
    student_latents = torch.randn(2, 32, 16, 16)
    
    # Teacher features: Batch = 2, Patches = 256, Dim = 1024
    teacher_features = torch.randn(2, 256, 1024)
    
    loss = loss_fn(student_latents, teacher_features)
    
    assert loss.shape == ()
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
def test_disentangled_alignment_loss_shape_flexibility():
    """Test that Disentangled handles (B, N, C) input automatically."""
    loss_fn = DisentangledAlignmentLoss(
        student_dim=32,
        teacher_dim=1024,
        spatial_size=16,
        patch_size=1,
        hidden_size=128,
        projector_dim=256,
        num_heads=4,
    )
    
    # Student as (B, N, C)
    student_latents = torch.randn(2, 256, 32)
    teacher_features = torch.randn(2, 256, 1024)
    
    loss = loss_fn(student_latents, teacher_features)
    assert not torch.isnan(loss)
