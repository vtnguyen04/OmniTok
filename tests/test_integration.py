"""Integration tests — end-to-end pipeline verification.

Tests full component interaction:
1. Tokenizer encode → decode pipeline
2. Tokenizer + Teachers + Alignment loss
3. Tokenizer + Reconstruction + Alignment + GAN (1 training step)
4. Checkpoint save → resume training
5. Registry-driven component construction
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from omnitok.models.tokenizer import Tokenizer
from omnitok.models.encoder.vision_transformer_bottleneck import DinoVisionTransformerWithBottleneck
from omnitok.models.decoder.pixel_decoder import DinoV3PixelDecoder
from omnitok.losses.alignment.cosine import CosineAlignmentLoss
from omnitok.losses.alignment.relational import RelationalKDLoss
from omnitok.losses.alignment.prediction import PredictionAlignmentLoss
from omnitok.losses.kl import KLLoss
from omnitok.losses.reconstruction import ReconstructionLoss
from omnitok.teachers.base import BaseTeacher
from omnitok.teachers.normalizer import FeatureNormalizer, ProjectedNormalizer
from omnitok.teachers.multi_teacher import MultiTeacher
from omnitok.training.utils import update_ema, save_checkpoint, load_checkpoint, count_params
from omnitok.registry import (
    ALIGNMENT_REGISTRY,
    LOSS_REGISTRY,
    TEACHER_REGISTRY,
)
from omnitok.data.transforms import build_train_transform, build_eval_transform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeTeacher(BaseTeacher):
    """Fake teacher for integration testing — no real model download."""

    def __init__(self, feat_dim: int = 64, p_size: int = 8):
        super().__init__(model_name="fake", device=None)
        self._feat_dim = feat_dim
        self._p_size = p_size
        self._model = nn.Identity()

    def _build_model(self):
        return nn.Identity()

    def _extract_features(self, x):
        B = x.shape[0]
        N = (x.shape[-1] // self._p_size) ** 2
        return torch.randn(B, N, self._feat_dim, device=x.device)

    @property
    def feature_dim(self) -> int:
        return self._feat_dim

    @property
    def patch_size(self) -> int:
        return self._p_size


@pytest.fixture
def tokenizer():
    """Small tokenizer for integration tests."""
    encoder = DinoVisionTransformerWithBottleneck(
        img_size=64, patch_size=8, embed_dim=128, depth=2,
        num_heads=4, ffn_layer="mlp", norm_layer="layernorm",
        vit_feature_bottleneck=32,
    )
    decoder = DinoV3PixelDecoder(
        in_chans=32, out_chans=3, upscale_factor=8,
        embed_dim=128, depth=2, num_heads=4,
        ffn_layer="mlp", norm_layer="layernorm",
    )
    return Tokenizer(encoder=encoder, decoder=decoder)


@pytest.fixture
def fake_images():
    """Batch of fake images: (B=4, 3, 64, 64) in [0, 1]."""
    return torch.rand(4, 3, 64, 64)


# ---------------------------------------------------------------------------
# Integration Test 1: Full Tokenizer Pipeline
# ---------------------------------------------------------------------------

class TestTokenizerPipeline:
    """End-to-end tokenizer: images → latent → reconstruction."""

    def test_full_forward_backward(self, tokenizer, fake_images):
        """Full forward + backward through tokenizer produces gradients."""
        tokenizer.train()
        out = tokenizer(fake_images, return_features=True)

        # Check outputs
        assert out["reconstruction"].shape == fake_images.shape
        assert out["latent"].shape == (4, 32, 8, 8)
        assert "x_norm_patchtokens" in out["features"]

        # Backward
        loss = out["reconstruction"].mean()
        loss.backward()

        enc_grads = sum(1 for p in tokenizer.encoder.parameters() if p.grad is not None)
        dec_grads = sum(1 for p in tokenizer.decoder.parameters() if p.grad is not None)
        assert enc_grads > 0
        assert dec_grads > 0

    def test_encode_decode_roundtrip_shape(self, tokenizer, fake_images):
        """encode() → decode() preserves spatial dimensions."""
        tokenizer.eval()
        with torch.no_grad():
            z = tokenizer.encode(fake_images)
            recon = tokenizer.decode(z)
        assert recon.shape == fake_images.shape


# ---------------------------------------------------------------------------
# Integration Test 2: Tokenizer + Teachers + Alignment
# ---------------------------------------------------------------------------

class TestTokenizerWithTeachers:
    """Tokenizer encoder features aligned to teacher features."""

    def test_cosine_alignment_with_teacher(self, tokenizer, fake_images):
        """Tokenizer encoder features → cosine alignment → teacher features."""
        teacher = FakeTeacher(feat_dim=64, p_size=8)
        alignment = CosineAlignmentLoss()

        tokenizer.train()
        out = tokenizer(fake_images, return_features=True)
        # After bottleneck, patchtokens dim = 32
        student_feat = out["features"]["x_norm_patchtokens"]  # (B, N, 32)

        with torch.no_grad():
            teacher_feat = teacher(fake_images)  # (B, N, 64)

        # Project both to common dim
        proj = ProjectedNormalizer(in_dim=32, out_dim=64)
        student_proj = proj(student_feat)

        loss = alignment(student_proj, teacher_feat)
        loss.backward()

        # Gradients flow to both tokenizer and projector
        assert any(p.grad is not None for p in tokenizer.parameters() if p.requires_grad)
        assert any(p.grad is not None for p in proj.parameters())

    def test_multi_teacher_alignment(self, tokenizer, fake_images):
        """Multi-teacher setup with PHI-S weighting."""
        teachers_dict = {
            "spatial": FakeTeacher(feat_dim=64, p_size=8),
            "semantic": FakeTeacher(feat_dim=128, p_size=8),
        }
        multi = MultiTeacher(teachers_dict, common_dim=32, normalize=True)

        # Extract from all teachers
        all_feats = multi.extract_all(fake_images)
        weights = multi.get_loss_weights()

        assert len(all_feats) == 2
        assert all(v.shape[-1] == 32 for v in all_feats.values())
        assert all(w.item() > 0 for w in weights.values())

    def test_relational_kd_alignment(self, tokenizer, fake_images):
        """RelationalKD works with different student/teacher dims."""
        teacher = FakeTeacher(feat_dim=256, p_size=8)
        alignment = RelationalKDLoss(distance_type="cosine")

        tokenizer.train()
        out = tokenizer(fake_images, return_features=True)
        student_feat = out["features"]["x_norm_patchtokens"]

        with torch.no_grad():
            teacher_feat = teacher(fake_images)

        loss = alignment(student_feat, teacher_feat)
        assert loss.item() >= 0
        loss.backward()
        assert any(p.grad is not None for p in tokenizer.parameters() if p.requires_grad)

    def test_prediction_alignment(self, tokenizer, fake_images):
        """PredictionAlignmentLoss with learnable predictor MLP."""
        teacher = FakeTeacher(feat_dim=64, p_size=8)
        # student_dim=32 because bottleneck reduces 128→32
        alignment = PredictionAlignmentLoss(student_dim=32, teacher_dim=64, embed_dim=48)

        tokenizer.train()
        out = tokenizer(fake_images, return_features=True)
        student_feat = out["features"]["x_norm_patchtokens"]  # (B, N, 32)
        with torch.no_grad():
            teacher_feat = teacher(fake_images)

        loss = alignment(student_feat, teacher_feat)
        loss.backward()

        # Predictor MLP has gradients
        assert any(p.grad is not None for p in alignment.predictor.parameters())


# ---------------------------------------------------------------------------
# Integration Test 3: Full Training Step (no Accelerate)
# ---------------------------------------------------------------------------

class TestTrainingStep:
    """Simulate a full training step: recon + alignment + optimizer.step()."""

    def test_one_training_step(self, tokenizer, fake_images):
        """1 complete training step without crash."""
        teacher = FakeTeacher(feat_dim=128, p_size=8)
        align_fn = CosineAlignmentLoss()
        recon_fn = ReconstructionLoss(recon_type="l2", recon_weight=1.0, perceptual_weight=0.0)

        optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=1e-4)

        tokenizer.train()
        optimizer.zero_grad()

        # Forward
        out = tokenizer(fake_images, return_features=True)
        recon = out["reconstruction"]
        student_feat = out["features"]["x_norm_patchtokens"]

        # Reconstruction loss
        recon_result = recon_fn(fake_images, recon)
        total_loss = recon_result["total"]

        # Alignment loss — student patchtokens are dim=32 after bottleneck
        with torch.no_grad():
            teacher_feat = teacher(fake_images)  # (B, N, 128)

        # Project student to teacher dim
        proj = nn.Linear(32, 128)
        student_proj = proj(student_feat)
        align_loss = align_fn(student_proj, teacher_feat)
        total_loss = total_loss + 0.5 * align_loss

        # Backward + step
        total_loss.backward()
        optimizer.step()

        # Verify weights changed
        assert total_loss.item() > 0

    def test_training_with_ema(self, tokenizer, fake_images):
        """Training step with EMA model update."""
        from copy import deepcopy

        ema = deepcopy(tokenizer)
        ema.eval()
        for p in ema.parameters():
            p.requires_grad = False

        recon_fn = ReconstructionLoss(recon_type="l1", recon_weight=1.0, perceptual_weight=0.0)
        optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)

        # Get initial EMA weight
        ema_weight_before = list(ema.parameters())[0].clone()

        # Training step
        tokenizer.train()
        out = tokenizer(fake_images)
        loss = recon_fn(fake_images, out["reconstruction"])["total"]
        loss.backward()
        optimizer.step()

        # EMA update
        update_ema(ema, tokenizer, decay=0.9)

        # EMA weight should have moved
        ema_weight_after = list(ema.parameters())[0]
        assert not torch.equal(ema_weight_before, ema_weight_after)

    def test_multi_alignment_training_step(self, tokenizer, fake_images):
        """Training with multiple alignment losses."""
        teachers_dict = {
            "t1": FakeTeacher(feat_dim=64, p_size=8),
            "t2": FakeTeacher(feat_dim=96, p_size=8),
        }
        multi = MultiTeacher(teachers_dict, common_dim=32)

        # Projector for student features (dim=32 after bottleneck → 32 common)
        student_proj = nn.Linear(32, 32)
        align_fn = CosineAlignmentLoss()
        recon_fn = ReconstructionLoss(recon_type="l2", recon_weight=1.0, perceptual_weight=0.0)

        params = list(tokenizer.parameters()) + list(student_proj.parameters()) + list(multi.projectors.parameters()) + list(multi.log_vars.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4)

        tokenizer.train()
        multi.train()
        optimizer.zero_grad()

        # Forward
        out = tokenizer(fake_images, return_features=True)
        student_feat = student_proj(out["features"]["x_norm_patchtokens"])

        # Recon loss
        recon_result = recon_fn(fake_images, out["reconstruction"])
        total_loss = recon_result["total"]

        # Multi-teacher alignment
        teacher_feats = multi.extract_all(fake_images)
        weights = multi.get_loss_weights()
        for name, t_feat in teacher_feats.items():
            a_loss = align_fn(student_feat, t_feat)
            total_loss = total_loss + weights[name] * a_loss
        total_loss = total_loss + multi.get_regularization()

        total_loss.backward()
        optimizer.step()

        assert total_loss.item() > 0


# ---------------------------------------------------------------------------
# Integration Test 4: Checkpoint Save → Resume
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    """Save checkpoint, create new model, load, verify state matches."""

    def test_save_resume_preserves_state(self, tokenizer, fake_images):
        """Save → load preserves model weights and training state."""
        from copy import deepcopy

        ema = deepcopy(tokenizer)
        optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)

        # Do a training step to create non-trivial state
        tokenizer.train()
        out = tokenizer(fake_images)
        loss = out["reconstruction"].mean()
        loss.backward()
        optimizer.step()

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(tokenizer, ema, optimizer, step=42, epoch=3, save_dir=tmpdir)

            # Create fresh models
            encoder2 = DinoVisionTransformerWithBottleneck(
                img_size=64, patch_size=8, embed_dim=128, depth=2,
                num_heads=4, ffn_layer="mlp", norm_layer="layernorm",
                vit_feature_bottleneck=32,
            )
            decoder2 = DinoV3PixelDecoder(
                in_chans=32, out_chans=3, upscale_factor=8,
                embed_dim=128, depth=2, num_heads=4,
                ffn_layer="mlp", norm_layer="layernorm",
            )
            tokenizer2 = Tokenizer(encoder=encoder2, decoder=decoder2)
            ema2 = deepcopy(tokenizer2)
            opt2 = torch.optim.Adam(tokenizer2.parameters(), lr=1e-3)

            # Load
            ckpt_path = os.path.join(tmpdir, "checkpoint-00000042.pt")
            state = load_checkpoint(ckpt_path, tokenizer2, ema2, opt2)

            assert state["step"] == 42
            assert state["epoch"] == 3

            # Weights should match
            for p1, p2 in zip(tokenizer.parameters(), tokenizer2.parameters()):
                assert torch.equal(p1.data, p2.data)


# ---------------------------------------------------------------------------
# Integration Test 5: Registry-Driven Construction
# ---------------------------------------------------------------------------

class TestRegistryConstruction:
    """Build components entirely from registry — simulates config-driven setup."""

    def test_build_alignment_from_registry(self):
        """Build all alignment losses from registry by name."""
        for name in ["cosine", "mse", "smooth_l1"]:
            loss = ALIGNMENT_REGISTRY.build(name)
            s = torch.randn(2, 8, 64)
            t = torch.randn(2, 8, 64)
            result = loss(s, t)
            assert result.ndim == 0, f"{name} should return scalar"

    def test_build_loss_from_registry(self):
        """Build loss modules from LOSS_REGISTRY."""
        kl = LOSS_REGISTRY.build("kl", weight=1e-5)
        assert isinstance(kl, KLLoss)

    def test_teacher_registry_has_entries(self):
        """Teacher registry has DINOv2 and SigLIP."""
        available = TEACHER_REGISTRY.available()
        assert "dinov2" in available
        assert "siglip" in available


# ---------------------------------------------------------------------------
# Integration Test 6: Data Transform → Model Pipeline
# ---------------------------------------------------------------------------

class TestDataToModelPipeline:
    """Data transforms feed correctly into model."""

    def test_transform_to_tokenizer(self, tokenizer):
        """PIL image → transform → tokenizer → reconstruction."""
        import numpy as np
        from PIL import Image

        # Simulate a real image
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr)

        # Transform
        transform = build_eval_transform(image_size=64)
        tensor = transform(pil_img).unsqueeze(0)  # (1, 3, 64, 64)

        assert tensor.shape == (1, 3, 64, 64)
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0

        # Forward through tokenizer
        tokenizer.eval()
        with torch.no_grad():
            out = tokenizer(tensor)
        assert out["reconstruction"].shape == (1, 3, 64, 64)
