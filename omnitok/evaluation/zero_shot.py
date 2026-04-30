"""Zero-shot classification evaluator for tokenizer understanding quality.

Uses CLIP-style zero-shot classification: encodes images with the tokenizer
encoder, projects to a shared space, and computes cosine similarity with
text class embeddings.

This evaluator is applicable when the tokenizer includes a CLIP/SigLIP
alignment head. For tokenizers without text alignment, use LinearProbeEvaluator.

Reference:
    CLIP: Learning Transferable Visual Models From Natural Language Supervision
    OpenAI zero-shot evaluation protocol
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ImageNet class names (subset for quick eval)
IMAGENET_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
]


class ZeroShotEvaluator:
    """Zero-shot classification accuracy evaluator.

    Computes zero-shot top-1 accuracy by encoding images with the tokenizer
    encoder and comparing against text class embeddings via cosine similarity.

    Args:
        text_encoder: Frozen text encoder (CLIP/SigLIP text tower).
        class_names: List of class names (e.g., ImageNet-1K class names).
        templates: Prompt templates for text encoding.
        device: Compute device.
    """

    def __init__(
        self,
        text_encoder: Optional[nn.Module] = None,
        class_names: Optional[List[str]] = None,
        templates: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.text_encoder = text_encoder
        self.class_names = class_names or []
        self.templates = templates or IMAGENET_TEMPLATES
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._text_features: Optional[Tensor] = None

    @torch.no_grad()
    def _build_text_features(self) -> Tensor:
        """Build text class embeddings from class names and templates.

        Returns:
            Text features tensor (num_classes, D), L2-normalized.
        """
        if self.text_encoder is None:
            raise RuntimeError(
                "text_encoder required for zero-shot eval. "
                "Set text_encoder in ZeroShotEvaluator constructor."
            )

        self.text_encoder.eval()
        all_text_features = []

        for class_name in self.class_names:
            texts = [template.format(class_name) for template in self.templates]
            # Tokenize and encode text
            text_features = self.text_encoder(texts)  # (n_templates, D)
            text_features = F.normalize(text_features, dim=-1)
            # Average over templates
            class_feature = text_features.mean(dim=0)
            class_feature = F.normalize(class_feature, dim=-1)
            all_text_features.append(class_feature)

        return torch.stack(all_text_features, dim=0)  # (C, D)

    @torch.no_grad()
    def compute(
        self,
        image_features: Tensor,
        labels: Tensor,
        text_features: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Compute zero-shot accuracy from precomputed features.

        Args:
            image_features: (N, D) image features, L2-normalized.
            labels: (N,) ground truth class indices.
            text_features: (C, D) text class embeddings. If None, builds from text_encoder.

        Returns:
            Dict with 'zero_shot_acc' (top-1 %) and 'n_samples'.
        """
        if text_features is None:
            if self._text_features is None:
                self._text_features = self._build_text_features()
            text_features = self._text_features

        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1).to(image_features.device)

        # Cosine similarity → (N, C)
        logits = image_features @ text_features.T

        # Top-1 predictions
        preds = logits.argmax(dim=-1)
        correct = (preds == labels.to(preds.device)).float().sum().item()
        total = labels.shape[0]
        acc = 100.0 * correct / total

        logger.info(f"Zero-shot: acc={acc:.2f}% ({int(correct)}/{total})")

        return {
            "zero_shot_acc": acc,
            "n_samples": total,
            "n_classes": text_features.shape[0],
        }

    @torch.no_grad()
    def compute_from_model(
        self,
        encoder: nn.Module,
        dataloader: DataLoader,
        projector: Optional[nn.Module] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Extract image features and compute zero-shot accuracy.

        Args:
            encoder: Image encoder (frozen or trainable).
            dataloader: DataLoader yielding (images, labels).
            projector: Optional projection head to shared space.
            max_batches: Max batches to process.

        Returns:
            Zero-shot accuracy dict.
        """
        encoder.eval()
        if projector is not None:
            projector.eval()

        all_features = []
        all_labels = []

        for i, (images, labels) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            images = images.to(self.device)
            features = encoder(images)

            # Pool to (B, D) if needed
            if features.ndim == 3:
                features = features.mean(dim=1)  # mean pool tokens
            elif features.ndim == 4:
                features = features.reshape(features.shape[0], features.shape[1], -1).mean(dim=-1)

            # Project if needed
            if projector is not None:
                features = projector(features)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

        if not all_features:
            raise RuntimeError("No features extracted — empty dataloader?")

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        return self.compute(features, labels)
