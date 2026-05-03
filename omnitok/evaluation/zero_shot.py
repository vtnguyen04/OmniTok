"""Zero-shot classification evaluator for OmniTok.

Refactored from VTP/tools/test_zero_shot_hf.py to evaluate alignment of the
continuous latent space with CLIP/SigLIP text embeddings.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .imagenet_classes import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

logger = logging.getLogger(__name__)


class ZeroShotEvaluator:
    """Zero-shot classification evaluator.

    Builds a text classifier from class names + prompt templates (ported from
    VTP's test_zero_shot_hf.py), then computes top-1/top-5 accuracy by
    comparing image features against the text classifier weights.

    Args:
        text_encoder: Module that maps list[str] → (N, D) normalized features.
        class_names: Class name list (defaults to ImageNet-1k).
        templates: Prompt template functions, each callable(class_name) → str.
        device: Compute device.
    """

    def __init__(
        self,
        text_encoder: Optional[nn.Module] = None,
        class_names: Optional[Tuple] = None,
        templates: Optional[Tuple] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.text_encoder = text_encoder
        self.class_names = class_names or IMAGENET_CLASSNAMES
        self.templates = templates or OPENAI_IMAGENET_TEMPLATES
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_weights: Optional[torch.Tensor] = None

    def _build_classifier(self) -> torch.Tensor:
        """Build zero-shot classifier weight matrix from text embeddings.

        Returns:
            (D, num_classes) weight matrix — each column is the mean-pooled,
            normalized text embedding for one class.
        """
        if self.text_encoder is None:
            raise RuntimeError("text_encoder is required for zero-shot eval.")

        self.text_encoder.eval()
        all_text_features = []

        with torch.no_grad():
            for class_name in tqdm(self.class_names, desc="Building zero-shot classifier"):
                texts = [template(class_name) for template in self.templates]
                text_features = self.text_encoder(texts)  # (n_templates, D)
                text_features = F.normalize(text_features, dim=-1)
                class_feature = F.normalize(text_features.mean(dim=0), dim=-1)
                all_text_features.append(class_feature)

        return torch.stack(all_text_features, dim=0).T  # (D, num_classes)

    def compute_from_model(
        self,
        encoder: nn.Module,
        dataloader: DataLoader,
        projector: Optional[nn.Module] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Extract image features and compute zero-shot top-1/top-5 accuracy.

        Args:
            encoder: Image encoder. Output: (B, D), (B, L, D), or (B, C, h, w).
            dataloader: DataLoader yielding (images, labels).
            projector: Optional projection head applied after encoder.
            max_batches: Limit evaluation batches.

        Returns:
            Dict with 'zero_shot_acc' (top-1 %), 'zero_shot_top5', 'n_samples'.
        """
        if self.classifier_weights is None:
            self.classifier_weights = self._build_classifier()

        classifier = self.classifier_weights.to(self.device)
        encoder.eval()
        if projector is not None:
            projector.eval()

        top1_count, top5_count, n = 0.0, 0.0, 0

        with torch.inference_mode():
            for i, (images, targets) in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break

                images = images.to(self.device)
                targets = targets.to(self.device)

                features = encoder(images)

                # Pool to (B, D)
                if features.ndim == 4:
                    B, C, h, w = features.shape
                    features = features.reshape(B, C, -1).mean(dim=-1)
                elif features.ndim == 3:
                    features = features.mean(dim=1)

                if projector is not None:
                    features = projector(features)

                features = F.normalize(features.float(), dim=-1)
                logits = 100.0 * features @ classifier

                acc1, acc5 = _topk_accuracy(logits, targets, topk=(1, 5))
                top1_count += acc1
                top5_count += acc5
                n += images.size(0)

        if n == 0:
            raise RuntimeError("No samples evaluated — empty dataloader?")

        top1 = top1_count / n * 100
        top5 = top5_count / n * 100

        logger.info(f"Zero-shot: top1={top1:.2f}%, top5={top5:.2f}% (n={n})")

        return {
            "zero_shot_acc": top1,
            "zero_shot_top5": top5,
            "n_samples": n,
        }


def _topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> List[float]:
    """Return per-k correct prediction counts (not percentages)."""
    maxk = max(topk)
    pred = output.topk(maxk, dim=1, largest=True, sorted=True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum().item()) for k in topk]
