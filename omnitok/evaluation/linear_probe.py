"""Linear probing evaluation for OmniTok.

Refactored from VTP to evaluate the semantic quality of the
continuous latent space.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("linear_probing_hf")


# ============================================================================
# Constants
# ============================================================================

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
CROP_SIZE = 224
RESIZE_SIZE = 256

DEFAULT_LEARNING_RATES = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1)


# ============================================================================
# Distributed utilities
# ============================================================================


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


# ============================================================================
# Transforms
# ============================================================================


def make_train_transform(crop_size: int = CROP_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def make_eval_transform(resize_size: int = RESIZE_SIZE, crop_size: int = CROP_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


# ============================================================================
# Feature Extraction Model
# ============================================================================


class FeatureExtractor(nn.Module):
    """Wrapper that extracts intermediate layer features from OmniTok."""

    def __init__(self, encoder: nn.Module, n_last_blocks: int, autocast_dtype: torch.dtype):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_dtype = autocast_dtype

    def forward(self, images: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract intermediate layer features.

        Returns:
            List of (patch_tokens, cls_token) tuples for each requested layer
        """
        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype):
                # OmniTok encoder outputs (B, L, D) or (B, D) directly
                features = self.encoder(images)

                # Mock the VTP intermediate_output format [(patch_tokens, cls_token)]
                if features.ndim == 3:
                    patch_tokens = features[:, 1:] if features.shape[1] > 256 else features
                    cls_token = features[:, 0] if features.shape[1] > 256 else features.mean(dim=1)
                else:
                    patch_tokens = features.unsqueeze(1)
                    cls_token = features

                return [(patch_tokens, cls_token)]


# ============================================================================
# Linear Classifier
# ============================================================================


def create_linear_input(x_tokens_list, use_n_blocks: int, use_avgpool: bool) -> torch.Tensor:
    """Create input for linear classifier from intermediate features."""
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)

    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)

    return output.float()


class LinearClassifier(nn.Module):
    """Linear classifier on top of frozen features."""

    def __init__(self, out_dim: int, use_n_blocks: int, use_avgpool: bool, num_classes: int = 1000):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear(output)


class AllClassifiers(nn.Module):
    """Container for multiple linear classifiers."""

    def __init__(self, classifiers_dict: Dict[str, nn.Module]):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs) -> Dict[str, torch.Tensor]:
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


# ============================================================================
# Infinite Sampler
# ============================================================================


class InfiniteSampler(Sampler):
    """Wraps another sampler to yield an infinite stream of indices."""

    def __init__(self, sampler: Sampler, shuffle: bool = True, seed: int = 0):
        self.sampler = sampler
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        while True:
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(self.epoch)
            yield from iter(self.sampler)
            self.epoch += 1

    def __len__(self):
        return int(1e18)  # Effectively infinite


# ============================================================================
# Training
# ============================================================================


def scale_lr(learning_rate: float, batch_size: int) -> float:
    """Scale learning rate based on batch size."""
    return learning_rate * (batch_size * get_world_size()) / 256.0


def setup_linear_classifiers(
    sample_output,
    n_last_blocks_list: Tuple[int, ...],
    learning_rates: Tuple[float, ...],
    batch_size: int,
    num_classes: int = 1000,
    device: torch.device = None,
) -> Tuple[AllClassifiers, List[Dict]]:
    """Setup linear classifiers with different configurations."""
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []

    for n in n_last_blocks_list:
        for avgpool in [True]:
            for _lr in learning_rates:
                lr = scale_lr(_lr, batch_size)
                out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                linear_classifier = LinearClassifier(
                    out_dim, use_n_blocks=n, use_avgpool=avgpool, num_classes=num_classes
                )
                linear_classifier = linear_classifier.to(device)
                classifier_key = f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.5f}".replace(".", "_")
                if is_main_process():
                    logger.info(f"Create linear classifier {classifier_key} with input_dim={out_dim}")
                linear_classifiers_dict[classifier_key] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if dist.is_initialized():
        linear_classifiers = nn.parallel.DistributedDataParallel(
            linear_classifiers, device_ids=[get_rank() % torch.cuda.device_count()]
        )

    return linear_classifiers, optim_param_groups


def train_one_epoch(
    feature_model: nn.Module,
    linear_classifiers: AllClassifiers,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    train_loader,
    epoch: int,
    epoch_length: int,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    linear_classifiers.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, total=epoch_length, desc=f"Epoch {epoch}") if is_main_process() else train_loader

    for batch_idx, (images, labels) in enumerate(progress_bar):
        if batch_idx >= epoch_length:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = feature_model(images)
        outputs = linear_classifiers(features)

        losses = {f"loss_{k}": criterion(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if is_main_process() and batch_idx % 50 == 0:
            progress_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    feature_model: nn.Module,
    linear_classifiers: AllClassifiers,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate all classifiers and return accuracies."""
    linear_classifiers.eval()

    classifiers_dict = (
        linear_classifiers.module.classifiers_dict
        if hasattr(linear_classifiers, "module")
        else linear_classifiers.classifiers_dict
    )

    correct = {k: 0 for k in classifiers_dict.keys()}
    total = 0

    progress_bar = tqdm(val_loader, desc="Evaluating") if is_main_process() else val_loader

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        features = feature_model(images)
        outputs = linear_classifiers(features)

        for k, logits in outputs.items():
            preds = logits.argmax(dim=1)
            correct[k] += (preds == labels).sum().item()

        total += labels.size(0)

    # Aggregate across processes
    if dist.is_initialized():
        # Gather correct counts and total from all processes
        total_tensor = torch.tensor([total], device=device, dtype=torch.long)
        dist.all_reduce(total_tensor)
        total = total_tensor.item()

        for k in correct:
            correct_tensor = torch.tensor([correct[k]], device=device, dtype=torch.long)
            dist.all_reduce(correct_tensor)
            correct[k] = correct_tensor.item()

    accuracies = {k: 100.0 * v / total for k, v in correct.items()}
    return accuracies


# ============================================================================
# Main
# ============================================================================


class LinearProbeEvaluator:
    """Linear probing evaluator using VTP's original SGD logic.

    Refactored from VTP/tools/test_linear_probing_hf.py to integrate with OmniTok.
    Note: Full SGD linear probing is slow. For fast inline evaluation during
    training, consider reducing epochs or learning_rates.
    """

    def __init__(
        self,
        batch_size: int = 256,
        epochs: int = 10,
        epoch_length: int = 1250,
        learning_rates: Tuple[float, ...] = DEFAULT_LEARNING_RATES,
        precision: str = "bf16",
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_length = epoch_length
        self.learning_rates = learning_rates
        self.precision = precision

    def compute_from_model(
        self,
        encoder: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Run full linear probing evaluation on the encoder."""
        num_classes = 1000  # Default ImageNet
        if hasattr(train_loader.dataset, "classes"):
            num_classes = len(train_loader.dataset.classes)

        n_last_blocks_list = (1,)
        n_last_blocks = 1

        if self.precision in ("bf16", "bfloat16"):
            autocast_dtype = torch.bfloat16
        elif self.precision in ("fp16", "float16"):
            autocast_dtype = torch.float16
        else:
            autocast_dtype = torch.float32

        feature_model = FeatureExtractor(encoder, n_last_blocks, autocast_dtype).to(device)

        # Get sample output for dimension calculation
        sample_image = next(iter(train_loader))[0][0:1].to(device)
        sample_output = feature_model(sample_image)

        linear_classifiers, optim_param_groups = setup_linear_classifiers(
            sample_output,
            n_last_blocks_list,
            self.learning_rates,
            self.batch_size,
            num_classes,
            device,
        )

        max_iter = self.epochs * self.epoch_length
        optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
        criterion = nn.CrossEntropyLoss()

        best_accuracy = 0.0
        best_classifier = ""

        for epoch in range(self.epochs):
            train_loss = train_one_epoch(
                feature_model,
                linear_classifiers,
                optimizer,
                scheduler,
                criterion,
                train_loader,
                epoch,
                self.epoch_length,
                device,
            )
            if is_main_process():
                logger.info(f"Epoch {epoch} train loss: {train_loss:.4f}")

            accuracies = evaluate(feature_model, linear_classifiers, val_loader, device)

            if accuracies:
                current_best_acc = max(accuracies.values())
                current_best_key = max(accuracies, key=accuracies.get)

                if current_best_acc > best_accuracy:
                    best_accuracy = current_best_acc
                    best_classifier = current_best_key

        return {
            "linear_probe_acc": best_accuracy,
            "best_classifier": best_classifier,
        }
