"""Training utilities — EMA, checkpointing, seeding.

Ported from REPA-E and MAETok.
"""

import logging
import os
import random
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    """Update EMA model parameters toward current model.

    Ported from REPA-E — also handles batch norm buffers.

    Args:
        ema_model: EMA model to update.
        model: Current training model.
        decay: EMA decay rate (0.9999 typical).
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # EMA on float buffers (e.g., BN running stats)
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        name = name.replace("module.", "")
        if buffer.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)
        else:
            ema_buffers[name].copy_(buffer)


def requires_grad(model: nn.Module, flag: bool = True) -> None:
    """Set requires_grad for all model parameters."""
    for p in model.parameters():
        p.requires_grad = flag


def fix_seeds(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: nn.Module,
    ema_model: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    save_dir: str,
    max_keep: int = 5,
) -> str:
    """Save training checkpoint.

    Args:
        model: Training model.
        ema_model: EMA model (optional).
        optimizer: Optimizer state.
        step: Current global step.
        epoch: Current epoch.
        save_dir: Directory to save checkpoints.
        max_keep: Max checkpoints to keep (oldest deleted).

    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"checkpoint-{step:08d}.pt")

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    if ema_model is not None:
        state["ema_model"] = ema_model.state_dict()

    torch.save(state, ckpt_path)
    logger.info(f"Saved checkpoint: {ckpt_path}")

    # Manage old checkpoints
    _cleanup_checkpoints(save_dir, max_keep)

    return ckpt_path


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    ema_model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load checkpoint and restore state.

    Args:
        ckpt_path: Path to checkpoint file.
        model: Model to load state into.
        ema_model: Optional EMA model.
        optimizer: Optional optimizer.

    Returns:
        Dict with 'step' and 'epoch'.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    logger.info(f"Loaded model from {ckpt_path}")

    if ema_model is not None and "ema_model" in state:
        ema_model.load_state_dict(state["ema_model"])
        logger.info("Loaded EMA model")

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        logger.info("Loaded optimizer state")

    return {"step": state.get("step", 0), "epoch": state.get("epoch", 0)}


def _cleanup_checkpoints(save_dir: str, max_keep: int) -> None:
    """Delete oldest checkpoints beyond max_keep."""
    ckpts = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("checkpoint-") and f.endswith(".pt")]
    )
    while len(ckpts) > max_keep:
        old = ckpts.pop(0)
        os.remove(os.path.join(save_dir, old))
        logger.info(f"Removed old checkpoint: {old}")


def count_params(model: nn.Module) -> dict:
    """Count model parameters.

    Returns:
        Dict with 'total', 'trainable', 'frozen' param counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }
