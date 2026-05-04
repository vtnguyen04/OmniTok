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
    loss: Optional[float] = None,
) -> str:
    """Save training checkpoint — only keeps 'last.pt' and 'best.pt'.

    Ported checkpoint strategy from continuous_tokenizer (manage_checkpoints).

    Args:
        model: Training model.
        ema_model: EMA model (optional).
        optimizer: Optimizer state.
        step: Current global step.
        epoch: Current epoch.
        save_dir: Directory to save checkpoints.
        loss: Current loss value for best checkpoint tracking.

    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss": loss,
    }
    if ema_model is not None:
        state["ema_model"] = ema_model.state_dict()

    # Always save last.pt — write to tmp then atomic rename to avoid partial writes
    last_path = os.path.join(save_dir, "last.pt")
    tmp_path = last_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, last_path)  # atomic on POSIX
    logger.info(f"Saved checkpoint: {last_path} (step={step})")

    # Save best.pt if loss is lower than previous best
    if loss is not None:
        best_path = os.path.join(save_dir, "best.pt")
        save_best = True
        if os.path.exists(best_path):
            prev = torch.load(best_path, map_location="cpu", weights_only=False)
            prev_loss = prev.get("loss")
            if prev_loss is not None and loss >= prev_loss:
                save_best = False
        if save_best:
            torch.save(state, best_path)
            logger.info(f"New best checkpoint: loss={loss:.6f} (step={step})")

    return last_path


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    ema_model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    resume_weights_only: bool = False,
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
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        logger.info(f"Loaded model (new params randomly init: {missing[:3]}{'...' if len(missing) > 3 else ''})")
    else:
        logger.info(f"Loaded model from {ckpt_path}")

    if ema_model is not None and "ema_model" in state:
        ema_model.load_state_dict(state["ema_model"], strict=False)
        logger.info("Loaded EMA model")

    if optimizer is not None and "optimizer" in state and not resume_weights_only:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except ValueError as e:
            logger.warning(f"Optimizer state mismatch (model architecture changed?) — starting fresh: {e}")
        logger.info("Loaded optimizer state")

    return {"step": state.get("step", 0), "epoch": state.get("epoch", 0)}


def _cleanup_checkpoints(save_dir: str, max_keep: int) -> None:
    """Delete oldest checkpoints beyond max_keep."""
    ckpts = sorted([f for f in os.listdir(save_dir) if f.startswith("checkpoint-") and f.endswith(".pt")])
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
