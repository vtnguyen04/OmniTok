#!/usr/bin/env python
"""OmniTok — Evaluation entry point.

Loads a trained tokenizer from an experiment output directory (which contains
config.yaml + checkpoints/), runs all evaluation metrics, saves results and
publication-quality figures.

Metrics:
    - PSNR          : Pixel-level fidelity (always on, cheap)
    - Gaussianity   : UNE hypothesis — fraction of Gaussian latent dims
    - rFID          : Reconstruction FID via cleanfid (optional, slow)
    - LinearProbe   : Semantic quality of encoder features (optional)

Outputs (all inside --output-dir):
    eval_results.json / eval_results.csv   — metric numbers
    figures/recon_grid.png                 — orig vs. recon comparison grid
    figures/loss_curves.pdf                — training loss curves (if metrics.json present)

Usage:
    # Full eval from a training run
    python evaluate.py --exp-dir outputs/smoke_test

    # Quick eval — PSNR + Gaussianity only
    python evaluate.py --exp-dir outputs/smoke_test --skip-rfid --skip-linear-probe

    # Override data dir and batch size
    python evaluate.py --exp-dir outputs/smoke_test --data-dir /path/to/imagenet/val --batch-size 64

    # Just regenerate plots
    python evaluate.py --exp-dir outputs/smoke_test --plots-only
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional

import torch
from omegaconf import OmegaConf

from omnitok.utils.logger import OmniTokLogger
from omnitok.utils.plots import PlotGenerator
from omnitok.utils.artifacts import ArtifactManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniTok Tokenizer Evaluation")
    parser.add_argument(
        "--exp-dir", type=str, required=True,
        help="Experiment output dir (contains config.yaml + checkpoints/)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to specific checkpoint .pt (default: latest in exp-dir/checkpoints/)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Validation data directory (default: uses data.root from training config)"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Results output dir (default: exp-dir/eval)")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--n-batches", type=int, default=None, help="Limit batches (None = full dataset)")
    parser.add_argument("--max-images", type=int, default=5000, help="Max images for rFID")
    parser.add_argument("--skip-rfid", action="store_true", help="Skip rFID (requires cleanfid, slow)")
    parser.add_argument("--skip-linear-probe", action="store_true", help="Skip linear probe")
    parser.add_argument("--skip-gaussianity", action="store_true", help="Skip Gaussianity Score")
    parser.add_argument("--no-ema", action="store_true", help="Do not load EMA model, evaluate the main model instead (recommended for runs < 50k steps)")
    parser.add_argument("--n-recon-images", type=int, default=8, help="Images to show in recon grid")
    parser.add_argument("--plots-only", action="store_true", help="Only regenerate plots from saved results")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    return parser.parse_args()


def _find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return the latest checkpoint in ckpt_dir, or None if empty."""
    if not os.path.isdir(ckpt_dir):
        return None
    ckpts = sorted(
        [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
        key=lambda x: int(x.split("-")[-1].split(".")[0]) if "-" in x else 0,
    )
    return os.path.join(ckpt_dir, ckpts[-1]) if ckpts else None


def load_tokenizer(exp_dir: str, ckpt_path: str, device: str, log, args):
    """Load model from experiment directory and checkpoint."""
    # Load config
    config_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    config_dict = OmegaConf.load(config_path)
    log.info(f"Loaded config from {config_path}")

    from omnitok.models.tokenizer import build_tokenizer
    # Build model using registry
    tokenizer = build_tokenizer(OmegaConf.to_container(config_dict, resolve=True))

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "ema_model" in state and not args.no_ema:
        log.info(f"Loaded EMA weights from step {state.get('global_step', 'unknown')}")
        missing, unexpected = tokenizer.load_state_dict(state["ema_model"], strict=False)
        if missing:
            log.warning(f"Missing keys when loading EMA: {missing[:5]}")
    elif "model" in state:
        log.info(f"Loaded MAIN weights from step {state.get('global_step', 'unknown')} (EMA skipped or missing)")
        missing, unexpected = tokenizer.load_state_dict(state["model"], strict=False)
        if missing:
            log.warning(f"Missing keys when loading MAIN model: {missing[:5]}")
    else:
        # Fallback if checkpoint format is different
        log.info("Loading raw state dict")
        tokenizer.load_state_dict(state, strict=False)

    tokenizer.eval().to(device)
    log.success(f"Tokenizer loaded → {device}")
    return tokenizer, config_dict


def save_recon_images(
    tokenizer: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    artifact_mgr: ArtifactManager,
    device: torch.device,
    n_images: int,
    log: OmniTokLogger,
) -> None:
    """Run encode→decode and save a reconstruction comparison grid.

    Pulls the first batch, encodes + decodes, saves side-by-side grid:
    [orig0 | recon0 | orig1 | recon1 | ...]
    """
    tokenizer.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        images = batch[0][:n_images].to(device)
        z = tokenizer.encode(images)
        recon = tokenizer.decode(z)

    # Denormalize from [-1, 1] → [0, 1] for visualization
    imgs_vis = (images.cpu() * 0.5 + 0.5).clamp(0, 1)
    recon_vis = (recon.cpu() * 0.5 + 0.5).clamp(0, 1)

    path = artifact_mgr.save_recon_grid(imgs_vis, recon_vis, step=0, nrow=min(4, n_images))
    log.success(f"Recon grid saved → {path}")


def generate_plots(exp_dir: str, output_dir: str, log: OmniTokLogger) -> None:
    """Generate figures from saved metrics.json and eval_results.json."""
    plots = PlotGenerator(output_dir=os.path.join(output_dir, "figures"))

    # Loss curves from training metrics
    metrics_path = os.path.join(exp_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics_data = json.load(f)
        if "history" in metrics_data and metrics_data["history"]:
            path = plots.plot_loss_curves(metrics_data["history"], title="Training Loss", fmt="png")
            log.info(f"Loss curves → {path}")
        else:
            log.warning("metrics.json has no history — skipping loss curve plot")
    else:
        log.warning(f"No metrics.json in {exp_dir} — skipping loss curves")

    # Ablation bar chart from eval results
    eval_path = os.path.join(output_dir, "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_results = json.load(f)
        # Only plot scalar metrics
        scalar_results = {k: v for k, v in eval_results.items() if isinstance(v, (int, float))}
        if scalar_results and "psnr" in scalar_results:
            path = plots.plot_ablation_bar(
                {"current": scalar_results},
                metric="psnr",
                title="PSNR",
                filename="eval_psnr_bar.png",
            )
            log.info(f"Metrics bar chart → {path}")

    log.success(f"All figures → {os.path.join(output_dir, 'figures')}")


def save_results(results: Dict[str, float], output_dir: str, log: OmniTokLogger) -> None:
    """Save results to JSON and CSV."""
    import csv
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(output_dir, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(results.items()):
            writer.writerow([k, v])

    log.info(f"Results → {json_path}")
    log.info(f"Results → {csv_path}")


def main() -> None:
    args = parse_args()
    exp_dir = os.path.abspath(args.exp_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(exp_dir, "eval"))
    os.makedirs(output_dir, exist_ok=True)

    log = OmniTokLogger(
        name="omnitok-eval",
        rank=0,
        log_dir=os.path.join(output_dir, "logs"),
    )
    log.print_banner()

    # --- Plots only mode ---
    if args.plots_only:
        log.info(f"Regenerating plots from {exp_dir}")
        generate_plots(exp_dir, output_dir, log)
        return

    # --- Device ---
    device = torch.device(
        args.device if (args.device == "cpu" or not torch.cuda.is_available()) else "cuda"
    )
    log.info(f"Device: {device}")

    # --- Find checkpoint ---
    ckpt_path = args.checkpoint or _find_latest_checkpoint(
        os.path.join(exp_dir, "checkpoints")
    )
    if ckpt_path is None:
        log.error(f"No checkpoint found in {exp_dir}/checkpoints/. Use --checkpoint.")
        sys.exit(1)
    log.info(f"Checkpoint: {ckpt_path}")

    # 2. Load model
    try:
        tokenizer, cfg = load_tokenizer(exp_dir, ckpt_path, device, log, args)
    except Exception as e:
        log.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # --- Build val dataloader ---
    from omnitok.data.datasets import ImageFolderDataset
    from torch.utils.data import DataLoader

    data_dir = args.data_dir or cfg.data.get("val_dir", None)
    if not data_dir:
        log.error("No data dir: pass --data-dir or set data.val_dir in training config")
        sys.exit(1)

    image_size = cfg.model.encoder.get("img_size", 256)
    dataset = ImageFolderDataset(root=data_dir, image_size=image_size, split="val")
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    
    # Create train loader for linear probe
    train_dir = data_dir.replace("val", "train") if "val" in data_dir else data_dir
    train_dataset = ImageFolderDataset(root=train_dir, image_size=image_size, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    log.info(f"Val dataset: {len(dataset)} images. Train dataset: {len(train_dataset)} images.")

    # --- Run metrics ---
    from omnitok.evaluation.evaluator import TokenizerEvaluator

    evaluator = TokenizerEvaluator(
        run_rfid=not args.skip_rfid,
        run_psnr=True,
        run_linear_probe=not args.skip_linear_probe,
        run_gaussianity=not args.skip_gaussianity,
        rfid_max_images=args.max_images,
        device=device,
    )

    log.info("Running evaluation metrics...")
    results = evaluator.evaluate(
        model=tokenizer,
        val_loader=val_loader,
        train_loader=train_loader,
        n_batches=args.n_batches,
    )

    log.print_metrics_table(results, step=0, title="Evaluation Results")

    # --- Save reconstruction images ---
    artifact_mgr = ArtifactManager(
        output_dir=os.path.join(output_dir, "figures"),
        dpi=150,
    )
    save_recon_images(tokenizer, val_loader, artifact_mgr, device, args.n_recon_images, log)

    # --- Save metric numbers ---
    save_results(results, output_dir, log)

    # --- Generate plots ---
    generate_plots(exp_dir, output_dir, log)

    log.success(f"Evaluation complete! → {output_dir}")


if __name__ == "__main__":
    main()
