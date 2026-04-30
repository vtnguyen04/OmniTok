#!/usr/bin/env python
"""OmniTok — Evaluation entry point.

Runs all evaluation metrics on a trained tokenizer checkpoint:
- rFID: Reconstruction FID
- PSNR: Pixel-level fidelity
- Linear Probe: Semantic quality
- Gaussianity Score: UNE hypothesis validation

Results are saved to JSON/CSV and optionally plotted.

Usage:
    # Full evaluation
    python evaluate.py --checkpoint outputs/tokenizer/checkpoints/ckpt_100000.pt

    # Quick eval (PSNR + Gaussianity only)
    python evaluate.py --checkpoint ckpt.pt --skip-rfid --skip-linear-probe

    # Generate plots from existing results
    python evaluate.py --results-dir outputs/tokenizer --plots-only
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional

import torch

from omnitok.utils.logger import OmniTokLogger
from omnitok.utils.plots import PlotGenerator
from omnitok.utils.artifacts import ArtifactManager
from omnitok.utils.experiment import ExperimentManager


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="OmniTok Tokenizer Evaluation")
    parser.add_argument("--checkpoint", type=str, required=False, help="Path to tokenizer checkpoint (.pt)")
    parser.add_argument("--data-dir", type=str, default="/path/to/imagenet/val", help="Validation data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/eval", help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--max-images", type=int, default=50000, help="Max images for rFID")
    parser.add_argument("--skip-rfid", action="store_true", help="Skip rFID (slow)")
    parser.add_argument("--skip-linear-probe", action="store_true", help="Skip linear probe")
    parser.add_argument("--skip-gaussianity", action="store_true", help="Skip Gaussianity Score")
    parser.add_argument("--plots-only", action="store_true", help="Only generate plots from existing results")
    parser.add_argument("--results-dir", type=str, default=None, help="Load results from this dir (for --plots-only)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    return parser.parse_args()


def load_tokenizer(ckpt_path: str, device: torch.device, log: OmniTokLogger):
    """Load tokenizer from checkpoint.

    Args:
        ckpt_path: Path to checkpoint .pt file.
        device: Target device.
        log: OmniTokLogger.

    Returns:
        Loaded tokenizer model in eval mode.
    """
    from omnitok.training.utils import load_checkpoint

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Reconstruct tokenizer from checkpoint config
    from omnitok.models.tokenizer import Tokenizer
    cfg = state.get("config", {})

    tokenizer = Tokenizer(
        img_size=cfg.get("img_size", 256),
        patch_size=cfg.get("patch_size", 16),
        embed_dim=cfg.get("embed_dim", 768),
        encoder_depth=cfg.get("encoder_depth", 12),
        encoder_num_heads=cfg.get("encoder_num_heads", 12),
        decoder_embed_dim=cfg.get("decoder_embed_dim", 768),
        decoder_depth=cfg.get("decoder_depth", 12),
        decoder_num_heads=cfg.get("decoder_num_heads", 12),
        bottleneck_dim=cfg.get("bottleneck_dim", 32),
    )

    # Load EMA weights if available, else model weights
    if "ema_model" in state:
        tokenizer.load_state_dict(state["ema_model"])
        log.info("Loaded EMA model weights")
    elif "model" in state:
        tokenizer.load_state_dict(state["model"])
        log.info("Loaded model weights")
    else:
        raise RuntimeError(f"No model weights found in {ckpt_path}")

    tokenizer.eval().to(device)
    log.success(f"Loaded tokenizer from {ckpt_path}")
    return tokenizer


def run_evaluation(
    tokenizer: torch.nn.Module,
    args: argparse.Namespace,
    log: OmniTokLogger,
) -> Dict[str, float]:
    """Run all evaluation metrics.

    Args:
        tokenizer: Loaded tokenizer model.
        args: CLI arguments.
        log: OmniTokLogger.

    Returns:
        Dict of all metric results.
    """
    from omnitok.evaluation.evaluator import TokenizerEvaluator

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    evaluator = TokenizerEvaluator(
        run_rfid=not args.skip_rfid,
        run_psnr=True,
        run_linear_probe=not args.skip_linear_probe,
        run_gaussianity=not args.skip_gaussianity,
        rfid_max_images=args.max_images,
        device=device,
    )

    # Build val dataloader
    from omnitok.data.datasets import ImageFolderDataset
    from omnitok.data.transforms import build_eval_transform
    from torch.utils.data import DataLoader

    transform = build_eval_transform(256)
    dataset = ImageFolderDataset(root=args.data_dir, transform=transform)
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    log.info(f"Evaluating on {len(dataset)} images from {args.data_dir}")

    # Run evaluation
    results = evaluator.evaluate(
        model=tokenizer,
        dataloader=val_loader,
    )

    # Print results table
    log.print_metrics_table(results, step=0, title="Evaluation Results")

    return results


def save_results(results: Dict[str, float], output_dir: str, log: OmniTokLogger) -> None:
    """Save evaluation results to JSON and CSV.

    Args:
        results: Metric results dict.
        output_dir: Output directory.
        log: OmniTokLogger.
    """
    import csv

    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results JSON → {json_path}")

    # CSV (for pgfplots / pandas)
    csv_path = os.path.join(output_dir, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(results.items()):
            writer.writerow([k, v])
    log.info(f"Results CSV → {csv_path}")


def generate_plots(output_dir: str, log: OmniTokLogger) -> None:
    """Generate publication-quality plots from saved results.

    Args:
        output_dir: Directory containing eval_results.json / metrics.json.
        log: OmniTokLogger.
    """
    plots = PlotGenerator(output_dir=os.path.join(output_dir, "figures"))

    # If metrics.json exists (from training), plot loss curves
    metrics_path = os.path.join(output_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics_data = json.load(f)

        if "history" in metrics_data:
            plots.plot_loss_curves(metrics_data["history"], fmt="pdf")
            log.info("Generated loss curves (PDF)")

    log.success(f"Figures → {os.path.join(output_dir, 'figures')}")


def main() -> None:
    """Evaluation entry point."""
    args = parse_args()

    log = OmniTokLogger(
        name="omnitok-eval",
        rank=0,
        log_dir=os.path.join(args.output_dir, "logs"),
    )
    log.print_banner()

    if args.plots_only:
        results_dir = args.results_dir or args.output_dir
        log.info(f"Generating plots from {results_dir}")
        generate_plots(results_dir, log)
        return

    if not args.checkpoint:
        log.error("--checkpoint is required for evaluation")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load model
    tokenizer = load_tokenizer(args.checkpoint, device, log)

    # Run evaluation
    results = run_evaluation(tokenizer, args, log)

    # Save results
    save_results(results, args.output_dir, log)

    # Generate plots
    generate_plots(args.output_dir, log)

    log.success("Evaluation complete!")


if __name__ == "__main__":
    main()
