#!/usr/bin/env python
"""Compute Gaussianity Score for a trained tokenizer.

Standalone tool for UNE hypothesis validation.
Measures how Gaussian each latent dimension is using Anderson-Darling test.

Usage:
    python tools/compute_gaussianity.py \\
        --checkpoint outputs/T5/checkpoints/ckpt_200000.pt \\
        --data-dir /path/to/imagenet/val \\
        --output results/gaussianity/T5.json

    # Compare multiple checkpoints
    python tools/compute_gaussianity.py \\
        --checkpoint outputs/T0/ckpt.pt outputs/T5/ckpt.pt outputs/T9/ckpt.pt \\
        --data-dir /path/to/imagenet/val \\
        --output results/gaussianity_comparison.json
"""

import argparse
import json
import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnitok.evaluation.gaussianity import GaussianityEvaluator
from omnitok.utils.logger import OmniTokLogger
from omnitok.utils.plots import PlotGenerator


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="OmniTok Gaussianity Score Tool")
    parser.add_argument("--checkpoint", type=str, nargs="+", required=True, help="Checkpoint path(s)")
    parser.add_argument("--data-dir", type=str, required=True, help="ImageNet val directory")
    parser.add_argument("--output", type=str, default="results/gaussianity.json", help="Output JSON path")
    parser.add_argument("--significance", type=float, default=5.0, help="Significance level (15, 10, 5, 2.5, 1)")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max latent samples")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--n-batches", type=int, default=None, help="Max batches (None=all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--plot", action="store_true", help="Generate histogram plots")
    return parser.parse_args()


def main() -> None:
    """Compute Gaussianity Score."""
    args = parse_args()

    log = OmniTokLogger(name="gaussianity", rank=0)
    log.print_banner()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    evaluator = GaussianityEvaluator(
        significance=args.significance,
        max_samples=args.max_samples,
    )

    # Build val dataloader
    from omnitok.data.datasets import ImageFolderDataset
    from omnitok.data.transforms import build_eval_transform
    from torch.utils.data import DataLoader

    transform = build_eval_transform(256)
    dataset = ImageFolderDataset(root=args.data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    results = {}

    for ckpt_path in args.checkpoint:
        name = os.path.basename(os.path.dirname(ckpt_path))
        log.info(f"Processing: {name} ({ckpt_path})")

        # Load tokenizer
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

        if "ema_model" in state:
            tokenizer.load_state_dict(state["ema_model"])
        else:
            tokenizer.load_state_dict(state["model"])

        tokenizer.eval().to(device)

        # Compute Gaussianity Score
        result = evaluator.compute_from_model(
            model=tokenizer,
            dataloader=loader,
            device=device,
            n_batches=args.n_batches,
        )

        results[name] = result
        log.metric(
            f"{name}: score={result['gaussianity_score']:.4f} "
            f"({result['n_gaussian_dims']}/{result['total_dims']} dims)"
        )

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results → {args.output}")

    # Print comparison table
    log.print_metrics_table(
        {k: v["gaussianity_score"] for k, v in results.items()},
        step=0,
        title="Gaussianity Score Comparison",
    )

    # Optional: generate plots
    if args.plot:
        plot_dir = os.path.join(os.path.dirname(args.output), "figures")
        plots = PlotGenerator(output_dir=plot_dir)
        plots.plot_ablation_bar(
            results={k: {"Gaussianity": v["gaussianity_score"]} for k, v in results.items()},
            metric="Gaussianity",
            title="Gaussianity Score by Experiment",
            lower_is_better=False,
            fmt="pdf",
        )
        log.info(f"Plots → {plot_dir}")

    log.success("Gaussianity analysis complete!")


if __name__ == "__main__":
    main()
