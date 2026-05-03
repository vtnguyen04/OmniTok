# OmniTok

> Where, What, and How to Align: A Unified Study of Vision Foundation Alignment in Visual Tokenizers

OmniTok is a modular, high-performance research framework designed to systematically investigate the alignment of Vision Foundation (VF) models within Visual Tokenizers. By unifying representation learning (DINOv2, SigLIP, SAM, Depth Anything) and generative modeling (LightningDiT, SiT), OmniTok enables seamless experimentation across different tokenizer architectures, alignment strategies, and generative paradigms.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Hydra](https://img.shields.io/badge/Config-Hydra-orange)

## Features

- **Modular Architecture**: Powered by a global `Registry` pattern. Easily swap encoders, decoders, bottlenecks, teachers, and alignment losses without touching the training loop.
- **Multi-Teacher Support**: Native support for distilling from multiple frozen Vision Foundation models simultaneously (e.g., DINOv2 + SigLIP + SAM) using dynamically injected projector heads.
- **Config-Driven Experiments**: Zero hardcoded hyperparameters. Everything is controlled via `Hydra` YAML composition, enabling massive ablations with a single command.
- **End-to-End Pipeline**: Includes Stage 1 (Visual Tokenizer Training) and Stage 2 (Diffusion Transformer Training via LightningDiT/SiT).

## Installation

OmniTok requires Python 3.10+ and PyTorch 2.0+. We recommend using `uv` or `pip` in a dedicated virtual environment.

```bash
# Clone the repository
git clone https://github.com/vtnguyen04/OmniTok.git
cd OmniTok

# Install dependencies using uv
uv sync
```

## Usage

OmniTok uses Hydra for configuration management. All experiment presets are located in `configs/experiment/`.

### 1. Train Tokenizer (Stage 1)
To run a specific tokenizer ablation experiment (e.g., training with frozen DINOv2 using Relational KD alignment):

```bash
bash scripts/train_tokenizer.sh +experiment=alignment/T2_frozen_dino_relkd
```

To run the full suite of Phase 7 ablations automatically:
```bash
bash scripts/run_all_experiments.sh
```

### 2. Train Diffusion Transformer (Stage 2)
*(Under active development)*
Once the tokenizer is trained and latents are extracted, you can train the DiT model:

```bash
bash scripts/train_dit.sh
```

### 3. Evaluation
Evaluate trained models using rFID, Zero-Shot classification, or Linear Probing:
```bash
bash scripts/evaluate.sh
```

## Architecture Map

```text
omnitok/
├── configs/             # Hydra configuration compositions (yaml)
├── omnitok/             # Core library
│   ├── data/            # ImageNet loaders and augmentations
│   ├── evaluation/      # Zero-shot, Linear Probe, rFID, Gaussianity metrics
│   ├── losses/          # Alignment, Reconstruction, KL, GAN, and Discriminator
│   ├── models/          # ViT/CNN Tokenizers, LightningDiT, SiT, Bottlenecks
│   ├── teachers/        # Frozen VF models (DINOv2, SigLIP, SAM, etc.)
│   └── training/        # BaseTrainer and TokenizerTrainer loops
├── scripts/             # Execution wrappers for cluster environments
└── train.py             # Single entry point for all training tasks
```

## Contributing

Contributions are welcome! Please ensure you follow the project's strict GitFlow and development rules outlined in `Rules.md`:
1. Never commit directly to `main` or `dev`. Always create a `feat/*` or `fix/*` branch.
2. Run tests before committing: `pytest tests/ -x --tb=short`.
3. Pass the linter: `ruff check omnitok/ --fix`.
4. Submit pull requests with squash merges.

## Acknowledgments

This framework is built upon and inspired by several foundational works:
- [LightningDiT](https://github.com/Lightning-AI/lightning-dit) & [SiT](https://github.com/willisma/SiT)
- [VTP](https://github.com/vtp-team/vtp)
- [REPA-E](https://github.com/repa-e/repa-e)

## License

This project is licensed under the MIT License.
