#!/usr/bin/env python
"""OmniTok — Single entry point for all training stages.

Usage:
    # Stage 1: Train tokenizer (default)
    python train.py

    # With experiment override
    python train.py +experiment=T0_vtp_baseline

    # CLI overrides
    python train.py training.max_steps=50000 data.batch_size=64

    # Multi-GPU
    accelerate launch train.py +experiment=T5_multi_dino_siglip
"""

import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from omnitok.utils.logger import OmniTokLogger
from omnitok.utils.wandb_logger import OmniTokWandBLogger
from omnitok.utils.artifacts import ArtifactManager
from omnitok.utils.metrics import MetricsTracker


def _build_tokenizer(cfg: DictConfig, log: OmniTokLogger) -> "Tokenizer":
    """Build Tokenizer model from config.

    Args:
        cfg: Full Hydra config.
        log: OmniTokLogger for structured output.

    Returns:
        Tokenizer model instance.
    """
    from omnitok.models.tokenizer import Tokenizer

    tokenizer = Tokenizer(
        img_size=cfg.encoder.img_size,
        patch_size=cfg.encoder.patch_size,
        embed_dim=cfg.encoder.embed_dim,
        encoder_depth=cfg.encoder.depth,
        encoder_num_heads=cfg.encoder.num_heads,
        decoder_embed_dim=cfg.decoder.embed_dim,
        decoder_depth=cfg.decoder.depth,
        decoder_num_heads=cfg.decoder.num_heads,
        bottleneck_dim=cfg.encoder.get("vit_feature_bottleneck", 32),
        decoder_out_chans=cfg.decoder.get("out_chans", 3),
    )
    from omnitok.training.utils import count_params
    params = count_params(tokenizer)
    log.encoder(f"Built Tokenizer: {params.get('total', 0):,} params")
    log.print_model_summary(params)
    return tokenizer


def _build_teachers(cfg: DictConfig, log: OmniTokLogger):
    """Build multi-teacher wrapper from config (frozen VFMs).

    Args:
        cfg: Full Hydra config.
        log: OmniTokLogger for structured output.

    Returns:
        MultiTeacher instance or None if no teachers configured.
    """
    from omnitok.teachers.multi_teacher import MultiTeacher

    teacher_configs = {}
    if hasattr(cfg, "teachers"):
        for name, t_cfg in cfg.teachers.items():
            if t_cfg is not None:
                teacher_configs[name] = t_cfg

    if not teacher_configs:
        log.warning("No teachers configured — training without alignment")
        return None

    teacher = MultiTeacher(teacher_configs)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    log.teacher(f"Built {len(teacher_configs)} teachers: {list(teacher_configs.keys())}")
    return teacher


def _build_alignment_loss(cfg: DictConfig, log: OmniTokLogger):
    """Build alignment loss from registry.

    Args:
        cfg: Full Hydra config.
        log: OmniTokLogger.

    Returns:
        Alignment loss module.
    """
    from omnitok.registry import ALIGNMENT_REGISTRY

    align_type = cfg.alignment.get("type", "cosine")
    kwargs = OmegaConf.to_container(cfg.alignment, resolve=True)
    kwargs.pop("type", None)
    kwargs.pop("weight", None)
    # Remove non-constructor keys
    kwargs.pop("projector", None)

    loss_fn = ALIGNMENT_REGISTRY.build(align_type, **kwargs)
    log.loss(f"Alignment: {align_type} (weight={cfg.alignment.get('weight', 1.0)})")
    return loss_fn


def _build_losses(cfg: DictConfig, log: OmniTokLogger):
    """Build reconstruction + optional GAN + optional Gaussianity loss.

    Args:
        cfg: Full Hydra config.
        log: OmniTokLogger.

    Returns:
        Tuple of (recon_loss, gan_loss, gaussianity_loss).
    """
    from omnitok.losses.reconstruction import ReconstructionLoss
    from omnitok.losses.gan import GANLoss

    recon_cfg = cfg.losses.reconstruction
    recon_loss = ReconstructionLoss(
        recon_type=recon_cfg.get("recon_type", "l1"),
        recon_weight=recon_cfg.get("recon_weight", 1.0),
        perceptual_weight=recon_cfg.get("perceptual_weight", 1.0),
    )
    log.loss(f"Recon: {recon_cfg.get('recon_type', 'l1')} + LPIPS(w={recon_cfg.get('perceptual_weight', 1.0)})")

    gan_loss = None
    if cfg.losses.gan.get("enabled", False):
        gan_loss = GANLoss(
            n_layers=cfg.losses.gan.get("n_layers", 3),
            disc_start=cfg.losses.gan.get("disc_start", 50000),
            disc_weight=cfg.losses.gan.get("disc_weight", 0.5),
            lecam_weight=cfg.losses.gan.get("lecam_weight", 0.01),
        )
        log.gan(f"GAN enabled (start={cfg.losses.gan.disc_start}, weight={cfg.losses.gan.disc_weight})")

    gaussianity_loss = None
    gauss_cfg = cfg.losses.get("gaussianity", {})
    if gauss_cfg and gauss_cfg.get("enabled", False):
        from omnitok.losses.gaussianity import GaussianityLoss
        gaussianity_loss = GaussianityLoss(
            weight=gauss_cfg.get("weight", 1e-4),
            mean_penalty=gauss_cfg.get("mean_penalty", True),
        )
        log.loss(f"GaussianityLoss enabled (weight={gauss_cfg.get('weight', 1e-4)})")

    return recon_loss, gan_loss, gaussianity_loss


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module, log: OmniTokLogger):
    """Build optimizer from config.

    Args:
        cfg: Full Hydra config.
        model: Model to optimize.
        log: OmniTokLogger.

    Returns:
        Optimizer instance.
    """
    opt_cfg = cfg.optimizer
    opt_type = opt_cfg.get("type", "adamw")
    params = [p for p in model.parameters() if p.requires_grad]

    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.get("weight_decay", 0.05),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        )
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(params, lr=opt_cfg.lr)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    log.info(f"Optimizer: {opt_type} (lr={opt_cfg.lr}, wd={opt_cfg.get('weight_decay', 0)})")
    return optimizer


def _build_dataloader(cfg: DictConfig, log: OmniTokLogger):
    """Build training DataLoader from config.

    Args:
        cfg: Full Hydra config.
        log: OmniTokLogger.

    Returns:
        DataLoader instance.
    """
    from omnitok.data.datasets import ImageFolderDataset
    from omnitok.data.transforms import build_train_transform
    from torch.utils.data import DataLoader

    transform = build_train_transform(cfg.data.image_size)
    dataset = ImageFolderDataset(root=cfg.data.root, transform=transform)

    log.info(f"Dataset: {len(dataset)} images from {cfg.data.root}")
    log.info(f"Batch: {cfg.data.batch_size} × {cfg.data.get('num_workers', 8)} workers")

    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.get("num_workers", 8),
        pin_memory=cfg.data.get("pin_memory", True),
        drop_last=True,
    )


def _setup_research_infra(cfg: DictConfig, log: OmniTokLogger):
    """Setup research infrastructure: WandB, ArtifactManager, PlotGenerator.

    Args:
        cfg: Full Hydra config.
        log: OmniTokLogger.

    Returns:
        Tuple of (wandb_logger, artifact_manager).
    """
    output_dir = cfg.training.get("output_dir", "outputs/tokenizer")
    exp_name = cfg.get("exp_name", "omnitok")

    # WandB
    wandb_logger = OmniTokWandBLogger(
        project="omnitok",
        name=exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["stage1", "tokenizer"],
        enabled=cfg.training.get("use_wandb", True),
    )
    if wandb_logger.run_url:
        log.info(f"WandB: {wandb_logger.run_url}")

    # Artifact Manager
    artifact_manager = ArtifactManager(
        output_dir=os.path.join(output_dir, "artifacts"),
        dpi=150,
    )
    log.info(f"Artifacts → {os.path.join(output_dir, 'artifacts')}")

    return wandb_logger, artifact_manager


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Hydra manages config composition and CLI overrides.
    Use +experiment=T0_vtp_baseline to load full experiment presets.
    """
    # Initialize OmniTokLogger — the ONLY logging interface
    exp_name = cfg.get("exp_name", "omnitok")
    output_dir = cfg.training.get("output_dir", "outputs/tokenizer")
    log_dir = os.path.join(output_dir, "logs")

    log = OmniTokLogger(
        name=exp_name,
        rank=int(os.environ.get("LOCAL_RANK", 0)),
        log_dir=log_dir,
        verbose=cfg.training.get("verbose", False),
    )

    # Banner + config summary
    log.print_banner()
    log.print_config_table(cfg, title=f"Experiment: {exp_name}")
    log.info(f"Resolved config saved to {output_dir}/config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Build all components with structured logging
    log.info("Building components...", phase="setup")

    tokenizer = _build_tokenizer(cfg, log)
    teachers = _build_teachers(cfg, log)
    alignment_loss = _build_alignment_loss(cfg, log) if teachers is not None else None
    recon_loss, gan_loss, gaussianity_loss = _build_losses(cfg, log)
    optimizer = _build_optimizer(cfg, tokenizer, log)

    disc_optimizer = None
    if gan_loss is not None:
        disc_optimizer = torch.optim.Adam(
            gan_loss.discriminator.parameters(),
            lr=cfg.optimizer.lr,
            betas=(0.5, 0.999),
        )

    train_dataloader = _build_dataloader(cfg, log)

    # Setup research infrastructure
    wandb_logger, artifact_manager = _setup_research_infra(cfg, log)

    # Training config dict for trainer
    train_config = OmegaConf.to_container(cfg.training, resolve=True)
    train_config["alignment_weight"] = cfg.alignment.get("weight", 1.0)
    train_config["use_wandb"] = cfg.training.get("use_wandb", True)
    train_config["exp_name"] = exp_name
    train_config["seed"] = cfg.training.get("seed", 42)

    # Build trainer
    from omnitok.training.trainer import TokenizerTrainer

    trainer = TokenizerTrainer(
        tokenizer=tokenizer,
        teachers=teachers,
        alignment_loss=alignment_loss,
        recon_loss=recon_loss,
        gan_loss=gan_loss,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        disc_optimizer=disc_optimizer,
        config=train_config,
    )

    # Attach research infra to trainer
    trainer.wandb_logger = wandb_logger
    trainer.artifact_manager = artifact_manager

    log.success("All components built successfully")

    # Resume if checkpoint exists
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if os.path.isdir(ckpt_dir):
        ckpts = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")],
            key=lambda x: int(x.split("_")[-1].split(".")[0]) if "_" in x else 0,
        )
        if ckpts:
            latest = os.path.join(ckpt_dir, ckpts[-1])
            log.info(f"Resuming from {latest}", phase="checkpoint")
            trainer.resume(latest)

    # Train
    log.info(f"Starting training: {cfg.training.max_steps} steps", phase="train")
    trainer.train()

    # Finalize
    wandb_logger.finish()
    log.success(f"Training complete! Results → {output_dir}")


if __name__ == "__main__":
    main()
