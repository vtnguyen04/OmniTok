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

import inspect
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from omnitok.utils.artifacts import ArtifactManager

# Import components to trigger registry decorators
from omnitok.utils.logger import OmniTokLogger
from omnitok.utils.wandb_logger import OmniTokWandBLogger


def _build_tokenizer(cfg: DictConfig, log: OmniTokLogger) -> torch.nn.Module:
    """Build Tokenizer model from config using modular registry.

    This replaces the hardcoded build logic with a flexible factory.
    """
    from omnitok.models.tokenizer import build_tokenizer
    from omnitok.training.utils import count_params

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    tokenizer = build_tokenizer(config_dict)

    # Handle pretrained weights for the encoder
    pretrained = cfg.model.encoder.get("pretrained", None)
    if pretrained:
        log.encoder(f"Loading pretrained encoder weights: {pretrained}")
        if hasattr(tokenizer.encoder, "load_pretrained_dinov2"):
            model_name = None if isinstance(pretrained, bool) else pretrained
            tokenizer.encoder.load_pretrained_dinov2(model_name)
            log.success(f"Pretrained weights loaded: {pretrained}")
        else:
            log.warning(f"Encoder doesn't support load_pretrained_dinov2. Ignoring pretrained={pretrained}")

    params = count_params(tokenizer)
    log.encoder(f"Built Modular Tokenizer: {params.get('total', 0):,} params")
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
    from omnitok.registry import TEACHER_REGISTRY
    from omnitok.teachers.multi_teacher import MultiTeacher

    teacher_configs = {}
    teacher_node = getattr(cfg, "teacher", getattr(cfg, "teachers", None))

    if teacher_node is not None and teacher_node.get("enabled", True):
        names = teacher_node.get("names", [])
        if not names and "dinov2" in teacher_node:
            names = ["dinov2"] # fallback
        for name in names:
            if name in teacher_node:
                teacher_configs[name] = teacher_node[name]

    if not teacher_configs:
        log.warning("No teachers configured — training without alignment")
        return None

    instantiated_teachers = {}
    for name, t_cfg in teacher_configs.items():
        kwargs = {k: v for k, v in t_cfg.items() if k != "type"}
        teacher_type = t_cfg.get("type", name)

        if teacher_type in TEACHER_REGISTRY:
            # Filter kwargs to only those accepted by the teacher's constructor
            teacher_cls = TEACHER_REGISTRY._registry[teacher_type]
            sig = inspect.signature(teacher_cls.__init__)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

            instantiated_teachers[name] = TEACHER_REGISTRY.build(teacher_type, **valid_kwargs)
        else:
            log.warning(f"Teacher {teacher_type} not found in registry. Skipping.")

    if not instantiated_teachers:
        log.warning("No teachers could be instantiated — training without alignment")
        return None

    teacher = MultiTeacher(instantiated_teachers)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    log.teacher(f"Built {len(instantiated_teachers)} teachers: {list(instantiated_teachers.keys())}")
    from omnitok.training.utils import count_params
    params = count_params(teacher)
    log.teacher(f"Teacher Params (Frozen): {params.get('total_M', 0):.2f}M")
    return teacher

def _build_alignment_loss(cfg: DictConfig, tokenizer, teachers, log: OmniTokLogger):
    """Build alignment loss from registry.

    Args:
        cfg: Full Hydra config.
        log: OmniTokLogger.

    Returns:
        Alignment loss module.
    """
    from omnitok.registry import ALIGNMENT_REGISTRY

    align_type = cfg.alignment.get("type", "cosine")
    align_cfg = cfg.alignment
    align_type = align_cfg.get("type", "cosine")

    # Explicit "none" — no alignment loss
    if align_type in (None, "none", ""):
        log.info("Alignment: disabled (type=none)")
        return None

    kwargs = OmegaConf.to_container(align_cfg, resolve=True)
    kwargs.pop("type", None)
    kwargs.pop("weight", None)
    kwargs.pop("adaptive_weighting", None)

    # Auto-inject dimensions if not manually overridden.
    # Alignment uses pre-bottleneck features (REPA-E style) → student_dim = original embed_dim,
    # NOT the bottleneck dim, so the projector maps 768→teacher_dim instead of 32→teacher_dim.
    if "student_dim" not in kwargs and tokenizer is not None:
        kwargs["student_dim"] = tokenizer.encoder.original_embed_dim \
            if hasattr(tokenizer.encoder, "original_embed_dim") \
            else tokenizer.encoder.embed_dim

    if "teacher_dim" not in kwargs and teachers is not None:
        if hasattr(teachers, "feature_dim"):
            kwargs["teacher_dim"] = teachers.feature_dim

    loss_fn = ALIGNMENT_REGISTRY.build(align_type, **kwargs)
    log.loss(f"Alignment: {align_type} (weight={cfg.alignment.get('weight', 1.0)})")
    
    from omnitok.training.utils import count_params
    params = count_params(loss_fn)
    if params.get('total', 0) > 0:
        log.loss(f"Alignment Projector Params: {params.get('total_M', 0):.2f}M")
        
    return loss_fn

def _build_losses(cfg: DictConfig, log: OmniTokLogger):
    """Build reconstruction + optional GAN + optional Gaussianity + optional LatentNorm loss.

    Args:
        cfg: Full Hydra config.
        log: OmniTokLogger.

    Returns:
        Tuple of (recon_loss, gan_loss, gaussianity_loss, latent_norm_loss).
    """
    from omnitok.losses.gan import GANLoss
    from omnitok.losses.reconstruction import ReconstructionLoss

    recon_loss = None
    if "loss" in cfg and "reconstruction" in cfg.loss:
        recon_cfg = cfg.loss.reconstruction
        if recon_cfg.get("recon_weight", 0.0) > 0.0 or recon_cfg.get("perceptual_weight", 0.0) > 0.0:
            recon_loss = ReconstructionLoss(
                recon_type=recon_cfg.get("recon_type", "l1"),
                recon_weight=recon_cfg.get("recon_weight", 1.0),
                perceptual_weight=recon_cfg.get("perceptual_weight", 1.0),
            )
            log.loss(f"Recon: {recon_cfg.get('recon_type', 'l1')} + LPIPS(w={recon_cfg.get('perceptual_weight', 1.0)})")

    gan_loss = None
    if "loss" in cfg and "gan" in cfg.loss:
        gan_cfg = cfg.loss.gan
        if gan_cfg.get("enabled", False):
            gan_loss = GANLoss(
                n_layers=gan_cfg.get("n_layers", 3),
                disc_start=gan_cfg.get("disc_start", 50000),
                disc_weight=gan_cfg.get("disc_weight", 0.5),
                lecam_weight=gan_cfg.get("lecam_weight", 0.01),
            )
            log.gan(f"GAN enabled (start={gan_cfg.get('disc_start', 50000)}, weight={gan_cfg.get('disc_weight', 0.5)})")
            
            from omnitok.training.utils import count_params
            params = count_params(gan_loss.discriminator)
            log.gan(f"Discriminator Params: {params.get('total_M', 0):.2f}M")

    gaussianity_loss = None
    if "loss" in cfg and "gaussianity" in cfg.loss:
        gauss_cfg = cfg.loss.gaussianity
        if gauss_cfg.get("enabled", False) and gauss_cfg.get("weight", 0.0) > 0.0:
            from omnitok.losses.gaussianity import GaussianityLoss
            gaussianity_loss = GaussianityLoss(
                weight=gauss_cfg.get("weight", 1e-4),
                mean_penalty=gauss_cfg.get("mean_penalty", True),
            )
            log.loss(f"GaussianityLoss enabled (weight={gauss_cfg.get('weight', 1e-4)})")

    latent_norm_loss = None
    if "loss" in cfg and "latent_norm" in cfg.loss:
        ln_cfg = cfg.loss.latent_norm
        if ln_cfg.get("enabled", False) and ln_cfg.get("weight", 0.0) > 0.0:
            from omnitok.losses.latent_norm import LatentNormLoss
            latent_norm_loss = LatentNormLoss(weight=ln_cfg.get("weight", 0.01))
            log.loss(f"LatentNormLoss enabled (weight={ln_cfg.get('weight', 0.01)})")

    return recon_loss, gan_loss, gaussianity_loss, latent_norm_loss


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module, log: OmniTokLogger, extra_modules: dict = None):
    """Build optimizer from config.

    Args:
        cfg: Full Hydra config.
        model: Model to optimize.
        log: OmniTokLogger.
        extra_modules: Dictionary of {module_prefix: module} (like alignment loss projector) to include in optimization.

    Returns:
        Optimizer instance.
    """
    opt_cfg = cfg.optimizer
    opt_type = opt_cfg.get("type", "adamw")
    base_lr = opt_cfg.lr

    # Retrieve lr_multipliers from config (or convert backbone_lr_multiplier for backward compatibility)
    lr_multipliers = opt_cfg.get("lr_multipliers", {})
    if "backbone_lr_multiplier" in opt_cfg and not lr_multipliers:
        lr_multipliers = {"encoder": opt_cfg.backbone_lr_multiplier, "encoder.feature_bottleneck": 1.0}

    # Delegate parameter grouping to the model (SOLID: Open/Closed Principle)
    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(base_lr=base_lr, lr_multipliers=lr_multipliers)
    else:
        # Fallback for models without specific lr multiplier needs
        params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": params, "lr": base_lr}]

    # Inject extra module parameters (e.g., Alignment Projector)
    if extra_modules:
        for prefix_key, extra_module in extra_modules.items():
            if extra_module is not None:
                # Same prefix matching logic for extra modules, with prepended module name
                for name, p in extra_module.named_parameters():
                    if not p.requires_grad:
                        continue

                    full_name = f"{prefix_key}.{name}"
                    best_match = ""
                    mult = 1.0
                    for prefix, m in lr_multipliers.items():
                        if full_name.startswith(prefix) and len(prefix) > len(best_match):
                            best_match = prefix
                            mult = m

                    lr = base_lr * mult

                    # Find existing group with this lr, or create new
                    found_group = False
                    for group in param_groups:
                        if group["lr"] == lr:
                            group["params"].append(p)
                            found_group = True
                            break

                    if not found_group:
                        param_groups.append({"params": [p], "lr": lr})

    if opt_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=opt_cfg.get("weight_decay", 0.05),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        )
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(param_groups)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    log.info(f"Optimizer: {opt_type} (lr={opt_cfg.lr}, multipliers={lr_multipliers}, wd={opt_cfg.get('weight_decay', 0.0)})")

    return optimizer


def _build_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer, log: OmniTokLogger):
    """Build learning rate scheduler with warmup."""
    sched_cfg = cfg.get("scheduler", {})
    if not sched_cfg:
        return None

    sched_type = sched_cfg.get("type", "cosine")
    warmup_steps = sched_cfg.get("warmup_steps", 5000)
    max_steps = cfg.training.get("max_steps", 200000)

    if sched_type == "cosine":
        import math

        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)
        log.info(f"Scheduler: {sched_type} (warmup={warmup_steps}, max_steps={max_steps})")
        return scheduler

    return None


def _build_dataloader(cfg: DictConfig, log: OmniTokLogger):
    """Build training DataLoader from config.

    Args:
        cfg: Full Hydra config.
        log: OmniTokLogger.

    Returns:
        DataLoader instance.
    """
    from torch.utils.data import DataLoader

    from omnitok.data.datasets import ImageFolderDataset

    # Merge training.data overrides into cfg.data (e.g., mini.yaml sets training.data.batch_size=2)
    if hasattr(cfg, "training") and hasattr(cfg.training, "data"):
        training_data = OmegaConf.to_container(cfg.training.data, resolve=True)
        for k, v in training_data.items():
            OmegaConf.update(cfg, f"data.{k}", v)

    batch_size = cfg.data.batch_size
    num_workers = cfg.data.get("num_workers", 8)

    # Get data directory from CLI args or config
    import sys
    data_dir = cfg.data.get("train_dir", None)
    if not data_dir:
        log.error("No data dir: set data.train_dir in config")
        sys.exit(1)

    dataset = ImageFolderDataset(root=data_dir, image_size=cfg.data.image_size, split="train")

    if cfg.data.get("val_dir", None):
        val_dir = cfg.data.get("val_dir")
        val_dataset = ImageFolderDataset(root=val_dir, image_size=cfg.data.image_size, split="val")
    else:
        val_dataset = None

    log.info(f"Dataset: {len(dataset)} images from {data_dir}")
    log.info(f"Batch: {batch_size} × {num_workers} workers")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg.data.get("pin_memory", True),
        drop_last=True,
    )


def _setup_research_infra(cfg: DictConfig, log: OmniTokLogger):
    """Setup research infrastructure: WandB, ArtifactManager, PlotGenerator.

    Args:
        cfg: Full Hydra config.
        log: OmniTokLogger.

    Returns:
        Tuple of (wandb_logger, artifact_manager, plot_generator).
    """
    output_dir = cfg.training.get("output_dir", f"outputs/{cfg.get('exp_name', 'omnitok')}")
    exp_name = cfg.get("exp_name", "omnitok")

    # WandB
    wandb_logger = OmniTokWandBLogger(
        project="omnitok",
        name=exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.training.get("tags", ["omnitok"]),
        enabled=cfg.training.get("use_wandb", True),
        run_id=cfg.training.get("run_id", cfg.get("exp_name")),
    )
    if wandb_logger.run_url:
        log.info(f"WandB: {wandb_logger.run_url}")

    # Artifact Manager
    artifact_manager = ArtifactManager(
        output_dir=os.path.join(output_dir, "artifacts"),
        dpi=150,
    )
    log.info(f"Artifacts → {os.path.join(output_dir, 'artifacts')}")

    # Plot Generator
    from omnitok.utils.plots import PlotGenerator
    plot_generator = PlotGenerator(output_dir=os.path.join(output_dir, "plots"))
    log.info(f"Plots → {os.path.join(output_dir, 'plots')}")

    return wandb_logger, artifact_manager, plot_generator


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Hydra manages config composition and CLI overrides.
    Use +experiment=T0_vtp_baseline to load full experiment presets.
    """
    # Load configs
    from omegaconf import OmegaConf

    # 1. Initialize research infrastructure (robust exp_name resolution)
    # Priority: CLI override > Experiment config > Training config > Default
    exp_name = cfg.get("exp_name")
    if not exp_name and cfg.get("experiment") and cfg.experiment.get("exp_name"):
        exp_name = cfg.experiment.exp_name
    if not exp_name and cfg.training.get("exp_name"):
        exp_name = cfg.training.exp_name
    if not exp_name:
        exp_name = "omnitok_run"

    output_dir = cfg.training.get("output_dir")
    if not output_dir:
        output_dir = f"outputs/{exp_name}"
    log_dir = os.path.join(output_dir, "logs")

    log = OmniTokLogger(
        name=exp_name,
        rank=int(os.environ.get("LOCAL_RANK", 0)),
        log_dir=log_dir,
        verbose=cfg.training.get("verbose", False),
    )

    # 2. Merge experiment config if present
    if cfg.get("experiment") is not None:
        log.info(f"Merging experiment config: {cfg.experiment.get('exp_name', 'unknown')}")
        experiment_cfg = cfg.experiment

        # Capture CLI overrides to re-apply them last
        import sys
        cli_cfg = OmegaConf.from_cli(sys.argv[1:])

        OmegaConf.set_struct(cfg, False)
        # Priority: Defaults < Experiment < CLI
        cfg = OmegaConf.merge(cfg, experiment_cfg, cli_cfg)
        OmegaConf.set_struct(cfg, True)

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
    alignment_loss = _build_alignment_loss(cfg, tokenizer, teachers, log) if teachers is not None else None
    recon_loss, gan_loss, gaussianity_loss, latent_norm_loss = _build_losses(cfg, log)

    extra_modules = {
        "alignment_loss": alignment_loss,
        "gaussianity_loss": gaussianity_loss,
        "latent_norm_loss": latent_norm_loss,
    }
    optimizer = _build_optimizer(cfg, tokenizer, log, extra_modules=extra_modules)
    scheduler = _build_scheduler(cfg, optimizer, log)

    disc_optimizer = None
    disc_scheduler = None
    if gan_loss is not None:
        disc_optimizer = torch.optim.Adam(
            gan_loss.discriminator.parameters(),
            lr=cfg.optimizer.lr,
            betas=(0.5, 0.999),
        )
        disc_scheduler = _build_scheduler(cfg, disc_optimizer, log)

    train_dataloader = _build_dataloader(cfg, log)

    # Setup research infrastructure
    wandb_logger, artifact_manager, plot_generator = _setup_research_infra(cfg, log)

    # Training config dict for trainer
    train_config = OmegaConf.to_container(cfg.training, resolve=True)
    train_config["alignment_weight"] = cfg.alignment.get("weight", 1.0)
    train_config["use_adaptive_weighting"] = cfg.alignment.get("adaptive_weighting", False)
    train_config["use_wandb"] = cfg.training.get("use_wandb", True)
    train_config["exp_name"] = exp_name
    train_config["output_dir"] = output_dir
    train_config["log_dir"] = log_dir
    train_config["seed"] = cfg.training.get("seed", 42)

    # Build trainer
    from omnitok.training.trainer import TokenizerTrainer

    trainer = TokenizerTrainer(
        tokenizer=tokenizer,
        teachers=teachers,
        alignment_loss=alignment_loss,
        recon_loss=recon_loss,
        gan_loss=gan_loss,
        gaussianity_loss=gaussianity_loss,
        latent_norm_loss=latent_norm_loss,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        disc_optimizer=disc_optimizer,
        disc_scheduler=disc_scheduler,
        config=train_config,
    )

    # Attach research infra to trainer
    trainer.wandb_logger = wandb_logger
    trainer.artifact_manager = artifact_manager
    trainer.plot_generator = plot_generator

    log.success("All components built successfully")

    # Resume logic
    resume_from = cfg.training.get("resume_from", None)
    if resume_from and os.path.isfile(resume_from):
        log.info(f"Resuming from specified checkpoint: {resume_from}", phase="checkpoint")
        trainer.resume(resume_from)
    else:
        last_ckpt = os.path.join(output_dir, "checkpoints", "last.pt")
        if os.path.isfile(last_ckpt):
            log.info(f"Resuming from {last_ckpt}", phase="checkpoint")
            trainer.resume(last_ckpt)

    # Spawn background system monitor on rank 0
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        import subprocess
        import atexit
        log.info("Spawning background system monitor (auto-managed)...", phase="setup")
        import sys
        cmd = [sys.executable, "omnitok/utils/system_monitor.py"]
        if wandb_logger and wandb_logger._run is not None:
            cmd.extend(["--run_id", wandb_logger._run.id])
            
        monitor_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        def cleanup_monitor():
            if monitor_process.poll() is None:
                monitor_process.terminate()
        atexit.register(cleanup_monitor)

    # Train
    log.info(f"Starting training: {cfg.training.max_steps} steps", phase="train")
    trainer.train()

    # Finalize
    wandb_logger.finish()
    log.success(f"Training complete! Results → {output_dir}")


if __name__ == "__main__":
    main()
