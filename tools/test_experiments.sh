#!/bin/bash
set -e

DATA_ROOT="$HOME/Tan/AIOTok/data/mini"
CONFIGS=(
    "T0_vtp_baseline"
    "T1_vanilla_vae"
    "T2_frozen_dino_relkd"
    "T3_frozen_dino_cosine"
    "T4_frozen_dino_prediction"
    "T5_multi_dino_siglip"
    "T6_multi_dino_siglip_sam"
    "T7_dino_depth"
    "T8_masking75"
    "T9_gaussianity_reg"
    "T10_linear_projector"
    "T_final"
)

echo "Starting OmniTok Pipeline Verification..."
echo "Data Root: $DATA_ROOT"
echo "----------------------------------------"

for config in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Testing Experiment: $config"
    echo "========================================"
    
    uv run python train.py \
        experiment=$config \
        data.root=$DATA_ROOT \
        data.batch_size=2 \
        data.num_workers=0 \
        training.max_steps=2 \
        training.save_every=2 \
        training.log_every=1 \
        training.use_wandb=False
        
    echo "[SUCCESS] $config passed."
    echo ""
done

echo "All 12 experiments passed successfully!"
