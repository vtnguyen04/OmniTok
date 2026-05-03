#!/bin/bash
# OmniTok — Automated Ablation Training Script
# Runs T1 to T4 sequentially to ensure complete ablation data.
# Run this inside a tmux session: `bash scripts/train_ablations.sh`

set -e # Exit on error

# Ensure we're in the right directory
cd "$(dirname "$0")/.." || exit

# Training configuration
MAX_STEPS=10000          # 10,000 steps is enough to see good structural convergence on mini
BATCH_SIZE=8             # Suitable for single GPU dry runs
DATA_ROOT="/home/truongchau/Tan/AIOTok/data/mini/train"

echo "=========================================================="
echo "🚀 Starting OmniTok Ablation Training Pipeline"
echo "Data Root: $DATA_ROOT"
echo "Steps per exp: $MAX_STEPS"
echo "=========================================================="

# List of experiments to run
EXPERIMENTS=("T1_vanilla_vae" "T2_frozen_dino_relkd" "T3_frozen_dino_cosine" "T4_frozen_dino_prediction")

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "----------------------------------------------------------"
    echo "▶️ Running Experiment: $EXP"
    echo "----------------------------------------------------------"
    
    # Run training
    uv run python train.py \
        experiment=$EXP \
        training.max_steps=$MAX_STEPS \
        data.batch_size=$BATCH_SIZE \
        training.grad_accum_steps=32 \
        training.scheduler.warmup_steps=250 \
        data.root=$DATA_ROOT
        
    echo "✅ Finished Experiment: $EXP"
done

echo "=========================================================="
echo "🎉 All ablation experiments (T1-T4) completed successfully!"
echo "Check the 'outputs/' directory for checkpoints and artifacts."
echo "=========================================================="
