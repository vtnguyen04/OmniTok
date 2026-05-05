#!/bin/bash
# Sequential experiment runner for OmniTok
# Runs ablation experiments sequentially with clean outputs.

set -e  # Exit on first error

EXPS=(
    "multi_teacher/T7_linear_projector"
    "baseline/T1_vanilla_vae"
    "ablations/decoder/D1_cnn_decoder"
    "sparse_routing/T_sparse_top1"
    "alignment/T8_maetok_full"
)

# No common overrides — each config uses its own max_steps (20000)
COMMON_ARGS=""

echo "================================================================"
echo "  STARTING OMNITOK NIGHTLY EXPERIMENTS (${#EXPS[@]} CONFIGS)"
echo "  $(date)"
echo "================================================================"

for EXP in "${EXPS[@]}"; do
    # Derive exp_name from config path for cleanup
    EXP_NAME=$(basename "$EXP" .yaml)

    echo ""
    echo "================================================================"
    echo "  RUNNING: $EXP ($EXP_NAME)"
    echo "  $(date)"
    echo "================================================================"

    # Clean previous outputs for this experiment
    rm -rf outputs/${EXP_NAME}*

    WANDB_MODE=online uv run python train.py experiment="$EXP" $COMMON_ARGS

    if [ $? -ne 0 ]; then
        echo "❌ Error in experiment $EXP. Stopping execution."
        exit 1
    fi

    echo "✅ Finished $EXP at $(date)."
    echo ""
done

echo "================================================================"
echo " 🎉 ALL EXPERIMENTS FINISHED SUCCESSFULLY! 🎉"
echo "  $(date)"
echo "================================================================"
