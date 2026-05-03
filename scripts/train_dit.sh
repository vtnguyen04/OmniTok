#!/bin/bash
# Stage 2: Train DiT on Frozen Tokenizer
# Usage: ./scripts/train_dit.sh [experiment_path] [tokenizer_ckpt] [num_gpus]
# Example: ./scripts/train_dit.sh G3_our_tokenizer outputs/tokenizer/core/T_final/checkpoints/best_model.pt 8

EXP_NAME=${1:-"G3_our_tokenizer"}
TOKENIZER_CKPT=${2:-""}
NUM_GPUS=${3:-8}
PORT=${4:-29501}

echo "========================================================="
echo " Starting OmniTok Stage 2: DiT Training"
echo " Experiment: $EXP_NAME"
echo " Tokenizer:  $TOKENIZER_CKPT"
echo " GPUs:       $NUM_GPUS"
echo "========================================================="

if [ -z "$TOKENIZER_CKPT" ]; then
    echo "ERROR: Must provide a tokenizer checkpoint."
    echo "Usage: ./scripts/train_dit.sh [experiment_name] [tokenizer_ckpt]"
    exit 1
fi

mkdir -p outputs/dit/$EXP_NAME

# Assuming train_dit.py is similar to train.py but for Stage 2
accelerate launch \
    --num_processes=$NUM_GPUS \
    --main_process_port=$PORT \
    train.py \
    training=dit_train \
    +experiment=$EXP_NAME \
    model.tokenizer.checkpoint=$TOKENIZER_CKPT \
    2>&1 | tee outputs/dit/$EXP_NAME/train.log
