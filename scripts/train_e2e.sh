#!/bin/bash
# Stage 3: Train E2E (Tokenizer + DiT)
# Usage: ./scripts/train_e2e.sh [experiment_name] [tokenizer_ckpt] [dit_ckpt] [num_gpus]

EXP_NAME=${1:-"E3_through_alignment"}
TOKENIZER_CKPT=${2:-""}
DIT_CKPT=${3:-""}
NUM_GPUS=${4:-8}
PORT=${5:-29502}

echo "========================================================="
echo " Starting OmniTok Stage 3: E2E Training"
echo " Experiment: $EXP_NAME"
echo " GPUs:       $NUM_GPUS"
echo "========================================================="

mkdir -p outputs/e2e/$EXP_NAME

accelerate launch \
    --num_processes=$NUM_GPUS \
    --main_process_port=$PORT \
    train.py \
    training=e2e_train \
    +experiment=$EXP_NAME \
    model.tokenizer.checkpoint=$TOKENIZER_CKPT \
    model.dit.checkpoint=$DIT_CKPT \
    2>&1 | tee outputs/e2e/$EXP_NAME/train.log
