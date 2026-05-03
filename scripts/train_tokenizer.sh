#!/bin/bash
# Stage 1: Train Tokenizer
# Usage: ./scripts/train_tokenizer.sh [experiment_path] [num_gpus]
# Example: ./scripts/train_tokenizer.sh multi_teacher/T5_multi_dino_siglip 8

EXP_NAME=${1:-"baseline/T0_vtp_baseline"}
NUM_GPUS=${2:-8}
PORT=${3:-29500}

echo "========================================================="
echo " Starting OmniTok Stage 1: Tokenizer Training"
echo " Experiment: $EXP_NAME"
echo " GPUs:       $NUM_GPUS"
echo "========================================================="

# Create outputs dir
mkdir -p outputs/tokenizer/$EXP_NAME

# Run via accelerate
accelerate launch \
    --num_processes=$NUM_GPUS \
    --main_process_port=$PORT \
    train.py \
    +experiment=$EXP_NAME \
    training.data.num_workers=8 \
    training.mixed_precision="bf16" \
    2>&1 | tee outputs/tokenizer/$EXP_NAME/train.log
