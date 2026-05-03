#!/bin/bash
# Evaluate Tokenizer or DiT
# Usage: ./scripts/evaluate.sh [checkpoint_path] [dataset_path]

CHECKPOINT=${1:-""}
DATASET=${2:-"/path/to/imagenet/val"}

echo "========================================================="
echo " OmniTok Evaluation"
echo " Checkpoint: $CHECKPOINT"
echo " Dataset:    $DATASET"
echo "========================================================="

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: Must provide a checkpoint path."
    echo "Usage: ./scripts/evaluate.sh [checkpoint_path] [dataset_path]"
    exit 1
fi

python evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --data "$DATASET" \
    --batch-size 128 \
    --metrics rfid,zeroshot,linprobe,gaussianity \
    --device cuda
