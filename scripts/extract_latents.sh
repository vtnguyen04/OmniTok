#!/bin/bash
# Pre-cache latents from a frozen tokenizer
# Usage: ./scripts/extract_latents.sh [tokenizer_ckpt] [dataset_path] [output_dir]

CHECKPOINT=${1:-""}
DATASET=${2:-"/path/to/imagenet/train"}
OUTPUT_DIR=${3:-"data/imagenet_latents"}

echo "========================================================="
echo " OmniTok Latent Extraction"
echo " Checkpoint: $CHECKPOINT"
echo " Dataset:    $DATASET"
echo " Output:     $OUTPUT_DIR"
echo "========================================================="

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: Must provide a tokenizer checkpoint."
    echo "Usage: ./scripts/extract_latents.sh [tokenizer_ckpt] [dataset_path] [output_dir]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python tools/extract_latents.py \
    --checkpoint "$CHECKPOINT" \
    --data "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 256 \
    --num-workers 16 \
    --device cuda
