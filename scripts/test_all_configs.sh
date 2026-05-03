#!/bin/bash
# scripts/test_all_configs.sh
# Test all experiment configs for 1 step each to ensure no crashes.

set -e

CONFIGS=$(find configs/experiment -name "*.yaml")

echo "Starting smoke tests for $(echo "$CONFIGS" | wc -l) configs..."

for cfg_path in $CONFIGS; do
    # Get relative path from configs/experiment
    cfg_name=$(echo "$cfg_path" | sed 's|configs/experiment/||' | sed 's|.yaml||')
    
    echo "----------------------------------------------------------------"
    echo "TESTING: $cfg_name"
    echo "----------------------------------------------------------------"
    
    # Kill any stale processes first
    pkill -9 -f train.py || true
    
    # Run 15 steps to check convergence
    uv run python train.py experiment="$cfg_name" \
        training.max_steps=15 \
        training.data.batch_size=2 \
        training.use_wandb=false \
        training.save_every=15 \
        training.log_every=1 \
        data.num_workers=0
    
    echo "SUCCESS: $cfg_name"
done

echo "----------------------------------------------------------------"
echo "ALL CONFIGS PASSED SMOKE TEST!"
echo "----------------------------------------------------------------"
