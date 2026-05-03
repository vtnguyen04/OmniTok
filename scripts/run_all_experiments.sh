#!/bin/bash
# Sequential experiment runner for OmniTok
# Each runs for ~2 epochs on ImageNet-10k

EXPS=(
    "baseline/T0_vtp_baseline"
    "baseline/T1_vanilla_vae"
    "alignment/T2_frozen_dino_relkd"
    "alignment/T3_frozen_dino_cosine"
    "alignment/T4_frozen_dino_prediction"
    "multi_teacher/T5_multi_dino_siglip"
    "multi_teacher/T6_multi_dino_siglip_sam"
    "multi_teacher/T7_dino_depth"
    "alignment/T8_maetok_full"
    "alignment/T9_gaussianity_reg"
    "ablations/architecture/T10_linear_projector"
    "core/T_final"
)

for EXP in "${EXPS[@]}"; do
    echo "================================================================"
    echo "RUNNING EXPERIMENT: $EXP"
    echo "================================================================"
    
    python train.py +experiment="$EXP"
    
    if [ $? -ne 0 ]; then
        echo "Error in experiment $EXP. Skipping..."
    fi
    
    echo "Finished $EXP."
    echo ""
done

echo "All experiments finished."
