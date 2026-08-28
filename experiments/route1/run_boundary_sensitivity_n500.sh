#!/bin/bash
# Route 1 Phase 1: Boundary-guarded sensitivity analysis (n500 fast validation)
# Run 3 boundary_reserve_ratio values with seed=1, n500 subset
# Expected total time: ~1.2h (3 × 0.4h)

set -e

BASE_DIR="/root/autodl-tmp/OmniSAM-main"
PYTHON="/root/miniconda3/envs/omnisam/bin/python"
REUSE_PL_DIR="/root/autodl-tmp/OmniSAM-main/data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1"
cd $BASE_DIR

SEED=1
DATASET="Cityscapes"
CLASS_SET=13
BACKBONE="sam2_s"
DATA_ROOT="data"
SRC_CKPT="/root/autodl-tmp/omnisam_s.pth"
TGT_CKPT="/root/autodl-tmp/omnisam_s.pth"
NUM_EPOCHS=1
BATCH_SIZE=4
PL_BATCH_SIZE=1
PL_NUM_WORKERS=2

# Use n500 for fast validation
SUBSET_FILE="experiments/route1/runs/_subsets/Cityscapes_13cls_seed1_n500.txt"

# Configs to test
declare -a CONFIGS=(
    "boundary_0625:experiments/route1/boundary_0625.yaml"
    "boundary_125:experiments/route1/boundary_125.yaml"
    "boundary_25:experiments/route1/boundary_25.yaml"
)

echo "=========================================="
echo "Route 1 Phase 1: Boundary Sensitivity (n500 FAST)"
echo "Start time: $(date)"
echo "Seed: $SEED"
echo "Subset: n500 (fast validation)"
echo "Configs: ${#CONFIGS[@]}"
echo "=========================================="

for config_entry in "${CONFIGS[@]}"; do
    IFS=: read -r name config_path <<< "$config_entry"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="${name}_n500_seed${SEED}_${TIMESTAMP}"
    OUTPUT_DIR="experiments/route1/runs/${RUN_NAME}"
    
    echo ""
    echo "=========================================="
    echo "Running: ${name} (n500)"
    echo "Config: ${config_path}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Time: $(date)"
    echo "=========================================="
    
    $PYTHON run_adaptation.py \
        --config $config_path \
        --seed $SEED \
        --dataset $DATASET \
        --class_set $CLASS_SET \
        --backbone $BACKBONE \
        --data_root $DATA_ROOT \
        --pd_root $OUTPUT_DIR \
        --src_ckpt $SRC_CKPT \
        --tgt_ckpt $TGT_CKPT \
        --amp \
        --tf32 \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --pl_batch_size $PL_BATCH_SIZE \
        --pl_num_workers $PL_NUM_WORKERS \
        --no-pl_shuffle \
        --pl_subset_file $SUBSET_FILE \
        --warmup_pl 2000 \
        --iters_pl 2000 \
        --no-use_text_proto \
        --text_proto_path "" \
        --lambda_txt 0.0 \
        --weight_txt_align 0.0 \
        --no-use_text_gate_ssl \
        --no-use_wandb \
        --no-use_tensorboard
    
    echo ""
    echo "Completed: ${name}"
    echo "Time: $(date)"
    
    # Extract results if available
    if [ -f "${OUTPUT_DIR}/eval_results.json" ]; then
        echo "Results:"
        $PYTHON -c "import json; d=json.load(open('${OUTPUT_DIR}/eval_results.json')); print(f'  mIoU: {d.get(\"mIoU\", \"N/A\"):.2f}'); print(f'  BFScore: {d.get(\"BFScore\", \"N/A\"):.2f}')" 2>/dev/null || echo "  (parsing failed)"
    fi
    
done

echo ""
echo "=========================================="
echo "Route 1 Phase 1 (n500) completed!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Summary:"
for config_entry in "${CONFIGS[@]}"; do
    IFS=: read -r name config_path <<< "$config_entry"
    RUN_DIR=$(ls -dt experiments/route1/runs/${name}_n500_seed${SEED}_* 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/eval_results.json" ]; then
        echo -n "$name (n500): "
        $PYTHON -c "import json; d=json.load(open('$RUN_DIR/eval_results.json')); print(f'mIoU={d.get(\"mIoU\", \"N/A\"):.2f}, BFScore={d.get(\"BFScore\", \"N/A\"):.2f}')" 2>/dev/null || echo "N/A"
    fi
done
