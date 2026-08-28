#!/bin/bash
# Route 3: Class-Aware Token Protection (n500 fast validation)
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
SUBSET_FILE="experiments/route3/runs/_subsets/Cityscapes_13cls_seed1_n500.txt"

declare -a CONFIGS=(
    "route3_rare010:experiments/route3/route3_rare010.yaml"
    "route3_rare015:experiments/route3/route3_rare015.yaml"
    "route3_rare020:experiments/route3/route3_rare020.yaml"
)

LOG_FILE="experiments/route3/route3_n500_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Route 3: Class-Aware Token Protection (n500 FAST)"
echo "Start time: $(date)"
echo "Seed: $SEED"
echo "Configs: ${#CONFIGS[@]}"
echo "Log: $LOG_FILE"
echo "=========================================="

exec > >(tee -a "$LOG_FILE") 2>&1

for config_entry in "${CONFIGS[@]}"; do
    IFS=: read -r name config_path <<< "$config_entry"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="${name}_n500_seed${SEED}_${TIMESTAMP}"
    OUTPUT_DIR="experiments/route3/runs/${RUN_NAME}"
    
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
    
    if [ -f "${OUTPUT_DIR}/eval_results.json" ]; then
        echo "Results:"
        $PYTHON -c "import json; d=json.load(open('${OUTPUT_DIR}/eval_results.json')); print(f'  mIoU: {d.get(\"mIoU\", \"N/A\"):.2f}'); print(f'  BFScore: {d.get(\"BFScore\", \"N/A\"):.2f}')" 2>/dev/null || echo "  (parsing failed)"
    fi
done

echo ""
echo "=========================================="
echo "Route 3 Phase 3A completed!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Summary:"
for config_entry in "${CONFIGS[@]}"; do
    IFS=: read -r name config_path <<< "$config_entry"
    RUN_DIR=$(ls -dt experiments/route3/runs/${name}_n500_seed${SEED}_* 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/eval_results.json" ]; then
        echo -n "$name (n500): "
        $PYTHON -c "import json; d=json.load(open('$RUN_DIR/eval_results.json')); print(f'mIoU={d.get(\"mIoU\", \"N/A\"):.2f}, BFScore={d.get(\"BFScore\", \"N/A\"):.2f}')" 2>/dev/null || echo "N/A"
    fi
done

echo ""
echo "Complete log: $LOG_FILE"
