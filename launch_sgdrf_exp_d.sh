#!/bin/bash
# 方案D: 降低density权重，减少动态模块影响

set -e
cd /root/autodl-tmp/OmniSAM-main

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="sgdrf_tune_d_lowweight"
PD_ROOT="pd_${EXP_NAME}"
LOG_FILE="logs/${EXP_NAME}_${TIMESTAMP}.log"

echo "========================================"
echo "  方案D: 降低Density权重"
echo "========================================"
echo "参数:"
echo "  - density_weight: 0.01 (从0.1降低10倍)"
echo "  - grad_clip: 0.3 (更强约束)"
echo "  - 减少动态模块对主模型的干扰"
echo ""
echo "日志: ${LOG_FILE}"
echo "========================================"

mkdir -p ${PD_ROOT}
if [ -d "pd_densepass_uda_100img_15ep_dynamic/DensePASS" ]; then
    cp -r pd_densepass_uda_100img_15ep_dynamic/DensePASS ${PD_ROOT}/
    echo "[✓] 已复用伪标签"
fi
mkdir -p ${PD_ROOT}/checkpoints

echo "[$(date)] 启动实验..."

nohup /root/miniconda3/bin/python run_adaptation.py \
    --backbone sam2_s \
    --dataset Cityscapes \
    --data_root data \
    --pd_root ${PD_ROOT} \
    --src_ckpt /root/autodl-tmp/omnisam_s.pth \
    --tgt_ckpt /root/autodl-tmp/omnisam_s.pth \
    --num_epochs 10 \
    --batch_size 1 \
    --warmup_pl 100 \
    --iters_pl 0 \
    --enable_sgdrf \
    --sgdrf_cold_start_epochs 2 \
    --sgdrf_density_weight 0.01 \
    --sgdrf_grad_clip_norm 0.3 \
    > ${LOG_FILE} 2>&1 &

PID=$!
echo "[$(date)] 实验已启动，PID: ${PID}"
echo "${PID}" > /tmp/sgdrf_exp_d_pid.txt
echo "预计完成: 约60-70分钟"
