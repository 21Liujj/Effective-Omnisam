#!/bin/bash
# 方案A: 延长Cold-Start到5 epochs
# 核心思路: 给density estimator更多训练时间

set -e
cd /root/autodl-tmp/OmniSAM-main

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="sgdrf_tune_a_cold5"
PD_ROOT="pd_${EXP_NAME}"
LOG_FILE="logs/${EXP_NAME}_${TIMESTAMP}.log"

echo "========================================"
echo "  方案A: 延长Cold-Start (5 epochs)"
echo "========================================"
echo "参数:"
echo "  - sgdrf_cold_start_epochs: 5 (默认2)"
echo "  - 其他参数: 默认"
echo ""
echo "预期:"
echo "  Epoch 1-5: 60% (cold_start)"
echo "  Epoch 6+: ≥58% (joint开始)"
echo ""
echo "日志: ${LOG_FILE}"
echo "========================================"

# 复用伪标签
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
    --sgdrf_cold_start_epochs 5 \
    > ${LOG_FILE} 2>&1 &

PID=$!
echo "[$(date)] 实验已启动，PID: ${PID}"
echo "${PID}" > /tmp/sgdrf_exp_a_pid.txt
echo ""
echo "监控: tail -f ${LOG_FILE}"
echo "结果: grep 'Epoch.*val mIoU' ${LOG_FILE}"
echo ""
echo "预计完成: 约60-70分钟"
