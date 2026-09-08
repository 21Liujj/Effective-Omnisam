#!/bin/bash

# 最优配置实验：固定SCM + Memory压缩 + 关闭动态SG-DRF

set -e

cd /root/autodl-tmp/OmniSAM-main

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"

EXP_NAME="optimal_scm_100img_5ep"
PD_ROOT="pd_${EXP_NAME}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_${TIMESTAMP}.log"

echo "========================================"
echo "  最优配置实验 + SCM (100张图)"
echo "========================================"
echo "配置:"
echo "  - 固定SCM (pad_width=256) ✅"
echo "  - Memory压缩 (pool, ratio=2) ✅"
echo "  - 关闭动态SG-DRF ✅"
echo "  - 固定伪标签 ✅"
echo "  - 5 epochs"
echo ""
echo "日志: ${LOG_FILE}"
echo "========================================"

# 复用之前最佳的伪标签
mkdir -p ${PD_ROOT}
if [ -d "pd_densepass_uda_100img_15ep_dynamic/DensePASS" ]; then
    cp -r pd_densepass_uda_100img_15ep_dynamic/DensePASS ${PD_ROOT}/
    echo "[✓] 已复用Exp1 Epoch1伪标签"
else
    echo "[✗] 警告: 未找到之前的伪标签目录"
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
    --num_epochs 5 \
    --batch_size 1 \
    --warmup_pl 100 \
    --iters_pl 0 \
    --no-enable_sgdrf \
    --use_memory_compression \
    --memory_compress_mode pool \
    --memory_downsample_ratio 2 \
    --num_maskmem 9 \
    --enable_scm \
    --scm_pad_width 256 \
    > ${LOG_FILE} 2>&1 &

PID=$!
echo "[$(date)] 实验已启动，PID: ${PID}"
echo "${PID}" > /tmp/optimal_scm_pid.txt
echo ""
echo "监控: tail -f ${LOG_FILE}"
echo "结果: grep 'Epoch.*val mIoU' ${LOG_FILE}"
echo ""
echo "预计完成: 约30-40分钟"
echo "预期性能: 60.5-61.5% mIoU (稳定，无下降)"

