#!/bin/bash
# Padding Width Ablation Study

cd /root/autodl-tmp/OmniSAM-main
CKPT="experiments/route1/runs/boundary_0625_n500_seed1_20260826_101721/checkpoints/Cityscapes_13cls/sam2_s/best_iou59.99.pth"
RESULTS_DIR="experiments/tecma/results/scm_ablation"

mkdir -p $RESULTS_DIR

echo "=========================================="
echo "SCM Padding Width Ablation Study"
echo "=========================================="
echo "Testing pad_width: 256, 512, 768, 1024"
echo "Estimated time: ~15-20 minutes"
echo ""

for pad_width in 256 512 768 1024; do
    echo "----------------------------------------"
    echo "Testing pad_width=$pad_width"
    echo "----------------------------------------"
    
    /root/miniconda3/bin/python validation.py \
        --dataset DensePASS \
        --class_set 13 \
        --backbone sam2_s \
        --use_mem_bank \
        --batch_size 1 \
        --ckpt $CKPT \
        --enable_scm \
        --scm_pad_width $pad_width \
        > $RESULTS_DIR/pad${pad_width}.log 2>&1
    
    # Extract mIoU from log
    miou=$(grep "mIoU:" $RESULTS_DIR/pad${pad_width}.log | tail -1 | awk '{print $2}')
    echo "  pad_width=$pad_width → mIoU=$miou"
    echo ""
done

echo "=========================================="
echo "Ablation Study Complete!"
echo "=========================================="
echo ""
echo "Results summary:"
for pad_width in 256 512 768 1024; do
    miou=$(grep "mIoU:" $RESULTS_DIR/pad${pad_width}.log | tail -1 | awk '{print $2}')
    echo "  pad_width=$pad_width: $miou mIoU"
done
