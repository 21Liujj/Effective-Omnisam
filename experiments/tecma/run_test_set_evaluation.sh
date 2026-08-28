#!/bin/bash
# Test Set Evaluation for SCM

cd /root/autodl-tmp/OmniSAM-main
CKPT="experiments/route1/runs/boundary_0625_n500_seed1_20260826_101721/checkpoints/Cityscapes_13cls/sam2_s/best_iou59.99.pth"
RESULTS_DIR="experiments/tecma/results/test_set"

mkdir -p $RESULTS_DIR

echo "=========================================="
echo "Test Set Evaluation"
echo "=========================================="
echo "Purpose: Validate generalization of SCM"
echo ""

# Check if test list exists
TEST_LIST="dataset/densepass_list/test.txt"
if [ ! -f "$TEST_LIST" ]; then
    echo "Warning: Test list not found at $TEST_LIST"
    echo "Using validation list as fallback"
    TEST_LIST="dataset/densepass_list/val.txt"
fi

echo "Using test list: $TEST_LIST"
echo ""

# 1. Baseline on test set
echo "----------------------------------------"
echo "Step 1/2: Baseline (no SCM)"
echo "----------------------------------------"

/root/miniconda3/bin/python validation.py \
    --dataset DensePASS \
    --class_set 13 \
    --backbone sam2_s \
    --use_mem_bank \
    --batch_size 1 \
    --ckpt $CKPT \
    --val_list $TEST_LIST \
    > $RESULTS_DIR/baseline_test.log 2>&1

baseline_miou=$(grep "mIoU:" $RESULTS_DIR/baseline_test.log | tail -1 | awk '{print $2}')
echo "  Baseline Test mIoU: $baseline_miou"
echo ""

# 2. SCM on test set
echo "----------------------------------------"
echo "Step 2/2: +SCM (pad_width=512)"
echo "----------------------------------------"

/root/miniconda3/bin/python validation.py \
    --dataset DensePASS \
    --class_set 13 \
    --backbone sam2_s \
    --use_mem_bank \
    --batch_size 1 \
    --ckpt $CKPT \
    --val_list $TEST_LIST \
    --enable_scm \
    --scm_pad_width 512 \
    > $RESULTS_DIR/scm_test.log 2>&1

scm_miou=$(grep "mIoU:" $RESULTS_DIR/scm_test.log | tail -1 | awk '{print $2}')
echo "  +SCM Test mIoU: $scm_miou"
echo ""

# Summary
echo "=========================================="
echo "Test Set Results Summary"
echo "=========================================="
echo ""
echo "Validation Set (已完成):"
echo "  Baseline: 60.51 mIoU"
echo "  +SCM:     60.72 mIoU (+0.21)"
echo ""
echo "Test Set (刚完成):"
echo "  Baseline: $baseline_miou mIoU"
echo "  +SCM:     $scm_miou mIoU"
echo ""

# Calculate improvement
if [ ! -z "$baseline_miou" ] && [ ! -z "$scm_miou" ]; then
    improvement=$(python3 -c "print(f'{float('$scm_miou') - float('$baseline_miou'):+.2f}')")
    echo "  Improvement: $improvement mIoU"
    echo ""
    
    if (( $(echo "$improvement > 0" | bc -l) )); then
        echo "✓ SCM shows consistent improvement on test set!"
    else
        echo "⚠ Performance differs from validation set, needs investigation"
    fi
fi

echo ""
echo "Detailed logs:"
echo "  Baseline: $RESULTS_DIR/baseline_test.log"
echo "  +SCM:     $RESULTS_DIR/scm_test.log"
echo ""
echo "=========================================="
