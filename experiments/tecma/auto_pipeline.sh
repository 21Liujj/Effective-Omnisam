#!/bin/bash
# Automatic Pipeline: Ablation → Test Set Evaluation

cd /root/autodl-tmp/OmniSAM-main

echo "=========================================="
echo "Automatic Experiment Pipeline"
echo "=========================================="
echo ""
echo "Stage 1: Padding Width Ablation (running...)"
echo "Stage 2: Test Set Evaluation (待启动)"
echo ""

# Wait for ablation to complete
ABLATION_COMPLETE=false
CHECK_INTERVAL=60  # Check every 60 seconds

while [ "$ABLATION_COMPLETE" = false ]; do
    # Check if all 4 tests are complete
    completed=0
    for pad_width in 256 512 768 1024; do
        if grep -q "Validation Finished" experiments/tecma/results/scm_ablation/pad${pad_width}.log 2>/dev/null; then
            ((completed++))
        fi
    done
    
    echo "[$(date '+%H:%M:%S')] Ablation进度: $completed/4 完成"
    
    if [ $completed -eq 4 ]; then
        ABLATION_COMPLETE=true
        echo ""
        echo "✓ Ablation实验全部完成！"
        echo ""
        
        # Print ablation results
        echo "Ablation Results:"
        for pad_width in 256 512 768 1024; do
            miou=$(grep "mIoU:" experiments/tecma/results/scm_ablation/pad${pad_width}.log | tail -1 | awk '{print $2}')
            echo "  pad_width=$pad_width: $miou mIoU"
        done
        echo ""
    else
        sleep $CHECK_INTERVAL
    fi
done

# Start test set evaluation
echo "=========================================="
echo "Starting Test Set Evaluation"
echo "=========================================="
echo ""

bash experiments/tecma/run_test_set_evaluation.sh

echo ""
echo "=========================================="
echo "All Experiments Complete!"
echo "=========================================="
