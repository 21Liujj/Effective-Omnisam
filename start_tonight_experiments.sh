#!/bin/bash
# Tonight Experiments: Pano-SMB (Task 4) + Route 1 (Task 5) auto-chain
# Usage: nohup ./start_tonight_experiments.sh &
# Start after 18:00 when attn_topk finishes

set -e

BASE_DIR="/root/autodl-tmp/OmniSAM-main"
cd $BASE_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="tonight_experiments_${TIMESTAMP}.log"
MASTER_PID="tonight_experiments_${TIMESTAMP}.pid"

echo "=========================================="
echo "Tonight Experiments Master Controller"
echo "Start time: $(date)"
echo "Tasks: Pano-SMB sensitivity + Route 1 Phase 1"
echo "Master log: $MASTER_LOG"
echo "Master PID: $$"
echo "=========================================="

# Save master PID
echo $$ > $MASTER_PID

echo ""
echo "Task 4: Starting Pano-SMB sensitivity analysis..."
echo "Expected duration: ~6h (18:00-24:00)"
echo ""

# Start Pano-SMB (assume the script exists)
cd experiments/pano_smb_stage2
if [ -f "run_pano_smb_sensitivity.sh" ]; then
    nohup ./run_pano_smb_sensitivity.sh > pano_smb_${TIMESTAMP}.log 2>&1 &
    PANO_PID=$!
    echo $PANO_PID > batch_pano_smb_${TIMESTAMP}.pid
    echo "Pano-SMB started with PID: $PANO_PID"
else
    echo "ERROR: Pano-SMB script not found!"
    echo "Please ensure experiments/pano_smb_stage2/run_pano_smb_sensitivity.sh exists"
    exit 1
fi

cd $BASE_DIR

echo ""
echo "Task 5: Starting auto-chain monitor..."
echo "Will auto-start Route 1 when Pano-SMB completes"
echo ""

# Start the auto-chain monitor in background
nohup ./experiments/auto_chain_pano_route1.sh > auto_chain_${TIMESTAMP}.log 2>&1 &
CHAIN_PID=$!
echo "Auto-chain monitor started with PID: $CHAIN_PID"

echo ""
echo "=========================================="
echo "All background tasks started!"
echo "=========================================="
echo ""
echo "Timeline:"
echo "  18:00-24:00: Pano-SMB sensitivity (4 configs × 1.5h)"
echo "  24:00-01:12: Route 1 Phase 1 n500 (3 configs × 0.4h)"
echo "  ~01:12: ALL DONE"
echo ""
echo "Monitor commands:"
echo "  - Pano-SMB log: tail -f experiments/pano_smb_stage2/pano_smb_${TIMESTAMP}.log"
echo "  - Route 1 log: tail -f auto_chain_${TIMESTAMP}.log"
echo "  - GPU status: nvidia-smi"
echo "  - Kill all: kill $PANO_PID $CHAIN_PID"
echo ""
echo "PIDs saved to:"
echo "  - Master: $MASTER_PID"
echo "  - Pano-SMB: experiments/pano_smb_stage2/batch_pano_smb_${TIMESTAMP}.pid"
echo "  - Auto-chain: (embedded in master)"
