#!/bin/bash
# Auto-chain: Pano-SMB sensitivity (Task 4) -> Route 1 Phase 1 (Task 5)
# This script monitors Pano-SMB completion and auto-starts Route 1

set -e

BASE_DIR="/root/autodl-tmp/OmniSAM-main"
cd $BASE_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="experiments/auto_chain_${TIMESTAMP}.log"
PID_FILE="experiments/auto_chain_${TIMESTAMP}.pid"

echo "=========================================="
echo "Auto-Chain: Pano-SMB -> Route 1"
echo "Start time: $(date)"
echo "Log: $LOG_FILE"
echo "PID: $$"
echo "=========================================="

# Save PID
echo $$ > $PID_FILE

echo ""
echo "Step 1: Waiting for Pano-SMB to complete..."
echo "Checking for running batch_pano_smb process..."

# Wait for Pano-SMB to complete
while true; do
    # Check if Pano-SMB batch script is still running
    if pgrep -f "batch_pano_smb" > /dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] Pano-SMB still running, checking again in 5 min..."
        sleep 300  # Check every 5 minutes
    else
        echo "[$(date +%H:%M:%S)] Pano-SMB process not found, checking completion..."
        
        # Extra wait to ensure clean completion
        sleep 60
        
        # Verify no training processes
        if ! pgrep -f "run_adaptation.py.*pano_smb" > /dev/null 2>&1; then
            echo "[$(date +%H:%M:%S)] Pano-SMB confirmed completed!"
            break
        fi
    fi
done

echo ""
echo "=========================================="
echo "Step 2: Starting Route 1 Phase 1 (n500)"
echo "Time: $(date)"
echo "=========================================="

cd $BASE_DIR/experiments/route1

# Run Route 1
./run_boundary_sensitivity_n500.sh 2>&1 | tee route1_autostart_${TIMESTAMP}.log

echo ""
echo "=========================================="
echo "Auto-Chain Completed!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "All tasks finished:"
echo "  Task 4: Pano-SMB sensitivity (completed before this)"
echo "  Task 5: Route 1 Phase 1 n500 (just completed)"
