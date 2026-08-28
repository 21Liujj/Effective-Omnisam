#!/bin/bash
# Launcher for Route 1 Phase 1 experiments
# Usage: nohup ./start_route1.sh > route1_phase1.log 2>&1 &

BASE_DIR="/root/autodl-tmp/OmniSAM-main"
cd $BASE_DIR/experiments/route1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="route1_phase1_${TIMESTAMP}.log"
PID_FILE="route1_phase1_${TIMESTAMP}.pid"

echo "=========================================="
echo "Route 1 Phase 1: Starting Boundary Sensitivity Analysis"
echo "Time: $(date)"
echo "Log: $LOG_FILE"
echo "PID will be saved to: $PID_FILE"
echo "=========================================="

# Save PID
echo $$ > $PID_FILE

# Run the batch script
./run_boundary_sensitivity.sh 2>&1 | tee $LOG_FILE

echo ""
echo "=========================================="
echo "Route 1 Phase 1: Completed"
echo "Time: $(date)"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
