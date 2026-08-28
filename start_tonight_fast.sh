#!/bin/bash
# Tonight Fast Experiments (Plan B): Both n500 for rapid validation
# Pano-SMB (4×n500) + Route 1 (3×n500) = ~2.8h total

set -e

BASE_DIR="/root/autodl-tmp/OmniSAM-main"
cd $BASE_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="tonight_fast_${TIMESTAMP}.log"

echo "=========================================="
echo "Tonight Fast Experiments (Plan B - n500)"
echo "Start time: $(date)"
echo "Timeline: ~2.8h total"
echo "  Step 1: Pano-SMB 4×n500 (~1.6h)"
echo "  Step 2: Route 1 3×n500 (~1.2h)"
echo "Master log: $MASTER_LOG"
echo "=========================================="

exec > >(tee -a "$MASTER_LOG") 2>&1

echo ""
echo "Step 1: Starting Pano-SMB sensitivity (n500)..."
echo "Start time: $(date)"

cd experiments/pano_smb_stage2
./run_pano_smb_sensitivity_n500.sh

echo ""
echo "Step 1 completed at $(date)"
echo ""
echo "Step 2: Starting Route 1 Phase 1 (n500)..."
echo "Start time: $(date)"

cd $BASE_DIR/experiments/route1
./run_boundary_sensitivity_n500.sh

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Pano-SMB: experiments/pano_smb_stage2/B*_n500/run/"
echo "  - Route 1: experiments/route1/runs/*_n500_*/"
echo ""
echo "View results:"
echo "  find experiments -name eval_results.json | xargs cat"
