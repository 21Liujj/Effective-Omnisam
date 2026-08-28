#!/bin/bash
# Tonight All Routes: Pano-SMB + Route 1 + Route 3 (all n500)
# Total: 1.6h + 1.2h + 1.2h = 4h

set -e

BASE_DIR="/root/autodl-tmp/OmniSAM-main"
cd $BASE_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="tonight_all_routes_${TIMESTAMP}.log"

echo "=========================================="
echo "Tonight All Routes (n500 Fast)"
echo "Start time: $(date)"
echo "Timeline:"
echo "  Step 1: Pano-SMB (1.6h)"
echo "  Step 2: Route 1 Boundary (1.2h)"
echo "  Step 3: Route 3 Class-Aware (1.2h)"
echo "  Total: ~4h"
echo "Master log: $MASTER_LOG"
echo "=========================================="

exec > >(tee -a "$MASTER_LOG") 2>&1

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║ Step 1: Pano-SMB Sensitivity (n500)  ║"
echo "╚═══════════════════════════════════════╝"
echo "Start time: $(date)"

cd experiments/pano_smb_stage2
./run_pano_smb_sensitivity_n500.sh

echo ""
echo "Step 1 completed at $(date)"
echo ""
echo "╔═══════════════════════════════════════╗"
echo "║ Step 2: Route 1 Boundary Guard (n500)║"
echo "╚═══════════════════════════════════════╝"
echo "Start time: $(date)"

cd $BASE_DIR/experiments/route1
./run_boundary_sensitivity_n500.sh

echo ""
echo "Step 2 completed at $(date)"
echo ""
echo "╔═══════════════════════════════════════╗"
echo "║ Step 3: Route 3 Class-Aware (n500)   ║"
echo "╚═══════════════════════════════════════╝"
echo "Start time: $(date)"

cd $BASE_DIR/experiments/route3
./run_route3_n500.sh

echo ""
echo "Step 3 completed at $(date)"
echo ""
echo "=========================================="
echo "🎉 All Routes Completed!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results Summary:"
echo ""
echo "=== Pano-SMB ==="
find $BASE_DIR/experiments/pano_smb_stage2 -name "eval_results.json" -path "*n500*" 2>/dev/null | while read f; do
    echo -n "$(dirname $f | xargs basename): "
    python -c "import json; d=json.load(open()); print(f\"mIoU={d.get(mIoU,N/A):.2f}\")" 2>/dev/null || echo "N/A"
done

echo ""
echo "=== Route 1 Boundary ==="
find $BASE_DIR/experiments/route1/runs -name "eval_results.json" -path "*n500*" 2>/dev/null | while read f; do
    echo -n "$(dirname $f | xargs basename): "
    python -c "import json; d=json.load(open()); print(f\"mIoU={d.get(mIoU,N/A):.2f}, BFScore={d.get(BFScore,N/A):.2f}\")" 2>/dev/null || echo "N/A"
done

echo ""
echo "=== Route 3 Class-Aware ==="
find $BASE_DIR/experiments/route3/runs -name "eval_results.json" -path "*n500*" 2>/dev/null | while read f; do
    echo -n "$(dirname $f | xargs basename): "
    python -c "import json; d=json.load(open()); print(f\"mIoU={d.get(mIoU,N/A):.2f}, BFScore={d.get(BFScore,N/A):.2f}\")" 2>/dev/null || echo "N/A"
done

echo ""
echo "=========================================="
echo "Next steps:"
echo "  1. Analyze which configurations work best"
echo "  2. Tomorrow: n2000 confirmation + combinations"
echo "  3. Full System integration"
echo "=========================================="
