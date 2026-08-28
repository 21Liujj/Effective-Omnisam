#!/usr/bin/env bash
# experiments/pl_refiner/scripts/run_r4_proto_threshold_calibration_5step_diag.sh
#
# Phase 4 / Task 4-3c: R4 current-batch prototype similarity threshold calibration.
# Diagnostic-only: no proto_gate filtering, no pseudo_label changes, no valid_mask changes, no loss changes.
#
# Outputs:
#   JSONL: experiments/pl_refiner/results/r4_proto_threshold_calibration_5step_diag.jsonl
#   Log  : experiments/pl_refiner/notes/l3_r4_proto_threshold_calibration_5step_diag.log
#
# Usage (on remote server):
#   cd /root/autodl-tmp/OmniSAM-main
#   bash experiments/pl_refiner/scripts/run_r4_proto_threshold_calibration_5step_diag.sh

set -euo pipefail
cd /root/autodl-tmp/OmniSAM-main

PY=/root/miniconda3/envs/omnisam/bin/python3
SRC_CKPT=/root/autodl-tmp/omnisam_s.pth
TGT_CKPT=/root/autodl-tmp/omnisam_s.pth
DATA_ROOT=data
REUSE_PL=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
CONFIG=experiments/pl_refiner/configs/R4_proto_threshold_calibration.yaml
LOG=experiments/pl_refiner/notes/l3_r4_proto_threshold_calibration_5step_diag.log
JSONL=experiments/pl_refiner/results/r4_proto_threshold_calibration_5step_diag.jsonl

mkdir -p experiments/pl_refiner/results experiments/pl_refiner/notes experiments/pl_refiner/sanity_runs
rm -f "$JSONL" "$LOG"

echo "========================================"
echo "  R4 prototype similarity threshold calibration — 5-step diagnostic"
echo "  Config: $CONFIG"
echo "  JSONL : $JSONL"
echo "  Log   : $LOG"
echo "  Note  : diagnostic-only; no proto_gate filtering; no loss change"
echo "========================================"

$PY run_adaptation.py \
  --config "$CONFIG" \
  --data_root "$DATA_ROOT" \
  --pd_root experiments/pl_refiner/sanity_runs/R4_proto_threshold_calibration_5step \
  --dataset Cityscapes \
  --backbone sam2_s \
  --class_set 13 \
  --src_ckpt "$SRC_CKPT" \
  --tgt_ckpt "$TGT_CKPT" \
  --reuse_pl_dir "$REUSE_PL" \
  --train_max_iters 5 \
  --seed 1 \
  --batch_size 4 \
  2>&1 | tee "$LOG"

echo ""
echo "========================================"
echo "  Done."
echo "  Threshold calibration diagnostics: $JSONL"
echo "  Log: $LOG"
echo "  If validation appears in log, treat it as framework artifact; do not use val/mIoU as R4 result."
echo "========================================"
