#!/usr/bin/env bash
# experiments/pl_refiner/scripts/run_r3_plq_5step_diag.sh
#
# Task 3-1 rescue: R3 PLQ_basic 5-step diagnostic run.
# Uses run_adaptation.py (confirmed working) with R3_plq_basic.yaml.
# Substitutes for standalone L2 diagnosis (which had import/path issues).
#
# What this run does:
#   - 5 training steps only, no full epoch
#   - No eval, no mIoU measurement
#   - No PLQ-weighted CE (pl_weight is computed but NOT applied to loss)
#   - Records PLQ diagnostics per step to:
#       experiments/pl_refiner/results/r3_plq_5step_diag.jsonl
#   - stdout log -> experiments/pl_refiner/notes/l3_r3_plq_5step_diag.log
#
# Dataset / ckpt / seed / batch_size / num_workers: same as R1/R2 short-run.
# Reuses existing PL dir (no PL regeneration).
#
# Usage (on remote server):
#   cd /root/autodl-tmp/OmniSAM-main
#   bash experiments/pl_refiner/scripts/run_r3_plq_5step_diag.sh

set -euo pipefail
cd /root/autodl-tmp/OmniSAM-main

PY=/root/miniconda3/envs/omnisam/bin/python3
SRC_CKPT=/root/autodl-tmp/omnisam_s.pth
TGT_CKPT=/root/autodl-tmp/omnisam_s.pth
DATA_ROOT=data
REUSE_PL=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
LOG=experiments/pl_refiner/notes/l3_r3_plq_5step_diag.log

mkdir -p experiments/pl_refiner/results
mkdir -p experiments/pl_refiner/notes

echo "========================================"
echo "  R3 PLQ_basic — 5-step diagnostic run"
echo "  Config: experiments/pl_refiner/configs/R3_plq_basic.yaml"
echo "  PLQ diag output: experiments/pl_refiner/results/r3_plq_5step_diag.jsonl"
echo "  Log: $LOG"
echo "========================================"

$PY run_adaptation.py \
  --config experiments/pl_refiner/configs/R3_plq_basic.yaml \
  --data_root "$DATA_ROOT" \
  --pd_root experiments/pl_refiner/sanity_runs/R3_plq_basic_5step \
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
echo "  PLQ diagnostics: experiments/pl_refiner/results/r3_plq_5step_diag.jsonl"
echo "  Log: $LOG"
echo "========================================"
