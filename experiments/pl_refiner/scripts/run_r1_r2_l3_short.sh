#!/usr/bin/env bash
# experiments/pl_refiner/scripts/run_r1_r2_l3_short.sh
#
# L3 short-run: R1_conf_mask vs R2_class_balanced, 5 steps each.
# NO full training. NO eval. NO checkpoint saving (checkpoints ARE written
# by run_adaptation.py to avoid --no-save patching, but output dir is under
# experiments/pl_refiner/sanity_runs/ which is NOT data_pd_exp_final_2000/).
#
# Usage (on remote server):
#   cd /root/autodl-tmp/OmniSAM-main
#   bash experiments/pl_refiner/scripts/run_r1_r2_l3_short.sh
#
# Both runs share:
#   - reuse_pl_dir = data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
#   - src_ckpt = /root/autodl-tmp/omnisam_s.pth
#   - tgt_ckpt = /root/autodl-tmp/omnisam_s.pth
#   - seed = 1, batch_size = 4, train_max_iters = 5
#   - dataset = Cityscapes, class_set = 13, backbone = sam2_s
#   - NO eval (num_epochs limit via train_max_iters)

set -euo pipefail
cd /root/autodl-tmp/OmniSAM-main

PY=/root/miniconda3/envs/omnisam/bin/python3
SRC_CKPT=/root/autodl-tmp/omnisam_s.pth
TGT_CKPT=/root/autodl-tmp/omnisam_s.pth
DATA_ROOT=data
REUSE_PL=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1

echo "========================================"
echo "  L3 short-run: R1_conf_mask (5 steps)"
echo "========================================"
$PY run_adaptation.py \
  --config experiments/pl_refiner/configs/R1_conf_mask.yaml \
  --data_root "$DATA_ROOT" \
  --pd_root experiments/pl_refiner/sanity_runs/R1_conf_mask_l3 \
  --dataset Cityscapes \
  --backbone sam2_s \
  --class_set 13 \
  --src_ckpt "$SRC_CKPT" \
  --tgt_ckpt "$TGT_CKPT" \
  --reuse_pl_dir "$REUSE_PL" \
  --train_max_iters 5 \
  --seed 1 \
  --batch_size 4 \
  2>&1 | tee experiments/pl_refiner/notes/l3_r1_conf_mask.log

echo ""
echo "========================================"
echo "  L3 short-run: R2_class_balanced (5 steps)"
echo "========================================"
$PY run_adaptation.py \
  --config experiments/pl_refiner/configs/R2_class_balanced.yaml \
  --data_root "$DATA_ROOT" \
  --pd_root experiments/pl_refiner/sanity_runs/R2_class_balanced_l3 \
  --dataset Cityscapes \
  --backbone sam2_s \
  --class_set 13 \
  --src_ckpt "$SRC_CKPT" \
  --tgt_ckpt "$TGT_CKPT" \
  --reuse_pl_dir "$REUSE_PL" \
  --train_max_iters 5 \
  --seed 1 \
  --batch_size 4 \
  2>&1 | tee experiments/pl_refiner/notes/l3_r2_class_balanced.log

echo ""
echo "========================================"
echo "  L3 short-run complete."
echo "  Logs:"
echo "    experiments/pl_refiner/notes/l3_r1_conf_mask.log"
echo "    experiments/pl_refiner/notes/l3_r2_class_balanced.log"
echo "========================================"
