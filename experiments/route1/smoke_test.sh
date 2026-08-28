#!/usr/bin/env bash
# Route 1 Boundary-Guarded Mode - Smoke Test
# 仅跑2个iteration验证代码不崩溃/不OOM

set -euo pipefail
cd /root/autodl-tmp/OmniSAM-main

source /root/miniconda3/etc/profile.d/conda.sh
conda activate omnisam

CUDA_VISIBLE_DEVICES=0 python run_adaptation.py \
  --config experiments/route1/boundary_only.yaml \
  --seed 1 \
  --dataset Cityscapes \
  --class_set 13 \
  --backbone sam2_s \
  --data_root data \
  --pd_root experiments/route1/smoke_test_run \
  --src_ckpt /root/autodl-tmp/omnisam_s.pth \
  --tgt_ckpt /root/autodl-tmp/omnisam_s.pth \
  --amp --tf32 \
  --num_epochs 1 \
  --batch_size 1 \
  --pl_batch_size 1 \
  --pl_num_workers 2 \
  --no-pl_shuffle \
  --pl_subset_file /root/autodl-tmp/OmniSAM-main/experiments/tecma/runs/_subsets/Cityscapes_13cls_seed1_n2000.txt \
  --warmup_pl 2 \
  --iters_pl 2 \
  --no-use_text_proto \
  --text_proto_path '' \
  --lambda_txt 0.0 \
  --weight_txt_align 0.0 \
  --no-use_text_gate_ssl \
  --no-use_wandb \
  --no-use_tensorboard
