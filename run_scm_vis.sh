#!/bin/bash
set -e

CKPT=/root/autodl-tmp/OmniSAM-main/data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.16.pth
OUT_DIR=paper/scm_qualitative_temp

echo "=== 运行SCM推理，保存可视化结果 ==="
python validation.py \
  --dataset DensePASS \
  --class_set 13 \
  --backbone sam2_s \
  --use_mem_bank \
  --batch_size 1 \
  --ckpt $CKPT \
  --enable_scm \
  --scm_pad_width 512 \
  --save_pred_debug 4 \
  --save_pred_out $OUT_DIR \
  --num_workers 0

echo "=== 推理完成，结果保存到 $OUT_DIR ==="
ls -lh $OUT_DIR/ 2>/dev/null || echo "输出目录为空，检查参数"
