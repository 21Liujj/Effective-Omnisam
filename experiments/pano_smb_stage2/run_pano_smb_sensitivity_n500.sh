#!/usr/bin/env bash
# Pano-SMB sensitivity with n500 (FAST validation)
set -euo pipefail

ROOT="/root/autodl-tmp/OmniSAM-main"
cd "${ROOT}"

PYTHON="/root/miniconda3/envs/omnisam/bin/python"
GPU_ID="${GPU_ID:-0}"
SRC_CKPT="/root/autodl-tmp/omnisam_s.pth"
TGT_CKPT="/root/autodl-tmp/omnisam_s.pth"
FIXED_PL_DIR="${ROOT}/data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1"
PL_SUBSET_FILE="${ROOT}/experiments/pano_smb_stage2/Cityscapes_13cls_seed1_n500.txt"

LOG_FILE="${ROOT}/experiments/pano_smb_stage2/pano_smb_n500_$(date +%Y%m%d_%H%M%S).log"
echo "Starting Pano-SMB min_tokens sensitivity (n500 FAST) at $(date)" | tee "${LOG_FILE}"

run_one() {
  local label="$1" config="$2" run_dir="$3"
  echo "" | tee -a "${LOG_FILE}"
  echo "======================================" | tee -a "${LOG_FILE}"
  echo "Running: ${label} (n500)" | tee -a "${LOG_FILE}"
  echo "Config: ${config}" | tee -a "${LOG_FILE}"
  echo "Output: ${run_dir}" | tee -a "${LOG_FILE}"
  echo "======================================" | tee -a "${LOG_FILE}"
  
  env WANDB_MODE=offline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" run_adaptation.py \
    --config "${config}" --seed 1 --dataset Cityscapes --class_set 13 --backbone sam2_s \
    --data_root data --pd_root "${run_dir}" --src_ckpt "${SRC_CKPT}" --tgt_ckpt "${TGT_CKPT}" \
    --amp --tf32 --num_epochs 1 --batch_size 4 --pl_batch_size 1 --pl_num_workers 2 \
    --no-pl_shuffle --pl_subset_file "${PL_SUBSET_FILE}" --warmup_pl 2000 --iters_pl 2000 \
    --no-use_text_proto --text_proto_path "" --lambda_txt 0.0 --weight_txt_align 0.0 \
    --no-use_text_gate_ssl --no-use_wandb --no-use_tensorboard --reuse_pl_dir "${FIXED_PL_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"
  
  local rc=${PIPESTATUS[0]}
  if [[ ${rc} -ne 0 ]]; then
    echo "ERROR: ${label} failed with exit code ${rc}" | tee -a "${LOG_FILE}"
    exit ${rc}
  fi
  echo "${label} completed successfully" | tee -a "${LOG_FILE}"
}

EXP_ROOT="${ROOT}/experiments/pano_smb_stage2"

run_one "B0_legacy_n500" "${EXP_ROOT}/B0_legacy/config.yaml" "${EXP_ROOT}/B0_legacy_n500/run"
run_one "B1_min128_n500" "${EXP_ROOT}/B1_min128/config.yaml" "${EXP_ROOT}/B1_min128_n500/run"
run_one "B1_min256_n500" "${EXP_ROOT}/B1_pano_smb/config.yaml" "${EXP_ROOT}/B1_pano_smb_n500/run"
run_one "B1_min512_n500" "${EXP_ROOT}/B1_min512/config.yaml" "${EXP_ROOT}/B1_min512_n500/run"

echo "" | tee -a "${LOG_FILE}"
echo "All experiments completed at $(date)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Summary:" | tee -a "${LOG_FILE}"
for variant in B0_legacy_n500 B1_min128_n500 B1_pano_smb_n500 B1_min512_n500; do
  result_file="${EXP_ROOT}/${variant}/run/eval_results.json"
  if [[ -f "${result_file}" ]]; then
    echo -n "${variant}: " | tee -a "${LOG_FILE}"
    python -c "import json; d=json.load(open('${result_file}')); print(f'mIoU={d.get(\"mIoU\", \"N/A\"):.2f}, BFScore={d.get(\"BFScore\", \"N/A\"):.2f}')" 2>/dev/null | tee -a "${LOG_FILE}" || echo "N/A" | tee -a "${LOG_FILE}"
  fi
done

echo "" | tee -a "${LOG_FILE}"
echo "Complete log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
