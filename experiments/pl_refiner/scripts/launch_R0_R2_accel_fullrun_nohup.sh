#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/OmniSAM-main

# Activate conda env safely
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f "/root/miniconda/etc/profile.d/conda.sh" ]; then
  source /root/miniconda/etc/profile.d/conda.sh
fi
conda activate omnisam

TS=$(date +%Y%m%d_%H%M%S)
RUN_TAG="R0_R2_accel_seed1_${TS}"
LOG_DIR="experiments/pl_refiner/fullrun_logs"
META_DIR="experiments/pl_refiner/fullrun_meta"
LOG_FILE="${LOG_DIR}/${RUN_TAG}.log"
META_FILE="${META_DIR}/${RUN_TAG}_meta.txt"
PID_FILE="${META_DIR}/${RUN_TAG}.pid"

mkdir -p "${LOG_DIR}" "${META_DIR}"

echo "========== LAUNCH META ==========" | tee "${META_FILE}"
echo "timestamp: ${TS}" | tee -a "${META_FILE}"
echo "pwd: $(pwd)" | tee -a "${META_FILE}"
echo "runs: R0 R2" | tee -a "${META_FILE}"
echo "gpu: 0" | tee -a "${META_FILE}"
echo "seed: 1" | tee -a "${META_FILE}"
echo "num_epochs: 1" | tee -a "${META_FILE}"
echo "log_file: ${LOG_FILE}" | tee -a "${META_FILE}"
echo "" | tee -a "${META_FILE}"

echo "========== DISK ==========" | tee -a "${META_FILE}"
df -h /root/autodl-tmp | tee -a "${META_FILE}"
echo "" | tee -a "${META_FILE}"

echo "========== GPU ==========" | tee -a "${META_FILE}"
nvidia-smi | tee -a "${META_FILE}" || true
echo "" | tee -a "${META_FILE}"

echo "========== CKPT ==========" | tee -a "${META_FILE}"
ls -lh /root/autodl-tmp/omnisam_s.pth | tee -a "${META_FILE}" || true
sha256sum /root/autodl-tmp/omnisam_s.pth | tee -a "${META_FILE}" || true
echo "" | tee -a "${META_FILE}"

echo "========== FIXED PL DIR ==========" | tee -a "${META_FILE}"
ls -ld data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1 | tee -a "${META_FILE}" || true
echo "" | tee -a "${META_FILE}"

echo "========== CONFIG SNAPSHOT ==========" | tee -a "${META_FILE}"
echo "--- R0_identity_fullrun.yaml ---" | tee -a "${META_FILE}"
cat experiments/pl_refiner/configs/R0_identity_fullrun.yaml | tee -a "${META_FILE}"
echo "" | tee -a "${META_FILE}"
echo "--- R2_class_balanced_fullrun.yaml ---" | tee -a "${META_FILE}"
cat experiments/pl_refiner/configs/R2_class_balanced_fullrun.yaml | tee -a "${META_FILE}"
echo "" | tee -a "${META_FILE}"

echo "========== GIT SNAPSHOT ==========" | tee -a "${META_FILE}"
git rev-parse HEAD 2>/dev/null | tee -a "${META_FILE}" || true
git status --short 2>/dev/null | tee -a "${META_FILE}" || true
echo "" | tee -a "${META_FILE}"

echo "========== START TRAINING =========="
echo "Log file: ${LOG_FILE}"
echo "Meta file: ${META_FILE}"

nohup env GPU=0 SEED=1 NUM_EPOCHS=1 RUNS="R0 R2" \
  bash experiments/pl_refiner/scripts/run_pl_refiner_fullrun.sh \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Started R0/R2 accelerated full-run"
echo "PID: ${PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Meta file: ${META_FILE}"
