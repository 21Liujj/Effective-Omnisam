#!/usr/bin/env bash
# ==============================================================================
# Multi-seed batch runner for TeCMA experiments (Week 1 / Day 1-4)
#
# Usage:
#   bash experiments/tecma/scripts/run_multiseed_batch.sh baseline_omnisam
#   bash experiments/tecma/scripts/run_multiseed_batch.sh A_Dtrain_Etrain
#
# Env overrides:
#   SEEDS="1 2 3 42 100"  (default)
#   GPU_ID=0
#   SERIAL=1  (wait each seed to finish before next; default)
#   PARALLEL=1  (launch all seeds in background, NOT recommended if GPU mem limited)
# ==============================================================================
set -euo pipefail

CONFIG_NAME="${1:-}"
if [[ -z "${CONFIG_NAME}" ]]; then
  echo "Usage: bash $0 <config_name>"
  echo "Example: bash $0 baseline_omnisam"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

SEEDS=${SEEDS:-"1 2 3 42 100"}
GPU_ID=${GPU_ID:-0}
SERIAL=${SERIAL:-1}
PARALLEL=${PARALLEL:-0}

echo "========================================"
echo "Multi-seed batch: ${CONFIG_NAME}"
echo "Seeds: ${SEEDS}"
echo "GPU: ${GPU_ID}"
echo "Mode: $( [[ ${SERIAL} -eq 1 ]] && echo SERIAL || echo PARALLEL )"
echo "========================================"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_LOG="experiments/tecma/runs/batch_${CONFIG_NAME}_${TIMESTAMP}.log"
mkdir -p experiments/tecma/runs

echo "[$(date)] Batch started" | tee "${BATCH_LOG}"

for SEED in ${SEEDS}; do
  echo "" | tee -a "${BATCH_LOG}"
  echo "[$(date)] === Launching seed ${SEED} ===" | tee -a "${BATCH_LOG}"
  
  if [[ ${PARALLEL} -eq 1 ]]; then
    # Background launch (NOT recommended unless you have multi-GPU)
    SEED=${SEED} GPU_ID=${GPU_ID} bash experiments/tecma/scripts/run_main_experiments.sh "${CONFIG_NAME}" >> "${BATCH_LOG}" 2>&1 &
    PID=$!
    echo "  Launched in background (PID ${PID})" | tee -a "${BATCH_LOG}"
  else
    # Serial: wait for completion
    SEED=${SEED} GPU_ID=${GPU_ID} bash experiments/tecma/scripts/run_main_experiments.sh "${CONFIG_NAME}" 2>&1 | tee -a "${BATCH_LOG}"
    EXIT_CODE=${PIPESTATUS[0]}
    if [[ ${EXIT_CODE} -ne 0 ]]; then
      echo "  [ERROR] Seed ${SEED} failed with exit code ${EXIT_CODE}" | tee -a "${BATCH_LOG}"
      echo "  Stopping batch." | tee -a "${BATCH_LOG}"
      exit ${EXIT_CODE}
    fi
    echo "  [OK] Seed ${SEED} completed" | tee -a "${BATCH_LOG}"
  fi
done

if [[ ${PARALLEL} -eq 1 ]]; then
  echo "" | tee -a "${BATCH_LOG}"
  echo "[$(date)] Waiting for all background jobs..." | tee -a "${BATCH_LOG}"
  wait
fi

echo "" | tee -a "${BATCH_LOG}"
echo "[$(date)] Batch completed" | tee -a "${BATCH_LOG}"
echo "Log: ${BATCH_LOG}"
