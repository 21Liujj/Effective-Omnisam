#!/usr/bin/env bash
# ==============================================================================
# TeCMA Phase 5 — Token-budget speed/accuracy trade-off runner (thin wrapper)
#
# Runs one or more token-budget configs SERIALLY by delegating to the existing
# run_main_experiments.sh. No new training logic is duplicated here: this wrapper
# only (1) isolates output under runs/token_budget/, (2) enforces canonical-PL
# reuse via REUSE_PL_DIR, and (3) prints each delegated command with NO arg
# concatenation (printf '%q').
#
# Usage:
#   bash experiments/tecma/scripts/run_token_budget_experiments.sh A_Etrain_keep075
#   bash experiments/tecma/scripts/run_token_budget_experiments.sh A_Etrain_keep075 A_Etrain_keep025
#
# Dry-run (prints commands only, runs nothing):
#   PL_DIR=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
#   DRY_RUN=1 REUSE_PL_DIR=$PL_DIR SEED=1 \
#     bash experiments/tecma/scripts/run_token_budget_experiments.sh A_Etrain_keep075 A_Etrain_keep025
#
# Env vars (forwarded to run_main_experiments.sh by export):
#   SEED=1            seed for every config (default 1)
#   REUSE_PL_DIR=...  canonical PL dir; STRONGLY recommended (warns if empty)
#   DRY_RUN=1         print only, run nothing
#   BACKGROUND=1      detach the whole queue with nohup, return immediately
#   DATASET/CLASS_SET/BACKBONE/GPU_ID/... inherited by the runner unchanged
#
# Output (never overwrites clean ablation in runs/<CONFIG>_...):
#   experiments/tecma/runs/token_budget/<CONFIG>_seed<SEED>_<STAMP>/
#     +-- run.log  run_summary.json
#     +-- checkpoints/.../{metrics.jsonl, metrics/train_timing.jsonl, ...}
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

if [[ $# -lt 1 ]]; then
  echo "用法: [SEED=1] [REUSE_PL_DIR=...] [DRY_RUN=1] bash $0 <config1> [config2] ..."
  echo "示例: DRY_RUN=1 REUSE_PL_DIR=\$PL_DIR SEED=1 bash $0 A_Etrain_keep075 A_Etrain_keep025"
  exit 1
fi

CONFIGS=("$@")
SEED="${SEED:-1}"
DRY_RUN="${DRY_RUN:-0}"
BACKGROUND="${BACKGROUND:-0}"
REUSE_PL_DIR="${REUSE_PL_DIR:-}"
RUNNER="${SCRIPT_DIR}/run_main_experiments.sh"

# Isolate token-budget outputs in their own subtree (prevents overwriting clean ablation)
TB_OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/tecma/runs/token_budget}"

# ---- sanity: every config must exist before doing anything ----
for cfg in "${CONFIGS[@]}"; do
  cfg_file="experiments/tecma/configs/${cfg}.yaml"
  if [[ ! -f "${cfg_file}" ]]; then
    echo "[ERROR] config not found: ${cfg_file}" >&2
    echo "        existing configs:" >&2
    ls experiments/tecma/configs/*.yaml >&2
    exit 1
  fi
done

# ---- canonical-PL guard ----
if [[ -z "${REUSE_PL_DIR}" ]]; then
  echo "[WARN] REUSE_PL_DIR is empty. Token-budget experiments SHOULD reuse the canonical" >&2
  echo "       pseudo labels so all points share an identical PL set. Without it, each run" >&2
  echo "       would regenerate PL and the trade-off curve becomes non-comparable." >&2
  echo "       Recommended:" >&2
  echo "         PL_DIR=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1" >&2
  echo "         REUSE_PL_DIR=\$PL_DIR bash $0 ${CONFIGS[*]}" >&2
elif [[ ! -d "${REUSE_PL_DIR}" ]]; then
  echo "[ERROR] REUSE_PL_DIR does not exist: ${REUSE_PL_DIR}" >&2
  exit 2
else
  echo "[reuse-pl] canonical PL: ${REUSE_PL_DIR}"
fi

echo "=============================================================="
echo " TeCMA token-budget queue"
echo "   configs       = ${CONFIGS[*]}"
echo "   seed          = ${SEED}"
echo "   output root   = ${TB_OUTPUT_ROOT}"
echo "   reuse_pl_dir  = ${REUSE_PL_DIR:-<empty>}"
echo "   dry_run       = ${DRY_RUN}, background=${BACKGROUND}"
echo "=============================================================="

# ---- background detach (whole queue) ----
if [[ "${BACKGROUND}" == "1" && "${_TECMA_TB_DETACHED:-0}" != "1" ]]; then
  BG_LOG_DIR="${TB_OUTPUT_ROOT}/_bg_logs"
  mkdir -p "${BG_LOG_DIR}"
  QSTAMP="$(date +%Y%m%d_%H%M%S)"
  QUEUE_LOG="${BG_LOG_DIR}/token_budget_seed${SEED}_${QSTAMP}.log"
  QUEUE_PID="${BG_LOG_DIR}/token_budget_seed${SEED}_${QSTAMP}.pid"
  SELF="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
  echo "[bg] detaching token-budget queue -> ${QUEUE_LOG}"
  _TECMA_TB_DETACHED=1 SEED="${SEED}" REUSE_PL_DIR="${REUSE_PL_DIR}" \
    OUTPUT_ROOT="${TB_OUTPUT_ROOT}" DRY_RUN="${DRY_RUN}" \
    nohup bash "${SELF}" "${CONFIGS[@]}" >"${QUEUE_LOG}" 2>&1 &
  echo $! > "${QUEUE_PID}"
  echo "[bg] PID=$(cat "${QUEUE_PID}") ; tail -f ${QUEUE_LOG}"
  exit 0
fi

# ---- serial execution: delegate each config to run_main_experiments.sh ----
fail=0
declare -a results=()
for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "########## [token_budget] BEGIN ${cfg} (seed=${SEED}) ##########"

  # Build the exact delegated invocation as an array, print it with %q (no concatenation).
  delegate_env=(
    "SEED=${SEED}"
    "OUTPUT_ROOT=${TB_OUTPUT_ROOT}"
    "DRY_RUN=${DRY_RUN}"
  )
  [[ -n "${REUSE_PL_DIR}" ]] && delegate_env+=("REUSE_PL_DIR=${REUSE_PL_DIR}")
  delegate_cmd=(env "${delegate_env[@]}" bash "${RUNNER}" "${cfg}")

  echo "[delegate] command:"
  printf '  %q' "${delegate_cmd[@]}"; printf '\n'

  if "${delegate_cmd[@]}"; then
    echo "########## [token_budget] DONE  ${cfg} ##########"
    results+=("${cfg}=ok")
  else
    rc=$?
    echo "########## [token_budget] FAIL  ${cfg} (rc=${rc}) ##########"
    results+=("${cfg}=fail(${rc})")
    fail=1
  fi
done

echo ""
echo "===== token-budget queue END ====="
for r in "${results[@]}"; do echo "  - ${r}"; done
echo "(fail=${fail})"
exit "${fail}"
