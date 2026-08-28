#!/usr/bin/env bash
# experiments/pl_refiner/scripts/run_pl_refiner_fullrun.sh
#
# PseudoLabelRefiner FULL-RUN driver: R0 baseline -> R2 -> R2+R4.
# Each run trains a FULL epoch (train_max_iters=0) and runs epoch-end val.
# All three reuse the SAME fixed base PL dir, so the ONLY variable is the
# refiner's valid_mask refinement (PL pixels are identical across runs).
#
# Reproducibility (per .claude/rules/experiments.md):
#   - seed / GPU / commit / config path / launch cmd recorded into each run dir
#   - fresh timestamped pd_root (never reuse an old dir)
#   - does NOT touch data_pd_exp_final_2000/ checkpoints or PLs (read-only reuse)
#
# IMPORTANT — run order:
#   1) R0_identity (matched baseline)   MUST run first
#   2) R2_class_balanced                run second; compare vs R0
#   3) R2R4_proto_gate                  run third; compare vs R2
# You may run only a subset by editing RUNS below.
#
# Usage (on remote server, in conda env omnisam):
#   cd /root/autodl-tmp/OmniSAM-main
#   bash experiments/pl_refiner/scripts/run_pl_refiner_fullrun.sh
#
# Optional overrides (env vars):
#   GPU=0 SEED=1 NUM_EPOCHS=1 RUNS="R0 R2 R2R4" bash .../run_pl_refiner_fullrun.sh

set -euo pipefail
cd /root/autodl-tmp/OmniSAM-main

# ---- fixed knobs ----
PY=/root/miniconda3/envs/omnisam/bin/python3
GPU="${GPU:-0}"
SEED="${SEED:-1}"
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
DATA_ROOT="${DATA_ROOT:-data}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
REUSE_PL="${REUSE_PL:-data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1}"

# Which runs to execute (space-separated): R0 R2 R2R4
RUNS="${RUNS:-R0 R2 R2R4}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CFG_DIR=experiments/pl_refiner/configs
PD_BASE="${PD_BASE:-experiments/pl_refiner/fullruns}"
NOTE_DIR=experiments/pl_refiner/notes
RES_DIR=experiments/pl_refiner/results
mkdir -p "${PD_BASE}" "${NOTE_DIR}" "${RES_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo NO_GIT)"

run_one() {
  local tag="$1"          # R0_identity / R2_class_balanced / R2R4_proto_gate
  local cfg="$2"
  local pd_root="${PD_BASE}/${tag}_${STAMP}"
  local log="${NOTE_DIR}/fullrun_${tag}_${STAMP}.log"
  local meta="${pd_root}/run_meta.txt"

  mkdir -p "${pd_root}"
  {
    echo "experiment_tag : ${tag}"
    echo "config         : ${cfg}"
    echo "seed           : ${SEED}"
    echo "gpu            : ${GPU}"
    echo "commit         : ${COMMIT}"
    echo "dataset        : ${DATASET} / class_set=${CLASS_SET} / backbone=${BACKBONE}"
    echo "batch_size     : ${BATCH_SIZE}"
    echo "num_epochs     : ${NUM_EPOCHS}"
    echo "train_max_iters: 0 (FULL epoch — val mIoU is a real result)"
    echo "reuse_pl_dir   : ${REUSE_PL}"
    echo "pd_root        : ${pd_root}"
    echo "log            : ${log}"
    echo "stamp          : ${STAMP}"
    echo "started_at     : $(date -Iseconds)"
  } | tee "${meta}"

  echo "=================================================="
  echo "[fullrun] ${tag}"
  echo "  cfg=${cfg}"
  echo "  pd_root=${pd_root}"
  echo "  log=${log}"
  echo "=================================================="

  env CUDA_VISIBLE_DEVICES="${GPU}" \
  "${PY}" run_adaptation.py \
    --config "${cfg}" \
    --data_root "${DATA_ROOT}" \
    --pd_root "${pd_root}" \
    --dataset "${DATASET}" \
    --backbone "${BACKBONE}" \
    --class_set "${CLASS_SET}" \
    --src_ckpt "${SRC_CKPT}" \
    --tgt_ckpt "${TGT_CKPT}" \
    --reuse_pl_dir "${REUSE_PL}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    2>&1 | tee "${log}"

  echo "finished_at    : $(date -Iseconds)" | tee -a "${meta}"
  echo "[done] ${tag}"
}

for r in ${RUNS}; do
  case "${r}" in
    R0)   run_one "R0_identity"       "${CFG_DIR}/R0_identity_fullrun.yaml" ;;
    R2)   run_one "R2_class_balanced" "${CFG_DIR}/R2_class_balanced_fullrun.yaml" ;;
    R2R4) run_one "R2R4_proto_gate"   "${CFG_DIR}/R2R4_proto_gate_fullrun.yaml" ;;
    *)    echo "[WARN] unknown run tag: ${r} (skip)";;
  esac
done

echo ""
echo "=================================================="
echo "  ALL REQUESTED FULL-RUNS DONE (stamp=${STAMP})"
echo "  per-run dir : ${PD_BASE}/<tag>_${STAMP}/"
echo "  per-run log : ${NOTE_DIR}/fullrun_<tag>_${STAMP}.log"
echo "  Compare epoch-end mIoU/mAcc/BFScore: R2 vs R0, then R2R4 vs R2."
echo "=================================================="
