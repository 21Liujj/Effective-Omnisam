#!/usr/bin/env bash
# ==============================================================================
# TeCMA 主实验运行脚本 (Phase 3 / Task 3-1)
#
# 为论文主结果表生成可复现实验。一条命令跑一个 config。
#
# 用法:
#   bash experiments/tecma/scripts/run_main_experiments.sh baseline_omnisam
#   bash experiments/tecma/scripts/run_main_experiments.sh A_Dtrain_Etrain
#
# 可选环境变量覆盖 (均有默认值):
#   DATASET=Cityscapes  CLASS_SET=13  BACKBONE=sam2_s
#   GPU_ID=0  SEED=1  PROFILE_MODE=1
#   QUICK_IMAGES=2000  NUM_EPOCHS=1  TRAIN_MAX_ITERS=0  BATCH_SIZE=4
#   OUTPUT_ROOT=experiments/tecma/runs
#   SRC_CKPT=...  TGT_CKPT=...  DATA_ROOT=data
#   DRY_RUN=1   # 只打印命令, 不执行
#   BACKGROUND=1  # nohup 后台运行, 立即返回 (PID/日志写入 runs/_bg_logs/)
#
# 输出 (每个实验独立目录, 带 config 名 + seed + 时间戳):
#   <OUTPUT_ROOT>/<CONFIG>_seed<SEED>_<STAMP>/
#     ├── checkpoints/<DATASET>_<CLASS_SET>cls/<BACKBONE>/
#     │     ├── best_iou*.pth            (checkpoint, mIoU 编码在文件名)
#     │     ├── last_<BACKBONE>_<CLS>cls.pth
#     │     ├── metrics.jsonl            (mIoU / BFScore / mAcc / PixAcc / FWIoU / mDice)
#     │     └── metrics/
#     │           ├── train_timing.jsonl  (profile: D/E token, attn time, gpu_mem)
#     │           └── pl_timing.jsonl      (PL 阶段 profile)
#     ├── run.log                        (train + evaluation 全量日志)
#     └── run_summary.json               (本次运行汇总: config/seed/路径/关键指标)
# ==============================================================================
set -euo pipefail

# ---- 0. 解析位置参数: config 名 ----
CONFIG_NAME="${1:-}"
if [[ -z "${CONFIG_NAME}" ]]; then
  echo "用法: bash $0 <config_name>"
  echo "可选 config: baseline_omnisam | A_Dtrain_Etrain"
  echo "(也可传入 experiments/tecma/configs/ 下任意 *.yaml 的名字)"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # experiments/tecma/scripts -> repo root
cd "${ROOT}"

CONFIG_FILE="experiments/tecma/configs/${CONFIG_NAME}.yaml"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] 找不到 config: ${CONFIG_FILE}"
  echo "现有 config:"; ls experiments/tecma/configs/*.yaml
  exit 1
fi

# ---- 1. 清晰的变量区 ----
DATASET="${DATASET:-Cityscapes}"
CLASS_SET="${CLASS_SET:-13}"
BACKBONE="${BACKBONE:-sam2_s}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-1}"
PROFILE_MODE="${PROFILE_MODE:-1}"      # 1=收集 profile jsonl (Task 2-2A 字段始终写入; 此处用于文档化/标记)
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/tecma/runs}"

DATA_ROOT="${DATA_ROOT:-data}"
SRC_CKPT="${SRC_CKPT:-/root/autodl-tmp/omnisam_s.pth}"
TGT_CKPT="${TGT_CKPT:-/root/autodl-tmp/omnisam_s.pth}"

# 训练规模 (默认论文主实验规模: 2000 PL 子集, 跑满 1 epoch)
QUICK_IMAGES="${QUICK_IMAGES:-2000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
TRAIN_MAX_ITERS="${TRAIN_MAX_ITERS:-0}"   # 0 = 不截断 (跑满一个 epoch)
BATCH_SIZE="${BATCH_SIZE:-4}"
PL_BATCH_SIZE="${PL_BATCH_SIZE:-1}"
PL_NUM_WORKERS="${PL_NUM_WORKERS:-2}"

AMP="${AMP:-1}"
TF32="${TF32:-1}"
DRY_RUN="${DRY_RUN:-0}"

# 复用 canonical pseudo labels (clean ablation 用): 若设置则透传 --reuse_pl_dir,
# 跳过 PL 生成, 直接复用同一套伪标签。为空则不传, 由 run_adaptation.py 自行生成/探测。
REUSE_PL_DIR="${REUSE_PL_DIR:-}"

# ---- 2. 输出目录 (带 config 名 + seed + 时间戳, 防覆盖) ----
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${CONFIG_NAME}_seed${SEED}_${STAMP}"
PD_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
RUN_LOG="${PD_ROOT}/run.log"
SUMMARY_JSON="${PD_ROOT}/run_summary.json"

# ---- 2b. 后台模式 (BACKGROUND=1): 自我 detach, 立即返回 ----
BACKGROUND="${BACKGROUND:-0}"
if [[ "${BACKGROUND}" == "1" && "${_TECMA_DETACHED:-0}" != "1" ]]; then
  BG_LOG_DIR="${OUTPUT_ROOT}/_bg_logs"
  mkdir -p "${BG_LOG_DIR}"
  BOOT_LOG="${BG_LOG_DIR}/${RUN_TAG}.boot.log"
  PID_FILE="${BG_LOG_DIR}/${RUN_TAG}.pid"
  SELF="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
  echo "[bg] 后台启动: ${CONFIG_NAME} (seed=${SEED})"
  echo "[bg] 引导日志 : ${BOOT_LOG}"
  echo "[bg] 训练日志 : ${RUN_LOG}"
  echo "[bg] PID 文件 : ${PID_FILE}"
  STAMP="${STAMP}" _TECMA_DETACHED=1 \
    nohup bash "${SELF}" "${CONFIG_NAME}" >"${BOOT_LOG}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "[bg] 已在后台运行, PID=$(cat "${PID_FILE}")"
  echo "[bg] 查看进度: tail -f ${BOOT_LOG}"
  exit 0
fi


# 防覆盖: 目录已存在则拒绝 (时间戳已基本保证唯一)
if [[ -e "${PD_ROOT}" ]]; then
  echo "[ERROR] 输出目录已存在, 拒绝覆盖: ${PD_ROOT}"
  echo "        请改 STAMP 或删除该目录后重试。"
  exit 1
fi

# ---- 3. PL 固定子集 (可复现) ----
SUBSET_DIR="${OUTPUT_ROOT}/_subsets"
INPUT_LIST="${INPUT_LIST:-dataset/densepass_list/train.txt}"
PL_SUBSET_FILE="${PL_SUBSET_FILE:-${SUBSET_DIR}/${DATASET}_${CLASS_SET}cls_seed${SEED}_n${QUICK_IMAGES}.txt}"

# ---- 4. flag 拼装 ----
amp_flag="--no-amp";  [[ "${AMP}"  == "1" || "${AMP}"  == "true" ]] && amp_flag="--amp"
tf32_flag="--no-tf32"; [[ "${TF32}" == "1" || "${TF32}" == "true" ]] && tf32_flag="--tf32"

train_iters_arg=""
if [[ "${TRAIN_MAX_ITERS}" != "0" && -n "${TRAIN_MAX_ITERS}" ]]; then
  train_iters_arg="--train_max_iters ${TRAIN_MAX_ITERS}"
fi

echo "=============================================================="
echo " TeCMA 主实验: ${CONFIG_NAME}"
echo "   config       = ${CONFIG_FILE}"
echo "   dataset      = ${DATASET} (${CLASS_SET}cls), backbone=${BACKBONE}"
echo "   seed         = ${SEED}, gpu=${GPU_ID}, profile=${PROFILE_MODE}"
echo "   PL images    = ${QUICK_IMAGES}, epochs=${NUM_EPOCHS}, train_max_iters=${TRAIN_MAX_ITERS}"
echo "   output dir   = ${PD_ROOT}"
echo "   run log      = ${RUN_LOG}"
echo "=============================================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] 仅打印, 不执行 (不创建输出目录)。"
  mkdir -p "${SUBSET_DIR}"
else
  mkdir -p "${PD_ROOT}" "${SUBSET_DIR}"
fi

# ---- 5. 构建固定 PL 子集 (若不存在) ----
build_subset_cmd=(
  python "${ROOT}/scripts/time_ablation/build_fixed_subset.py"
  --input_list "${INPUT_LIST}"
  --output_list "${PL_SUBSET_FILE}"
  --num_images "${QUICK_IMAGES}"
  --offset 0 --shuffle --seed "${SEED}"
)
if [[ ! -f "${PL_SUBSET_FILE}" ]]; then
  echo "[subset] 生成固定子集: ${PL_SUBSET_FILE}"
  if [[ "${DRY_RUN}" != "1" ]]; then "${build_subset_cmd[@]}"; else echo "  ${build_subset_cmd[*]}"; fi
else
  echo "[subset] 复用已有子集: ${PL_SUBSET_FILE}"
fi

# ---- 6. 主训练命令 ----
train_cmd=(
  env
  WANDB_MODE=offline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
  python run_adaptation.py
    --config "${CONFIG_FILE}"
    --seed "${SEED}"
    --dataset "${DATASET}"
    --class_set "${CLASS_SET}"
    --backbone "${BACKBONE}"
    --data_root "${DATA_ROOT}"
    --pd_root "${PD_ROOT}"
    --src_ckpt "${SRC_CKPT}"
    --tgt_ckpt "${TGT_CKPT}"
    ${amp_flag} ${tf32_flag}
    --num_epochs "${NUM_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --pl_batch_size "${PL_BATCH_SIZE}"
    --pl_num_workers "${PL_NUM_WORKERS}"
    --no-pl_shuffle
    --pl_subset_file "${PL_SUBSET_FILE}"
    --warmup_pl "${QUICK_IMAGES}"
    --iters_pl "${QUICK_IMAGES}"
    ${train_iters_arg}
    --no-use_text_proto
    --text_proto_path ""
    --lambda_txt 0.0
    --weight_txt_align 0.0
    --no-use_text_gate_ssl
    --no-use_wandb
    --no-use_tensorboard
)

# ---- 6b. 复用 canonical pseudo labels (--reuse_pl_dir 透传) ----
if [[ -n "${REUSE_PL_DIR}" ]]; then
  if [[ ! -d "${REUSE_PL_DIR}" ]]; then
    echo "[ERROR] REUSE_PL_DIR does not exist: ${REUSE_PL_DIR}" >&2
    exit 2
  fi
  train_cmd+=(--reuse_pl_dir "${REUSE_PL_DIR}")
  echo "[reuse-pl] using canonical pseudo labels: ${REUSE_PL_DIR}"
  if [[ "${CONFIG_NAME}" == "A_Dtrain_Etrain_PLon" ]]; then
    echo "[WARN] A_Dtrain_Etrain_PLon should NOT use REUSE_PL_DIR, because it is designed to test PL-stage compression/sparse retrieval. Passing --reuse_pl_dir will invalidate the PL-on comparison." >&2
  fi
else
  echo "[reuse-pl] disabled; pseudo labels will be generated or auto-detected by run_adaptation.py"
fi

echo "[train] 命令:"
printf '  %s\n' "${train_cmd[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] 结束 (未执行训练)。"
  exit 0
fi

# 同时写屏幕 + run.log
"${train_cmd[@]}" 2>&1 | tee "${RUN_LOG}"
TRAIN_RC="${PIPESTATUS[0]}"

# ---- 7. 汇总 run_summary.json ----
SAVE_DIR="${PD_ROOT}/checkpoints/${DATASET}_${CLASS_SET}cls/${BACKBONE}"
METRICS_JSONL="${SAVE_DIR}/metrics.jsonl"
TRAIN_TIMING="${SAVE_DIR}/metrics/train_timing.jsonl"
PL_TIMING="${SAVE_DIR}/metrics/pl_timing.jsonl"

python - "$SUMMARY_JSON" "$CONFIG_NAME" "$CONFIG_FILE" "$DATASET" "$CLASS_SET" "$BACKBONE" \
        "$SEED" "$GPU_ID" "$PROFILE_MODE" "$STAMP" "$PD_ROOT" "$RUN_LOG" \
        "$METRICS_JSONL" "$TRAIN_TIMING" "$PL_TIMING" "$TRAIN_RC" <<'PY_SUMMARY_EOF'
import json, os, sys
(out, cfg_name, cfg_file, dataset, class_set, backbone, seed, gpu, profile,
 stamp, pd_root, run_log, metrics_jsonl, train_timing, pl_timing, train_rc) = sys.argv[1:17]

def last_json_line(p):
    try:
        with open(p) as f:
            lines = [l for l in f if l.strip()]
        return json.loads(lines[-1]) if lines else None
    except Exception:
        return None

m  = last_json_line(metrics_jsonl) or {}
tt = last_json_line(train_timing)  or {}

summary = {
    "config_name": cfg_name,
    "config_file": cfg_file,
    "dataset": dataset,
    "class_set": int(class_set),
    "backbone": backbone,
    "seed": int(seed),
    "gpu_id": int(gpu),
    "profile_mode": profile,
    "stamp": stamp,
    "output_dir": pd_root,
    "run_log": run_log,
    "train_return_code": int(train_rc),
    "artifacts": {
        "metrics_jsonl": metrics_jsonl if os.path.exists(metrics_jsonl) else None,
        "train_timing_jsonl": train_timing if os.path.exists(train_timing) else None,
        "pl_timing_jsonl": pl_timing if os.path.exists(pl_timing) else None,
        "best_ckpt": m.get("best_ckpt_path"),
    },
    "metrics": {
        "val_mIoU": m.get("val_mIoU"),
        "best_mIoU": m.get("best_mIoU"),
        "BFScore": (m.get("best_metrics") or {}).get("BFScore", m.get("BFScore")),
        "val_mAcc": m.get("val_mAcc"),
        "val_PixAcc": m.get("val_PixAcc"),
        "val_FWIoU": m.get("val_FWIoU"),
        "val_mDice": m.get("val_mDice"),
    },
    "profile": {
        "step_time": tt.get("step_time"),
        "iter_compute_time": tt.get("iter_compute_time"),
        "memory_attention_time": tt.get("memory_attention_time"),
        "gpu_mem_peak": tt.get("gpu_mem_peak"),
        "memory_tokens_before_final": tt.get("memory_tokens_before_final"),
        "memory_tokens_after_final": tt.get("memory_tokens_after_final"),
        "compression_tokens_before": tt.get("compression_tokens_before"),
        "compression_tokens_after": tt.get("compression_tokens_after"),
        "compression_ratio": tt.get("compression_ratio"),
        "sparse_tokens_before": tt.get("sparse_tokens_before"),
        "sparse_tokens_after": tt.get("sparse_tokens_after"),
        "sparse_keep_ratio": tt.get("sparse_keep_ratio"),
        "memory_tokens_scope": tt.get("memory_tokens_scope"),
    },
}
with open(out, "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[summary] 写入 {out}")
print(json.dumps(summary["metrics"], ensure_ascii=False))
PY_SUMMARY_EOF

echo "=============================================================="
echo "[done] ${CONFIG_NAME} 完成 (train_rc=${TRAIN_RC})"
echo "  输出目录 : ${PD_ROOT}"
echo "  汇总     : ${SUMMARY_JSON}"
echo "=============================================================="
exit "${TRAIN_RC}"
