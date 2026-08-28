#!/usr/bin/env bash
# ==============================================================================
# TeCMA 串行队列运行脚本
#
# 单 GPU 上把多个 config 串行跑完 (一个结束再跑下一个), 整个队列在后台运行,
# 命令立即返回。每个 config 仍各自生成独立的带时间戳输出目录。
#
# 用法:
#   SEED=1 bash experiments/tecma/scripts/run_queue.sh A_Dtrain A_Etrain Dtrain_Etrain A_Dtrain_Etrain_PLon
#
# 环境变量:
#   SEED=1            传给每个实验 (默认 1)
#   FG=1              前台运行 (不 detach), 方便调试
#   其余变量 (DATASET/CLASS_SET/GPU_ID/...) 会被 run_main_experiments.sh 继承
#
# 输出:
#   队列日志 : experiments/tecma/runs/_bg_logs/queue_seed<SEED>_<STAMP>.log
#   队列 PID : experiments/tecma/runs/_bg_logs/queue_seed<SEED>_<STAMP>.pid
#   各实验目录: experiments/tecma/runs/<CONFIG>_seed<SEED>_<STAMP>/
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT}"

if [[ $# -lt 1 ]]; then
  echo "用法: [SEED=1] bash $0 <config1> [config2] ..."
  echo "示例: SEED=1 bash $0 A_Dtrain A_Etrain Dtrain_Etrain A_Dtrain_Etrain_PLon"
  exit 1
fi

CONFIGS=("$@")
SEED="${SEED:-1}"
FG="${FG:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/experiments/tecma/runs}"
BG_LOG_DIR="${OUTPUT_ROOT}/_bg_logs"
QSTAMP="${QSTAMP:-$(date +%Y%m%d_%H%M%S)}"
QUEUE_LOG="${BG_LOG_DIR}/queue_seed${SEED}_${QSTAMP}.log"
QUEUE_PID="${BG_LOG_DIR}/queue_seed${SEED}_${QSTAMP}.pid"
RUNNER="${SCRIPT_DIR}/run_main_experiments.sh"

# ---- 后台 detach (FG=1 跳过) ----
if [[ "${FG}" != "1" && "${_TECMA_QUEUE_DETACHED:-0}" != "1" ]]; then
  mkdir -p "${BG_LOG_DIR}"
  SELF="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
  echo "[queue] 单 GPU 串行运行 ${#CONFIGS[@]} 个实验: ${CONFIGS[*]}"
  echo "[queue] seed=${SEED}"
  echo "[queue] REUSE_PL_DIR=${REUSE_PL_DIR:-<empty>}"
  echo "[queue] 队列日志: ${QUEUE_LOG}"
  echo "[queue] PID 文件: ${QUEUE_PID}"
  QSTAMP="${QSTAMP}" _TECMA_QUEUE_DETACHED=1 \
    nohup bash "${SELF}" "${CONFIGS[@]}" >"${QUEUE_LOG}" 2>&1 &
  echo $! > "${QUEUE_PID}"
  echo "[queue] 已后台运行, PID=$(cat "${QUEUE_PID}")"
  echo "[queue] 查看进度: tail -f ${QUEUE_LOG}"
  exit 0
fi

# ---- detached: 串行执行 ----
echo "===== QUEUE START $(date) ====="
echo "configs : ${CONFIGS[*]}"
echo "seed    : ${SEED}"
echo "runner  : ${RUNNER}"
echo "[queue] REUSE_PL_DIR=${REUSE_PL_DIR:-<empty>}"

fail=0
declare -a results=()
for cfg in "${CONFIGS[@]}"; do
  cfg_file="experiments/tecma/configs/${cfg}.yaml"
  if [[ ! -f "${cfg_file}" ]]; then
    echo "########## [$(date)] SKIP  ${cfg} (找不到 ${cfg_file}) ##########"
    results+=("${cfg}=missing")
    fail=1
    continue
  fi
  echo ""
  echo "########## [$(date)] BEGIN ${cfg} ##########"
  if [[ "${cfg}" == "A_Dtrain_Etrain_PLon" && -n "${REUSE_PL_DIR:-}" ]]; then
    echo "[WARN] A_Dtrain_Etrain_PLon is a PL-on ablation. Do NOT use REUSE_PL_DIR for this config unless you intentionally want to invalidate the PL-on comparison." >&2
  fi
  if SEED="${SEED}" bash "${RUNNER}" "${cfg}"; then
    echo "########## [$(date)] DONE  ${cfg} (ok) ##########"
    results+=("${cfg}=ok")
  else
    rc=$?
    echo "########## [$(date)] FAIL  ${cfg} (rc=${rc}) ##########"
    results+=("${cfg}=fail(${rc})")
    fail=1
  fi
done

echo ""
echo "===== QUEUE END $(date) ====="
echo "结果汇总:"
for r in "${results[@]}"; do echo "  - ${r}"; done
echo "(fail=${fail})"
exit "${fail}"
