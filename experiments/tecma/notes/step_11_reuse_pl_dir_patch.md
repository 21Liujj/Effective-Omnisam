# Step 11 — REUSE_PL_DIR 透传补丁 (clean ablation 修复)

> 日期: 2026-06-16
> 目标: 让 train-only 消融能真正复用同一套 canonical pseudo labels (PL), 升级 Phase 4 v0 → clean ablation。
> 限制: 只改 shell 脚本, 不改 run_adaptation.py / model / engine / trainer / plugin。本次未跑训练, 仅 bash -n + DRY_RUN 验证。

---

## 1. 修改了哪些文件

| 文件 | 操作 |
|---|---|
| `experiments/tecma/scripts/run_main_experiments.sh` | 修改: 新增 `REUSE_PL_DIR` 变量 + 透传 `--reuse_pl_dir` 逻辑 |
| `experiments/tecma/scripts/run_queue.sh` | 修改: 新增队列级 `REUSE_PL_DIR` 日志 + PLon 警告 |
| `run_main_experiments.sh.bak_reusepl_20260616_101328` | 备份 (修改前原文) |
| `run_queue.sh.bak_reusepl_20260616_101328` | 备份 (修改前原文) |
| `experiments/tecma/notes/step_11_reuse_pl_dir_patch.md` | 新增: 本报告 |

未改动: `run_adaptation.py`、任何 model/engine/trainer/plugin 代码、任何已有实验结果 (`data_pd_exp_final_2000/`、`runs/`)。

---

## 2. 每个文件具体改了什么

### 2.1 run_main_experiments.sh

**(a) 变量区 (DRY_RUN 之后) 新增声明:**

```bash
# 复用 canonical pseudo labels (clean ablation 用): 若设置则透传 --reuse_pl_dir,
# 跳过 PL 生成, 直接复用同一套伪标签。为空则不传, 由 run_adaptation.py 自行生成/探测。
REUSE_PL_DIR="${REUSE_PL_DIR:-}"
```

**(b) `train_cmd` 数组构建完成后、打印命令之前 (新增 step 6b)**, 用数组 append 方式安全透传 (路径含空格也安全):

```bash
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
```

要点:
- 使用 bash 数组 `train_cmd+=(...)`, 与脚本原有 `train_cmd=( env ... python run_adaptation.py ... )` 风格一致, 不存在路径空格问题。
- `--reuse_pl_dir` 真正进入 `run_adaptation.py` 命令 (不只是写 log)。
- `REUSE_PL_DIR` 为空 → 不 append flag, 打印 disabled 提示, 行为与补丁前完全一致 (向后兼容)。
- 非空但目录不存在 → `[ERROR]` + `exit 2`, 提前失败避免误跑。

### 2.2 run_queue.sh

**(a) detach 前的 header 增加一行可见日志:**

```bash
echo "[queue] REUSE_PL_DIR=${REUSE_PL_DIR:-<empty>}"
```

**(b) detached 执行段的队列 header 同样打印:**

```bash
echo "[queue] REUSE_PL_DIR=${REUSE_PL_DIR:-<empty>}"
```

**(c) 主循环内, 每个 config 调用 runner 之前, 针对 PLon 打强警告:**

```bash
if [[ "${cfg}" == "A_Dtrain_Etrain_PLon" && -n "${REUSE_PL_DIR:-}" ]]; then
  echo "[WARN] A_Dtrain_Etrain_PLon is a PL-on ablation. Do NOT use REUSE_PL_DIR for this config unless you intentionally want to invalidate the PL-on comparison." >&2
fi
```

要点:
- 环境变量 `REUSE_PL_DIR` 通过 `bash "${RUNNER}" "${cfg}"` 自然继承 (queue 与 runner 同进程环境), 无需显式 export, 已由 DRY_RUN 验证透传成功。
- 仅警告, **不自动清空 REUSE_PL_DIR, 不阻止运行**。

---

## 3. 为什么之前 v0 消融没有复用 PL

Phase 4 v0 的运行日志里, 传给 `run_adaptation.py` 的命令中**根本没有 `--reuse_pl_dir`**。原因: 当时的 `run_main_experiments.sh` 在构建 `train_cmd` 数组时, 完全没有读取 `REUSE_PL_DIR` 环境变量, 也没有任何 append `--reuse_pl_dir` 的分支。

即使运行者在命令行设置了 `REUSE_PL_DIR=...`, 该变量只是一个未被任何代码引用的 shell 环境变量, 被静默忽略。结果: 每个 train-only 消融都走 `run_adaptation.py` 的默认路径 (`reuse_pl_dir=""`), 各自重新生成了一套 pseudo labels。

这使得 4 个 train-only 消融的 PL 互不相同, 引入了 PL 随机性这一额外混杂变量, 因此只能作为 Phase 4 v0, 不能作为 clean ablation。

(`run_adaptation.py` 本身一直支持该 flag: dataclass 字段 `reuse_pl_dir`, default `""`; 主循环 `if getattr(args, "reuse_pl_dir", ""): engine.save_dir = args.reuse_pl_dir ... 跳过生成`。所以问题纯粹在 shell 层, 无需改 Python。)

---

## 4. 现在如何确认 REUSE_PL_DIR 已透传成功

`bash -n` 语法检查: 两个脚本均通过 (rc=0)。

DRY_RUN 验证 (FG=1 前台, 不 detach, 不训练):

```bash
PL_DIR=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
DRY_RUN=1 FG=1 REUSE_PL_DIR=$PL_DIR SEED=1 \
  bash experiments/tecma/scripts/run_queue.sh \
  A_ccpm A_Dtrain A_Etrain Dtrain_Etrain A_Dtrain_Etrain
```

5 个 config 打印的 `run_adaptation.py` 命令**末尾全部包含**:

```
--reuse_pl_dir data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
```

并且每个都打印 `[reuse-pl] using canonical pseudo labels: ...`; 队列头打印 `[queue] REUSE_PL_DIR=...`。

补充边界验证:
- **未设置 REUSE_PL_DIR**: 打印 `[reuse-pl] disabled; ...`, 命令中无 `--reuse_pl_dir` (向后兼容)。
- **路径不存在** (`REUSE_PL_DIR=/no/such/dir`): 打印 `[ERROR] REUSE_PL_DIR does not exist: ...`, 退出码 2。

---

## 5. 为什么 A_Dtrain_Etrain_PLon 不应使用 REUSE_PL_DIR

`A_Dtrain_Etrain_PLon` 的实验目的就是测量 D (memory compression) / E (sparse retrieval) 在 **PL 生成阶段** (`apply_in_pl: true`) 的效果。它需要用插件**自己生成**的 pseudo labels 才有意义。

如果给它传 `--reuse_pl_dir`, 训练会直接复用 baseline 的 canonical PL, 完全跳过 PL 生成阶段 —— 这恰好把它要测的东西短路掉了, PL-on 对比随之失效。

因此脚本会发警告 (queue 级 + runner 级各一条), 但**遵循"提醒而非禁止"**: 若用户坚持传入, 命令里仍会带 `--reuse_pl_dir` (这是用户主动行为, 不建议真实这样跑)。clean ablation 队列**不应包含** PLon; PLon 已在 v0 跑过, 是否重跑由用户决定。

---

## 6. 推荐下一步 clean ablation 运行命令

```bash
cd /root/autodl-tmp/OmniSAM-main

PL_DIR=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1

REUSE_PL_DIR=$PL_DIR SEED=1 bash experiments/tecma/scripts/run_queue.sh \
  baseline_omnisam A_ccpm A_Dtrain A_Etrain Dtrain_Etrain A_Dtrain_Etrain
```

- 不含 `A_Dtrain_Etrain_PLon` (它不能复用 PL)。
- 6 个 config 全部复用同一套 canonical PL → 干净的 train-only 消融。
- 默认后台 detach; 加 `FG=1` 可前台调试。真实跑前可先加 `DRY_RUN=1` 复核命令。
