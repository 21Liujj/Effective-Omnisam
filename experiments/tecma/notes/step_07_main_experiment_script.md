# Task 3-1 主实验运行脚本 — step_07

> 为论文主结果表生成可复现运行脚本。本步骤**只生成脚本，不运行完整实验**。
> Generated: 2026-06-15

---

## 1. 交付物

| 文件 | 说明 |
|---|---|
| `experiments/tecma/scripts/run_main_experiments.sh` | 主实验运行脚本，一条命令跑一个 config |

已通过：`bash -n` 语法检查 + 两个 config 的 `DRY_RUN` 验证（命令正确生成、未启动训练、未建残留目录）。

---

## 2. 如何运行（验收：复制一条命令即可）

在仓库根目录 `/root/autodl-tmp/OmniSAM-main` 下：

**跑 baseline：**
```bash
bash experiments/tecma/scripts/run_main_experiments.sh baseline_omnisam
```

**跑 final（A+D_train+E_train）：**
```bash
bash experiments/tecma/scripts/run_main_experiments.sh A_Dtrain_Etrain
```

就这两条。其余参数都有默认值。

> 建议先 dry-run 看一眼将要执行的命令（不会训练、不建目录）：
> ```bash
> DRY_RUN=1 bash experiments/tecma/scripts/run_main_experiments.sh baseline_omnisam
> ```

---

## 3. 脚本变量（全部可用环境变量覆盖）

| 变量 | 默认值 | 含义 |
|---|---|---|
| `DATASET` | `Cityscapes` | 数据集 |
| `CLASS_SET` | `13` | 类别集 |
| `BACKBONE` | `sam2_s` | 主干 |
| `GPU_ID` | `0` | GPU 编号 |
| `SEED` | `1` | 随机种子（也进输出目录名） |
| `PROFILE_MODE` | `1` | 收集 profile jsonl 标记（见 §6） |
| `OUTPUT_ROOT` | `experiments/tecma/runs` | 输出根目录 |
| `QUICK_IMAGES` | `2000` | PL 子集大小（论文主实验规模） |
| `NUM_EPOCHS` | `1` | 训练 epoch 数 |
| `TRAIN_MAX_ITERS` | `0` | 0=跑满一个 epoch；>0 截断（调试用） |
| `BATCH_SIZE` | `4` | 训练 batch |
| `SRC_CKPT` / `TGT_CKPT` | `/root/autodl-tmp/omnisam_s.pth` | 源/目标初始权重 |
| `DATA_ROOT` | `data` | 数据根目录 |
| `STAMP` | 当前时间戳 | 输出目录时间戳后缀 |
| `DRY_RUN` | `0` | 1=只打印命令，不执行、不建目录 |

覆盖示例：
```bash
GPU_ID=1 SEED=2 bash experiments/tecma/scripts/run_main_experiments.sh A_Dtrain_Etrain
```

---

## 4. 输出目录结构（每个实验独立、带 config 名 + seed + 时间戳）

```
experiments/tecma/runs/<CONFIG>_seed<SEED>_<STAMP>/
├── checkpoints/Cityscapes_13cls/sam2_s/
│   ├── best_iou<mIoU>.pth          ← checkpoint（mIoU 编码在文件名）
│   ├── last_sam2_s_13cls.pth       ← 最后一个 epoch 权重
│   ├── resume_sam2_s.pth           ← resume 用
│   ├── metrics.jsonl               ← mIoU / BFScore / mAcc / PixAcc / FWIoU / mDice
│   └── metrics/
│       ├── train_timing.jsonl      ← profile：D/E token、attn time、gpu_mem（Task 2-2A 字段）
│       └── pl_timing.jsonl         ← PL 阶段 profile
├── run.log                         ← train + evaluation 全量日志（tee 屏幕同时落盘）
└── run_summary.json                ← 本次运行汇总（脚本自动生成，见 §5）
```

输出目录名带 `config + seed + 时间戳`，**天然防覆盖**；若目录已存在脚本会直接报错退出。
固定 PL 子集落在 `experiments/tecma/runs/_subsets/`，按 `dataset_classset_seed_n` 命名，跨 config 复用、保证可复现。

任务要求的 6 项输出对应关系：
| 要求 | 落盘位置 |
|---|---|
| checkpoint | `checkpoints/.../best_iou*.pth`、`last_*.pth` |
| train log | `run.log` |
| evaluation log | `run.log`（验证阶段同写）+ `metrics.jsonl` |
| mIoU | `metrics.jsonl.val_mIoU` / `run_summary.json.metrics.val_mIoU` |
| BFScore | `metrics.jsonl`（best_metrics.BFScore）/ `run_summary.json.metrics.BFScore` |
| profile jsonl | `metrics/train_timing.jsonl`、`metrics/pl_timing.jsonl` |
| summary json | `run_summary.json` |

---

## 5. run_summary.json（脚本训练结束后自动生成）

脚本在训练结束后读取 `metrics.jsonl` 末行 + `train_timing.jsonl` 末行，汇总成一个文件，便于直接做主结果表：

```json
{
  "config_name": "A_Dtrain_Etrain",
  "config_file": "experiments/tecma/configs/A_Dtrain_Etrain.yaml",
  "dataset": "Cityscapes", "class_set": 13, "backbone": "sam2_s",
  "seed": 1, "gpu_id": 0, "profile_mode": "1", "stamp": "...",
  "output_dir": "...", "run_log": "...", "train_return_code": 0,
  "artifacts": { "metrics_jsonl": "...", "train_timing_jsonl": "...",
                 "pl_timing_jsonl": "...", "best_ckpt": "..." },
  "metrics": { "val_mIoU": ..., "best_mIoU": ..., "BFScore": ...,
               "val_mAcc": ..., "val_PixAcc": ..., "val_FWIoU": ..., "val_mDice": ... },
  "profile": { "step_time": ..., "iter_compute_time": ..., "memory_attention_time": ...,
               "gpu_mem_peak": ..., "memory_tokens_before_final": ..., "memory_tokens_after_final": ...,
               "compression_tokens_before": ..., "compression_tokens_after": ..., "compression_ratio": ...,
               "sparse_tokens_before": ..., "sparse_tokens_after": ..., "sparse_keep_ratio": ...,
               "memory_tokens_scope": "per_attention_call_all_chunks_concat" } }
```

baseline 时 `compression_*` / `sparse_*` 为 `null`（与 Task 2-3 sanity 验证一致）。

---

## 6. 关于 PROFILE_MODE

代码中**没有** `--profile_mode` 这个 CLI 参数。Task 2-2A 的 profile 字段（D/E token、attn time、gpu_mem）是**始终写入** `train_timing.jsonl` 的，不需要开关。

因此脚本里的 `PROFILE_MODE`（默认 1）仅作为**文档化标记**写入 `run_summary.json`，记录这次运行期望收集 profile。它不改变训练行为，也不会传给 `run_adaptation.py`。这样做避免凭空发明一个代码里不存在的 flag。

---

## 7. 安全性 / 防覆盖

- 输出目录名含时间戳，且脚本对已存在目录**拒绝写入并退出**，不会覆盖任何已有 checkpoint。
- 不触碰 `data_pd_exp_final_2000/`（旧主结果）和 `experiments/tecma/sanity_runs/`（Task 2-3 产物）。
- `DRY_RUN=1` 时不创建输出目录，仅打印命令。
- 脚本默认**只跑传入的那一个 config**，不会自动批量跑全部。

---

## 8. 验收结论

- ✅ 一条命令跑 baseline：`bash experiments/tecma/scripts/run_main_experiments.sh baseline_omnisam`
- ✅ 一条命令跑 final：`bash experiments/tecma/scripts/run_main_experiments.sh A_Dtrain_Etrain`
- ✅ 每个实验独立目录，带 config 名 + seed + 时间戳
- ✅ 6 项输出（checkpoint / train log / eval log / mIoU / BFScore / profile jsonl / summary json）全部覆盖
- ✅ 本步骤未启动完整训练（仅 `bash -n` + `DRY_RUN` 验证）

> 下一步（Task 3-2）：真正执行这两条命令跑主实验，产出论文主结果表数据。由用户在确认 GPU 空闲时执行（单 epoch、2000 PL，参考既有 final_2000 实验耗时约 30–45 min/config）。


---

## 9. 运行分工规则（Task 3-1b 新增）

**Claude 不运行完整训练，也不长时间监控日志。** 明确分工如下：

1. **Claude 不运行完整训练**：不执行 `run_main_experiments.sh`、`run_adaptation.py` 等长耗时命令。
2. **Claude 不长时间监控日志**：不后台起跑、不挂着 tail/Monitor 训练日志。
3. **用户手动运行训练命令**：所有训练/评测命令由用户自己在服务器终端执行。
4. **用户只把关键日志或 summary 粘贴回来**：如 `run_summary.json`、`metrics.jsonl`、`train_timing.jsonl` 末行，或报错片段。
5. **Claude 只负责解析结果、定位错误、生成表格或下一步脚本**：基于贴回的内容做分析，把结果登记到 `experiments/tecma/tables/main_results_registry.md`。

> 例外：秒级~分钟级的功能性 sanity（仅用于验证字段能否落盘，如 Task 2-3）可由 Claude 跑；但能避免则避免。

主结果登记文件：`experiments/tecma/tables/main_results_registry.md`（跑完一个实验后，把 summary 贴回，Claude 负责追加一行）。
