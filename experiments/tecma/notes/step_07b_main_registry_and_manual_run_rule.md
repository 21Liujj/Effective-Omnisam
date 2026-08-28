# Task 3-1b 报告 — step_07b 主结果登记 + 人工运行规则

> 收尾任务：补齐主结果登记文件，写入"人工运行、Claude 不跑长实验"规则。
> 本任务不跑训练、不运行 run_main_experiments.sh、不启动 evaluation、不改模型代码。
> Generated: 2026-06-15

---

## 1. main_results_registry.md 是否已存在

**原本不存在**（`experiments/tecma/tables/` 目录此前未建）。本任务**新建**：
- `experiments/tecma/tables/main_results_registry.md`（49 行）

---

## 2. 是否补齐两行主结果

已补齐两行，列头与要求一致（Method / Config / Dataset / Seed / Epoch / mIoU / BFScore / Step Time / Memory Attention Time / Memory Tokens / Peak Memory / Result Path / Status）：

| Method | Config | mIoU | BFScore | Step Time | Mem Attn | Mem Tokens | Peak Mem | Status |
|---|---|---|---|---|---|---|---|---|
| baseline / base_last | baseline_omnisam | 60.26 | 39.97 | 5.27 s/it | 263.0 s | 32768 | 26.82 GB | existing_verified_main_result |
| A_Dtrain_Etrain | A_Dtrain_Etrain | 60.16 | 39.86 | 4.08 s/it | 93.6 s | 4096 | 25.34 GB | existing_verified_main_result |

**数据核对**：mIoU / Step Time / Memory Attention Time / Memory Tokens / Peak Memory 五项已逐一与远端日志（`metrics.jsonl` + `train_timing.jsonl`）核对**全部一致**。

⚠ **BFScore 一处需注意**：两个实验的 `metrics.jsonl` 中 `BFScore` 字段实际为 `None`（该次运行未落盘边界 F 分数）。表中 39.97 / 39.86 采用**用户提供值**登记备查。已在 registry 文件中明确标注此来源差异；若论文要引用 BFScore，建议后续重新评测回填以保证可追溯。

---

## 3. 是否更新 step_07 文档

已更新 `experiments/tecma/notes/step_07_main_experiment_script.md`，新增 **§9 运行分工规则**，含 5 条：
1. Claude 不运行完整训练；
2. Claude 不长时间监控日志；
3. 用户手动运行训练命令；
4. 用户只把关键日志 / summary 粘贴回来；
5. Claude 只负责解析结果、定位错误、生成表格或下一步脚本。

同时把该长期规则更新进 Claude 记忆 `feedback-no-script-execution`（含明确的"写脚本 / 静态检查 / 写解析脚本 / 解析贴回日志"四项分工）。

---

## 4. 是否确认不运行训练

确认。本任务**未**执行任何训练 / 评测 / 长耗时命令：
- 未运行 `run_main_experiments.sh`；
- 未运行 `run_adaptation.py`；
- 未启动 evaluation；
- 未修改模型代码；
- 未覆盖任何已有主结果（仅新建 registry + 追加文档段落）。
- 唯一的远端操作是只读日志（核对数字）+ 写两个文档文件。

---

## 5. 下一步建议

- **Task 3-2（由用户手动跑）**：在 GPU 空闲时执行
  ```bash
  bash experiments/tecma/scripts/run_main_experiments.sh baseline_omnisam
  bash experiments/tecma/scripts/run_main_experiments.sh A_Dtrain_Etrain
  ```
  预计 baseline ≈ 2.7h、A_Dtrain_Etrain ≈ 2.5h（PL 生成各占 ~2h 为大头）。跑完把 `run_summary.json` 贴回，Claude 追加新行到 registry。
- **BFScore 回填**：若主结果表需要 BFScore，建议在评测阶段确认其计算与落盘路径（当前 `metrics.jsonl` 为 None），避免论文数字无源。
- registry 中新跑结果的 Status 用 `pending` → 核对后改 `verified`，与现有两行的 `existing_verified_main_result` 区分。
