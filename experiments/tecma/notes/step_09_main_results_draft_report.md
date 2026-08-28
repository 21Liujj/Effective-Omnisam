# Task 3-3 主结果说明 — step_09_main_results_draft_report.md

> 复用已有正式结果生成主结果表初版 + BFScore 证据审计。未跑训练 / 未跑主实验 / 未启动评测。
> Generated: 2026-06-15

---

## 交付物

| 文件 | 说明 |
|---|---|
| `experiments/tecma/tables/main_results_draft.md` | 主结果表初版（含 Δ、Speedup、Token Reduction 等派生量） |
| `experiments/tecma/notes/step_09_bfscore_evidence_audit.md` | BFScore 日志证据审计 |
| `experiments/tecma/notes/step_09_main_results_draft_report.md` | 本说明 |

registry **未改动**（只读派生）。

---

## 1. 主结果来自已有 full DensePASS protocol，不是短程 sanity

主结果表的两行数据来自 2026-04-25 跑的正式 final_2000 实验：
- `data_pd_exp_final_2000/base_last_20260425_232941/`
- `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/`

完整 DensePASS protocol（2000 张 PL、跑满 1 epoch、100 张验证集），mIoU/BFScore/效率指标均来自这两个目录的 `train_timing.jsonl` + `metrics.jsonl` + 训练日志。

## 2. 短程 sanity 只证明脚本和 profiling 字段可用

Task 3-2 的短程 sanity（8 张 PL、2 iter）**仅用于验证** `run_main_experiments.sh` 链路、输出目录隔离、`train_timing.jsonl` 新字段（D/E token 拆分）能正确落盘。

其 mIoU（baseline 60.25 / A 59.98）反映"几乎未训练"的初始权重，**不进入任何论文表格**，本任务已严格排除。

## 3. 当前主结果是 single seed / one epoch

- Seed = 1（单一种子），Epoch = 1（单轮）。
- 因此**无方差估计**，无法报告均值±标准差。
- 主表已在"限制"一节明确标注此点。

## 4. 是否建议后续补 seed2

**建议补 seed2（中优先级）。**
- 理由：主结果是单 seed，审稿人通常要求至少 2–3 个 seed 以证明结论稳健（尤其 ΔmIoU 仅 −0.10，需排除随机波动）。
- 做法：用 `run_main_experiments.sh` 传 `SEED=2` 跑两个 config（由用户手动执行，各约 2.5–2.7h）。
- 不阻塞当前 draft —— 单 seed 已足够支撑"效率显著提升、精度持平"的核心论点；seed2 用于补强统计可信度。

## 5. 是否建议进入 Phase 4 消融脚本

**建议进入 Phase 4（消融）。**
- 主结果链路（脚本 / profiling / 表格 / 证据审计）已全部打通且可复现。
- registry 已有 9 个 config（baseline + A/D/E 各种组合），消融的配置基础已就绪（Task 1 的 config check 全过）。
- Phase 4 建议：为 D-only（Dtrain_fsmc）、E-only（Etrain_sgsmr）、A-only（A_ccpm）、D+E（Dtrain_Etrain）等消融 config 复用同一套 `run_main_experiments.sh` + draft 派生流程，量化各模块对 mIoU/token/attention 的独立贡献。
- 同样遵守分工：脚本由 Claude 写/检查，训练由用户手动跑，结果贴回后 Claude 解析登记。

---

## 核心结论速览

A_Dtrain_Etrain vs baseline（Cityscapes→DensePASS, seed1, epoch1）：

| | mIoU | BFScore | Step | Mem-Attn | Mem-Tokens | Peak-Mem |
|---|---|---|---|---|---|---|
| 变化 | −0.10 | −0.11 | **1.29×** | **2.81×** | **8.0×** | −1.48 GB |

精度持平，效率显著提升。BFScore 已 verified_from_train_log（可追溯）。
