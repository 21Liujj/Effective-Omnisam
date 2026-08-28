# TeCMA 主结果登记表 (main_results_registry.md)

> 论文主结果表的权威登记文件。每行一个已验证/待跑的主实验。
> 运行分工：训练命令由用户手动在服务器执行；Claude 只写脚本、检查、解析贴回的结果。
> Last updated: 2026-06-15 (Task 3-1b)

---

## 主结果表

| Method | Config | Dataset | Seed | Epoch | mIoU | BFScore | Step Time | Memory Attention Time | Memory Tokens | Peak Memory | Result Path | Status |
| ------ | ------ | ------- | ---- | ----- | ---- | ------- | --------- | --------------------- | ------------- | ----------- | ----------- | ------ |
| baseline / base_last | baseline_omnisam | Cityscapes→DensePASS | 1 | 1 | 60.26 | 39.97 | 5.27 s/it | 263.0 s | 32768 | 26.82 GB | `data_pd_exp_final_2000/base_last_20260425_232941/` | existing_verified_main_result |
| A_Dtrain_Etrain | A_Dtrain_Etrain | Cityscapes→DensePASS | 1 | 1 | 60.16 | 39.86 | 4.08 s/it | 93.6 s | 4096 | 25.34 GB | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/` | existing_verified_main_result |

---

## 字段说明

- **Method / Config**：方法名 / 对应 `experiments/tecma/configs/*.yaml`。
- **Memory Tokens**：单次 memory_attention 调用 concat 全部历史 chunk 的 K/V token 总数（口径 `per_attention_call_all_chunks_concat`）。baseline 32768 = D/E 关闭；A_Dtrain_Etrain 4096 = D 压缩后。
- **Memory Attention Time**：epoch 累计 memory attention 耗时（来自 `train_timing.jsonl.memory_attention_time`）。
- **Step Time / Peak Memory**：来自 `train_timing.jsonl` 的 `step_time` / `gpu_mem_peak`。
- **Status**：`existing_verified_main_result` = 已跑完并核对过日志的正式结果；`pending` = 待用户手动跑。

## 数据来源核对（2026-06-15）

下列字段已与远端日志逐一核对一致：

| 字段 | base_last | A_Dtrain_Etrain | 来源 |
|---|---|---|---|
| mIoU | 60.26 ✓ | 60.16 ✓ | `metrics.jsonl.val_mIoU` |
| Step Time | 5.27 ✓ | 4.08 ✓ | `train_timing.jsonl.step_time` |
| Mem Attn Time | 263.0 ✓ | 93.6 ✓ | `train_timing.jsonl.memory_attention_time` |
| Memory Tokens | 32768 ✓ | 4096 ✓ | `train_timing.jsonl.memory_tokens_before` |
| Peak Memory | 26.82 ✓ | 25.34 ✓ | `train_timing.jsonl.gpu_mem_peak` |
| BFScore | 39.97 ⚠ | 39.86 ⚠ | 用户提供（见下方说明） |

⚠ **BFScore 说明**：这两个实验的 `metrics.jsonl` 中 `BFScore` 字段为 `None`（该次运行未计算/未落盘边界 F 分数）。表中 39.97 / 39.86 为用户提供的数值，登记备查；若需在论文中引用，建议后续在评测阶段重新计算 BFScore 并回填，确保可追溯。

## 效率对比要点（A vs baseline，同 seed/epoch/数据）

- mIoU 几乎持平（60.16 vs 60.26，−0.10），精度无损。
- Step Time 4.08 vs 5.27 s/it（**−22.6%**）。
- Memory Attention Time 93.6 vs 263.0 s（**−64.4%**）。
- Memory Tokens 4096 vs 32768（**8× 压缩**）。
- Peak Memory 25.34 vs 26.82 GB（−1.48 GB）。

这是 D/E 模块带来的核心效率结论：精度持平下显著降低 memory attention 开销与显存。
