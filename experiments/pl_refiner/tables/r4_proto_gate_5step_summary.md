# r4_proto_gate_5step_summary.md
# R4 proto_gate 5-step Diagnostic Summary Table

**实验标识**: R4-4b  
**日期**: 2026-06-25（收口整理）  
**数据来源**: `experiments/pl_refiner/results/r4_proto_gate_5step_diag.jsonl`（5 条正式 records，global_step=9/18/27/36/45）  
**注意**: 本表格不包含训练性能结论，不含 mIoU，不含 full-run 结果。

---

## Table 1. 核心字段统计（5 条正式 records）

| metric | mean | min | max | comment |
|--------|-----:|----:|----:|---------|
| r2_total_valid_ratio | 0.4687 | 0.4637 | 0.4741 | R2 过滤后保留比例（所有类） |
| r2_common_valid_ratio | 0.4921 | 0.4902 | 0.4959 | R2 过滤后 common 类保留比例 |
| r2_rare_valid_ratio | 0.7193 | 0.7074 | 0.7402 | R2 过滤后 rare 类保留比例 |
| r2_extreme_rare_valid_ratio | 0.6260 | 0.0000 | 0.7933 | R2 过滤后 extreme rare 保留比例（step3 无 ext_rare 像素→0） |
| proto_gate_final_valid_ratio | 0.4290 | 0.4233 | 0.4343 | R4 proto_gate 过滤后最终保留比例 |
| proto_gate_final_common_valid_ratio | 0.4500 | 0.4479 | 0.4525 | R4 后 common 最终保留比例 |
| proto_gate_final_rare_valid_ratio | 0.7006 | 0.6856 | 0.7266 | R4 后 rare 最终保留比例 |
| proto_gate_final_extreme_rare_valid_ratio | 0.6260 | 0.0000 | 0.7933 | R4 后 extreme rare 保留比例（fallback keep，与 R2 一致） |
| proto_gate_filter_ratio_total | 0.0991 | 0.0979 | 0.0996 | R4 在 r2_valid 基础上额外过滤比例（总计） |
| proto_gate_filter_ratio_common | 0.1000 | 0.0999 | 0.1001 | R4 common 额外过滤比例 |
| proto_gate_filter_ratio_rare | 0.0504 | 0.0501 | 0.0508 | R4 rare 额外过滤比例 |
| proto_gate_filter_ratio_extreme_rare | 0.0000 | 0.0000 | 0.0000 | R4 extreme rare 额外过滤比例（1 null→step3） |
| proto_sim_mean | 0.8669 | 0.8432 | 0.8948 | prototype 余弦相似度均值 |
| proto_sim_std | 0.0733 | 0.0560 | 0.0927 | prototype 余弦相似度标准差 |
| proto_sim_valid_ratio | 0.4704 | 0.4679 | 0.4738 | 有效 prototype 相似度像素比例 |
| loss_sup | 0.0986 | 0.0680 | 0.1226 | 源域监督 loss |
| loss_unsup | 0.0009 | 0.0002 | 0.0023 | 目标域无监督 loss（经 proto_gate 过滤） |
| nan_count | 0.0000 | 0.0000 | 0.0000 | loss NaN 计数 |
| proto_sim_nan_count | 0.0000 | 0.0000 | 0.0000 | proto 相似度 NaN 计数 |

**注**: `proto_gate_filter_ratio_extreme_rare` 在 step=27（global_step=27）时为 null（当前 batch 无 extreme rare 像素），其余步骤为 0.0。均值按 4 个非 null 值=0.0 处理。

---

## Table 2. R2 前置 valid mask vs R4 proto_gate 后置 final valid mask 对比

| group | R2 valid ratio | R4 final valid ratio | delta | interpretation |
|-------|---------------:|---------------------:|------:|----------------|
| total | 0.4687 | 0.4290 | −0.0397 | R4 在 R2 基础上进一步过滤约 4.0%，总保留从 46.9% → 42.9% |
| common | 0.4921 | 0.4500 | −0.0422 | common 类过滤约 4.2 pp（底层 filter_ratio~10%，作用于 R2_valid 内部） |
| rare | 0.7193 | 0.7006 | −0.0187 | rare 类过滤约 1.9 pp（filter_ratio~5%，保护设计） |
| extreme rare | 0.6260 | 0.6260 | +0.0000 | extreme rare fallback keep，R4 不额外过滤 |

**说明**:  
- delta 均为负值（extreme rare 为零），符合设计预期：R4 只过滤，不扩充。  
- rare 的 delta 绝对值（1.9 pp）< common（4.2 pp），符合"rare 保护"策略。  
- extreme rare delta = 0 证明 fallback keep 机制工作正常。

---

## Table 3. R4 proto_gate 过滤比例分析

| group | filter ratio | keep ratio | expected | pass? |
|-------|------------:|----------:|---------:|-------|
| total | 0.0991 | 0.9009 | 0.08–0.12 | ✓ PASS |
| common | 0.1000 | 0.9000 | ~0.10 | ✓ PASS |
| rare | 0.0504 | 0.9496 | ~0.05 | ✓ PASS |
| extreme rare | 0.0000 | 1.0000 | 0 or null | ✓ PASS (fallback keep) |

**说明**: filter_ratio 基于 R2_valid 中每个类别有效像素的比例，不是原始图像像素。

---

## Table 4. Per-class 过滤分析（5 步聚合像素统计）

| class id | class name | total pixels | filtered pixels | kept pixels | filter ratio | status |
|----------|------------|-------------:|----------------:|------------:|-------------:|--------|
| 0 | road | 86,060 | 8,609 | 77,451 | 0.1000 | normal_filtered |
| 1 | sidewalk | 48,767 | 4,878 | 43,889 | 0.1000 | normal_filtered |
| 2 | building | 232,074 | 23,210 | 208,864 | 0.1000 | normal_filtered |
| 3 | wall | 0 | 0 | 0 | N/A | no_valid_pixels |
| 4 | fence | 4,172 | 208 | 3,964 | 0.0499 | below_min_filter_pixels |
| 5 | pole | 3,226 | 164 | 3,062 | 0.0508 | normal_filtered |
| 6 | traffic light | 0 | 0 | 0 | N/A | no_valid_pixels |
| 7 | traffic sign | 629 | 0 | 629 | 0.0000 | extreme_rare_keep |
| 8 | vegetation | 129,303 | 12,931 | 116,372 | 0.1000 | normal_filtered |
| 9 | terrain | 6,378 | 630 | 5,748 | 0.0988 | below_min_filter_pixels (step2) |
| 10 | sky | 83,529 | 8,355 | 75,174 | 0.1000 | normal_filtered |
| 11 | person | 2,585 | 131 | 2,454 | 0.0507 | normal_filtered |
| 12 | car | 19,881 | 1,990 | 17,891 | 0.1001 | normal_filtered |

**分类说明**:
- `normal_filtered`: 有足够像素，执行正常 bottom-10% percentile filtering
- `no_valid_pixels`: 该 batch 无该类有效像素（wall=3, traffic light=6），fallback keep
- `below_min_filter_pixels`: 像素过少，无法按 10% 过滤（filter 绝对数 < min_filter_pixels 阈值），fallback keep 或 partial filter
- `extreme_rare_keep`: traffic sign (7) 设计为 extreme rare，不做 R4 filtering，直接 keep
- **注**: wall(3) 和 traffic light(6) 在 5 个 batch 中均无有效像素，这是正常的 rare/absent class 现象，不是实现错误。

---

## Table 5. 字段命名对照（供后续报告和论文使用）

| 字段名 | 语义 | 推荐用法 |
|--------|------|---------|
| `r2_total_valid_ratio` | R2 class-balanced 过滤后保留比例 | ✓ 推荐：R2 前置比例 |
| `r2_common_valid_ratio` | R2 后 common 类保留比例 | ✓ 推荐 |
| `r2_rare_valid_ratio` | R2 后 rare 类保留比例 | ✓ 推荐 |
| `r2_extreme_rare_valid_ratio` | R2 后 extreme rare 类保留比例 | ✓ 推荐 |
| `proto_gate_final_valid_ratio` | R2+R4 联合过滤后最终保留比例 | ✓ 推荐：R4 后置比例 |
| `proto_gate_final_common_valid_ratio` | R4 后 common 最终保留比例 | ✓ 推荐 |
| `proto_gate_final_rare_valid_ratio` | R4 后 rare 最终保留比例 | ✓ 推荐 |
| `proto_gate_final_extreme_rare_valid_ratio` | R4 后 extreme rare 最终保留比例 | ✓ 推荐 |
| `refiner_total_valid_ratio` | 与 `proto_gate_final_valid_ratio` 完全相同（验证：数值一致） | ⚠ 不推荐论文使用：语义不区分 mode |
| `total_valid_ratio_after_R2` | 与 `proto_gate_final_valid_ratio` 完全相同（验证：数值一致） | ⚠ 不推荐论文使用：命名暗示 R2 后、实为 R4 后 |
| `rare_valid_ratio_after_R2` | 与 `r2_rare_valid_ratio` 完全相同 | ⚠ 不推荐论文使用：有歧义 |
| `extreme_rare_valid_ratio_after_R2` | 与 `r2_extreme_rare_valid_ratio` 完全相同 | ⚠ 不推荐论文使用：有歧义 |

