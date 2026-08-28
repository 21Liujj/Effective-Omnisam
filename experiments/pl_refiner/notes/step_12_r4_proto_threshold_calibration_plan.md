# step_12_r4_proto_threshold_calibration_plan.md
# Phase 4 / Task 4-3c — R4 prototype similarity threshold calibration diagnostic 准备

**Run Level**: L3-style diagnostic only  
**日期**: 2026-06-24  
**执行人**: Claude Code  
**是否执行 diagnostic 脚本**: No（只生成脚本与命令，等待用户手动执行）  
**是否实现 proto_gate filtering**: No  
**是否改变 valid_mask**: No  
**是否改变 pseudo_label**: No  
**是否改变 loss**: No  
**是否 correction / memory consistency**: No  
**是否 full-run**: No  
**是否保存正式 checkpoint**: No  
**是否看 mIoU**: No  
**是否进入 R4-4**: No

---

## 1. 本次任务说明

本次是 **Phase 4 / Task 4-3c：R4 prototype similarity threshold calibration diagnostic**。

目标是在真实 batch 上继续统计 current-batch prototype similarity 分布，用于后续决定 `proto_gate` threshold。  
本阶段只做 calibration diagnostics，不做任何 filtering。

本次扩展的统计包括：

- 全局 similarity quantiles：q01/q05/q10/q25/q50/q75/q90/q95/q99
- 更高 absolute threshold 的 low-ratio：0.65/0.70/0.75/0.80/0.85/0.90
- class-wise threshold / quantile 统计
- common / rare / extreme rare group-wise 统计
- percentile filtering simulation，仅模拟过滤比例，不改变 mask

---

## 2. 修改文件列表

| 文件 | 类型 | 变更内容 |
|---|---|---|
| `adaptation/pseudo_label/prototype_gate.py` | 修改 | 扩展 `summarize_proto_similarity()`：新增 quantiles、threshold ratios、class-wise/group-wise stats、filter simulation |
| `adaptation/pseudo_label/refiner.py` | 修改 | 将 threshold calibration 相关 summary 字段写入 diagnostics；仍 diagnostic-only |
| `adaptation/strategies.py` | 修改 | 将新增 calibration diagnostics 展平到 `extra_metrics`，供 trainer 收集；不参与 loss |
| `adaptation/trainer.py` | 修改 | 扩展 `proto_sim_diag_records` 收集字段，包含 quantiles/class-wise/group-wise/simulation stats |
| `adaptation/engine.py` | 修改 | R4 prototype diagnostics jsonl 文件名支持从 config 的 `prototype_diagnostics.output_jsonl` 指定 |
| `experiments/pl_refiner/configs/R4_proto_threshold_calibration.yaml` | 新增 | R4-3c calibration-only 配置；不启用 filtering |
| `experiments/pl_refiner/scripts/run_r4_proto_threshold_calibration_5step_diag.sh` | 新增 | 用户手动执行的 5-step calibration diagnostic 脚本 |
| `experiments/pl_refiner/results/r4_proto_threshold_calibration_5step_diag.jsonl` | 初始化 | 占位空文件；用户手动执行脚本后生成真实内容 |
| `experiments/pl_refiner/notes/l3_r4_proto_threshold_calibration_5step_diag.log` | 初始化 | 占位日志；用户手动执行脚本后生成真实内容 |
| `experiments/pl_refiner/notes/step_12_r4_proto_threshold_calibration_plan.md` | 新增 | 本报告 |

已执行轻量语法检查：

```bash
/root/miniconda3/envs/omnisam/bin/python -m py_compile \
  adaptation/pseudo_label/prototype_gate.py \
  adaptation/pseudo_label/refiner.py \
  adaptation/strategies.py \
  adaptation/trainer.py \
  adaptation/engine.py
```

结果：通过。未运行 diagnostic 脚本。

---

## 3. 为什么需要 R4-3c

R4-3b 已跑通真实 batch current-batch prototype similarity diagnosis，观察到：

- `proto_sim_mean` 大约 `0.84–0.89`
- `proto_sim_std` 大约 `0.056–0.093`
- `proto_sim_min` 大约 `0.43–0.54`
- `proto_sim_max` 大约 `0.99`
- `nan_count = 0`
- `low_sim_ratio_at_0.2 = 0`
- `low_sim_ratio_at_0.4 = 0`
- `low_sim_ratio_at_0.6` 通常很小，最终 jsonl 大约 `0.0005–0.0173`

结论：

1. prototype similarity 比 R3 raw PLQ 更有区分度；
2. 但 `threshold=0.6` 太保守，过滤比例太小；
3. 不能直接进入 `proto_gate filtering`；
4. 需要先统计更高 threshold 与 quantile 分布，评估 0.75/0.80/0.85 等阈值或 percentile strategy 的潜在过滤比例。

因此本次 R4-3c 只做 threshold calibration，不做 filtering。

---

## 4. 新增 diagnostics 字段

### 4.1 全局 similarity 分布

保留：

- `proto_sim_mean`
- `proto_sim_std`
- `proto_sim_min`
- `proto_sim_max`
- `proto_sim_valid_ratio`
- `proto_valid_classes`
- `proto_missing_classes`
- `proto_counts_by_class`
- `proto_sim_mean_by_class`
- `nan_count`

新增 quantiles：

- `proto_sim_q01`
- `proto_sim_q05`
- `proto_sim_q10`
- `proto_sim_q25`
- `proto_sim_q50`
- `proto_sim_q75`
- `proto_sim_q90`
- `proto_sim_q95`
- `proto_sim_q99`

新增 threshold ratios：

- `low_sim_ratio_at_0.65`
- `low_sim_ratio_at_0.70`
- `low_sim_ratio_at_0.75`
- `low_sim_ratio_at_0.80`
- `low_sim_ratio_at_0.85`
- `low_sim_ratio_at_0.90`

### 4.2 class-wise threshold / quantile 统计

新增：

- `low_sim_ratio_at_0.75_by_class`
- `low_sim_ratio_at_0.80_by_class`
- `low_sim_ratio_at_0.85_by_class`
- `proto_sim_q10_by_class`
- `proto_sim_q25_by_class`
- `proto_sim_q50_by_class`

如果某类 prototype missing 或没有有效 similarity pixel，对应字段为 `null`。

### 4.3 group-wise 统计

使用 Cityscapes-13 分组：

```text
common:       0,1,2,8,9,10,12
rare:         3,4,5,6,7,11
extreme rare: 6,7
```

新增：

- `proto_sim_mean_common`
- `proto_sim_mean_rare`
- `proto_sim_mean_extreme_rare`
- `low_sim_ratio_at_0.75_common`
- `low_sim_ratio_at_0.75_rare`
- `low_sim_ratio_at_0.80_common`
- `low_sim_ratio_at_0.80_rare`
- `low_sim_ratio_at_0.85_common`
- `low_sim_ratio_at_0.85_rare`

如果某 group 没有有效像素，记为 `null`，不报错。

### 4.4 percentile filtering simulation

只做模拟，不改变 valid_mask。

A. global absolute threshold：

- `0.75`
- `0.80`
- `0.85`

记录前缀：

- `simulated_abs_0_75_*`
- `simulated_abs_0_80_*`
- `simulated_abs_0_85_*`

B. global percentile：

- bottom 5%
- bottom 10%

记录前缀：

- `simulated_global_bottom5_*`
- `simulated_global_bottom10_*`

C. class-wise percentile：

- common classes filter bottom 10%
- rare classes filter bottom 5%
- extreme rare classes filter bottom 0%，fallback keep

记录前缀：

- `simulated_classwise_percentile_*`

同时提供任务要求的简单 alias：

- `simulated_filter_ratio_total`
- `simulated_filter_ratio_common`
- `simulated_filter_ratio_rare`
- `simulated_filter_ratio_extreme_rare`
- `simulated_keep_ratio_total`
- `simulated_keep_ratio_rare`

这些 alias 对应推荐的 class-wise percentile simulation。

---

## 5. 本次是否改变训练行为

| 项目 | 是否改变 | 说明 |
|---|---|---|
| 是否改变 valid_mask | No | 所有 threshold/percentile 都只模拟，不回写 mask |
| 是否改变 pseudo_label | No | 不做 correction，不改类别 |
| 是否改变 loss | No | diagnostics 不参与 loss |
| 是否训练 full-run | No | 只准备 5-step diagnostic 脚本，未执行 |
| 是否保存正式 checkpoint | No | 脚本使用 `experiments/pl_refiner/sanity_runs/R4_proto_threshold_calibration_5step`，不写 `data_pd_exp_final_2000/` |
| 是否看 mIoU | No | 不读取、不报告 mIoU；若框架打印 validation，视为 artifact |
| 是否进入 R4-4 | No | 本次只准备 R4-3c |

---

## 6. 用户下一步手动执行命令

请在远端服务器执行：

```bash
cd /root/autodl-tmp/OmniSAM-main
bash experiments/pl_refiner/scripts/run_r4_proto_threshold_calibration_5step_diag.sh
```

脚本会输出：

```text
experiments/pl_refiner/results/r4_proto_threshold_calibration_5step_diag.jsonl
experiments/pl_refiner/notes/l3_r4_proto_threshold_calibration_5step_diag.log
```

---

## 7. 成功标准

本次 R4-3c calibration 成功标准：

1. 能跑完 5-step diagnostic。
2. 生成：

```text
experiments/pl_refiner/results/r4_proto_threshold_calibration_5step_diag.jsonl
```

3. similarity quantiles 正常：q01/q05/q10/q25/q50/q75/q90/q95/q99 均存在，或在无有效像素时为 `null`。
4. `nan_count = 0`。
5. 能估计 threshold `0.75 / 0.80 / 0.85` 的过滤比例。
6. 能估计 global percentile bottom 5% / 10% 的过滤比例。
7. 能估计 class-wise percentile strategy 的过滤比例：common bottom 10%，rare bottom 5%，extreme rare fallback keep。
8. 不改变 valid_mask / pseudo_label / loss。

---

## 8. 如果失败，用户应贴回什么

如果脚本失败，请贴回：

```bash
tail -80 experiments/pl_refiner/notes/l3_r4_proto_threshold_calibration_5step_diag.log
```

如果 JSONL 生成但内容异常，也请贴回：

```bash
head -5 experiments/pl_refiner/results/r4_proto_threshold_calibration_5step_diag.jsonl
```

---

## 9. 输出文件说明

已准备：

```text
experiments/pl_refiner/scripts/run_r4_proto_threshold_calibration_5step_diag.sh
experiments/pl_refiner/configs/R4_proto_threshold_calibration.yaml
experiments/pl_refiner/notes/step_12_r4_proto_threshold_calibration_plan.md
```

已初始化占位：

```text
experiments/pl_refiner/results/r4_proto_threshold_calibration_5step_diag.jsonl
experiments/pl_refiner/notes/l3_r4_proto_threshold_calibration_5step_diag.log
```

说明：

- 因本阶段明确要求“不要执行脚本”，所以 JSONL/log 当前只是占位。
- 用户手动执行脚本后，脚本会删除旧占位并生成真实内容。

---

## 10. 最终结论

**READY_TO_RUN：已准备好用户手动运行 R4 threshold calibration 5-step diagnostic。**

不要进入 R4-4；下一步只应由用户手动运行上述 5-step calibration diagnostic，并回贴 log/jsonl 摘要用于分析 threshold。
