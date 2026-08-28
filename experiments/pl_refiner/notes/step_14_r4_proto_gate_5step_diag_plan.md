# step_14_r4_proto_gate_5step_diag_plan.md
# Phase 4 / Task 4-4b — R4 proto_gate 真实入口 5-step diagnostic 准备

**Run Level**: L3-style 5-step diagnostic only  
**日期**: 2026-06-24  
**执行人**: Claude Code  
**是否执行脚本**: No（只生成脚本与命令，等待用户手动执行）  
**是否改变 valid_mask**: Yes，仅作为 R4 proto_gate diagnostic  
**是否改变 pseudo_label**: No  
**是否改变 loss 公式**: No  
**是否 correction / memory consistency**: No  
**是否依赖 FPA proto_bank**: No  
**是否 full-run**: No  
**是否保存正式 checkpoint**: No  
**是否看 mIoU**: No  
**是否进入 R4-4c**: No

---

## 1. 本次任务说明

本次是 **Phase 4 / Task 4-4b：真实入口 proto_gate 5-step diagnostic**。

目标：将 R4 current-batch class-wise percentile `proto_gate` 接入 `PseudoLabelRefiner`，通过真实 `run_adaptation.py` 入口做 5-step diagnostic。

本阶段允许 R4 改变 `valid_mask`，用于测试 filtering 效果；但仍严格保持：

- 不修改 `pseudo_label` 类别；
- 不改变 loss 公式；
- 不做 correction；
- 不做 memory consistency；
- 不依赖 FPA `proto_bank`；
- 不 full-run；
- 不查看 mIoU。

---

## 2. 修改文件列表

| 文件 | 类型 | 变更内容 |
|---|---|---|
| `adaptation/pseudo_label/refiner.py` | 修改 | 新增 `mode='proto_gate'`；实现 R2 class-balanced → current-batch prototype similarity → class-wise percentile gate → final valid_mask |
| `adaptation/strategies.py` | 修改 | 展平 R4 proto_gate diagnostics 到 `extra_metrics`，供 log/trainer 收集；loss 公式不变 |
| `adaptation/trainer.py` | 修改 | 扩展 `proto_sim_diag_records` 收集 proto_gate 字段，写入 JSONL 前的 records |
| `adaptation/engine.py` | 修改 | R4 prototype diagnostics JSONL 文件名支持 config 指定；proto_gate 默认文件名为 `r4_proto_gate_5step_diag.jsonl` |
| `experiments/pl_refiner/configs/R4_proto_gate_5step_diag.yaml` | 新增 | R4 proto_gate 5-step diagnostic 配置 |
| `experiments/pl_refiner/scripts/run_r4_proto_gate_5step_diag.sh` | 新增 | 用户手动执行的 5-step diagnostic 脚本 |
| `experiments/pl_refiner/results/r4_proto_gate_5step_diag.jsonl` | 初始化 | 占位空文件；用户执行脚本后生成真实内容 |
| `experiments/pl_refiner/notes/l3_r4_proto_gate_5step_diag.log` | 初始化 | 占位日志；用户执行脚本后生成真实内容 |
| `experiments/pl_refiner/notes/step_14_r4_proto_gate_5step_diag_plan.md` | 新增 | 本报告 |

已运行轻量语法检查：

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

## 3. R4 proto_gate 数据流

本次接入的数据流严格为：

```text
R2 class-balanced selection
→ r2_valid_mask
→ feat_t_de + pseudo_label + r2_valid_mask
→ downsample label/mask 到 feature resolution
→ current-batch class prototype
→ cosine similarity map
→ class-wise percentile proto_gate filtering
→ keep_f / filter_f
→ keep_highres nearest upsample 回 label resolution
→ final_valid_mask = r2_valid_mask & keep_highres
→ 原有 unsup CE loss 使用 final_valid_mask
```

核心约束：

```text
final_valid_mask <= r2_valid_mask
```

代码中加入 hard invariant：若 `final_valid_mask` 重新激活 R2 已过滤像素，则直接报错。

---

## 4. 本次是否改变训练行为

| 项目 | 是否改变 | 说明 |
|---|---|---|
| 是否改变 valid_mask | Yes | 仅作为 R4 proto_gate diagnostic，`final_valid_mask = r2_valid_mask & keep_highres` |
| 是否改变 pseudo_label | No | `refined_label = pseudo_label`，不做 correction |
| 是否改变 loss | No | 仍走原有 CE loss；仅 mask 中 ignore 的像素变化 |
| 是否 full-run | No | 只准备 5-step diagnostic 脚本，未执行 |
| 是否依赖 FPA proto_bank | No | prototype 来源为 current-batch `feat_t_de + pseudo_label + r2_valid_mask` |
| 是否保存正式 checkpoint | No | 脚本输出目录在 `experiments/pl_refiner/sanity_runs/R4_proto_gate_5step` |
| 是否看 mIoU | No | 不读取、不报告 mIoU；若框架打印 val，视为 artifact |

---

## 5. R4 proto_gate 配置

新增配置：

```text
experiments/pl_refiner/configs/R4_proto_gate_5step_diag.yaml
```

关键语义：

```yaml
pseudo_label_refiner:
  mode: "proto_gate"
  inherit_class_balanced: true

  class_balance:
    common_classes: [0,1,2,8,9,10,12]
    rare_classes: [3,4,5,11]
    extreme_rare_classes: [6,7]
    common_top_ratio: 0.50
    rare_top_ratio: 0.75
    extreme_rare_top_ratio: 0.80
    common_min_conf: 0.75
    rare_min_conf: 0.60
    extreme_rare_min_conf: 0.55
    min_pixels_per_class: 50

  proto_gate:
    prototype_source: current_batch
    min_proto_pixels: 10
    common_filter_percentile: 0.10
    rare_filter_percentile: 0.05
    extreme_rare_filter_percentile: 0.0
    min_filter_pixels_per_class: 100
    min_keep_pixels_per_class: 10
```

---

## 6. 用户下一步手动执行命令

请在远端服务器执行：

```bash
cd /root/autodl-tmp/OmniSAM-main
bash experiments/pl_refiner/scripts/run_r4_proto_gate_5step_diag.sh
```

脚本会输出：

```text
experiments/pl_refiner/results/r4_proto_gate_5step_diag.jsonl
experiments/pl_refiner/notes/l3_r4_proto_gate_5step_diag.log
```

---

## 7. 成功标准

R4-4b 成功标准：

1. 能跑完真实入口 5-step。
2. 无 NaN。
3. 无 OOM。
4. 无 shape mismatch。
5. `final_valid_mask <= r2_valid_mask`。
6. R4 总过滤比例约 8%–12%，或接近 R4-3c simulation。
7. rare 过滤比例约 4%–6%，不应明显超过 10%。
8. extreme rare 不过滤或 fallback keep。
9. `loss_unsup` 不爆炸，不长期为 0。
10. JSONL 正常落盘。
11. 不改变 `pseudo_label`。
12. 不改变 loss 公式。

每 step JSONL 预期包含：

- R2 前置信息：`r2_total_valid_ratio`, `r2_common_valid_ratio`, `r2_rare_valid_ratio`, `r2_extreme_rare_valid_ratio`
- R4 gate 信息：`proto_gate_filter_ratio_total`, `proto_gate_keep_ratio_total`, group ratios, final valid ratios, per-class filtered/kept/total/threshold/fallback
- similarity 信息：`proto_sim_mean/std/min/max`, `proto_sim_valid_ratio`, `proto_valid_classes`, `proto_missing_classes`, `proto_sim_mean_by_class`, `nan_count`
- 训练稳定性信息：`loss_sup`, `loss_unsup`, `pseudo_valid_ratio`/`refiner_total_valid_ratio`, `ignore_ratio`/`refiner_total_ignore_ratio`, `refiner_mode`

---

## 8. 如果失败，用户应贴回什么

如果脚本失败，请贴回：

```bash
tail -100 experiments/pl_refiner/notes/l3_r4_proto_gate_5step_diag.log
```

如果 JSONL 生成但内容异常，也请贴回：

```bash
head -5 experiments/pl_refiner/results/r4_proto_gate_5step_diag.jsonl
```

---

## 9. 输出文件说明

已准备：

```text
experiments/pl_refiner/configs/R4_proto_gate_5step_diag.yaml
experiments/pl_refiner/scripts/run_r4_proto_gate_5step_diag.sh
experiments/pl_refiner/notes/step_14_r4_proto_gate_5step_diag_plan.md
```

已初始化占位：

```text
experiments/pl_refiner/results/r4_proto_gate_5step_diag.jsonl
experiments/pl_refiner/notes/l3_r4_proto_gate_5step_diag.log
```

说明：

- 因本阶段明确要求“不要执行脚本”，所以 JSONL/log 当前只是占位。
- 用户手动执行脚本后，脚本会删除旧占位并生成真实内容。

---

## 10. 最终结论

**READY_TO_RUN：已准备好用户手动运行 R4 proto_gate 5-step diagnostic。**

不要进入 R4-4c；下一步只应由用户手动运行上述 5-step diagnostic，并回贴 log/jsonl 摘要用于判断 R4 proto_gate 是否可继续。
