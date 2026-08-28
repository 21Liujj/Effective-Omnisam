# step_11_r4_proto_similarity_diag_plan.md
# Phase 4 / Task 4-3b — 真实 batch current-batch prototype similarity diagnosis 准备

**Run Level**: L3-style 5-step diagnostic substitute  
**日期**: 2026-06-24  
**执行人**: Claude Code  
**是否执行 diagnostic 脚本**: No（只生成脚本与命令，等待用户手动执行）  
**是否训练 full-run**: No  
**是否实现 proto_gate filtering**: No  
**是否改变 valid_mask**: No  
**是否改变 pseudo_label**: No  
**是否改变 loss**: No  
**是否实现 correction / memory consistency**: No  
**是否保存正式 checkpoint**: No  
**是否查看 mIoU**: No  
**是否进入 R4-4**: No  
**是否继续 FPA proto_bank adapter**: No

---

## 1. 本次任务说明

本次是 **Phase 4 / Task 4-3b**：准备真实 batch 的 current-batch prototype similarity diagnosis。

当前 R4 策略已经确定为：

```text
R4 v1 使用 current-batch prototype，不依赖已有 FPA proto_bank。
```

本阶段目标是在真实 `run_adaptation.py` 训练入口中，用真实 batch 的：

- `feat_t_de`
- `pseudo_label`
- R2 class-balanced refined `valid_mask`

计算 current-batch class prototype 与 per-location cosine similarity 分布。

本阶段只做 **diagnostics**：

- 不做 threshold filtering
- 不改变 R2 valid_mask
- 不改变 pseudo_label
- 不改变 loss
- 不实现 correction
- 不实现 memory consistency

---

## 2. 修改文件列表

| 文件 | 类型 | 变更内容 |
|---|---|---|
| `adaptation/pseudo_label/refiner.py` | 修改 | 在 `mode='class_balanced'` 下增加 diagnostic-only `prototype_diagnostics.enabled` 分支；先正常执行 R2 class-balanced selection，然后用 R2 valid_mask 计算 current-batch prototype similarity diagnostics；不改变返回的 `valid_mask` / `pseudo_label` / `pl_weight` |
| `adaptation/strategies.py` | 修改 | 将 refiner diagnostics 中的 prototype similarity 字段展平到 `extra_metrics`，并打印 `[proto_sim_diag]` 日志行；不参与 loss 计算 |
| `adaptation/trainer.py` | 修改 | 新增 `proto_sim_diag_records`，每 step 从 `iter_metrics` 收集 R4 prototype similarity diagnostics |
| `adaptation/engine.py` | 修改 | epoch 结束后将 `trainer.proto_sim_diag_records` 写入 `experiments/pl_refiner/results/r4_proto_similarity_5step_diag.jsonl` |
| `experiments/pl_refiner/configs/R4_proto_similarity_diag.yaml` | 新增 | R4 diagnostic-only 配置：`mode: class_balanced` + `prototype_diagnostics.enabled: true`；不启用 proto_gate filtering |
| `experiments/pl_refiner/scripts/run_r4_proto_similarity_5step_diag.sh` | 新增 | 用户手动执行的 5-step diagnostic 脚本 |
| `experiments/pl_refiner/results/r4_proto_similarity_5step_diag.jsonl` | 初始化 | 占位空文件；真正内容由用户手动执行脚本后生成 |
| `experiments/pl_refiner/notes/l3_r4_proto_similarity_5step_diag.log` | 初始化 | 占位日志；真正 stdout log 由用户手动执行脚本后生成 |
| `experiments/pl_refiner/notes/step_11_r4_proto_similarity_diag_plan.md` | 新增 | 本报告 |

补充：已运行轻量 `py_compile` 语法检查，未运行训练/diagnostic 脚本。

---

## 3. 诊断入口说明

本阶段使用 `run_adaptation.py` 真实入口，而不是 standalone dataloader 脚本。

原因：

1. 前序 standalone dataloader 诊断容易出现 DensePASS import/path 问题。
2. R3 PLQ 诊断已验证 `run_adaptation.py + train_max_iters=5` 是可用的真实入口替代方案。
3. 本次需要真实 batch 中的 `feat_t_de`，而 `feat_t_de` 只在 model forward 后自然可得；复用训练入口最稳。
4. 当前实现将诊断挂在 `PseudoLabelRefiner` 的 `class_balanced` 路径中，确保先得到真实 R2 refined valid_mask，再计算 prototype similarity。

注意：如果当前训练框架仍触发 validation 或打印 validation 相关内容，必须视为框架 artifact，不允许将 val/mIoU 指标作为 R4 结果。

---

## 4. 数据流

真实 batch 中的数据流如下：

```text
model target forward
    → pred_t, feat_t, feat_t_de

softmax(pred_t)
    → conf_t

PseudoLabelRefiner(mode='class_balanced')
    inputs:
      pseudo_label = frame_pl_t
      confidence   = conf_t
      features     = feat_t_de
      valid_mask   = frame_pl_t != ignore_index

R2 class-balanced selection
    → R2 refined valid_mask

R4 diagnostic-only current-batch prototype similarity:
    feat_t_de / pseudo_label / R2 valid_mask
    → downsample_label_and_mask_to_feature(...)
    → compute_current_batch_prototypes(...)
    → compute_proto_similarity_map(...)
    → summarize_proto_similarity(...)
    → diagnostics only

return:
    refined_label = pseudo_label          # unchanged
    valid_mask    = R2 refined valid_mask # unchanged by prototype diagnostics
    pl_weight     = None                  # unchanged
```

本次调用的纯函数来自：

```text
adaptation/pseudo_label/prototype_gate.py
```

包括：

- `downsample_label_and_mask_to_feature(...)`
- `compute_current_batch_prototypes(...)`
- `compute_proto_similarity_map(...)`
- `summarize_proto_similarity(...)`

---

## 5. 本次是否改变训练行为

必须确认：

| 项目 | 是否改变 | 说明 |
|---|---|---|
| 是否改变 valid_mask | No | prototype diagnostics 只读取 R2 refined valid_mask；返回的 `valid_mask` 仍是 R2 class-balanced mask |
| 是否改变 pseudo_label | No | `refined_label=pseudo_label`，不做 correction |
| 是否改变 loss | No | `loss_unsup` / `loss_sup` / `loss_fpa` 公式不变；diagnostics 不参与 loss |
| 是否训练 full-run | No | 只准备 5-step diagnostic 脚本，未执行 |
| 是否保存正式 checkpoint | No | 脚本使用 `experiments/pl_refiner/sanity_runs/R4_proto_similarity_5step`，不写入 `data_pd_exp_final_2000/` |
| 是否看 mIoU | No | 不读取/报告 mIoU |
| 是否实现 proto_gate filtering | No | 没有阈值过滤，没有 `final_valid_mask = R2 & proto_keep` |

---

## 6. 用户下一步手动执行命令

请在远端服务器执行：

```bash
cd /root/autodl-tmp/OmniSAM-main
bash experiments/pl_refiner/scripts/run_r4_proto_similarity_5step_diag.sh
```

脚本会输出：

```text
experiments/pl_refiner/results/r4_proto_similarity_5step_diag.jsonl
experiments/pl_refiner/notes/l3_r4_proto_similarity_5step_diag.log
```

---

## 7. 成功标准

本次 R4-3b diagnostic 成功标准：

1. 能跑完 5-step diagnostic。
2. 生成：

```text
experiments/pl_refiner/results/r4_proto_similarity_5step_diag.jsonl
```

3. JSONL 至少包含每 step 的：
   - `proto_sim_mean`
   - `proto_sim_std`
   - `proto_sim_min`
   - `proto_sim_max`
   - `proto_sim_valid_ratio`
   - `proto_valid_classes`
   - `proto_missing_classes`
   - `proto_counts_by_class`
   - `proto_sim_mean_by_class`
   - `low_sim_ratio_at_0.2`
   - `low_sim_ratio_at_0.4`
   - `low_sim_ratio_at_0.6`
   - `total_valid_ratio_after_R2`
   - `rare_valid_ratio_after_R2`
   - `extreme_rare_valid_ratio_after_R2`
   - `loss_sup`
   - `loss_unsup`
   - `nan_count`
   - `refiner_mode`

4. similarity 无 NaN：`nan_count == 0`。
5. similarity 范围在 `[-1,1]`：`proto_sim_min >= -1` 且 `proto_sim_max <= 1`。
6. `proto_valid_classes` 非空。
7. `proto_missing_classes` 有记录，即使为空列表也必须记录。
8. `low_sim_ratio_at_0.4` / `low_sim_ratio_at_0.6` 有非平凡分布，用于判断后续 threshold 是否可能误杀。
9. 不改变 valid_mask / pseudo_label / loss。

---

## 8. 如果失败，用户应贴回什么

如果脚本失败，请贴回：

```bash
tail -80 experiments/pl_refiner/notes/l3_r4_proto_similarity_5step_diag.log
```

如果 JSONL 生成但内容异常，也请贴回：

```bash
head -5 experiments/pl_refiner/results/r4_proto_similarity_5step_diag.jsonl
```

---

## 9. 输出文件说明

已准备：

```text
experiments/pl_refiner/scripts/run_r4_proto_similarity_5step_diag.sh
experiments/pl_refiner/configs/R4_proto_similarity_diag.yaml
experiments/pl_refiner/notes/step_11_r4_proto_similarity_diag_plan.md
```

已初始化占位：

```text
experiments/pl_refiner/results/r4_proto_similarity_5step_diag.jsonl
experiments/pl_refiner/notes/l3_r4_proto_similarity_5step_diag.log
```

说明：

- 因本阶段明确要求“不要执行脚本”，所以 JSONL/log 当前只是占位。
- 用户手动执行脚本后，脚本会删除旧占位并生成真实内容。

---

## 10. 最终结论

**READY_TO_RUN：已准备好用户手动运行 R4 prototype similarity 5-step diagnostic。**

不要进入 R4-4；下一步只应由用户手动运行上述 5-step diagnostic，并回贴 log/jsonl 摘要用于分析。
