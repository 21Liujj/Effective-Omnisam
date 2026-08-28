# step_03_config_dryrun_report.md
# Phase 1 / Task 1-2 — R0-R7 固定配置与 Dry-Run 检查报告

**Run Level**: L1 dry-run（无训练、无模型加载、无数据集读取）  
**日期**: 2026-06-23  
**执行人**: Claude Code  
**修改代码**: 否（只新增 YAML + 检查脚本）  
**跑训练**: 否

---

## 1. 本次做了什么

- 生成 R0-R7 共 8 个固定配置 YAML（`experiments/pl_refiner/configs/R*.yaml`）
- 生成 dry-run 检查脚本（`experiments/pl_refiner/scripts/check_pl_refiner_configs.py`）
- 执行 dry-run 脚本验证 8 个配置的自洽性
- 明确区分 `R1_identity_sanity.yaml`（工程验收用）与 `R1_conf_mask.yaml`（论文消融用）
- 本步骤不实现任何新 mode（R2-R7 全部标记 NOT_IMPLEMENTED_YET）

---

## 2. 新增/修改文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `experiments/pl_refiner/configs/R0_baseline.yaml` | 更新 | 原有文件重写，补全完整注释和所有字段 |
| `experiments/pl_refiner/configs/R1_conf_mask.yaml` | **新增** | 论文消融 R1（pseudo_conf_mask plugin，mode=conf_mask） |
| `experiments/pl_refiner/configs/R2_class_balanced.yaml` | **新增** | 论文消融 R2（mode=class_balanced，NOT_IMPLEMENTED_YET） |
| `experiments/pl_refiner/configs/R3_plq_basic.yaml` | **新增** | 论文消融 R3（mode=plq_basic，NOT_IMPLEMENTED_YET） |
| `experiments/pl_refiner/configs/R4_proto_gate.yaml` | **新增** | 论文消融 R4（mode=proto_gate，NOT_IMPLEMENTED_YET） |
| `experiments/pl_refiner/configs/R5_rbpc.yaml` | **新增** | 论文消融 R5（mode=rbpc，NOT_IMPLEMENTED_YET） |
| `experiments/pl_refiner/configs/R6_memory_plq.yaml` | **新增** | 论文消融 R6（mode=memory_plq，NOT_IMPLEMENTED_YET） |
| `experiments/pl_refiner/configs/R7_full.yaml` | **新增** | 论文消融 R7（mode=full + A_Dtrain_Etrain，NOT_IMPLEMENTED_YET） |
| `experiments/pl_refiner/scripts/check_pl_refiner_configs.py` | **新增** | Dry-run 检查脚本（纯 PyYAML + stdlib，无模型依赖） |
| `experiments/pl_refiner/notes/step_03_config_dryrun_report.md` | **本文件** | 本步骤报告 |

**注意**：`experiments/pl_refiner/configs/R1_identity_sanity.yaml` 原有文件**未修改**，仅用于 L3 sanity 工程验收。

---

## 3. R0-R7 配置总表

| 配置 | refiner_mode | cb | plq | proto | corr | mem_con | memory_config | full_train_allowed | 状态 |
|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|
| R0_baseline | off | ✗ | ✗ | ✗ | ✗ | ✗ | none | true | IMPLEMENTED |
| R1_conf_mask | conf_mask | ✗ | ✗ | ✗ | ✗ | ✗ | none | true | IMPLEMENTED (plugin) |
| R2_class_balanced | class_balanced | ✓ | ✗ | ✗ | ✗ | ✗ | none | true | NOT_IMPLEMENTED_YET |
| R3_plq_basic | plq_basic | ✓ | ✓ | ✗ | ✗ | ✗ | none | true | NOT_IMPLEMENTED_YET |
| R4_proto_gate | proto_gate | ✓ | ✓ | ✓ | ✗ | ✗ | none | true | NOT_IMPLEMENTED_YET |
| R5_rbpc | rbpc | ✓ | ✓ | ✓ | ✓ | ✗ | none | conditional | NOT_IMPLEMENTED_YET |
| R6_memory_plq | memory_plq | ✓ | ✓ | ✗ | ✗ | ✓ | none | conditional | NOT_IMPLEMENTED_YET |
| R7_full | full | ✓ | ✓ | ✓ | ✗* | ✗* | A_Dtrain_Etrain | true | NOT_IMPLEMENTED_YET |

> *R7 correction 和 memory_consistency 保守默认 enable:false，等 R5/R6 稳定后开启。

**缩写**：cb=use_class_balance，plq=use_plq，proto=use_prototype，corr=use_correction，mem_con=use_memory_consistency

---

## 4. Dry-Run 检查结果

执行命令：
```bash
cd /root/autodl-tmp/OmniSAM-main
/root/miniconda3/envs/omnisam/bin/python3 \
    experiments/pl_refiner/scripts/check_pl_refiner_configs.py
```

**结果**：

```
Configs loaded       : 8/8
Implemented modes    : 2/8  (R2-R7 = NOT_IMPLEMENTED_YET is expected)
Consistency check    : PASS
Training started     : NO
Model/data loaded    : NO

✓ All 8 configs loaded and internally consistent.
✓ R2-R7 correctly marked NOT_IMPLEMENTED_YET.
✓ R1_identity_sanity and R1_conf_mask are distinct configs.
✓ Dry-run PASS — safe to proceed to Phase 2 / Task 2-1.
```

R0-R7 全部 PASS：
- R0, R1: 已实现，一致性 PASS
- R2-R7: 配置自洽，正确标记 NOT_IMPLEMENTED_YET，一致性 PASS
- 无 FORBIDDEN 组合错误（如 R2 use_plq=true 等被禁止的混乱开关）
- 命名 guard PASS（identity_sanity mode=identity vs conf_mask mode=conf_mask）

---

## 5. 哪些 mode 已实现，哪些 mode 尚未实现

### 已实现

| mode | 实现方式 | 位置 |
|---|---|---|
| `off` | `PseudoLabelRefiner.refine()` 返回 None；caller 原逻辑不变 | `adaptation/pseudo_label/refiner.py` |
| `identity` | 返回原 `frame_pl_t` / `valid_mask` 不修改 | `adaptation/pseudo_label/refiner.py` |
| `conf_mask` | 通过现有 `pseudo_conf_mask` plugin（A 模块）实现 | `adaptation/plugins/pseudo_conf_mask.py` + `strategies.py` |

### 尚未实现（NOT_IMPLEMENTED_YET）

| mode | 等待 Phase | 所需新文件 |
|---|---|---|
| `class_balanced` | Phase 2 | `adaptation/pseudo_label/class_balance.py` |
| `plq_basic` | Phase 3 | `adaptation/pseudo_label/plq.py`（当前为占位桩） |
| `proto_gate` | Phase 4 | `adaptation/pseudo_label/prototype_bank.py`（当前为占位桩） |
| `rbpc` | Phase 5 | `adaptation/pseudo_label/correction.py`（当前为占位桩） |
| `memory_plq` | Phase 6 | `adaptation/pseudo_label/plq.py` memory path（需先做 Task 6-1 可行性扫描）|
| `full` | Phase 7 | 上述全部 |

当前 `refiner.py` 的 `SUPPORTED_MODES = frozenset({'off', 'identity'})`。  
调用 R2-R7 config 的 mode 时会触发 `NotImplementedError`（设计行为，正确）。  
训练时直接使用 R2-R7 config 会报错，这是预期的安全保护。

---

## 6. R1_identity_sanity 与 R1_conf_mask 的区别

| | R1_identity_sanity.yaml | R1_conf_mask.yaml |
|---|---|---|
| 目的 | 工程验收：验证 refiner 插入点不改变 loss 数值 | 论文消融 R1：confidence masking baseline |
| refiner mode | `identity` | `conf_mask` |
| pseudo_conf_mask plugin | disabled | **enabled** (threshold=0.85) |
| full_train_allowed | 否（仅 5-step sanity） | **true** |
| 对应已有实验 | 无（仅验证等价性） | A_ccpm（clean ablation，mIoU=60.26） |
| 用于哪个阶段 | Phase 1 Task 1-1 L3 sanity（已完成） | Phase 1 Task 1-2 配置固化 + Phase 1 全量训练 |
| 命名含义 | identity = 不修改任何内容 | conf_mask = 置信度过滤 |

**严格禁止混用**：两个配置的 `mode` 字段、`pseudo_conf_mask.enabled` 字段均不同。  
论文消融表中 R1 行对应 `R1_conf_mask`，不对应 `R1_identity_sanity`。

---

## 7. 当前风险

| 风险 | 严重度 | 说明 |
|---|---|---|
| R2-R7 的 YAML 参数（top_ratio、gate_threshold 等）是合理默认值，未经实验验证 | 中 | 等 Phase 2-7 实现时按实验结果调参；YAML 只是起点，不是最终值 |
| R7 correction/memory_consistency 保守默认 false | 低 | 设计决策，等 R5/R6 稳定后再开启，已在 YAML 注释中说明 |
| R6 memory_plq 可行性未确认 | 中 | memory PLQ 需要 patch 坐标 / overlap logits，需 Phase 6 Task 6-1 扫描；已在 YAML 注释中标注 `TBD Phase 6` |
| output_dir 字段：所有配置当前使用 reuse_pl_dir 而非 output_dir | 低 | 全量训练时需要新的 output_dir（含时间戳）；当前 5-step sanity 复用 base_last PL 合理 |
| L3 sanity 日志缺少 loss_sup / loss_unsup 等字段 | 中 | 见下方 §8 建议字段列表；不阻塞 Task 1-2，但会影响后续诊断 |

---

## 8. Recommended Sanity Logging Fields

> **本阶段不修改 loss 逻辑，不为了 logging 改训练行为。**  
> 以下字段是后续 L2/L3 诊断中建议记录的指标，供未来 Phase 2+ 实现时参考。

当前 L3 sanity 日志（sanity_r0_off.log / sanity_r1_identity.log）**缺少**以下字段，  
导致 CONDITIONAL PASS（有训练完成，但无法对比 loss 等价性）。

### 建议在 strategies.py 或 trainer.py 中新增的日志字段

| 字段 | 含义 | 位置 | 优先级 |
|---|---|---|---|
| `loss_sup` | source CE loss（每 iter） | strategies.py forward 返回 | 高 |
| `loss_unsup` | target unsup CE loss（每 iter） | strategies.py forward 返回 | 高 |
| `pseudo_valid_ratio` | mask_t 有效像素比例 `mask_t.float().mean()` | strategies.py | 高 |
| `ignore_ratio` | 被 ignore 的像素比例 `1 - valid_mask.float().mean()` | strategies.py | 高 |
| `valid_pixel_ratio` | PL 文件中非 255 像素比例（离线 PL 质量） | strategies.py | 中 |
| `pl_weight_mean` | PLQ 权重均值（R3+ 用）| strategies.py | 中（R3+） |
| `refiner_mode` | 当前 refiner mode（每 iter 或每 epoch） | strategies.py / trainer.py | 中 |
| `effective_pixel_ratio` | 实际参与 loss 计算的像素比例（mask_t AND valid_mask） | strategies.py | 中 |
| `gpu_mem_peak` | 每 iter 结束时的 GPU 显存峰值（MB） | trainer.py | 低 |

**当前状态**：strategies.py 已有 `pseudo_valid_ratio`、`pseudo_mean_confidence`、`pseudo_filtered_ratio`
等字段输出（约 line 364-366），但不在 step-level 日志中，只在某些 debug 模式下输出。  
建议 Phase 2 实现 R2 时统一在 trainer.py 的 per-step log 中输出 `loss_sup`/`loss_unsup`/`pseudo_valid_ratio`，
无需修改 loss 计算逻辑，只需在已有变量上增加日志打印。

---

## 9. 是否允许进入 Phase 2 / Task 2-1

**允许进入 Phase 2 / Task 2-1**。

条件确认：

| 条件 | 状态 |
|---|---|
| R0-R7 YAML 已生成 | ✓ 8/8 |
| dry-run 检查 PASS | ✓ 8/8 PASS |
| R1_identity_sanity vs R1_conf_mask 命名区分 | ✓ 已确认不同 mode |
| R2-R7 正确标记 NOT_IMPLEMENTED_YET | ✓ |
| 当前代码无破坏性修改 | ✓（只新增 YAML + 脚本） |
| L3 sanity CONDITIONAL PASS 已记录 | ✓（logging 缺失风险已登记）|

**进入 Phase 2 的注意事项**：
1. Task 2-1 实现 `class_balanced` mode 时，先修改 `adaptation/pseudo_label/class_balance.py`（当前为占位桩）
2. 将 `refiner.py` 的 `SUPPORTED_MODES` 加入 `'class_balanced'`
3. L3 sanity 时建议同步补充 `loss_sup`/`loss_unsup` per-step 日志（见 §8），以达到 strict PASS
4. Phase 2 的 output_dir 必须用新目录名（含实验名/时间戳），不复用 sanity_runs/

---

## 10. 本次任务总结

- **做了什么**：生成 R0-R7 YAML 配置（8 个），dry-run 检查脚本，执行验证（全部 PASS）
- **修改了哪些文件**：只新增 YAML + 脚本，无 Python 代码修改
- **是否跑训练**：否
- **是否改动 pseudo label**：否
- **是否改动 loss**：否
- **是否实现新 mode**：否（R2-R7 仅有 YAML 定义，无实现）
- **当前风险**：中（YAML 参数未调参；logging 缺失；R6 可行性待扫描）
- **是否允许进入 Phase 2**：**允许**
