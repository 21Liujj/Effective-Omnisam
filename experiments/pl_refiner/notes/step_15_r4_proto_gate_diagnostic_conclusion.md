# step_15_r4_proto_gate_diagnostic_conclusion.md
# Phase 4 / Task 4-4c — R4 proto_gate 5-step Diagnostic 结果收口报告

**Run Level**: Report / code-audit only  
**日期**: 2026-06-25  
**执行人**: Claude Code  
**是否训练**: No  
**是否运行 run_adaptation.py**: No  
**是否 full-run**: No  
**是否看 mIoU**: No（val mIoU=60.22 是 diagnostic artifact，不作为 R4 结果，见 §9）  
**是否保存 checkpoint**: No  
**是否修改 pseudo_label**: No  
**是否修改 loss**: No  
**是否进入 R5/R6**: No  

---

## 1. R4-4c 任务说明

本阶段是 **R4 proto_gate 5-step diagnostic 的结果收口、字段审计和 full-run 决策阶段**。

明确声明：
- 不是新实验
- 不训练
- 不看 mIoU
- 不做 full-run
- 不改算法
- 只整理 R4-4b 诊断结果，审计字段命名，给出 full-run 决策建议

---

## 2. 读取文件列表

| 文件 | 状态 |
|------|------|
| `experiments/pl_refiner/results/r4_proto_gate_5step_diag.jsonl` | ✓ 读取成功（5 条正式 records） |
| `experiments/pl_refiner/notes/l3_r4_proto_gate_5step_diag.log` | ✓ 读取成功（255 行） |
| `experiments/pl_refiner/notes/step_13_r4_proto_gate_filter_unit.md` | ✓ 读取成功 |
| `experiments/pl_refiner/notes/step_12_r4_proto_threshold_calibration_plan.md` | ✓ 读取成功 |
| `experiments/pl_refiner/notes/step_11_r4_proto_similarity_diag_plan.md` | ✓ 读取成功 |
| `experiments/pl_refiner/notes/latest_status.md` | MISSING（本次创建） |

---

## 3. R4-4b 正式统计结果

数据来源：5 条正式 records（global_step=9/18/27/36/45，epoch=1，iter=1–5）

| metric | mean | min | max | comment |
|--------|-----:|----:|----:|---------|
| r2_total_valid_ratio | 0.4687 | 0.4637 | 0.4741 | R2 前置保留比例 |
| r2_common_valid_ratio | 0.4921 | 0.4902 | 0.4959 | R2 common 保留 |
| r2_rare_valid_ratio | 0.7193 | 0.7074 | 0.7402 | R2 rare 保留 |
| r2_extreme_rare_valid_ratio | 0.6260 | 0.0000 | 0.7933 | R2 ext_rare 保留（step3 无像素→0） |
| proto_gate_final_valid_ratio | 0.4290 | 0.4233 | 0.4343 | R4 后最终保留比例 |
| proto_gate_final_common_valid_ratio | 0.4500 | 0.4479 | 0.4525 | R4 后 common 最终 |
| proto_gate_final_rare_valid_ratio | 0.7006 | 0.6856 | 0.7266 | R4 后 rare 最终 |
| proto_gate_final_extreme_rare_valid_ratio | 0.6260 | 0.0000 | 0.7933 | R4 后 ext_rare（fallback keep = R2 值） |
| proto_gate_filter_ratio_total | 0.0991 | 0.0979 | 0.0996 | R4 额外过滤比例 |
| proto_gate_filter_ratio_common | 0.1000 | 0.0999 | 0.1001 | common 额外过滤 |
| proto_gate_filter_ratio_rare | 0.0504 | 0.0501 | 0.0508 | rare 额外过滤 |
| proto_gate_filter_ratio_extreme_rare | 0.0000 | 0.0000 | 0.0000 | ext_rare 额外过滤（1 null→step3） |
| proto_sim_mean | 0.8669 | 0.8432 | 0.8948 | prototype 余弦相似度均值 |
| proto_sim_std | 0.0733 | 0.0560 | 0.0927 | 标准差 |
| proto_sim_valid_ratio | 0.4704 | 0.4679 | 0.4738 | 有效 proto 像素比例 |
| loss_sup | 0.0986 | 0.0680 | 0.1226 | 源域 CE loss |
| loss_unsup | 0.0009 | 0.0002 | 0.0023 | 目标域无监督 loss |
| nan_count | 0.0000 | 0.0000 | 0.0000 | loss NaN 计数 |
| proto_sim_nan_count | 0.0000 | 0.0000 | 0.0000 | proto 相似度 NaN 计数 |

---

## 4. R2 前置 valid mask vs R4 proto_gate 后置 valid mask 对比

| group | R2 valid ratio | R4 final valid ratio | delta | interpretation |
|-------|---------------:|---------------------:|------:|----------------|
| total | 0.4687 | 0.4290 | −0.0397 | R4 在 R2 基础上进一步过滤约 4.0 pp |
| common | 0.4921 | 0.4500 | −0.0422 | common 额外过滤约 4.2 pp（filter_ratio~10%） |
| rare | 0.7193 | 0.7006 | −0.0187 | rare 额外过滤约 1.9 pp（filter_ratio~5%，保护设计生效） |
| extreme rare | 0.6260 | 0.6260 | 0.0000 | fallback keep，R4 不额外过滤 ext_rare |

**解释**:
- R4 在 R2 基础上进一步减少约 4.0% 的总有效 token，方向正确
- total valid ratio 从约 46.9% 降到约 42.9%，符合设计预期（~42–44%）
- common 过滤约 10%，rare 过滤约 5%，非对称保护策略按预期工作
- extreme rare fallback keep 机制完全生效，delta=0

---

## 5. R4 proto_gate 过滤比例分析

| group | filter ratio | keep ratio | expected | pass? |
|-------|------------:|----------:|---------:|-------|
| total | 0.0991 | 0.9009 | 0.08–0.12 | ✓ PASS |
| common | 0.1000 | 0.9000 | ~0.10 | ✓ PASS |
| rare | 0.0504 | 0.9496 | ~0.05 | ✓ PASS |
| extreme rare | 0.0000 | 1.0000 | 0 or null | ✓ PASS (fallback keep) |

**结论**: 所有组的过滤比例均在预期范围内，偏差 < 1‰。

---

## 6. Per-class 过滤分析

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
| 9 | terrain | 6,378 | 630 | 5,748 | 0.0988 | below_min_filter_pixels (partial) |
| 10 | sky | 83,529 | 8,355 | 75,174 | 0.1000 | normal_filtered |
| 11 | person | 2,585 | 131 | 2,454 | 0.0507 | normal_filtered |
| 12 | car | 19,881 | 1,990 | 17,891 | 0.1001 | normal_filtered |

**重要说明**:
- `wall`(3) 和 `traffic light`(6) 在 5 个 diagnostic batch 中均无有效像素（DensePASS 场景下这两类极为稀少），fallback keep 是正确行为，不是实现错误
- `fence`(4) 总像素极少（4,172），部分 step 触发 below_min_filter_pixels fallback，aggregated filter_ratio=4.99%，符合 rare 组设计
- `traffic sign`(7) 是设计定义的 extreme_rare，完全不做 R4 filtering，filtered=0 是正确的
- `terrain`(9) 在 step=2 中触发 `below_min_filter_pixels_95` fallback，其余 step 正常过滤约 10%，aggregated 结果合理
- 所有 fallback 行为均有 `proto_gate_fallback_classes` 和 `proto_gate_fallback_reasons` 记录，可追溯

---

## 7. 字段命名检查与审计

### 7.1 身份验证

数值验证结果（5 step 全部一致）：
- `refiner_total_valid_ratio` ≡ `proto_gate_final_valid_ratio` ≡ `total_valid_ratio_after_R2`（差值 < 1e-8）
- `rare_valid_ratio_after_R2` ≡ `r2_rare_valid_ratio`
- `extreme_rare_valid_ratio_after_R2` ≡ `r2_extreme_rare_valid_ratio`

### 7.2 字段命名问题分析

| 字段 | 问题 | 严重程度 |
|------|------|---------|
| `total_valid_ratio_after_R2` | 命名暗示"R2 后"，但实际等于 R4 后的 final ratio（因为 proto_gate 是在 R2 之后追加的） | ⚠ 中等：在论文中直接引用会误导读者 |
| `rare_valid_ratio_after_R2` | 命名表示 R2 后的 rare ratio，数值也确实是 R2 后（r2_rare），但与 r2_rare_valid_ratio 重复 | ⚠ 低：冗余字段，不会造成数值错误 |
| `extreme_rare_valid_ratio_after_R2` | 同上 | ⚠ 低：冗余字段 |
| `refiner_total_valid_ratio` | 等于 proto_gate_final，在 proto_gate mode 下语义准确，但换成其他 mode 时可能不同 | ⚠ 低：mode-dependent，论文中不建议直接引用 |

### 7.3 字段命名根本问题

`total_valid_ratio_after_R2` 的命名逻辑是"经过 R2 过滤后的 valid ratio"，但在 `refiner_mode=proto_gate` 时，代码链路是：

```
R2_valid_mask → proto_gate → final_valid_mask
```

最终写入 `total_valid_ratio_after_R2` 的值是 `final_valid_mask` 的比例（即 R2+R4 联合结果），而不是 R2 的中间结果。这是命名歧义的根源。

### 7.4 后续报告和论文推荐字段

**R2 前置比例（仅反映 R2 class-balanced selection 的结果）**：
- `r2_total_valid_ratio`
- `r2_common_valid_ratio`
- `r2_rare_valid_ratio`
- `r2_extreme_rare_valid_ratio`

**R4 后置比例（R2+R4 联合过滤后的最终结果）**：
- `proto_gate_final_valid_ratio`
- `proto_gate_final_common_valid_ratio`
- `proto_gate_final_rare_valid_ratio`
- `proto_gate_final_extreme_rare_valid_ratio`

**不建议在论文中使用**（除非代码中明确修正命名）：
- `total_valid_ratio_after_R2`（命名误导，应改为 `final_valid_ratio` 或 `proto_gate_final_valid_ratio`）
- `rare_valid_ratio_after_R2`（冗余，用 `r2_rare_valid_ratio`）
- `extreme_rare_valid_ratio_after_R2`（冗余，用 `r2_extreme_rare_valid_ratio`）
- `refiner_total_valid_ratio`（mode-agnostic，语义不精确）

### 7.5 修正建议（不重跑实验，不改算法逻辑）

在下一次 full-run 的 diagnostic 代码中，建议将 jsonl 写入代码中的 `total_valid_ratio_after_R2` 改为 `final_valid_ratio`，或直接废弃该字段，只保留 `proto_gate_final_valid_ratio`。这是纯日志字段改名，不影响算法逻辑。

---

## 8. R4-4b 稳定性判断

| check | result | pass? |
|-------|--------|-------|
| run through 5-step real entry | 5/5 steps 完成，log 确认 proto_sim_diag + refiner_diag 均有输出 | ✓ PASS |
| refiner_mode = proto_gate | 所有 5 条 records 均有 `"refiner_mode": "proto_gate"` | ✓ PASS |
| no NaN | nan_count=0, proto_sim_nan_count=0（所有 5 步） | ✓ PASS |
| no OOM | log 正常完成，无 CUDA OOM 报错 | ✓ PASS |
| no shape mismatch | log 无任何 shape mismatch / broadcasting error | ✓ PASS |
| loss_unsup stable | 范围 [0.000220, 0.002347]，无爆炸，无归零异常 | ✓ PASS |
| pseudo_label unchanged | R4 只修改 valid_mask，代码路径不触及 pseudo_label tensor；not directly logged，inferred from implementation | INFERRED |
| loss unchanged | loss 计算公式未修改，仅 valid_mask 传入；not directly logged，inferred from implementation | INFERRED |
| final_valid_mask <= R2 valid_mask | proto_gate_final_valid_ratio ≤ r2_total_valid_ratio（所有步骤验证）；delta 均为负值 | ✓ PASS |
| total filter ratio expected | 0.0979–0.0996，在 0.08–0.12 范围内 | ✓ PASS |
| rare not over-filtered | filter_ratio_rare~0.050，< filter_ratio_common~0.100 | ✓ PASS |
| extreme rare fallback keep | filter_ratio_extreme_rare=0（或 null），与设计一致 | ✓ PASS |

**说明**: `pseudo_label unchanged` 和 `loss unchanged` 标记为 INFERRED，因为 diagnostic 日志未显式 assert pseudo_label tensor 不变。建议在 full-run 前增加一个显式 assert（`assert (pl_before == pl_after).all()`），但这不阻止当前决策。

---

## 9. 不允许写入论文的内容

**严格声明**：

1. `val mIoU=60.22` 出现在 `l3_r4_proto_gate_5step_diag.log` 末尾，是 diagnostic 脚本触发的 val artifact（`max_train_iters=5` 导致 epoch 结束后自动 val），**不作为 R4 结果**，不得引用。

2. 当前**不能 claim R4 improves mIoU**——没有完整 epoch 对比，没有 full-run baseline vs R4 的对比。

3. 当前**不能 claim R4 improves BFScore**——同上，60.22 是 diagnostic artifact。

4. 当前**只能 claim**：
   > R4 proto_gate can stably filter low prototype-similarity pseudo labels in a real 5-step training entry, with total filter_ratio ≈ 9.9%, common ≈ 10.0%, rare ≈ 5.0%, extreme_rare = 0 (fallback keep), no NaN, no OOM, in epoch=1 5-step diagnostic.

5. **full-run 后才能写最终性能提升**（mIoU, mAcc, BFScore 等）。

---

## 10. 结论与 full-run 决策

### 10.1 R4-4b 诊断结论

**PASS**（有 2 项 INFERRED，但不构成阻塞条件）

具体判断：
- 5/5 step 正常完成，无 NaN、无 OOM、无 shape error
- 过滤比例精确匹配设计值（common=10%, rare=5%, ext_rare=0%）
- R2→R4 的 final valid ratio 降低路径正确（均为负 delta）
- fallback keep 对所有边缘类（wall, traffic light, traffic sign, fence, terrain 部分 step）正常触发
- loss_unsup 范围合理，无爆炸
- proto_sim 全程无 NaN，proto_sim_mean ≈ 0.87（高相似度，说明当前批次原型质量高）

### 10.2 是否允许进入 full-run

**CONDITIONAL YES**

条件：
1. 用户确认 full-run 目录名不复用已有目录（使用新 `pd_root` 目录名）
2. full-run 前保存配置 YAML 快照、seed、GPU
3. 建议在 full-run 启动脚本中添加 `assert` 检查 pseudo_label 不被修改（可选，不强制阻塞）
4. R2 full-run 先于 R2+R4 full-run 执行（提供 baseline 对比）

### 10.3 推荐 full-run 矩阵

| 实验 | 建议 | 原因 |
|------|------|------|
| R2 class-balanced full-run | **必须，优先** | 建立 R4 对比 baseline；R2 有完整 diagnostic 支撑 |
| R2+R4 proto_gate full-run | **建议，排第二** | R4 diagnostic 已通过，逻辑正确，需 full-run 验证性能 |
| R3 PLQ full-run | **不建议** | R3-4b 已结论 PLQ 饱和，继续无收益 |
| R5 correction | **不建议** | R4 full-run 结果未出前不进入 R5 |
| R6 memory consistency | **仅做 lite scan** | 不进入 full implementation，仅概念验证扫描 |

### 10.4 推荐下一步操作顺序

1. **准备 R2 full-run 配置**：新建 `pd_root` 目录，准备配置 YAML（包含 seed、GPU、启动命令）
2. **执行 R2 full-run**：等待完整训练，读取 `metrics.jsonl`，确认 mIoU vs baseline
3. **若 R2 稳定**，准备 R2+R4 full-run 配置
4. **执行 R2+R4 full-run**：对比 R2 alone，评估 proto_gate 的增量收益
5. **GPU 闲置时可并行做 R6-lite scan-only**：只做概念验证，不接入训练

---

## 最终结论

```
READY_FOR_FULLRUN_DECISION
```

R4 proto_gate 5-step diagnostic 已完全收口：
- 5/5 step 正常运行
- 过滤机制按设计精确工作
- 字段命名歧义已识别并有明确处理建议
- 诊断 artifact（val mIoU=60.22）已标注，不作为结论
- 字段审计完成，后续论文字段用法已规范

由用户决定是否启动 full-run 以及时间安排。

