# latest_status.md
# PseudoLabelRefiner 当前状态（2026-06-25 更新）

---

## 当前阶段：Phase 4 / Task 4-4c 完成

**状态**: `READY_FOR_FULLRUN_DECISION`

---

## 已完成阶段总结

| 阶段 | 任务 | 结论 | 文档 |
|------|------|------|------|
| R2 | class-balanced pseudo-label selection | ✓ 实现完成，diagnostic PASS | step_06–step_10 |
| R3 | PLQ basic diagnostic | PLQ 饱和，不进入 full-run | step_09 |
| R4-4a | proto_gate filter unit test | ✓ 单元测试全通过 | step_13 |
| R4-4b | proto_gate 5-step real entry diagnostic | ✓ 5/5 step 通过，PASS | step_14（待补）|
| R4-4c | 结果收口 + 字段审计 + full-run 决策 | ✓ 完成，READY_FOR_FULLRUN_DECISION | step_15 |

---

## 当前关键指标（R4-4b diagnostic，不是 full-run 结果）

| 指标 | 值 |
|------|----|
| proto_gate_filter_ratio_total | 0.0991（mean，5 steps） |
| proto_gate_filter_ratio_common | 0.1000（mean） |
| proto_gate_filter_ratio_rare | 0.0504（mean） |
| proto_gate_filter_ratio_extreme_rare | 0.0000（fallback keep） |
| r2_total_valid_ratio | 0.4687（mean） |
| proto_gate_final_valid_ratio | 0.4290（mean） |
| loss_unsup | 0.0009（mean，stable） |
| nan_count | 0（所有步骤） |

---

## 不允许引用的数据

- `val mIoU=60.22`：diagnostic artifact（max_train_iters=5 触发的自动 val），不是 R4 结果

---

## 字段命名规范（已审计）

**推荐用于论文/报告的字段**：
- R2 前置：`r2_total_valid_ratio`, `r2_common_valid_ratio`, `r2_rare_valid_ratio`, `r2_extreme_rare_valid_ratio`
- R4 后置：`proto_gate_final_valid_ratio`, `proto_gate_final_common_valid_ratio`, `proto_gate_final_rare_valid_ratio`, `proto_gate_final_extreme_rare_valid_ratio`

**不建议论文使用**：`total_valid_ratio_after_R2`, `rare_valid_ratio_after_R2`, `extreme_rare_valid_ratio_after_R2`, `refiner_total_valid_ratio`

---

## 下一步

1. 准备 R2 full-run 配置（新 `pd_root` 目录）
2. 执行 R2 full-run
3. 视 R2 结果决定是否执行 R2+R4 full-run
4. GPU 闲置时可做 R6 lite scan（不进入 full implementation）

---

## 严格禁止

- 不执行 R3 full-run
- 不进入 R5 correction
- 不进入 R6 full implementation（仅 lite scan）
- 不覆盖 `data_pd_exp_final_2000/` 目录

