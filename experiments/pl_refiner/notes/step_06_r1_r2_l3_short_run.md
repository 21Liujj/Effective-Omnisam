# step_06_r1_r2_l3_short_run.md
# Phase 2 / Task 2-2b — R1 vs R2 L3 Short-Run 记录

**Run Level**: L3 short-run（5 train steps，no eval，no checkpoint for mIoU）  
**日期**: 2026-06-24（补记 2026-06-23 已完成结果）  
**执行人**: Claude Code  
**是否跑训练**: Yes，L3 极短训练诊断（R1/R2 各 5 steps，不是 L4 full-run）  
**是否改动 pseudo label**: No（读取既有 PL，不覆盖、不重生成）  
**是否改动 loss**: No  
**是否保存 checkpoint**: No

---

## 1. 本次做了什么

- 补记 2026-06-23 已完成的 **R1_conf_mask vs R2_class_balanced** L3 short-run 结果。
- 对比 R1 与 R2 在前 5 个 train steps 内的：
  - `loss_sup`
  - `loss_unsup`
  - `pseudo_valid_ratio / total_valid_ratio`
  - rare / common 类有效像素趋势
  - NaN / CUDA OOM / shape mismatch 风险
- 生成并确认诊断表：
  - `experiments/pl_refiner/tables/r1_r2_short_diagnosis.md`
- 确认 R2 可以进入下一阶段 R3 前的状态总结。

---

## 2. 修改了哪些文件

| 文件 | 类型 | 变更内容 |
|---|---|---|
| `experiments/pl_refiner/tables/r1_r2_short_diagnosis.md` | 已有输出 | R1/R2 L3 short-run 对比表 |
| `experiments/pl_refiner/notes/l3_r1_conf_mask.log` | 已有日志 | R1 5-step short-run 原始日志 |
| `experiments/pl_refiner/notes/l3_r2_class_balanced.log` | 已有日志 | R2 5-step short-run 原始日志 |
| `experiments/pl_refiner/notes/step_06_r1_r2_l3_short_run.md` | **新增** | 本 step 记录文件 |

**未修改**：backbone、MemoryBank、strategies loss 公式、dataset loader、optimizer、checkpoint、已有 pseudo label PNG。

---

## 3. L3 Short-Run 设置

| 项 | 值 |
|---|---|
| Run Level | L3 short-run |
| steps | R1/R2 各 5 train steps |
| eval | No |
| checkpoint for mIoU | No |
| PL dir | `data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1` |
| ckpt | `/root/autodl-tmp/omnisam_s.pth`（src/tgt） |
| seed | 1 |
| batch_size | 4 |

注意：这是稳定性/诊断 short-run，不生成论文 mIoU，不等价于 L4 full-run。

---

## 4. 关键结果

来源：`experiments/pl_refiner/tables/r1_r2_short_diagnosis.md`

| Run | steps | loss_sup mean | loss_unsup mean | total_valid_ratio | common_valid_ratio | rare_valid_ratio | extreme_rare_valid_ratio | ignore_ratio | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R1_conf_mask | 5 | ~0.093 | ~0.021 | N/A（refiner diag empty） | N/A | N/A | 0.00% | N/A | PASS |
| R2_class_balanced | 5 | ~0.093 | ~0.007 | **49.3%** | **49.0%** | **70.0%** | **73.2%** | **50.7%** | PASS |

观察：

1. R2 `loss_sup` 与 R1 基本一致（~0.093），说明 source supervised 分支未被 R2 破坏。
2. R2 `loss_unsup` 低于 R1（~0.007 vs ~0.021），符合 R2 对 common 类更严格筛选、有效像素减少的预期。
3. R2 `total_valid_ratio` ≈ 49.3%，与 L2 proxy 诊断的 50.04% 接近。
4. R2 rare 类有效比例明显提高，短程平均约 70%。
5. short-run 未观察到 NaN、CUDA OOM、明显 shape mismatch。

---

## 5. Per-Class 现象摘要

R2 per-class valid ratio 在 short-run 中呈现：

| class_id | class_name | group | R2 valid ratio 现象 | 风险 |
|---:|---|---|---|---|
| 0 | road | common | ~49% | 符合 top_ratio=0.50 |
| 1 | sidewalk | common | ~45% | 符合预期 |
| 2 | building | common | ~49% | 符合预期 |
| 3 | wall | rare-hard | ~3%，多数 batch no_candidates | **需要后续监控** |
| 4 | fence | rare-hard | ~50% | 可接受 |
| 5 | pole | rare-hard | ~69%，常触发 starvation | 可接受但需监控 |
| 6 | traffic light | rare-hard | ~14%，常 no_candidates/starvation | **需要后续监控** |
| 7 | traffic sign | rare-hard | ~76% | 可接受 |
| 8 | vegetation | common | ~49% | 符合预期 |
| 9 | terrain | common | ~38% | 偶有 all_below_min_conf |
| 10 | sky | common | ~50% | 符合预期 |
| 11 | person | rare-hard | ~69%，常触发 starvation | 可接受但需监控 |
| 12 | car | common | ~48% | 符合预期 |

关键风险点：

- `wall` 与 `traffic light` 在 5-step batch 内非常稀疏。
- 这是 DensePASS PL 类分布与 batch sampling 共同导致的，R2 无法在 batch 中凭空创造不存在的类像素。
- L4 前必须继续监控 full epoch 或更长 short-run 中这些类是否长期 starvation。

---

## 6. 是否满足 Task 2-2b 验收

| 条件 | 状态 |
|---|---|
| R2 YAML 可跑 5 steps | PASS |
| 记录 `loss_sup` | PASS |
| 记录 `loss_unsup` | PASS |
| 记录 pseudo/total valid ratio | PASS |
| R1 vs R2 前 5 步有对比 | PASS |
| total_valid_ratio 接近 50% | PASS（49.3%） |
| 无 CUDA OOM | PASS |
| 无 NaN | PASS |
| 不跑完整训练 | PASS |
| 不保存 checkpoint | PASS |
| 不覆盖 PL | PASS |

---

## 7. 当前风险

| 风险 | 严重度 | 说明 |
|---|---|---|
| wall / traffic light batch 内极稀疏 | 中 | 5-step 内多数 batch 无候选或触发 starvation，需在后续更长诊断或 L4 前确认 |
| R2 loss_unsup 下降 | 中 | 有效像素减少后 unsup loss 变小，可能影响 UDA 强度，需要后续 R3/R4 或 L4 观察 mIoU |
| R1 per-class valid ratio 不完整 | 低 | R1 主要由 pseudo_conf_mask plugin 处理，refiner diag 不提供 per-class 统计；不影响 R2 验收 |
| gpu_mem_peak 未完整采集 | 低 | 日志未捕获峰值字段，估计约 20GB；R2 计算额外显存可忽略 |

---

## 8. 下一步建议

进入 **R3_plq_basic**，但仍必须按施工方案顺序执行：

1. 先 L2，不跑完整训练。
2. 实现或检查：
   - `PLQ = confidence × entropy_reliability × local_agreement`
   - `use_plq_weighted_ce = true`
3. 必须输出：
   - `mean_plq`
   - `plq_histogram`
   - `plq_by_class`
   - `loss_unsup_mean`
4. L2 通过后，再考虑 R3 L3 short-run。
5. 没有明确写 L4 前，禁止启动完整训练。

---

## 9. 结论

R2 `class_balanced` 已完成：

- L2 forward diagnosis PASS
- L3 5-step short-run PASS
- R2 总体 valid ratio 符合设计（约 50%）
- rare 类覆盖明显改善（约 70%）
- source supervised 分支未受影响
- 没有 NaN/OOM/shape mismatch

因此可以进入下一阶段 **R3_plq_basic 的 L2 施工与诊断**。
