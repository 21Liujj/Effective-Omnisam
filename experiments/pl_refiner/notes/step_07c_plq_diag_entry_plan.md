# step_07c: PLQ Diagnostic Entry Plan — Task 3-1 Rescue

**Date**: 2026-06-24  
**Status**: code complete, awaiting user manual 5-step run  
**Run Level**: L3-style diagnostic substitute (not full training, not L4)

---

## 1. 为什么停止 standalone L2 diagnosis 路线

`experiments/pl_refiner/scripts/l2_plq_forward_diagnosis.py` 反复出现：

- `ImportError: cannot import name 'DensePASS' from 'dataset.DensePASS'`
- standalone dataloader 路径问题（模块导入顺序、sys.path 设置）
- 修复多次后仍不稳定，消耗时间超过预期

根本原因：standalone 脚本脱离了 `run_adaptation.py` 的 build_loaders / build_model 初始化链，
需要独立复现 dataloader + model setup，容易产生 import / path 差异。

---

## 2. 为什么改用 run_adaptation.py 真实入口

- `run_adaptation.py` 已经被 R1/R2 short-run 验证可稳定运行
- 真实训练入口已经有 conf_t / pred_t / pseudo_label / valid_mask 在 `strategies.py` 插入点可用
- `refiner.py` 的 `mode=plq_basic` 已实现，无需重新搭建 dataloader
- 只需补 diagnostics logging 即可复用整条 pipeline

---

## 3. 本次是否改代码

**是**，只补 diagnostics logging，严格不改 loss / optimizer / backbone：

| 文件 | 修改内容 | 是否改 loss |
|------|---------|------------|
| `adaptation/pseudo_label/refiner.py` | `_refine_plq_basic` diagnostics 补全（nan_count, zero_ratio, class-group means, histogram, per-class PLQ, valid_ratio_by_bin） | No |
| `adaptation/strategies.py` | `if _pl_refiner_diag:` 块内补 `mode==plq_basic` PLQ fields → extra_metrics；补 `[plq_diag]` print | No |
| `adaptation/trainer.py` | 增加 `plq_diag_records` 列表；每 step 从 `iter_metrics` 收集 PLQ 字段 | No |
| `adaptation/engine.py` | epoch 结束后写 `experiments/pl_refiner/results/r3_plq_5step_diag.jsonl` | No |

---

## 4. 本次是否改 loss

**No**。`pl_weight`（PLQ score）在 `_refine_plq_basic` 里已计算为 `PseudoLabelRefineResult.pl_weight`，
但 `strategies.py` 的 loss 计算路径**不使用** `pl_weight`（未接入 CE weighting）。
本次只是诊断 PLQ 值域、分布、NaN，不触碰 loss。

---

## 5. 本次是否训练

**不是完整训练**。用户后续手动执行 5-step diagnostic，不是 full-run。
目的是观察 PLQ 在真实 batch 上的值域和分布，替代失败的 standalone L2 diagnosis。

---

## 6. 已实现的 diagnostics 字段

每 step 记录到 `r3_plq_5step_diag.jsonl`（每行一个 JSON，含以下字段）：

| 字段 | 来源 |
|------|------|
| `mean_plq` (= `plq_mean`) | refiner diagnostics |
| `plq_min`, `plq_max`, `plq_std` | refiner diagnostics |
| `confidence_mean` | refiner diagnostics |
| `entropy_reliability_mean` | refiner diagnostics |
| `local_agreement_mean` | refiner diagnostics |
| `plq_nan_count` | refiner diagnostics |
| `plq_zero_ratio` | refiner diagnostics |
| `plq_common_mean`, `plq_rare_mean`, `plq_extreme_rare_mean` | refiner diagnostics (from cb_diag group ids) |
| `plq_by_class` → `plq_cls{c}_mean` | refiner diagnostics |
| `plq_histogram` → `plq_hist_{lo}-{hi}` | refiner diagnostics (10 bins) |
| `valid_ratio_by_plq_bin` → `plq_vr_bin_{lo}-{hi}` | refiner diagnostics (10 bins) |
| `loss_sup_s`, `loss_unsup` | extra_metrics (upstream) |
| `refiner_total_valid_ratio` | strategies refiner diag block |
| `refiner_rare_valid_ratio` | strategies refiner diag block |
| `gpu_mem_peak_mb` | strategies refiner diag block |
| `epoch`, `iter`, `global_step` | trainer |

---

## 7. 用户下一步应执行的命令

```bash
cd /root/autodl-tmp/OmniSAM-main
bash experiments/pl_refiner/scripts/run_r3_plq_5step_diag.sh
```

---

## 8. 成功后应该生成哪些文件

| 文件 | 内容 |
|------|------|
| `experiments/pl_refiner/results/r3_plq_5step_diag.jsonl` | 5 行 JSON，每行一 step 的 PLQ diagnostics |
| `experiments/pl_refiner/notes/l3_r3_plq_5step_diag.log` | 完整 stdout log，含 `[plq_diag]` 和 `[refiner_diag]` 行 |
| `experiments/pl_refiner/sanity_runs/R3_plq_basic_5step/` | checkpoint（sanity dir，不影响 final 结果） |

---

## 9. 如果命令失败，用户应该贴回什么错误

请贴回以下内容：

1. 报错的完整 traceback（最后 30 行）
2. 关键词：`ImportError` / `KeyError` / `AttributeError` / `CUDA out of memory`
3. log 文件末尾：`tail -50 experiments/pl_refiner/notes/l3_r3_plq_5step_diag.log`

---

## 10. 已修改文件列表（便于 git diff 检查）

```
adaptation/pseudo_label/refiner.py
adaptation/strategies.py
adaptation/trainer.py
adaptation/engine.py
experiments/pl_refiner/scripts/run_r3_plq_5step_diag.sh  (新增)
experiments/pl_refiner/notes/step_07c_plq_diag_entry_plan.md  (本文件)
```
