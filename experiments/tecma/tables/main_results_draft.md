# TeCMA 主结果表（初版 draft）— main_results_draft.md

> 由 `main_results_registry.md` 派生。主结果来自已有 full DensePASS protocol（final_2000），**非**短程 sanity。
> 短程 sanity 的 mIoU/BFScore 不进入本表。
> Generated: 2026-06-15 (Task 3-3 revised)

---

## 主结果表（Cityscapes→DensePASS, Seed=1, Epoch=1）

| Method | Config | Dataset | Seed | Epoch | mIoU | BFScore | ΔmIoU | ΔBFScore | Step Time | Speedup | Memory Attention Time | Attention Speedup | Memory Tokens | Token Reduction | Peak Memory | Memory Saving | Status |
| ------ | ------ | ------- | ---- | ----- | ---- | ------- | ----- | -------- | --------- | ------- | --------------------- | ----------------- | ------------- | --------------- | ----------- | ------------- | ------ |
| baseline / base_last | baseline_omnisam | Cityscapes→DensePASS | 1 | 1 | 60.26 | 39.97 | — (ref) | — (ref) | 5.27 s/it | 1.00× (ref) | 263.0 s | 1.00× (ref) | 32768 | — (ref) | 26.82 GB | — (ref) | existing_verified_main_result |
| A_Dtrain_Etrain | A_Dtrain_Etrain | Cityscapes→DensePASS | 1 | 1 | 60.16 | 39.86 | −0.10 | −0.11 | 4.08 s/it | 1.29× | 93.6 s | 2.81× | 4096 | 8.0× (−87.5%) | 25.34 GB | −1.48 GB (−5.5%) | existing_verified_main_result |

---

## 衍生量计算说明

相对 baseline（对照组）：

| 指标 | baseline | A_Dtrain_Etrain | 计算 | 结果 |
|---|---|---|---|---|
| ΔmIoU | 60.26 | 60.16 | 60.16 − 60.26 | **−0.10**（精度几乎无损） |
| ΔBFScore | 39.97 | 39.86 | 39.86 − 39.97 | **−0.11** |
| Speedup (step) | 5.27 s/it | 4.08 s/it | 5.27 / 4.08 | **1.29×**（−22.6%） |
| Attention Speedup | 263.0 s | 93.6 s | 263.0 / 93.6 | **2.81×**（−64.4%） |
| Token Reduction | 32768 | 4096 | 32768 / 4096 | **8.0×**（−87.5%） |
| Memory Saving | 26.82 GB | 25.34 GB | 26.82 − 25.34 | **−1.48 GB**（−5.5%） |

---

## 关键结论

A_Dtrain_Etrain 相对 baseline：**精度近乎持平（mIoU −0.10, BFScore −0.11），但显著降低计算与显存开销** —— memory token 8× 压缩、memory attention 2.81× 提速、整体 step 1.29× 提速、峰值显存省 1.48 GB。这是 D（空间记忆压缩）+ E（稀疏记忆检索）模块的核心效率收益。

## 数据溯源

- mIoU / Step Time / Memory Attention Time / Memory Tokens / Peak Memory：来自各实验 `train_timing.jsonl` + `metrics.jsonl`，已在 registry 核对。
- BFScore：来自训练日志 `[ValSummary] BFScore=` 行（见 `step_09_bfscore_evidence_audit.md`，Evidence Status = verified_from_train_log）。

## 限制（须随表注明）

- **Single seed（seed=1）、单 epoch（epoch=1）**：尚无方差估计；论文最终表建议补 seed2（见主结果说明报告）。
- 本表为 draft，未覆盖 registry；任何修改先改 registry 再重新派生本表。
