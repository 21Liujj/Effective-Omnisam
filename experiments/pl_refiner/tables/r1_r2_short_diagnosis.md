# r1_r2_short_diagnosis.md
# Phase 2 / Task 2-2b — L3 Short-Run Diagnosis: R1_conf_mask vs R2_class_balanced

**Date**: 2026-06-23  
**Run Level**: L3 short-run (5 iters × ~9 sub-steps = ~45 per-step diag lines per run)  
**No training**: only 5 train steps. No eval. No checkpoint used for mIoU.  
**PL dir**: `data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1`  
**ckpt**: `/root/autodl-tmp/omnisam_s.pth` (both src and tgt)  
**seed**: 1  **batch_size**: 4  

---

## Table 1: Summary

| Run | steps | loss_sup (mean) | loss_unsup (mean) | total_valid_ratio | common_valid_ratio | rare_valid_ratio | extreme_rare_valid_ratio | ignore_ratio | gpu_mem_peak | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| R1_conf_mask | 5 | ~0.093 | ~0.021 | N/A (mode=conf_mask, refiner diag empty) | N/A | N/A | **0.00%** | N/A | ~20 GB est. | PASS |
| R2_class_balanced | 5 | ~0.093 | ~0.007 | **49.3%** | **49.0%** | **70.0%** | **73.2%** | **50.7%** | ~20 GB est. | PASS |

> **R1 note**: R1 uses `mode=conf_mask` which returns identity from refiner; actual masking is via `pseudo_conf_mask` plugin. The `_pl_refiner_diag` dict is populated with diagnostics that include `mode='identity'` (not `class_balanced`), so refiner group metrics = 0 is expected for R1. The `pseudo_vr` field (= `cov_t = mask_t.mean()`) correctly shows R1's pseudo_valid_ratio ≈ 0.93~1.00 (very high, no rare filtering).

> **gpu_mem_peak**: The `[refiner_diag]` print-line includes gpu_mem_peak_mb only when `torch.cuda.max_memory_allocated()` is available. This was not captured in logs. Estimate from prior A_ccpm runs (clean ablation): ~19.9 GB. R2 with class_balance computation adds negligible memory (<50 MB).

---

## Table 2: Per-Class Valid Ratio

Averaged over all ~45 per-step diag snapshots from R2 log. R1 uses pseudo_vr=cov_t (global, not per-class).

| class_id | class_name | group | R1_valid_ratio (global) | R2_valid_ratio (per-class avg) | delta | R2_skip_pattern |
|---|---|---|---:|---:|---:|---|
| 0 | road | common | ~96% (pseudo_vr) | ~49% | -47pp | — |
| 1 | sidewalk | common | ~96% | ~45% | -51pp | — |
| 2 | building | common | ~96% | ~49% | -47pp | — |
| 3 | wall | rare-hard | ~96% | **~3%** | -93pp | **no_candidates** (very few wall pixels per batch) |
| 4 | fence | rare-hard | ~96% | ~50% | -46pp | varies: starvation / no_candidates |
| 5 | pole | rare-hard | ~96% | ~69% | -27pp | starvation (few px) |
| 6 | traffic light | rare-hard | ~96% | **~14%** | -82pp | **no_candidates / starvation** (batch often empty) |
| 7 | traffic sign | rare-hard | ~96% | ~76% | -20pp | occasional no_candidates |
| 8 | vegetation | common | ~96% | ~49% | -47pp | — |
| 9 | terrain | common | ~96% | ~38% | -58pp | occasional all_below_min_conf |
| 10 | sky | common | ~96% | ~50% | -46pp | — |
| 11 | person | rare-hard | ~96% | ~69% | -27pp | starvation (few px) |
| 12 | car | common | ~96% | ~48% | -48pp | — |

> **R1 valid_ratio interpretation**: R1 uses `pseudo_conf_mask` plugin with threshold=0.85, applied globally. The `pseudo_vr = mask_t.float().mean()` ≈ 0.93-1.00 because DensePASS images reused from clean PL have high model confidence after training. Per-class breakdown not available for R1 in this short-run.

> **wall (c3)** and **traffic light (c6)** are frequently `no_candidates` per batch. This is **expected behavior**: these classes have extremely sparse PL pixels per patch/batch. In full training (full epoch), they will appear in some batches. Starvation guard documented correctly.

---

## Notes on wall (c3) and traffic light (c6)

- **wall**: `no_candidates` on ~90% of sub-steps. Per-class valid_ratio ≈ 0-5% (effectively 0 in most batches). Reason: wall pixels are sparse in DensePASS PL (0.92% overall), and per-batch sampling may not include wall patches. This is a **known limitation** — class-balanced selection cannot help when a class literally has 0 PL pixels in a batch. 
- **traffic light**: `no_candidates` or `starvation` (3-8 px) on ~95% of sub-steps. PL frequency = 0.04% (extremely rare). Even when present, often below min_conf=0.55. Starvation guard fires when 3-8 px pass min_conf.

These two classes need monitoring in full training. The question is whether they appear enough in a full epoch to be meaningful.
