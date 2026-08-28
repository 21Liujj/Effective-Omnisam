# r2_class_balance_forward_diagnosis.md
# Phase 2 / Task 2-2a — L2 Forward Diagnosis: R1 vs R2

**Date**: 2026-06-23  
**Run Level**: L2 forward diagnosis (no training, no checkpoint, no mIoU)  
**N_SAMPLES**: 50 PL images (from base_last_20260425_232941 epoch1)  
**Confidence proxy**: per-class heuristic (class_conf_proxy dict), not from model  

## Per-Class Table

| class_id | class_name | group | top_ratio | min_conf | R1 cand | R1 selected | R1 valid_ratio | R2 cand | R2 selected | R2 valid_ratio | skipped? | skip_reason |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | road | common | 0.50 | 0.75 | 31,207,606 | 27,617,060 | 88.49% | 31,207,606 | 15,603,793 | 50.00% | . | — |
| 1 | sidewalk | common | 0.55 | 0.70 | 17,543,237 | 399,258 | 2.28% | 17,543,237 | 8,119,149 | 46.28% | . | — |
| 2 | building | common | 0.50 | 0.75 | 67,672,431 | 63,966,331 | 94.52% | 67,672,431 | 33,836,203 | 50.00% | . | — |
| 3 | wall | rare-hard | 0.75 | 0.60 | 42,109 | 68 | 0.16% | 42,109 | 30,882 | 73.34% | Y | no_candidates |
| 4 | fence | rare-hard | 0.70 | 0.60 | 792,234 | 3,639 | 0.46% | 792,234 | 550,004 | 69.42% | Y | no_candidates |
| 5 | pole | rare-hard | 0.75 | 0.60 | 1,086,681 | 1,489 | 0.14% | 1,086,681 | 796,614 | 73.31% | Y | starvation_kept_all_3 |
| 6 | traffic light | rare-hard | 0.80 | 0.55 | 681 | 0 | 0.00% | 681 | 461 | 67.69% | Y | no_candidates |
| 7 | traffic sign | rare-hard | 0.80 | 0.55 | 370,291 | 0 | 0.00% | 370,291 | 272,321 | 73.54% | Y | no_candidates |
| 8 | vegetation | common | 0.50 | 0.75 | 41,593,162 | 34,998,721 | 84.15% | 41,593,162 | 20,796,570 | 50.00% | Y | no_candidates |
| 9 | terrain | common | 0.55 | 0.70 | 4,584,044 | 164,280 | 3.58% | 4,584,044 | 2,231,036 | 48.67% | Y | no_candidates |
| 10 | sky | common | 0.50 | 0.80 | 30,026,652 | 29,610,732 | 98.61% | 30,026,652 | 15,013,315 | 50.00% | . | — |
| 11 | person | rare-hard | 0.75 | 0.60 | 1,168,282 | 320,664 | 27.45% | 1,168,282 | 876,205 | 75.00% | Y | starvation_kept_all_40 |
| 12 | car | common | 0.50 | 0.75 | 9,339,517 | 8,583,978 | 91.91% | 9,339,517 | 4,669,744 | 50.00% | . | — |

## Summary

| Run | total_valid_ratio | common_valid_ratio | rare_valid_ratio | extreme_rare_valid_ratio | status |
|---|---:|---:|---:|---:|---|
| R1_conf_mask | 80.64% | 81.87% | 9.42% | 0.00% | reference |
| R2_class_balanced | 50.04% | 49.65% | 73.01% | 73.53% | target |
| Δ (R2-R1) | -30.60pp | -32.22pp | +63.60pp | +73.53pp | — |

## Checks

- [PASS] rare classes not all zero: rare sel counts: [30882, 550004, 796614, 461, 272321, 876205]
- [PASS] common valid_ratio <= 85%: common_ratio=49.65%
- [PASS] no NaN in r2 ratios: clean
- [PASS] traffic light skip documented: no_candidates
- [PASS] traffic sign skip documented: no_candidates
- [PASS] R2 candidate set == original valid (no re-activation): OK

## Notes

- Confidence values are from a **heuristic proxy** (no model forward pass).
  Real confidence will come from the OmniSAM model during L3 short-run.
- The goal of L2 is to verify structural correctness: class distribution,
  skip handling, no NaN, no re-activation of invalid pixels.
- mIoU should NOT be inferred from this diagnosis.