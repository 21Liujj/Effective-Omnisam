# Pseudo-label Refiner Full-run Result Summary

## Main Results

| Method | mIoU | mAcc | BFScore |
|---|---:|---:|---:|
| Efficient + Identity PL | 60.02 | 71.71 | 39.35 |
| Efficient + R2 Class-balanced PL | 60.07 | 72.28 | 39.54 |
| Efficient + R2 Class-balanced + R4 Proto-gate | 60.05 | 72.24 | 39.57 |

## Delta Summary

| Comparison | d_mIoU | d_mAcc | d_BFScore |
|---|---:|---:|---:|
| R2 - R0 | 0.05 | 0.57 | 0.19 |
| R2R4 - R2 | -0.02 | -0.05 | 0.03 |
| R2R4 - R0 | 0.03 | 0.53 | 0.22 |

## Interpretation

- R2 class-balanced filtering is the primary effective PL refinement module.
- R2 maintains mIoU and improves mAcc / BFScore under the memory-efficient setting.
- R4 proto-gate slightly improves BFScore over R2 but does not further improve mIoU.
- Therefore, R2 should be treated as the main PL module, and R4 should be positioned as a boundary-oriented auxiliary diagnostic/refinement module.

## Evidence Paths

| Run | Metrics Path | Log Path |
|---|---|---|
| R0 | `experiments/pl_refiner/fullruns/R0_identity_20260625_135318/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | `experiments/pl_refiner/notes/fullrun_R0_identity_20260625_135318.log` |
| R2 | `experiments/pl_refiner/fullruns/R2_class_balanced_20260625_135318/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | `experiments/pl_refiner/notes/fullrun_R2_class_balanced_20260625_135318.log` |
| R2R4 | `experiments/pl_refiner/fullruns/R2R4_proto_gate_20260625_154537/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | `experiments/pl_refiner/notes/fullrun_R2R4_proto_gate_20260625_154537.log` |

## Generated Files

- `experiments/pl_refiner/tables/pl_fullrun_main_metrics.md`
- `experiments/pl_refiner/tables/pl_fullrun_main_metrics.csv`
- `experiments/pl_refiner/tables/pl_fullrun_per_class_iou.md`
- `experiments/pl_refiner/tables/pl_fullrun_per_class_iou.csv`
- `experiments/pl_refiner/tables/pl_fullrun_per_class_delta.md`
- `experiments/pl_refiner/tables/pl_fullrun_diag_summary.md`
- `experiments/pl_refiner/tables/pl_fullrun_diag_summary.csv`
