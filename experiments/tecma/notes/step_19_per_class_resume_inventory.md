# Step 19 Per-class IoU Resume Inventory

Date: 2026-06-18

本报告是 Phase 7.1 的断点恢复盘点。仅扫描已有产物；未运行训练、未生成伪标签、未启动评估。

## Scan scope

- Requested directories: `experiments/tecma/tables/`, `experiments/tecma/notes/`, `experiments/tecma/raw_results/`, `experiments/tecma/logs/`, `experiments/tecma/profiled_results/`.
- Observed: `tables/` and `notes/` exist; requested `raw_results/`, `logs/`, `profiled_results/` are absent in `experiments/tecma/`.
- Additional relevant location found by recursive scan: `experiments/tecma/runs/**/run.log`.
- Keywords scanned: `per-class`, `per class`, `IoU`, `road`, `sidewalk`, `building`, `pole`, `traffic`, `person`, `rider`, `mIoU`, `BFScore`.

## Inventory table

| Config | Found per-class IoU? | Source File | Complete Classes? | Need Re-eval? | Notes |
|---|---|---|---|---|---|
| `A_ccpm` | Yes | `experiments/tecma/runs/A_ccpm_seed1_20260616_104911/run.log` | Yes, 13/13 | No | Contains `========== per-class IoU (0-100) ==========` block; final val line mIoU=60.26, BFScore=39.59. |
| `A_Dtrain` | Yes | `experiments/tecma/runs/A_Dtrain_seed1_20260616_111019/run.log` | Yes, 13/13 | No | Contains `========== per-class IoU (0-100) ==========` block; final val line mIoU=60.12, BFScore=39.23. |
| `A_Etrain_keep025` | Yes | `experiments/tecma/runs/token_budget/A_Etrain_keep025_seed1_20260617_114629/run.log` | Yes, 13/13 | No | Contains `========== per-class IoU (0-100) ==========` block; final val line mIoU=60.10, BFScore=39.29. |
| `A_Dtrain_Etrain` | Yes | `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/run.log` | Yes, 13/13 | No | Contains `========== per-class IoU (0-100) ==========` block; final val line mIoU=60.02, BFScore=39.23. |

## Existing Phase 7.1 assets found before this resume step

- `experiments/tecma/tables/per_class_iou.csv`
- `experiments/tecma/tables/per_class_iou.md`
- `experiments/tecma/notes/step_19_per_class_iou_report.md`

These existing files compare `baseline_omnisam`, `A_ccpm`, and `A_Dtrain_Etrain`. They do **not** include the full requested 4-config diagnostic columns for `A_Dtrain` and `A_Etrain_keep025`; therefore new diagnostic files are generated without overwriting existing assets.

## Class mapping status

Class mapping is confirmed from `validation.py::DATASET_NAME2CLASSES` for the DensePASS / Cityscapes 13-class setting:

road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, car.

`rider`, `bicycle`, and `motorcycle` are not part of the current 13-class setting; they are not analyzed here.

## Resume decision

All four requested configs already have complete per-class IoU blocks in existing `run.log` files. Therefore **no re-evaluation is needed** and **no training is allowed or performed**. Phase 7.1 can proceed directly to table/report generation from the recovered logs.
