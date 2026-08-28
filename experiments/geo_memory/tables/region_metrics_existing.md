# Existing Result Region Metrics

Values are percentages. GT uses DensePASS13 mapping before evaluation; ignore_index=255 is excluded.

| Method | Overall mIoU | Seam mIoU | Polar mIoU | Equator mIoU | Polar Error | Source Path |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GM0 baseline_omnisam | N/A | N/A | N/A | N/A | N/A | missing prediction path(s): experiments/tecma/eval_predictions_confirmed/baseline_omnisam, experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/predictions, experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/debug_pred_vis; gt=data/DensePASS/gtFine/val |
| GM1 tecmem_base / A_Dtrain_Etrain | 51.49 | 45.57 | 57.19 | 52.35 | 14.67 | pred=experiments/tecma/eval_predictions_confirmed/A_Dtrain_Etrain; gt=data/DensePASS/gtFine/val; evaluated=12 matched masks; missing_pred=88 |

## Input Inventory

| Method | Selected Prediction Dir | GT Dir | matched | missing_pred | missing_gt | size_mismatch | Dry-run | Full Eval |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| GM0 baseline_omnisam | N/A | data/DensePASS/gtFine/val | 0 | 100 | 0 | 0 | N/A | N/A |
| GM1 tecmem_base / A_Dtrain_Etrain | experiments/tecma/eval_predictions_confirmed/A_Dtrain_Etrain | data/DensePASS/gtFine/val | 12 | 88 | 0 | 0 | rc=0, status=ok | rc=0, status=ok |
