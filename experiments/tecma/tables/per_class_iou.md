# Per-class IoU Analysis for TeCMA

## Source

Per-class IoU values are extracted from `run.log` files:

- `experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/run.log`
- `experiments/tecma/runs/A_ccpm_seed1_20260616_104911/run.log`
- `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/run.log`

The class mapping is confirmed from `validation.py::DATASET_NAME2CLASSES` for DensePASS / Cityscapes 13-class setting.

## Class mapping

| id | class |
|---:|---|
| 0 | road |
| 1 | sidewalk |
| 2 | building |
| 3 | wall |
| 4 | fence |
| 5 | pole |
| 6 | traffic light |
| 7 | traffic sign |
| 8 | vegetation |
| 9 | terrain |
| 10 | sky |
| 11 | person |
| 12 | car |

Note: the current 13-class setting does not include `rider`, `bicycle`, or `motorcycle`, so these classes cannot be analyzed in this per-class IoU table.

## Full per-class IoU comparison

| class_id | class_name | baseline_omnisam_IoU | A_ccpm_IoU | A_Dtrain_Etrain_IoU | delta_IoU | delta_vs_baseline | rank_drop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | road | 80.23 | 80.25 | 80.25 | 0.00 | 0.02 | 10 |
| 1 | sidewalk | 47.33 | 47.35 | 47.62 | 0.27 | 0.29 | 12 |
| 2 | building | 89.67 | 89.67 | 89.53 | -0.14 | -0.14 | 7 |
| 3 | wall | 43.77 | 43.78 | 44.40 | 0.62 | 0.63 | 13 |
| 4 | fence | 48.92 | 48.92 | 48.15 | -0.77 | -0.77 | 2 |
| 5 | pole | 43.64 | 43.64 | 43.36 | -0.28 | -0.28 | 5 |
| 6 | traffic light | 27.88 | 27.88 | 26.23 | -1.65 | -1.65 | 1 |
| 7 | traffic sign | 32.52 | 32.52 | 32.23 | -0.29 | -0.29 | 4 |
| 8 | vegetation | 78.77 | 78.77 | 78.63 | -0.14 | -0.14 | 8 |
| 9 | terrain | 45.15 | 45.15 | 44.76 | -0.39 | -0.39 | 3 |
| 10 | sky | 94.61 | 94.61 | 94.62 | 0.01 | 0.01 | 11 |
| 11 | person | 67.50 | 67.50 | 67.42 | -0.08 | -0.08 | 9 |
| 12 | car | 83.33 | 83.34 | 83.08 | -0.26 | -0.25 | 6 |

## Largest drops: A_Dtrain_Etrain vs A_ccpm

| class_name | A_ccpm_IoU | A_Dtrain_Etrain_IoU | delta_IoU | note |
| --- | --- | --- | --- | --- |
| traffic light | 27.88 | 26.23 | -1.65 | small/sparse object; inspect in qualitative/failure cases |
| fence | 48.92 | 48.15 | -0.77 | thin structure / boundary-sensitive class |
| terrain | 45.15 | 44.76 | -0.39 | regular semantic class |
| traffic sign | 32.52 | 32.23 | -0.29 | small/sparse object; inspect in qualitative/failure cases |
| pole | 43.64 | 43.36 | -0.28 | small/sparse object; inspect in qualitative/failure cases |

## Largest gains / most stable classes

| class_name | A_ccpm_IoU | A_Dtrain_Etrain_IoU | delta_IoU | note |
| --- | --- | --- | --- | --- |
| wall | 43.78 | 44.40 | 0.62 | thin structure / boundary-sensitive class |
| sidewalk | 47.35 | 47.62 | 0.27 | regular semantic class |
| sky | 94.61 | 94.62 | 0.01 | large-area semantic class; mostly stable |
| road | 80.25 | 80.25 | 0.00 | large-area semantic class; mostly stable |
| person | 67.50 | 67.42 | -0.08 | small/sparse object; inspect in qualitative/failure cases |

Strictly positive gains are observed on `wall`, `sidewalk`, and `sky`. `road` remains unchanged.

## Small-object analysis

| class_name | A_ccpm_IoU | A_Dtrain_Etrain_IoU | delta_IoU | note |
| --- | --- | --- | --- | --- |
| pole | 43.64 | 43.36 | -0.28 | small/sparse object; inspect in qualitative/failure cases |
| traffic light | 27.88 | 26.23 | -1.65 | small/sparse object; inspect in qualitative/failure cases |
| traffic sign | 32.52 | 32.23 | -0.29 | small/sparse object; inspect in qualitative/failure cases |
| person | 67.50 | 67.42 | -0.08 | small/sparse object; inspect in qualitative/failure cases |

The small-object classes do not collapse systematically. The largest degradation appears on `traffic light` (-1.65 IoU), while `traffic sign` (-0.29), `pole` (-0.28), and `person` (-0.08) remain relatively stable.

## Implication for qualitative visualization

Phase 7-2 should prioritize:

1. Failure-oriented cases: traffic light, traffic sign, pole, fence.
2. Boundary / thin-structure cases: fence and wall.
3. Stable positive examples: wall, sidewalk, sky, road.
4. Small-object stability examples: person.

## Paper-ready conclusion

Compared with the full-memory A_ccpm reference, TeCMA reduces memory tokens substantially while only slightly changing per-class IoU. The overall mIoU decreases by 0.24, and most classes remain stable. The largest class-level degradation occurs on the small and sparse `traffic light` category, suggesting that aggressive memory-token reduction may affect tiny objects more than large-area classes. This motivates the qualitative and failure-case analysis in the next stage.
