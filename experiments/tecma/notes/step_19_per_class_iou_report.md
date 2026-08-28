# Step 19: Per-class IoU Report

## Status

Phase 7-1 is completed.

No training, pseudo-label generation, or validation was executed. This step only consolidates per-class IoU values already present in existing `run.log` files.

## Compared runs

Main comparison:

- Full-memory reference: `A_ccpm`
- Final TeCMA: `A_Dtrain_Etrain`

Optional reference:

- `baseline_omnisam`

## Per-class IoU availability

Per-class IoU exists in `run.log` files rather than standalone JSON/CSV files.

Sources:

- `experiments/tecma/runs/A_ccpm_seed1_20260616_104911/run.log`
- `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/run.log`
- `experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/run.log`

## Class mapping

The class mapping is confirmed from `validation.py::DATASET_NAME2CLASSES`.

The current setting uses 13 classes:

road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, car.

The classes `rider`, `bicycle`, and `motorcycle` are not included in the current 13-class setting.

## Largest drops

| class_name | A_ccpm_IoU | A_Dtrain_Etrain_IoU | delta_IoU |
| --- | --- | --- | --- |
| traffic light | 27.88 | 26.23 | -1.65 |
| fence | 48.92 | 48.15 | -0.77 |
| terrain | 45.15 | 44.76 | -0.39 |
| traffic sign | 32.52 | 32.23 | -0.29 |
| pole | 43.64 | 43.36 | -0.28 |

## Largest gains / most stable classes

| class_name | A_ccpm_IoU | A_Dtrain_Etrain_IoU | delta_IoU |
| --- | --- | --- | --- |
| wall | 43.78 | 44.40 | 0.62 |
| sidewalk | 47.35 | 47.62 | 0.27 |
| sky | 94.61 | 94.62 | 0.01 |
| road | 80.25 | 80.25 | 0.00 |
| person | 67.50 | 67.42 | -0.08 |

## Small-object conclusion

The observable small-object classes are `pole`, `traffic light`, `traffic sign`, and `person`.

- `traffic light`: -1.65
- `traffic sign`: -0.29
- `pole`: -0.28
- `person`: -0.08

Small objects do not collapse systematically, but `traffic light` is the most sensitive class and should be emphasized in failure-case analysis.

## Paper interpretation

The per-class IoU analysis supports a conservative claim: TeCMA maintains most semantic categories while substantially reducing memory tokens. The largest degradation occurs on a sparse small-object class, which is consistent with the intuition that aggressive memory reduction may remove some fine-grained cues.

Recommended wording:

> Per-class IoU shows that most categories remain stable under token-efficient memory adaptation. The largest degradation appears on traffic light, a sparse and small-object category, while large-area classes such as road, sky, and building are nearly unchanged. This suggests that TeCMA preserves global semantic consistency but may still lose fine details for tiny objects under aggressive token reduction.

## Guidance for Phase 7-2 qualitative visualization

Prioritize samples containing:

1. traffic light;
2. traffic sign;
3. pole;
4. fence;
5. person;
6. road / sky / sidewalk as stable positive references.

## Output files

- `experiments/tecma/tables/per_class_iou.csv`
- `experiments/tecma/tables/per_class_iou.md`
- `experiments/tecma/notes/step_19_per_class_iou_report.md`
