# Step 19 Per-class IoU Diagnostic Report — TeCMA Phase 7.1

## Status

Phase 7.1 is resumed and completed from existing artifacts. No training, pseudo-label generation, or validation/evaluation was executed. All values below are extracted from existing `run.log` per-class IoU blocks.

## Compared configs

- `A_ccpm`: `experiments/tecma/runs/A_ccpm_seed1_20260616_104911/run.log`; mIoU=60.26, BFScore=39.59
- `A_Dtrain`: `experiments/tecma/runs/A_Dtrain_seed1_20260616_111019/run.log`; mIoU=60.12, BFScore=39.23
- `A_Etrain_keep025`: `experiments/tecma/runs/token_budget/A_Etrain_keep025_seed1_20260617_114629/run.log`; mIoU=60.10, BFScore=39.29
- `A_Dtrain_Etrain`: `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/run.log`; mIoU=60.02, BFScore=39.23

## Class mapping

Confirmed 13-class mapping from `validation.py::DATASET_NAME2CLASSES`:

road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, car.

`rider`, `bicycle`, and `motorcycle` are not included in this 13-class setting; they are therefore not analyzed. If a later 19-class protocol is used, the mapping must be re-confirmed before extending this table.

## Final diagnostic table

| Class | A_ccpm | A_Dtrain | A_Etrain_keep025 | A_Dtrain_Etrain | Delta_A_Dtrain | Delta_A_Etrain_keep025 | Delta_final_vs_A |
|---|---:|---:|---:|---:|---:|---:|---:|
| road | 80.25 | 80.26 | 80.11 | 80.25 | +0.01 | -0.14 | +0.00 |
| sidewalk | 47.35 | 47.51 | 47.16 | 47.62 | +0.16 | -0.19 | +0.27 |
| building | 89.67 | 89.49 | 89.51 | 89.53 | -0.18 | -0.16 | -0.14 |
| wall | 43.78 | 43.75 | 44.69 | 44.40 | -0.03 | +0.91 | +0.62 |
| fence | 48.92 | 48.83 | 48.66 | 48.15 | -0.09 | -0.26 | -0.77 |
| pole | 43.64 | 43.37 | 43.30 | 43.36 | -0.27 | -0.34 | -0.28 |
| traffic light | 27.88 | 27.04 | 27.17 | 26.23 | -0.84 | -0.71 | -1.65 |
| traffic sign | 32.52 | 32.53 | 31.67 | 32.23 | +0.01 | -0.85 | -0.29 |
| vegetation | 78.77 | 78.72 | 78.63 | 78.63 | -0.05 | -0.14 | -0.14 |
| terrain | 45.15 | 44.96 | 44.67 | 44.76 | -0.19 | -0.48 | -0.39 |
| sky | 94.61 | 94.62 | 94.58 | 94.62 | +0.01 | -0.03 | +0.01 |
| person | 67.50 | 67.56 | 67.69 | 67.42 | +0.06 | +0.19 | -0.08 |
| car | 83.34 | 82.88 | 83.40 | 83.08 | -0.46 | +0.06 | -0.26 |

## 1. Largest drops: `A_Dtrain_Etrain` vs `A_ccpm`

| Rank | Class | A_ccpm | A_Dtrain_Etrain | Delta_final_vs_A | Likely category |
|---:|---|---:|---:|---:|---|
| 1 | traffic light | 27.88 | 26.23 | -1.65 | small object / sparse; pseudo-label noise sensitive |
| 2 | fence | 48.92 | 48.15 | -0.77 | thin boundary / boundary-sensitive |
| 3 | terrain | 45.15 | 44.76 | -0.39 | boundary/seam-adjacent candidate; label ambiguity possible |
| 4 | traffic sign | 32.52 | 32.23 | -0.29 | small object / sparse; pseudo-label noise sensitive |
| 5 | pole | 43.64 | 43.36 | -0.28 | small/thin object; boundary-sensitive |

## 2. Largest gains / most stable classes: `A_Dtrain_Etrain` vs `A_ccpm`

| Rank | Class | A_ccpm | A_Dtrain_Etrain | Delta_final_vs_A | Notes |
|---:|---|---:|---:|---:|---|
| 1 | wall | 43.78 | 44.40 | +0.62 | strict positive gain; thin/boundary class but improved in this run |
| 2 | sidewalk | 47.35 | 47.62 | +0.27 | strict positive gain |
| 3 | sky | 94.61 | 94.62 | +0.01 | near-stable positive |
| 4 | road | 80.25 | 80.25 | +0.00 | exactly stable |
| 5 | person | 67.50 | 67.42 | -0.08 | near-stable, not a strict gain |

Strictly positive gains are only observed for `wall` (+0.62), `sidewalk` (+0.27), and `sky` (+0.01). `road` is unchanged. The fifth row is a near-stable class, not a true gain.

## 3. Module-specific damage patterns

### `A_Dtrain` damage vs `A_ccpm`

| Rank | Class | A_ccpm | A_Dtrain | Delta_A_Dtrain | Interpretation |
|---:|---|---:|---:|---:|---|
| 1 | traffic light | 27.88 | 27.04 | -0.84 | largest D-only loss; small/sparse object likely sensitive to spatial memory compression |
| 2 | car | 83.34 | 82.88 | -0.46 | regular foreground object; possible detail loss but still high absolute IoU |
| 3 | pole | 43.64 | 43.37 | -0.27 | small/thin object |
| 4 | terrain | 45.15 | 44.96 | -0.19 | boundary/label-ambiguity candidate |
| 5 | building | 89.67 | 89.49 | -0.18 | large class, small magnitude loss |

Main `A_Dtrain`损伤集中在 `traffic light` (-0.84)，其次是 `car` (-0.46)、`pole` (-0.27)、`terrain` (-0.19)、`building` (-0.18)。这说明 D/FSMC 的空间压缩对小目标最敏感，但也会带来少量普通类别波动。

### `A_Etrain_keep025` damage vs `A_ccpm`

| Rank | Class | A_ccpm | A_Etrain_keep025 | Delta_A_Etrain_keep025 | Interpretation |
|---:|---|---:|---:|---:|---|
| 1 | traffic sign | 32.52 | 31.67 | -0.85 | largest E-only 25% loss; sparse retrieval may discard rare sign tokens |
| 2 | traffic light | 27.88 | 27.17 | -0.71 | small/sparse object |
| 3 | terrain | 45.15 | 44.67 | -0.48 | boundary/label-ambiguity candidate |
| 4 | pole | 43.64 | 43.30 | -0.34 | small/thin object |
| 5 | fence | 48.92 | 48.66 | -0.26 | thin boundary-sensitive class |

Main `A_Etrain_keep025`损伤集中在 `traffic sign` (-0.85) 和 `traffic light` (-0.71)，说明 aggressive sparse retrieval 在 25% token budget 下对稀疏小目标更敏感。

## 4. Categories relevant to small objects, boundaries, seam, and pseudo-label noise

- **Small / sparse objects**: `traffic light`, `traffic sign`, `pole`, `person`. 本表中最明显的是 `traffic light`；`person` 基本稳定。
- **Boundary / thin structures**: `fence`, `wall`, `pole`, and partly `traffic sign`/`traffic light`. `fence` 是 final vs A 的第二大下降；`wall` 反而提升，说明边界类不是系统性崩塌。
- **Seam-related candidates**: 本表没有逐图 seam 定位，不能直接证明 seam 影响。可作为后续可视化候选的类别包括 `road`, `sidewalk`, `building`, `vegetation`, `terrain`, `sky` 等横向连续区域；其中 `terrain` 有 -0.39 drop，值得在 ERP seam 附近样本中检查。
- **Pseudo-label noise candidates**: `traffic light`, `traffic sign`, `pole`, `fence`, `terrain`。理由是这些类面积小、边界薄或标签边界含混，容易受置信度过滤、memory token selection 或 PL 噪声影响。该归因是诊断假设，仍需 Phase 7.2 的定性样本验证。

## Conclusion

The final `A_Dtrain_Etrain` configuration mainly loses IoU on sparse small objects and thin/boundary-sensitive classes, especially `traffic light` (-1.65) and `fence` (-0.77). Large-area classes remain stable (`road` +0.00, `sky` +0.01, `building` -0.14, `vegetation` -0.14). `A_Dtrain` primarily hurts `traffic light`, while `A_Etrain_keep025` primarily hurts `traffic sign` and `traffic light`. This supports using the next pseudo-label quality calibration module to emphasize small-object and boundary-aware confidence calibration rather than globally changing all classes.

## Output files

- `experiments/tecma/notes/step_19_per_class_resume_inventory.md`
- `experiments/tecma/tables/per_class_iou_diagnostic.csv`
- `experiments/tecma/tables/per_class_iou_diagnostic.md`
- `experiments/tecma/notes/step_19_per_class_iou_diagnostic_report.md`
