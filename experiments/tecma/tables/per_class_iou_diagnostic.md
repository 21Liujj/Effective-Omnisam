# Per-class IoU Diagnostic Table — TeCMA Phase 7.1

单位：IoU 0–100。Delta 列均以 `A_ccpm` 为参照。

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

## Sources
- `A_ccpm`: `experiments/tecma/runs/A_ccpm_seed1_20260616_104911/run.log`
- `A_Dtrain`: `experiments/tecma/runs/A_Dtrain_seed1_20260616_111019/run.log`
- `A_Etrain_keep025`: `experiments/tecma/runs/token_budget/A_Etrain_keep025_seed1_20260617_114629/run.log`
- `A_Dtrain_Etrain`: `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/run.log`
