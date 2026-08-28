# Token Budget Speed-Accuracy Trade-off Table

> Generated: 2026-06-17 | Seed=1 | DensePASS val (100 imgs) | canonical PL reused

## Summary

| Budget | Config | Curve | Mem Tokens | Token Ratio | Reduction | mIoU | ΔmIoU | BFScore | ΔBFScore | mAcc | PixAcc | FWIoU | mDice |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100% | A_ccpm | E-only curve | 32,768 | 1.0000 | 1.00× | 60.26 | +0.00 | 39.59 | +0.00 | 71.72 | 88.64 | 80.73 | 72.85 |
| 75% | A_Etrain_keep075 | E-only curve | 24,576 | 0.7500 | 1.33× | 60.12 | -0.14 | 39.28 | -0.31 | 71.62 | 88.54 | 80.58 | 72.73 |
| 50% | A_Etrain | E-only curve | 16,384 | 0.5000 | 2.00× | 60.09 | -0.17 | 39.23 | -0.36 | 71.53 | 88.53 | 80.57 | 72.69 |
| 25% | A_Etrain_keep025 | E-only curve | 8,192 | 0.2500 | 4.00× | 60.10 | -0.16 | 39.29 | -0.30 | 71.62 | 88.56 | 80.62 | 72.67 |
| final (12.5%) | A_Dtrain_Etrain | Combined D+E final endpoint | 4,096 | 0.1250 | 8.00× | 60.02 | -0.24 | 39.23 | -0.36 | 71.53 | 88.57 | 80.66 | 72.60 |

## Speed & Memory

| Budget | Config | Step Time (s) | Step Δ% | Attn Time (s) | Attn Reduction | GPU Peak (GB) | GPU Save (GB) | GPU Save % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100% | A_ccpm | 2.4089 | +0.00% | 117.5635 | +0.00% | 19.98 | 0.00 | +0.00% |
| 75% | A_Etrain_keep075 | 2.4614 | +2.18% | 98.6208 | +16.11% | 19.64 | 0.34 | +1.70% |
| 50% | A_Etrain | 2.3798 | -1.21% | 79.6928 | +32.21% | 19.28 | 0.70 | +3.48% |
| 25% | A_Etrain_keep025 | 2.2941 | -4.77% | 60.5447 | +48.50% | 18.93 | 1.05 | +5.26% |
| final (12.5%) | A_Dtrain_Etrain | 2.3153 | -3.89% | 51.9189 | +55.84% | 18.78 | 1.20 | +6.01% |

## Notes

- **E-only curve**: 100% / 75% / 50% / 25% — only E (SG-SMR / sparse retrieval) active, D off.
- **Combined D+E final endpoint**: 12.5% — D (FSMC) + E (SG-SMR) both active; D compresses 8192→2048 tokens (compress_ratio=0.5, pool), E keeps top-50% of those (4096). Not a point on the E-only curve.
- All new runs reuse canonical PL from `data_pd_exp_final_2000/base_last_20260425_232941/...` (evidence in run.log).
- ΔmIoU / ΔBFScore relative to 100% baseline (A_ccpm).
- step_time Δ% relative to 100% baseline; **75% is slightly slower than 100%** (+2.17%), likely due to sparse retrieval overhead + measurement noise — see report §8.
