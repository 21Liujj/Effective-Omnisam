# Phase 4-clean Ablation Results

Date: 2026-06-16  
Project: OmniSAM / TeCMA  
Setting: Cityscapes → DensePASS, seed=1, fixed canonical pseudo labels

## 1. Scope

This file records the official Phase 4-clean ablation results for TeCMA. The clean ablation evaluates three modules under a fair fixed-pseudo-label setting:

- **A / CCPM** (`pseudo_conf_mask`): Confidence-Calibrated Pseudo-label Masking
- **D / FSMC** (`efficient_tam_memory`): FoV-aware Structured Memory Compression
- **E / SG-SMR** (`efficient_sam2_sparse`): Saliency-Guided Sparse Memory Retrieval

The following six runs are included:

| Config | Run directory |
|---|---|
| baseline_omnisam | `experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834` |
| A_ccpm | `experiments/tecma/runs/A_ccpm_seed1_20260616_104911` |
| A_Dtrain | `experiments/tecma/runs/A_Dtrain_seed1_20260616_111019` |
| A_Etrain | `experiments/tecma/runs/A_Etrain_seed1_20260616_113023` |
| Dtrain_Etrain | `experiments/tecma/runs/Dtrain_Etrain_seed1_20260616_115113` |
| A_Dtrain_Etrain | `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135` |

## 2. Source files and evidence

For each run, the following files were checked:

- `run_summary.json`
- `run.log`
- `checkpoints/Cityscapes_13cls/sam2_s/metrics/train_timing.jsonl`

Important notes:

1. `run_summary.json` currently stores `BFScore: null`.
2. The correct BFScore was extracted from the final validation line in `run.log`:
   ```text
   [ValSummary] BFScore=...
   ```
3. `pl_timing_jsonl` is `null` for these clean runs. This is expected because Phase 4-clean reused existing pseudo labels instead of regenerating them.
4. Each of the six runs contains the fixed-PL evidence line:
   ```text
   Reuse fixed pseudo-label dir for epoch 1: data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
   ```
   The checked count is 1 occurrence per run.

All values below match the provided expected clean-ablation reference values.

## 3. Official clean ablation table

| Config | mIoU | BFScore | mAcc | PixAcc | FWIoU | mDice | Step Time (s/it) | Iter Compute (s/it) | Mem. Attn. Time (s) | Peak GPU Mem (GB) | Mem. Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_omnisam | 60.26 | 39.59 | 71.71 | 88.63 | 80.73 | 72.84 | 2.3505 | 1.5573 | 117.4825 | 19.5405 | 32768 |
| A_ccpm | 60.26 | 39.59 | 71.72 | 88.64 | 80.73 | 72.85 | 2.4089 | 1.5815 | 117.5635 | 19.9780 | 32768 |
| A_Dtrain | 60.12 | 39.23 | 71.56 | 88.58 | 80.65 | 72.71 | 2.2938 | 1.4540 | 60.9970 | 18.9858 | 8192 |
| A_Etrain | 60.09 | 39.23 | 71.53 | 88.53 | 80.57 | 72.69 | 2.3798 | 1.5388 | 79.6928 | 19.2829 | 16384 |
| Dtrain_Etrain | 60.02 | 39.23 | 71.52 | 88.57 | 80.65 | 72.59 | 2.3237 | 1.4687 | 52.3798 | 18.3393 | 4096 |
| A_Dtrain_Etrain | 60.02 | 39.23 | 71.53 | 88.57 | 80.66 | 72.60 | 2.3153 | 1.4643 | 51.9189 | 18.7768 | 4096 |

## 4. Delta vs. baseline

Baseline: mIoU 60.26, BFScore 39.59, memory attention time 117.4825s, peak GPU memory 19.5405GB, memory tokens 32768.

| Config | ΔmIoU | ΔBFScore | Mem. Attn. Reduction | Token Reduction | Peak Mem Saving | Step Time Change |
|---|---:|---:|---:|---:|---:|---:|
| baseline_omnisam | +0.00 | +0.00 | 0.0% | 1.0× | 0.00 GB | +0.0% |
| A_ccpm | +0.00 | +0.00 | -0.1% | 1.0× | -0.44 GB | +2.5% |
| A_Dtrain | -0.14 | -0.36 | 48.1% | 4.0× | 0.55 GB | -2.4% |
| A_Etrain | -0.17 | -0.36 | 32.2% | 2.0× | 0.26 GB | +1.2% |
| Dtrain_Etrain | -0.24 | -0.36 | 55.4% | 8.0× | 1.20 GB | -1.1% |
| A_Dtrain_Etrain | -0.24 | -0.36 | 55.8% | 8.0× | 0.76 GB | -1.5% |

Notes:

- Positive memory saving means lower peak GPU memory than baseline.
- Negative memory-attention reduction for `A_ccpm` means a tiny increase relative to baseline, consistent with no token reduction.
- Step time is whole training iteration time; memory-attention time isolates the measured memory-attention component.

## 5. Module-level interpretation

### A / CCPM

Under fixed high-fidelity pseudo labels, `A_ccpm` almost does not change mIoU or BFScore relative to baseline:

- mIoU: 60.26 → 60.26
- BFScore: 39.59 → 39.59
- memory tokens: unchanged at 32768
- memory attention time: effectively unchanged, 117.48s → 117.56s

Therefore, CCPM should **not** be described as a significant accuracy-improving module in this clean setting. A more accurate description is that CCPM acts as a conservative confidence filter / pseudo-label stability guard.

### D / FSMC

`A_Dtrain` reduces memory tokens from 32768 to 8192, i.e. a 4× reduction, and lowers memory-attention time from 117.48s to 61.00s, a 48.1% reduction. The mIoU drop is 0.14. This indicates that FSMC is the major efficiency contributor.

### E / SG-SMR

`A_Etrain` reduces memory tokens from 32768 to 16384, i.e. a 2× reduction, and lowers memory-attention time by 32.2%. Its efficiency gain is clear but weaker than FSMC when used alone.

### D + E

`Dtrain_Etrain` and `A_Dtrain_Etrain` both reach 4096 memory tokens, i.e. an 8× token reduction. The final `A_Dtrain_Etrain` configuration reduces memory-attention time from 117.48s to 51.92s, a 55.8% reduction, while mIoU drops by 0.24 and BFScore drops by 0.36.

## 6. Short conclusion

Phase 4-clean supports the core TeCMA claim under fixed pseudo labels: TeCMA can substantially reduce memory-token volume and memory-attention training cost while approximately preserving segmentation accuracy and boundary consistency.

The strongest official configuration, `A_Dtrain_Etrain`, achieves:

- 8× memory-token reduction: 32768 → 4096
- 55.8% memory-attention time reduction: 117.48s → 51.92s
- only 0.24 mIoU degradation: 60.26 → 60.02
- only 0.36 BFScore degradation: 39.59 → 39.23
- 0.76GB peak GPU memory saving: 19.54GB → 18.78GB

Recommended paper wording:

> Under fixed pseudo labels, TeCMA reduces memory tokens by 8× and memory-attention time by 55.8%, while incurring only 0.24 mIoU and 0.36 BFScore degradation.

## 7. Phase 4-v0 vs. Phase 4-clean

- **Phase 4-v0**: five configurations completed, but `REUSE_PL_DIR` was not actually propagated, so each configuration regenerated its own pseudo labels. It can be used as preliminary sanity/trend analysis only.
- **Phase 4-clean**: six configurations reused the same canonical pseudo-label directory. This is the official ablation candidate and should be used as the basis for the paper Table X.

Do not mix v0 and clean results in the final formal ablation table.

## 8. PLon note

`A_Dtrain_Etrain_PLon` was already run in Phase 4-v0. It is a negative ablation for enabling D/E during pseudo-label generation and therefore cannot reuse the canonical fixed pseudo labels by design. It is reasonable that Phase 4-clean does not include PLon. If needed, PLon can be reported separately as supplementary / negative ablation, but it should not be mixed into the fixed-PL clean table.
