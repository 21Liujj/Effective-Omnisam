# Step 20/21 Diagnostic Visualization Report — TeCMA Phase 7.2 + 7.3

## Status

This step was executed as a **diagnostic-only, no-training, no-re-evaluation** resume step. I scanned for reusable prediction masks / eval outputs / visualization outputs first.

Important limitation: canonical prediction masks for the four requested configs were **not found**. Therefore, confirmed qualitative comparison and confirmed failure cases cannot be produced without a separate inference/eval export. I did **not** run training, pseudo-label generation, validation, or inference.

## Requested configs and prediction-mask availability

| Config | Canonical source checked | Prediction masks found? | Count | Action taken |
|---|---|---:|---:|---|
| A_ccpm | `experiments/tecma/runs/A_ccpm_seed1_20260616_104911/DensePASS/sam2_s/Cityscapes_13cls/epoch1/gtFine` | No | 0 | No re-eval; marked missing |
| A_Dtrain | `experiments/tecma/runs/A_Dtrain_seed1_20260616_111019/DensePASS/sam2_s/Cityscapes_13cls/epoch1/gtFine` | No | 0 | No re-eval; marked missing |
| A_Etrain_keep025 | `experiments/tecma/runs/token_budget/A_Etrain_keep025_seed1_20260617_114629/DensePASS/sam2_s/Cityscapes_13cls/epoch1/gtFine` | No | 0 | No re-eval; marked missing |
| A_Dtrain_Etrain | `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/DensePASS/sam2_s/Cityscapes_13cls/epoch1/gtFine` | No | 0 | No re-eval; marked missing |

## Reusable but non-canonical masks found

These are not used as final Phase 7.2/7.3 comparison evidence because they are DensePASS training pseudo-label style outputs (`used_path.txt` contains city `.jpg` names, 2000 images) or a different config (`A_Etrain` default, not `A_Etrain_keep025`). They can only be used as auxiliary references, not for the requested validation-set four-config comparison.

| Asset | Count | Why not used as canonical comparison |
|---|---:|---|
| `A_ccpm_PL_train_masks` | 2000 | train/PL-style DensePASS city outputs, not canonical val predictions from `A_ccpm_seed1_20260616_104911` |
| `A_Dtrain_PL_train_masks` | 2000 | train/PL-style DensePASS city outputs, not canonical val predictions from `A_Dtrain_seed1_20260616_111019` |
| `A_Etrain_default_PL_train_masks` | 2000 | `A_Etrain` default, not requested `A_Etrain_keep025` |
| `A_Dtrain_Etrain_PLon_train_masks` | 2000 | PL-on/train-mask run, not canonical final `A_Dtrain_Etrain_seed1_20260616_121135` val predictions |

## What was exported

Because prediction masks are missing, I exported **candidate diagnostic sample panels** from existing validation input + ground-truth labels only. These are not model-comparison failure cases.

Output directories:

- `experiments/tecma/figures/qualitative_diagnostic/`
- `experiments/tecma/figures/failure_cases_diagnostic/`

Each qualitative sample directory contains:

- `input_panorama.png`
- `ground_truth_color.png`
- `ground_truth_focus_overlay.png` for traffic light / traffic sign / pole / fence / terrain
- `seam_candidate_overlay.png` marking left/right ERP edge strips as candidates only
- `diagnostic_candidate_panel.png`
- `missing_predictions.txt`

## Class/color mapping

The class mapping is confirmed from `validation.py::DATASET_NAME2CLASSES` for DensePASS/Cityscapes 13-class setting:

`road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, car`.

Colorization uses a fixed Cityscapes-style palette for visualization only.

## Diagnostic sample table

| Sample | Main Classes | Error Type | A_ccpm behavior | A_Dtrain behavior | A_Etrain behavior | Final behavior | Implication for PLQ |
|---|---|---|---|---|---|---|---|
| 122 | traffic light / traffic sign / pole / terrain | Candidate: small object missing + boundary complex candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | small-object and boundary-aware confidence calibration |
| 155 | traffic light / fence / pole / terrain | Candidate: small object missing + thin structure candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | avoid suppressing sparse object tokens under compression |
| 289 | traffic light / pole / fence / building | Candidate: small object missing + boundary erosion candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | class-conditional confidence for tiny vertical structures |
| 101 | traffic sign / fence / terrain / person | Candidate: class confusion + thin boundary candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | traffic-sign/pole/fence calibration; boundary-aware PL filtering |
| 111 | traffic sign / pole / terrain / person | Candidate: small object missing + class confusion candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | rare-object confidence calibration |
| 263 | pole / traffic sign / wall / road | Candidate: thin structure broken candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | thin-structure preservation in memory-token selection |
| 68 | pole / traffic sign / road / sky | Candidate: small object missing candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | pole/sign recall guard |
| 319 | fence / building / road / sky | Candidate: thin structure broken + boundary erosion candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | fence boundary calibration |
| 305 | fence / wall / building / sky | Candidate: thin structure broken + boundary erosion candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | boundary-aware confidence for fence/wall |
| 24 | terrain / vegetation / sky | Candidate: terrain boundary + seam candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | terrain/vegetation boundary calibration |
| 210 | terrain / road / building / sky | Candidate: terrain boundary + seam candidate | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | seam-aware large-region consistency check |
| 363 | building / road / traffic sign / sky | Candidate: seam candidate + large background stable reference | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | Not assessable: canonical prediction mask directory missing; no inference/eval run. | seam-consistency audit; not confirmed as seam failure |

## Failure-case status

No **confirmed** failure cases were exported. The directory `experiments/tecma/figures/failure_cases_diagnostic/` contains candidate panels and `failure_case_candidates.csv` only.

Candidate failure types queued for later visual confirmation after prediction export:

| Sample | Candidate failure type | Reason |
|---|---|---|
| 122 | small object missing | rich traffic light/sign/pole GT content |
| 319 | thin structure broken | fence-rich GT content |
| 305 | boundary erosion | fence/wall boundary-rich GT content |
| 210 | seam inconsistency | terrain + edge-region candidate; seam failure not automatically confirmed |
| 101 | class confusion | traffic sign/fence/person candidate |
| 24 | pseudo-label noise suspected | terrain/vegetation boundary candidate; PL noise only a hypothesis |

## What is needed to complete confirmed Phase 7.2/7.3

To produce the requested side-by-side panels and confirmed failure cases, run a separate inference/eval export for the four canonical checkpoints and save prediction masks without overwriting existing outputs. Suggested target layout:

```text
experiments/tecma/eval_predictions/
  A_ccpm/gtFine/<sample>_labelTrainIds.png
  A_Dtrain/gtFine/<sample>_labelTrainIds.png
  A_Etrain_keep025/gtFine/<sample>_labelTrainIds.png
  A_Dtrain_Etrain/gtFine/<sample>_labelTrainIds.png
```

Use the existing best checkpoints reported in each `run_summary.json`. This should be inference/evaluation only, not training.

## PLQ implications from candidate selection only

Even without predictions, the sample candidates align with Phase 7.1 per-class IoU drops: traffic light/sign/pole, fence, and terrain. Once predictions are exported, PLQ calibration should be checked for:

1. small-object recall preservation (`traffic light`, `traffic sign`, `pole`);
2. thin/boundary structure continuity (`fence`, `pole`, `wall`);
3. terrain/vegetation/road boundary ambiguity;
4. seam consistency at ERP left/right edges, but only after automated or manual seam-localized comparison.

## Strict non-actions

- Did not modify model/training code.
- Did not run training.
- Did not run validation/inference/eval.
- Did not overwrite existing prediction masks.
- Did not claim seam failure or model failure without predictions.
