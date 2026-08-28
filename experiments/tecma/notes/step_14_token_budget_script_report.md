# Step 14: Phase 5-2 Token-Budget Config & Script Report

Scope: generate configs + runner script for the two new E-only token-budget
points of Plan B (from `step_13_token_budget_plan.md`). **No training, no PL
generation, no validation, no queue launch.** Verification limited to `bash -n`,
`check_configs.py`, and `DRY_RUN=1`.

## 1. New YAML configs

| File | Budget point | Derived from |
|---|---|---|
| `experiments/tecma/configs/A_Etrain_keep075.yaml` | 75% / ~24576 tokens | `A_Etrain.yaml` |
| `experiments/tecma/configs/A_Etrain_keep025.yaml` | 25% / ~8192 tokens  | `A_Etrain.yaml` |

Only `efficient_sam2_sparse.keep_ratio` differs from the `A_Etrain` template
(0.5 → 0.75 / 0.25). All other experiment knobs (dataset, seed, backbone,
epochs, A threshold) are untouched.

## 2. New script

`experiments/tecma/scripts/run_token_budget_experiments.sh` — a **thin wrapper**.
It does NOT duplicate training logic; it delegates each config to the existing
`run_main_experiments.sh`, which already implements `--reuse_pl_dir`, `DRY_RUN`,
`BACKGROUND`, anti-overwrite timestamped dirs, and `run_summary.json`.

Wrapper responsibilities:
- isolates output to `experiments/tecma/runs/token_budget/` (never touches clean ablation in `runs/<CONFIG>_...`);
- forwards `REUSE_PL_DIR`, warns loudly if empty, errors if the dir doesn't exist;
- runs configs serially; supports single point, multi point, `DRY_RUN=1`, and `BACKGROUND=1` (detached queue);
- prints every delegated command with `printf '%q'` — no argument concatenation.

`check_configs.py` was minimally patched: two acceptance entries added
(`A_Etrain_keep075`, `A_Etrain_keep025`) asserting A on / D off / E train-on /
E PL-off / keep_ratio. The matrix itself auto-scans `configs/*.yaml`, so no
scan change was needed. Backup saved as `check_configs.py.bak_tokenbudget`.

## 3. Key switch table (both new configs)

| Switch | A_Etrain_keep075 | A_Etrain_keep025 |
|---|---|---|
| `pseudo_conf_mask.enabled` (A) | true | true |
| `pseudo_conf_mask.threshold` | 0.85 | 0.85 |
| `efficient_tam_memory.enabled` (D) | false | false |
| `efficient_tam_memory.apply_in_train` | false | false |
| `efficient_tam_memory.apply_in_pl` | false | false |
| `efficient_sam2_sparse.enabled` (E) | true | true |
| `efficient_sam2_sparse.keep_ratio` | **0.75** | **0.25** |
| `efficient_sam2_sparse.apply_in_train` | true | true |
| `efficient_sam2_sparse.apply_in_pl` | false | false |

D off, E train-only, E off in PL — matches Plan B exactly.

## 4. REUSE_PL_DIR pass-through

Confirmed. In dry-run, both delegated train commands end with:
`--reuse_pl_dir data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1`.
The wrapper forwards `REUSE_PL_DIR` to `run_main_experiments.sh`, which appends
the flag as separate array elements (no string interpolation). If `REUSE_PL_DIR`
is empty, the wrapper prints a warning and the recommended PL_DIR command.

## 5. bash -n results

| Script | Result |
|---|---|
| `run_token_budget_experiments.sh` | OK |
| `run_main_experiments.sh` | OK |
| `run_queue.sh` | OK |

## 6. check_configs.py result

11 configs checked, **0 validation errors**; 38 acceptance checks, **0 failed → PASS**.
New points confirmed: `A_Etrain_keep075` keep_ratio=0.75, `A_Etrain_keep025`
keep_ratio=0.25, both A on / D off / E train-on / E PL-off.

## 7. Dry-run result

`DRY_RUN=1 REUSE_PL_DIR=$PL_DIR SEED=1 ... A_Etrain_keep075 A_Etrain_keep025`:
- no real training started; no `Pseudolabel Generation`; no `Epoch 1 [Train]`;
- each command uses its own config and ends with `--reuse_pl_dir $PL_DIR`;
- output dirs under `runs/token_budget/<CONFIG>_seed1_<STAMP>/`;
- command print has **no argument concatenation** (every flag space-separated);
- queue summary: `A_Etrain_keep075=ok`, `A_Etrain_keep025=ok`, `fail=0`.

## 8. Phase 5-3 real-run command (DO NOT run here)

```bash
cd /root/autodl-tmp/OmniSAM-main
PL_DIR=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
REUSE_PL_DIR=$PL_DIR SEED=1 bash experiments/tecma/scripts/run_token_budget_experiments.sh \
  A_Etrain_keep075 A_Etrain_keep025
```

Background (detached queue, returns immediately):

```bash
cd /root/autodl-tmp/OmniSAM-main
PL_DIR=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
BACKGROUND=1 REUSE_PL_DIR=$PL_DIR SEED=1 bash experiments/tecma/scripts/run_token_budget_experiments.sh \
  A_Etrain_keep075 A_Etrain_keep025
# log: experiments/tecma/runs/token_budget/_bg_logs/token_budget_seed1_<STAMP>.log
```

## 9. Expected time & space

Per run reuses canonical PL (PL stage skipped, the ~2h cost is avoided). Train
is the only cost: ≈25–35 min/epoch on the confirmed 2000-PL / 1-epoch scale
(E-only train, no PL regen). Two runs ≈ 50–70 min serial. Disk: one checkpoint
set + logs per run, on the order of the existing A_Etrain run directories
(hundreds of MB each); written under `runs/token_budget/`. **(estimate — confirm
from logs after the real run.)**

## 10. Risks

- **Empty REUSE_PL_DIR**: would silently trigger PL regen and break curve comparability. Mitigated by an explicit warning; always pass `PL_DIR`.
- **PL dir path drift**: if the canonical PL dir is moved/renamed, the wrapper errors out (dir-exists check) rather than regenerating.
- **keep_ratio → realized tokens** is approximate (24576 / 8192 are expected, not measured). Confirm realized `sparse_tokens_after` / `memory_tokens` from `train_timing.jsonl` after the real run before quoting in the paper.
- **GPU contention**: serial queue assumes one free GPU (default GPU_ID=0). Set GPU_ID if shared.

## 11. Next step

Phase 5-3 = execute the two real runs with the command in §8 (reusing canonical
PL), then collect realized token counts + mIoU + train timing into the
speed-accuracy table from `step_13_token_budget_plan.md`.
