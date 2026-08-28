# Step 15 — Token Budget Preflight Report (Phase 5-3, Step A)

> Date: 2026-06-17 (server time ~11:11)
> Scope: read-only preflight + dry-run for the two new token-budget points.
> NO real training started.

## 0. Summary verdict

- Preflight: **PASS** (通过)
- Real training started this step: **NO**
- Recommendation: **safe to start the two real runs manually.**

## 1. Related training processes

`ps aux | grep run_adaptation|run_token_budget|run_queue|run_main_experiments`
(excluding grep / ssh helper):

**NO related process running.** GPU is free for the serial queue.

## 2. Disk space

```
Filesystem      Size  Used Avail Use% Mounted on
/dev/md0        100G   72G   29G  72% /root/autodl-tmp
```

- Free: **29 GB**. `experiments/tecma/runs` currently 23 GB.
- Two new runs reuse canonical PL (no PL regen, no new 4000-png set), each ≈
  one checkpoint set + logs (order of hundreds of MB). 29 GB is sufficient.

## 3. Config files

| Config | Status | Key settings |
|---|---|---|
| `experiments/tecma/configs/A_Etrain_keep075.yaml` | exists (985 B) | A on (thr 0.85), D off, E on keep_ratio=**0.75**, E apply_in_train=true, apply_in_pl=false |
| `experiments/tecma/configs/A_Etrain_keep025.yaml` | exists (984 B) | A on (thr 0.85), D off, E on keep_ratio=**0.25**, E apply_in_train=true, apply_in_pl=false |

Both match Plan B: A on / D off / E train-only / E off in PL.

## 4. Canonical PL_DIR

```
data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
```

- **Exists.** Structure: `gtFine/` + `gtFine_overlap/`, 40 city subdirs each,
  **4000 PL .png total** (2000 + 2000 overlap). Valid canonical PL set.

## 5. Script

- `experiments/tecma/scripts/run_token_budget_experiments.sh` — exists, **executable** (`-rwxr-xr-x`, 5.7 KB).

## 6. Dry-run result (re-run this step)

`DRY_RUN=1 REUSE_PL_DIR=$PL_DIR SEED=1 bash run_token_budget_experiments.sh A_Etrain_keep075 A_Etrain_keep025`

- queue summary: `A_Etrain_keep075=ok`, `A_Etrain_keep025=ok`, **fail=0**
- no real training; **no `Pseudolabel Generation`**; **no `Epoch 1 [Train]`**
- each config → its own output dir under
  `experiments/tecma/runs/token_budget/<CONFIG>_seed1_<STAMP>/`
- subset file: `_subsets/Cityscapes_13cls_seed1_n2000.txt` (n=2000, seed=1)

## 7. `--reuse_pl_dir` present?

**YES.** Both delegated train commands end with:
`--reuse_pl_dir data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1`
Wrapper also prints `[reuse-pl] using canonical pseudo labels: ...` per run.

## 8. Argument concatenation?

**NONE.** Every flag is space-separated in the printed command; no glued args
(`--config ... --seed 1 ... --reuse_pl_dir ...` all distinct).

## 9. Start real run? 

**Yes — recommended.** All six preflight checks pass and the dry-run is clean.

## 10. Recommended foreground (serial) command

```bash
cd /root/autodl-tmp/OmniSAM-main
PL_DIR=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
REUSE_PL_DIR=$PL_DIR SEED=1 bash experiments/tecma/scripts/run_token_budget_experiments.sh \
  A_Etrain_keep075 A_Etrain_keep025
```

## 11. Recommended background command

```bash
cd /root/autodl-tmp/OmniSAM-main
PL_DIR=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
LOG=experiments/tecma/runs/token_budget/_bg_logs/token_budget_seed1_$(date +%Y%m%d_%H%M%S).log
mkdir -p experiments/tecma/runs/token_budget/_bg_logs
nohup env REUSE_PL_DIR=$PL_DIR SEED=1 FG=1 \
  bash experiments/tecma/scripts/run_token_budget_experiments.sh \
  A_Etrain_keep075 A_Etrain_keep025 \
  > "$LOG" 2>&1 &
echo "Token budget experiments started."
echo "Log: $LOG"
```

Watch: `tail -f "$LOG"`

## 12. Expected time & risks

- Each run reuses canonical PL → PL stage skipped (~2h cost avoided). Train only:
  ≈25–35 min/epoch at 2000-PL / 1-epoch E-only scale. **Two runs ≈ 50–70 min serial** (estimate; confirm from logs).
- Disk OK (29 GB free; runs reuse PL, hundreds of MB each).

Risks:
- Empty `REUSE_PL_DIR` would silently regenerate PL and break curve comparability — always export `PL_DIR`. (Wrapper warns if empty.)
- keep_ratio → realized tokens (24576 / 8192) is **expected**, not measured. Confirm realized `memory_tokens` / `sparse_tokens_after` from `train_timing.jsonl` after the run before quoting in the paper.
- GPU contention: serial queue assumes GPU 0 free (confirmed free now).

## 13. Note on background command variant

Step 14 report used `BACKGROUND=1` for detached mode; the task's recommended
command uses `nohup ... FG=1`. Both are valid: `BACKGROUND=1` is the wrapper's
own detach path, `nohup ... FG=1` detaches at shell level and forces foreground
inside. Use whichever; the `nohup` variant gives an explicit captured log path.
