# Time Ablation Infrastructure

## Naming Convention

Run name pattern:

`<dataset>_seed<seed>_bank<bank>_comp-<mode>_loc-<location>_sched-<setting>`

Example:

`cs13dp13_seed0_bank9_comp-hybrid_loc-before_attention_sched-default`

## Output Layout

Each run writes:

- `logs/` (stdout, tensorboard)
- `ckpts/` (checkpoint path under `pd_root/checkpoints/...`)
- `metrics/` (`pl_timing.jsonl`, `train_timing.jsonl`)
- `summaries/` (reserved for post-processing)

## Launchers

- Main 4 methods:
  - `scripts/time_ablation/run_time_ablation_main4.sh`
- Memory internal ablation:
  - `scripts/time_ablation/run_time_ablation_memory_internal.sh`
- Patch internal ablation:
  - `scripts/time_ablation/run_time_ablation_patch_internal.sh`
- Smoke:
  - `scripts/time_ablation/run_time_ablation_smoke.sh`

## Summary Tool

```bash
python tools/summarize_time_ablation.py \
  --root logs/time_ablation/main4 \
  --out_dir logs/time_ablation/main4/summaries
```
