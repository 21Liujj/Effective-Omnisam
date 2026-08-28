# Phase 4 Status

Date: 2026-06-16  
Project: OmniSAM / TeCMA

## Status summary

```text
Phase 4-v0: completed, preliminary only, not final table
Phase 4-clean: completed, official ablation candidate
Blocking issue: none
Remaining issue: command printing has occasional formatting artifacts, but actual execution succeeded
Next phase: Phase 5 complexity analysis and paper experiment writing
```

## Details

### Phase 4-v0

- Status: completed.
- Positioning: preliminary sanity / trend analysis only.
- Reason: `REUSE_PL_DIR` was not actually propagated, so each configuration regenerated pseudo labels.
- Usage: may be referenced internally for trend checking, but should not be used as the formal paper ablation table.

### Phase 4-clean

- Status: completed.
- Positioning: official ablation candidate.
- Runs: six configurations completed successfully.
- Fixed PL directory:

```text
data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
```

- Evidence: every run log contains:

```text
Reuse fixed pseudo-label dir for epoch 1: data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
```

### Official files

- `experiments/tecma/tables/clean_ablation_results.md`
- `experiments/tecma/tables/clean_ablation_results.csv`
- `experiments/tecma/notes/phase4_clean_ablation_conclusion.md`
- `experiments/tecma/tables/paper_ablation_table_draft.md`

## Key outcome

The full `A_Dtrain_Etrain` configuration achieves 8× memory-token reduction and 55.8% memory-attention time reduction under fixed pseudo labels, with only 0.24 mIoU and 0.36 BFScore degradation.

## PLon handling

`A_Dtrain_Etrain_PLon` belongs to v0 / negative-ablation analysis. It tests enabling D/E during pseudo-label generation and therefore cannot reuse canonical fixed PL by design. It should not be mixed into the Phase 4-clean main table. If needed, report it separately in supplementary material.
