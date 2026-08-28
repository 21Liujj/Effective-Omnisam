# Step 13: Token Budget / Speed-Accuracy Trade-off Plan (Reworked)

## Goal

Design a minimal Phase 5-1 experiment to show that FSMC (Module D) and SG-SMR (Module E) provide a *stable* memory-token-efficient speed/accuracy trade-off, not a single lucky hyperparameter point.

Strict scope: this file is an experiment design only. Do NOT run training, pseudo-label generation, validation, or timing scripts from this step. No new YAML, no script edits, no model/engine/trainer/plugin changes.

## 0. Why the previous design was reworked (core problem)

The earlier draft had three defects:

1. **Duplicate points.** `tecma_token_budget_050.yaml` (D=0.50, E=0.50) and `tecma_token_budget_final_adaptive.yaml` (D=0.50, E=0.50) were the *same* setting. The confirmed final config `A_Dtrain_Etrain` is already D=0.5 + E=0.5, so both rows collapse onto one operating point — the "final" row added nothing.
2. **Configured ratio conflated with token budget.** The draft treated the D/E ratio knobs (0.75 / 0.50 / 0.25) as if they were the realized token budget. They are not. D and E compose multiplicatively, and a single knob value (e.g. 0.50) maps to very different realized budgets depending on whether it is D-only, E-only, or D+E.
3. **A new-curve framing that ignored existing clean-ablation results.** The draft proposed 5 brand-new runs while a clean ablation already pins down four budget points for free.

### Configured ratio vs realized token budget

Three distinct quantities must never be merged:

- **Configured ratio** — the YAML knob (`compress_ratio` for D, `keep_ratio` for E).
- **Realized memory tokens** — actual per-frame token count measured from logs.
- **Realized token budget** — realized tokens as a fraction of the full-memory baseline (32768 tokens = 100%).

The existing clean ablation makes the distinction concrete:

| Config          | D ratio | E ratio | Realized memory tokens | Realized token budget |
| --------------- | ------: | ------: | ---------------------: | --------------------: |
| A_ccpm          |     off |     off |                  32768 |                  100% |
| A_Etrain        |     off |     0.5 |                  16384 |                   50% |
| A_Dtrain        |     0.5 |     off |                   8192 |                   25% |
| A_Dtrain_Etrain |     0.5 |     0.5 |                   4096 |                 12.5% |

Note D=0.5 alone yields 25% budget (8192), and D=0.5 + E=0.5 yields 12.5% (4096), NOT 50%. The "final" config is a **12.5% effective token budget point**, not a 50% point.

## Checked Parameters

- Module D / FSMC config block: `plugins.efficient_tam_memory`
  - Main YAML parameter: **`plugins.efficient_tam_memory.compress_ratio`** (final value `0.5`)
  - Execution controls: `apply_in_pl`, `apply_in_train`
- Module E / SG-SMR config block: `plugins.efficient_sam2_sparse`
  - Main YAML parameter: **`plugins.efficient_sam2_sparse.keep_ratio`** (final value `0.5`, mode `attn_topk`)
  - Execution controls: `apply_in_pl`, `apply_in_train`
- Module A / pseudo confidence mask: `plugins.pseudo_conf_mask` (`threshold: 0.85`)
  - Keep A enabled and identical for every point so token budget is never confounded with pseudo-label filtering.

## Hard constraints shared by both plans

1. All D/E settings are **train-only**: `apply_in_pl: false`, `apply_in_train: true`.
2. D/E **disabled in PL** for every point — PL must stay on the canonical full-memory pseudo-labels.
3. All new runs **reuse the canonical PL set**; no PL regeneration.
4. Identical data split, seed, source checkpoint, epochs, validation protocol, logging format across all points.
5. A enabled with `threshold: 0.85` for every point.
6. Report speed separately for Train / PL / Val / E2E. The main curve plots Train step time (or memory-attention time) against DensePASS val mIoU.

## Plan A — Zero-new-run minimum plan

Reuse the existing clean ablation as-is. Four budget points, **no new training required**.

| Budget Point  | Config          | Modules                  | Realized Tokens | Existing Result |
| ------------- | --------------- | ------------------------ | --------------: | --------------- |
| 100%          | A_ccpm          | A only, D/E off          |           32768 | yes             |
| 50%           | A_Etrain        | A + E keep_ratio=0.5     |           16384 | yes             |
| 25%           | A_Dtrain        | A + D compress_ratio=0.5 |            8192 | yes             |
| final / 12.5% | A_Dtrain_Etrain | A + D=0.5 + E=0.5        |            4096 | yes             |

**What Plan A is and is not.** This plan needs **0 new runs** — every point already has a confirmed result. But it is NOT a single-hyperparameter curve: the four points differ in *which modules* are active (E-only, D-only, D+E), so it is a **"token-budget trade-off across TeCMA memory-reduction variants"**, not a clean sweep of one knob. It demonstrates the trade-off exists across the variant family; it does not isolate one mechanism.

## Plan B — Recommended clean curve plan

Add two E-only points so 100/75/50/25 form a single clean sparse-keep-ratio curve, then attach the D+E final as the combined endpoint.

| Budget Point  | Config           | A  | D    | E keep_ratio | Expected Tokens | Existing? | Need Run? |
| ------------- | ---------------- | -- | ---- | -----------: | --------------: | --------- | --------- |
| 100%          | A_ccpm           | on | off  |          off |           32768 | yes       | no        |
| 75%           | A_Etrain_keep075 | on | off  |         0.75 |           24576 | no        | yes       |
| 50%           | A_Etrain         | on | off  |         0.50 |           16384 | yes       | no        |
| 25%           | A_Etrain_keep025 | on | off  |         0.25 |            8192 | no        | yes       |
| final / 12.5% | A_Dtrain_Etrain  | on | 0.50 |         0.50 |            4096 | yes       | no        |

**Clarifications (mandatory):**

1. The 100 / 75 / 50 / 25 rows are an **E-only `keep_ratio` curve** — D stays off, only `efficient_sam2_sparse.keep_ratio` varies. This isolates one mechanism along one axis.
2. The **final row is the D+E combined endpoint** (D=0.5 + E=0.5), at **12.5% effective token budget** — it is deliberately off the E-only curve and must be labeled as a separate combined point.
3. This design needs **only two new runs**: `A_Etrain_keep075` and `A_Etrain_keep025`.
4. Both new runs **must reuse the canonical PL set** (no PL regeneration).
5. D/E must be enabled **train-only** (`apply_in_train: true`).
6. PL stage must keep D/E **off** (`apply_in_pl: false`).
7. **Do not plot `final` as a 50% point.** It is a 12.5% effective token budget; mislabeling it collapses the curve.

## Recommendation

- **Primary: adopt Plan B.** Two extra runs buy a clean, single-axis E-only keep-ratio curve (100/75/50/25) plus a clearly-marked D+E endpoint — far more defensible in the paper than a mixed-variant scatter.
- **Fallback: adopt Plan A** only if time is extremely tight. It costs 0 runs but must be reported honestly as a trade-off across memory-reduction variants, not a single-knob sweep.

## Expected outputs for the paper

- Table columns: config name, configured ratio, realized memory tokens, realized token budget, train step time, memory-attention time, mIoU, mAcc, PixAcc.
- Figure: x-axis = realized token budget (or speed metric `s/it`), y-axis = mIoU; annotate 100 / 75 / 50 / 25 on the E-only curve and mark the 12.5% D+E final as a distinct point.
- Interpretation: if mIoU is near-flat from 100% down to 25% while train time drops, the trade-off is stable; the point where mIoU starts to fall defines the practical lower budget bound.

## Next step (Phase 5-2)

Phase 5-2 should **only generate the run scripts/configs** for the two Plan B points (`A_Etrain_keep075`, `A_Etrain_keep025`) — reusing canonical PL, D/E train-only, PL D/E off. **Do not run any experiment in Phase 5-2.**
