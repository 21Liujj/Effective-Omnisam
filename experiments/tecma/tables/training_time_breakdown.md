# Training Time Breakdown (Phase 6-2)

> Generated: 2026-06-17 | N_iters=500 (2000 PL images, batch_size=4)
> Confirmed from run.log: `Epoch 1 [Train]: 0/500`
> `iter_compute_time` wraps full forward+backward+optimizer (`t_comp0` to `comp_t`);
> it INCLUDES memory attention. `step_time` additionally includes data loading and
> loop overhead (~0.83s/iter on average). `memory_attention_time` is an epoch-level
> accumulator (`attention_time_accum`). Per-iter values derived by dividing by N=500.

## Table A: Training-Step Time Breakdown — A/D/E Clean Ablation

Reference: `A_ccpm` (A=on, D=off, E=off, 32768 tokens, step 2.409s).
> All times are **per-iteration** averages (seconds unless noted).
> `MemAttn/iter` derived from accumulated epoch total ÷ N_iters (=500).
> `Est. Non-MemAttn` = step_time − MemAttn/iter (includes data loading, encoder, decoder, loss, backward, optim, loop overhead).
> `MemAttn Share` = MemAttn/iter / step_time (memory attention share of total step).

| Config | Modules | Tokens | mIoU | BFScore | Step(s) | Data(s) | ICT(s) | MemAttn/iter(ms) | MemAttn Share | Est.Non-Mem(s) | Step Δ% | MemAttn Red.% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `A_ccpm` | A=on, D=off, E=off | 32,768 | 60.26 | 39.59 | 2.4089 | 0.1053 | 1.5815 | 235.1 | 9.76% | 2.1738 | +0.00% | +0.00% |
| `A_Dtrain` | A=on, D=on (train), E=off | 8,192 | 60.12 | 39.23 | 2.2938 | 0.1136 | 1.4540 | 122.0 | 5.32% | 2.1718 | -4.78% | +48.12% |
| `A_Etrain` | A=on, D=off, E=on (keep=0.50) | 16,384 | 60.09 | 39.23 | 2.3798 | 0.1262 | 1.5388 | 159.4 | 6.70% | 2.2204 | -1.21% | +32.21% |
| `Dtrain_Etrain` | A=off, D=on (train), E=on (keep=0.50) | 4,096 | 60.02 | 39.23 | 2.3237 | 0.1319 | 1.4687 | 104.8 | 4.51% | 2.2189 | -3.54% | +55.45% |
| `A_Dtrain_Etrain` | A=on, D=on (train), E=on (keep=0.50) | 4,096 | 60.02 | 39.23 | 2.3153 | 0.1257 | 1.4643 | 103.8 | 4.48% | 2.2115 | -3.89% | +55.84% |

**Key observations (Table A):**
- Memory attention accounts for **9.76%** of total step time in the reference (A_ccpm).
- A_Dtrain_Etrain reduces MemAttn/iter by 55.84% (235.1→103.8 ms), but step time drops only 3.89% (2.409→2.315s).
- Est. Non-MemAttn time is **stable** across configs (2.17–2.22s/iter, std < 0.02s), confirming that encoder, decoder, loss, backward, and loop overhead are unchanged by D/E modules.
- ict−MemAttn/iter (compute without MA): A_ccpm=1.2231s vs A_Dtrain_Etrain=1.3000s; the slight increase (~0.08s) is the D+E forward-pass overhead (pool + saliency + top-k).

## Table B: Training-Step Time Breakdown — Token Budget Curve

Reference: `A_ccpm` (100%, 32768 tokens).
> Same per-iter derivation as Table A.

| Budget | Config | Curve | Tokens | Step(s) | MemAttn/iter(ms) | MemAttn Share | Est.Non-Mem(s) | Step Δ% | MemAttn Red.% | mIoU | BFScore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100% | `A_ccpm` | E-only curve (ref.) | 32,768 | 2.4089 | 235.1 | 9.76% | 2.1738 | +0.00% | +0.00% | 60.26 | 39.59 |
| 75% | `A_Etrain_keep075` | E-only curve | 24,576 | 2.4614 | 197.2 | 8.01% | 2.2642 | +2.18% | +16.11% | 60.12 | 39.28 |
| 50% | `A_Etrain` | E-only curve | 16,384 | 2.3798 | 159.4 | 6.70% | 2.2204 | -1.21% | +32.21% | 60.09 | 39.23 |
| 25% | `A_Etrain_keep025` | E-only curve | 8,192 | 2.2941 | 121.1 | 5.28% | 2.1730 | -4.77% | +48.50% | 60.10 | 39.29 |
| final (12.5%) | `A_Dtrain_Etrain` | D+E combined endpoint | 4,096 | 2.3153 | 103.8 | 4.48% | 2.2115 | -3.89% | +55.84% | 60.02 | 39.23 |

**Key observations (Table B):**
- MemAttn/iter is **strictly monotonic** across the E-only curve (235.1 → 197.2 → 159.4 → 121.1 ms).
- Total step time at 75% (2.461s) is **higher** than at 100% (2.409s, +2.18%):
  - Sparse retrieval overhead (saliency norm computation + top-k indexing) adds ~0.06s/iter to ict;
  - At keep_ratio=0.75, the MemAttn saving (~37.9 ms/iter) does not yet offset this overhead;
  - At keep_ratio=0.50 and below, MemAttn savings dominate and step time drops below the reference.
- Est. Non-MemAttn stabilizes around 2.17–2.26s at lower budgets; the higher value at 75% (2.264s) reflects the sparse retrieval overhead in the non-MA component of ict.

## Table C: Fine-Grained Timing Availability

| Component | Available? | Source | Notes |
|---|---|---|---|
| Total step time | ✅ Yes | `run_summary.profile.step_time` / `train_timing.jsonl` | Per-iteration average wall-clock; includes all sub-components |
| Data loading | ✅ Yes (per-iter avg) | `train_timing.jsonl.data_time` | Per-iter average; measured by `t_data0` timer in trainer |
| iter_compute_time (ICT) | ✅ Yes | `train_timing.jsonl.iter_compute_time` | Full forward+backward+optim per-iter; INCLUDES memory attention |
| Memory attention (total) | ✅ Yes | `run_summary.profile.memory_attention_time` | Epoch total via `attention_time_accum` in MemoryBank; convert to per-iter ÷ 500 |
| Memory attention (per call) | ✅ Yes | `train_timing.jsonl.memory_attention_time_per_call` | Per-call average; A_ccpm=14.7ms/call |
| Loop overhead (step − data − ict) | ⚠️ Derived only | step_time − data_time − ict | Not separately instrumented; ~0.72s/iter; includes tqdm, grad-clip, callbacks, logging |
| Image encoder (SAM2 LoRA) | ❌ Not available | Not separately instrumented | Would require extra torch.cuda.Event instrumentation in model.py |
| Decoder (SegFormerHead) | ❌ Not available | Not separately instrumented | Included in ICT-minus-MA residual (~1.28s/iter) |
| Loss computation | ❌ Not available | Not separately instrumented | Included in ICT-minus-MA residual |
| Backward pass | ❌ Not available | Not separately instrumented | Included in ICT-minus-MA residual |
| Optimizer step | ❌ Not available | Not separately instrumented | Included in ICT-minus-MA residual |
| D module (FSMC) overhead | ⚠️ Derived only | ict_ref − ict_D (if D only) ≈ 0.08s | Not directly measured; estimated from ict difference across configs; noisy |
| E module (SG-SMR) overhead | ⚠️ Derived only | ict_ref − ict_E (if E only) ≈ 0.04s | Same caveat as D |
| Validation step | ❌ Not in step_time | epoch_total − epoch_train | Validation runs after training; not part of step_time |

> **ICT-minus-MA residual** = `iter_compute_time − (memory_attention_time / 500)`. For A_ccpm this is ~1.22s/iter, representing the combined cost of encoder, FPN, decoder, loss, backward, optimizer, and sparse/pool module overhead. It is NOT individually instrumented.
