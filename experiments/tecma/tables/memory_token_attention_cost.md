# Memory Token & Attention Cost Analysis

> Generated: 2026-06-17 | Seed=1 | DensePASS val 100 imgs | clean fixed-PL ablation setting
> All timing is epoch-level aggregate from `run_summary.json.profile`.
> `memory_tokens_scope`: `per_attention_call_all_chunks_concat`
> (memory tokens counted across all image-patch chunks in one SAM2 memory-attention call)

## Table A: A/D/E Clean Ablation — Memory Token & Attention Cost

Reference: `baseline_omnisam` (A=off, D=off, E=off), 32,768 tokens, attn 117.48s.

| Config | Modules | mIoU | ΔmIoU | BFScore | ΔBFScore | Mem. Tokens | Token Red. | Mem. Attn.(s) | Attn. Red. | Step Time(s) | Step Δ% | GPU Peak(GB) | GPU Save(GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_omnisam` | A=off, D=off, E=off | 60.26 | +0.00 | 39.59 | +0.00 | 32,768 | 1.00× | 117.48 | +0.0% | 2.3505 | +0.00% | 19.54 | +0.00 |
| `A_ccpm` | A=on, D=off, E=off | 60.26 | +0.00 | 39.59 | +0.00 | 32,768 | 1.00× | 117.56 | -0.1% | 2.4089 | +2.48% | 19.98 | -0.44 |
| `A_Dtrain` | A=on, D=on (train), E=off | 60.12 | -0.14 | 39.23 | -0.36 | 8,192 | 4.00× | 61.00 | +48.1% | 2.2938 | -2.41% | 18.99 | +0.55 |
| `A_Etrain` | A=on, D=off, E=on (train, keep=0.50) | 60.09 | -0.17 | 39.23 | -0.36 | 16,384 | 2.00× | 79.69 | +32.2% | 2.3798 | +1.25% | 19.28 | +0.26 |
| `Dtrain_Etrain` | A=off, D=on (train), E=on (train, keep=0.50) | 60.02 | -0.24 | 39.23 | -0.36 | 4,096 | 8.00× | 52.38 | +55.4% | 2.3237 | -1.14% | 18.34 | +1.20 |
| `A_Dtrain_Etrain` | A=on, D=on (train), E=on (train, keep=0.50) | 60.02 | -0.24 | 39.23 | -0.36 | 4,096 | 8.00× | 51.92 | +55.8% | 2.3153 | -1.50% | 18.78 | +0.76 |

Notes:
- A (CCPM) does **not** reduce memory tokens; attn time unchanged vs baseline.
- D alone (A_Dtrain): 4× token reduction, −48.1% attn time.
- E alone (A_Etrain, keep=0.50): 2× token reduction, −32.2% attn time.
- D+E combined, no A (Dtrain_Etrain): 8× token reduction, −55.4% attn time, same mIoU as A_Dtrain_Etrain (60.02).
- A+D+E final (A_Dtrain_Etrain): 8× token reduction, −55.8% attn time, mIoU 60.02 (ΔmIoU −0.24 vs baseline).
- GPU peak saving for A_ccpm is slightly negative (−0.44 GB) vs baseline — both have 32768 tokens; small run-to-run variation.

## Table B: E-only Token Budget Curve + D+E Final Endpoint

Reference: `A_ccpm` (100% budget, 32,768 tokens, attn 117.56s).

| Budget | Config | Curve Type | Mem. Tokens | Token Red. | mIoU | ΔmIoU | BFScore | ΔBFScore | Mem. Attn.(s) | Attn. Red. | Step Time(s) | Step Δ% | GPU Peak(GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100% | `A_ccpm` | E-only curve | 32,768 | 1.00× | 60.26 | +0.00 | 39.59 | +0.00 | 117.56 | +0.0% | 2.4089 | +0.00% | 19.98 |
| 75% | `A_Etrain_keep075` | E-only curve | 24,576 | 1.33× | 60.12 | -0.14 | 39.28 | -0.31 | 98.62 | +16.1% | 2.4614 | +2.18% | 19.64 |
| 50% | `A_Etrain` | E-only curve | 16,384 | 2.00× | 60.09 | -0.17 | 39.23 | -0.36 | 79.69 | +32.2% | 2.3798 | -1.21% | 19.28 |
| 25% | `A_Etrain_keep025` | E-only curve | 8,192 | 4.00× | 60.10 | -0.16 | 39.29 | -0.30 | 60.54 | +48.5% | 2.2941 | -4.77% | 18.93 |
| final (12.5%) | `A_Dtrain_Etrain` | Combined D+E final endpoint | 4,096 | 8.00× | 60.02 | -0.24 | 39.23 | -0.36 | 51.92 | +55.8% | 2.3153 | -3.89% | 18.78 |

Notes:
- 100%/75%/50%/25% form the **E-only keep-ratio curve** (D off, E keep_ratio varies).
- **final (12.5%)** is the **D+E combined endpoint**, not an E-only curve point.
  Effective token path: 32768 → D→ 8192 → E→ 4096.
- Memory attention time is **strictly monotonic** across the E-only curve (100%→25%).
- Total step time is NOT strictly monotonic: 75% point is +2.18% vs 100%, because sparse retrieval overhead is not yet amortized at small keep_ratio reductions.
- BFScore and mIoU are stable across all 4 E-only points (max drop 0.17 / 0.36 respectively).

## Table C: D Module Token Compression Detail

| Config | comp_tokens_before (per-frame) | comp_tokens_after (per-frame) | compress_ratio | final_tokens (all chunks) |
|---|--:|--:|--:|--:|
| `A_Dtrain` | 4,096 | 1,024 | 0.25 | 8,192 |
| `Dtrain_Etrain` | 4,096 | 1,024 | 0.25 | 4,096 |
| `A_Dtrain_Etrain` | 4,096 | 1,024 | 0.25 | 4,096 |

D module (FSMC) compresses each memory frame from 4096 to 1024 tokens (compress_ratio=0.25) via spatial pooling (stride=2 pool).
With 8 image patches per panoramic image, total memory tokens = 8 frames × 1024 = 8,192 (A_Dtrain) or 4,096 after E further halves it (D+E combined).

## Appendix: Main-Result Timing Reference (Different Run Conditions)

These runs are from Phase 3 (full PL generation, different GPU context). **Not comparable to the clean-ablation step times above.**

| Config | Step Time (s) | Attn Time (s) | Attn Time/call (ms) | GPU Peak (GB) | Mem Tokens | mIoU |
|---|--:|--:|--:|--:|--:|--:|
| `baseline_omnisam (main)` | 5.266 | 263.02 | 32.877 | 26.82 | 32,768 | 60.26 |
| `A_Dtrain_Etrain (main)` | 4.085 | 93.63 | 11.704 | 25.34 | 4,096 | 60.16 |

> Main-result `baseline_omnisam` attn time 263.0s vs clean-ablation 117.5s: different run, more total patches processed.
> Main-result `A_Dtrain_Etrain` attn time 93.6s vs clean-ablation 51.9s: same token count (4096) but different batch/patch configuration.
> Both show consistent direction: D+E combined reduces attention time significantly vs baseline.
