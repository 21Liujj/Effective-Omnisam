# Legacy Geo-TeCMA / Pano-SMB Overlap Audit

Date: 2026-07-15  
Run Level: L0 scan-only  
Code modified: No  
Training run: No

## 1. Executive Summary

Final classification: **OLD_HAS_DIAGNOSTICS_ONLY**.

Legacy assets separate into: (1) MTC 2D pooling; (2) SG-SMR content-only Top-K, effectively feature norm in all verified runs; (3) Geo-Memory geometry metadata/masks/region diagnostics with `selection_effect=diagnosis_only`.

The retriever has dormant attention/confidence/hybrid/overlap/boundary interfaces, but the real caller passes attention and confidence as None and never supplies overlap/boundary masks. Verified configs use `attn_topk` plus `saliency_source=feature_norm`, therefore fallback to feature norm. No hybrid or geometry-aware selection run was found.

No per-view hard quota, protected overlap/seam reservation, cap, residual fill, view coverage metric, or panorama-global exact-budget correction exists. Old work does not duplicate Pano-SMB hard budgeting.

## 2. Search Scope

Scanned `model/`, `adaptation/`, `experiments/geo_memory/`, `experiments/tecma/`, configs, scripts, docs, paper inventories, logs, metrics, JSON/JSONL, and checkpoints. No Geo selector config or hard-budget implementation was found.

## 3. Memory Selection Call Chain

```text
run_adaptation.py -> Engine.setup -> build_model -> WithMemStrategy
-> OmniSAM_adapt.forward -> MemoryBank.forward
-> previous frame chunks -> optional MTC -> optional retriever per chunk
-> concatenate memory/memory_pos -> SAM2 memory_attention
-> _encode_new_memory
```

Memory is `(L,B,C)` with `mem_hw`. Retrieval chooses ratio-based `k` independently per previous frame chunk. Memory and memory_pos use the same index. Both model paths pass `num_obj_ptr_tokens=0`; no-memory embeddings bypass selection.

## 4. Token Importance Implementations

| Mode | Logic | Status |
|---|---|---|
| feature norm | `norm(tokens,dim=-1).mean(dim=1)` | IMPLEMENTED_AND_TRAINED |
| attention | detached score reduced to L | INTERFACE_ONLY; caller passes None |
| confidence | map flattened to L | INTERFACE_ONLY; caller passes None |
| random | `rand(L)` | IMPLEMENTED_NOT_VERIFIED |
| hybrid | norm + optional attention/confidence/overlap/boundary | INTERFACE_ONLY |
| prototype similarity | PL diagnostics/gates | not memory selection |
| query-memory similarity / overlap agreement | not found | NOT_IMPLEMENTED |
| latitude geo score | `abs(sin(latitude))` | DIAGNOSTICS_ONLY |

## 5. Panorama Geometry Signals

| Signal | Constructed? | Passed to Selector? | Reliability |
|---|---:|---:|---|
| frame/view index | yes | no | physical view conditional |
| local token xy / `mem_hw` | yes | no | reliable for local grid |
| ERP x/y range | inferred | no | training x often unknown |
| overlap/boundary mask | interface only | no | unestablished |
| seam | diagnostic | no | only known order/range |
| polar/equator/latitude | yes | no | reliable by token row |
| actual reverse flag/crop box | no | no | missing |

## 6. Reconstructed Legacy Algorithm

```python
for each previous memory chunk:
    if MTC: feats, pos = pool_2d(feats, pos)
    score = feature_norm(feats)  # attention input is None
    k = safe_round(L * keep_ratio)
    idx = sort(topk(score, k))
    feats, pos = feats[idx], pos[idx]
memory_attention(concat(feats_chunks))
```

Active category: **A. Content-only Top-K per chunk**. Geometry does not affect score. There is no hard quota, reservation, cap, seam reliability gate in selection, or global exact budget. A processed chunk keeps minimum tokens incidentally, but that is not an explicit panorama view-coverage constraint.

## 7. Runtime Plumbing Verification

Engine configures the retriever and its output becomes attention input. However, attention/confidence are None, overlap/boundary are omitted, and returned selected indices are not persisted. Geo writer runs after forward and is diagnosis-only. `fallback_full_when_uncertain` is stored but audited retrieval falls back to feature norm, not full memory.

## 8. Existing Configs and Runs

| Run | Classification | Geometry selection? | mIoU | BFScore |
|---|---|---:|---:|---:|
| TeCMA baseline clean | COMPLETE_FULL_RUN | no | 60.26 | 39.59 |
| A_Etrain clean | COMPLETE_FULL_RUN | norm Top-K; no geo | 60.09 | 39.23 |
| A_Dtrain_Etrain clean | COMPLETE_FULL_RUN | MTC+norm Top-K; no geo | 60.02 | 39.23 |
| E 75% / 25% | COMPLETE_FULL_RUN | no geo | 60.12 / 60.10 | 39.28 / 39.29 |
| Geo Task 0-4 | FORWARD_DIAGNOSIS_ONLY | no | N/A | N/A |
| Geo Task 02-3 | COMPLETE_SHORT_RUN | diagnosis only; one train iter | 59.14 | 37.76 |
| GM1 region metrics | FORWARD_DIAGNOSIS_ONLY | no | 12 matched masks only | N/A |
| hybrid/overlap selector | INTERFACE_ONLY | no | N/A | N/A |

Complete training results exist only for content-only legacy compression/retrieval, not Geo-aware selection.

## 9. Legacy vs Pano-SMB Capability Matrix

| Capability | Legacy Norm Top-K | Old Geo-TeCMA | Proposed Pano-SMB |
|---|---|---|---|
| feature norm | IMPLEMENTED_AND_TRAINED | diagnostic read only | reuse |
| attention/confidence | INTERFACE_ONLY | NOT_IMPLEMENTED | optional |
| panorama/overlap soft bonus | INTERFACE_ONLY | DESIGN_ONLY | not core |
| seam/latitude diagnostics | NOT_IMPLEMENTED | IMPLEMENTED_AND_FORWARD_VERIFIED | reusable ideas |
| per-view hard quota | NOT_IMPLEMENTED | NOT_IMPLEMENTED | core |
| zero-view prevention | incidental chunk minimum | NOT_IMPLEMENTED | explicit |
| overlap reservation/cap | NOT_IMPLEMENTED | NOT_IMPLEMENTED | core |
| seam reliability gate | NOT_IMPLEMENTED | diagnostics only | core |
| seam hard reservation | NOT_IMPLEMENTED | NOT_IMPLEMENTED | core |
| exact global spatial budget | per-chunk count only | NOT_IMPLEMENTED | core |
| spatial/pointer separation | pointer always zero; nonzero unverified | same | explicit/tested |
| memory/position sync | IMPLEMENTED_AND_TRAINED | unchanged | preserve |
| view coverage/CV/retention | NOT_IMPLEMENTED | NOT_IMPLEMENTED or N/A | core diagnostics |
| complete Geo-aware run | NOT_IMPLEMENTED | NOT_IMPLEMENTED | required later |

## 10. Duplication Risk

Reuse MTC, feature-norm score, legacy Top-K, `mem_hw`, row-major mapping, synchronized memory/position indexing, geometry utilities, and unknown/N/A policy.

Still-new Pano-SMB content is hard per-view coverage, bounded topology reservation, residual global saliency fill, exact global budget, and reliability-gated fallback. Do not claim novelty for pooling, norm Top-K, token-row geometry, or seam/polar diagnostics.

## 11. Final Classification

**OLD_HAS_DIAGNOSTICS_ONLY**

## 12. Recommended Next Step

**STOP_AND_FIX_METADATA_MAPPING** - provide per-sample physical crop/order metadata, actual reverse flag, and reliable ERP x range. Do not infer seam from loop order under random reversal. No next task was started.

## 13. Evidence Index

| ID | Path | Supports |
|---|---|---|
| E1 | `model/modules/sparse_memory_retriever.py` | scores, fallback, Top-K, same index |
| E2 | `model/memory_bank.py` | token shapes, per-chunk caller, attention path |
| E3 | `adaptation/engine.py` | runtime configuration |
| E4 | `adaptation/strategies.py` | Geo context and diagnosis-only warning |
| E5 | `experiments/geo_memory/geometry_ops.py` | standalone geometry diagnostics |
| E6 | Geo step 00-1/00-2 notes | missing crop/reverse metadata |
| E7 | Geo JSONL/writer notes | diagnosis-only, null keep/score |
| E8 | TeCMA YAML/run logs | attn_topk, feature-norm fallback |
| E9 | clean ablation/token-budget reports | content-only full runs |
| E10 | Geo step 02-4 | kept/dropped/base-score N/A |
| E11 | repository-wide searches | no hard budgeting/callers |

## 14. Unknowns and Risks

1. Nonzero object-pointer safety is untested.
2. Sparse selected indices are not persisted.
3. ERP x/order metadata is inferred, not first-class.
4. Local `mem_hw` is reliable, but global/sparse original coordinates need type/index metadata.
5. Region metrics lack GM0 comparison and cover only 12 GM1 masks.
6. Per-chunk minimum tokens is not Pano-SMB view coverage.
7. Full TeCMA runs are single-seed and not geometry-aware.
