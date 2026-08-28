# Step 01 — Memory Chunk / View / Budget Semantics Audit

- Date: 2026-07-15
- Repository: `/root/autodl-tmp/OmniSAM-main`
- Commit inspected: `57b7654`
- Run Level: **L0 scan-only**
- Code modified: **No**
- Training run: **No**
- Pano-SMB implementation: **No**

## 1. Executive Summary

**Final classification: `CHUNK_EQUALS_VIEW_FIXED_K`.**

Qualification: a code-level chunk is a batched tensor `(L,B,C)`. It corresponds to one historical sequence frame/view slot, while its batch dimension contains one view instance from each of the `B` panoramas in the batch. It does **not** concatenate multiple view positions of one panorama along the token dimension. Therefore, at the per-sample logical level, one chunk equals one historical patch/view.

In the verified `A_Dtrain_Etrain` setting:

```text
one raw history chunk:       (4096, 4, C), mem_hw=64×64
MTC pool stride 2:           (1024, 4, C), effective grid 32×32
legacy Top-K keep_ratio=.5:  ( 512, 4, C)
maximum history at frame 8:  8 chunks
concatenated attention mem:  (4096, 4, C)
object-pointer tokens:       0
special/no-memory tokens:    0 in the attention memory tensor
```

Thus legacy TeCMA already gives every available historical view chunk the same nonzero `K_chunk=512` under the standard fixed-shape configuration. Ordinary “every view receives at least some tokens” and “prevent an available view from receiving zero tokens” are not new capabilities. What is absent—and can remain new—is a **panorama-global fixed budget that reallocates tokens non-uniformly among history views while enforcing a floor**, plus bounded topology protection within that global budget.

The previous Pano-SMB premise that all view tokens enter one unconstrained global Top-K is incorrect for this code. The existing Top-K runs before concatenation, independently per history chunk.

**Recommended next step: `REDESIGN_AS_GLOBAL_BUDGET_REALLOCATION`.**

## 2. Memory Chunk Generation

| Step | File | Class/Function | Variable | Shape | Semantic Meaning | Evidence |
|---|---|---|---|---|---|---|
| Panorama input | `dataset/WildPASS.py` | `WildPASSforAdaptation.__getitem__` | `image` before split | `(3,1024,4096)` after resize | One target panorama sample | lines 192–210 |
| Patch/view split | same | `_sliding_windows` | `patched_image` | list of nine `(3,1024,1024)` tensors | Nine horizontally overlapping patch/view instances | lines 174–185; crop 4096, window 1024, stride 384 |
| Sequence stack | same | `_sliding_windows` return | `image` | `(9,3,1024,1024)` per sample | Tensor-order patch sequence; may be reversed as a whole | lines 182–185 |
| Batched sequence | `utils/build.py` | `build_tgt_from_pd` DataLoader | `images_t` | `(B,9,3,1024,1024)`, verified run `B=4` | Batch of four panoramas, each with nine view slots | lines 374–389; main script `BATCH_SIZE=4` |
| Frame/view extraction | `adaptation/strategies.py` | `WithMemStrategy.forward` | `frame_img_t=images_t[:,frame_idx]` | `(B,3,1024,1024)` | One logical sequence view slot for every sample in the batch | lines 199, 233, 242–263 |
| Model forward | `model/model.py` | `OmniSAM_adapt.forward` | `current_vision_feats` | BCHW | Current patch/view feature map | lines 217–229 |
| Memory encoding | same + `model/memory_bank.py` | `_encode_new_memory` | `current_memory_feats` | before flatten `(B,C,64,64)`; after `(4096,B,C)` | Spatial memory generated from exactly the current frame/view slot | model lines 293–297; bank lines 81–87 |
| Bank entry | `model/memory_bank.py` | `_encode_new_memory` | `output_dict_target['frame_outputs'][frame_idx]` | dict holding `(4096,B,C)`, pos, `mem_hw` | One history-frame entry keyed by sequence `frame_idx` | lines 117–130 |
| History lookup | same | `MemoryBank.forward` | `t_pos_and_prevs` | Python list of `(t_pos, entry-or-None)` | Up to `num_maskmem` prior frame entries | lines 325–342 |
| Chunk materialization | same | `MemoryBank.forward` | `feats`, `maskmem_enc` | `(L,B,C)` each | One stored historical frame/view slot, batch-packed | lines 340–347 |
| MTC | `model/modules/memory_compressor.py` | `compress` | `feats`, `maskmem_enc` | standard `4096→1024` along L | Per-chunk 2D pooling; B is preserved | lines 90–101, 134–187 |
| Per-chunk retriever | `memory_bank.py`; `sparse_memory_retriever.py` | `retrieve` | `feats2`, `pos2`, `idx2` | standard `1024→512`, `(512,B,C)` | One Top-K over spatial positions in this history chunk | bank lines 374–399; retriever lines 202–239 |
| Chunk list | `memory_bank.py` | `MemoryBank.forward` | `to_cat_memory`, `to_cat_memory_pos_embed` | Python lists of `(K_chunk,B,C)` tensors | Ordered selected history chunks | lines 403–405 |
| Concatenation | same | `torch.cat(...,dim=0)` | `memory`, `memory_pos_embed` | `(sum_j K_j,B,C)` | Full memory sequence passed to attention | lines 438–452 |
| Attention | same | `sam_model.memory_attention` | `memory`, `memory_pos` | verified maximum `(4096,4,C)` | All selected prior-view chunks concatenated on L | lines 447–453; run profile |
| Reset boundary | `adaptation/strategies.py` | `reset_memory` | source/target frame dictionaries | emptied after one strategy forward/batch | History does not continue into the next training batch | lines 956–968 |

### Chunk container, origin, and token order

- Before concatenation, chunks live in Python lists: `to_cat_memory` and `to_cat_memory_pos_embed`.
- Each element originates from one `frame_outputs[prev_frame_idx]` dictionary entry.
- A raw chunk uses row-major spatial order because memory encoding does `flatten(2).permute(2,0,1)` and stores `mem_hw`.
- MTC preserves a coarser row-major grid.
- Sparse retrieval sorts selected indices numerically before `index_select`; retained tokens therefore return to ascending original spatial-index order.
- The entry is keyed by `frame_idx`, but the feature tensor itself carries no `frame_idx`, `view_idx`, or `patch_idx` field.
- The retriever receives `memory_id` constructed from current frame, `t_pos`, and iteration, not an explicit physical view ID.
- No object pointers are appended; `num_obj_ptr_tokens=0` is passed by `OmniSAM_adapt`.
- `maskmem_tpos_enc` is added elementwise to positional embeddings; it is not an additional token.
- `no_mem_embed` is used only when no history is available and is not concatenated into `memory`.

## 3. Chunk-View-Frame Mapping

### 3.1 Static relationship

For current sequence frame `f`, the lookup is:

```text
t_pos = 0 ... num_maskmem-1
prev_frame_idx = f - (num_maskmem - t_pos)
skip entries whose prev_frame_idx is absent
```

With `num_maskmem=9` and a nine-view training sequence, the maximum occurs at current `frame_idx=8`: available previous entries are frame indices `0...7`, hence eight chunks. The current frame is encoded only after its attention/decoder forward, so it is not part of its own history.

### 3.2 Real training sample example

The canonical fixed-PL `used_path.txt` contains `Amsterdam/1.jpg`, and the target DataLoader does not enable shuffle. The standard run uses batch size four. Static code proves its tensor-order view at each `frame_idx` is encoded into the correspondingly keyed history entry. The physical patch ID and ERP range are marked `UNKNOWN` because `_sliding_windows` may reverse the sequence and returns no reverse flag; this audit does not investigate or repair that metadata.

At current `frame_idx=8` for the batch containing `Amsterdam/1.jpg`:

| Chunk ID (append order) | History Index `t_pos` | Frame Index | View/Patch ID | ERP Range | Raw Spatial Tokens | After MTC / Before Top-K | After Top-K | Reliable Mapping? |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 0 | 1 | 0 | tensor-order view 0; physical patch UNKNOWN | UNKNOWN | 4096 | 1024 | 512 | history frame yes; physical patch no |
| 1 | 2 | 1 | tensor-order view 1; physical patch UNKNOWN | UNKNOWN | 4096 | 1024 | 512 | history frame yes; physical patch no |
| 2 | 3 | 2 | tensor-order view 2; physical patch UNKNOWN | UNKNOWN | 4096 | 1024 | 512 | history frame yes; physical patch no |
| 3 | 4 | 3 | tensor-order view 3; physical patch UNKNOWN | UNKNOWN | 4096 | 1024 | 512 | history frame yes; physical patch no |
| 4 | 5 | 4 | tensor-order view 4; physical patch UNKNOWN | UNKNOWN | 4096 | 1024 | 512 | history frame yes; physical patch no |
| 5 | 6 | 5 | tensor-order view 5; physical patch UNKNOWN | UNKNOWN | 4096 | 1024 | 512 | history frame yes; physical patch no |
| 6 | 7 | 6 | tensor-order view 6; physical patch UNKNOWN | UNKNOWN | 4096 | 1024 | 512 | history frame yes; physical patch no |
| 7 | 8 | 7 | tensor-order view 7; physical patch UNKNOWN | UNKNOWN | 4096 | 1024 | 512 | history frame yes; physical patch no |

`t_pos=0` maps to `prev_frame_idx=-1` and is skipped.

### 3.3 Required answers

1. **Does one chunk correspond to only one view?** Logically yes per sample: one chunk is one historical sequence frame/view slot. Physically the tensor is batch-packed and contains one such view instance for each of B panoramas.
2. **Can the same view produce multiple chunks?** Not within one source/target memory bank during one training sequence: each `frame_idx` key is written once and read as one entry. Overlapping patches may share ERP content, but they remain distinct views.
3. **Can one chunk contain multiple views?** It does not contain multiple sequence positions from one panorama. It does contain B per-sample view instances at the same tensor-order position along the batch dimension.
4. **Is `frame_idx` equal to `view_idx`?** It equals tensor-order sequence-view index. It is not proven equal to a stable physical patch ID.
5. **Is history index only memory temporal order?** Yes. `t_pos` controls temporal positional encoding and maps to a previous `frame_idx`; it is not a separate panorama view identifier.
6. **Does patch-sequence order match memory-history order?** Yes for tensor order: strategy iterates increasing `frame_idx`, writes with that key, and memory reads earlier keys in increasing append order. Physical ERP order is UNKNOWN here.
7. **Can the selector directly obtain `view_id`?** No explicit `view_id` is passed. Its caller knows `prev_frame_idx`/`t_pos`, but only `memory_id` with current frame and `t_pos` is passed; the tensor carries no view metadata.

## 4. Budget Scope

| Budget Variable | Verified Value | Scope | Code Location | Formula | Evidence |
|---|---:|---|---|---|---|
| `keep_ratio` | 0.5 | per chunk | TeCMA YAML → Engine → retriever | base for `round(L*ratio)` | config and engine lines 265–292 |
| `min_keep_tokens` / `min_tokens` | 64 | per chunk | same | `k=max(round(L*r),min_tokens,1)` | retriever lines 39–49 |
| `max_tokens` | `None` in caller | per chunk if supplied | retriever API | optional `k=min(k,max_tokens)` | API/caller |
| raw spatial L | 4096 | per chunk | memory encoder | `64*64` | `mem_hw`, Geo/runtime evidence |
| MTC kernel/stride | 2/2 | per chunk | `A_Dtrain_Etrain.yaml` | 64×64→32×32, 4096→1024 | config/compressor/profile |
| Top-K `k` | 512 | per chunk in standard run | retriever | `round(1024*0.5)` | code + run profile |
| `num_maskmem` | 9 | maximum history lookup slots | TrainConfig/MemoryBank | loop over 9 temporal slots | `run_adaptation.py`, profile `memory_bank_size=9` |
| available chunk count | current frame index, max 8 in 9-frame training sequence | per panorama sequence/batch | history lookup | prior stored frames only | memory-bank loop |
| concatenated L | max 4096 | per attention call, all available chunks | `torch.cat(dim=0)` | `N_chunks*512` in standard run | run `memory_tokens_scope` |
| batch size B | 4 | training batch | main experiment script | tensor batch dimension | script/run setup |
| explicit `token_budget` | absent | N/A | repository search | N/A | no selector variable |
| `num_memory_tokens` | absent as control | N/A | repository search | N/A | diagnostics only use counts |
| `max_memory_tokens` | absent | N/A | repository search | N/A | no config/control |

### Budget conclusions

- Total memory sequence length grows linearly with available chunk count: `L_attention = Σ_j K_j`; under fixed standard shape, `=N_chunks*512`.
- The same formula and same K apply to each standard chunk. No score-based cross-chunk budget reallocation exists.
- There is no panorama-global exact budget independent of history length.
- Different chunks cannot dynamically receive different K based on importance under the verified fixed-shape setting. K can differ only mechanically if chunk L differs, a max is configured, or the below-minimum bypass applies.
- A nonempty processed chunk cannot select zero tokens: `_safe_topk_count` enforces at least one, and chunks with `L<=min_tokens` bypass selection and retain all tokens.
- Because selection index is along L and shared across B, every sample-view slice in the chunk retains the same number K, although the selected spatial positions are computed from a score averaged across B.

## 5. Real Token Count Example

Verified `A_Dtrain_Etrain_seed1_20260616_121135`, final/current frame with eight history chunks:

| Quantity | Per Chunk | Chunk Count | Concatenated / Attention Scope |
|---|---:|---:|---:|
| Raw spatial memory | 4096 | 8 | 32768 token positions before MTC |
| After MTC | 1024 | 8 | 8192 positions before sparse Top-K |
| After Top-K | 512 | 8 | 4096 positions entering attention |
| Object pointer | 0 | N/A | 0 |
| Special/no-memory tokens in `memory` | 0 | N/A | 0 |
| Attention memory sequence length L | N/A | N/A | 4096 |
| Batch-packed tensor | `(512,4,C)` | 8 | `(4096,4,C)` |
| Physical token instances if multiplying by B | 2048 | 8 | 16384 |

Direct run evidence:

```text
memory_bank_size=9
compression_tokens_before=4096
compression_tokens_after=1024
compression_ratio=0.25
sparse_tokens_before=8192
sparse_tokens_after=4096
sparse_keep_ratio=0.5
memory_tokens_after_final=4096
memory_tokens_scope=per_attention_call_all_chunks_concat
```

The compressor fields reflect the last per-chunk compression stats, while sparse fields aggregate all chunks in the final recorded call. Their combination exactly supports eight chunks × 1024 before Top-K and eight × 512 after Top-K.

At frame 0, no history is used. At frames 1...8, available chunks and selected attention length are respectively `1...8` and `512...4096`, assuming all entries and standard shapes are present.

## 6. Legacy Selection Pseudocode

```python
# One training batch contains B panoramas and F patch/view slots.
for frame_idx in range(F):
    current_view_batch = images[:, frame_idx]      # (B,3,H,W)

    chunks = []
    for temporal_slot in range(num_maskmem):
        prev_frame_idx = frame_idx - (num_maskmem - temporal_slot)
        entry = frame_outputs.get(prev_frame_idx)
        if entry is None:
            continue

        memory = entry.features                   # (4096,B,C)
        pos = entry.pos
        memory, pos = mtc_pool(memory, pos)       # (1024,B,C)

        # score is (L,), averaged across B; one index set shared by B views.
        score = feature_norm(memory).mean_over_batch()
        K_chunk = max(round(L * keep_ratio), min_tokens, 1)
        idx = topk(score, K_chunk)
        chunks.append(memory[idx])                # (512,B,C)
        pos_chunks.append(pos[idx])

    memory_attention(
        memory=cat(chunks, dim=0),
        memory_pos=cat(pos_chunks, dim=0),
        num_obj_ptr_tokens=0,
    )

    frame_outputs[frame_idx] = encode_current_view(current_view_batch)

reset_memory_after_batch()
```

## 7. Implications for Pano-SMB

### 7.1 Claims that must change

The old Pano-SMB document should not claim:

- legacy Norm Top-K places all views into one global pool before selection;
- a historical view can be assigned zero tokens by the existing selector;
- ordinary equal/fixed per-view quota is a new mechanism;
- zero-view prevention alone is a new contribution.

The verified baseline is better described as:

```text
uniform independent per-history-view budgeting
= each available history chunk receives the same ratio-derived K_chunk
  before chunks are concatenated for memory attention
```

### 7.2 What remains new

A defensible revised Pano-SMB direction is:

```text
same total budget as the uniform per-chunk legacy baseline
→ allocate a minimum floor to every valid history view
→ redistribute the remaining panorama-global budget non-uniformly
→ apply bounded protected reservations within that shared budget
→ exact correction to a fixed total K_global
```

This changes the method from “rescuing views dropped by global Top-K” to **adaptive global budget reallocation versus a uniform per-view baseline**.

### 7.3 Required boundary answers

1. **Is per-view quota new?** A generic fixed/equal quota is not new in effect: legacy already retains fixed K for every available standard chunk/view. A *minimum floor combined with non-uniform global residual reallocation* would be new.
2. **Can zero-view prevention be claimed?** Not as a new baseline deficiency fix. Legacy nonempty chunks cannot become zero. It may be stated only as a preserved invariant of the redesigned allocator.
3. **Is panorama-global exact budget new?** Yes. Legacy total grows with history chunk count and has no cross-chunk exact-K allocator.
4. **Where should overlap/seam reservations operate?** If pursued later, they should compete inside the panorama-global cross-chunk budget; implementing them independently inside each already-fixed chunk would not establish budget reallocation. This is an architectural implication only; no seam metadata was audited here.
5. **Which document claims need revision?** Replace “all view tokens compete globally and some views may be fully removed” with “legacy uses uniform independent per-view budgets and cannot adapt token allocation to heterogeneous cross-FoV importance.” Reframe view quota as a floor within global reallocation, not a standalone novelty. Reframe exact budget as fixed total across history views, not merely exact per-chunk Top-K.

### 7.4 Batch-level risk

Legacy feature norm averages scores across B, then applies the same spatial indices to all B panorama views in that chunk. Therefore each view gets K tokens, but per-sample salient locations may be suppressed by batch averaging. This is a separate potential issue from view-level budget allocation and should not be silently mixed into Pano-SMB claims.

## 8. Final Classification

```text
CHUNK_EQUALS_VIEW_FIXED_K
```

Rationale: each history entry is generated by one `frame_idx` view forward and is selected independently before concatenation. The tensor is batch-packed, but it does not combine multiple view positions from one panorama along L. Under standard fixed shapes, every available logical history view receives exactly the same K and cannot be emptied.

## 9. Recommended Next Step

```text
REDESIGN_AS_GLOBAL_BUDGET_REALLOCATION
```

Revise Pano-SMB before implementation so its baseline is uniform per-view/chunk Top-K and its new operation is matched-total-budget redistribution with per-view floors and exact global K. Do not start selector construction until that method definition is updated and approved.

No seam metadata work and no Pano-SMB construction were performed.

## 10. Evidence Index

| Evidence ID | Path | Function/Lines | Supports |
|---|---|---|---|
| E1 | `dataset/WildPASS.py` | `_sliding_windows`, 174–185 | one panorama becomes nine tensor-order patch/views |
| E2 | `utils/build.py` | `build_tgt_from_pd`, 358–389 | DensePASS target dimensions and batched loader |
| E3 | `adaptation/strategies.py` | 199, 233–263 | one `frame_idx` slice forwarded at a time |
| E4 | `model/model.py` | 217–229, 293–297 | memory read then current-frame write |
| E5 | `model/memory_bank.py` | 81–138 | one keyed frame entry and `(L,B,C)` flattening |
| E6 | `model/memory_bank.py` | 325–405 | history lookup, per-chunk MTC/retrieval, chunk lists |
| E7 | `model/memory_bank.py` | 438–453 | concatenate then attention; pointer count propagation |
| E8 | `model/modules/memory_compressor.py` | 90–101, 134–187 | row-major grid and per-chunk 2D pooling |
| E9 | `model/modules/sparse_memory_retriever.py` | 39–55, 150–239 | per-call K formula, batch-mean norm score, nonzero floor |
| E10 | `experiments/tecma/configs/A_Dtrain_Etrain.yaml` | pool stride 2, keep ratio .5, minimum 64 | standard budget configuration |
| E11 | `experiments/tecma/.../run_summary.json` | profile fields | measured 4096→1024 and aggregate 8192→4096 |
| E12 | corresponding `train_timing.jsonl` | `memory_bank_size`, token scope | max history and final attention length |
| E13 | `experiments/tecma/scripts/run_main_experiments.sh` | default BATCH_SIZE=4 | batch-packed real example |
| E14 | canonical `used_path.txt` | first entries include `Amsterdam/1.jpg` | real target-sample evidence |
| E15 | `adaptation/strategies.py` | 956–968 | memory reset after one batch/sequence |

## 11. Unknowns and Risks

1. Physical patch ID/ERP range is UNKNOWN when random reversal is unrecorded; only tensor-order frame/view mapping is proven.
2. The exact first four samples are ordered by the canonical path file and unshuffled target loader, but worker-side reversal decisions remain per sample and unknown.
3. `last_compression_stats` records the latest chunk rather than all chunk stats; aggregate sparse and final counts make the standard maximum example consistent, but no new forward was run.
4. The selected index is shared across B because the score averages over batch. Per-sample token-location quality is not diagnosed.
5. `num_maskmem=9` is a lookup capacity; the nine-frame training sequence provides at most eight previous chunks at the last frame.
6. If chunk shape changes, per-chunk K changes mechanically through `round(L*keep_ratio)`; “fixed K” refers to the verified standard fixed-shape configuration.
7. Nonzero object pointers are outside the current OmniSAM path and remain unverified.
8. Legacy optional frame-level `select_mode` could drop whole chunks if separately enabled, but verified TeCMA plugin configs use the sparse token retriever path and do not enable that mode.
