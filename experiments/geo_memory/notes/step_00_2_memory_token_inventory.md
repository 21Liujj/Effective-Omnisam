# Phase 0 / Task 0-2 — Memory Token Inventory for Geo-TeCMA

Date: 2026-06-29
Repo: `/root/autodl-tmp/OmniSAM-main`
Commit: `57b7654`
Mode: scan-only. No training, no model/config change, no checkpoint write.
Depends on: `experiments/geo_memory/notes/step_00_1_patch_view_pipeline.md`

## 1. Question

Task 0-1 confirmed the panorama view sequence geometry:

- ERP width is split into horizontal patches: `crop_size=(4096, 1024)`, `window_size=1024`, `stride=384`, `frame_cnt=9`.
- Each patch covers the full ERP height, so patch-level geometry can locate horizontal seam/overlap regions but cannot distinguish polar/equator rows.
- DensePASS validation and PL generation use deterministic forward + reversed passes.
- `WildPASSforAdaptation` can randomly reverse patch order.

Task 0-2 checks whether memory tokens can be mapped back to patch/view and to a 2D token grid, and whether seam/latitude geometry priors are safe to build on top of the current memory path.

## 2. Memory token generation path

Main path:

1. `model/model.py::OmniSAM_adapt.forward(frame_idx, x, mem_type=...)`
2. encoder produces `current_vision_feats` and `current_vision_pos_embeds`
3. `self.memory_bank(frame_idx, ...)` reads previous memories
4. decoder predicts logits
5. `self.memory_bank._encode_new_memory(frame_idx, current_vision_feats, mask_for_mem, mem_type)` writes new memory

Important code evidence:

- `model/model.py:293-297`
  - `mask_for_mem = torch.argmax(pred, 1).unsqueeze(1)`
  - `_encode_new_memory(frame_idx, current_vision_feats, mask_for_mem, mem_type)`
- `model/memory_bank.py:78-84`
  - memory encoder output is read as BCHW feature map
  - `mem_hw = tuple(current_memory_feats.shape[-2:])`
  - tokens are flattened by `current_memory_feats.flatten(2).permute(2, 0, 1)`
- `model/memory_bank.py:114-126`
  - memory is stored in `output_dict_source` or `output_dict_target`
  - key is `frame_idx`
  - stored fields include `maskmem_features`, `maskmem_pos_enc`, `importance`, and `mem_hw`

Shape convention:

- Before flatten: memory feature map is `(B, C, H_mem, W_mem)`.
- Stored token tensor is `(L, B, C)` where `L = H_mem * W_mem` for local spatial tokens.
- `mem_hw = (H_mem, W_mem)` is retained in the memory entry.
- Source and target memory dictionaries are separate, so Geo metadata must respect `mem_type` and avoid mixing source/target outputs.

## 3. Mapping memory token back to patch/view

### Deterministic mapping exists if patch order is known

Memory entries are stored under the same `frame_idx` that is passed from the dataset/loop into `OmniSAM_adapt.forward`. Therefore, a memory token can map back to a patch/view through:

```text
memory entry -> frame_idx -> patch/view index -> ERP x-start
```

For the standard 4096-wide ERP setup from Task 0-1:

```text
x_start(frame_idx) = frame_idx * 384, for frame_idx in [0, 8]
x_end = x_start + 1024
```

This is valid for deterministic forward-order sequences and for deterministic second-pass validation/PL logic after accounting for the pass direction.

### Missing explicit metadata

Current dataset returns do not carry explicit patch metadata such as:

- patch id
- crop box / x_start / x_end
- pass id or pass direction
- random reverse flag

So the mapping is currently inferred from loop order and known config, not stored as first-class metadata.

## 4. Mapping memory token back to a 2D token grid

This is directly feasible for local spatial memory tokens.

Evidence:

- `model/memory_bank.py:82-84` saves `mem_hw` before row-major flatten.
- `model/modules/memory_compressor.py:91-101` defines reversible conversion:
  - `_tokens_to_map(tokens_lbc, hw)` checks `L == H * W`, then reshapes `(L,B,C)` to `(B,C,H,W)`.
  - `_map_to_tokens(feat_bchw)` reshapes `(B,C,H,W)` back to `(L,B,C)`.

Therefore, for token index `i` in a local spatial memory tensor:

```text
token_row = i // mem_w
token_col = i % mem_w
```

Caveats:

- If compression is applied before bank, `mem_hw` may become the compressed grid size.
- If compression is applied before attention, the bank stores original grid tokens, while attention consumes compressed tokens.
- If `keep_global_tokens > 0`, appended global tokens are not ordinary spatial grid tokens and must be excluded or handled separately.
- Sparse retrieval selects token indices (`idx`) in `model/modules/sparse_memory_retriever.py`, but `MemoryBank.forward` currently receives `idx2` without storing it in diagnostics. Sparse-selected tokens remain mappable in principle if `idx2` is retained, but the current diagnostics do not persist selected row/col.

## 5. Latitude / polar prior feasibility

Patch-level latitude is insufficient because every horizontal patch covers the full ERP height.

Token-row-level latitude is feasible for local spatial tokens because `mem_hw` provides the memory grid height. A geometry prior can derive latitude/polar/equator position from `token_row`, not from `frame_idx`:

```text
row_center = (token_row + 0.5) / mem_h
latitude-like coordinate can be derived from row_center over the ERP height
```

This should work for DensePASS/WildPASS ERP patches because each patch preserves full vertical extent. Compression by spatial pooling still preserves a coarser row grid, so row-level latitude remains feasible at compressed resolution.

Important limitation: this conclusion applies to local grid tokens. It does not automatically apply to appended global tokens or to sparse-selected tokens unless their original selected indices are retained.

## 6. Seam / horizontal prior feasibility

Seam-aware geometry is patch/view based and depends on correct horizontal patch order.

For deterministic order, it is directly feasible:

```text
frame_idx -> x_start = frame_idx * stride -> overlap/seam distance
```

For validation and PL generation, the two-pass layout is deterministic and the merge utilities already assume the same width-only patch sequence. Seam geometry can be reconstructed from `frame_idx`, pass id, `window_size`, and `stride`.

For adaptation training with random reverse, seam-aware geometry is not directly safe unless order metadata is added or random reverse is disabled/normalized.

## 7. WildPASSforAdaptation random reverse risk

`dataset/WildPASS.py` contains random reversal in adaptation datasets:

- `WildPASSforAdaptation._sliding_windows`, lines 174-185
- `WildPASSforAdaptation19._sliding_windows`, lines 523-534

Pattern:

```text
if np.random.randint(0, 2):
    patched_image.reverse()
    patched_label.reverse()
```

This reverses the full patch sequence but does not return a reverse flag. As a result, during training:

```text
loop frame_idx 0 may mean leftmost patch in one sample/iteration
loop frame_idx 0 may mean rightmost patch in another sample/iteration
```

Impact:

- Seam / horizontal-position prior based only on `frame_idx` can be assigned to the wrong physical ERP x-range.
- Cross-view memory relation based on previous/next `frame_idx` can become geometrically ambiguous.
- Token-row latitude prior is much less affected, because vertical row meaning is unchanged by horizontal sequence reversal.
- A combined seam+latitude prior still inherits seam misalignment risk if it uses `frame_idx` without knowing actual order.

## 8. Acceptance answers

### Can memory tokens map back to patch/view?

Yes, conditionally.

- Technically yes: memory entries are keyed by `frame_idx`, and standard patch geometry can infer patch/view position.
- Operationally unsafe for random-reversed adaptation sequences unless reverse/order metadata is recorded or random reverse is removed/controlled.

### Can memory tokens map back to a 2D token grid?

Yes for local spatial memory tokens.

- `mem_hw` is stored with each memory entry.
- Flattening is row-major via `flatten(2).permute(2, 0, 1)`.
- Existing compressor utilities already convert `(L,B,C) <-> (B,C,H,W)`.

### Can latitude / polar prior be token-row level?

Yes.

- It should be token-row level, not patch-level.
- Each patch covers full ERP height, so `token_row / mem_h` can encode polar/equator position.
- This is directly feasible for local grid tokens and remains feasible after spatial pooling at lower resolution.

### Can seam-aware be directly done?

Partially.

- Directly feasible for deterministic validation/PL paths and any training path with known patch order.
- Not directly safe for current `WildPASSforAdaptation` random-reverse path without metadata/order fix.

### Can latitude-aware be directly done?

Yes for local spatial memory tokens.

- Latitude depends on token row and is not broken by horizontal random reverse.
- It still needs careful handling for global tokens and sparse-selected diagnostics.

### Minimal feasible Geo-TeCMA version

Recommended minimum for the next construction step: diagnosis-first, then guarded seam+latitude.

Reason:

- Token-row latitude is technically feasible now.
- Seam is technically feasible now only where patch order is deterministic.
- Current adaptation data can random-reverse patches without recording the flag, so an unguarded training-time seam prior would be unreliable.

Practical recommendation:

1. First implement diagnosis-only geometry metadata logging/verification:
   - `frame_idx`
   - inferred `x_start/x_end`
   - `mem_hw`
   - optional token-row latitude bins
   - reverse/order status where available
2. Then allow seam-aware only when patch order is known.
3. Allow latitude-aware for local grid tokens.
4. Do not enable an unguarded seam+latitude training prior on `WildPASSforAdaptation` until reverse/order metadata is fixed.

## 9. Required follow-up before implementation

Before Phase 1 code changes, decide one of these safe fixes:

1. Return patch geometry metadata from datasets: patch id, x_start/x_end, pass direction, reverse flag.
2. Disable random reverse for geometry-prior experiments.
3. Keep random reverse but convert `frame_idx` to physical patch id using the recorded reverse flag.

Any new Geo-TeCMA module should default to `enabled: false` and be registered in `docs/claude/module-map.md` when implemented.

