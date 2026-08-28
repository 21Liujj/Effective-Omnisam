# Step 00-1 Patch / View Pipeline Scan

- Run level: L0 scan-only
- Date: 2026-06-29
- Scope: OmniSAM panorama patch / view pipeline for Geo-TeCMA feasibility
- Code changed: No training/model code changed. This markdown report is the only output.
- Training run: No
- Full training run: No
- Memory selection changed: No
- Inference changed: No
- Depends on old memory line: Yes, this scan is for inserting geometry priors on top of the existing TeCMA/memory-efficient line.

## 1. Relevant Code Paths

| Purpose | Path | Key lines / symbols |
|---|---|---|
| DensePASS val dataset | `dataset/DensePASS.py` | `DensePASS13`, `DensePASS19`, `_sliding_windows`, `_get_frame_cnt`, `__getitem__` |
| WildPASS PL dataset | `dataset/WildPASS.py` | `WildPASSforPL._sliding_windows`, `WildPASSforPL.__getitem__` |
| WildPASS pseudo-label adaptation dataset | `dataset/WildPASS.py` | `WildPASSforAdaptation._sliding_windows`, `WildPASSforAdaptation.__getitem__` |
| Source Cityscapes dataset | `dataset/Cityscapes.py` | `Cityscapes13._sliding_windows`, `Cityscapes13.__getitem__` |
| Target / val dataloader construction | `utils/build.py` | `build_tgt_for_pl`, `build_tgt_from_pd`, DensePASS/WildPASS crop and sliding params |
| Single-model validation loop | `validation.py` | `validation()`, two pass loops, `merge_width_scatter_np` |
| Pseudo-label generation loop | `adaptation/pl_generator.py` | `PseudoLabelGenerator.run()`, forward/backward patch pass, merge |
| UDA train frame loop | `adaptation/strategies.py` | `WithMemStrategy.forward()`, `for frame_idx in range(frame_cnt)` |
| Model input / output | `model/model.py` | `OmniSAM_adapt.forward(frame_idx, x, mem_type, ...)` |
| Panorama merge utilities | `utils/tools.py` | `merge_width_scatter_np`, `merge_width_scatter_np_uncertainty` |
| PSR-PLD merge utility | `utils/pl_merge.py` | `merge_patch_sequence` |

## 2. Input Tensor Shape Into Model

### DensePASS / WildPASS validation

`validation.py` constructs DensePASS validation with:

```text
crop_size=(4096, 1024)
sliding_window=(384, 1024)
passes=2
```

For one DataLoader batch:

```text
image:     (B, 18, 3, 1024, 1024)
label:     (B, 18, 1024, 1024)
ori_label: (B, 1024, 4096) before later resize/alignment for DensePASS metric reporting
```

Reasoning from code:

```text
frame_cnt = (4096 - 1024) // 384 + 1 = 9
passes=2 => 18 patches/views per panorama
```

In the validation loop, each patch is passed to the model as:

```text
frame_img = image[:, frame_idx]
model(frame_idx, frame_img)
frame_img shape: (B, 3, 1024, 1024)
```

`OmniSAM_adapt.forward()` reads:

```text
_, _, height, width = x.shape
backbone_out = self.encoder(x)
...
pred = F.interpolate(..., size=(height, width))
return pred, feat, feat_de
```

So patch-level logits returned to validation / PL generation are:

```text
output / pred: (B, num_classes, 1024, 1024)
```

### Pseudo-label generation

`PseudoLabelGenerator.run()` receives `images` from target PL dataset and computes:

```text
B = images.shape[0]
frame_cnt = images.shape[1] // 2
H, W = images.shape[-2:]
logits_maps = np.zeros((B, 2 * frame_cnt, C, H, W))
```

With current 4096x1024 / stride 384 / window 1024 settings:

```text
images:      (B, 18, 3, 1024, 1024)
frame_cnt:   9
logits_maps: (B, 18, C, 1024, 1024)
```

### UDA training after pseudo-labels

`WildPASSforAdaptation._sliding_windows()` uses the same width-only crop rule but randomly reverses patch order instead of always appending a second pass. With 4096x1024 / stride 384 / window 1024:

```text
image: (B, 9, 3, 1024, 1024)
label: (B, 9, 1024, 1024)
```

`WithMemStrategy.forward()` then does:

```text
frame_cnt = images_t.shape[1]
for frame_idx in range(frame_cnt):
    frame_img_t = images_t[:, frame_idx]
    pred_t, feat_t, feat_t_de = model(frame_idx, frame_img_t, mem_type='target')
```

## 3. Patch / View Split Logic

The split is horizontal-only over ERP width. There is no vertical split.

DensePASS split logic in `dataset/DensePASS.py`:

```text
for x in range(0, w, stride):
    if x + window_size <= w:
        patched_image.append(image[:, :, x:x + window_size])
        patched_label.append(label[:, x:x + window_size])
if self.passes == 2:
    patched_image.extend(reversed(patched_image))
```

For DensePASS val / PL-like 4096 width:

| frame_idx in forward pass | ERP x_start | ERP x_end |
|---:|---:|---:|
| 0 | 0 | 1024 |
| 1 | 384 | 1408 |
| 2 | 768 | 1792 |
| 3 | 1152 | 2176 |
| 4 | 1536 | 2560 |
| 5 | 1920 | 2944 |
| 6 | 2304 | 3328 |
| 7 | 2688 | 3712 |
| 8 | 3072 | 4096 |

For `passes=2`, patch indices 9-17 are the reversed patch list. In validation and PL generation, code processes them as the second pass via:

```text
image[:, num_frames + frame_idx]
```

and later the merge utility flips that second pass back to match forward order.

## 4. Patch / View Metadata Availability

Current dataloaders return only tensors and names/sizes; they do not return explicit patch metadata.

### DensePASS validation return

`DensePASS13.__getitem__()` / `DensePASS19.__getitem__()` returns:

```text
image, label, size, name, ori_label
```

Where:

```text
image: stacked patch tensor
label: stacked patch label tensor
size: np.asarray(label).shape, e.g. (18, 1024, 1024)
name: file-relative image name with leading slash stripped
ori_label: resized full ERP label tensor, e.g. (1024, 4096)
```

No explicit fields are returned for:

```text
patch_id / view_id
x_start / y_start
crop box
original unresized image size
pass direction
```

### WildPASS PL return

`WildPASSforPL.__getitem__()` returns:

```text
image, size, name
```

or, when depth prior is enabled:

```text
image, size, name, depth_patch
```

Again, no explicit crop boxes or per-view coordinates are returned.

### UDA adaptation target return

`WildPASSforAdaptation.__getitem__()` returns:

```text
image, label, size, name
```

or with depth:

```text
image, label, size, name, depth_patch
```

No explicit crop box / view metadata is returned.

## 5. Patch Prediction Aggregation Back To ERP Panorama

### Validation aggregation

`validation.py` stores patch logits as:

```text
logits_maps: (B, 2 * num_frames, C, H, W)
```

Then DensePASS/WildPASS predictions are merged by:

```text
merge_width_scatter_np(
    logits_np=logits_maps,
    orig_size=(1024, 4096),
    W_patch=W,
    stride_w=384,
    passes=2,
    return_probs=False,
)
```

`utils/tools.py::merge_width_scatter_np()`:

1. Applies softmax per patch.
2. Reshapes from `(N, B, C, H, Wp)` to `(N, passes, L, C, H, Wp)`.
3. For `passes=2`, flips the second pass along patch dimension.
4. Averages forward/backward pass probabilities.
5. Uses `starts = arange(0, W_full - W_patch + 1, stride_w)` and `scatter_add_` along width.
6. Returns `argmax` as `(N, H_full, W_full)`.

Important note: the current merge does `scatter_add_` on probabilities without dividing by per-pixel coverage. This is current behavior; this report does not change it.

For DensePASS metrics, validation then resizes merged prediction from 4096x1024 to 2048x400 before computing metrics:

```text
pred_resized: (B, 400, 2048)
gt_eval:      (B, 400, 2048)
```

### Pseudo-label aggregation

`adaptation/pl_generator.py` uses the same patch sequence shape and either:

```text
merge_width_scatter_np_uncertainty(..., threshold=uc_threshold)
```

or PSR-PLD path:

```text
merge_patch_sequence(..., cfg=PlMergeConfig(...))
```

Both paths use the same width-only mapping:

```text
orig_size=(1024, 4096)
W_patch=1024
stride_w=384
passes=2
```

## 6. ERP Coordinate Mapping Feasibility

### Can patch / view map back to ERP seam / polar / equator?

Yes, for DensePASS/WildPASS validation and PL generation, patch/view-to-ERP mapping is feasible from deterministic sliding-window parameters, even though metadata is not explicitly returned.

The mapping can be derived as:

```text
L = (W_full - W_patch) // stride_w + 1
base_idx = frame_idx % L
if second pass is present, frame_idx >= L means reversed pass before merge alignment
x_start = base_idx * stride_w   for forward-aligned order
y_start = 0
x_end = x_start + W_patch
y_end = H_full
```

For the second pass, raw dataloader order is reversed, but validation/PL generation explicitly processes `frame_cnt + frame_idx` and merge code flips the second pass back. A geometry hook should therefore normalize to forward-aligned patch id before assigning region.

### Seam mapping

In resized ERP coordinates `(H_full=1024, W_full=4096)`, seam pixels are near:

```text
x ~= 0 or x ~= W_full - 1
```

The patches overlapping seam are directly identifiable from `x_start/x_end`. For example:

```text
patch 0: x=[0, 1024)        overlaps left seam
patch 8: x=[3072, 4096)     overlaps right seam
```

### Polar mapping

There is no vertical crop. Every patch spans full ERP height:

```text
y=[0, 1024)
```

Therefore polar/equator assignment is not patch-level exclusive. It must be computed at token/pixel rows inside each patch, or approximated as a patch-level mixed region. For Geo-TeCMA, polar/equator is feasible only if memory token spatial position within the patch is available or reconstructed from token grid coordinates.

### Token-to-ERP mapping implication for later phases

For patch-level mapping:

```text
global_x = x_start + local_x
global_y = local_y
```

For memory-token mapping, later tasks must confirm token grid shape and token ordering. If memory tokens preserve a 2D grid or `mem_hw`, then local token `(tx, ty)` can be mapped approximately to ERP coordinates by scaling token centers into the patch box:

```text
global_x = x_start + (tx + 0.5) * W_patch / token_W
global_y =          (ty + 0.5) * H_full  / token_H
```

This part is not fully proven in Task 0-1; it belongs to Task 0-2 / 0-3.

## 7. Dataloader Metadata Summary

| Dataset / stage | Return fields | Explicit image_id? | Explicit crop box? | Original size? | Patch order |
|---|---|---:|---:|---:|---|
| `DensePASS13/19` val | `image, label, size, name, ori_label` | `name` only | No | Resized full label only; no raw original size | deterministic forward + reversed if `passes=2` |
| `WildPASSforPL` | `image, size, name[, depth_patch]` | `name` only | No | `size` after resize | deterministic forward + reversed |
| `WildPASSforAdaptation` | `image, label, size, name[, depth_patch]` | `name` only | No | `size` after resize/label shape | forward or reversed randomly |
| `Cityscapes13` source | `image, label, size, name` | `name` only | No | resized image shape | forward or reversed randomly |

## 8. Uncertain Items / Risks

1. `DensePASS.py` does not store explicit crop boxes. Mapping is derivable from `crop_size`, `window_size`, `stride`, `passes`, and `frame_idx`, but future hooks must be careful with reversed order.
2. Training target (`WildPASSforAdaptation`) randomly reverses patch order. Without an explicit reverse flag, `frame_idx` alone may not always map to the same ERP x range during training. This is a risk for geometry-aware training hooks.
3. Polar/equator cannot be represented accurately as a single patch-level label because every patch spans the full ERP height. It requires token-level or pixel-row-level geometry assignment.
4. Validation/PL generation use a fixed `orig_size=(1024, 4096)` in merge. DensePASS metrics later resize to `(400, 2048)`, so region metrics must choose and document the coordinate system.
5. This scan does not yet prove memory token ordering or token-to-patch mapping. That is Task 0-2.

## 9. Acceptance Answer

Patch / view can be mapped back to ERP seam / polar / equator regions with conditions:

- Seam mapping is directly feasible at patch/view level because horizontal crop starts are deterministic.
- Polar/equator mapping is feasible at pixel/token-row level, not as a single pure patch label, because each patch spans the full ERP height.
- Current dataloaders do not return explicit crop metadata, so Geo-TeCMA should either reconstruct it from `frame_idx + sliding_window config` or add a diagnostic metadata hook later.
- For training datasets that randomly reverse patch order, robust geometry mapping will require either disabling/replacing random reversal for geometry-aware hooks or recording the actual patch order/reverse flag.

## 10. Next Step Recommendation

Proceed to Phase 0 / Task 0-2: scan memory token generation and determine whether memory tokens preserve enough spatial grid information (`mem_hw`, token order, patch/frame id) to support token-level seam / polar / equator assignment.

