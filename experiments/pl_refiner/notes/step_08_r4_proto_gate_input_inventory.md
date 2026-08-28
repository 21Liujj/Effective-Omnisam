# step_08_r4_proto_gate_input_inventory.md
# Phase 4 / Task 4-1 — R4 prototype gate 输入与可行性扫描

**Run Level**: L0 scan-only  
**日期**: 2026-06-24  
**执行人**: Claude Code  
**是否改代码**: No  
**是否训练**: No  
**是否跑脚本**: No  
**是否实现 R4**: No  
**是否看 mIoU**: No

---

## 0. 本次只读检查范围

按要求只读检查以下文件：

| 文件 | 状态 |
|---|---|
| `adaptation/strategies.py` | 已读取 |
| `adaptation/pseudo_label/refiner.py` | 已读取 |
| `adaptation/pseudo_label/prototype_bank.py` | 已读取 |
| `adaptation/pseudo_label/class_balance.py` | 已读取 |
| `experiments/pl_refiner/configs/R4_proto_gate.yaml` | 已读取，存在 |
| `experiments/pl_refiner/notes/latest_status.md` | 缺失 |
| `experiments/pl_refiner/notes/step_05_r2_class_balance_impl.md` | 已读取 |
| `experiments/pl_refiner/notes/step_06_r1_r2_short_run.md` | 缺失 |
| `experiments/pl_refiner/notes/step_07_plq_basic_impl.md` | 缺失 |

---

## 1. R4 需要哪些输入

R4 的目标限定为：**prototype-guided pseudo-label filtering / valid_mask refinement**。

因此 R4 proto_gate 最小输入集为：

1. `pseudo_label`
   - 离线 pseudo label / 当前 refiner 链路中的 label。
   - R4 不修改 label 内容，只用于按类索引 prototype 与构造候选像素。

2. `valid_mask`
   - R2 class-balanced 后的 mask。
   - R4 应继承 R2 输出，并只在其基础上进一步过滤。

3. `confidence`
   - 在线 softmax max confidence。
   - R2 class-balanced 需要它；R4 本身也可用于诊断或 fallback，但 prototype gate 的核心不是 confidence threshold。

4. `feature map`
   - 每个 target pixel / feature location 的 embedding，用于和对应 class prototype 做 similarity。
   - 当前配置文件写的是 `feat_key: "feat_t_de"`。

5. `class prototype`
   - 每类一个 prototype vector，用于和 feature 做 cosine similarity / distance。
   - 必须按 class 存储，且 feature dim 与 `feat_t_de` channel dim 一致。

6. `ignore_index`
   - 当前为 `255`。
   - 必须保证 ignore 像素不被重新激活。

7. `num_classes`
   - 当前 DensePASS / Cityscapes 13-class 设定下为 `13`。
   - 当前 refiner 用 `pred_t.shape[1]` 传入。

---

## 2. 当前 PseudoLabelRefiner 插入点输入可用性

当前插入点在 `adaptation/strategies.py` 的 `WithMemStrategy.forward()` 中，位置是：

```python
with torch.no_grad():
    probs = torch.softmax(pred_t, 1)
    conf_t, pseudo_t = probs.max(1)
    valid_mask = (frame_pl_t != ignore_index)
    visual_mask = (conf_t > conf_thr) & valid_mask
    ...
    _pl_result = _refiner.refine(
        pseudo_label=frame_pl_t,
        confidence=conf_t,
        student_logits=pred_t,
        features=feat_t_de,
        valid_mask=valid_mask,
        ignore_index=ignore_index,
        step=global_step,
    )
```

| input | variable name | source file/function | expected shape | available? | note |
| ----- | ------------- | -------------------- | -------------- | ---------- | ---- |
| pseudo_label | `frame_pl_t` | `adaptation/strategies.py` / `WithMemStrategy.forward()` | `(B,H,W)` int, values `0..C-1` or `ignore_index` | YES | 当前 refiner 已直接收到。R4 必须保持 `refined_label=pseudo_label`，不做 correction。 |
| valid_mask | initial: `valid_mask`; R2 output: `cb_result.new_valid_mask` / final `mask_t` | `strategies.py` creates `valid_mask=(frame_pl_t != ignore_index)`; `refiner.py::_refine_class_balanced()` / `class_balance.py::ClassBalancedSelector.select()` creates R2 mask | `(B,H,W)` bool | YES | 插入点传入的是原始 offline valid mask。R4 若要继承 R2，需要在 `proto_gate` 内先跑 R2 class-balanced，得到 R2 mask，再做 prototype gate。 |
| confidence | `conf_t` | `strategies.py` / `probs=torch.softmax(pred_t,1); conf_t,_=probs.max(1)` | `(B,H,W)` float | YES | 当前 R2/R3 已使用；R4 继承 R2 时仍需要它。 |
| feature map | `feat_t_de` | `strategies.py` / target forward: `pred_t, feat_t, feat_t_de = model(... mem_type='target' ...)` | expected `(B,Cf,Hf,Wf)`; R4 yaml 注释写 `(B,256,H',W')` | PARTIAL | 已传入 refiner 的 `features` 参数，但 refiner 当前未使用。是否与 PL 分辨率一致未在允许文件中确认；R4 需要 shape guard/interpolate。 |
| class prototype | existing training object: `proto_bank`; R4 placeholder: `adaptation/pseudo_label/prototype_bank.py` | `strategies.py` has `proto_bank.snapshot()`, `proto_bank.update(feat_s_de_ori, frame_lbl_s)`, `proto_bank.mse_pull_loss(feat_t_de, frame_pl_t_f)`; `adaptation/pseudo_label/prototype_bank.py` is placeholder | expected `(num_classes,Cf)` or equivalent class-indexed table | PARTIAL | 训练链路已有 FPA `proto_bank` 对象，但当前 refiner 调用未传入 `proto_bank`，且 `adaptation/pseudo_label/prototype_bank.py` 明确是 placeholder：`PROTOTYPE_BANK_AVAILABLE=False`。prototype API/shape 还需进一步确认。 |
| ignore_index | `ignore_index` | `strategies.py` from `args.ignore_index` default `255`; passed to refiner | scalar int, default `255` | YES | R2 已保证不重新激活 ignore；R4 必须继续保持 `proto_mask & R2_valid_mask`。 |
| num_classes | `self.num_classes` in refiner; constructed from `pred_t.shape[1]` | `strategies.py` / `PseudoLabelRefiner.from_args(args, num_classes=int(pred_t.shape[1]))` | scalar int, expected `13` | YES | R4 yaml 也写 `num_classes: 13`；实际以 model output channel 更可靠。 |

---

## 3. R4 应该使用哪个 feature

### 3.1 是否用 `feat_t_de`

结论：**应该优先使用 `feat_t_de`，但需要 shape guard。**

理由：

- `R4_proto_gate.yaml` 已指定：

```yaml
prototype:
  feat_key: "feat_t_de"   # use decoder features (B, 256, H', W')
  distance_metric: "cosine"
```

- `strategies.py` 当前已经把 `feat_t_de` 传给 refiner：

```python
features=feat_t_de
```

- 现有 FPA 原型链路也使用 decoder feature：

```python
proto_bank.update(feat_s_de_ori, frame_lbl_s)
loss_proto = proto_bank.mse_pull_loss(feat_t_de, frame_pl_t_f)
```

这说明 `feat_t_de` 是当前代码中已用于 source/target prototype alignment 的 feature，作为 R4 prototype gate 的 feature 来源最一致。

### 3.2 feature shape 是什么

从允许读取文件中能确认的是：

- `pred_t` 用于 CE loss，和 `frame_pl_t_f` 对齐，因此 `pred_t` 应为 `(B,num_classes,H,W)`。
- `feat_t_de` 被传给 `proto_bank.mse_pull_loss(feat_t_de, frame_pl_t_f)`，说明其语义上应能与 label 对齐或由 proto_bank 内部处理分辨率差异。
- `R4_proto_gate.yaml` 注释明确写 `feat_t_de` 是 decoder features：`(B,256,H',W')`。

因此当前 L0 判断：

```text
feat_t_de expected shape: (B, 256, Hf, Wf)
pseudo_label shape      : (B, H, W)
valid_mask shape        : (B, H, W)
```

其中 `Hf,Wf` 是否等于 `H,W`，**仅从本次允许文件无法确认**。

### 3.3 feature 分辨率是否和 pseudo_label / valid_mask 一致

结论：**不能假设一致。**

理由：

- R4 yaml 注释使用 `H'`,`W'`，而不是 `H`,`W`。
- `class_balance.py` 和 `refiner.py` 的 mask 逻辑全部假设 `pseudo_label/confidence/valid_mask` 是 `(B,H,W)`。
- 现有 prototype loss 能接收 `feat_t_de + frame_pl_t_f`，但其内部如何对齐不在本次允许文件内。

### 3.4 是否需要 interpolate

结论：**R4 需要准备 interpolate 分支。**

推荐的 R4 gate 形态是：

1. 对 `feat_t_de` 和 class prototype 做 cosine similarity，得到 similarity map：`sim_map`。
2. 若 `sim_map.shape[-2:] != pseudo_label.shape[-2:]`，将 `sim_map` interpolate 到 `(H,W)`。
3. 只在 `R2_valid_mask` 内根据阈值过滤。

原因：R4 的输出必须是 `(B,H,W)` bool `valid_mask`，而 loss path 使用的是 label resolution。

### 3.5 是否需要 normalize

结论：**需要 normalize。**

R4 yaml 指定：

```yaml
distance_metric: "cosine"
```

因此应对：

- feature：按 channel 维 normalize，即 `F.normalize(features, dim=1)`；
- prototype：按 feature dim normalize，即 `F.normalize(prototypes, dim=1)`；

再计算每个 pixel/location 与其 pseudo-label class prototype 的 cosine similarity。

---

## 4. prototype 从哪里来

### 4.1 当前是否已有 `proto_bank`

结论：**训练链路已有一个 `proto_bank` 对象，但 R4 专用 `adaptation/pseudo_label/prototype_bank.py` 尚未实现。**

证据：

`strategies.py` 中 `WithMemStrategy.forward()` 签名已有：

```python
def forward(..., proto_bank, alpha, device)
```

并且当前训练中已有：

```python
proto_bank.snapshot()
proto_bank.update(feat_s_de_ori, frame_lbl_s)
loss_proto = proto_bank.mse_pull_loss(feat_t_de, frame_pl_t_f)
```

但 `adaptation/pseudo_label/prototype_bank.py` 目前只有占位：

```python
"""adaptation/pseudo_label/prototype_bank.py
PrototypeBank -- placeholder for R4/R5.
Will be implemented in Phase 3 / Task 4-1.
"""
PROTOTYPE_BANK_AVAILABLE = False
```

### 4.2 `proto_bank` 是否能 snapshot

结论：**现有训练链路里的 `proto_bank` 能 snapshot。**

证据：`strategies.py` 当前已有：

```python
proto_bank.snapshot()
proto_bank.update(feat_s_de_ori, frame_lbl_s)
```

注意：snapshot 发生在当前 refiner 调用之后：

```text
_refiner.refine(... features=feat_t_de ...)
proto_bank.snapshot()
proto_bank.update(feat_s_de_ori, frame_lbl_s)
```

因此如果 R4 要使用 snapshot prototype，需要明确使用：

- 当前 step 之前的 prototype snapshot；或
- source update 后的新 prototype；

否则会有时序歧义。对于 filtering，建议优先使用**进入当前 target frame 前已经稳定存在的 source/global prototype snapshot**，避免使用当前 target pseudo-label 更新后的信息。

### 4.3 prototype shape 是什么

当前允许文件不能直接确认现有 FPA `proto_bank` 内部张量名和 shape。

R4 所需 shape 应为：

```text
class_prototypes: (num_classes, feature_dim)
expected here  : (13, 256) if using feat_t_de with C=256
```

如果现有 `proto_bank` 的内部 prototype dim 与 `feat_t_de.shape[1]` 不一致，则 R4 不能直接使用。

### 4.4 prototype 是否按 class 存储

从当前调用语义判断：**应当是按 class 存储**，但本次 L0 不能完全确认内部字段。

证据：

- `proto_bank.update(feat_s_de_ori, frame_lbl_s)` 使用 source label 更新；
- `proto_bank.mse_pull_loss(feat_t_de, frame_pl_t_f)` 使用 pseudo label 对 target feature 做 prototype pull；
- 这类 FPA 原型通常必须按 class id 查询。

但由于本次没有读取 `adaptation/methods.py` 或 proto_bank 类定义，内部是否为 `(C,D)`、是否有 valid flag、是否有 count，不在本次允许检查范围内。

### 4.5 source / target / online prototype 分别是否存在

基于本次允许文件，判断如下：

| prototype type | exists? | evidence | note |
|---|---:|---|---|
| source/global prototype | PARTIAL/likely | `proto_bank.update(feat_s_de_ori, frame_lbl_s)` | source feature + source GT label 更新，应该是当前 FPA 主要 prototype 来源。内部 shape/API 未确认。 |
| target prototype | PARTIAL/unclear | 旧路径片段中有 `calculate_batch_prototypes(feat_t_de, frame_pl_t)` 和 `prototype_loss(frame_idx,t_pt)` | 当前主 forward 使用 `mse_pull_loss(feat_t_de, frame_pl_t_f)`，未看到 target prototype 持久更新。 |
| online/snapshot prototype | PARTIAL/likely | `proto_bank.snapshot()` + `proto_bank.delta_mean()` | 能 snapshot，但 snapshot 内容、读取接口、是否能安全提供给 refiner 未确认。 |
| R4-specific pseudo_label PrototypeBank | NO | `adaptation/pseudo_label/prototype_bank.py` only placeholder, `PROTOTYPE_BANK_AVAILABLE=False` | 不能作为当前 R4 输入来源。 |

### 4.6 是否需要新增 prototype 逻辑

结论：**需要新增或暴露 prototype 访问逻辑，但不一定需要重新发明一套 prototype。**

最低需求：

1. 让 R4 refiner 能拿到 class prototype。
2. 明确 prototype tensor shape：`(num_classes, feature_dim)`。
3. 明确 prototype valid flag / zero prototype fallback。
4. 明确使用 source/global/snapshot 哪一种 prototype。
5. 在 prototype 缺失时 fallback 到 R2 mask，而不是误杀全部像素。

当前阻塞点：

- `PseudoLabelRefiner.refine()` 当前没有 `proto_bank` 或 `class_prototypes` 参数。
- `prototype_bank.py` 是 placeholder。
- 现有 `proto_bank` 虽然存在于 `strategies.py`，但没有在当前 refiner call 中传入。

---

## 5. R4 是否能继承 R2 valid_mask

结论：**能，且应该这样设计。**

要求路径：

```text
R2 class-balanced valid_mask
→ prototype similarity gate
→ refined valid_mask
```

对应当前代码结构最自然的设计是：

```text
proto_gate mode inside PseudoLabelRefiner:

1. ClassBalancedSelector.select(
       pseudo_label=pseudo_label,
       confidence=confidence,
       original_valid_mask=(pseudo_label != ignore_index),
       ignore_index=ignore_index,
   )
   -> r2_valid_mask

2. proto_similarity = cosine(feature, prototype[pseudo_label])

3. proto_keep = proto_similarity >= threshold

4. final_valid_mask = r2_valid_mask & proto_keep

5. return PseudoLabelRefineResult(
       refined_label=pseudo_label,   # unchanged
       valid_mask=final_valid_mask,
       pl_weight=None,               # R4 filtering only, no weighted CE
       diagnostics=...
   )
```

必须保持的不变性：

- 不修改 `pseudo_label` 类别。
- 不修改 loss。
- 不做 correction。
- 不重新激活原始 ignore 像素。
- 若 prototype 缺失/非法，应 fallback 为 R2 mask，而不是输出全 False。

---

## 6. R4 最大工程风险

| 风险 | 严重度 | 分析 | 建议 |
|---|---:|---|---|
| feature 和 pseudo_label shape 不一致 | 高 | `feat_t_de` 预期 `(B,256,H',W')`，而 `pseudo_label/valid_mask/confidence` 是 `(B,H,W)`。R4 输出必须回到 `(B,H,W)`。 | R4-2 前必须确认 `feat_t_de.shape[-2:]`；实现时必须有 interpolate 分支和 shape diagnostics。 |
| prototype 缺失或全零 | 高 | `adaptation/pseudo_label/prototype_bank.py` 当前为 placeholder；现有 FPA `proto_bank` 未传入 refiner；rare class prototype 可能尚未初始化。 | 需要 prototype valid flag；缺失/零 norm class 应 fallback keep R2，不参与 prototype gate。 |
| rare class prototype 不稳定 | 高 | R2 的价值在于提高 rare/extreme rare 覆盖；若 rare prototype 来自极少样本，cosine gate 会放大噪声。 | 对 rare class 使用更宽松 threshold 或 warmup；记录 per-class proto_valid_count / reject_ratio。 |
| threshold 误杀 rare class | 高 | 单一 `gate_threshold=0.5` 可能对 common 类合适，但对 traffic light/sign 等极少类过严。 | R4 不应默认全类同阈值；至少 diagnostics 要输出 per-class reject ratio；必要时 per-class threshold。 |
| R2 已经降低 valid ratio，R4 再过滤导致 loss_unsup 太弱 | 高 | R2 已将 total valid ratio 显著降低；R4 再做 AND gate，可能使 unsup loss 有效像素不足。 | 加 `min_valid_ratio/min_valid_pixels` 或 fallback 到 R2；重点观察 `loss_unsup` 和 `pseudo_valid_ratio`，但本次不训练。 |
| prototype 时序歧义 | 中 | 当前 refiner 在 `proto_bank.snapshot()/update()` 之前调用；R4 使用哪个时刻的 prototype 未定义。 | 明确使用 previous snapshot/source global prototype，避免当前 target pseudo label leakage。 |
| refiner 实例每次创建，状态无法长期保存 | 中 | `strategies.py` 在 no_grad 内每次创建 `PseudoLabelRefiner.from_args()`；R2 selector 缓存只在单次实例内有效。 | 若 R4 PrototypeBank 放在 refiner 内，会每步丢状态；prototype 应来自外部 `proto_bank` 或稳定对象。 |
| `proto_gate` mode 当前不在 SUPPORTED_MODES | 中 | `refiner.py` 当前支持 `off/identity/conf_mask/class_balanced/plq_basic`，不支持 `proto_gate`。 | R4-2 才可扩展 mode；本次仅扫描，不实现。 |

---

## 7. 回答本次 6 个问题的压缩结论

### Q1. R4 需要哪些输入？

需要：

- `pseudo_label`: `(B,H,W)`
- `valid_mask`: R2 class-balanced 后 `(B,H,W)` bool
- `confidence`: `(B,H,W)`，用于继承 R2
- `feature map`: 推荐 `feat_t_de`，预期 `(B,256,Hf,Wf)`
- `class prototype`: 预期 `(num_classes,256)`
- `ignore_index`: int，默认 `255`
- `num_classes`: int，当前 `13`

### Q2. 当前插入点是否可用？

- `pseudo_label/confidence/features/valid_mask/ignore_index/num_classes` 在 refiner call 处大体可用。
- `class prototype` **不直接可用**：训练函数有 `proto_bank`，但未传给 refiner；R4 专用 `prototype_bank.py` 是 placeholder。

### Q3. R4 应该使用哪个 feature？

- 应使用 `feat_t_de`。
- 预期 shape 为 `(B,256,Hf,Wf)`。
- 不能假设与 `(B,H,W)` label/mask 同分辨率。
- 需要 interpolate 或等价对齐机制。
- 使用 cosine 时需要 normalize feature 和 prototype。

### Q4. prototype 从哪里来？

- 现有训练链路已有 FPA `proto_bank`，支持 `snapshot()`、`update()`、`mse_pull_loss()`。
- 但 `adaptation/pseudo_label/prototype_bank.py` 仍是 placeholder，`PROTOTYPE_BANK_AVAILABLE=False`。
- 当前不能确认 prototype 内部 shape/API。
- source/global prototype likely exists；target/online prototype 状态不清楚。
- R4 需要新增或暴露 prototype 读取逻辑，但应优先复用现有 FPA proto_bank，而不是重复造一套冲突 prototype。

### Q5. R4 是否能继承 R2 valid_mask？

能，且应该严格走：

```text
R2 class-balanced valid_mask
→ prototype similarity gate
→ refined valid_mask
```

只修改 `valid_mask`，不修改 `pseudo_label`，不改 loss。

### Q6. 最大工程风险是什么？

最大风险是：**prototype 来源/API/shape 未确认 + `feat_t_de` 与 label/mask 分辨率可能不一致**。

其次是：prototype 缺失/全零、rare class prototype 不稳定、统一 threshold 误杀 rare class、R2 后再过滤使 unsup loss 太弱。

---

## 8. 最终结论

**PARTIAL：输入大体可用，但 prototype 来源或 shape 需要进一步检查。**

允许进入下一步的条件应是：R4-2 先明确现有 `proto_bank` 的可读 prototype API、prototype shape、valid flag，以及 `feat_t_de` 与 pseudo label 的实际 shape 对齐方式。当前不应直接实现 full R4，更不应训练。
