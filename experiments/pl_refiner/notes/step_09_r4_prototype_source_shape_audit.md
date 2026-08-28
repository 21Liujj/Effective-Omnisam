# step_09_r4_prototype_source_shape_audit.md
# Phase 4 / Task 4-2 — prototype 来源/API 与 feature shape 细查

**Run Level**: L0 scan-only / code inventory  
**日期**: 2026-06-24  
**执行人**: Claude Code  
**是否改代码**: No  
**是否训练**: No  
**是否跑脚本**: No  
**是否实现 R4**: No  
**是否实现 similarity diagnosis**: No  
**是否进入 R5/R6**: No  
**是否 full-run**: No  
**是否看 mIoU**: No

---

## 1. 本次任务说明

Task 4-1 的结论是 `PARTIAL`：

- `pseudo_label / valid_mask / confidence / feat_t_de / ignore_index / num_classes` 在 `PseudoLabelRefiner` 插入点大体可用。
- `feat_t_de` 应作为 R4 feature，预期 shape 为 `(B,256,H',W')`。
- 不能确认 `feat_t_de` 与 `pseudo_label / valid_mask` 是否同分辨率。
- 训练链路已有 FPA `proto_bank`，但当前 refiner call 未传入，且 `adaptation/pseudo_label/prototype_bank.py` 是 placeholder。

本次唯一目标：查清 R4 prototype 来源/API 与 feature-label/mask shape 对齐问题，并给出后续最稳实现路线。

---

## 2. 读取 / 搜索文件列表

### 2.1 直接读取文件

| 文件 | 读取目的 | 结果 |
|---|---|---|
| `adaptation/strategies.py` | 检查 `proto_bank` 传入、`snapshot/update/mse_pull_loss`、`feat_t_de` 来源、refiner 参数 | 已读取 |
| `adaptation/methods.py` | 检查 FPA prototype bank 类定义/API/shape | 已读取 |
| `adaptation/trainer.py` | 检查 `proto_bank` 创建位置与传入 strategy | 已读取 |
| `adaptation/engine.py` | 检查 `adaptation_kwargs`，尤其 `feature_dim` | 已读取片段 |
| `model/model.py` | 检查 `feat_t_de` 从 model forward 的来源 | 已读取 |
| `model/modules/decoder_sam.py` | 检查 `SegFormerHead_adaption` 输出 feature shape 逻辑 | 已读取 |
| `adaptation/pseudo_label/prototype_bank.py` | 检查 R4/R5 prototype placeholder | 已读取 |
| `experiments/pl_refiner/configs/R4_proto_gate.yaml` | 检查 R4 prototype 配置 | 已读取 |
| `experiments/pl_refiner/notes/step_00_pl_pipeline_inventory.md` | 检查既有 PL/refiner inventory 中的 feature shape 记录 | 已搜索/读取相关行 |
| `experiments/pl_refiner/notes/step_08_r4_proto_gate_input_inventory.md` | Task 4-1 上下文 | 已搜索/读取相关行 |

### 2.2 搜索关键词

按要求搜索了：

- `class Proto`
- `Prototype`
- `proto_bank`
- `snapshot`
- `mse_pull_loss`
- `update(`
- `prototypes`
- `class_proto`
- `source_proto`
- `target_proto`
- `PROTOTYPE_BANK_AVAILABLE`
- `proto / prototype / fpa / mse_pull / proto_bank` configs 相关关键词

主要命中：

- `adaptation/methods.py`
- `adaptation/strategies.py`
- `adaptation/trainer.py`
- `adaptation/engine.py`
- `adaptation/pseudo_label/prototype_bank.py`
- `model/model.py`
- `model/modules/decoder_sam.py`
- `experiments/pl_refiner/configs/R4_proto_gate.yaml`
- `experiments/pl_refiner/configs/R5_rbpc.yaml`
- `experiments/pl_refiner/configs/R6_memory_plq.yaml`
- `experiments/pl_refiner/configs/R7_full.yaml`
- `experiments/pl_refiner/notes/*`
- 文本原型相关文件 `model/text_align.py / model/text_semantic.py`（与 R4 visual prototype gate 不是同一条 prototype 线）

---

## 3. 已有 FPA proto_bank 可用性结论

### 3.1 proto_bank 创建与传入

`adaptation/trainer.py`：

```python
from adaptation.methods import PrototypicalAdaptation, PrototypicalAdaptationEMA
...
self.proto_bank = PrototypicalAdaptationEMA(**adaptation_kwargs)
```

`adaptation/engine.py`：

```python
self.trainer = Trainer(
    ...
    adaptation_kwargs=dict(num_classes=self.num_classes, feature_dim=256),
    ...
)
```

因此当前训练主链路使用的是：

```text
PrototypicalAdaptationEMA(num_classes=self.num_classes, feature_dim=256)
```

`adaptation/trainer.py` 之后将其传入 strategy：

```python
global_step = self.mem_strategy.forward(
    ...
    self.proto_bank,
    self.alpha,
    ...
)
```

### 3.2 PrototypicalAdaptationEMA API / shape

`adaptation/methods.py`：

```python
class PrototypicalAdaptationEMA:
    def __init__(self, num_classes, feature_dim, momentum=0.99, device="cuda", ignore_index=255, conf_thres=0.85):
        self.num_classes = num_classes
        self.feat_dim = feature_dim
        self.momentum = momentum
        self.ignore_index = ignore_index
        self.conf_thres = conf_thres

        self.prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.counts     = torch.zeros(num_classes, device=device)
```

直接可见：

```text
prototypes shape = (num_classes, feature_dim) = expected (13, 256)
counts shape     = (num_classes,)
```

### 3.3 snapshot / delta

`adaptation/methods.py`：

```python
def snapshot(self):
    self._proto_prev = self.prototypes.detach().clone()

def delta_mean(self):
    if not hasattr(self, "_proto_prev"):
        return 0.0
    return (self.prototypes - self._proto_prev).abs().mean().item()
```

结论：

- `snapshot()` **返回 None**。
- 它只写入内部字段 `_proto_prev`。
- 当前 `strategies.py` 中没有使用 `snapshot()` 返回值。
- `delta_mean()` 用 `_proto_prev` 和当前 `self.prototypes` 计算平均变化量。

### 3.4 update timing

`adaptation/strategies.py` 当前顺序是：

```text
1. model target forward -> pred_t, feat_t, feat_t_de
2. compute conf_t / valid_mask / mask_t
3. PseudoLabelRefiner.refine(... features=feat_t_de ...)
4. proto_bank.snapshot()
5. proto_bank.update(feat_s_de_ori, frame_lbl_s)
6. later: loss_proto = proto_bank.mse_pull_loss(feat_t_de, frame_pl_t_f)
```

也就是说：

- 当前 refiner 插入点发生在 `proto_bank.snapshot()` 与 `proto_bank.update()` 之前。
- FPA `proto_bank` 在当前 step 的 source GT prototype update 之后，才用于 target `mse_pull_loss()`。
- 如果 R4 直接用 FPA prototype，需要明确使用：
  - update 前旧 prototype；还是
  - 当前 source update 后新 prototype；还是
  - `_proto_prev` snapshot。

### 3.5 update() 对 shape 的处理

`PrototypicalAdaptationEMA.update(feats, labels, conf=None)`：

```python
if feats.dim() == 4:
    B, C, H, W = feats.shape
    feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)
    if labels.shape[-2:] != (H, W):
        labels = F.interpolate(labels.float(), size=(H, W), mode='nearest').long()
    labels = labels.view(-1)
...
feats = F.normalize(feats[mask], dim=1)
...
batch_proto = sums[valid] / cnts[valid].unsqueeze(1)
self.prototypes[valid] = m * self.prototypes[valid] + (1 - m) * batch_proto
self.counts[valid] += cnts[valid]
```

关键点：

- FPA update 选择**把 labels nearest downsample / resize 到 feature resolution**。
- update 时对 features 做 `F.normalize(feats[mask], dim=1)`。
- prototype 是 EMA 更新。
- `counts` 存在，可作为 `valid flag/count` 来源：`counts > 0`。

### 3.6 mse_pull_loss() 对 shape 的处理

`PrototypicalAdaptationEMA.mse_pull_loss(feats, labels)`：

```python
if feats.dim() == 4:
    B, C, H, W = feats.shape
    feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
    if labels.dim() == 3:
        labels = labels.unsqueeze(1)
    if labels.shape[-2:] != (H, W):
        labels = F.interpolate(labels.float(), size=(H, W), mode='nearest').long()
    labels = labels.view(-1)
...
batch_proto[valid] = sums[valid] / cnts[valid].unsqueeze(1)
return F.mse_loss(batch_proto[valid].float(), self.prototypes[valid].float())
```

关键点：

- `mse_pull_loss()` 输入是 `feats` 与 `labels`。
- 如果分辨率不一致，仍然是 resize labels 到 feature resolution。
- 它返回 scalar loss。
- 它不是 pixel-level similarity map API。
- 它不会返回 per-class prototype，也不会返回 per-pixel distance。

### 3.7 FPA proto_bank 是否可直接用于 R4？

| item | finding | evidence | risk |
| --------------------------- | ------- | -------- | ---- |
| where created | `Trainer.__init__()` 中创建 `self.proto_bank = PrototypicalAdaptationEMA(**adaptation_kwargs)` | `adaptation/trainer.py` line around 25；`adaptation/engine.py` passes `adaptation_kwargs=dict(num_classes=self.num_classes, feature_dim=256)` | 创建清楚，风险低 |
| update timing | 在 `PseudoLabelRefiner.refine()` 之后，`loss_proto` 之前：`snapshot()` → `update(source feature/source label)` → later `mse_pull_loss(target feature/target PL)` | `adaptation/strategies.py` 中 refiner call 后紧接 `proto_bank.snapshot(); proto_bank.update(feat_s_de_ori, frame_lbl_s)`；后面 `loss_proto = proto_bank.mse_pull_loss(...)` | R4 若要用 FPA prototype，必须决定用 update 前还是 update 后 prototype；否则有时序歧义 |
| snapshot return type | `None`；内部写 `_proto_prev = self.prototypes.detach().clone()` | `adaptation/methods.py::snapshot()` | 不能直接写 `protos = proto_bank.snapshot()`；需要 adapter 明确读取 `_proto_prev` 或 `prototypes` |
| prototype shape | `(num_classes, feature_dim)`，当前应为 `(13,256)` | `self.prototypes = torch.zeros(num_classes, feature_dim, device=device)`；engine `feature_dim=256` | shape 适合 R4，但需要确认与实际 `feat_t_de.shape[1]` 一致，代码层面预期一致 |
| class-wise available? | YES，按 class row 存储 | `self.prototypes[valid] = ...`；`sums.index_add_(0, labels, feats)` | 可按 pseudo label class index 查询 |
| valid flag/count available? | YES，`self.counts` 可作为 class prototype valid count；`counts > 0` 是 valid flag | `self.counts = torch.zeros(num_classes)`；`self.counts[valid] += cnts[valid]` | counts 是累计像素数，不是 EMA 有效性置信度；zero prototype 仍需 norm guard |
| can be passed to refiner? | PARTIAL：对象当前在 `strategies.py` 可见，但 refiner call 没有 `proto_bank` / `class_prototypes` 参数 | 当前 `_refiner.refine(... features=feat_t_de, valid_mask=..., ...)` 没传 `proto_bank` | 需要改 refiner API 或在 strategies 中先抽取 prototypes 再传入；不是“直接可用” |

### 3.8 FPA 可用性结论

**结论：PARTIAL。**

已有 FPA `proto_bank` **可能可用**，因为：

- `prototypes` shape 清楚：`(C,256)`。
- `counts` 可作为 valid flag/count。
- prototype 是 class-wise 存储。
- 现有 update/mse_pull_loss 已处理 feature-label 分辨率不一致。

但它**不能直接用于 R4**，因为：

1. `PseudoLabelRefiner.refine()` 当前未接收 `proto_bank` 或 `class_prototypes`。
2. `snapshot()` 返回 `None`，不能直接作为 prototype tensor。
3. 若直接读 `proto_bank.prototypes`，需要明确时序：refiner 之前读旧 prototype，还是把 source update 提前后读新 prototype。
4. `mse_pull_loss()` 只是 scalar loss API，不是 R4 所需的 pixel-wise similarity gate API。
5. 直接在 R4 里依赖 `proto_bank.prototypes/_proto_prev/counts` 会形成强耦合，最好写 wrapper/adapter。

因此本项不是 `YES_FPA`，而是：

```text
PARTIAL：已有 proto_bank 可能可用，但需要 wrapper / adapter。
```

---

## 4. current-batch prototype gate 可行性结论

### 4.1 方案定义

不依赖已有 FPA `proto_bank`，仅使用当前 batch 的：

- `feat_t_de`
- `pseudo_label`
- R2 refined `valid_mask`

在线计算当前 batch class prototype：

```text
class_proto[k] = mean(feature_i)
where pseudo_label_i == k and valid_mask_i == 1
```

然后对每个 valid pixel / feature location 计算：

```text
cosine(feature_i, class_proto[pseudo_label_i])
```

最后：

```text
final_valid_mask = R2_valid_mask & proto_keep
```

R4 仍只做 filtering，不改 label，不改 loss。

### 4.2 是否需要 detach

**需要 detach。**

理由：

- 当前 refiner 位于 `torch.no_grad()` block 内，语义上就是在线过滤/诊断，不应改变梯度路径。
- prototype gate 是 mask 决策，不应通过 prototype mean 产生可导路径。
- 即使后续实现位置不完全在 `no_grad()` 中，也应显式使用 `features.detach()`。

建议：

```python
feat = features.detach()
```

然后 normalize 和 prototype mean 都在 no-grad / detached tensor 上做。

### 4.3 rare class 像素太少怎么办

current-batch prototype 最大风险就是 rare/extreme rare class 样本过少。

处理原则：

- **少于 min pixels 的 class 不做 prototype gate**。
- 对该 class 的像素 fallback：保留 R2 valid mask，不额外过滤。
- 不能因为 class prototype 缺失就把该类全部置 False。

建议 diagnostics：

- `proto_valid_class_count`
- `proto_missing_class_ids`
- `proto_pixel_count_by_class`
- `proto_reject_ratio_by_class`
- `proto_rare_reject_ratio`
- `proto_extreme_rare_reject_ratio`

### 4.4 min_pixels_per_class 应该设多少

建议 R4 v1 使用：

```text
min_proto_pixels = 10  # feature-resolution locations
```

理由：

- R4 config 当前已有 `class_balance.min_pixels_per_class: 10`，可与 R4 v1 原型计算保持一致的保守尺度。
- `feat_t_de` 分辨率约为 `H/4,W/4`，feature-resolution 的 10 个点约对应 label-resolution 的约 160 个像素，已经比单点/少量点稳定。
- 对 traffic light / traffic sign 等 extreme rare 类，如果 feature-resolution 少于 10 个点，不应强行 gate，应 fallback 到 R2。

备选更保守策略：

```text
common min_proto_pixels = 10
rare/extreme rare min_proto_pixels = 5 或 fallback keep R2
```

但 R4 v1 为了避免自由设计过多，建议先统一 `10`，并且缺失 class fallback keep R2。

### 4.5 没有 prototype 的 class 怎么处理

必须 fallback：

```text
if class k prototype missing / count < min_proto_pixels / zero-norm / NaN:
    proto_keep[pseudo_label == k] = True inside R2_valid_mask
```

即：

```text
final_valid_mask for missing class = R2_valid_mask for that class
```

不能输出全 False。

### 4.6 是否比依赖 FPA proto_bank 更稳

工程上：**current-batch 更稳。**

原因：

- 不需要修改 strategy/refiner 间额外传 `proto_bank` 对象。
- 不依赖 FPA `snapshot()` 时序。
- 不依赖 `proto_bank.prototypes/_proto_prev/counts` 这些内部字段。
- 不受 source update 前/后 prototype 的语义歧义影响。
- 只需要 `features/pseudo_label/valid_mask/num_classes/ignore_index`，这些 Task 4-1 已确认在 refiner 插入点可用。

算法上：**current-batch prototype 不一定比 FPA source prototype 更强**，因为它来自 target pseudo-label 本身，可能有 confirmation bias。但 R4 v1 的目标是安全过滤和诊断，不是最终最强版本，因此更适合作为 first implementation / diagnosis。

### 4.7 是否适合作为 R4 v1

**适合。**

推荐 current-batch prototype gate 作为 R4 v1，尤其先做 R4-3a similarity diagnosis：

```text
R2 valid_mask
→ current-batch class prototype from feat_t_de + R2 valid pixels
→ cosine similarity map
→ diagnostics first
→ later threshold gate
```

---

## 5. feature / label / mask shape 对齐方案

### 5.1 shape 来源确认

`model/model.py`：

```python
x, feat_de = self.decoder(feat)
l_vis = F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)
...
return pred, feat, feat_de
```

`model/modules/decoder_sam.py::SegFormerHead_adaption.forward()`：

```python
_c3 = ... reshape(... c3.shape[2], c3.shape[3])
_c3 = F.interpolate(_c3, size=c1.size()[2:], ...)
_c2 = ... reshape(... c2.shape[2], c2.shape[3])
_c2 = F.interpolate(_c2, size=c1.size()[2:], ...)
_c1 = ... reshape(... c1.shape[2], c1.shape[3])
_c = self.linear_fuse(torch.cat([_c3, _c2, _c1], dim=1))
_c = self.out_proj(_c)
x = self.linear_pred(self.dropout(_c))
return x, _c
```

因此：

- `feat_t_de = _c`
- channel dim = decoder embedding dim = `256`
- spatial resolution = `c1` resolution
- `pred_t` 被 interpolate 到 input patch resolution `(height,width)`，但 `feat_t_de` 不会被 interpolate 到 input resolution

既有 note 也记录：

```text
feat_t_de shape (B,256,H',W')，H'≈H/4
```

### 5.2 shape 对齐表

| tensor | expected shape | resolution | risk | alignment strategy |
| ------ | -------------- | ---------- | ---- | ------------------ |
| `feat_t_de` | `(B,256,Hf,Wf)` | decoder/c1 resolution，通常约 `H/4,W/4` | 与 label/mask/confidence 不同分辨率；直接按 `(B,H,W)` index 会 shape mismatch | 保持 feature resolution 计算 prototype/similarity；`features.detach()` + `F.normalize(dim=1)` |
| `pseudo_label` | `(B,H,W)` | CE/loss label resolution，等于 input patch resolution | 若直接用于 feature-resolution feature，会 mismatch | 对 prototype/similarity 计算使用 nearest downsample 到 `(Hf,Wf)`；最终输出仍回到 `(H,W)` |
| `valid_mask` | `(B,H,W)` bool | CE/loss label resolution | R2 mask 是最终 loss path 使用的 mask，不能永久降到 low-res | 对 prototype 计算 downsample 到 `(Hf,Wf)`；最终 gate map upsample 回 `(H,W)` 后与原 R2 valid_mask 做 AND |
| `confidence` | `(B,H,W)` float | pred/loss resolution | R2 需要 high-res confidence；R4 current-batch proto 不一定需要 confidence | R2 仍用 high-res；若 R4 diagnostics 需要按 proto class 统计，可 downsample confidence 到 `(Hf,Wf)` 或只在 high-res final mask 上统计 |
| `similarity_map` | recommended low-res `(B,Hf,Wf)` then high-res `(B,H,W)` | low-res for compute, high-res for final filtering | 若只保留 low-res mask，无法直接对齐 CE loss；若 upsample bool hard mask 可能块状 | 先在 low-res 计算 cosine similarity；将 similarity map bilinear interpolate 到 `(H,W)`，再 threshold；final `valid_mask=(R2_valid_mask & proto_keep_highres)` |

### 5.3 upsample feature 还是 downsample label/mask？

推荐：**downsample label/mask 到 feature resolution 计算 prototype/similarity，再 upsample similarity_map 回 label resolution。**

理由：

1. 与现有 FPA `update()` / `mse_pull_loss()` 行为一致：它们都是把 labels nearest resize 到 feature resolution。
2. 避免把 `(B,256,Hf,Wf)` feature upsample 到 `(B,256,H,W)` 导致显存/计算开销暴涨。
3. R4 只产生 mask，不需要改变 CE logits 或 label，因此低分辨率计算 similarity 是可接受的。
4. 最终 `valid_mask` 必须回到 `(B,H,W)`，与原 loss path 对齐。

### 5.4 对 CE loss 路径是否有影响

不会改变 CE loss 公式。

R4 只改变：

```python
frame_pl_t_f = frame_pl_t.clone()
frame_pl_t_f[~mask_t] = ignore_index
loss_unsup_raw = criteria['t'](pred_t, frame_pl_t_f)
```

其中 `mask_t` 必须是 `(B,H,W)`。

因此 R4 的 final output 必须是：

```text
final_valid_mask: (B,H,W) bool
```

只要最终 mask 对齐 `(B,H,W)`，CE loss 路径不需要修改。

### 5.5 对 R4 filtering 是否有影响

有两个影响：

1. 小目标 / rare class 可能在 downsample 到 feature resolution 时消失。
2. upsample similarity 回 high-res 后的边界会较粗。

处理策略：

- missing / too few prototype class fallback keep R2。
- diagnostics 必须记录 per-class proto count 和 reject ratio。
- R4 v1 不做 correction，不追求边界精修，只做 conservative filtering。

---

## 6. 推荐 R4 v1 实现路线

候选：

- A. 使用已有 FPA proto_bank snapshot 作为 prototype 来源。
- B. 不依赖 FPA proto_bank，先使用 current-batch prototype 作为 R4 v1。

### 推荐：B. current-batch prototype gate

推荐理由：

1. **工程最稳**：不依赖 FPA `proto_bank` 的时序、snapshot 返回值、内部字段读取，也不用先设计 adapter。
2. **输入已可用**：`PseudoLabelRefiner.refine()` 当前已经拿到 `pseudo_label / confidence / features=feat_t_de / valid_mask / ignore_index / num_classes`。
3. **不阻塞于 FPA API**：FPA prototype 虽然有 `.prototypes` 和 `.counts`，但当前 refiner 未接入，直接用需要改 signature 或 wrapper。
4. **符合 R4 目标**：只做 filtering / valid_mask refinement，不改 label，不改 loss。
5. **适合先做 diagnosis**：可以先输出 similarity 分布、per-class proto count、reject ratio，再决定 threshold，避免直接误杀 rare class。

### 不推荐 A 作为 R4 v1 的原因

FPA `proto_bank` 已经比 Task 4-1 更清楚，但仍不建议作为 R4 v1 直接依赖：

- `snapshot()` 返回 None。
- `refiner` 当前没有 `proto_bank` 参数。
- refiner 发生在 FPA source update 前，时序需要重排或明确读取旧 prototype。
- 直接读 `proto_bank.prototypes/counts` 是内部耦合，应该先写 adapter。
- R4 的 pixel-wise similarity API 仍需要新增逻辑；`mse_pull_loss()` 不能直接复用。

因此：

```text
FPA proto_bank: PARTIAL, later can do adapter.
R4 v1: 推荐 current-batch prototype gate.
```

---

## 7. 风险分析

| 风险 | 严重度 | 分析 | R4 v1 建议 |
|---|---:|---|---|
| current-batch prototype confirmation bias | 中 | prototype 来自 target pseudo-label 本身，错误 pseudo-label 会污染 prototype | R4-3 先做 diagnosis，不直接训练；阈值保守；只过滤极低相似度 outlier |
| rare/extreme rare prototype 像素太少 | 高 | traffic light/sign 等类在 feature resolution 可能只有很少点甚至消失 | `min_proto_pixels=10`；不足则 fallback keep R2；记录 missing class |
| downsample label/mask 丢小目标 | 高 | nearest downsample 到 `Hf,Wf` 可能让小目标消失 | 缺失 class 不 gate；必要时后续比较 feature-upsample 方案，但 v1 不做 |
| threshold 误杀 rare class | 高 | 单一 threshold 对 common 和 rare 不一定都合理 | R4-3a 先做 similarity diagnosis；输出 per-class reject ratio；v1 threshold conservative 或只诊断 |
| R2 后再 gate 导致 unsup loss 太弱 | 高 | R2 已降低 valid ratio，R4 再 AND 会进一步减少有效像素 | 加 `min_valid_ratio/min_valid_pixels` fallback；若 final valid 过低则 fallback R2 |
| FPA proto_bank 时序歧义 | 中 | refiner 在 `snapshot/update` 前，FPA loss 在 update 后 | R4 v1 不依赖 FPA；后续 R4-3b 再设计 adapter |
| FPA direct internal coupling | 中 | 直接读 `.prototypes/.counts/_proto_prev` 会绑定内部实现 | 若走 FPA，必须 wrapper/adapter，不直接散落读取 |
| similarity map high/low-res 边界粗糙 | 低-中 | low-res sim 上采样会导致块状 gate | R4 是 filtering，不做 boundary correction；保持 conservative |

---

## 8. 是否允许进入 R4-3

**允许进入 R4-3，但建议只进入 R4-3a。**

推荐下一步：

```text
R4-3a：实现 current-batch prototype similarity diagnosis
```

理由：

- R4 v1 推荐 current-batch route。
- 在真正 filtering 前，必须先看 similarity 分布、per-class proto count、missing class、潜在 reject ratio。
- R3 已出现 PLQ raw score 饱和，不应直接用新分数进入 loss/filtering；R4 也应先诊断。

不建议立即做：

- `R4-3b：实现 FPA proto_bank adapter`：可以后续做，但不是 v1 最短稳路径。
- `R4-3c：继续 scan 其他 prototype 文件`：当前关键 API/shape 已足够支持 v1 方案，不需要继续无界扫描。

---

## 9. 下一步建议，但不执行

### 推荐 R4-3a 内容

只建议，不执行：

1. 在 `proto_gate` 之前或诊断模式中，先实现 `current_batch_proto_similarity_diagnosis`。
2. 输入：
   - `features=feat_t_de.detach()`
   - `pseudo_label`
   - R2 `valid_mask`
   - `ignore_index`
   - `num_classes`
3. 对齐：
   - downsample `pseudo_label/valid_mask` 到 feature resolution。
   - normalize features。
   - 按 class 计算 current-batch prototype。
   - 计算 low-res similarity。
   - upsample similarity map 到 label resolution，仅用于 diagnostics / future threshold。
4. 输出 diagnostics：
   - `proto_valid_class_count`
   - `proto_missing_class_ids`
   - `proto_pixel_count_by_class`
   - `proto_sim_mean/std/min/max`
   - `proto_sim_mean_by_class`
   - candidate thresholds 下的 `would_reject_ratio_by_class`
   - `would_final_valid_ratio`
5. 不改 loss，不做训练，不看 mIoU。

---

## 10. 本次必须回答问题汇总

### Q1. 已有 FPA proto_bank 是否可直接用于 R4？

**PARTIAL：已有 proto_bank 可能可用，但需要 wrapper / adapter。**

不是 `YES`，因为当前 refiner 未接收 proto_bank，`snapshot()` 返回 None，且 `mse_pull_loss()` 是 scalar loss API，不是 pixel-wise gate API。

### Q2. 如果不依赖已有 proto_bank，是否可以做 current-batch prototype gate？

**可以，且推荐作为 R4 v1。**

关键约束：

- `features.detach()`。
- feature resolution 上计算 prototype/similarity。
- class count `< min_proto_pixels` 时 fallback keep R2。
- prototype missing / zero norm / NaN 时 fallback keep R2。
- 先做 diagnosis，再决定 threshold。

### Q3. feat_t_de shape 和 pseudo_label shape 如何对齐？

推荐：

```text
feat_t_de 保持低分辨率 (B,256,Hf,Wf)
pseudo_label / valid_mask nearest downsample 到 (Hf,Wf) 用于 prototype/similarity
similarity_map bilinear upsample 到 (H,W)
final_valid_mask = R2_valid_mask(B,H,W) & proto_keep_highres(B,H,W)
```

最终 valid_mask 必须和原 CE loss 路径对齐，即 `(B,H,W)`。

### Q4. R4 v1 最稳实现路线是什么？

选择：

```text
B. 不依赖 FPA proto_bank，先使用 current-batch prototype 作为 R4 v1。
```

### Q5. 后续 R4-3 应该做什么？

建议：

```text
R4-3a：实现 current-batch prototype similarity diagnosis。
```

不要先做 FPA adapter，不要继续无界 scan。

---

## 11. 最终结论

**YES_BATCH：建议先做 current-batch prototype gate。**

摘要：

1. **最终结论**：`YES_BATCH`
2. **推荐 prototype 来源**：current-batch prototype；FPA proto_bank 暂不作为 R4 v1 直接来源
3. **feature 和 label/mask 对齐**：feature resolution 计算，label/mask nearest downsample；similarity map upsample 回 label resolution；最终 mask `(B,H,W)` 对齐 CE loss
4. **rare class prototype 风险处理**：`min_proto_pixels=10` feature locations；不足 / missing / zero-norm / NaN 时 fallback keep R2，不额外过滤
5. **是否允许进入 R4-3**：允许，但建议只进入 `R4-3a current-batch prototype similarity diagnosis`
