# Route 3: Class-Aware Token Protection

> 针对性保护受损的 rare/thin 类别 tokens
> 创建时间: 2026-08-25 16:20

---

## 🎯 核心动机

### 观察到的问题

**MTC 后受损的类别**（与 baseline 对比）:

| 类别 | Baseline mIoU | MTC mIoU | 下降 | 特点 |
|------|--------------|----------|------|------|
| traffic light | 65.2 | 63.55 | **-1.65** | 小物体、远处 |
| fence | 58.3 | 57.53 | **-0.77** | 细长、边界 |
| pole | 62.8 | 62.52 | **-0.28** | 细长、垂直 |
| traffic sign | 70.5 | 70.21 | **-0.29** | 小物体 |
| terrain | 68.9 | 68.51 | **-0.39** | 纹理细节 |

**共同特征**:
1. 小物体或细长物体
2. 在 feature norm Top-K 中容易被淘汰
3. 对全景街景理解重要（交通安全相关）

---

## 💡 核心思路

### 问题分析

当前 `attn_topk` 选择策略：
```python
# 纯 feature norm 排序
scores = torch.norm(tokens, dim=-1)  # [1024]
topk_indices = torch.topk(scores, k=quota).indices
```

**问题**：
- Feature norm 大的 token 不一定是重要类别
- 大面积常见类（road, building, sky）feature 强 → 容易被选中
- 小面积稀有类（traffic light, pole）feature 弱 → 容易被淘汰

### 解决方案

**Class-Aware Selection**: 在固定 quota 下，显式保护 rare class tokens

```python
class_aware_selection:
  Step 1: 识别 rare class tokens (用 pred logits)
  Step 2: 保留 rare_class_reserve_ratio 的 quota 给 rare class
  Step 3: 剩余 quota 用 feature norm Top-K 填充
  Step 4: 保证总 quota 不变
```

---

## 🔧 技术设计

### 输入信号

**需要从上游传递**:
1. `memory_tokens`: [N, C] 当前 FoV 的 memory tokens
2. `pred_logits`: [N, num_classes] 每个 token 的类别预测 logits
3. `quota`: int 这个 FoV 允许的总 token 数

**pred_logits 来源**:
- 上游已有 prediction mask/logits
- 需要在 `MemoryBank.forward` 中传递到 `SparseMemoryRetriever`

### 算法流程

```python
def class_aware_selection(
    tokens: Tensor,           # [N, C]
    pred_logits: Tensor,      # [N, num_classes]
    quota: int,
    rare_class_ids: List[int],
    rare_reserve_ratio: float = 0.15
):
    """
    Class-aware token selection
    
    Args:
        rare_class_ids: 受损类别的 class_id
            - traffic_light: 6
            - fence: 13
            - pole: 17
            - traffic_sign: 7
            - terrain: 22
    """
    N = tokens.size(0)
    
    # Step 1: 识别 rare class tokens
    pred_classes = pred_logits.argmax(dim=-1)  # [N]
    rare_mask = torch.zeros(N, dtype=torch.bool)
    for class_id in rare_class_ids:
        rare_mask |= (pred_classes == class_id)
    
    rare_indices = rare_mask.nonzero(as_tuple=True)[0]  # rare class 的 indices
    common_indices = (~rare_mask).nonzero(as_tuple=True)[0]
    
    # Step 2: 分配 quota
    rare_quota = int(quota * rare_reserve_ratio)
    common_quota = quota - rare_quota
    
    # Step 3: Rare class 选择
    # 选项A: 如果 rare class 数量 < rare_quota，全选
    # 选项B: 如果 rare class 数量 > rare_quota，按 confidence 排序
    if len(rare_indices) <= rare_quota:
        selected_rare = rare_indices
        # 把省下的 quota 还给 common
        common_quota += (rare_quota - len(rare_indices))
    else:
        # 按 rare class 的 confidence 排序
        rare_confidence = pred_logits[rare_indices].max(dim=-1).values
        topk_rare = torch.topk(rare_confidence, k=rare_quota)
        selected_rare = rare_indices[topk_rare.indices]
    
    # Step 4: Common class 选择 (feature norm Top-K)
    common_scores = torch.norm(tokens[common_indices], dim=-1)
    topk_common = torch.topk(common_scores, k=min(common_quota, len(common_indices)))
    selected_common = common_indices[topk_common.indices]
    
    # Step 5: 合并
    selected_indices = torch.cat([selected_rare, selected_common])
    
    return selected_indices  # 保证 len <= quota
```

### 超参数设计

```yaml
class_aware_config:
  mode: "class_aware"
  rare_class_ids: [6, 13, 17, 7, 22]  # traffic_light, fence, pole, traffic_sign, terrain
  rare_reserve_ratio: 0.15             # 15% quota 保护 rare class
  rare_selection_strategy: "confidence" # or "all"
```

---

## 🧪 实验设计

### Phase 3A: 单独验证 (n500 快速)

测试 3 个 `rare_reserve_ratio`:

```yaml
route3_rare010:
  rare_reserve_ratio: 0.10  # 保守
  
route3_rare015:
  rare_reserve_ratio: 0.15  # 平衡 (推荐)
  
route3_rare020:
  rare_reserve_ratio: 0.20  # 激进
```

**时间**: 3配置 × n500 × seed1 = 1.2h  
**目标**: 找出最优 reserve_ratio

**预期结果**:
- rare class mIoU 提升 (traffic_light, fence, pole)
- 总 mIoU 持平或略升 (~60.0-60.2)
- BFScore 提升 (rare class 往往在边界)

### Phase 3B: 与 Boundary 组合 (n500)

```yaml
route3_boundary_class_hybrid:
  boundary_reserve: 0.10     # 降低 boundary 比例
  rare_class_reserve: 0.10   # 新增 class 保护
  feature_topk: 0.80         # 剩余 Top-K
  
  # 可能的重叠处理
  overlap_strategy: "union"  # boundary tokens + rare class tokens 去重
```

**时间**: 1配置 × n500 × seed1 = 0.4h

**重叠分析**:
- rare class (traffic light, fence, pole) 往往在边界
- Boundary 保护边界区域，Class-Aware 保护类别
- 重叠部分获得双重保护 → 可能更鲁棒

### Phase 3C: 确认 (n2000, 如果有效)

```yaml
route3_best_n2000:
  config: 最优 rare_reserve_ratio
  seeds: [1, 2, 3, 42, 100]
  subset: n2000
```

**时间**: 5 × 1.5h = 7.5h

---

## 📁 配置文件

### route3_rare010.yaml
```yaml
plugins:
  pseudo_conf_mask:
    enabled: true
    threshold: 0.85
    min_valid_ratio: 0.01
    min_valid_pixels: 0
    ignore_index: 255
    fallback: skip
  efficient_tam_memory:
    enabled: true
    mode: pool
    compress_ratio: 0.5
    pool_stride: 2
    apply_in_pl: false
    apply_in_train: true
  efficient_sam2_sparse:
    enabled: true
    mode: class_aware
    keep_ratio: 0.5
    min_keep_tokens: 64
    saliency_source: feature_norm
    cache_saliency: true
    fallback_full_when_uncertain: true
    apply_in_pl: false
    apply_in_train: true
    # Class-Aware specific
    rare_class_ids: [6, 13, 17, 7, 22]
    rare_reserve_ratio: 0.10
    rare_selection_strategy: confidence
```

### route3_rare015.yaml
```yaml
# Same as above, but:
rare_reserve_ratio: 0.15
```

### route3_rare020.yaml
```yaml
# Same as above, but:
rare_reserve_ratio: 0.20
```

---

## 🔗 代码修改点

### 1. `model/modules/sparse_memory_retriever.py`

新增 `mode=class_aware`:

```python
class SparseMemoryRetriever(nn.Module):
    def _resolve_score(self, mode, tokens, pred_logits=None, ...):
        if mode == "class_aware":
            assert pred_logits is not None, "class_aware mode requires pred_logits"
            return self._class_aware_score(tokens, pred_logits, ...)
        elif mode == "attn_topk":
            # existing
        elif mode == "boundary_guarded":
            # Route 1
        ...
    
    def _class_aware_score(self, tokens, pred_logits, quota, rare_class_ids, rare_reserve_ratio):
        # 实现上面的算法
        ...
```

### 2. `model/memory_bank.py`

传递 `pred_logits` 到 retriever:

```python
class MemoryBank(nn.Module):
    def forward(self, current_vision_feats, pred_logits=None, ...):
        # ...
        if self.sparse_memory_retriever:
            selected_indices = self.sparse_memory_retriever(
                memory_tokens,
                pred_logits=pred_logits,  # NEW
                ...
            )
```

### 3. 上游调用处

确保 `pred_logits` 被传递:

```python
# 在 train/eval loop 中
memory_output = memory_bank(
    current_vision_feats,
    pred_logits=pred_masks.logits,  # NEW: 从预测结果获取
    ...
)
```

---

## 📊 预期结果

### 单独效果 (Route 3 alone)

| Metric | Baseline | MTC | Route 3 | 改善 |
|--------|----------|-----|---------|------|
| mIoU | 60.22 | 60.01 | **60.20** | +0.19 |
| BFScore | 39.71 | 39.86 | **39.90** | +0.04 |
| traffic_light | 65.2 | 63.55 | **65.0** | +1.45 |
| fence | 58.3 | 57.53 | **58.5** | +0.97 |
| pole | 62.8 | 62.52 | **63.0** | +0.48 |

### 与 Boundary 组合

| Metric | Route 1 | Route 3 | Route1+3 | 叠加效果 |
|--------|---------|---------|----------|---------|
| mIoU | 60.25 | 60.20 | **60.35** | ✅ 正向 |
| BFScore | 39.85 | 39.90 | **39.95** | ✅ 正向 |
| traffic_light | 65.8 | 65.0 | **66.2** | ✅ 最佳 |
| fence | 59.1 | 58.5 | **59.5** | ✅ 正向 |

### Full System (MTC + Pano-SMB + Boundary + Class-Aware)

| Metric | Baseline | Full System | 改善 |
|--------|----------|-------------|------|
| mIoU | 60.22 | **60.45** | +0.23 |
| BFScore | 39.71 | **40.00** | +0.29 |
| traffic_light | 65.2 | **66.5** | +1.3 |
| fence | 58.3 | **59.5** | +1.2 |
| pole | 62.8 | **63.8** | +1.0 |

---

## 💡 与现有工作的区分

### vs SAMITE (prototype-guided)
- **SAMITE**: Frame-level foreground prototype selection
  - 跨帧选择哪些帧的 memory
  - 用 prototype 相似度
- **Ours**: Token-level rare-class protection
  - 帧内选择哪些 tokens
  - 固定总 quota，显式保护受损类

### vs Lean-SAM2 (target-anchor)
- **Lean-SAM2**: Prompt-anchor similarity
  - Object tracking 场景，有明确 object prompt
  - Target-aware pruning
- **Ours**: Class-aware without prompts
  - Dense semantic segmentation，无 prompt
  - Rare-class protection under fixed budget

### 核心差异
我们是: **Fixed-budget panoramic rare-class token protection for dense semantic UDA**

---

## 🚨 潜在问题与对策

### 问题1: Pseudo-label 噪声

**问题**: 目标域的 pred_logits 是 pseudo-label，可能有噪声

**对策**:
- 只在高置信度的 pseudo-label 上应用 class-aware
- 结合 `pseudo_conf_mask` 的 threshold=0.85
- 或者用 confidence weighting

### 问题2: Rare class 数量不稳定

**问题**: 不同图像中 rare class 出现数量差异大

**对策**:
- 设置 `min_rare_reserve` 和 `max_rare_reserve`
- 动态调整: 如果 rare class 很少，省下的 quota 还给 common
- 如果 rare class 很多，按 confidence 排序 Top-K

### 问题3: 与 Boundary 重叠

**问题**: Rare class 往往在边界，可能重复保护

**对策**:
- 策略1: Union 去重，总 quota 不变
- 策略2: 降低各自的 reserve_ratio (boundary 10% + class 10% = 20%)
- 策略3: 优先 boundary，class-aware 只作用于非边界区域

---

## 📅 实验排期建议

### 今晚 20:48 后立即启动

```bash
# 接在 Route 1 后面
20:48-22:00  Route 3 Phase 3A (3配置 × n500) = 1.2h
22:00-22:24  Route 3 Phase 3B (hybrid × n500) = 0.4h
22:24       完成，明早分析结果
```

**总时长**: 1.6h (vs 白天 7.5h 的 n2000 5-seed)

### 明天根据结果

如果 n500 显示有效 (rare class mIoU 提升):
```bash
上午: Route 3 最优配置 n2000 × 5seeds (7.5h)
晚上: Full System 组合实验
```

---

**文档创建时间**: 2026-08-25 16:20  
**预计首次实验**: 今晚 20:48  
**预期完成**: 今晚 22:24
