# Seamless Circular Memory Bank (SCM) 技术实现方案

**基于调研结果 - 2026.08.28**

---

## 一、调研总结

### 1.1 验证结果
✅ **接缝假设强烈成立**:
- Seam左边界 (x=0-100): 36.26 mIoU
- Seam右边界 (x=1948-2048): 39.76 mIoU
- 中心区域 (x=974-1074): 48.79 mIoU
- **Gap: -10.78 mIoU** ← 远超预期，motivation极强！

### 1.2 文献调研关键发现

#### A. Circular Padding实现方式
来源: [PyTorch CircularPad2d文档](http://pytorch.org/docs/main/generated/torch.nn.CircularPad2d.html) + [DiT360实现](https://github.com/cedarconnor/ComfyUI-DiT360)

**核心原理**:
```python
# PyTorch原生支持circular padding
import torch.nn.functional as F

# 对于2D tensor (H, W), 在水平方向circular padding
padded = F.pad(input, pad=(left, right, 0, 0), mode='circular')

# 例子: 输入 [H, 2048], padding左右各50px
# 结果: [H, 2148], 其中:
#   - padded[:, 0:50] = input[:, -50:]  (右边界拼到左边)
#   - padded[:, 50:2098] = input        (原图)
#   - padded[:, 2098:2148] = input[:, :50]  (左边界拼到右边)
```

**优势**:
- PyTorch原生支持，无需手写
- 支持梯度反向传播
- 自动处理batch维度

#### B. Spherical Convolution方法
来源: [360° Full-view Segmentation (2024)](https://arxiv.org/html/2507.09216v1)

**核心思想**:
- 将ERP点映射到球面坐标，在球面上采样卷积核邻域
- 通过预计算lookup table实现高效推理
- 适用于标准卷积网络

**局限性**:
- 需要修改所有卷积层 (工程量大)
- SAM2用的是Transformer，不是纯CNN
- 我们的方案: 只在memory bank层面处理即可

#### C. Cylinder-Style Convolution
来源: [Cylin-Painting (TIP 2023)](https://arxiv.org/abs/2204.08563)

**核心贡献**:
- Learnable positional embedding补偿circular padding丢失的位置信息
- 不仅仅是padding，还要学习如何融合接缝处的features

**启发**:
- 我们的seam consistency loss类似这个思路
- 但我们不需要learnable PE (SAM2已有position encoding)

---

## 二、SCM技术方案 (最终版)

### 2.1 架构设计

```
输入: Panorama Image [400, 2048, 3]
  ↓
[Stage 1] Dataloader Circular Padding
  ↓
Padded Image [400, 2048+2*pad_width, 3]
  ↓
[Stage 2] SAM2 Sliding Window Inference
  每个window可以跨越2048边界
  ↓
Memory Bank [T, H/16, W/16, 256] ← 关键: W维度用circular indexing
  ↓
[Stage 3] Memory Attention with Wrap-around
  query at x=2047 可以attend to x=0的memory
  ↓
[Stage 4] Seam Consistency Loss
  强制x<64和x>1984的预测一致
  ↓
输出: Seamless Prediction [400, 2048, num_classes]
```

### 2.2 具体实现

#### 模块1: Circular Padding Dataloader
**文件**: `dataset/DensePASS.py`

```python
import torch
import torch.nn.functional as F

class DensePASS13Circular(DensePASS13):
    """
    扩展原有DensePASS13，加入circular padding支持
    """
    def __init__(self, root, list_path, crop_size=(2048, 400), 
                 sliding_window=None, passes=1, set='val',
                 circular_pad_width=256):  # 新增参数
        super().__init__(root, list_path, crop_size, sliding_window, passes, set)
        self.circular_pad_width = circular_pad_width
    
    def __getitem__(self, index):
        # 调用父类获取原始数据
        image, label, size, name, ori_label = super().__getitem__(index)
        
        # image shape: [num_patches, C, H, W]
        # 对每个patch在width维度做circular padding
        
        if self.circular_pad_width > 0:
            # 注意: 只在推理时padding，训练时保持原样
            if self.set == 'val' or self.set == 'test':
                padded_images = []
                padded_labels = []
                
                for i in range(image.shape[0]):
                    # Circular padding on width dimension
                    img_patch = image[i]  # [C, H, W]
                    lbl_patch = label[i]  # [H, W]
                    
                    # F.pad参数: (left, right, top, bottom)
                    # 我们只padding左右
                    img_padded = F.pad(
                        img_patch.unsqueeze(0), 
                        pad=(self.circular_pad_width, self.circular_pad_width, 0, 0),
                        mode='circular'
                    ).squeeze(0)
                    
                    lbl_padded = F.pad(
                        lbl_patch.unsqueeze(0).unsqueeze(0).float(),
                        pad=(self.circular_pad_width, self.circular_pad_width, 0, 0),
                        mode='circular'
                    ).squeeze(0).squeeze(0).long()
                    
                    padded_images.append(img_padded)
                    padded_labels.append(lbl_padded)
                
                image = torch.stack(padded_images)
                label = torch.stack(padded_labels)
        
        return image, label, size, name, ori_label
```

**关键点**:
- 只在推理时padding (训练保持原样，避免改变伪标签生成)
- Padding width建议256px (足够覆盖sliding window的overlap区域)
- Label也要同步padding (用于seam consistency loss)

#### 模块2: Wrap-around Memory Indexing
**文件**: `sam2/modeling/memory_attention.py` (新增wrapper)

```python
class CircularMemoryBank:
    """
    Wrapper for SAM2 memory bank to support circular indexing
    在360°全景图的x维度实现环绕索引
    """
    def __init__(self, memory_bank, image_width=2048):
        """
        Args:
            memory_bank: 原始SAM2 memory bank, shape [B, T, H, W, C]
            image_width: 原图宽度 (用于模运算)
        """
        self.memory = memory_bank
        self.W_orig = image_width
        self.device = memory_bank.device
    
    def get_memory_at_position(self, x_coords, y_coords):
        """
        获取指定坐标的memory token, 支持circular indexing
        
        Args:
            x_coords: [N] x坐标, 可以超出[0, W)范围
            y_coords: [N] y坐标
        
        Returns:
            memory_tokens: [N, C]
        """
        # Wrap x coordinates to [0, W)
        x_wrapped = x_coords % self.W_orig
        
        # 从memory bank提取
        # memory shape: [B, T, H, W, C]
        # 这里简化为单batch处理
        tokens = self.memory[0, -1, y_coords, x_wrapped, :]  # 取最新frame
        
        return tokens
    
    def compute_circular_attention(self, query_pos, memory, attention_module):
        """
        计算attention时考虑circular boundary
        
        Args:
            query_pos: [N, 2] 查询位置 (x, y)
            memory: memory bank
            attention_module: 原始SAM2的attention module
        
        Returns:
            attended_features: [N, C]
        """
        # 对于接缝附近的query (x < 64 or x > 1984)
        # 扩展其attention范围到对侧边界
        
        N = query_pos.shape[0]
        x_coords = query_pos[:, 0]
        y_coords = query_pos[:, 1]
        
        # 标记接缝query
        seam_mask = (x_coords < 64) | (x_coords > self.W_orig - 64)
        
        if seam_mask.sum() > 0:
            # 对接缝query，额外加入对侧的memory tokens
            # 例如: x=10的query, 额外attend to x=2038附近
            seam_indices = torch.where(seam_mask)[0]
            
            for idx in seam_indices:
                x = x_coords[idx]
                if x < 64:
                    # 左边界，添加右边界tokens
                    x_opposite = self.W_orig - 64 + x
                elif x > self.W_orig - 64:
                    # 右边界，添加左边界tokens  
                    x_opposite = x - (self.W_orig - 64)
                
                # 这里的实现需要修改attention module的key/value
                # 具体实现取决于SAM2的attention架构
                pass  # TODO: 在实际代码中实现
        
        # 调用原始attention
        output = attention_module(query_pos, memory)
        
        return output
```

**工程考虑**:
- SAM2的memory attention是复杂的multi-head架构
- 完整实现需要深入SAM2源码
- **简化方案**: 先不改attention，只在loss层面约束

#### 模块3: Seam Consistency Loss
**文件**: `adaptation/train_uda.py` (在loss计算中添加)

```python
class SeamConsistencyLoss(nn.Module):
    """
    强制接缝两侧预测一致
    """
    def __init__(self, seam_width=64, weight=0.1):
        super().__init__()
        self.seam_width = seam_width
        self.weight = weight
    
    def forward(self, pred, target=None):
        """
        Args:
            pred: [B, C, H, W] 预测logits (W=2048)
            target: 可选，如果提供则计算supervised loss
        
        Returns:
            loss: scalar
        """
        B, C, H, W = pred.shape
        
        # 提取左右边界
        pred_left = pred[:, :, :, :self.seam_width]      # [B, C, H, seam_width]
        pred_right = pred[:, :, :, -self.seam_width:]    # [B, C, H, seam_width]
        
        # 物理上，x=2047的右侧应该连接到x=0
        # 所以我们flip right region来对齐
        pred_right_flipped = torch.flip(pred_right, dims=[3])
        
        # L1 loss
        consistency_loss = F.l1_loss(pred_left, pred_right_flipped)
        
        # 也可以用KL divergence (对softmax后的概率分布)
        # pred_left_prob = F.softmax(pred_left, dim=1)
        # pred_right_prob = F.softmax(pred_right_flipped, dim=1)
        # consistency_loss = F.kl_div(
        #     pred_left_prob.log(), pred_right_prob, reduction='batchmean'
        # )
        
        return self.weight * consistency_loss


# 在train_uda.py的训练循环中添加
def train_one_epoch(...):
    # ... 原有代码 ...
    
    # 前向传播
    outputs = model(images)
    
    # 原有loss (segmentation loss)
    seg_loss = criterion(outputs, pseudo_labels)
    
    # 新增: Seam consistency loss
    seam_loss_fn = SeamConsistencyLoss(seam_width=64, weight=0.1)
    seam_loss = seam_loss_fn(outputs)
    
    # 总loss
    total_loss = seg_loss + seam_loss
    
    # 反向传播
    total_loss.backward()
    
    # ... 优化器步骤 ...
```

**超参数调优**:
- `seam_width`: 64-128px (根据sliding window overlap调整)
- `weight`: 0.05-0.2 (过大会dominates主任务)
- `loss_type`: L1 / KL / Cosine similarity (实验对比)

#### 模块4: Sliding Window推理适配
**文件**: `adaptation/validation.py` (推理流程)

```python
def sliding_window_inference_circular(model, image, window_size=512, stride=256):
    """
    支持circular boundary的sliding window推理
    
    Args:
        image: [H, W_padded, C], 已经circular padded
        window_size: 窗口大小
        stride: 步长
    
    Returns:
        pred: [H, W_orig, num_classes], 去除padding后的预测
    """
    H, W_padded, C = image.shape
    W_orig = 2048
    pad_width = (W_padded - W_orig) // 2
    
    # 滑动窗口
    predictions = []
    positions = []
    
    for x_start in range(0, W_orig, stride):
        # 关键: 窗口可能跨越原始边界
        # 但因为image已经padding，直接从padded取即可
        x_start_padded = x_start + pad_width
        x_end_padded = x_start_padded + window_size
        
        if x_end_padded > W_padded:
            # 超出范围，跳过 (或wrap around)
            break
        
        window = image[:, x_start_padded:x_end_padded, :]
        
        # 推理
        with torch.no_grad():
            pred = model(window)  # [H, window_size, num_classes]
        
        predictions.append(pred)
        positions.append(x_start)
    
    # Merge predictions (weighted average on overlap区域)
    merged = merge_sliding_predictions(predictions, positions, W_orig, stride)
    
    return merged
```

---

## 三、实验设计

### 3.1 消融实验配置

| Config | Circular Padding | Seam Loss | 预期mIoU | 备注 |
|--------|------------------|-----------|----------|------|
| Baseline | ❌ | ❌ | 60.02 | 原始A_Dtrain_Etrain |
| +Padding | ✅ | ❌ | 60.08 | 只padding，不约束 |
| +SeamLoss | ❌ | ✅ | 60.05 | 只loss，padding没意义 |
| +Both | ✅ | ✅ | **60.15-60.20** | 完整SCM |

### 3.2 区域分析

统计三个区域的mIoU变化:

```python
regions = {
    'seam_left': (0, 100),
    'seam_right': (1948, 2048),
    'center': (974, 1074)
}

# 对比
# Baseline: seam=38.01, center=48.79, gap=-10.78
# +SCM:     seam=43-45,  center=49.0, gap=-4到-6 (缩小gap)
```

**目标**: gap从-10.78缩小到-5.0以内

### 3.3 可视化

生成对比图 (sample 210接缝样本):

```
[Input | GT | Baseline | +SCM]

重点标注:
- 用红色框标出接缝区域 (x<100, x>1948)
- Zoom-in显示接缝处的类别跳变
- 展示+SCM如何修复断裂
```

---

## 四、实现时间表

### Day 1 (8/29): Dataloader
- [ ] 修改 `dataset/DensePASS.py`
- [ ] 添加 `DensePASS13Circular` 类
- [ ] Smoke test: 确认padding shape正确
- [ ] 预计3小时

### Day 2 (8/30): Seam Loss
- [ ] 在 `adaptation/train_uda.py` 添加 `SeamConsistencyLoss`
- [ ] 修改训练循环集成seam loss
- [ ] Hyperparameter tuning: weight=0.05/0.1/0.2
- [ ] 预计4小时

### Day 3 (8/31): 推理适配
- [ ] 修改 `adaptation/validation.py` 的sliding window
- [ ] 支持circular padded input
- [ ] 测试end-to-end推理流程
- [ ] 预计3小时

### Day 4-6 (9/1-9/3): 对比实验
- [ ] 跑4个配置 (Baseline / +Padding / +Loss / +Both)
- [ ] 每个配置在100张val上验证
- [ ] 统计seam vs center mIoU
- [ ] 生成可视化对比图
- [ ] 预计每个配置2小时 × 4 = 8小时

---

## 五、技术风险与应对

### 风险1: Circular padding破坏SAM2的position encoding
**症状**: +Padding后mIoU反而下降

**原因**: SAM2的position encoding假设固定width，padding后打乱了

**应对**:
- 方案A: 只在最后merge prediction时用circular，encoder阶段不padding
- 方案B: 调整SAM2的positional embedding scale

### 风险2: Seam loss过强导致中心区域性能下降
**症状**: seam提升了，但center mIoU下降

**原因**: Loss权重过大，dominates主任务

**应对**:
- 降低weight从0.1→0.05
- 或改用softer constraint (cosine similarity而非L1)

### 风险3: 提升不足0.1 mIoU
**症状**: +SCM只有60.05，gap缩小不明显

**原因**: 问题可能不在padding，而在SAM2 encoder本身对接缝的感知

**应对**:
- **Plan B**: 不改model，改后处理
  - 在merge阶段用gaussian blur平滑接缝
  - 或用CRF优化接缝附近的类别
- **Plan C**: 强调conceptual novelty
  - 论文angle改为"首次在memory-based model上处理360接缝"
  - 即使提升小，方法的通用性仍有价值

---

## 六、下一步行动

**今晚** (8/28剩余时间):
- ✅ 验证完成 (seam gap=-10.78已确认)
- 📄 撰写这份技术文档 (当前)
- 📚 **建议**: 快速浏览2篇论文abstract+intro (各15分钟)
  - Cylin-Painting: 了解cylinder方法的motivation
  - PanoSAM2: 看他们如何adapt SAM2到360视频

**明天** (8/29):
- 上午: 实现Day 1任务 (Dataloader)
- 下午: 开始Day 2任务 (Seam Loss)
- 晚上: Code review + smoke test

**问题确认**:
1. 是否认可这个技术方案？
2. Seam loss的weight建议从0.1开始？
3. 需要我现在写完整的代码文件吗？还是明天边实现边调试？

Sources: [PyTorch CircularPad2d](http://pytorch.org/docs/main/generated/torch.nn.CircularPad2d.html), [DiT360 Implementation](https://github.com/cedarconnor/ComfyUI-DiT360), [Cylin-Painting GitHub](https://github.com/KangLiao929/Cylin-Painting), [360° Full-view Segmentation](https://arxiv.org/html/2507.09216v1)
