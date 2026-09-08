# SCM Hyperparameter Experiments Plan

## 当前结果（pad_width=512）
- Baseline: 60.51 mIoU
- +SCM (pad_width=512): 60.72 mIoU (+0.21)
- Traffic sign: +5.38 (最大提升)

## 实验计划

### Experiment 1: Padding Width Ablation

**目的**: 找到最优padding width

| pad_width | 预期patches数 | 理论依据 |
|-----------|--------------|----------|
| 256 | 20 (10×2) | 最小overlap，快速 |
| **512** | **22 (11×2)** | **当前最优** ✓ |
| 768 | 24 (12×2) | 更大overlap |
| 1024 | 26 (13×2) | 最大overlap，慢 |

**运行命令**:
```bash
for pad_width in 256 512 768 1024; do
    echo "Testing pad_width=$pad_width"
    python validation.py \
        --dataset DensePASS \
        --class_set 13 \
        --backbone sam2_s \
        --use_mem_bank \
        --batch_size 1 \
        --ckpt best_iou59.99.pth \
        --enable_scm \
        --scm_pad_width $pad_width \
        | tee results/scm_pad${pad_width}.log
done
```

**预期结果**:
- pad_width=256: 可能不够，边界信息不足
- pad_width=512: 当前最优 (60.72)
- pad_width=768/1024: 可能饱和，推理更慢

### Experiment 2: Seam Loss Weight Tuning (Training)

**目的**: 在训练时加入seam consistency loss

```python
# adaptation/losses.py
class SeamConsistencyLoss(nn.Module):
    def __init__(self, seam_width=128, weight=0.1):
        self.seam_width = seam_width
        self.weight = weight
    
    def forward(self, pred_logits):
        # pred_logits: [B, C, H, W]
        left = pred_logits[:, :, :, :self.seam_width]
        right = pred_logits[:, :, :, -self.seam_width:]
        loss = F.l1_loss(left, right)
        return self.weight * loss
```

**Weight ablation**:
- 0.0 (baseline)
- 0.05 (light)
- 0.1 (medium)
- 0.2 (strong)

**注**: 这需要重新训练，暂时优先级较低

### Experiment 3: Different Seam Widths for Analysis

**目的**: 分析不同宽度的seam region

| seam_width | 分析区域 |
|------------|----------|
| 64 | Narrow (immediate boundary) |
| 128 | Medium (current) |
| 256 | Wide (extended boundary) |

### Experiment 4: SCM + Other Techniques

**目的**: 组合其他优化

| Config | mIoU (expected) |
|--------|-----------------|
| Baseline | 60.51 |
| +SCM | 60.72 |
| +SCM +TTA | ~61.0 |
| +SCM +CAC | ~61.2 |
| +SCM +CAC +LPE | ~61.5+ |

## 当前优先级

### 高优先级（今天完成）
1. ✅ SCM实现和验证 (60.72 mIoU)
2. ✅ Comprehensive metrics分析
3. ⏳ Padding width ablation (256/768/1024)
4. ⏳ 生成可视化对比图

### 中优先级（本周完成）
5. CAC (Class-Adaptive Compression) 实现
6. 超参数调优
7. 论文Section 3.1撰写

### 低优先级（硕士论文）
8. Seam consistency loss训练
9. LPE (Latitude-aware Positional Encoding)
10. 更多ablation studies

## 时间估算

- Padding width ablation: 3×4分钟 = 12分钟
- 可视化生成: 30分钟
- CAC实现: 2-3小时
- 论文撰写: 3-4小时

**总计**: 今天可以完成padding ablation + 可视化
