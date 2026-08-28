# 项目进度总结 - 2026年8月28日

**项目**: 360°全景分割优化（EI会议论文）  
**状态**: 创新点1和2已完成，创新点3设计完成待实验  
**下次讨论**: 2026年9月1日（周一）

---

## 已完成工作

### ✅ 创新点1: Memory Bank优化 (A+D+E模块)

**模块组成**:
- **A (CCPM)**: Class-Conditional Prompt Modulation
- **D (FSMC)**: Feature-Space Memory Compression - 4×压缩
- **E (Token Pruning)**: 动态token剪枝 (keep_ratio=0.50) - 2×压缩

**实验结果**:
- 组合效果: 8×memory压缩
- 注意力时间: -55.8%
- 性能损失: mIoU仅降0.24 (60.26 → 60.02)
- GPU显存节省: ~1.2GB

**状态**: ✅ 已完成，有完整实验数据和表格

**文件位置**:
- `experiments/tecma/tables/memory_token_attention_cost.md`
- 相关runs: `experiments/tecma/runs/A_Dtrain_Etrain_*`

---

### ✅ 创新点2: SCM (Seamless Circular Memory)

**核心思想**:
360°全景图在x=0和x=4096处物理连续，但模型处理时出现"接缝断裂"。
SCM通过循环padding让模型看到跨边界的连续上下文。

**实现**:
```python
# Dataset层面：循环padding
I_padded = [I[:, -w:], I, I[:, :w]]  # 4096 → 4096+2w

# Inference层面：crop回原始尺寸
pred_cropped = pred_padded[:, :, w:-w]
```

**Ablation Study结果**:

| Method | Pad Width | Val mIoU | Δ vs Baseline | Relative Gain |
|--------|-----------|----------|---------------|---------------|
| Baseline | - | 60.51 | - | - |
| SCM | 256 | 60.60 | +0.09 | +0.15% |
| SCM | 512 | 60.72 | +0.21 | +0.35% |
| SCM | 768 | 60.80 | +0.29 | +0.48% |
| **SCM** | **1024** | **60.84** | **+0.33** | **+0.55%** |

**Per-Class分析（Baseline vs SCM pad=1024）**:

| Class | Baseline | SCM (1024) | Δ | Analysis |
|-------|----------|------------|---|----------|
| **traffic sign** | 26.07 | **31.11** | **+5.04** | ⭐ 最显著提升（边界敏感） |
| wall | 45.19 | 46.07 | +0.88 | 边界类别受益 |
| fence | 49.43 | 49.69 | +0.26 | 边界类别 |
| car | 84.35 | 84.62 | +0.27 | 提升 |
| road | 81.00 | 80.91 | -0.09 | 大类别稳定 |
| building | 89.96 | 89.76 | -0.20 | 大类别稳定 |
| sky | 94.60 | 94.58 | -0.02 | 大类别稳定 |

**关键发现**:
1. ✅ Traffic Sign显著提升 (+5.04 IoU) - 验证边界连续性假设
2. ✅ 单调递增趋势 - pad_width越大性能越好
3. ✅ 大类别稳定 - 无负面影响

**Bug修复记录**:
- **致命Bug**: validation.py第367-370行重复crop代码
- **影响**: 预测图被crop两次 (4608→4096→3840)，导致mIoU暴跌到34.77
- **修复**: 删除重复代码
- **验证**: 修复后60.60 mIoU ✓

**状态**: ✅ 已完成，已提交git (commit 4658693)

**文件位置**:
- `experiments/tecma/results/SCM_EXPERIMENT_SUMMARY.md`
- `experiments/tecma/results/scm_ablation_table.md`
- `experiments/tecma/results/scm_per_class_analysis.md`
- `experiments/tecma/results/scm_ablation/pad{256,512,768,1024}.log`
- `dataset/DensePASS_circular.py`

---

### 🚧 创新点3: BEPL (Boundary-Enhanced Prompt Learning) - 设计完成

**核心思想**:
在边界区域添加可学习的视觉prompt，增强边界特征表达。

**重要澄清 - "Prompt"不是多模态！**

#### ❌ 不是：Text Prompt（多模态）
```python
# 不是这种
text_prompt = "segment traffic sign at boundary"
model(image, text_prompt)  # CLIP-style
```

#### ✅ 是：Visual Prompt（可学习特征向量）
```python
# 是这种：纯视觉，无文本
boundary_prompts = nn.Parameter(torch.randn(1, 256, 64))  # 可学习向量
enhanced_features = features + boundary_prompts * seam_mask
```

**技术本质**:
- 空间自适应的特征增强
- 类似Positional Encoding，但针对边界区域
- 借用"prompt"概念，但纯视觉

**备选命名**（如果"prompt"引起误解）:
- Boundary-Aware Feature Enhancement (BAFE)
- Learnable Boundary Augmentation (LBA)
- Boundary-Specific Adapter (BSA)

**实现方案**:

##### 方案A: 推理版本（无需训练，快速验证）
- 直接对logits应用边界增强
- 使用高斯权重 + 对比度增强
- **预计实验时间**: 15分钟
- **预期提升**: +0.1~0.2 mIoU

##### 方案B: 训练版本（完整版）
- 可学习的boundary prompt参数
- 需要重新训练模型
- **预计实验时间**: 数小时
- **预期提升**: +0.2~0.4 mIoU

**与已有创新点的协同**:
- **与Memory优化**: BEPL可以用更少memory专注边界
- **与SCM**: SCM提供数据连续性，BEPL提供特征增强
- **形成系统**: Memory压缩 → 数据连续 → 特征增强

**状态**: 🚧 设计完成，代码已实现，待实验验证

**文件位置**:
- `adaptation/boundary_prompt.py` (训练版本)
- `adaptation/boundary_prompt_inference.py` (推理版本)
- `validation.py` (已集成BEPL支持，备份在validation.py.backup_before_bepl)

**命令行参数**:
```bash
python validation.py \
    --enable_scm --scm_pad_width 1024 \
    --enable_bepl --bepl_seam_width 128 --bepl_strength 0.2
```

---

## 论文结构规划

### Title (待定)
- "Efficient 360° Panoramic Segmentation with Boundary-Aware Memory and Seamless Context"
- "Memory-Efficient Panoramic Segmentation via Seamless Boundary Enhancement"

### Abstract结构
1. 问题: 360°全景分割的挑战（边界不连续 + 高计算量）
2. 创新点1: Memory优化（8×压缩，-55.8%时间）
3. 创新点2: SCM边界连续性（+0.33 mIoU）
4. 创新点3: BEPL特征增强（预期+0.2）
5. 结果: DensePASS SOTA，高效推理

### Method Section
- **Section 3.1**: Memory Bank优化 (A+D+E)
- **Section 3.2**: Seamless Circular Memory (SCM)
- **Section 3.3**: Boundary-Enhanced Prompt Learning (BEPL)
- **Section 3.4**: 系统集成

### Experiments Section
- **Section 4.1**: 实验设置
- **Section 4.2**: Memory优化ablation
- **Section 4.3**: SCM ablation + per-class分析
- **Section 4.4**: BEPL验证
- **Section 4.5**: 完整系统对比
- **Section 4.6**: 可视化

---

## Git提交记录

### Commit 4658693 (2026-08-28)
```
feat(scm): Complete SCM ablation study (+0.33 mIoU, +5.04 traffic sign)

- Implement circular padding for 360° panorama boundary continuity
- Add DensePASS13Circular and DensePASS19Circular dataset classes
- Ablation on pad_width: 256/512/768/1024
- Best config: pad_width=1024, 60.84 mIoU (+0.33 vs baseline 60.51)
- Traffic sign: 31.11 (+5.04), validates boundary hypothesis
- Fix: Remove duplicate crop bug in validation.py line 367-370
```

---

## 下周工作计划（2026年9月1日开始）

### 高优先级
1. **BEPL实验验证**
   - [ ] 运行推理版本快速测试（15分钟）
   - [ ] 如果有效，考虑训练完整版本
   - [ ] 记录结果，更新论文表格

2. **系统集成测试**
   - [ ] Memory优化 + SCM + BEPL组合实验
   - [ ] 生成最终性能表格

3. **可视化生成**
   - [ ] Baseline vs SCM边界对比图
   - [ ] Traffic sign可视化案例
   - [ ] Attention map可视化（如果有BEPL）

### 中优先级
4. **论文撰写**
   - [ ] Method Section 3.2 (SCM) - 已有完整材料
   - [ ] Method Section 3.3 (BEPL) - 等实验结果
   - [ ] Results Section 4.3-4.4

5. **补充实验（可选）**
   - [ ] WildPASS数据集验证泛化性
   - [ ] 计算量分析 (FLOPs, inference time)
   - [ ] 不同backbone ablation (SAM2-B, SAM2-L)

### 低优先级
6. **扩展工作（硕士论文用）**
   - [ ] Seam consistency loss训练版本
   - [ ] Latitude-aware positional encoding
   - [ ] 更多ablation studies

---

## 实验数据汇总

### Dataset
- **DensePASS**: 2000 train + 100 val
- **Resolution**: 4096×1024 (全景图)
- **Classes**: 13类 (Cityscapes subset)

### Model
- **Backbone**: SAM2-S
- **Checkpoint**: `best_iou59.99.pth`
- **Memory Bank**: 启用 (num_maskmem=9)

### GPU
- **平台**: AutoDL RTX 5090
- **SSH**: bjb2:33884 (2026-08-24新实例)

### 实验耗时估算
- Baseline验证: ~4分钟
- SCM单个pad_width: ~5分钟
- 完整ablation (4个配置): ~20分钟
- BEPL快速测试: ~15分钟
- 可视化生成: ~30分钟

---

## 技术笔记

### SCM实现要点
1. Dataset层需要同步padding image和label
2. Validation需要动态计算padded_width
3. 注意crop逻辑，避免重复crop bug
4. frame_cnt计算要考虑padded width

### BEPL实现要点
1. 边界mask使用高斯衰减，平滑过渡
2. 增强强度需要ablation（0.1~0.3）
3. 可以对features或logits应用
4. 推理版本无需训练，训练版本效果更好

### 常见问题
1. **Q**: pad_width越大越好吗？
   **A**: 理论上是，但有边际效应递减。1024已接近饱和，且计算量增加~20%

2. **Q**: BEPL的"prompt"是多模态吗？
   **A**: 不是！是纯视觉的可学习特征向量，不涉及text

3. **Q**: 三个创新点够吗？
   **A**: 对EI会议足够。三个点环环相扣：效率+边界+增强

---

## 文件清单

### 代码
```
dataset/DensePASS_circular.py          # SCM dataset实现
adaptation/boundary_prompt.py          # BEPL训练版本
adaptation/boundary_prompt_inference.py # BEPL推理版本
validation.py                          # 主验证脚本（已集成SCM+BEPL）
experiments/tecma/run_ablation.sh      # SCM ablation脚本
```

### 结果与文档
```
experiments/tecma/results/
├── SCM_EXPERIMENT_SUMMARY.md          # SCM完整总结
├── scm_ablation_table.md              # Ablation表格（LaTeX）
├── scm_per_class_analysis.md          # Per-class对比
├── scm_ablation/
│   ├── pad256.log
│   ├── pad512.log
│   ├── pad768.log
│   └── pad1024.log
├── baseline_quick.log                 # Baseline结果
└── PROJECT_STATUS_2026_08_28.md       # 本文档
```

### Memory优化相关
```
experiments/tecma/tables/memory_token_attention_cost.md
experiments/tecma/runs/A_Dtrain_Etrain_*/
```

---

## 联系信息

**AutoDL服务器**:
- Host: bjb2:33884
- Password: H/kJPbUbNG3V
- 有效期: 至2026-08-24新实例到期

**项目路径**: `/root/autodl-tmp/OmniSAM-main`

**分支**: `feature/scm-circular-memory`

---

## 下周一见！

**待讨论事项**:
1. BEPL实验结果评审
2. 是否需要训练完整版BEPL
3. 论文写作分工与时间规划
4. 可视化需求确认
5. 投稿目标会议确认（deadline？）

**准备工作**（周末可选）:
- [ ] 思考论文title
- [ ] 整理相关论文（360°分割 + boundary增强）
- [ ] 思考可视化方案（想展示什么效果）

---

**状态**: ✅ 两个创新点完成，第三个设计完成  
**信心**: 💪 实验数据完整，技术路线清晰  
**期待**: 🚀 下周一继续冲刺！

