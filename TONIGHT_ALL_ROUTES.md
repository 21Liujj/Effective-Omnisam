# 今晚完整实验 - 三大创新路线

> 一次性验证 Pano-SMB + Route 1 + Route 3，明早全面分析
> 创建时间: 2026-08-25 16:25

---

## ⚡ 终极方案：三路并进

**所有实验用 n500 → 今晚 22:00 完成全部验证！**

```
Pano-SMB:  4配置 × n500 = 1.6h
Route 1:   3配置 × n500 = 1.2h  
Route 3:   3配置 × n500 = 1.2h  (🆕 新增)
─────────────────────────────────
总计:      10配置        = 4.0h
```

**vs 原计划**: 
- 原计划: Pano-SMB (6h) + Route 1 (4.5h) = 10.5h
- 新方案: 三路全验证 = 4.0h
- **节省**: 6.5h + 获得 Route 3 数据

---

## 📅 时间线

```
16:25 (现在)  ✅ 三路准备完成
18:00         → attn_topk 完成，启动实验
18:00-19:36   → Pano-SMB 4×n500 (1.6h)
19:36-20:48   → Route 1 3×n500 (1.2h)
20:48-22:00   → Route 3 3×n500 (1.2h) 🆕
22:00         → 全部完成！
```

---

## 🚀 一键启动 (18:00后)

```bash
ssh -p 33884 root@connect.bjb2.seetacloud.com
cd /root/autodl-tmp/OmniSAM-main
nohup ./start_tonight_all_routes.sh &
```

**然后关闭终端，明早 7:00 查看结果，8:00 讨论！**

---

## 📊 三大创新路线

### Route 0: Pano-SMB (FoV级优化)
**测试**: min_tokens = 128/256/512 + uniform baseline

| 配置 | 说明 | 预期 |
|------|------|------|
| B0_legacy | Uniform quota | 对照组 |
| B1_min128 | 激进分配 | 可能过度 |
| B1_min256 | 平衡分配 | 预期最优 ⭐ |
| B1_min512 | 保守分配 | 退化uniform |

**时间**: 1.6h  
**目标**: 找到最优 FoV 级分配策略

---

### Route 1: Boundary Guard (边界保护)
**测试**: boundary_reserve_ratio = 6.25%/12.5%/25%

| 配置 | 边界保留 | 特征Top-K | 预期 |
|------|---------|----------|------|
| boundary_0625 | 256 (6.25%) | 3840 | 保守 |
| boundary_125 | 512 (12.5%) | 3584 | 平衡 ⭐ |
| boundary_25 | 1024 (25%) | 3072 | 激进 |

**时间**: 1.2h  
**目标**: BFScore 恢复，细小类提升

---

### Route 3: Class-Aware (类别保护) 🆕
**测试**: rare_class_reserve_ratio = 10%/15%/20%

| 配置 | Rare Class保留 | 特征Top-K | 预期 |
|------|--------------|----------|------|
| route3_rare010 | 10% | 90% | 保守 |
| route3_rare015 | 15% | 85% | 平衡 ⭐ |
| route3_rare020 | 20% | 80% | 激进 |

**保护类别**: traffic_light, fence, pole, traffic_sign, terrain

**时间**: 1.2h  
**目标**: Rare class mIoU 直接提升

---

## 🎯 预期结果对比

### 各路线单独效果

| 路线 | mIoU | BFScore | 特点 |
|------|------|---------|------|
| Baseline | 60.22 | 39.71 | 无压缩 |
| MTC | 60.01 | 39.86 | 4×压缩 |
| +Pano-SMB | 60.15 | 39.92 | FoV级优化 |
| +Route 1 | 60.25 | 39.85 | 边界保护 |
| +Route 3 | 60.20 | 39.90 | 类别保护 🆕 |

### 受损类别恢复预期

| 类别 | Baseline | MTC | Route 1 | Route 3 | 组合 |
|------|----------|-----|---------|---------|------|
| traffic_light | 65.2 | 63.55 | **65.8** | **65.0** | **66.2** |
| fence | 58.3 | 57.53 | **59.1** | **58.5** | **59.5** |
| pole | 62.8 | 62.52 | **63.2** | **63.0** | **63.8** |

**关键观察**:
- Route 1 (Boundary): 边界类恢复好
- Route 3 (Class-Aware): 小物体类恢复好
- 组合效果: 两者互补，全面恢复

---

## 💡 明天分析重点

### 早上 7:00-8:00 查看结果

```bash
# 快速汇总
cd /root/autodl-tmp/OmniSAM-main
find experiments -name "eval_results.json" -path "*n500*" \
  -exec echo {} \; -exec cat {} \;
```

### 明天 8:00 讨论决策

**问题1**: Pano-SMB 最优配置？
- 预期: min256 最好
- 决策: n2000 确认 or 直接用

**问题2**: Route 1 vs Route 3 谁更好？
- Route 1: 边界保护，BFScore 导向
- Route 3: 类别保护，rare class 导向
- 决策: 都好 → 组合；一个好 → 深化

**问题3**: 组合策略？
```yaml
组合A: Pano-SMB + Route 1 (best)
组合B: Pano-SMB + Route 3 (best)
组合C: Route 1 + Route 3 (boundary + class)
组合D: Full System (全开)
```

**问题4**: 明天跑什么？
- 选项A: 最优单路线 n2000 5-seed (7.5h)
- 选项B: 多个组合 n500 快速 (3-4h)
- 选项C: 直接 Full System n2000 (1.5h)

---

## 📊 论文实验矩阵 (预期)

### Ablation Study (单独效果)

| D | E1(Pano) | E2(Route1) | E3(Route3) | mIoU | BFScore |
|---|---------|-----------|-----------|------|---------|
| ✓ |         |           |           | 60.01 | 39.86 |
| ✓ | ✓       |           |           | 60.15 | 39.92 |
| ✓ | ✓       | ✓         |           | 60.30 | 39.95 |
| ✓ | ✓       |           | ✓         | 60.25 | 39.93 |
| ✓ | ✓       | ✓         | ✓         | **60.45** | **40.00** |

### Per-Class Analysis

| Class | Base | +Route1 | +Route3 | Full |
|-------|------|---------|---------|------|
| traffic_light | 65.2 | 65.8 | 65.0 | **66.5** |
| fence | 58.3 | 59.1 | 58.5 | **59.8** |
| pole | 62.8 | 63.2 | 63.0 | **64.0** |
| road | 98.5 | 98.4 | 98.5 | **98.5** |

---

## 🔧 技术细节

### Route 3 核心算法

```python
def class_aware_selection(tokens, pred_logits, quota):
    # 1. 识别 rare class tokens
    rare_mask = identify_rare_classes(pred_logits)
    
    # 2. 分配 quota
    rare_quota = quota * rare_reserve_ratio
    common_quota = quota - rare_quota
    
    # 3. Rare class 优先选择
    selected_rare = select_rare_tokens(rare_mask, rare_quota)
    
    # 4. Common class 用 feature norm Top-K
    selected_common = select_common_topk(common_quota)
    
    # 5. 合并
    return cat(selected_rare, selected_common)
```

**关键优势**:
- ✅ 显式保护受损类
- ✅ 信号现成 (pred logits)
- ✅ 与 Route 1 正交互补
- ✅ 固定预算，无额外开销

---

## 🚨 注意事项

1. **n500 是快速验证**
   - 结果可能波动 ±0.3-0.5 mIoU
   - 主要看趋势和相对排序
   - 明天需 n2000 确认

2. **三路可能都有效**
   - Pano-SMB: FoV 级自适应
   - Route 1: 边界保护
   - Route 3: 类别保护
   - 三者正交，组合效果可能 > 单独

3. **明天重点**
   - 分析哪个路线最有潜力
   - 优先深化最强路线
   - 测试组合效果
   - 准备 Full System

---

## 📚 相关文档

- **Route 1 设计**: `experiments/route1/README_phase1.md`
- **Route 3 设计**: `experiments/route3/ROUTE3_DESIGN.md`
- **完整路线图**: `docs/innovation_analysis/post_2048_roadmap.md`
- **创新总览**: `INNOVATION_SUMMARY.md`

---

**创建时间**: 2026-08-25 16:25  
**启动时间**: 2026-08-25 18:00  
**预计完成**: 2026-08-25 22:00  
**总时长**: 4小时 (验证3大创新路线！)
