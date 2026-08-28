# 今晚快速实验 - 方案B (n500)

## ⚡ 核心优势

**所有实验用 n500 子集 → 今晚就能看到结果！**

- Pano-SMB: 4配置 × n500 = 1.6h (vs 原6h)
- Route 1: 3配置 × n500 = 1.2h (vs 原4.5h)
- **总时长: 2.8h** (vs 原10.5h)
- **完成时间: 20:48** (vs 原次日01:12)

---

## 📅 时间线

```
15:54 (现在)  ✅ 方案B准备完成
18:00         → attn_topk 完成，一键启动
18:00-19:36   → Pano-SMB 4×n500 (1.6h)
19:36-20:48   → Route 1 3×n500 (1.2h，自动接续)
20:48         → 全部完成！当晚看结果
```

---

## 🚀 一键启动 (18:00后)

```bash
ssh -p 33884 root@connect.bjb2.seetacloud.com
cd /root/autodl-tmp/OmniSAM-main
nohup ./start_tonight_fast.sh &
```

**可以关闭终端，晚上9点就能看结果！**

---

## 📊 实验内容

### Step 1: Pano-SMB Sensitivity (n500, 1.6h)

测试 **4个 min_tokens** 配置：

| 配置 | min_tokens | 说明 | 预期 |
|------|-----------|------|------|
| B0_legacy | - | Uniform quota baseline | 对照组 |
| B1_min128 | 128 | 激进分配 | 可能过度 |
| B1_min256 | 256 | 平衡分配 (默认) | 预期最优 |
| B1_min512 | 512 | 保守分配 | 退化uniform |

**每配置**: ~0.4h  
**输出**: `experiments/pano_smb_stage2/B*_n500/run/`

---

### Step 2: Route 1 Phase 1 (n500, 1.2h)

测试 **3个 boundary_reserve_ratio** 配置：

| 配置 | boundary_reserve_ratio | 边界保留 | 特征Top-K |
|------|----------------------|---------|-----------|
| boundary_0625 | 0.0625 | 256 tokens (6.25%) | 3840 |
| boundary_125 | 0.125 | 512 tokens (12.5%) | 3584 |
| boundary_25 | 0.25 | 1024 tokens (25%) | 3072 |

**每配置**: ~0.4h  
**输出**: `experiments/route1/runs/*_n500_*/`

---

## 🎯 今晚目标

### n500 快速验证
- ✅ 找出趋势：哪个配置方向对
- ✅ 初步筛选：排除明显不work的
- ⚠️ 结果可能波动，只看大致趋势

### 明天 n2000 确认
- 对有潜力的配置用 n2000 正式验证
- 得到可靠的论文数据

---

## 📈 查看结果 (20:48后)

### 快速查看所有结果
```bash
cd /root/autodl-tmp/OmniSAM-main
find experiments -name "eval_results.json" -path "*n500*" -exec echo {} \; -exec cat {} \;
```

### Pano-SMB 结果
```bash
for dir in experiments/pano_smb_stage2/B*_n500/run; do
  echo "=== $(basename $(dirname $dir)) ==="
  cat $dir/eval_results.json 2>/dev/null | python -c "import json,sys; d=json.load(sys.stdin); print(f\"mIoU: {d[mIoU]:.2f}, BFScore: {d.get(BFScore,N/A)}\")"
done
```

### Route 1 结果
```bash
for dir in experiments/route1/runs/*_n500_*/; do
  echo "=== $(basename $dir) ==="
  cat $dir/eval_results.json 2>/dev/null | python -c "import json,sys; d=json.load(sys.stdin); print(f\"mIoU: {d[mIoU]:.2f}, BFScore: {d.get(BFScore,N/A)}\")"
done
```

---

## 💡 明天计划

根据今晚 n500 结果决定：

### 如果 Pano-SMB 有效配置
```bash
# 用 n2000 确认最优配置
B1_min256_n2000  (如果 n500 显示256最好)
时间: ~1.5h
```

### 如果 Route 1 有效配置
```bash
# 用 n2000 确认最优配置
boundary_125_n2000  (如果 n500 显示12.5%最好)
时间: ~1.5h
```

### 组合实验
```bash
# 测试最强组合
CAMS + Pano-SMB_best
Boundary_best + Pano-SMB_best
时间: 每个 ~1.5h
```

### 更多创新点
```bash
# Route 2: Overlap-Aware (如果geometry修复好)
# Route 3: Class-Aware
# 每个 3×n500 = 1.2h 快速验证
```

---

## 📊 与 baseline 对比

| 实验 | 数据集大小 | mIoU | BFScore | 状态 |
|------|----------|------|---------|------|
| **Baseline** | full | 60.22±0.04 | 39.71±0.19 | ✅ 完成 |
| **CAMS** | full | 59.96±0.04 | 39.46±0.13 | ✅ 完成 |
| **attn_topk** | full | ~60.02 (预计) | ? | 🔄 18:00完成 |
| **Pano-SMB** | n500 | ? | ? | ⏳ 今晚18:00-19:36 |
| **Route 1** | n500 | ? | ? | ⏳ 今晚19:36-20:48 |

---

## ⚠️ 注意事项

1. **n500 是快速验证**
   - 结果会波动 (±0.5 mIoU)
   - 只看趋势和相对排序
   - 不能直接用于论文

2. **明天需要 n2000 确认**
   - 对有效配置做正式验证
   - 得到稳定的 5-seed 数据

3. **今晚就能讨论下一步**
   - 20:48 看完结果
   - 决定明天重点方向
   - 继续讨论更多创新点

---

## 🔧 监控命令

```bash
# 实时日志
tail -f tonight_fast_*.log

# GPU状态
watch -n 5 nvidia-smi

# 当前阶段
ps aux | grep run_adaptation | grep -oE "(pano_smb|boundary)"

# 停止实验
pkill -f run_adaptation.py
```

---

**创建时间**: 2026-08-25 15:54  
**启动时间**: 2026-08-25 18:00  
**预计完成**: 2026-08-25 20:48  
**总时长**: 2.8小时 (节省7.2小时！)
