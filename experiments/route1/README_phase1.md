# Route 1 Phase 1: Boundary-Guarded Memory Selection

## 实验目标

验证 **Boundary-Guarded Selection** 创新点的有效性，针对当前系统的最大痛点：
- BFScore 下降（39.23 vs 39.59）
- 细小/边界类损伤（traffic light -1.65, fence -0.77, pole -0.28）

## 核心思路

在固定 4096 memory tokens 预算下：
1. 用上游 pred mask 提取边界（Sobel/morphology）
2. 保留 `boundary_reserve_ratio` 的 quota 给 boundary tokens
3. 剩余用 feature norm Top-K 填满

**关键优势**：
- ✅ 针对性强：直击 BFScore 和细小类问题
- ✅ 新颖度高：文献未发现同构方法
- ✅ 开销低：O(N) + 两次 Top-K
- ✅ 信号现成：上游已有 pred mask

## 实验设计

### Phase 1: 敏感性分析（1 seed 快速验证）

测试 3 个 `boundary_reserve_ratio` 值：

| Config | boundary_reserve_ratio | 保留边界 tokens | 剩余 Top-K | 总 tokens |
|--------|------------------------|----------------|-----------|----------|
| boundary_0625 | 0.0625 | 256 (6.25%) | 3840 | 4096 |
| boundary_125 | 0.125 | 512 (12.5%) | 3584 | 4096 |
| boundary_25 | 0.25 | 1024 (25%) | 3072 | 4096 |

**Baseline 对照**：
- baseline_omnisam: 60.22±0.04 mIoU, 39.71±0.19 BFScore (32768 tokens, 无压缩)
- A_Dtrain_Etrain (attn_topk): ~60.02 mIoU (预计，5-seed 进行中)
- CAMS: 59.96±0.04 mIoU, 39.46±0.13 BFScore (4096 tokens)

**时间安排**：
- 每个配置：~1.5h
- 总时间：~4.5h
- 预计完成：今晚 24:00 → 次日 4:30

### Phase 2: 最优配置验证（5 seeds，如果 Phase 1 有效）

如果 Phase 1 发现某个配置 > 60.2，再用 5 seeds (1/2/3/42/100) 验证稳定性。

## 文件结构

```
experiments/route1/
├── boundary_0625.yaml          # 6.25% boundary reserve
├── boundary_125.yaml           # 12.5% boundary reserve (默认)
├── boundary_25.yaml            # 25% boundary reserve
├── run_boundary_sensitivity.sh # 批处理脚本（串行跑3个配置）
├── start_route1.sh             # 启动器（带日志和PID）
├── README_phase1.md            # 本文档
└── runs/                       # 输出目录
    ├── boundary_0625_seed1_*/
    ├── boundary_125_seed1_*/
    └── boundary_25_seed1_*/
```

## 启动方式

### 方式1: 前台运行（实时查看输出）
```bash
cd /root/autodl-tmp/OmniSAM-main/experiments/route1
./run_boundary_sensitivity.sh
```

### 方式2: 后台运行（推荐）
```bash
cd /root/autodl-tmp/OmniSAM-main/experiments/route1
nohup ./start_route1.sh > route1_phase1.log 2>&1 &
echo $! > route1_phase1.pid
```

### 查看进度
```bash
# 查看日志
tail -f /root/autodl-tmp/OmniSAM-main/experiments/route1/route1_phase1_*.log

# 查看进程
ps aux | grep run_adaptation

# 查看 GPU 占用
nvidia-smi
```

### 停止实验（如需要）
```bash
# 找到 PID
cat /root/autodl-tmp/OmniSAM-main/experiments/route1/route1_phase1_*.pid

# 杀死进程
kill -9 <PID>
```

## 预期结果

### 成功标准
- 任何一个配置 mIoU ≥ 60.2（接近或超过 baseline 60.22）
- BFScore 提升（> 39.46，最好接近 39.71）
- 细小类 mIoU 提升（traffic light, fence, pole）

### 决策逻辑

**如果 boundary_125 达到 60.3+**：
→ Phase 2 用 5 seeds 验证稳定性
→ 准备写入论文，作为核心创新点

**如果 3 个都在 59.8-60.1 区间**：
→ 分析 per-class 结果，看边界类是否有提升
→ 考虑调整 boundary 提取方法（Sobel → morphology）
→ 或尝试 Route 2 (Overlap-Aware)

**如果全部 < 59.8**：
→ 该方向可能不如预期
→ 转向 Route 3 (Class-Aware) 或组合策略

## 技术细节

### Boundary 提取方法（代码实现）
预计使用以下方法之一：
1. **Sobel edge detection**：`cv2.Sobel()` 或 `kornia.filters.sobel()`
2. **Morphological gradient**：`dilation - erosion`
3. **Laplacian**：二阶导数边界

### 集成点
代码修改位置（预计）：
- `model/modules/sparse_memory_retriever.py`: 新增 `mode=boundary_guarded`
- 输入：memory tokens + pred mask（从上游传入）
- 输出：selected token indices（保证总数 = quota）

## 相关文档

- `docs/innovation_analysis/innovation_overview.md` - 四个创新点完整对比
- `docs/innovation_analysis/discussion_log.md` - 讨论记录
- `docs/omnisam_innovation_literature_scan_2026.md` - 文献综述

## 实验记录

### 2026-08-25
- ✅ 创建 3 个配置文件（boundary_0625/125/25）
- ✅ 创建批处理脚本（run_boundary_sensitivity.sh）
- ✅ 创建启动器（start_route1.sh）
- ⏳ 等待 18:00 attn_topk 实验完成后启动
