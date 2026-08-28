# 今晚实验自动化执行计划

## 📅 时间线

```
15:40 (现在)  → 准备完成 ✅
18:00         → attn_topk (Task 3) 完成
18:00-24:00   → Task 4: Pano-SMB sensitivity (自动运行)
24:00-01:12   → Task 5: Route 1 Phase 1 n500 (自动接续)
01:12         → 全部完成，明早查看结果
```

## 🚀 一键启动（18:00 后执行）

```bash
ssh -p 33884 root@connect.bjb2.seetacloud.com
cd /root/autodl-tmp/OmniSAM-main
nohup ./start_tonight_experiments.sh &
```

**然后就可以关闭终端，明早查看结果！**

## 📊 实验内容

### Task 4: Pano-SMB Sensitivity (6h)
- 测试 min_tokens: 128, 256, 512
- 每个配置 ~1.5h
- 总时间: ~6h
- 输出目录: `experiments/pano_smb_stage2/runs/`

### Task 5: Route 1 Phase 1 (1.2h) ⚡ **n500 快速验证**
- 测试 boundary_reserve_ratio: 0.0625, 0.125, 0.25
- 每个配置 ~0.4h (n500 子集)
- 总时间: ~1.2h
- 输出目录: `experiments/route1/runs/`

## 🔍 监控命令

### 查看实时进度
```bash
# 总日志
tail -f /root/autodl-tmp/OmniSAM-main/tonight_experiments_*.log

# Pano-SMB 进度
tail -f /root/autodl-tmp/OmniSAM-main/experiments/pano_smb_stage2/pano_smb_*.log

# Route 1 进度（24:00后才有）
tail -f /root/autodl-tmp/OmniSAM-main/auto_chain_*.log

# GPU 占用
nvidia-smi

# 进程检查
ps aux | grep run_adaptation
```

### 查看 PIDs
```bash
cat /root/autodl-tmp/OmniSAM-main/tonight_experiments_*.pid
```

### 紧急停止（如需要）
```bash
# 找到所有 PID
ps aux | grep -E "start_tonight|auto_chain|run_adaptation" | grep -v grep

# 停止主控制器
kill $(cat /root/autodl-tmp/OmniSAM-main/tonight_experiments_*.pid)

# 停止所有训练
pkill -f run_adaptation.py
```

## 📈 明早查看结果

### Pano-SMB 结果
```bash
cd /root/autodl-tmp/OmniSAM-main
find experiments/pano_smb_stage2/runs -name "eval_results.json" -exec echo {} \; -exec cat {} \;
```

### Route 1 结果
```bash
cd /root/autodl-tmp/OmniSAM-main
find experiments/route1/runs -name "eval_results.json" -exec echo {} \; -exec cat {} \;
```

### 一键汇总
```bash
cd /root/autodl-tmp/OmniSAM-main
python aggregate_multiseed.py  # 如果适用
```

## 🎯 成功标准

### Pano-SMB
- 找到最优 min_tokens (预期 256 最好)
- 至少一个配置 > 60.1

### Route 1 (n500 快速验证)
- 看趋势，找出最好的 boundary_reserve_ratio
- 如果有配置接近 60.0+，明天做 n2000 确认
- 重点关注 BFScore 是否提升

## ⚠️ 注意事项

1. **n500 是快速验证**：结果可能波动，主要看趋势
2. **明天需要做 n2000 确认**：如果 Route 1 显示有效
3. **不要在实验运行时重启服务器**
4. **SSH 断开不影响实验**：使用了 nohup

## 🔄 如果出错

### Pano-SMB 脚本不存在
```bash
# 检查路径
ls -la experiments/pano_smb_stage2/run_pano_smb_sensitivity.sh
# 如果不存在，只运行 Route 1
cd experiments/route1
nohup ./run_boundary_sensitivity_n500.sh &
```

### Route 1 提前手动启动（不等 Pano-SMB）
```bash
cd /root/autodl-tmp/OmniSAM-main/experiments/route1
nohup ./run_boundary_sensitivity_n500.sh &
```

## 📝 相关文档

- Route 1 详细说明: `experiments/route1/README_phase1.md`
- 创新点分析: `docs/innovation_analysis/innovation_overview.md`
- 文献综述: `docs/omnisam_innovation_literature_scan_2026.md`

---

**创建时间**: 2026-08-25 15:40  
**预计完成**: 2026-08-26 01:12  
**总耗时**: ~7.2h 自动化执行
