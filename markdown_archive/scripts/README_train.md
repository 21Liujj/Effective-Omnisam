# 训练脚本统一使用说明

本文档用于统一 `scripts/` 下训练脚本的启动方式、日志位置与后台运行规范。

## 通用约定

- **后台运行开关**：`RUN_IN_BACKGROUND=1`
- **日志目录**：默认 `./logs`（可通过 `LOG_DIR` 覆盖）
- **日志文件**：大多数脚本默认按时间戳自动命名（可通过 `LOG_FILE` 覆盖）
- **PID 文件**：后台启动时会写入 `logs/*.pid`（可通过 `PID_FILE` 覆盖）

## 通用命令模板

1) 启动（后台）

```bash
cd /root/autodl-tmp/OmniSAM-main
RUN_IN_BACKGROUND=1 bash scripts/<script_name>.sh
```

2) 看日志

```bash
tail -f logs/<log_file>.log
```

3) 看进程

```bash
cat logs/<pid_file>.pid
ps -fp "$(cat logs/<pid_file>.pid)"
```

4) 停止进程

```bash
kill "$(cat logs/<pid_file>.pid)"
```

---

## 1) 多模态 A/B/C/D 省时实验

脚本：`scripts/run_multimodal_stage1_stage2_abcd.sh`

### 推荐启动

```bash
cd /root/autodl-tmp/OmniSAM-main
RUN_IN_BACKGROUND=1 \
SRC_CKPT=/root/autodl-tmp/omnisam_s.pth \
TGT_CKPT=/root/autodl-tmp/omnisam_s.pth \
BATCH_SIZE=4 PL_BATCH_SIZE=2 \
bash scripts/run_multimodal_stage1_stage2_abcd.sh
```

### 默认日志与 PID

- 主日志：`logs/multimodal_abcd_<timestamp>.master.log`
- 分组日志：`logs/multimodal_abcd_*.log`
- PID：`logs/multimodal_abcd.pid`

### 常用查看

```bash
tail -f logs/multimodal_abcd_*.master.log
tail -f logs/multimodal_abcd_A_generate_only.log
```

---

## 2) 多模态 A0-A4 消融实验

脚本：`scripts/run_multimodal_ablation_a0_a4.sh`

### 推荐启动

```bash
cd /root/autodl-tmp/OmniSAM-main
RUN_IN_BACKGROUND=1 \
BATCH_SIZE=4 \
bash scripts/run_multimodal_ablation_a0_a4.sh
```

### 默认日志与 PID

- 主日志：`logs/multimodal_a1_a4_<timestamp>.master.log`
- 分组日志：`logs/multimodal_a1_a4_*.log`
- PID：`logs/multimodal_a1_a4.pid`

---

## 3) DensePASS UDA（智能续跑）

脚本：`scripts/run_adaptation_uda_densepass.sh`

### 推荐启动

```bash
cd /root/autodl-tmp/OmniSAM-main
RUN_IN_BACKGROUND=1 \
SRC_CKPT=/root/autodl-tmp/omnisam_s.pth \
TGT_CKPT=/root/autodl-tmp/omnisam_s.pth \
NUM_EPOCHS=3 BATCH_SIZE=2 \
bash scripts/run_adaptation_uda_densepass.sh
```

### 默认日志与 PID

- 日志：`logs/uda_densepass_<timestamp>.log`
- PID：`logs/uda_densepass.pid`

---

## 4) DensePASS UDA（恢复训练）

脚本：`scripts/resume_adaptation_uda_densepass.sh`

### 推荐启动

```bash
cd /root/autodl-tmp/OmniSAM-main
RUN_IN_BACKGROUND=1 \
NUM_EPOCHS=2 BATCH_SIZE=2 \
bash scripts/resume_adaptation_uda_densepass.sh
```

### 默认日志与 PID

- 日志：`logs/uda_densepass_resume_<timestamp>.log`
- PID：`logs/uda_densepass_resume.pid`

---

## 5) WildPASS UDA

脚本：`scripts/run_adaptation_uda_wildpass.sh`

### 推荐启动

```bash
cd /root/autodl-tmp/OmniSAM-main
RUN_IN_BACKGROUND=1 \
SRC_CKPT=/root/autodl-tmp/omnisam_s.pth \
TGT_CKPT=/root/autodl-tmp/omnisam_s.pth \
bash scripts/run_adaptation_uda_wildpass.sh
```

### 默认日志与 PID

- 日志：`logs/uda_wildpass_<timestamp>.log`
- PID：`logs/uda_wildpass.pid`

---

## 6) 自动迭代伪标签训练（overnight）

脚本：`scripts/auto_iterative_pl_train_overnight.sh`

> 该脚本本身就支持后台运行，且默认 `RUN_IN_BACKGROUND=1`。

### 推荐启动

```bash
cd /root/autodl-tmp/OmniSAM-main
RUN_IN_BACKGROUND=1 \
TOTAL_EPOCHS=15 CHUNK_EPOCHS=2 \
BATCH_SIZE=4 PL_BATCH_SIZE=2 \
bash scripts/auto_iterative_pl_train_overnight.sh
```

### 默认日志

- 日志：`logs/auto_iterative_pl_train_<timestamp>.log`

---

## 常见问题

### Q1: 关掉终端/断开 SSH 后任务还在吗？

- 使用 `RUN_IN_BACKGROUND=1` 后，脚本会通过 `nohup` 后台运行，终端断开后任务会继续。

### Q2: 如何确认是否真的在跑？

```bash
ps -fp "$(cat logs/<pid_file>.pid)"
```

若进程存在且日志仍在增长（`tail -f` 有新输出），说明正常运行。

### Q3: 如何切换日志名？

```bash
LOG_FILE=logs/my_exp.log RUN_IN_BACKGROUND=1 bash scripts/<script_name>.sh
```

### Q4: 如何切换 PID 文件名？

```bash
PID_FILE=logs/my_exp.pid RUN_IN_BACKGROUND=1 bash scripts/<script_name>.sh
```

