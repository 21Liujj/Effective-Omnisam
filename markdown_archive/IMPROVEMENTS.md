# OmniSAM UDA 省时提效改进整理

本文档整理了本次对 `OmniSAM-main` 的 UDA 训练流程做的 **省时 + 提效** 改进点、对应代码位置、以及推荐的运行方式（尤其适配“复用 epoch1 伪标签继续训练多个 epoch”的场景）。

---

## 1. 复用伪标签目录（Pseudo-label Reuse）

### 背景
UDA 中伪标签生成非常耗时（你的日志中 2000 张约 1h50）。很多实验并不需要每个 epoch 都重算伪标签，可以 **生成一次，多轮复用**，将时间更多花在训练与超参尝试上。

### 改动
- **新增参数**：`--reuse_pl_dir`
  - 含义：指定一个已有的伪标签 epoch 目录（例如 `.../epoch1`），后续所有 epoch **直接复用该目录**，并跳过伪标签生成流程。

### 代码位置
- `run_adaptation.py`
  - `TrainConfig.reuse_pl_dir`
  - `main()` 中若设置了 `reuse_pl_dir`，则每个 epoch 直接 `engine.train()`/`engine.validate()`，不再进入伪标签生成逻辑。
- `adaptation/engine.py`
  - 训练时若 `args.reuse_pl_dir` 非空，会校验该目录下伪标签完整性，并以该目录构建 `target_loader`。

### 用法示例
先生成一次伪标签（得到 `epoch1`），后续复用：

```bash
python run_adaptation.py \
  --backbone sam2_s \
  --dataset Cityscapes \
  --num_epochs 5 \
  --batch_size 6 \
  --reuse_pl_dir "/root/autodl-tmp/OmniSAM-main/pd_uda_densepass/DensePASS/sam2_s/Cityscapes/epoch1"
```

---

## 2. 动作 C：target `loss_unsup` 加“置信度掩码”（只学高置信像素）

### 背景
target 域无监督训练依赖伪标签。伪标签噪声常集中在边界/远处/小物体等区域。若对所有像素无差别监督，容易“确认偏差”并拖累 mIoU。

### 改动
在训练时对 target 伪标签做像素级过滤：
- 对每个像素计算 `conf_t = max(softmax(pred_t))`
- 仅保留 `conf_t > conf_thr` 的像素参与无监督损失
- 其余像素将伪标签置为 `ignore_index`（默认 255），在 CE 中被忽略

### 新增参数
- `--conf_thr`（默认 0.85）：max-softmax 过滤阈值
- `--ignore_index`（默认 255）：被过滤像素设置为该值（loss 忽略）

### 代码位置
- `run_adaptation.py`
  - `TrainConfig.conf_thr`
  - `TrainConfig.ignore_index`
- `adaptation/strategies.py`
  - `WithMemStrategy.forward(...)` 中构造 `frame_pl_t_f`，将低置信像素置为 `ignore_index`
  - `loss_unsup = CE(pred_t, frame_pl_t_f)`
  - 同步将 `loss_proto` 的 label 输入改为过滤后的 `frame_pl_t_f`，使原型损失也更干净

### 推荐阈值策略（经验）
- 先试：`--conf_thr 0.85`
- 更稳（更干净）：`--conf_thr 0.90`
- 若训练太慢/覆盖太少再降：`--conf_thr 0.80`

---

## 3. schedule 自适应：ramp 的 `total_steps` 不再写死 1000

### 背景
原实现中 `adaptation/strategies.py` 写死 `total_steps = 1000`，用于 `alpha/beta` 的 ramp。
但你在做动作 B（减少伪标签子集、复用多 epoch）时，`len(target_loader)` 会变化，写死的 `1000` 会导致 ramp 节奏不匹配，影响收敛。

### 改动
将 `total_steps` 改为 **每个 epoch 真实的迭代数**（`len(target_loader)`），并传入策略中使用。

### 代码位置
- `adaptation/trainer.py`
  - 计算 `total_iters = len(target_loader)`
  - 调用 `mem_strategy.forward(..., total_iters)`
- `adaptation/strategies.py`
  - `total_steps = total_iters if total_iters>0 else 1000`

---

## 4. 动作 A：训练加速开关（AMP + TF32）

### 背景
在 32G 显存 GPU 上（如 5090），打开 AMP/TF32 通常可以：
- 降低显存占用（更容易把 `batch_size` 提到 4/6）
- 提升吞吐（单步更快）

### 改动与参数
- `--amp`（默认 False）：开启 AMP 混合精度（CUDA 时生效）
- `--tf32`（默认 True）：开启 TF32 matmul（CUDA 时生效）

### 代码位置
- `run_adaptation.py`
  - `TrainConfig.amp`
  - `TrainConfig.tf32`
  - `main()` 中在 CUDA 下启用 TF32（并尽量设置 matmul 精度为 high）
- `adaptation/trainer.py`
  - 若 `args.amp` 且 CUDA，创建 `torch.cuda.amp.GradScaler()`，并传入策略
- `adaptation/strategies.py`
  - 使用 `torch.cuda.amp.autocast(enabled=use_amp)` 包裹 forward/loss
  - 使用 scaler 做 `scale/backward/step/update`

### 推荐用法
（32G GPU 上建议从 `batch_size=6` 开始试）

```bash
python run_adaptation.py \
  --batch_size 6 \
  --amp --tf32 \
  ...
```

### 技术原理（可用于论文/汇报说明）

#### 4.1 AMP（Automatic Mixed Precision，自动混合精度）
- 核心思想：将前向/反向中的大部分算子切换到低精度（FP16/BF16），少数数值敏感算子保留 FP32。
- 加速来源：卷积与矩阵乘会更多使用 Tensor Core 路径，单位时间吞吐更高。
- 显存收益：中间激活与部分梯度采用半精度，显存占用下降，通常可支持更大 `batch_size`。
- 稳定性机制：通过 `GradScaler` 做动态 loss scaling，减少半精度训练中的梯度下溢风险。

#### 4.2 TF32（TensorFloat-32）
- 适用硬件：Ampere 及以上 NVIDIA GPU（如 3090/4090/5090/A100）。
- 核心思想：在保持 FP32 累加稳定性的前提下，使用 TF32 快速路径执行 matmul/conv。
- 工程效果：通常可在几乎不改变最终精度的情况下，显著提升 FP32 路径下的训练吞吐。

#### 4.3 在本项目中的实际价值
- 你的任务为高分辨率全景语义分割，单 step 计算量大、显存压力高。
- 启用 AMP + TF32 后，一般可同时获得：
  1) 单 step 更快；
  2) 显存占用更低；
  3) 可以将 `batch_size` 从 2 提高到 4/6，进一步减少每个 epoch 的迭代数。
- 对于 `used_path=2000` 的场景，`batch_size=6` 时每 epoch 迭代约为 333（`drop_last=True`），显著低于 `batch_size=2` 的约 1000 迭代。

#### 4.4 AMP 常见兼容性问题与处理
- 典型问题：自定义模块内混合了 FP16 特征与 FP32 缓冲（如 `index_add_`）会触发 dtype mismatch。
- 本项目处理方式：对原型库相关特征显式 `.float()`，保证原型更新/原型损失计算使用 FP32，兼顾稳定性与 AMP 加速收益。

---

## 5. 无卡模式下的伪标签 PNG 统计（无需 GPU）

### 背景
无 GPU 时无法跑“模型置信度统计”（需要前向推理），但仍可以快速统计伪标签 PNG 的：
- ignore(255) 占比
- 每类像素占比
- ignore 占比最高的城市/图片

### 改动
增强了 `tools/analyze_pseudolabels.py`：
- 支持 `--use_used_path`：仅统计 `used_path.txt` 对应的伪标签文件
- 支持 `--topk_images`：打印 ignore 占比最高的图片列表
- 保留 `--max_files`：支持抽样快速估计

### 用法示例（你的 epoch1）

```bash
python tools/analyze_pseudolabels.py \
  --pl_dir "/root/autodl-tmp/OmniSAM-main/pd_uda_densepass/DensePASS/sam2_s/Cityscapes/epoch1" \
  --use_used_path \
  --max_files 200 \
  --topk_cities 10 \
  --topk_images 20
```

---

## 6. 有卡环境下的 target 置信度覆盖统计（建议在 GPU 上跑）

仓库内 `tools/analyze_target_confidence.py` 已修复导入问题并支持 `--class_set`。
该脚本通过前向推理统计：
- `kept_ratio(conf>thr)`：高置信像素覆盖率
- `pl_ignore_ratio`：伪标签 PNG 中 ignore 占比
- 按类聚合的 mean_conf 与 kept%

建议在 CUDA 环境运行（CPU 易超时/被 kill）。

---

## 7. 推荐训练配方（示例）

### 复用 epoch1，训练 5 个 epoch（32G GPU，建议）

```bash
python run_adaptation.py \
  --backbone sam2_s \
  --dataset Cityscapes \
  --batch_size 6 \
  --num_epochs 5 \
  --reuse_pl_dir "/root/autodl-tmp/OmniSAM-main/pd_uda_densepass/DensePASS/sam2_s/Cityscapes/epoch1" \
  --amp --tf32 \
  --conf_thr 0.85
```

### 阈值对照（建议跑一组 0.90）

```bash
python run_adaptation.py \
  ... \
  --conf_thr 0.90
```

---

## 8. 实测运行状态记录（RTX 5090, 32G）

以下为本次复现中 `reuse epoch1 + amp + tf32 + conf_thr=0.85 + batch_size=4` 的实测 `nvidia-smi` 观测：

- `Memory-Usage`: `29321MiB / 32607MiB`
- `GPU-Util`: `96%`
- `Power`: `504W / 575W`
- 单进程训练（PID 对应 `python`）显存占用约 `29312MiB`

### 结论解读
- GPU 计算利用率接近满载，说明训练吞吐优化已基本到位。
- 显存占用接近上限但仍可稳定运行，当前配置是“高利用率 + 可持续”区间。
- 在该配置下继续提升 `batch_size`（尤其回到 6）存在较高 OOM 风险。

### 实操建议
- 建议优先保持 `batch_size=4` 完整跑通 epoch2~epoch6，先拿稳定结果。
- 若需进一步尝试提速/提 batch，建议按小步策略：
  1) 先将 `conf_thr` 提高到 `0.90`（常见可略降有效像素量）；
  2) 再试 `batch_size=5`，不建议直接跳到 6。
- 可选显存碎片优化环境变量：
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 9. 伪标签生成：提速与提质方案（规范版）

本节聚焦“伪标签生成耗时长（2000 张约 1h+）”与“伪标签噪声影响后续 mIoU”的双重问题，给出可落地的优化路径。

### 9.1 总体原则
- **算力优先分配到难样本**：不要对所有样本使用同等高成本策略。
- **先快后精**：先得到可用伪标签，再做局部精修。
- **阶段性重打而非每轮重打**：每 3~5 epoch 重打一次通常性价比更高。
- **以验证 mIoU 曲线为最终判据**：伪标签统计是辅助，mIoU 才是目标。

### 9.2 提速策略（工程优先级从高到低）

#### A. 分阶段生成数量（最易落地）
- 第一阶段先生成 `500~800` 张，快速进入训练。
- 第二阶段再补齐（或仅补难样本）。
- 收益：显著缩短“首轮可见结果”的等待时间。

#### B. 双 pass 改为“默认单 pass + 难样本重算”
- 先用低成本配置（单 pass）全量生成。
- 对低置信/高不确定样本再做高成本重生成（双 pass 或更细滑窗）。
- 收益：在接近质量的情况下，明显降低总生成时长。

#### C. 提高伪标签阶段 batch / I/O 吞吐
- 生成阶段单独调大 batch（不影响训练稳定性）。
- 合理提升 dataloader worker 与 pin_memory（受 CPU/磁盘影响）。
- 收益：纯吞吐加速，风险低。

#### D. 伪标签生成阶段启用 AMP/TF32 推理加速（低风险高收益）
> 对应本次选择的“方案2”：不减少样本、不改变生成策略，仅让前向推理更快。

- 思路：伪标签生成本质是对目标域数据做大规模前向推理，时间开销大。训练阶段启用 AMP/TF32 的同时，伪标签生成阶段也应启用 AMP/TF32，以利用 Tensor Core 提升吞吐、降低显存峰值。
- 风险：理论上半精度会带来数值差异，但在推理阶段通常影响很小；本项目在落盘前对 logits 显式转为 `float32`，进一步降低类型/稳定性风险。
- 适用：CUDA 可用且 `--amp` 打开时自动生效；TF32 由 `--tf32` 控制并在程序启动时全局开启。

**代码位置**
- `adaptation/pl_generator.py`
  - 在伪标签生成的模型前向处加入 `autocast`（仅 CUDA + amp 时启用）
  - 在写入 numpy 前使用 `output.detach().float().cpu().numpy()`，避免保存 half 引发的类型问题
- `adaptation/engine.py`
  - 调用 `PseudoLabelGenerator.run(..., amp=args.amp)`，使 `--amp` 同时控制伪标签生成阶段

**使用方式**
- 只要你的启动命令（或自动脚本）包含 `--amp` 且 CUDA 可用，伪标签生成会自动启用 AMP 推理加速。
- 例如自动脚本中：
  - `AMP=1 TF32=1 bash scripts/auto_iterative_pl_train_overnight.sh`

### 9.3 提质策略（对 mIoU 更敏感）

#### D. 教师迭代提纯（推荐）
- 用当前 best adaptation ckpt 作为 teacher 重新生成伪标签。
- 再用新伪标签继续训练 2~4 epoch。
- 建议节奏：每 3~5 epoch 重打一次，不必每 epoch 重打。

##### 说明：它和“训练提速优化”是什么关系？
- **teacher 重打伪标签**只影响“伪标签生成阶段用哪个模型推理”，核心目的是 **提高伪标签质量**。
- 之前的训练优化（AMP/TF32、batch、conf-mask、schedule 自适应、复用伪标签与续训脚本）主要影响 **训练阶段的速度/稳定性/收敛节奏**。
- 两者是正交的：更换 teacher 打伪标签不会抵消训练提速；相反，**更高质量伪标签 + 更稳的训练策略**通常更容易带来 mIoU 提升。

##### 实践含义（结合你的实验现状）
- 你当前的第一轮 UDA 训练获得了更好的 best ckpt（例如 `mIoU=60.36`），相比论文/原始设置（例如 `60.29`）更强。
- 下一步使用该 `best ckpt` 作为 teacher 重新生成 2000 张伪标签，相当于做一次“自训练迭代提纯”：
  1) Round-1 teacher：原始/论文 ckpt → 生成伪标签 → 训练得到更强模型；
  2) Round-2 teacher：Round-1 的 best ckpt → 重打伪标签 → 再训练继续提升。
- 这一步在 UDA/自训练中非常常见，通常比“在同一份旧伪标签上继续硬训”更有效。

##### 为什么工程上常用“重打一次伪标签 → 复用训练若干轮”
该节奏在 UDA/自训练工程实践中非常普遍，原因主要来自 **计算效率** 与 **优化稳定性** 的双重约束：

1) **效率（wall-clock）角度**  
伪标签生成本质是对目标域数据做一次（近）全量推理，成本高且随样本数线性增长。在本项目中，生成 2000 张伪标签可达 1h+。若每个 epoch 都重打伪标签，总耗时会被推理阶段主导，性价比极低。

2) **稳定性（优化/收敛）角度**  
训练过程是在拟合“监督信号”（伪标签）所定义的目标函数。若每个 epoch 都重打伪标签，监督信号在频繁变化，梯度方向会随之摆动，导致收敛更不稳定、指标更易震荡，也更难诊断超参效果。固定一份伪标签训练若干轮，可使短周期目标更稳定，曲线更可解释。

3) **收益（边际效应）角度**  
相邻 epoch 的 teacher 通常变化有限，频繁重打的伪标签差异也有限，带来的质量提升往往不足以覆盖其时间成本。更实用的策略是让 teacher 在 3~5 个 epoch 内产生“可感知提升”后再重打，从而获得更高的收益/成本比。

##### 推荐重打频率（经验）
- **默认建议**：每 **3~5 个 epoch** 重打一次伪标签。  
- 若出现以下信号，优先重打而不是继续硬训：  
  - 最近 2~3 个 epoch `val mIoU` 持续下降或进入平台且波动变大；  
  - 小类（person/traffic sign/light）长期无改善；  
  - 训练损失下降但验证不升（疑似对伪标签噪声过拟合）。

#### E. 类别自适应阈值（class-wise threshold）
- 小类（如 person/traffic light/sign）阈值适度放宽，大类适度收紧。
- 目标：缓解小目标被过度过滤导致的召回损失。

#### F. 轻量后处理清噪
- 连通域过滤、边界平滑、类别先验约束。
- 目标：减少明显孤立噪声与伪标签毛刺。

### 9.4 推荐执行路线（当前项目）

#### 路线 1（时间受限，先拿结果）
1) 先复用现有 epoch1 伪标签跑通（当前已在执行）  
2) 记录 5 个 epoch 的 mIoU 曲线  
3) 若进入平台，使用 best ckpt 重打一版伪标签，再训 2~3 epoch

#### 路线 2（兼顾提速与质量）
1) 快速全量生成（低成本配置）  
2) 基于不确定性筛选底部 20% 难样本  
3) 仅对难样本高质量重生成  
4) 用混合伪标签继续训练

### 9.5 评估与对照建议（用于论文/汇报）
- 对照组：
  - Baseline：原始全量双 pass 生成 + 常规训练
  - Fast：单 pass / 子集先训
  - Fast+Refine：难样本重算
  - Iterative Teacher：best ckpt 重打伪标签
- 统计指标：
  - 生成时长（总时长、单图时长）
  - 伪标签质量（ignore 比例、类别覆盖、异常样本比例）
  - 训练效果（best mIoU、前 3 epoch 提升斜率、收敛稳定性）

### 9.6 实践判断规则（是否该重打伪标签）
- 若最近 2~3 个 epoch `val mIoU` 增量 < 0.1 且波动变大：优先重打伪标签。
- 若仍持续提升（每 epoch +0.2 以上）：可继续当前伪标签再跑若干 epoch。
- 若小类 mIoU 长期不上升：优先尝试类别自适应阈值或教师重打。

---

## 10. 多模态融合方案：RGB + 文本先验（详细技术原理）

本节给出“先不改代码、先定方法学设计”的完整方案。目标是将文本语义引入当前 UDA 全景分割流程，在不依赖额外传感器的前提下，提升跨域鲁棒性与长尾类表现。

### 10.1 任务定义与动机

#### 问题背景
- 当前管线属于单模态：`RGB -> 语义分割`。
- 在合成到真实域迁移中，视觉外观（颜色、纹理、曝光）变化显著，易导致语义错配。
- 小目标（person, traffic sign/light）与边界区域最易受噪声影响。

#### 核心动机
- 文本天然提供“类别语义锚点”，可缓解“外观变了但语义不变”的域偏移问题。
- 因此先从 `RGB + 文本` 入手，是成本最低、最容易出结果的多模态路线。

### 10.2 文本信息到底是什么

#### 不是逐图描述文本
- 本方案不要求每张图有 caption（例如“左边有车右边有树”）。
- WildPASS / DensePASS 本身也没有这类逐图文本标注。

#### 是类别语义文本
- 使用类别名及其模板句作为文本模态输入。
- 例如（13 类）：
  - `road`, `sidewalk`, `building`, `wall`, `fence`, `pole`,
  - `traffic light`, `traffic sign`, `vegetation`, `terrain`, `sky`, `person`, `car`
- 模板句如：`a photo of a road in urban driving scene`。

### 10.3 CLIP 冻结文本分支的原理

#### 双塔对齐思想
- CLIP 由图像编码器和文本编码器组成，通过对比学习将图文映射到同一语义空间。
- 匹配图文相似度高，不匹配相似度低。

#### 为什么“冻结 CLIP 文本编码器”
- 预训练语义先验强，迁移时稳定。
- 避免小数据场景过拟合。
- 显存与计算开销更低，工程更可控。

### 10.4 模型级方案设计（最小可行版）

#### 模块 1：类别文本原型库（离线构建）
1) 为每个类别写多条 prompt（3~8 条）。  
2) 用冻结文本编码器提取向量。  
3) 对同类 prompt 取均值得到类别原型 `T_c`。  
4) 保存为固定原型库，训练期间不更新（第一版建议）。

#### 模块 2：像素-文本相似度头（在线训练）
- 从分割网络 decoder 取像素特征图 `F ∈ R^{B×D×H×W}`。
- 归一化后与 `T_c` 做余弦相似度，得到文本语义 logits `L_txt`。
- 这一步可以理解为“每个像素与每个类别文本语义的匹配分数”。

#### 模块 3：融合策略（建议先后融合）
- 视觉分支 logits：`L_vis`
- 文本分支 logits：`L_txt`
- 融合输出：`L_fuse = L_vis + λ * L_txt`
- 其中 `λ` 可先固定（如 0.1~0.5），后续再升级为可学习门控。

### 10.5 损失函数设计（与现有 UDA 兼容）

保持你当前损失主干不变，仅新增文本对齐项：

- 现有：
  - `L_sup_src`（源域监督）
  - `L_unsup_tgt`（目标域伪标签监督，已含 conf-mask）
  - `L_proto`（原型约束）
- 新增：
  - `L_txt_align`：视觉分支与文本分支的一致性约束（KL/CE 均可）

总损失可写为：

`L_total = L_sup_src + a * L_unsup_tgt + b * L_proto + c * L_txt_align`

其中 `c` 建议从小到大（如 0.05 起步，稳定后到 0.2）。

### 10.6 训练流程建议（分阶段）

#### Stage A：单模态基线
- 先用当前已优化的 UDA 流程训练到稳定（你当前正在做）。

#### Stage B：引入文本先验（冻结）
- 加入 `L_txt_align`，小权重启动。
- 观察前 2~3 epoch 收敛稳定性与 mIoU 变化。

#### Stage C：迭代提纯
- 若曲线稳定上升，用 best ckpt 重打伪标签后继续训练（沿用第 9 节策略）。

### 10.7 提示词工程（Prompt Engineering）规范

#### 多模板集成（推荐）
- 每类至少 3 条模板，建议 5 条。
- 使用“场景 + 类别”形式：`a panoramic urban scene with {class}`。
- 同义词可适度引入（如 `car/automobile`），但避免语义漂移。

#### 模板质量控制
- 避免过长复杂句子，保持模板语义明确、结构统一。
- 若某类长期低性能，可专门优化该类模板。

### 10.8 预期收益与风险

#### 预期收益
- 跨域鲁棒性提升（外观变化下语义不易漂移）。
- 长尾类和小目标稳定性提升概率更高。
- 无需额外硬件模态，工程投入较低。

#### 主要风险
- 对 prompt 敏感，模板不佳会削弱收益。
- 文本分支权重过大可能干扰视觉细粒度边界。
- 仅文本先验对几何边界帮助有限（后续可扩展深度模态）。

### 10.9 最小实验矩阵（先验证方向）

建议先做 3 组，快速判断是否值得深入：

1) Baseline（当前 UDA 优化版，无文本）  
2) Baseline + 文本后融合（`λ=0.1`，`c=0.05`）  
3) Baseline + 文本后融合（`λ=0.3`，`c=0.1`）

观察指标：
- best mIoU（总体）
- 小类 mIoU（person/traffic sign/light）
- 收敛稳定性（前 3 epoch 波动）

### 10.10 论文表述建议（可直接改写）

可将该方法贡献描述为：
- 提出一个轻量级 RGB-Text 语义对齐框架，在不增加传感器依赖的条件下增强 UDA 分割鲁棒性。  
- 通过冻结文本编码器与类别文本原型库，将跨域稳定的语义先验注入像素级判别过程。  
- 通过视觉-文本一致性约束与现有伪标签训练协同优化，提升长尾类与跨域场景表现。

