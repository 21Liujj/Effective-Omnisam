# OmniSAM 多模态技术方案（RGB + Text，含 OPS 升级路线）

本文档给出一份完整、可执行、可迭代的技术方案，目标是在现有 OmniSAM UDA 流水线上实现并逐步升级多模态能力。

---

## 1. 目标与范围

### 1.1 总目标

在不增加额外传感器的前提下，通过文本语义先验增强跨域语义分割表现，重点提升：

- 跨域鲁棒性（外观变化下语义稳定）
- 长尾/小目标类别表现（person / traffic sign / traffic light）
- 训练过程稳定性（避免过拟合伪标签噪声）

### 1.2 当前范围

优先做 `RGB + Text`，暂不引入深度/点云。  
原因：当前数据管线为 RGB + label，文本模态是最小工程改动、最快验证收益的路线。

**可选后续**：生成式 **扩散模型** 与本文主链路的关系、接入角色与阶段划分见 **第 13 节**（默认不改变推理延迟门禁）。

---

## 2. 方法总览（通俗版）

原始模型只有视觉分支输出 `L_vis`。  
多模态版本增加文本分支输出 `L_txt`，再融合为：

`L_fuse = L_vis + lambda_txt * L_txt`

其中：

- `L_vis`：像素看起来像哪个类
- `L_txt`：像素语义上更接近哪个类（基于文本语义原型）
- `L_fuse`：最终用于监督/预测的 logits

训练时新增一致性约束 `L_txt_align`，让视觉分支与文本分支分布更一致，降低互相冲突。

---

## 3. 文本语义原型在本方案中的使用方式

### 3.1 为什么采用“冻结文本编码器”

本方案不微调文本编码器主干参数。文本编码器仅用于提供稳定语义空间：

- 文本编码器用于生成类别语义原型
- 参数固定，不参与反向传播

### 3.2 离线 vs 在线

- **离线（当前已实现）**：先生成 `text_prototypes.pt`，训练时只加载原型，不实时跑文本编码器
- **在线（后续可选）**：训练过程中动态调用/更新文本侧表示（工程复杂、开销更高）

当前建议采用离线方案先拿稳定收益。

后续是否引入新的图文融合支路，统一按 **第 11 节** 重评估流程执行。

---

## 4. 已实现的工程落地（代码对齐）

以下内容已在仓库中落地：

1. 文本原型构建脚本：`tools/build_text_prototypes.py`
2. 文本对齐模块：`model/text_align.py`
   - `PixelTextAlignHead`
   - `kl_align_loss`
3. 模型融合逻辑：`model/model.py`（`OmniSAM_adapt` 支持 `L_fuse` 与 `return_aux`）
4. 原型加载注入：`utils/build.py`（通过 `--text_proto_path`）
5. 训练损失接入：`adaptation/strategies.py`（新增 `loss_txt_align` + ramp）
6. 训练参数开关：`run_adaptation.py`（新增文本相关参数）
7. 验证脚本：`validation_muti.py`（支持多 ckpt 对比 + 文本分支开关）

---

## 5. 数学定义与损失设计

### 5.1 文本原型

对每个类别 `c`，构造多条模板句并编码为向量，均值后归一化：

`T_c = normalize(mean(encode_text(prompt_c_i)))`

### 5.2 文本 logits

decoder 像素特征 `F in R^{B x D x H x W}` 经投影到文本维度后，与各类原型余弦相似得到：

`L_txt in R^{B x C x H x W}`

### 5.3 融合输出

`L_fuse = L_vis + lambda_txt * L_txt`

### 5.4 总损失

`L_total = L_sup_src + a * L_unsup_tgt + b * L_proto + c * L_txt_align`

其中：

- `L_sup_src`：源域监督
- `L_unsup_tgt`：目标域伪标签监督（含置信度 mask）
- `L_proto`：原型约束（现有）
- `L_txt_align`：视觉-文本一致性（KL）

`c` 建议使用从小到大的 ramp（避免早期不稳定）。

---

## 6. 训练流程（标准执行）

### Stage A：基线确认（无文本）

目的：固定可复现基线，作为后续对照。

- `--text_proto_path ""`
- `--lambda_txt 0`
- `--weight_txt_align 0`

### Stage B：引入文本先验（轻权重）

目的：验证文本模态有效性。

推荐起步：

- `lambda_txt = 0.1`
- `weight_txt_align = 0.05`
- `txt_tau = 0.07`
- `txt_align_tau = 1.0`

### Stage C：增强权重与稳定性检查

若 Stage B 稳定提升，再试：

- `lambda_txt = 0.3`
- `weight_txt_align = 0.1`

若波动明显，回退到小权重。

---

## 7. 标准实验矩阵（必须做）

固定同一 teacher / 同一数据切分 / 同一验证口径，做三组：

1. Baseline（无文本）
2. Baseline + Text（`lambda_txt=0.1`, `weight_txt_align=0.05`）
3. Baseline + Text（`lambda_txt=0.3`, `weight_txt_align=0.1`）

---

## 8. 命令模板

### 8.1 安装依赖

```bash
cd /root/autodl-tmp/OmniSAM-main
pip install -r requirements.txt
```

### 8.2 生成文本原型（一次）

```bash
cd /root/autodl-tmp/OmniSAM-main
python tools/build_text_prototypes.py \
  --dataset Cityscapes \
  --class_set 13 \
  --text_model ViT-B-32 \
  --text_pretrained openai \
  --out ./assets/text_prototypes_cityscapes13_vitb32.pt
```

### 8.3 多模态训练（示例）

```bash
cd /root/autodl-tmp/OmniSAM-main
python run_adaptation.py \
  --backbone sam2_s \
  --dataset Cityscapes \
  --class_set 13 \
  --num_epochs 1 \
  --batch_size 2 \
  --text_proto_path ./assets/text_prototypes_cityscapes13_vitb32.pt \
  --lambda_txt 0.1 \
  --txt_tau 0.07 \
  --weight_txt_align 0.05 \
  --txt_align_tau 1.0
```

### 8.4 文本模态验证（多 ckpt 对比）

```bash
cd /root/autodl-tmp/OmniSAM-main
python validation_muti.py \
  --backbone sam2_s \
  --dataset DensePASS \
  --class_set 13 \
  --use_mem_bank \
  --data_root data \
  --ckpt /path/to/ckpt1.pth \
  --ckpt /path/to/ckpt2.pth \
  --ckpt_names model1,model2 \
  --text_proto_path ./assets/text_prototypes_cityscapes13_vitb32.pt \
  --lambda_txt 0.1 \
  --txt_tau 0.07 \
  --metrics_json ./logs/val_text_cmp.json \
  --log_file ./logs/val_text_cmp.log
```

---

## 9. 指标与结论口径

主指标：

- mIoU
- mAcc
- PixAcc
- FWIoU
- mDice

重点观察：

- 小类 IoU：person / traffic sign / traffic light
- 前 3 epoch 收敛稳定性
- 与 baseline 的稳定差值（避免单次偶然）

建议判定：

- 若 mIoU 与小类指标稳定提升（例如 >= 0.2），进入下一阶段
- 若边界类明显退化，优先降低 `lambda_txt` 和 `weight_txt_align`

---

## 9.5 推理加速与模型压缩（GPU 5090，mIoU ≥ 60.3 约束）

本章目标：在保持 `DensePASS/Cityscapes_13cls` 验证口径下 **mIoU ≥ 60.3** 的前提下，同时优化：

- **单张延迟**（p50/p95，单位 ms）
- **吞吐**（batch 推理 FPS / images/s）

> 说明：推理性能优化与训练策略解耦。建议先冻结一个“可交付 ckpt”（例如当前 best 约 60.3+），后续所有加速/压缩都以该 ckpt 为 teacher/基线。

### 9.5.1 统一性能测量口径（先做这个，否则容易“越优化越糊涂”）

**必须固定的变量**：

- 输入分辨率与预处理（resize/normalize）
- 推理模式（是否滑窗、窗口大小/步长、是否双 pass）
- 后处理（argmax / mask merge / overlap 可视化必须从测速链路剥离）
- batch size（分别测：BS=1 的延迟、BS>1 的吞吐）

**建议报告格式**（每次优化都要给出）：

- **精度**：`mIoU/mAcc/PixAcc/FWIoU/mDice`（用 `validation_muti.py` 或同等口径）
- **延迟**（BS=1）：warmup 50 次 + 正式 200 次，报告 `p50/p95`（ms）
- **吞吐**（BS=8/16）：warmup 20 次 + 正式 100 次，报告 `images/s`
- **资源**：显存峰值（MiB）、显卡功耗/频率（可选）

**最低验收线**：

- `mIoU >= 60.3`
- 若量化/剪枝导致 `mIoU < 60.3`：必须提供回滚方案（见 9.5.6）

### 9.5.2 推理加速优先级（低风险 → 中风险 → 高风险）

#### A. 低风险（优先做，通常不掉精度）

1) **FP16/BF16 推理**

- 优先走 **TensorRT FP16**；若先在 PyTorch 上验证，可用 autocast/half。
- 风险点：个别算子 FP16 数值不稳（一般语义分割较少见），用 `mIoU` 回归测试兜底。

2) **`torch.compile`（PyTorch 2.x）/ kernel fusion**

- 对纯 PyTorch 推理链路，可尝试 `torch.compile(mode="max-autotune")`（具体 mode 以你环境为准）。
- 优先把“前处理/后处理”从 compile 范围剥离，只编译模型 forward。

3) **减少 Python 开销**

- dataloader、可视化、日志、保存 overlap 等完全不进入 benchmark。
- 推理脚本里把张量创建、device 迁移提前做，循环内只做 forward + 同步计时。

4) **CUDA Graphs（追求更低延迟时）**

- 当输入形状固定、控制流稳定时，用 CUDA Graphs 录制能明显降低 launch overhead。
- 适配成本：需要固定 batch/shape，或维护少量 shape bucket。

#### B. 中风险（可能小幅掉精度或需要更多工程）

5) **ONNX 导出 + TensorRT 引擎**

- 目标：把核心推理放进 TRT，拿到更强的融合与调度优化。
- 关键约束：动态 shape（滑窗/多尺度）会增加工程复杂度；建议先做固定输入或固定窗口的 TRT。

6) **结构化剪枝（channel pruning）+ 轻量微调**

- 做法：对 decoder/head 或 backbone 部分通道做结构化剪枝，保证 TensorRT/硬件友好。
- 建议从“最不敏感”的模块开始（通常是 head/adapter 层），每次剪枝比例从 10% 起。

7) **知识蒸馏（Teacher=当前 60.3+ ckpt）**

- 做法：student 结构更小（宽度/层数/记忆 token 数减少等），训练时最小化 student 与 teacher 的 logits/KL + 真值/伪标签监督。
- 优势：在同等速度目标下，蒸馏通常比硬剪枝更稳。

#### C. 高风险（速度收益大，但最容易过线失败）

8) **INT8 量化（PTQ/QAT）**

- PTQ：最快落地，但对分割精度更敏感；建议优先做 per-channel weight + FP16 激活混合策略。
- QAT：工程与训练成本更高，但更可能把 mIoU 拉回到 ≥60.3。

### 9.5.3 “两条跑道”并行：先拿速度，再拿体积

建议把工作拆成两条跑道，避免互相干扰：

- **跑道 1（加速）**：FP16/TRT/compile/CUDA graphs → 目标是把延迟/FPS 做到可交付
- **跑道 2（压缩）**：剪枝/蒸馏/量化 → 目标是把模型 size/显存/算力需求压下去

每次只改一个主变量，且都必须跑 `mIoU` 回归。

### 9.5.4 推荐落地顺序（可执行 checklist）

**Step 0：冻结基线**

- 固定一个 ckpt（best/可交付），并用 `validation_muti.py` 记录一份 JSON/日志作为基线。

**Step 1：PyTorch FP16 推理回归**

- 先确认 FP16 推理下 `mIoU >= 60.3`（若掉点，先别急着做 TRT/量化）。

**Step 2：TensorRT FP16**

- 把“模型 forward”导出到 ONNX，再构建 TRT FP16 引擎。
- 先做固定输入形状版本（例如固定窗口），跑通后再考虑动态 shape。

**Step 3：吞吐与延迟分开优化**

- BS=1：优先 CUDA graphs / 减少同步 / 减少后处理开销
- BS=8/16：优先 TRT + 合理 batch + pinned memory + 流水线

**Step 4：结构化剪枝（小步快跑）**

- 每次剪 10% 左右，跑一次精度回归与性能回归。
- 只要过线且速度收益明显，就固化一个“剪枝档位”。

**Step 5：蒸馏到更小 student（若剪枝不够）**

- teacher 固定为 best 60.3+ ckpt；student 尽量结构简单、推理友好。

**Step 6：INT8（最后一步）**

- 先 PTQ 验证可行性；若精度不够，再考虑 QAT。

### 9.5.5 风险点与对策（最常见踩坑）

- **性能看起来提升了，但精度回归没跑**：必须把 `mIoU >= 60.3` 写成门禁。
- **测速包含了可视化/保存 overlap**：这些极慢，会掩盖真正的推理速度。
- **滑窗/双 pass 导致 TRT 动态 shape 很麻烦**：先固定窗口形状做 TRT，必要时做 shape bucket。
- **batch 变大吞吐上去但延迟变差**：分开交付两个 profile（实时 vs 离线吞吐）。
- **量化掉精度**：优先混合精度（INT8+FP16）、per-channel、校准集覆盖长尾场景；必要时 QAT。

### 9.5.6 回滚与交付策略（保证“永远有一个能用的版本”）

建议同时维护 3 个可交付档位：

- **Gold（精度优先）**：原始 best ckpt（mIoU 最高，速度一般）
- **Fast（速度优先）**：TRT FP16 / compile 版本（mIoU ≥ 60.3）
- **Tiny（体积优先，可选）**：剪枝/蒸馏/量化版本（mIoU ≥ 60.3，若做不到就不交付）

只要某条压缩路线掉到 `<60.3`，就立刻回退到上一个“过线档位”，继续下一条路线，不在死胡同里消耗时间。

---

## 10. 常见问题与排查

### 10.1 还没训练多模态，直接开 lambda 有意义吗

没有结论意义。  
应先训练多模态 ckpt，再按同口径验证。

### 10.2 旧纯视觉 ckpt 开文本后波动

属于常见现象：文本头未经过多模态训练。  
建议只把旧 ckpt 作为初始化，再进行多模态微调。

### 10.3 原型维度或类别数不匹配

检查：

- `class_set`（13/19）是否一致
- `text_prototypes` 的 `C` 是否等于当前 `num_classes`

### 10.4 OMP 警告

若出现 `Invalid value for environment variable OMP_NUM_THREADS`，确保环境变量为正整数：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

## 11. 多模态路线重评估（不含特定预训练语义模型）

本章用于在保留当前 V1 能力的前提下，重新评估后续技术路线。原则是先证明“文本分支是否有稳定收益”，再决定是否引入更复杂的图文融合模块。

### 11.1 重评估目标

1. 判定文本分支增益是否真实、可复现、可解释。  
2. 区分增益来源：语义先验、对齐损失、还是仅训练正则。  
3. 在单卡 32GB 预算下，筛选可持续迭代的模块。

### 11.2 分层实验设计

#### A 层：有效性（先判生死）

- A0：纯视觉 baseline。  
- A1：当前 V1（`L_fuse + L_txt_align`）。  
- A2：打乱类别原型（负对照）。  
- A3：关闭 `L_txt_align`（仅保留 `L_fuse`）。  
- A4：`lambda_txt=0` 但保留 `L_txt_align`（验证正则效应）。

#### B 层：轻量升级（仅在 A 层成立后）

- B1：原型侧轻量适配（Linear/LN/温度标定）。  
- B2：在线文本编码 + 低秩适配（可选，默认关闭）。  
- B3：单尺度视觉注入（门控相加）。  
- B4：双尺度视觉注入。  
- B5：末层轻量 Cross-Attention（可选）。

#### C 层：稳健性

- 固定最佳 1～2 个配置做多 seed（建议 >=3）。  
- 报告均值、方差、训练时间与显存峰值。  
- 记录小类与总体指标是否同向变化。

#### D 层：成本门禁

- 训练：峰值显存、每 iter 耗时、每 epoch 总耗时。  
- 推理：按第 9.5 节口径报告 p50/p95 与吞吐。  
- 若性能收益不足以覆盖成本，回退上一档位。

### 11.3 验收标准（DoD）

| 条目 | 通过条件 |
|------|----------|
| 有效性 | 相对 A0 的 mIoU 与关键小类有稳定正增益 |
| 可解释性 | A2 明显退化；A3/A4 能解释收益来源 |
| 稳健性 | 多 seed 波动可接受，无偶然单次峰值依赖 |
| 可部署性 | 满足第 9.5 节性能门禁，且有回滚方案 |

### 11.4 执行顺序

先 A 层 → 再 B 层 → 最后 C/D 层。  
禁止同一轮同时改动：文本权重、伪标签阈值、视觉注入结构。

---

## 11.5 文本模态 V1 到 V2 的方案改进记录（2026-04）

本节用于正式记录当前 Text-Conditioned OmniSAM 的实验演进结论，以及据此确定的下一阶段实现方向，避免后续再次把“工程是否接通”和“真实语义是否有效”混为一谈。

### 11.5.1 V1：`logit_add late fusion` 已完成的验证

当前 V1 结构为：

```text
OpenCLIP class text prototypes
-> trainable text adapter / projector
-> text logits
-> final_logits = visual_logits + alpha * text_logits
```

围绕该结构，已完成以下关键结论：

1. 文本原型三种模式均已打通：
   - `real`
   - `shuffle`
   - `random`
2. 已补齐训练与日志侧证据：
   - `fusion_alpha`
   - `text_scale`
   - `text_adapter_grad_norm`
   - `fusion_alpha_grad_norm`
   - `text_dynamics.jsonl`
3. 已确认 `alpha_init=0.0` 会导致文本分支早期几乎不参与训练：
   - `fusion_alpha` 仅在 `1e-6` 量级波动；
   - 文本分支对 `final_logits` 几乎没有实际贡献。
4. 当手动设置 `alpha_init=0.05 / 0.10 / 0.20` 并提高 text-branch 学习率后，已确认：
   - 文本分支有梯度；
   - `final_logits` 相对 `visual_logits` 的偏移随 `alpha` 增大而线性增大；
   - 文本支路已不再是“假接入”。

### 11.5.2 V1 当前暴露出的核心问题

虽然 V1 已证明“文本链路已接通”，但在 `fixed50` 的对照实验中仍持续出现：

```text
A1(real) ≈ A4(shuffle) ≈ A5(random)
```

这说明当前问题已经从：

```text
文本分支没学起来
```

转变为：

```text
文本分支学起来了，但当前结构仍未体现真实文本语义优势
```

更准确地说，当前 V1 更像是：

```text
class-embedding branch
```

而不是：

```text
language-semantic grounding branch
```

原因是：只要后续 projector 与 decoder 足够可训练，即使输入的是 `random` prototype，也可能被学成固定类别代码，从而使 `real / shuffle / random` 都表现出类似效果。

### 11.5.3 为什么不再继续只在 V1 上调大 `alpha`

`alpha=0.05 / 0.10 / 0.20` 的对比实验已经说明：

```text
alpha 越大，文本对 final logits 的扰动越明显
```

但同时也说明：

```text
扰动变大 ≠ 真实文本语义更强
```

因此，后续不再把“继续增大 late-fusion 权重”作为主方向。  
下一阶段应把文本从“最后加分”前移到 feature 构建阶段，即进行 **feature-level fusion**。

### 11.5.4 确定的下一阶段方案：V2 `image-to-text cross-attention`

V2 结构改为：

```text
OpenCLIP text prototypes
-> text adapter / projector
-> image-to-text cross-attention
-> enhanced decoder feature
-> 原 segmentation head
-> segmentation logits
```

形式上：

```text
Q = image tokens
K = text prototypes
V = text prototypes

F_text = CrossAttention(Q=image, K=text, V=text)
F_fused = F_visual + beta * F_text
logits = segmentation_head(F_fused)
```

这意味着：

- 不再只在 logits 末端做 `visual + alpha * text`
- 而是在 semantic decoder feature 阶段，让每个像素 token 主动从类别文本原型中获取语义补充

### 11.5.5 为什么 V2 先选 `image-to-text` 而不是 `text-to-image`

当前阶段优先采用 `image-to-text cross-attention` 的原因：

1. 输出仍为 `[B, C, H, W]`，可以直接接回现有 segmentation head；
2. 对 OmniSAM 主流程侵入最小，不破坏：
   - patch sequence
   - memory attention
   - pseudo-label generation
3. 与现有 `A0/A1/A4/A5` 负控框架兼容性最好；
4. 更适合作为 V1 的低风险升级版。

本阶段不直接跳到 `text-to-image` 的原因是：

- `text-to-image` 更接近 OpenWorldSAM 式 grounding，但工程复杂度更高；
- 其输出是 class-conditioned token，需要重新设计 dense logits 生成；
- 若第一版就同时改 attention 方向与 dense projection，失败时更难解释问题出在哪里。

因此当前建议路线为：

```text
V1: logit_add late fusion        （已验证：文本已接通，但语义优势未显化）
V2: image-to-text cross-attn     （当前主线）
V3: text-to-image cross-attn     （若 V2 仍无效，再进入）
```

### 11.5.6 V2 的关键日志与判据

为避免 V2 跑完后仍然“只看到 logits 有变化，但无法判断最终预测是否真的变了”，V2 必须重点记录以下指标：

#### 1) 预测变化率

```text
prediction_change_ratio = mean(argmax(final_logits) != argmax(visual_logits))
```

这是当前最重要指标，用于判断文本分支是否真正改变了类别预测，而不仅仅是改变了 logits 的均值。

#### 2) 按类 prediction change

优先记录以下类：

- `sidewalk`
- `wall`
- `fence`
- `pole`
- `traffic light`
- `traffic sign`
- `terrain`

用于判断文本改动是否主要发生在长尾类、边界混淆类或难分类上。

#### 3) cross-attention 本身的有效性

必须记录：

- `cross_beta`
- `cross_beta_grad_norm`
- `cross_attn_grad_norm`
- `cross_attn_mean`
- `cross_attn_std`
- `cross_attn_entropy`
- `cross_attn_max_mean`

其中：

- `cross_attn_entropy` 过高：说明像素平均看所有类，文本注意力不聚焦；
- `cross_attn_entropy` 过低但 `random` 也同样低：说明可能只是把 token 当类别代码。

#### 4) feature 改变量

必须记录：

- `visual_feature_mean/std`
- `fused_feature_mean/std`
- `feature_delta_mean`
- `feature_delta_norm`
- `feature_delta_ratio`

若 `feature_delta_ratio` 仍接近 0，则说明 feature-level fusion 实际仍未真正介入。

### 11.5.7 V2 推荐实验矩阵

#### Smoke（fixed50，1 epoch）

```text
B1_real
B4_shuffle
B5_random
```

目的：

1. 检查 shape / dtype / OOM 是否正常；
2. 检查 `cross_beta` 与 `cross_attn_grad_norm` 是否非零；
3. 检查 `feature_delta_ratio` 是否非零；
4. 检查 `prediction_change_ratio` 是否明显高于 V1；
5. 初步观察 `B1` 是否开始与 `B4/B5` 拉开。

#### 正式对照（fixed200，2~3 epoch）

```text
A0 baseline
B1 real
B4 shuffle
B5 random
```

核心判据：

- **强正向**
  - `B1_real > B4_shuffle / B5_random`
  - 且 `B1` 至少不低于 `A0`

- **弱正向**
  - 总 `mIoU` 提升不明显
  - 但小类或易混类对 `B4/B5` 有稳定优势

- **无效**
  - `B1 / B4 / B5` 继续几乎完全重合
  - 且 `prediction_change_ratio`、`feature_delta_ratio`、`cross_attn_entropy` 也无显著差异

### 11.5.8 如果 V2 仍无效，后续不再优先调 `beta`

如果在 `fixed200 + 2~3 epoch` 后仍出现：

```text
B1 ≈ B4 ≈ B5
```

则应认为问题不再是“文本没有接上”，而是：

```text
当前 static OpenCLIP prototype + projector + image-to-text cross-attn
仍不足以把真实文本语义转化成有效分割增益
```

此时后续优先级应转向：

1. 进一步限制 projector 容量，避免 random token 被轻易学成类别代码；
2. 加强 text-visual alignment loss；
3. 再评估更强的 `text-to-image cross-attention` / grounding 方案；
4. 必要时考虑 multi-template prompt 或 query-based class token 设计。

一句话总结：

> V1 已经证明“文本支路能接上”；V2 要验证“文本是否开始在特征层面体现真实语义优势”；如果 V2 仍无效，再进入更接近 OpenWorldSAM 的强 grounding 路线，而不是继续单纯调大 late-fusion 权重。

---

## 12. 最小复现 Checklist

### 12.1 V1（当前）

- [ ] 已生成 `text_prototypes.pt`
- [ ] baseline 与多模态仅文本参数不同
- [ ] 训练/验证时 `lambda_txt` 与 `text_proto_path` 设置一致
- [ ] 三组实验矩阵已跑完并保存日志/JSON
- [ ] 小类指标与总体指标都做了对比分析

### 12.2 V2（文本侧轻量升级，实施后勾选）

- [ ] 文本编码分支与 Adapter 参数分组已在优化器中配置
- [ ] 文本向量维度与 `PixelTextAlignHead` 维度一致
- [ ] 在线/缓存策略与每步耗时、峰值显存已记录
- [ ] V2-A / V2-B 对照实验（见 11.10）已跑完

### 12.3 V3（视觉注入升级，实施后勾选）

- [ ] 视觉注入分支与 SAM 输入预处理分流（归一化、分辨率）已写清并单测
- [ ] `fusion_indices` 与通道对齐在日志中可打印 shape
- [ ] 开启 V3 后按第 9.5 节完成延迟/显存回归（若达不到门禁则走 11.8 蒸馏或减层）
- [ ] V3-A / V3-B 单次单变量 ablation 已归档

### 12.4 扩散可选路线（实施后勾选；**当前定稿：角色 A**）

- [ ] 已明确采用角色 A/B/C/D 之一且未同时改多个主变量（**默认仅 A**）
- [ ] **角色 A**：已按 **§13.4** 落盘目录 / list、`manifest` 与 `aug_version` 齐全；**§13.5** 文献与借鉴点已写入实验记录
- [ ] **角色 A**：**§13.4.4** 质检（teacher 一致性等）规则与阈值已固定并有通过率统计
- [ ] 若为角色 B：推理路径是否保留扩散已书面约定（默认训练用、部署不引入多步扩散）
- [ ] 与第 9.5 节门禁冲突时有回滚 ckpt 与配置

完成第 12.1～12.3 后，可形成当前多模态实验闭环；若启用扩散，再勾选 12.4 并满足 **§13.9**（DoD）与 **§13.4**（执行定稿）。

---

## 13. 可选：扩散生成式模型与本方案的结合（增广与特征支路）

本章说明如何将 **潜空间扩散（LDM）** 等与现有 **SAM2 + UDA + RGB/Text（第 3、11 节）** 结合。原则：优先以 **离线数据** 或 **冻结特征支路** 接入，避免推理期多步去噪拖垮第 9.5 节延迟门禁。

### 13.1 目标与边界

- **目标**：利用大规模生成式模型中的 **外观/布局先验** 或 **目标域风格数据**，提升跨域与长尾表现；与文本原型、OPS 式视觉注入 **正交可叠加**。  
- **边界**：默认 **不把分割头整体替换为扩散分割**（工程范式差异过大）；若研究需要，单独立项（见 13.3 角色 D）。

### 13.2 与主路线的关系

| 模块 | 主要作用 | 与扩散的分工 |
|------|----------|----------------|
| 文本分支 | 类级语义先验、`L_txt` / `L_txt_align` | 扩散可用于 **合成图像的 prompt**，语义判据仍以标注与对齐损失为准 |
| 视觉注入（V3） | 空间语义补充 | 扩散 UNet 特征可作为另一支冻结先验；需注意双支路显存 |
| SAM2 + UDA | 几何与域自适应主任务 | 扩散侧建议冻结或仅训轻量融合，避免与伪标签梯度冲突 |

**文本侧注意**：若同时使用 SD 类模型的文本条件与主训练文本分支，应解耦用途（如 SD 只用于增广描述；主分割仍由训练分支监督）。

### 13.3 接入角色（按推荐顺序）

#### 角色 A：离线数据增广（首选，**本仓库当前执行定稿**）

- **做法**：用预训练 LDM（如 Stable Diffusion / SDXL）+ ControlNet / IP-Adapter 等，生成或编辑 **布局/语义保持、外观接近目标域** 的图像；**离线写入磁盘** 后接入现有 dataloader。  
- **优点**：**不改变推理图**；与第 9.5 节无冲突；与 V1/V2/V3 可独立 ablation。  
- **风险**：合成错误类别或错误布局会污染伪标签；需 **质检**（人工抽检 + 与源标注一致性校验 + 固定生成 seed/版本号）。  
- **执行规格与论文明细**：**§13.4（怎么做）**、**§13.5（参考了哪些文献、借鉴什么）**。

#### 角色 B：冻结扩散特征支路（与 11.5 V3 并列）

- **做法**：固定 **VAE 编码器或 UNet 指定层** 特征，经 1×1 / 上采样对齐后注入 SAM decoder（门控相加优先于 full cross-attn），权重 **不反传** 至扩散主干。  
- **优点**：补充 **纹理与生成式空间结构** 先验。  
- **风险**：显存与训练时间上升；分辨率与归一化需与 SAM 输入分流。

#### 角色 C：训练期域桥 / 重建正则（研究向）

- **思路**：参考 **域桥接 / 退化路径** 类工作（如 [DiDA](https://arxiv.org/abs/2412.10339)、[域自适应潜扩散相关路线](https://arxiv.org/abs/2412.16859)）在 **特征或图像空间** 构造中间域，辅助域不变表征。  
- **要求**：单独设计损失与权重 ramp，**不与** 现有 `L_unsup_tgt` 同时大改；建议仅在 **源域有标** 或 **高置信 mask** 上启用辅助项。

#### 角色 D：潜空间扩散分割头（不推荐作为本仓库第一阶段）

- **做法**：在 latent 中建模 mask / 全景布局（如 [Panoptic Diffusion](https://arxiv.org/abs/2412.02929)、潜空间全景/分割一类工作）。  
- **定位**：新架构级课题；若做，需单独数据格式、训练循环与推理步数预算，**默认不纳入** 与第 11 节同步交付的里程碑。

### 13.4 角色 A（离线目标域风格增广）— 执行定稿

本节为 **已定稿实施方案**：在 **不修改推理图、不改动伪标签生成主逻辑** 的前提下，通过 **离线扩散管线** 扩充源域训练图像的外观多样性，使其 **外观接近目标域** 而 **语义布局与标注一致**。

#### 13.4.1 策略定位：源域 S → 增广源域 S′（首选）

| 项目 | 定稿选择 | 说明 |
|------|----------|------|
| 增广对象 | **源域有标数据**（如 `data/cityscapes/.../train`） | 标签 **直接使用原 gt**，不经扩散，避免「无标合成」带来的语义漂移 |
| 监督 | 仍走现有 **`L_sup_src`**（真值监督） | 伪标签 **`L_unsup_tgt`** 仍 **只依赖目标域 teacher**，流程与 `pd_root` 逻辑不变 |
| 推理 | **不包含** 扩散 | 与 **第 9.5 节** 延迟门禁天然兼容 |

**暂不纳入第一版定稿（可作为后续扩展）**：仅对 **目标域无标图** 做风格化后做一致性损失（需额外损失与调参）；第一版以 **S→S′** 验证增益与工程可控性为主。

#### 13.4.2 生成管线（离线三步）

1. **清单（Manifest）**  
   - 为每一张待增广样本生成一条记录，写入 `manifest.jsonl`（或等价 CSV），**每行至少包含**：  
     `src_image_path`、`src_label_path`、`out_image_path`、`dataset_split`、`aug_version`、`base_model`（如 `SD1.5` / `SDXL`）、`control_type`（如 `seg_palette`）、`controlnet` 版本、`strength`（img2img）、`seed`、`prompt_template_id`、`pass_qc`（生成后填）、`qc_iou_teacher`（可选）。  
   - `aug_version` 示例：`cityscapes_densepass_style_v0.2`，**任何超参变更须 bump 版本号**，保证实验可复现。

2. **条件生成（推荐组合）**  
   - **主路径（保语义布局）**：将 **语义标注图** 按 **固定类别调色板** 映射为 RGB「伪彩色图」，作为 **ControlNet（语义/分割类条件）** 的控制输入；原图作为 **img2img** 内容锚，**strength 宜从低到高网格搜索**（例如 0.25～0.55 区间），优先 **小步改变外观**、少改结构。  
   - **辅路径（拉近目标域外观，可选）**：在控制强度足够的前提下，叠加 **IP-Adapter** 用少量 **目标域参考图** 抽取风格；**不得**单独依赖 IP-Adapter 而无语义控制（易跨类幻觉）。  
   - **文本条件**：使用 **短、固定、版本化** 的英文 prompt 模板（如 `prompt_v1`），描述目标域外观关键词（如 weather、camera、fisheye/panoramic 等 **与数据集一致** 的词汇）；**避免**在 prompt 中罗列细分类名替代布局条件。

3. **落盘与对齐**  
   - **推荐目录结构**（与 `README.md` 中 Cityscapes 树兼容，便于最小改动接 `DatasetBuilder`）：  
     - `data/cityscapes_aug_<tag>/leftImg8bit/train/...` 存增广 RGB；  
     - `data/cityscapes_aug_<tag>/gtFine/train/...` 存 **与原图同名** 的 **gt 拷贝**（硬链接或复制均可，**禁止**对标签做生成式修改）。  
   - 新增或切换 list 文件（如 `dataset/cityscapes_list/train_aug.txt`）：每行指向 **增广 RGB 路径 + 原 gt 路径**（若同树则与现有格式一致）。  
   - `run_adaptation.py` / `utils/build.py`：通过 **`--data_root` 指向增广根目录** 或增加「增广 list」参数（二选一，**团队内统一一种** 并写进实验记录）。

#### 13.4.3 分辨率与 Cityscapes 长图（2048×1024）

| 方案 | 做法 | 适用 |
|------|------|------|
| **整图缩放** | 长边缩至 512/768 → 生成 → 双三次插值回原分辨率 | 实现最快，细节偏软 |
| **分块 tile** | 与训练 sliding window 一致的块大小与 overlap，分块生成后 feather 融合 | 质量较好，**QC 建议整图再跑一遍 teacher** |
| **仅增广 patch** | 若训练入口本就是 patch，可直接对 patch 尺寸生成并写回 list | **算力最省**，与 `Cityscapes13` 等 `resize`/`window` 逻辑对齐成本最低 |

**定稿要求**：在 `manifest` 中记录采用的方案与 tile 参数；**与当前 `dataset/Cityscapes.py` 的 resize 与窗口设置一致**，避免增广图与训练预处理双重缩放导致域不一致。

#### 13.4.4 质检门禁（必须通过再进训练）

**自动（建议全量）**

- 使用 **当前域适应前的 best teacher 或固定 checkpoint（冻结）**，对 `I_orig` 与 `I_aug` 在 **相同预处理** 下各前向一次，计算 **教师预测一致性**（例：像素 `argmax` 一致率，可仅在 `gt != ignore_index` 上统计）。  
- **建议初值阈值**：全图一致率 **≥ 0.93～0.97**（需在 100～200 张上网格标定后固定）；低于阈值则 **丢弃该增广图** 或 **降低 strength 重试**（重试次数上限写入 manifest）。  
- **辅助**：各类像素占比与原始标注的 **L1 距离**（防整类区域塌陷/消失）。

**人工**

- 每 **500～1000** 张增广图随机抽 **10～20** 张，检查道路连通性、交通实例、小目标是否被纹理「吃掉」。

**训练侧保险**

- **混合比例 `p_aug`**：建议从 **0.1** 线性或阶梯 ramp 至 **0.2～0.3**；若验证集 **源域 val 掉点** 或目标域 mIoU 无增益，先降 `p_aug` 或收紧 QC 阈值，**勿同时** 改伪标签阈值与增广版本。

#### 13.4.5 实验矩阵（必做与选做）

| ID | 设置 | 目的 |
|----|------|------|
| E0 | 无增广 baseline | 对照 |
| E1 | 仅语义 ControlNet + 单一 `strength` + 整图缩放 | 主结论 |
| E2 | 改 `strength` 或 tile 策略（**单次只改一项**） | 敏感性 |
| E3 | + IP-Adapter（固定参考集） | 目标域外观更强时的收益/风险 |

**记录**：DensePASS（或实际目标域）mIoU、**person / traffic sign / traffic light**、源域 val mIoU（若有）、`p_aug`、`aug_version`、QC 通过率。

#### 13.4.6 工具链（实现任选其一，定稿只要求可复现）

- **diffusers**：便于批处理、固定 seed、与 Python 数据后处理同一仓库。  
- **ComfyUI / A1111**：便于调参出图，导出 **固定工作流 JSON** + 命令行批跑，同样写入 `manifest`。

**依赖**：与 OmniSAM 训练环境 **分离 conda 环境** 推荐，避免 `torch`/CUDA 与主训练冲突；增广产物以 **只读目录** 挂载到训练机。

#### 13.4.7 与多模态文本分支的衔接

- **文本原型**（`text_prototypes.pt`）默认不因增广而重算，除非类别定义或模板集变更。  
- 增广 **只改变 RGB**；`L_txt` / `L_txt_align` 与 **§11** 一致，无需为扩散单独开损失。

---

### 13.5 参考文献与思想来源（角色 A 为主，全章补充）

下表说明 **角色 A 定稿** 主要 **借鉴了哪些论文的哪些思想**（本仓库 **不承诺** 复现某篇论文的完整训练框架，而是 **抽取与数据增广、域偏移、条件生成相关的原则**）。

| 文献 | 链接 / 标识 | 在本方案中的借鉴点 |
|------|----------------|---------------------|
| Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | **潜空间扩散（LDM）** 作为离线生成骨干；**训练主网络仍在像素/特征空间做分割**，扩散不参与反传。 |
| Zhang & Rao, *Adding Conditional Control to Text-to-Image Diffusion Models*（ControlNet） | [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) | **用语义图等强结构条件** 约束生成，支撑「**保布局、改外观**」的主路径，降低跨类幻觉。 |
| Ye et al., *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models* | [arXiv:2308.06721](https://arxiv.org/abs/2308.06721) | **可选**：用目标域参考图注入 **全局外观/风格**，与 ControlNet **分工**（结构 vs 风格）。 |
| Zheng et al., *Open Panoramic Segmentation*（OPS，本仓库本地参考） | ECCV 2024 / [项目页](https://junweizheng93.github.io/publications/OPS/OPS.html) | 提供“预训练语义先验 + 轻量分割头”工程思路；本次重评估不绑定其特定语义编码实现。 |
| Zheng et al., *Panoptic Diffusion Models* | [arXiv:2412.02929](https://arxiv.org/abs/2412.02929) | 强调 **图像与分割布局在生成中的联合建模**；本方案 **不** 做联合生成式分割头，但借用其 **「布局语义必须显式约束」** 的结论，落到 **语义图 + ControlNet**。 |
| Li et al., *A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting* | [OpenReview / HTML](https://arxiv.org/html/2401.10227v2) | **在潜空间用分割相关信号做条件** 的思路；支撑 **语义控制 + img2img** 的选型，而非裸 text-to-image。 |
| *Adversarially Domain-adaptive Latent Diffusion for Unsupervised Semantic Segmentation* | [arXiv:2412.16859](https://arxiv.org/abs/2412.16859) | **UDA 与 LDM 结合** 的代表性方向；本方案取其 **「利用生成式模型缓解域差异」** 的动机，**落地为离线数据增广子集**，不实现文中完整对抗式 LDM 训练。 |
| *Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation*（DiDA） | [arXiv:2412.10339](https://arxiv.org/abs/2412.10339) | **渐进式域桥、避免一步跨域破坏语义**；对应工程上 **低 strength、强 QC、混合比例 ramp**，而非照搬其网络。 |
| Tianyu Lin et al., *Stable Diffusion Segmentation …*（SDSeg，生物医学场景） | [arXiv:2406.18361](https://arxiv.org/abs/2406.18361) | **SD 用于分割相关管线、潜空间与效率讨论**；支撑 **「扩散与主分割网络解耦、离线完成」** 的工程形态。 |
| *Image Editing with Diffusion Models: A Survey* | [arXiv:2504.13226](https://arxiv.org/abs/2504.13226) | **条件编辑、ControlNet/IP-Adapter/img2img** 等工具与风险综述；支撑工具选型与安全增广策略文档化。 |

**其他角色（B/C/D）延伸阅读**（不在角色 A 第一版必做）：可从 **§13.8** 所列关键词与文献继续检索 `latent diffusion semantic segmentation`、`diffusion features UNet block`、`Stable Diffusion domain adaptation segmentation` 等扩展。

---

### 13.6 分阶段落地路线（与第 11 节里程碑对齐）

| 阶段 | 内容 | 前置条件 |
|------|------|----------|
| **D0** | 文献与基线：固定 V1/V2 最佳 mIoU；冻结 **§13.4** 中 `aug_version` 与 prompt 模板 | 第 7、12.1 节完成；**§13.4.4 质检脚本** 可跑通 |
| **D1** | **仅角色 A**：离线生成 S′ → 接入 `data_root` / list（**§13.4.2～13.4.3**） | D0；manifest 完整 |
| **D2** | 角色 B：冻结 UNet 特征支路（**§13.3**） | D1 或明确跳过；显存预算通过 |
| **D3** | 角色 C / 扩散分割头（**§13.3**） | 单独评审 |

**建议顺序**：V1 文本稳定 →（可选）V3 视觉注入 → **D1（角色 A）** → **D2**；**禁止**同一迭代内同时 **大改伪标签阈值 + 新开视觉注入 + 上新 aug_version**。

### 13.7 与第 9.5 节（推理性能）的约定

- **默认交付**：推理路径 **不包含** 多步扩散；**角色 A 零推理开销**；角色 B 若保留到部署，须计入 **第 9.5 节** p50/p95 与显存；若超门禁，采用 **蒸馏**（学生去掉扩散支路）或 **仅训练期使用扩散**。  
- **训练期**：扩散前向仅在 **离线环境**；在线训练机只读增广结果，不承担多步去噪。

### 13.8 文献与关键词（其他角色与检索）

下列工作与本章 **角色 B/C/D** 或后续扩展更相关，角色 A 已定稿后可按需深入：

- 联合图像与分割/全景生成：[Panoptic Diffusion Models](https://arxiv.org/abs/2412.02929)  
- 潜空间分割与掩码建模：[A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting](https://arxiv.org/html/2401.10227v2)  
- UDA 与潜扩散结合：[Adversarially Domain-adaptive Latent Diffusion for Unsupervised Semantic Segmentation](https://arxiv.org/abs/2412.16859)  
- 域桥与退化：[Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](https://arxiv.org/abs/2412.10339)  
- SD 潜空间分割示例：[Stable Diffusion Segmentation 相关单步/潜空间工作](https://arxiv.org/html/2406.18361v2)  
- 编辑与条件生成总览：[Image Editing with Diffusion Models: A Survey](https://arxiv.org/html/2504.13226v1)

检索补充关键词：`latent diffusion semantic segmentation`、`diffusion features UNet block segmentation`、`Stable Diffusion domain adaptation segmentation`。

### 13.9 Definition of Done（扩散相关，含角色 A）

| 条目 | DoD |
|------|-----|
| **A-Data** | `manifest.jsonl`（或等价物）与 `aug_version` 齐全；QC 通过率与阈值记录在案 |
| **A-Train** | E0/E1（及可选 E2/E3）实验表完成；目标域 **mIoU 或小类** 相对 E0 有结论；源域 val **无未解释大跌** |
| **A-Compat** | `L_unsup_tgt` / 伪标签生成路径与无增广时 **逻辑一致**；推理路径 **无扩散** |
| D2 | 单层注入 ablation（无扩散 / 有扩散）已归档；峰值显存与 step 时间有表；推理是否保留扩散已书面说明 |
| 合规 | 若部署保留扩散支路（角色 B），**第 9.5 节** 指标仍过线，否则已回滚或已蒸馏 |

---

完成第 12 节 checklist（若启用扩散则一并满足第 **13.9** 节）后，可形成可复现、可汇报、可继续升级的多模态实验闭环。
