# OmniSAM DensePASS 实验排查记录

本文档记录本次围绕 `DensePASS` UDA 实验的完整排查思路，重点回答三个问题：

1. 为什么最开始复现时 `mIoU` 能从 `60.29` 提高到 `60.36`；
2. 为什么后续自己做的实验，尤其是复用伪标签、训练加速版本，会出现 `mIoU` 持续下降；
3. 为什么最终决定先恢复到更接近最早成功复现时的实验设置，再继续观察能否在 `60.36` 基础上提升。

---

## 1. 最开始的成功复现

最开始的基线验证结果：

- `validate_densepass_baseline_20260407_131045.log`
- 使用 checkpoint：`/root/autodl-tmp/omnisam_s.pth`
- 验证集 `sam2_s val mIoU: 60.29`

随后进行一次标准 UDA 训练：

- `uda_densepass_20260408_143548.log`
- 训练 1 个 epoch 后，验证集结果变为：
  - `Epoch 1: val mIoU=60.36, best=60.36`

这说明：

- 当时的训练流程至少在那一次实验中是有效的；
- 也说明“用当前 teacher 生成伪标签，再做一轮 UDA”是有可能带来小幅增益的；
- 因此 `60.36` 可以视为当前阶段一个重要参考点。

---

## 2. 后续实验为什么开始下降

后续掉点主要出现在两类实验里：

1. 固定复用某一轮伪标签继续训练多个 epoch；
2. 引入了一系列“省时 + 提效”改动后，再与最开始那条成功曲线直接比较。

### 2.1 固定复用同一份伪标签会导致过拟合噪声

在 `uda_densepass_reuse_epoch1_opt_20260409_195415.log` 中：

- 使用固定目录 `reuse_pl_dir=.../epoch1`
- 后续连续训练多个 epoch
- 指标从：
  - `60.36 -> 60.16 -> 60.06 -> 59.99 -> 59.85`

这说明：

- 同一份伪标签如果被连续使用很多次，模型容易逐渐去拟合伪标签中的噪声；
- 训练损失可能继续下降，但验证集真标签上的 `mIoU` 反而下滑；
- 这类现象在自训练 / 半监督 / UDA 场景中很常见，本质上是 confirmation bias。

### 2.2 训练加速与训练策略改动，会让实验不再和“最开始成功那次”可直接比较

在 `IMPROVEMENTS.md` 中加入过的改动包括：

- `reuse_pl_dir`：复用已有伪标签目录；
- `conf_thr`：按像素置信度过滤 target 无监督损失；
- `batch_size` 提大；
- AMP / TF32；
- ramp 的 `total_steps` 改成真实 `len(target_loader)`；
- iterative teacher / round 式训练。

这些改动里，最可能直接改变 `mIoU` 的不是“提速”本身，而是它们会改变训练目标和优化动态：

- `conf_thr` 改变了参与 target loss 的像素集合；
- `batch_size` 改变了每步梯度统计特性；
- `reuse_pl_dir` 把“每轮重打伪标签”变成了“固定伪标签多训几轮”；
- `total_steps` 修正后，loss 权重 ramp 节奏也和原来不同。

因此，后来那些实验虽然更快，但已经不是“和最开始成功复现完全同一个设定”。

---

## 3. 一个关键问题：best checkpoint 曾经被错误覆盖

在自动 iterative 实验中，还发现了一个非常关键的问题：

- 初始 teacher 是 `best_sam2_s_iou60.36.pth`
- 但训练代码里的 `best_mIoU` 初始值是 `0.0`
- 结果第一轮验证即使只有 `60.14`，也会被误认为是“当前 best”
- 于是后续 round 的 teacher 可能被错误更新成 `60.14` 这一版模型

这会造成两个后果：

1. 记录上看起来像是“best 变成了 60.14”；
2. 更严重的是后续生成伪标签时，teacher 真的可能退化成 `60.14`，导致后续 round 质量进一步下滑。

针对这个问题，已经做了两类修正：

- 在 `adaptation/engine.py` 中，支持从 `tgt_ckpt` 文件名中解析 `iou60.36` 作为初始 baseline；
- 在 `tools/auto_iterative_pl_train.py` 中，加了一层保护：伪标签生成时使用的 teacher 不会弱于初始化 teacher。

这一步是必要的，因为如果 best 逻辑都错了，后面的实验结论会被严重污染。

---

## 4. 为什么判断“先回退到接近最早成功复现的设置”更合理

在时间紧、目标明确（尽快超过 `60.36`）的情况下，最合理的策略不是继续在复杂改动上叠加，而是先缩小变量，做回归验证。

核心判断依据如下：

### 4.1 最开始那次成功实验，至少证明原始思路是能涨的

- `60.29 -> 60.36` 已经说明最原始的一轮 UDA 策略是可行的；
- 所以如果后续所有复杂实验都掉点，优先应怀疑“新改动破坏了原本有效的训练动态”。

### 4.2 当前目标是快速确认“能不能继续涨”，而不是完整做方法学扩展

如果此时继续同时保留：

- conf mask
- AMP / TF32
- 更大 batch
- iterative round
- 复用伪标签
- 改过的调度逻辑

那么一旦结果下降，将很难判断到底是哪一个因素导致的。

### 4.3 恢复旧设置更适合做对照实验

为了验证“是不是后续改动导致掉点”，最直接的方法就是：

- 尽量恢复到与首次成功复现接近的设置；
- 只保留必要修复，例如：
  - 初始 best baseline 修正；
  - 避免伪标签生成 teacher 退化；
- 其它会显著改变训练形态的改动先全部关掉。

---

## 5. 最终恢复的实验思路

基于上述分析，决定先跑一个“original-ish from 60.36”的对照实验。

目标是：

- 起点改为 `60.36` 的 best ckpt；
- 但训练流程尽量贴近最开始成功复现时的样子；
- 排除后续加速和训练策略改动的干扰。

### 5.1 恢复原则

恢复后的思路如下：

- 使用 `best_sam2_s_iou60.36.pth` 作为 `tgt_ckpt`；
- 不使用 `reuse_pl_dir`；
- 不使用 iterative driver；
- 不启用 AMP；
- `batch_size` 恢复到 `2`；
- 用 `conf_thr=-1` 的方式关闭置信度掩码；
- 使用新的 `PD_ROOT`，避免和旧实验的伪标签或 resume 状态混用。

这样做的目的不是完全回退代码到历史版本，而是尽可能恢复当时“训练动态”最接近的实验设置。

### 5.2 新建的对照脚本

新增脚本：

- `scripts/uda_densepass_originalish_from_60_36.sh`

其设计目的就是：

- 从 `60.36` 出发；
- 尽量模拟最开始那次复现的训练风格；
- 便于直接跑 `1~3` 个 epoch 做快速验证。

默认配置包含：

- `NUM_EPOCHS=1`
- `BATCH_SIZE=2`
- `CONF_THR=-1`
- `AMP=0`
- `TF32=1`

如果要跑 3 个 epoch：

```bash
cd /root/autodl-tmp/OmniSAM-main
PD_ROOT=/root/autodl-tmp/OmniSAM-main/pd_uda_clean_run1 NUM_EPOCHS=3 bash scripts/uda_densepass_originalish_from_60_36.sh
```

---

## 6. 关于“多 epoch 是否每轮都重打伪标签”的澄清

这里需要区分两个脚本：

### 6.1 `scripts/uda_densepass_originalish_from_60_36.sh`

这个脚本本质上只是调用一次 `run_adaptation.py`。

若：

- `NUM_EPOCHS=1`，则通常只生成一次 `epoch1` 伪标签；
- `NUM_EPOCHS>1`，且 `PD_ROOT` 是全新的，则每个 epoch 对应自己的目录：
  - `epoch1`
  - `epoch2`
  - `epoch3`

也就是每个 epoch 都会生成该 epoch 对应的一套伪标签。

这与最开始“每个 epoch 都按正常流程生成并训练”的理解是一致的。

### 6.2 `tools/auto_iterative_pl_train.py`

这个脚本则是 round 式驱动：

- 每个 round 先 `generate_only`
- 再用 `reuse_pl_dir` 固定那一轮生成的 `epoch1`
- 连续训练 `CHUNK_EPOCHS` 个 epoch

所以它是一种“每轮一套伪标签，多训几个 epoch”的策略，与 `run_adaptation.py` 的原始逐 epoch 逻辑不同。

---

## 7. 当前判断与下一步建议

当前最重要的判断是：

- 后续掉点并不能简单归因于“60.36 这个 teacher 不行”；
- 更大的可能性是：后续实验同时叠加了太多训练策略改动，导致已经不再与最开始成功那次实验可比；
- 再叠加 best 逻辑错误，进一步放大了掉点问题。

因此当前阶段最合理的实验顺序是：

1. 先用 `scripts/uda_densepass_originalish_from_60_36.sh` 跑一个尽量接近原始复现风格的 1~3 epoch 对照实验；
2. 观察：
   - `Epoch 1`
   - `Epoch 2`
   - `Epoch 3`
   的 `val mIoU` 是否仍全部低于 `60.36`；
3. 如果这条最接近原版的设置仍然不能超过 `60.36`，再考虑：
   - 是否已经接近当前 teacher 的上限；
   - 是否需要更换 teacher；
   - 是否应重新设计伪标签提纯策略，而不是继续堆训练技巧。

---

## 8. 本次排查的核心结论

本次排查最终形成的核心结论如下：

- `60.29 -> 60.36` 的成功实验说明原始 UDA 路线是能带来提升的；
- 后续下降并不意味着“teacher 更强反而更差”，更可能是训练设定被改得过多；
- `reuse_pl_dir` 固定伪标签多 epoch 训练，很容易造成对伪标签噪声的过拟合；
- `conf_thr`、更大 batch、AMP、schedule 修正等改动会改变训练动态，不能再和最开始那条曲线直接比较；
- 自动 iterative 流程里，初始 best 逻辑错误曾导致 teacher 错误退化到 `60.14`，这是一个必须先修掉的问题；
- 在时间紧的情况下，最优先策略应当是恢复到更接近最开始成功复现的实验设置，先做一个可信的对照实验。

这也是为什么当前决定：

- 暂时不继续叠加复杂提效改动；
- 而是先从 `60.36` 出发，跑一个尽量“原版风格”的对照实验，再决定下一步。

---

## 9. 多模态实验方案（RGB + 文本）执行规范

本节给出已经在仓库中落地的多模态训练方案（RGB + 类别文本先验）。目标是：在不增加额外传感器的前提下，用文本语义锚点提升跨域稳定性，尤其关注长尾类与小目标类。

### 9.1 方法原理（通俗版）

原始 OmniSAM 只看图像外观（`L_vis`）。多模态版新增一个文本分支：

1. 为每个语义类构造若干 prompt（如 road/person/car）；
2. 用冻结 CLIP 文本编码器得到每类语义原型 `T_c`；
3. 将 decoder 的像素特征投影到文本空间，计算与 `T_c` 的余弦相似度，得到 `L_txt`；
4. 训练/推理时融合输出：`L_fuse = L_vis + λ * L_txt`；
5. 训练时新增 `L_txt_align`（KL），约束视觉分支与文本分支的一致性。

直观理解：视觉分支告诉模型“看起来像什么”，文本分支告诉模型“语义上应该像什么”，融合后可缓解跨域外观漂移。

### 9.2 代码落地点（已实现）

- 文本原型构建脚本：`tools/build_text_prototypes.py`
- 文本对齐模块：`model/text_align.py`
  - `PixelTextAlignHead`
  - `kl_align_loss`
- 模型融合逻辑：`model/model.py`（`OmniSAM_adapt`）
- 文本原型加载与模型注入：`utils/build.py`
- 训练损失接入：`adaptation/strategies.py`
- 训练参数开关：`run_adaptation.py`（`TrainConfig`）
- 文本验证脚本（多 ckpt 对比）：`validation_muti.py`

### 9.3 训练前准备

#### 9.3.1 安装依赖

`requirements.txt` 已新增 `open_clip_torch`，确保环境内已安装：

```bash
cd /root/autodl-tmp/OmniSAM-main
pip install -r requirements.txt
```

#### 9.3.2 生成文本原型（一次即可）

```bash
cd /root/autodl-tmp/OmniSAM-main
python tools/build_text_prototypes.py \
  --dataset Cityscapes \
  --class_set 13 \
  --clip_model ViT-B-32 \
  --clip_pretrained openai \
  --out ./assets/text_prototypes_cityscapes13_vitb32.pt
```

输出文件包含：

- `text_prototypes`（`C x D`）
- `class_names`
- `templates`
- `clip_model` / `clip_pretrained`

### 9.4 训练配置说明（关键超参）

在 `run_adaptation.py` 中新增参数：

- `--text_proto_path`：文本原型文件路径；为空表示关闭文本分支
- `--lambda_txt`：融合权重（`L_fuse = L_vis + lambda_txt * L_txt`）
- `--txt_tau`：文本分支余弦 logits 温度
- `--weight_txt_align`：`L_txt_align` 权重
- `--txt_align_tau`：KL 对齐温度

建议起步值：

- `lambda_txt=0.1`
- `weight_txt_align=0.05`
- `txt_tau=0.07`
- `txt_align_tau=1.0`

稳定后可提升到：

- `lambda_txt=0.3`
- `weight_txt_align=0.1`

### 9.5 标准实验矩阵（与 §10.9 对齐）

建议固定同一 teacher 起点（如 `60.36` ckpt），先做三组：

1. **Baseline（无文本）**  
2. **Baseline + Text（`lambda_txt=0.1`, `weight_txt_align=0.05`）**  
3. **Baseline + Text（`lambda_txt=0.3`, `weight_txt_align=0.1`）**

对比口径保持一致：同数据切分、同验证集、同评价代码。

### 9.6 训练命令模板

以下以 DensePASS UDA 为例（在原有命令上补充多模态参数）：

```bash
cd /root/autodl-tmp/OmniSAM-main
python run_adaptation.py \
  --backbone sam2_s \
  --dataset Cityscapes \
  --class_set 13 \
  --batch_size 2 \
  --num_epochs 1 \
  --tgt_ckpt /root/autodl-tmp/OmniSAM-main/pd_uda_densepass/checkpoints/Cityscapes/sam2_s/best_sam2_s_iou60.36.pth \
  --src_ckpt /root/autodl-tmp/omnisam_s.pth \
  --text_proto_path ./assets/text_prototypes_cityscapes13_vitb32.pt \
  --lambda_txt 0.1 \
  --txt_tau 0.07 \
  --weight_txt_align 0.05 \
  --txt_align_tau 1.0
```

若要关闭文本分支做对照，只需：

- `--text_proto_path ""`
- `--lambda_txt 0`
- `--weight_txt_align 0`

### 9.7 验证与多模型对比

使用 `validation_muti.py`（格式同 `validation.py`，支持多个 ckpt 一次对比）：

```bash
cd /root/autodl-tmp/OmniSAM-main
python validation_muti.py \
  --backbone sam2_s \
  --dataset DensePASS \
  --class_set 13 \
  --use_mem_bank \
  --data_root data \
  --ckpt /root/autodl-tmp/omnisam_s.pth \
  --ckpt /root/autodl-tmp/OmniSAM-main/pd_uda_densepass/checkpoints/Cityscapes/sam2_s/best_sam2_s_iou60.36.pth \
  --ckpt_names official,uda60_36 \
  --text_proto_path ./assets/text_prototypes_cityscapes13_vitb32.pt \
  --lambda_txt 0.1 \
  --txt_tau 0.07 \
  --metrics_json ./logs/val_muti_text.json
```

### 9.8 指标与结论判定标准

主指标：

- `mIoU`
- `mAcc`
- `PixAcc`
- `FWIoU`
- `mDice`

重点关注：

- 小类 IoU（`person`, `traffic sign`, `traffic light`）
- 前 3 epoch 稳定性（是否出现明显震荡）
- 与 baseline 的真实差值（不是单次偶然波动）

建议判定规则：

- 若 `mIoU` 与小类指标均稳定提升（>= 0.2）且曲线平稳，可继续加大 `lambda_txt`；
- 若边界类掉点或波动变大，先降低 `lambda_txt` 与 `weight_txt_align`；
- 若全局无提升，优先检查 prompt 模板质量与原型类别顺序是否与训练类别完全一致。

### 9.9 常见问题与排查

1. **旧视觉 ckpt 开文本后效果不稳定**  
   原因：旧 ckpt 未训练过文本头。建议先用小权重微调再评估。

2. **文本原型维度/类别不匹配报错**  
   检查 `text_prototypes` 的 `C` 是否等于当前 `num_classes`，并确认 13/19 类设置一致。

3. **训练初期波动大**  
   将 `lambda_txt` 调小到 `0.05~0.1`，`weight_txt_align` 调小到 `0.02~0.05`，观察 1~2 epoch。

4. **显存上涨**  
   文本分支本身较轻；若仍吃紧，先保持 `batch_size=2` 并关闭其他额外开销配置。

### 9.10 复现实验最小 checklist

- [ ] 已生成 `text_prototypes.pt`
- [ ] `class_set` 与原型类别数匹配
- [ ] baseline 与多模态仅差在文本相关参数
- [ ] 使用同一验证脚本与同一数据切分
- [ ] 至少完成三组对照并记录 JSON 指标

完成以上步骤后，即可得到可复现实验结论：文本先验是否在当前 DensePASS UDA 场景中带来稳定收益。

### 9.11 V1 结论与 V2 改进记录（2026-04）

本小节记录当前 Text-Conditioned OmniSAM 的第一阶段结论，以及据此确定的第二阶段改进方向，避免后续重复走弯路。

#### 9.11.1 V1（text logits late fusion）当前结论

V1 的结构为：

```text
OpenCLIP class text prototypes
-> trainable text adapter / projector
-> text logits
-> final_logits = visual_logits + alpha * text_logits
```

围绕该 V1 结构，已经完成以下关键验证：

1. `real / shuffle / random` 文本原型三种模式均已打通；
2. `fusion_alpha`、`text_scale`、`text_adapter_grad_norm` 等日志已落盘；
3. 当 `alpha_init=0.0` 时，文本分支几乎不参与训练，`fusion_alpha` 只在 `1e-6` 量级波动；
4. 通过引入 `alpha_init=0.05 / 0.10 / 0.20` 以及更高的 text-branch 学习率，已经确认文本分支可被激活；
5. 激活后 `fusion_alpha` 非零、`text_adapter_grad_norm` 非零、`final_logits` 相对 `visual_logits` 的偏移也会随 `alpha` 增大而线性变大。

但多轮 `fixed50` ablation 的结果显示：

```text
A1(real) ≈ A4(shuffle) ≈ A5(random)
```

这意味着：

- 问题已不再是“文本分支没有接上”；
- 而是“文本分支接上了，但当前 late fusion 结构仍未稳定体现真实 OpenCLIP 语义优势”。

更准确地说，当前 V1 更像一个：

```text
class-embedding branch
```

而不是真正的：

```text
language-semantic grounding branch
```

原因在于：只要后续 projector / decoder 足够可训练，`random` prototype 也可能被模型学成“固定类别代码”，从而导致 `real/shuffle/random` 都能产生类似效果。

#### 9.11.2 已完成的关键诊断结论

截至当前，已经明确得到以下工程与实验结论：

1. **旧问题已解决：**
   - 文本分支不再是“假接入”；
   - `alpha=0` 导致文本梯度被压制的问题已被实验确认。

2. **新的核心问题：**
   - 即使文本分支有梯度、能影响 `final_logits`，`real` 仍没有在小样本短训练中稳定优于 `shuffle/random`；
   - 说明当前 `logit-add late fusion` 太晚、太弱，难以把 OpenCLIP 语义转成有效的像素级语义增益。

3. **因此，后续不应继续只在 V1 上盲目增大 `alpha`：**
   - `alpha` 变大只意味着文本扰动更大；
   - 但“扰动变大”并不自动等于“真实语义更强”。

#### 9.11.3 V2 改进方向：image-to-text cross-attention

基于上述结论，第二阶段改进方向确定为：

```text
V2 = image-to-text cross-attention feature fusion
```

其核心思想是：

```text
semantic decoder feature F
+ projected text prototypes T
-> cross-attention(Q=image, K/V=text)
-> fused feature F'
-> 原 segmentation head
-> segmentation logits
```

即：

```text
OpenCLIP text prototypes
-> text adapter/projector
-> image-to-text cross-attention fusion
-> enhanced decoder feature
-> 原 segmentation head
-> segmentation logits
```

相比 V1：

- V1：在 logits 阶段做 late fusion；
- V2：在 semantic decoder feature 阶段做 feature-level fusion。

这样做的目的不是简单“加文本分数”，而是让每个像素 token 在形成语义 feature 时，就可以主动从类别文本原型中获取语义补充。

#### 9.11.4 为什么 V2 先选 image-to-text，而不是 text-to-image

当前阶段优先做 `image-to-text cross-attention` 的原因：

1. 输出仍为 `[B, C, H, W]`，可直接接回原有 segmentation head；
2. 对 OmniSAM 主干侵入更小，不破坏现有 patch sequence / memory attention 主流程；
3. 对 `A1/A4/A5` 负控兼容性更好；
4. 适合作为从 V1 向更强文本融合过渡的低风险升级版。

需要强调：

- `image-to-text` 是“像素向文本取语义补充”；
- `text-to-image` 是“类别文本向图像找空间证据”；
- 后者更接近 OpenWorldSAM 的 grounding 思路，但工程代价更高，应作为后续 V3 再评估。

因此当前推荐路线为：

```text
V1: logit_add late fusion  （已验证，语义优势未显化）
V2: image-to-text cross-attn（当前主线）
V3: text-to-image cross-attn（若 V2 仍无效，再进入）
```

#### 9.11.5 V2 需要重点记录的新指标

为了避免 V2 跑完后仍无法判断“文本是否真正改变预测”，V2 除保留原有 `text_dynamics.jsonl` 外，必须新增或重点关注以下指标：

1. **`prediction_change_ratio`**
   - 定义：`mean(argmax(final_logits) != argmax(visual_logits))`
   - 用于判断文本融合是否真正改变了最终预测，而不只是轻微改变 logits 均值。

2. **按类 `prediction_change_ratio_cls_*`**
   - 当前优先看：
     - `sidewalk`
     - `wall`
     - `fence`
     - `pole`
     - `traffic light`
     - `traffic sign`
     - `terrain`
   - 用于判断文本扰动是否主要发生在语义混淆类或长尾小类上。

3. **`cross_attn_entropy_mean`**
   - 若过高：说明每个像素平均看所有类，文本注意力不聚焦；
   - 若过低但 `random` 也同样过低：说明可能只是把 token 当类别代码，而不是在利用真实语义。

4. **`feature_delta_ratio_mean`**
   - 衡量 cross-attn 对 decoder feature 的改变量；
   - 若为零或极小，则说明 feature-level fusion 仍未真正介入。

5. **梯度与门控动态**
   - `cross_beta`
   - `cross_beta_grad_norm`
   - `cross_attn_grad_norm`
   - `text_projector_grad_norm`
   - `semantic_decoder_grad_norm`

#### 9.11.6 V2 的实验判据

V2 先做 `fixed50, 1 epoch` smoke：

```text
B1_real
B4_shuffle
B5_random
```

主要检查：

1. 是否无 shape / dtype / OOM 报错；
2. `cross_beta` 是否非零；
3. `cross_attn_grad_norm` 是否非零；
4. `feature_delta_ratio` 是否非零；
5. `prediction_change_ratio` 是否明显高于 V1；
6. `B1` 是否开始与 `B4/B5` 拉开。

若 smoke 正常，再进入：

```text
fixed200, 2~3 epoch:
A0 baseline
B1 real
B4 shuffle
B5 random
```

最终判据：

- **强正向：**
  - `B1_real > B4_shuffle / B5_random`
  - 且 `B1` 至少不低于 `A0`

- **弱正向：**
  - 总 `mIoU` 提升不显著，但小类或易混类对 `B4/B5` 有稳定优势

- **无效：**
  - `B1/B4/B5` 仍几乎完全重合
  - 且 `prediction_change_ratio`、`cross_attn_entropy`、`feature_delta_ratio` 也没有显著差异

#### 9.11.7 若 V2 仍无效，后续不再优先调 `alpha/beta`

如果 `fixed200 + 2~3 epoch` 后 V2 仍然出现：

```text
B1 ≈ B4 ≈ B5
```

则应认为问题不再是“文本没有接上”，而是：

```text
当前 static text prototype + projector + image-to-text cross-attn
仍不足以把 OpenCLIP 真实语义转化为有效分割增益
```

此时后续优先级应转向：

1. 更严格地约束 projector（降低其把 random token 学成类别码的能力）；
2. 更明确的 text-visual alignment loss；
3. 再评估更强的 `text-to-image cross-attention` / language grounding 方案；
4. 必要时考虑 multi-template prompt 或更强的 query-based text conditioning。

简言之：

> V1 已经证明“文本能接上”；V2 要验证“文本是否开始在特征层面体现真实语义优势”；若 V2 仍无效，再进入更接近 OpenWorldSAM 的强 grounding 版本，而不是继续单纯调大 late fusion 权重。
