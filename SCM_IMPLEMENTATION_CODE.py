"""
Seamless Circular Memory Bank - Implementation Code
基于实际代码库架构的完整实现

关键发现:
- DensePASS实际resize到 4096×1024 (不是原始2048×400)
- Sliding window: 1024×1024, stride=384
- 这意味着circular padding要在4096宽度上处理
"""

# ============================================================================
# 文件1: dataset/DensePASS_circular.py
# 新增circular padding支持的DensePASS数据集
# ============================================================================

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils import data
from torchvision import transforms

from dataset.DensePASS import DensePASS13, DensePASS19


class DensePASS13Circular(DensePASS13):
    """
    扩展DensePASS13，添加circular padding支持

    关键修改:
    1. 在resize到4096×1024后，对width维度做circular padding
    2. Padding width = sliding_window_size//2 = 512 (确保边界窗口能访问对侧)
    3. 只在validation时padding (训练保持原样)
    """

    def __init__(self, root, list_path, crop_size=(4096, 1024),
                 sliding_window=None, passes=1, set='val',
                 enable_circular_padding=True,
                 circular_pad_width=512):
        """
        Args:
            enable_circular_padding: 是否启用circular padding
            circular_pad_width: padding宽度 (像素), 建议window_size//2
        """
        super().__init__(root, list_path, crop_size, sliding_window, passes, set)

        self.enable_circular_padding = enable_circular_padding
        self.circular_pad_width = circular_pad_width

        print(f"[DensePASS13Circular] Initialized with:")
        print(f"  - crop_size: {crop_size}")
        print(f"  - circular_padding: {enable_circular_padding}")
        print(f"  - pad_width: {circular_pad_width}px")

    def _circular_pad_image(self, img_tensor):
        """
        对图像tensor在width维度做circular padding

        Args:
            img_tensor: [C, H, W] tensor

        Returns:
            padded: [C, H, W+2*pad_width] tensor
        """
        # F.pad的参数顺序: (left, right, top, bottom)
        # 我们只padding左右
        padded = F.pad(
            img_tensor,
            pad=(self.circular_pad_width, self.circular_pad_width, 0, 0),
            mode='circular'
        )
        return padded

    def _circular_pad_label(self, label_tensor):
        """
        对label tensor在width维度做circular padding

        Args:
            label_tensor: [H, W] LongTensor

        Returns:
            padded: [H, W+2*pad_width] LongTensor
        """
        # Label需要先unsqueeze成[1, 1, H, W]再padding
        label_4d = label_tensor.unsqueeze(0).unsqueeze(0).float()

        padded_4d = F.pad(
            label_4d,
            pad=(self.circular_pad_width, self.circular_pad_width, 0, 0),
            mode='circular'
        )

        # 恢复成[H, W]
        padded = padded_4d.squeeze(0).squeeze(0).long()
        return padded

    def __getitem__(self, index):
        """
        重写__getitem__，在父类基础上添加circular padding
        """
        # 调用父类获取原始数据
        image, label, size, name, ori_label = super().__getitem__(index)

        # image shape: [num_patches, C, H, W]
        # label shape: [num_patches, H, W]

        # 只在validation时且enable_circular_padding=True时padding
        if self.enable_circular_padding and self.set == 'val':
            num_patches = image.shape[0]

            padded_images = []
            padded_labels = []

            for i in range(num_patches):
                img_patch = image[i]      # [C, H, W]
                lbl_patch = label[i]      # [H, W]

                # Circular padding
                img_padded = self._circular_pad_image(img_patch)
                lbl_padded = self._circular_pad_label(lbl_patch)

                padded_images.append(img_padded)
                padded_labels.append(lbl_padded)

            image = torch.stack(padded_images)
            label = torch.stack(padded_labels)

            # 更新size信息 (用于后续处理)
            # size原本是 (num_patches, H, W) 的tuple
            # 需要reflect padding后的width
            new_size = (size[0], size[1], size[2] + 2 * self.circular_pad_width)
            size = new_size

        return image, label, size, name, ori_label


class DensePASS19Circular(DensePASS19):
    """
    19-class版本的circular DensePASS
    """
    def __init__(self, root, list_path, crop_size=(4096, 1024),
                 sliding_window=None, passes=1, set='val',
                 enable_circular_padding=True,
                 circular_pad_width=512):
        super().__init__(root, list_path, crop_size, sliding_window, passes, set)

        self.enable_circular_padding = enable_circular_padding
        self.circular_pad_width = circular_pad_width

    def _circular_pad_image(self, img_tensor):
        return F.pad(img_tensor,
                    pad=(self.circular_pad_width, self.circular_pad_width, 0, 0),
                    mode='circular')

    def _circular_pad_label(self, label_tensor):
        label_4d = label_tensor.unsqueeze(0).unsqueeze(0).float()
        padded_4d = F.pad(label_4d,
                         pad=(self.circular_pad_width, self.circular_pad_width, 0, 0),
                         mode='circular')
        return padded_4d.squeeze(0).squeeze(0).long()

    def __getitem__(self, index):
        image, label, size, name, ori_label = super().__getitem__(index)

        if self.enable_circular_padding and self.set == 'val':
            num_patches = image.shape[0]
            padded_images = []
            padded_labels = []

            for i in range(num_patches):
                img_padded = self._circular_pad_image(image[i])
                lbl_padded = self._circular_pad_label(label[i])
                padded_images.append(img_padded)
                padded_labels.append(lbl_padded)

            image = torch.stack(padded_images)
            label = torch.stack(padded_labels)

            new_size = (size[0], size[1], size[2] + 2 * self.circular_pad_width)
            size = new_size

        return image, label, size, name, ori_label


# ============================================================================
# 文件2: adaptation/losses.py (新增)
# Seam Consistency Loss
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeamConsistencyLoss(nn.Module):
    """
    接缝一致性loss: 强制x=0和x=4095两侧的预测一致

    原理:
    - 360°全景图在物理上x=4095的右侧应该连接到x=0
    - 我们约束左边界(x<seam_width)和右边界(x>W-seam_width)的预测一致
    - 使用L1 loss或KL divergence
    """

    def __init__(self, seam_width=128, weight=0.1, loss_type='l1'):
        """
        Args:
            seam_width: 接缝区域宽度(像素), 对4096宽建议128
            weight: loss权重
            loss_type: 'l1' | 'kl' | 'cosine'
        """
        super().__init__()
        self.seam_width = seam_width
        self.weight = weight
        self.loss_type = loss_type

        print(f"[SeamConsistencyLoss] Initialized:")
        print(f"  - seam_width: {seam_width}px")
        print(f"  - weight: {weight}")
        print(f"  - loss_type: {loss_type}")

    def forward(self, pred_logits):
        """
        Args:
            pred_logits: [B, C, H, W] 模型输出logits

        Returns:
            loss: scalar tensor
        """
        B, C, H, W = pred_logits.shape

        # 提取左右边界区域
        pred_left = pred_logits[:, :, :, :self.seam_width]      # [B, C, H, seam_width]
        pred_right = pred_logits[:, :, :, -self.seam_width:]    # [B, C, H, seam_width]

        # 计算一致性loss
        if self.loss_type == 'l1':
            # 直接L1 loss (on logits)
            loss = F.l1_loss(pred_left, pred_right)

        elif self.loss_type == 'kl':
            # KL divergence (on probability distributions)
            pred_left_prob = F.softmax(pred_left, dim=1)
            pred_right_prob = F.softmax(pred_right, dim=1)

            # KL(left || right)
            loss = F.kl_div(
                pred_left_prob.log(),
                pred_right_prob,
                reduction='batchmean'
            )

        elif self.loss_type == 'cosine':
            # Cosine similarity (on features)
            # Flatten spatial dimensions
            left_flat = pred_left.view(B, C, -1)    # [B, C, H*seam_width]
            right_flat = pred_right.view(B, C, -1)

            # Cosine similarity, averaged over spatial positions
            cos_sim = F.cosine_similarity(left_flat, right_flat, dim=1)  # [B, H*seam_width]
            loss = 1.0 - cos_sim.mean()  # 1-similarity as loss

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return self.weight * loss


# ============================================================================
# 文件3: adaptation/train_uda_with_scm.py
# 训练脚本修改 (集成Seam Consistency Loss)
# ============================================================================

"""
这个文件展示如何在现有train_uda.py中集成SCM

关键修改点:
1. 导入SeamConsistencyLoss
2. 在训练循环中计算seam_loss
3. 加到total_loss中
4. 记录到tensorboard/wandb

实际使用时,可以直接修改原有的 adaptation/train_uda.py
"""

# 在train_uda.py顶部添加import
from adaptation.losses import SeamConsistencyLoss

# 在train函数中初始化seam loss
def train_main(...):
    # ... 原有代码 ...

    # 初始化Seam Consistency Loss
    seam_loss_fn = SeamConsistencyLoss(
        seam_width=128,      # 对4096宽, 128px是合理的
        weight=0.1,          # 初始权重0.1
        loss_type='l1'       # 先用L1, 可以试KL
    ).to(device)

    # ... 训练循环 ...

    for epoch in range(start_epoch, num_epochs):
        model.train()

        for batch_idx, (images, pseudo_labels) in enumerate(train_loader):
            images = images.to(device)
            pseudo_labels = pseudo_labels.to(device)

            # 前向传播
            outputs = model(images)  # [B, num_classes, H, W]

            # 原有segmentation loss
            seg_loss = criterion(outputs, pseudo_labels)

            # 新增: Seam consistency loss
            seam_loss = seam_loss_fn(outputs)

            # 总loss
            total_loss = seg_loss + seam_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 记录
            if batch_idx % log_interval == 0:
                print(f"Epoch [{epoch}][{batch_idx}/{len(train_loader)}] "
                      f"Total: {total_loss.item():.4f} | "
                      f"Seg: {seg_loss.item():.4f} | "
                      f"Seam: {seam_loss.item():.4f}")

                # Tensorboard/Wandb logging
                if writer is not None:
                    global_step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('train/total_loss', total_loss.item(), global_step)
                    writer.add_scalar('train/seg_loss', seg_loss.item(), global_step)
                    writer.add_scalar('train/seam_loss', seam_loss.item(), global_step)


# ============================================================================
# 文件4: validation_scm.py
# 修改validation.py以支持circular padding的数据集
# ============================================================================

"""
修改点:
1. 导入 DensePASS13Circular / DensePASS19Circular
2. 添加 --enable_scm 参数
3. 根据参数选择使用circular dataset

实际使用时可以:
方案A: 复制validation.py为validation_scm.py并修改
方案B: 直接在validation.py中添加参数支持
"""

# 在validation.py顶部添加
from dataset.DensePASS_circular import DensePASS13Circular, DensePASS19Circular

# 在main函数的argparse中添加
def main():
    parser = argparse.ArgumentParser()
    # ... 原有参数 ...

    # 新增SCM相关参数
    parser.add_argument('--enable_scm', action='store_true',
                        help='Enable Seamless Circular Memory (SCM) with circular padding')
    parser.add_argument('--scm_pad_width', type=int, default=512,
                        help='Circular padding width for SCM (default: 512)')

    args = parser.parse_args()

    # ... 在创建DensePASS dataset的地方修改 ...

    if dataset_name == "DensePASS":
        root_syn = os.path.join(data_root, "DensePASS")
        val_list = os.path.join(repo_root, "dataset/densepass_list/val.txt")
        syn_w, syn_h = 4096, 1024
        sw_w_stride, sw_size = 384, 1024

        # 根据--enable_scm选择dataset类
        if args.enable_scm:
            print(f"[SCM] Using circular padding with pad_width={args.scm_pad_width}")
            DensePASSClass13 = DensePASS13Circular
            DensePASSClass19 = DensePASS19Circular
            extra_kwargs = {
                'enable_circular_padding': True,
                'circular_pad_width': args.scm_pad_width
            }
        else:
            print("[Baseline] Using standard DensePASS (no circular padding)")
            DensePASSClass13 = DensePASS13
            DensePASSClass19 = DensePASS19
            extra_kwargs = {}

        if class_set == 19:
            val_dataset = DensePASSClass19(
                root_syn,
                val_list,
                crop_size=(syn_w, syn_h),
                sliding_window=(sw_w_stride, sw_size),
                passes=2,
                **extra_kwargs
            )
        else:
            val_dataset = DensePASSClass13(
                root_syn,
                val_list,
                crop_size=(syn_w, syn_h),
                sliding_window=(sw_w_stride, sw_size),
                passes=2,
                **extra_kwargs
            )


# ============================================================================
# 文件5: scripts/run_scm_experiments.sh
# 自动化实验脚本
# ============================================================================

"""
运行4个配置的对比实验:
1. Baseline (no SCM)
2. +Circular Padding only
3. +Seam Loss only (需要先训练)
4. +Both (完整SCM)
"""

# !/bin/bash

REPO_ROOT="/root/autodl-tmp/OmniSAM-main"
cd $REPO_ROOT

# 配置
DATASET="DensePASS"
CLASS_SET=13
CKPT_DIR="experiments/route1/runs/boundary_0625_n500_seed1_20260826_101721/checkpoints/Cityscapes_13cls/sam2_s"
CKPT_PATH="${CKPT_DIR}/best_iou59.99.pth"
OUTPUT_BASE="experiments/scm_ablation"

mkdir -p $OUTPUT_BASE

echo "=========================================="
echo "SCM Ablation Experiments"
echo "=========================================="

# Config 1: Baseline (no SCM)
echo "[1/4] Running Baseline (no circular padding, no seam loss)..."
python validation.py \
    --dataset $DATASET \
    --class_set $CLASS_SET \
    --resume $CKPT_PATH \
    --save_dir ${OUTPUT_BASE}/baseline \
    --enable_scm false \
    2>&1 | tee ${OUTPUT_BASE}/baseline.log

# Config 2: +Circular Padding only
echo "[2/4] Running +Circular Padding..."
python validation.py \
    --dataset $DATASET \
    --class_set $CLASS_SET \
    --resume $CKPT_PATH \
    --save_dir ${OUTPUT_BASE}/circular_padding \
    --enable_scm true \
    --scm_pad_width 512 \
    2>&1 | tee ${OUTPUT_BASE}/circular_padding.log

# Config 3 & 4需要先训练带seam loss的模型
# 这里假设已经训练好，直接加载checkpoint
# TODO: 添加训练脚本

echo "=========================================="
echo "Experiments completed!"
echo "Results saved to: $OUTPUT_BASE"
echo "=========================================="

# 运行seam region analysis
echo "Running seam region analysis..."
for config in baseline circular_padding; do
    python experiments/tecma/diagnostics/seam_region_miou_analysis.py \
        --pred_dir ${OUTPUT_BASE}/${config}/predictions \
        --gt_dir data/DensePASS/gtFine/val \
        --seam_width 200 \
        --output ${OUTPUT_BASE}/${config}_seam_analysis.json
done


# ============================================================================
# 使用说明
# ============================================================================

"""
立即执行步骤:

Day 1 (明天8/29):
-----------------
1. 创建 dataset/DensePASS_circular.py (上面文件1)
   复制粘贴代码，测试import

2. Smoke test:
   cd /root/autodl-tmp/OmniSAM-main
   python -c "
   from dataset.DensePASS_circular import DensePASS13Circular
   ds = DensePASS13Circular(
       'data/DensePASS',
       'dataset/densepass_list/val.txt',
       crop_size=(4096, 1024),
       sliding_window=(384, 1024),
       passes=2,
       enable_circular_padding=True,
       circular_pad_width=512
   )
   img, lbl, size, name, ori = ds[0]
   print(f'Image shape: {img.shape}')  # 应该看到width > 1024
   print(f'Label shape: {lbl.shape}')
   "

Day 2 (8/30):
-----------------
1. 创建 adaptation/losses.py (上面文件2)

2. 修改现有的训练脚本集成seam loss (参考文件3)

3. 运行短epoch测试 (n=100样本)
   验证loss不是NaN，training能正常运行

Day 3 (8/31):
-----------------
1. 修改 validation.py 支持--enable_scm参数 (参考文件4)

2. 测试推理:
   python validation.py --dataset DensePASS --enable_scm --resume <ckpt>

3. 运行对比实验 (Baseline vs +Padding)

关键点:
- seam_width要调整: 原来假设2048宽用100px，现在4096宽应该用200px
- circular_pad_width=512 (window_size 1024的一半)
- 第一版先不改memory attention，只用padding+loss
"""
