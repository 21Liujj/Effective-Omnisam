#!/usr/bin/env python3
"""
Inference baseline_omnisam on 12 DensePASS val samples for failure visualization.
Reuses export_confirmed_predictions.py logic, targets only baseline checkpoint.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = "/root/autodl-tmp/OmniSAM-main"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import utils.sam2_bootstrap  # noqa: F401

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torch.utils.data import DataLoader
from dataset.DensePASS import DensePASS13
from sam.build_sam import build_sam2
from model.model import OmniSAM_adapt
from utils.checkpoint_adapt import smart_load_checkpoint
from utils.tools import merge_width_scatter_np

BASELINE_CKPT = "experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.26.pth"
VAL_SUBSET_LIST = "experiments/tecma/eval_predictions_confirmed/val_subset12.txt"
DATA_ROOT = "data/DensePASS"
OUT_DIR = "experiments/tecma/eval_predictions_confirmed/baseline_omnisam"
NUM_CLASSES = 13
BACKBONE = "sam2_s"
NUM_MASKMEM = 9
BATCH_SIZE = 1
SEED = 42
BACKBONE_HYDRA_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT = "sam/checkpoints/sam2.1_hiera_small.pt"

def set_seed(seed: int):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(device):
    sam = build_sam2(BACKBONE_HYDRA_CFG, SAM2_CKPT, device=device)
    model = OmniSAM_adapt(sam, num_classes=NUM_CLASSES, num_maskmem=NUM_MASKMEM).to(device)
    return model

def load_ckpt(model, ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    smart_load_checkpoint(model, ckpt_path)
    print(f"Loaded checkpoint: {ckpt_path}", flush=True)

def build_val_loader(list_path):
    dataset = DensePASS13(
        root=DATA_ROOT,
        list_path=list_path,
        crop_size=(4096, 1024),
        sliding_window=(384, 1024),
        passes=2,
        set="val",
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

def run_inference(model, loader, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    num_frames = loader.dataset._get_frame_cnt()
    saved = 0

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            image, label, _, names, ori_label = batch
            image = image.to(device)
            B, _, H, W = image.shape[0], image.shape[1], image.shape[-2], image.shape[-1]

            raw_name = names[0] if isinstance(names, (list, tuple)) else str(names)
            sample_id = os.path.basename(str(raw_name)).replace("_.png", "").replace(".png", "").lstrip("/")
            out_path = os.path.join(out_dir, f"{sample_id}_pred_argmax.png")

            if os.path.isfile(out_path):
                print(f"[skip] {sample_id} exists", flush=True)
                model.memory_bank._init_output_dict_source()
                model.memory_bank._init_output_dict_source()
                continue

            logits_maps = np.zeros((B, 2 * num_frames, NUM_CLASSES, H, W))

            for frame_idx in range(num_frames):
                frame_img = image[:, frame_idx]
                output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, frame_idx] = output.cpu().numpy()

            model.memory_bank._init_output_dict_source()

            for frame_idx in range(num_frames):
                frame_img = image[:, num_frames + frame_idx]
                output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, num_frames + frame_idx] = output.cpu().numpy()

            model.memory_bank._init_output_dict_source()

            merged = merge_width_scatter_np(
                logits_np=logits_maps,
                orig_size=(1024, 4096),
                W_patch=W,
                stride_w=384,
                passes=2,
                device=str(device),
                return_probs=False,
            )

            pred_argmax = cv2.resize(
                merged[0].astype(np.float32), (2048, 400), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)

            PILImage.fromarray(pred_argmax).save(out_path)
            saved += 1
            print(f"[{idx+1:2d}/12] saved {sample_id}", flush=True)

    print(f"Done. saved={saved}", flush=True)
    return saved

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    if not os.path.isfile(VAL_SUBSET_LIST):
        raise FileNotFoundError(f"val_subset12.txt not found: {VAL_SUBSET_LIST}")

    loader = build_val_loader(VAL_SUBSET_LIST)
    print(f"Loader: {len(loader)} batches", flush=True)

    model = build_model(device)
    load_ckpt(model, BASELINE_CKPT)
    saved = run_inference(model, loader, device, OUT_DIR)

    print(f"\nBaseline predictions saved to: {OUT_DIR}", flush=True)
    print(f"Total: {saved} files", flush=True)

if __name__ == "__main__":
    main()
