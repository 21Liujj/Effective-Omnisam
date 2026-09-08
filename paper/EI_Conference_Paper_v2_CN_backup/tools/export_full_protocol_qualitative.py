#!/usr/bin/env python3
"""Export four DensePASS qualitative predictions from confirmed full-protocol checkpoints.

Inference only. Reuses the project's model, dataset, checkpoint loader, and sliding-window merge.
It refuses to overwrite any prediction output.
"""
import os
import random
import sys
import tempfile
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import utils.sam2_bootstrap  # noqa: F401
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset.DensePASS import DensePASS13
from sam.build_sam import build_sam2
from model.model import OmniSAM_adapt
from utils.checkpoint_adapt import smart_load_checkpoint
from utils.tools import merge_width_scatter_np

SAMPLE_IDS = (24, 68, 101, 111)
CHECKPOINTS = {
    "baseline": "data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.26.pth",
    "ours": "data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.16.pth",
}
OUT_ROOT = os.path.join(REPO_ROOT, "paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw")
DATA_ROOT = os.path.join(REPO_ROOT, "data/DensePASS")
NUM_CLASSES = 13
PALETTE_13 = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [0, 0, 142],
], dtype=np.uint8)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask_to_rgb(mask):
    if mask.ndim != 2 or mask.min() < 0 or mask.max() >= NUM_CLASSES:
        raise ValueError(f"invalid 13-class prediction range: {mask.min()}..{mask.max()}")
    return PALETTE_13[mask]


def build_model(device):
    sam = build_sam2(
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        os.path.join(REPO_ROOT, "sam/checkpoints/sam2.1_hiera_small.pt"),
        device=device,
    )
    return OmniSAM_adapt(sam, num_classes=NUM_CLASSES, num_maskmem=9).to(device)


def build_loader():
    # Dataset requires a list file. Keep the four-entry subset outside the repository.
    temp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="omnisam_qual_", delete=False)
    try:
        for sid in SAMPLE_IDS:
            temp.write(f"/{sid}_.png\n")
        temp.close()
        dataset = DensePASS13(
            root=DATA_ROOT,
            list_path=temp.name,
            crop_size=(4096, 1024),
            sliding_window=(384, 1024),
            passes=2,
            set="val",
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        return loader, temp.name
    except Exception:
        temp.close()
        if os.path.exists(temp.name):
            os.unlink(temp.name)
        raise


def infer_checkpoint(name, ckpt_rel, loader, device):
    ckpt_abs = os.path.join(REPO_ROOT, ckpt_rel)
    if not os.path.isfile(ckpt_abs):
        raise FileNotFoundError(ckpt_abs)
    model = build_model(device)
    smart_load_checkpoint(model, ckpt_abs)
    model.eval()
    num_frames = loader.dataset._get_frame_cnt()
    outputs = []
    with torch.no_grad():
        for batch in loader:
            image, _label, _unused, names, _ori_label = batch
            image = image.to(device)
            raw_name = names[0] if isinstance(names, (list, tuple)) else str(names)
            sid = int(os.path.basename(str(raw_name)).replace("_.png", "").replace(".png", "").lstrip("/"))
            if sid not in SAMPLE_IDS:
                raise RuntimeError(f"unexpected sample id: {sid}")
            out_dir = os.path.join(OUT_ROOT, name, f"sample_{sid:03d}")
            argmax_path = os.path.join(out_dir, "pred_argmax.png")
            color_path = os.path.join(out_dir, "pred_color.png")
            argmax_exists = os.path.exists(argmax_path)
            color_exists = os.path.exists(color_path)
            if argmax_exists and color_exists:
                print(f"SKIP_COMPLETE {name} sample={sid} {out_dir}", flush=True)
                model.memory_bank._init_output_dict_source()
                continue
            if argmax_exists or color_exists:
                raise FileExistsError(f"partial output exists; refusing to overwrite in {out_dir}")

            bsz, _, _, h, w = image.shape
            logits_maps = np.zeros((bsz, 2 * num_frames, NUM_CLASSES, h, w))
            for frame_idx in range(num_frames):
                output, _, _ = model(frame_idx, image[:, frame_idx])
                logits_maps[:, frame_idx] = output.cpu().numpy()
            model.memory_bank._init_output_dict_source()
            for frame_idx in range(num_frames):
                output, _, _ = model(frame_idx, image[:, num_frames + frame_idx])
                logits_maps[:, num_frames + frame_idx] = output.cpu().numpy()
            model.memory_bank._init_output_dict_source()

            merged = merge_width_scatter_np(
                logits_np=logits_maps,
                orig_size=(1024, 4096),
                W_patch=w,
                stride_w=384,
                passes=2,
                device=str(device),
                return_probs=False,
            )
            pred = cv2.resize(merged[0].astype(np.float32), (2048, 400), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            color = mask_to_rgb(pred)
            os.makedirs(out_dir, exist_ok=False)
            Image.fromarray(pred, mode="L").save(argmax_path)
            Image.fromarray(color, mode="RGB").save(color_path)
            outputs.append((sid, pred.shape, int(pred.min()), int(pred.max()), argmax_path, color_path))
            print(f"EXPORTED {name} sample={sid} shape={pred.shape} range={pred.min()}..{pred.max()}", flush=True)

    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return outputs


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    set_seed()
    device = torch.device("cuda")
    print(f"DEVICE {torch.cuda.get_device_name(0)}", flush=True)
    loader, temp_list = build_loader()
    try:
        all_outputs = {}
        for name, ckpt in CHECKPOINTS.items():
            print(f"CHECKPOINT {name} {ckpt}", flush=True)
            all_outputs[name] = infer_checkpoint(name, ckpt, loader, device)
        missing = []
        for name in CHECKPOINTS:
            for sid in SAMPLE_IDS:
                out_dir = os.path.join(OUT_ROOT, name, f"sample_{sid:03d}")
                for filename in ("pred_argmax.png", "pred_color.png"):
                    path = os.path.join(out_dir, filename)
                    if not os.path.isfile(path):
                        missing.append(path)
        if missing:
            raise RuntimeError(f"incomplete export; missing={missing}")
        print("EXPORT_COMPLETE", flush=True)
    finally:
        if os.path.exists(temp_list):
            os.unlink(temp_list)


if __name__ == "__main__":
    main()
