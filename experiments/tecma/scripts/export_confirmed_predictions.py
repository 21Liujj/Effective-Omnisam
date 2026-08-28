#!/usr/bin/env python3
"""
experiments/tecma/scripts/export_confirmed_predictions.py

Inference-only export for TeCMA Phase 7.2/7.3 confirmed visualization.
Runs each of the four configs' best checkpoint on 12 DensePASS val samples.
Saves prediction argmax masks WITHOUT modifying any existing code.

Usage (from repo root):
  conda run -n omnisam python experiments/tecma/scripts/export_confirmed_predictions.py

Outputs:
  experiments/tecma/eval_predictions_confirmed/
    val_subset12.txt              (subset list, already exists)
    A_ccpm/                       (NNN_pred_argmax.png for each sample)
    A_Dtrain/
    A_Etrain_keep025/
    A_Dtrain_Etrain/

Safety:
  - Does NOT overwrite existing prediction files (skips with warning).
  - Does NOT modify model weights.
  - Does NOT run training or pseudo-label generation.
  - Does NOT touch experiments/tecma/runs/ or any existing checkpoint.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure repo root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Bootstrap SAM2 path (must happen before any sam/model imports)
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

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = _REPO_ROOT

CONFIGS = [
    {
        "name": "A_ccpm",
        "ckpt": "experiments/tecma/runs/A_ccpm_seed1_20260616_104911/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.26.pth",
    },
    {
        "name": "A_Dtrain",
        "ckpt": "experiments/tecma/runs/A_Dtrain_seed1_20260616_111019/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.12.pth",
    },
    {
        "name": "A_Etrain_keep025",
        "ckpt": "experiments/tecma/runs/token_budget/A_Etrain_keep025_seed1_20260617_114629/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.10.pth",
    },
    {
        "name": "A_Dtrain_Etrain",
        "ckpt": "experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.02.pth",
    },
]

VAL_SUBSET_LIST = os.path.join(
    REPO_ROOT, "experiments/tecma/eval_predictions_confirmed/val_subset12.txt"
)
DATA_ROOT = os.path.join(REPO_ROOT, "data/DensePASS")
OUT_ROOT = os.path.join(REPO_ROOT, "experiments/tecma/eval_predictions_confirmed")

NUM_CLASSES = 13
BACKBONE = "sam2_s"
NUM_MASKMEM = 9   # must match training
BATCH_SIZE = 1
SEED = 42

# SAM2 backbone config
BACKBONE_HYDRA_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT = os.path.join(REPO_ROOT, "sam/checkpoints/sam2.1_hiera_small.pt")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(device: torch.device) -> OmniSAM_adapt:
    """Build model skeleton (no checkpoint loaded yet)."""
    sam = build_sam2(BACKBONE_HYDRA_CFG, SAM2_CKPT, device=device)
    model = OmniSAM_adapt(sam, num_classes=NUM_CLASSES, num_maskmem=NUM_MASKMEM).to(device)
    return model


def load_ckpt(model: OmniSAM_adapt, ckpt_rel: str):
    ckpt_abs = os.path.join(REPO_ROOT, ckpt_rel)
    if not os.path.isfile(ckpt_abs):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_abs}")
    smart_load_checkpoint(model, ckpt_abs)
    print(f"  Loaded checkpoint: {ckpt_abs}", flush=True)


def build_val_loader(list_path: str) -> DataLoader:
    """Build DensePASS13 loader from a custom list file (subset or full val)."""
    dataset = DensePASS13(
        root=DATA_ROOT,
        list_path=list_path,
        crop_size=(4096, 1024),
        sliding_window=(384, 1024),
        passes=2,
        set="val",
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return loader


def run_inference_one_config(model, loader, device, out_dir: str):
    """
    Run sliding-window inference on all samples in loader.
    Save prediction argmax to out_dir/{sample_id}_pred_argmax.png.
    Skips if file already exists.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # Count frames from dataset directly
    num_frames = loader.dataset._get_frame_cnt()

    saved = 0
    skipped = 0

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            image, label, _, names, ori_label = batch
            image = image.to(device)
            B = image.shape[0]
            H, W = image.shape[-2:]

            # Extract sample name: DensePASS returns e.g. "24_.png"
            raw_name = names[0] if isinstance(names, (list, tuple)) else str(names)
            # Strip leading slash and trailing "_" or extension to get clean ID
            sample_id = os.path.basename(str(raw_name)).replace("_.png", "").replace(".png", "").lstrip("/")

            out_path = os.path.join(out_dir, f"{sample_id}_pred_argmax.png")
            if os.path.isfile(out_path):
                print(f"  [skip] {sample_id} already exists: {out_path}", flush=True)
                skipped += 1
                # Still need to flush memory bank state
                model.memory_bank._init_output_dict_source()
                model.memory_bank._init_output_dict_source()
                continue

            logits_maps = np.zeros((B, 2 * num_frames, NUM_CLASSES, H, W))

            # Forward pass 1
            for frame_idx in range(num_frames):
                frame_img = image[:, frame_idx]
                output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, frame_idx] = output.cpu().numpy()

            model.memory_bank._init_output_dict_source()

            # Forward pass 2
            for frame_idx in range(num_frames):
                frame_img = image[:, num_frames + frame_idx]
                output, _, _ = model(frame_idx, frame_img)
                logits_maps[:, num_frames + frame_idx] = output.cpu().numpy()

            model.memory_bank._init_output_dict_source()

            # Merge sliding windows → (pred_h=400, pred_w=2048)
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
            print(f"  [{idx+1:3d}] saved {sample_id} → {out_path}", flush=True)

    print(f"  Done. saved={saved}, skipped={skipped}", flush=True)
    return saved, skipped

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import json, time

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Verify subset list exists
    if not os.path.isfile(VAL_SUBSET_LIST):
        raise FileNotFoundError(f"Subset val list not found: {VAL_SUBSET_LIST}")

    # Read sample IDs for reporting
    with open(VAL_SUBSET_LIST) as f:
        subset_entries = [l.strip() for l in f if l.strip()]
    print(f"Subset samples ({len(subset_entries)}): {subset_entries}", flush=True)

    loader = build_val_loader(VAL_SUBSET_LIST)
    print(f"Loader: {len(loader)} batches", flush=True)

    summary = {}
    t_total = time.perf_counter()

    for cfg in CONFIGS:
        name = cfg["name"]
        ckpt_rel = cfg["ckpt"]
        out_dir = os.path.join(OUT_ROOT, name)
        print(f"\n{'='*60}", flush=True)
        print(f"Config: {name}", flush=True)
        print(f"Ckpt:   {ckpt_rel}", flush=True)
        print(f"Out:    {out_dir}", flush=True)

        t0 = time.perf_counter()
        model = build_model(device)
        load_ckpt(model, ckpt_rel)
        saved, skipped = run_inference_one_config(model, loader, device, out_dir)
        elapsed = time.perf_counter() - t0

        summary[name] = {
            "ckpt": ckpt_rel,
            "out_dir": out_dir,
            "saved": saved,
            "skipped": skipped,
            "elapsed_sec": round(elapsed, 2),
            "files": sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else [],
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    total_elapsed = time.perf_counter() - t_total

    # Write summary JSON
    summary_path = os.path.join(OUT_ROOT, "export_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"configs": summary, "total_elapsed_sec": round(total_elapsed, 2)}, f, indent=2)
    print(f"\nSummary written: {summary_path}", flush=True)
    print(f"Total elapsed: {total_elapsed:.1f}s", flush=True)

    # Print inventory
    print("\n=== Export Inventory ===", flush=True)
    for name, info in summary.items():
        print(f"  {name}: saved={info['saved']}, skipped={info['skipped']}, files={info['files']}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
