#!/usr/bin/env python3
"""
experiments/tecma/scripts/generate_confirmed_panels.py

After export_confirmed_predictions.py has run, generates:
1. 6-column side-by-side panels for 12 samples:
   [Input | GT | A_ccpm | A_Dtrain | A_Etrain_keep025 | A_Dtrain_Etrain]
   → experiments/tecma/figures/qualitative_confirmed/sample_{ID}/confirmed_panel.png

2. Confirmed failure case panels for 6 candidate samples:
   → experiments/tecma/figures/failure_cases_confirmed/confirmed_{ID}_{type}.png

3. Writes experiments/tecma/notes/step_20_21_confirmed_prediction_visualization_report.md

Safety:
  - Does NOT overwrite existing files.
  - Does NOT run training.
  - Reads from eval_predictions_confirmed/ which must exist.
"""
import os
import sys
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = ["A_ccpm", "A_Dtrain", "A_Etrain_keep025", "A_Dtrain_Etrain"]

CANDIDATE_IDS = [122, 155, 289, 101, 111, 263, 68, 319, 305, 24, 210, 363]

FAILURE_CASES = [
    (122, "small_object_missing"),
    (319, "thin_structure_broken"),
    (305, "boundary_erosion"),
    (210, "seam_inconsistency"),
    (101, "class_confusion"),
    (24,  "pseudo-label_noise_suspected"),
]

PRED_ROOT    = os.path.join(_REPO_ROOT, "experiments/tecma/eval_predictions_confirmed")
DATA_ROOT    = os.path.join(_REPO_ROOT, "data/DensePASS")
QUAL_OUT     = os.path.join(_REPO_ROOT, "experiments/tecma/figures/qualitative_confirmed")
FAIL_OUT     = os.path.join(_REPO_ROOT, "experiments/tecma/figures/failure_cases_confirmed")
NOTES_PATH   = os.path.join(_REPO_ROOT, "experiments/tecma/notes",
                            "step_20_21_confirmed_prediction_visualization_report.md")

# Cityscapes 13-class palette (BGR for cv2, RGB for PIL)
PALETTE_13 = np.array([
    [128, 64,128],  # road
    [244, 35,232],  # sidewalk
    [ 70, 70, 70],  # building
    [102,102,156],  # wall
    [190,153,153],  # fence
    [153,153,153],  # pole
    [250,170, 30],  # traffic light
    [220,220,  0],  # traffic sign
    [107,142, 35],  # vegetation
    [152,251,152],  # terrain
    [ 70,130,180],  # sky
    [220, 20, 60],  # person
    [  0,  0,142],  # car
], dtype=np.uint8)  # RGB
IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)
NUM_CLASSES = 13
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "car",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert label mask (H, W) with values 0-12 / 255 to RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask == c] = PALETTE_13[c]
    rgb[mask == 255] = IGNORE_COLOR
    return rgb


def add_text_label(img: np.ndarray, text: str, font_scale: float = 0.6, thickness: int = 2) -> np.ndarray:
    """Add top-left text label to an RGB numpy image. Returns new array."""
    out = img.copy()
    # Convert to BGR for cv2 putText, then back
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    # Shadow
    cv2.putText(bgr, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    # White text
    cv2.putText(bgr, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_input_rgb(sample_id: int) -> np.ndarray:
    """Load DensePASS val input image, resize to (400, 2048, 3)."""
    img_path = os.path.join(DATA_ROOT, f"leftImg8bit/val/{sample_id}_.png")
    img = np.array(PILImage.open(img_path).convert("RGB"))
    img = cv2.resize(img, (2048, 400), interpolation=cv2.INTER_LINEAR)
    return img


def load_gt_mask(sample_id: int) -> np.ndarray:
    """Load DensePASS val GT label (labelTrainIds), return uint8 mask (400, 2048)."""
    gt_path = os.path.join(DATA_ROOT, f"gtFine/val/{sample_id}_labelTrainIds.png")
    gt = np.array(PILImage.open(gt_path))
    gt = cv2.resize(gt.astype(np.uint8), (2048, 400), interpolation=cv2.INTER_NEAREST)
    return gt


def load_pred_mask(config_name: str, sample_id: int) -> np.ndarray:
    """Load prediction argmax mask (400, 2048), or None if missing."""
    pred_path = os.path.join(PRED_ROOT, config_name, f"{sample_id}_pred_argmax.png")
    if not os.path.isfile(pred_path):
        return None
    pred = np.array(PILImage.open(pred_path))
    if pred.shape != (400, 2048):
        pred = cv2.resize(pred.astype(np.uint8), (2048, 400), interpolation=cv2.INTER_NEAREST)
    return pred.astype(np.uint8)


def make_panel_6col(sample_id: int) -> np.ndarray:
    """
    Build a 6-column side-by-side panel (H=400, W=2048 per col):
    [Input | GT | A_ccpm | A_Dtrain | A_Etrain_keep025 | A_Dtrain_Etrain]
    Returns HStack RGB image.
    """
    input_rgb = load_input_rgb(sample_id)
    gt_mask   = load_gt_mask(sample_id)
    gt_rgb    = mask_to_rgb(gt_mask)

    cols = [
        add_text_label(input_rgb, "Input"),
        add_text_label(gt_rgb, "GT"),
    ]

    missing_configs = []
    for cfg in CONFIGS:
        pred = load_pred_mask(cfg, sample_id)
        if pred is None:
            # Gray placeholder with "MISSING" text
            placeholder = np.full((400, 2048, 3), 128, dtype=np.uint8)
            tile = add_text_label(placeholder, f"{cfg} MISSING")
            missing_configs.append(cfg)
        else:
            pred_rgb = mask_to_rgb(pred)
            tile = add_text_label(pred_rgb, cfg)
        cols.append(tile)

    # Add dividers (2px black)
    divider = np.zeros((400, 2, 3), dtype=np.uint8)
    panel = cols[0]
    for c in cols[1:]:
        panel = np.concatenate([panel, divider, c], axis=1)

    return panel, missing_configs

def make_failure_panel_6col(sample_id: int, failure_type: str) -> np.ndarray:
    """
    Build a failure-case 6-column panel identical to make_panel_6col.
    Returns (panel_rgb, missing_configs).
    """
    return make_panel_6col(sample_id)


def compute_per_class_delta(gt_mask: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray):
    """
    Returns per-class IoU delta (B-A) for diagnostic notes.
    Returns None on error.
    """
    try:
        from utils.metrics import fast_hist, per_class_iu
        hist_a = fast_hist(gt_mask.flatten().astype(np.int64), pred_a.flatten().astype(np.int64), NUM_CLASSES)
        hist_b = fast_hist(gt_mask.flatten().astype(np.int64), pred_b.flatten().astype(np.int64), NUM_CLASSES)
        iou_a = per_class_iu(hist_a) * 100.0
        iou_b = per_class_iu(hist_b) * 100.0
        delta = iou_b - iou_a
        return iou_a.tolist(), iou_b.tolist(), delta.tolist()
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(QUAL_OUT, exist_ok=True)
    os.makedirs(FAIL_OUT, exist_ok=True)

    report_lines = [
        "# Step 20/21 Confirmed Prediction Visualization Report — TeCMA Phase 7.2 + 7.3",
        "",
        f"**生成时间**: 2026-06-18",
        "",
        "## 四配置 Prediction Masks 来源",
        "",
        "| Config | Checkpoint | val_mIoU |",
        "|---|---|---:|",
        "| A_ccpm | `runs/A_ccpm_seed1_20260616_104911/...best_iou60.26.pth` | 60.26 |",
        "| A_Dtrain | `runs/A_Dtrain_seed1_20260616_111019/...best_iou60.12.pth` | 60.12 |",
        "| A_Etrain_keep025 | `runs/token_budget/A_Etrain_keep025.../...best_iou60.10.pth` | 60.10 |",
        "| A_Dtrain_Etrain | `runs/A_Dtrain_Etrain_seed1_20260616_121135/...best_iou60.02.pth` | 60.02 |",
        "",
        "**推断策略**: 标准推断（D/E 不激活），对比纯 training effect。",
        "",
        "---",
        "",
        "## 12 样本 Confirmed Panel 状态",
        "",
        "| Sample | Panel 路径 | Missing Configs | Status |",
        "|---|---|---|---|",
    ]

    all_missing_any = []

    for sid in CANDIDATE_IDS:
        sample_dir = os.path.join(QUAL_OUT, f"sample_{sid}")
        panel_path = os.path.join(sample_dir, "confirmed_panel.png")

        if os.path.isfile(panel_path):
            report_lines.append(f"| {sid} | `qualitative_confirmed/sample_{sid}/confirmed_panel.png` | — | ✅ EXISTS (skip) |")
            continue

        os.makedirs(sample_dir, exist_ok=True)
        panel, missing = make_panel_6col(sid)
        PILImage.fromarray(panel).save(panel_path)

        # Also save individual tiles
        input_rgb = load_input_rgb(sid)
        gt_mask = load_gt_mask(sid)
        PILImage.fromarray(input_rgb).save(os.path.join(sample_dir, "input_panorama.png"))
        PILImage.fromarray(mask_to_rgb(gt_mask)).save(os.path.join(sample_dir, "ground_truth_color.png"))

        for cfg in CONFIGS:
            pred = load_pred_mask(cfg, sid)
            if pred is not None:
                PILImage.fromarray(mask_to_rgb(pred)).save(
                    os.path.join(sample_dir, f"pred_{cfg}_color.png")
                )
                np.save(os.path.join(sample_dir, f"pred_{cfg}_argmax.npy"), pred)

        missing_str = ", ".join(missing) if missing else "—"
        status = "⚠️ PARTIAL" if missing else "✅ OK"
        if missing:
            all_missing_any.append(sid)
        report_lines.append(f"| {sid} | `qualitative_confirmed/sample_{sid}/confirmed_panel.png` | {missing_str} | {status} |")
        print(f"  [{sid}] panel saved → {panel_path}", flush=True)

    report_lines += [
        "",
        "---",
        "",
        "## 6 Confirmed Failure Cases",
        "",
        "| Sample | Type | Panel | Missing Configs | Status |",
        "|---|---|---|---|---|",
    ]

    for sid, ftype in FAILURE_CASES:
        fail_path = os.path.join(FAIL_OUT, f"confirmed_{sid}_{ftype}.png")
        if os.path.isfile(fail_path):
            report_lines.append(f"| {sid} | {ftype} | `failure_cases_confirmed/confirmed_{sid}_{ftype}.png` | — | ✅ EXISTS |")
            continue

        panel, missing = make_failure_panel_6col(sid, ftype)
        PILImage.fromarray(panel).save(fail_path)

        missing_str = ", ".join(missing) if missing else "—"
        status = "⚠️ PARTIAL" if missing else "✅ OK"
        report_lines.append(f"| {sid} | {ftype} | `failure_cases_confirmed/confirmed_{sid}_{ftype}.png` | {missing_str} | {status} |")
        print(f"  [{sid}] failure case saved → {fail_path}", flush=True)

    # Summary
    report_lines += [
        "",
        "---",
        "",
        "## 严格非操作记录",
        "",
        "- 未训练模型。",
        "- 未生成伪标签。",
        "- 未覆盖已有文件（`qualitative_diagnostic/`、`failure_cases_diagnostic/` 保持不变）。",
        "- 推断仅使用 `eval_predictions_confirmed/` 中预先导出的预测 masks。",
        "",
    ]

    # Write report
    if os.path.isfile(NOTES_PATH):
        print(f"  [skip] report already exists: {NOTES_PATH}", flush=True)
    else:
        with open(NOTES_PATH, "w") as f:
            f.write("\n".join(report_lines))
        print(f"  Report written: {NOTES_PATH}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
