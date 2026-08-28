#!/usr/bin/env python3
"""
experiments/tecma/scripts/generate_baseline_comparison_panels.py

4-column panels: [Input | GT | baseline_omnisam | A_Dtrain_Etrain]
Prioritizes the 6 confirmed failure cases + 4 representative samples (10 total).
Saves to experiments/tecma/figures/qualitative_baseline_comparison/.
Writes a diagnostic report noting where A_Dtrain_Etrain degrades vs baseline.

Safety: read-only against predictions already exported. No training, no inference.
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image as PILImage

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

FAILURE_CASES = [
    (210, "seam_inconsistency"),
    (24,  "pseudo-label_noise_suspected"),
    (305, "boundary_erosion"),
    (319, "thin_structure_broken"),
    (122, "small_object_missing"),
    (101, "class_confusion"),
]
EXTRA_REPRESENTATIVE = [155, 289, 111, 363]
SAMPLE_IDS = [sid for sid, _ in FAILURE_CASES] + EXTRA_REPRESENTATIVE
FAILURE_TYPE = {sid: ftype for sid, ftype in FAILURE_CASES}

PRED_ROOT = os.path.join(_REPO_ROOT, "experiments/tecma/eval_predictions_confirmed")
DATA_ROOT = os.path.join(_REPO_ROOT, "data/DensePASS")
OUT_DIR   = os.path.join(_REPO_ROOT, "experiments/tecma/figures/qualitative_baseline_comparison")
NOTES_PATH = os.path.join(_REPO_ROOT, "experiments/tecma/notes",
                          "baseline_vs_Dtrain_Etrain_visual_diagnosis.md")

# Cityscapes 13-class palette (RGB)
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
], dtype=np.uint8)
IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)
NUM_CLASSES = 13
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain",
    "sky", "person", "car",
]

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask == c] = PALETTE_13[c]
    rgb[mask == 255] = IGNORE_COLOR
    return rgb

def add_text_label(img, text):
    out = img.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def load_input_rgb(sample_id):
    img_path = os.path.join(DATA_ROOT, f"leftImg8bit/val/{sample_id}_.png")
    img = np.array(PILImage.open(img_path).convert("RGB"))
    img = cv2.resize(img, (2048, 400), interpolation=cv2.INTER_LINEAR)
    return img

def load_gt_mask(sample_id):
    gt_path = os.path.join(DATA_ROOT, f"gtFine/val/{sample_id}_labelTrainIds.png")
    gt = np.array(PILImage.open(gt_path))
    gt = cv2.resize(gt.astype(np.uint8), (2048, 400), interpolation=cv2.INTER_NEAREST)
    return gt

def load_pred_mask(config_name, sample_id):
    pred_path = os.path.join(PRED_ROOT, config_name, f"{sample_id}_pred_argmax.png")
    if not os.path.isfile(pred_path):
        return None
    pred = np.array(PILImage.open(pred_path))
    if pred.shape != (400, 2048):
        pred = cv2.resize(pred.astype(np.uint8), (2048, 400), interpolation=cv2.INTER_NEAREST)
    return pred.astype(np.uint8)

def make_panel_4col(sample_id):
    input_rgb = load_input_rgb(sample_id)
    gt_mask = load_gt_mask(sample_id)
    gt_rgb = mask_to_rgb(gt_mask)
    baseline_pred = load_pred_mask("baseline_omnisam", sample_id)
    dtrain_pred = load_pred_mask("A_Dtrain_Etrain", sample_id)

    cols = [add_text_label(input_rgb, "Input"), add_text_label(gt_rgb, "GT")]
    missing = []
    for name, pred in [("baseline_omnisam", baseline_pred), ("A_Dtrain_Etrain", dtrain_pred)]:
        if pred is None:
            tile = add_text_label(np.full((400, 2048, 3), 128, dtype=np.uint8), f"{name} MISSING")
            missing.append(name)
        else:
            tile = add_text_label(mask_to_rgb(pred), name)
        cols.append(tile)

    divider = np.zeros((400, 2, 3), dtype=np.uint8)
    panel = cols[0]
    for c in cols[1:]:
        panel = np.concatenate([panel, divider, c], axis=1)
    return panel, missing, gt_mask, baseline_pred, dtrain_pred

def per_class_iou_delta(gt_mask, pred_baseline, pred_dtrain):
    if pred_baseline is None or pred_dtrain is None:
        return None
    try:
        from utils.metrics import fast_hist, per_class_iu
    except Exception:
        return None
    hist_b = fast_hist(gt_mask.flatten().astype(np.int64), pred_baseline.flatten().astype(np.int64), NUM_CLASSES)
    hist_d = fast_hist(gt_mask.flatten().astype(np.int64), pred_dtrain.flatten().astype(np.int64), NUM_CLASSES)
    iou_b = per_class_iu(hist_b) * 100.0
    iou_d = per_class_iu(hist_d) * 100.0
    delta = iou_d - iou_b
    return iou_b, iou_d, delta

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(NOTES_PATH), exist_ok=True)

    report = [
        "# Baseline vs A_Dtrain_Etrain 视觉失效诊断报告",
        "",
        "**对比**: `baseline_omnisam` (无压缩, best_iou60.26) vs `A_Dtrain_Etrain` (MTC+SMR, best_iou60.02)",
        "**样本**: 6 个已标注失效案例 + 4 个代表性样本",
        "",
        "| Sample | 失效类型 | Panel | 显著恶化类别 (dIoU<-2) | 显著改善类别 (dIoU>2) |",
        "|---|---|---|---|---|",
    ]

    for sid in SAMPLE_IDS:
        ftype = FAILURE_TYPE.get(sid, "representative")
        panel, missing, gt_mask, pred_b, pred_d = make_panel_4col(sid)
        out_path = os.path.join(OUT_DIR, f"{sid}_{ftype}_baseline_vs_dtrain.png")
        PILImage.fromarray(panel).save(out_path)
        print(f"  [{sid}] panel saved -> {out_path}", flush=True)

        worse, better = [], []
        delta_result = per_class_iou_delta(gt_mask, pred_b, pred_d)
        if delta_result is not None:
            _, _, delta = delta_result
            for c, d in enumerate(delta):
                if d < -2:
                    worse.append(f"{CLASS_NAMES[c]}({d:+.1f})")
                elif d > 2:
                    better.append(f"{CLASS_NAMES[c]}({d:+.1f})")

        worse_str = ", ".join(worse) if worse else "-"
        better_str = ", ".join(better) if better else "-"
        rel_path = f"qualitative_baseline_comparison/{os.path.basename(out_path)}"
        report.append(f"| {sid} | {ftype} | `{rel_path}` | {worse_str} | {better_str} |")

    report += [
        "",
        "---",
        "",
        "## 视觉观察记录 (待人工填写关键定性发现)",
        "",
        "> 说明: 上表按每样本的类别级 IoU 差值 (A_Dtrain_Etrain - baseline) 自动标出显著变化类别。",
        "> 请人工打开各 panel 图,记录具体的空间位置/边界/接缝处的定性观察。",
        "",
        "### 待观察要点",
        "- Seam (210): 全景图左右接缝处是否出现类别跳变或边界断裂",
        "- Pseudo-label noise (24): 压缩模型是否放大了伪标签噪声区域的误分类",
        "- Boundary erosion (305): 物体边界(如 person/car 轮廓)是否变得更模糊",
        "- Thin structure (319): 细长结构(如 pole/traffic light 杆)是否更容易丢失",
        "- Small object (122): 小物体(如远处的 person/traffic light)是否更容易被背景类吞掉",
        "- Class confusion (101): 语义相近类别(如 sidewalk/road, wall/building)混淆是否加重",
        "",
        "## 严格非操作记录",
        "",
        "- 未训练模型,未生成新伪标签。",
        "- 仅读取已导出的 baseline_omnisam / A_Dtrain_Etrain 预测 masks 和 DensePASS GT。",
        "- 未覆盖 qualitative_confirmed/、failure_cases_confirmed/ 中的既有文件。",
        "",
    ]

    with open(NOTES_PATH, "w") as f:
        f.write("\n".join(report))
    print(f"\nReport written: {NOTES_PATH}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
