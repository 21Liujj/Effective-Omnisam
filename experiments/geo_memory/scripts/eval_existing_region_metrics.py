#!/usr/bin/env python3
"""Inventory and run existing Geo-Memory region metrics diagnostics.

Task 1-4 constraints:
- input inventory first; no training, no model changes, no memory selection changes
- dry-run 3-5 matched samples before full available-mask metrics
- DensePASS13 label mapping follows dataset/DensePASS.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.geo_memory.geometry_ops import REGION_POLAR, REGION_SEAM_POLAR, build_region_mask

PYTHON = sys.executable
IGNORE_INDEX = 255
NUM_CLASSES = 13
SEAM_WIDTH = 32
POLAR_BAND_RATIO = 0.15
DRY_RUN_N = 5

# DensePASS13._key in dataset/DensePASS.py: map Cityscapes/DensePASS 19 trainIds to OmniSAM 13 classes.
DENSEPASS19_TO_13 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12, 255, 12, 12], dtype=np.int64)
CLASS_MAPPING_NOTE = (
    "DensePASS13 mapping from dataset/DensePASS.py: "
    "[0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12]; "
    "ignore_index=255; num_classes=13; valid GT follows validation.py fast_hist filter GT∈[0,13)."
)

GT_DIR = Path("data/DensePASS/gtFine/val")
METHODS = [
    {
        "key": "GM0_baseline_omnisam",
        "table_name": "GM0 baseline_omnisam",
        "pred_candidates": [
            Path("experiments/tecma/eval_predictions_confirmed/baseline_omnisam"),
            Path("experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/predictions"),
            Path("experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/debug_pred_vis"),
        ],
    },
    {
        "key": "GM1_A_Dtrain_Etrain",
        "table_name": "GM1 tecmem_base / A_Dtrain_Etrain",
        "pred_candidates": [Path("experiments/tecma/eval_predictions_confirmed/A_Dtrain_Etrain")],
    },
]

OUT_JSON = Path("experiments/geo_memory/results/region_metrics_existing.json")
OUT_TABLE = Path("experiments/geo_memory/tables/region_metrics_existing.md")
OUT_NOTE = Path("experiments/geo_memory/notes/step_01_4_existing_region_metrics.md")
RAW_ROOT = Path("experiments/geo_memory/raw_results")


def natural_key(s: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]


def canonical_id(path: Path, kind: str) -> str:
    stem = path.stem
    if kind == "pred":
        stem = re.sub(r"_pred_argmax$", "", stem)
        stem = re.sub(r"_pred$", "", stem)
    elif kind == "gt":
        stem = re.sub(r"_labelTrainIds$", "", stem)
        stem = re.sub(r"_gtFine_labelTrainIds$", "", stem)
    stem = stem.strip("/")
    stem = stem[:-1] if stem.endswith("_") else stem
    return stem


def list_masks(directory: Path, kind: str) -> Dict[str, Path]:
    if not directory.exists() or not directory.is_dir():
        return {}
    out: Dict[str, Path] = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff", ".npy"}:
            out[canonical_id(p, kind)] = p
    return out


def choose_pred_dir(candidates: List[Path]) -> Tuple[Path | None, str]:
    for p in candidates:
        if p.exists() and p.is_dir() and list_masks(p, "pred"):
            return p, "found"
    existing_dirs = [str(p) for p in candidates if p.exists()]
    if existing_dirs:
        return None, "candidate directories exist but contain no mask files"
    return None, "all candidate prediction paths missing"


def load_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return np.asarray(Image.open(path), dtype=np.int64)


def map_densepass13(gt_raw: np.ndarray) -> np.ndarray:
    gt = gt_raw.astype(np.int64, copy=False)
    mapped = np.full(gt.shape, IGNORE_INDEX, dtype=np.uint8)
    valid = (gt >= 0) & (gt < len(DENSEPASS19_TO_13))
    mapped[valid] = DENSEPASS19_TO_13[gt[valid]].astype(np.uint8)
    mapped[gt == IGNORE_INDEX] = IGNORE_INDEX
    return mapped


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def copy_normalized_inputs(method_key: str, ids: List[str], pred_files: Dict[str, Path], gt_files: Dict[str, Path], split: str) -> Tuple[Path, Path]:
    base = RAW_ROOT / method_key / split
    pred_out = base / "pred"
    gt_out = base / "gt_densepass13"
    if base.exists():
        shutil.rmtree(base)
    pred_out.mkdir(parents=True, exist_ok=True)
    gt_out.mkdir(parents=True, exist_ok=True)
    for mid in ids:
        pred = load_mask(pred_files[mid]).astype(np.uint8)
        gt = map_densepass13(load_mask(gt_files[mid]))
        save_png(pred, pred_out / f"{mid}.png")
        save_png(gt, gt_out / f"{mid}.png")
    return pred_out, gt_out


def run_eval_region_metrics(pred_dir: Path, gt_dir: Path, out_prefix: Path) -> Dict:
    cmd = [
        PYTHON,
        "experiments/geo_memory/scripts/eval_region_metrics.py",
        "--pred-dir", str(pred_dir),
        "--gt-dir", str(gt_dir),
        "--ignore-index", str(IGNORE_INDEX),
        "--seam-width", str(SEAM_WIDTH),
        "--polar-band-ratio", str(POLAR_BAND_RATIO),
        "--num-classes", str(NUM_CLASSES),
        "--output-json", str(out_prefix.with_suffix(".json")),
        "--output-md", str(out_prefix.with_suffix(".md")),
        "--self-test",
    ]
    proc = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = {
        "command": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "output_json": str(out_prefix.with_suffix(".json")),
        "output_md": str(out_prefix.with_suffix(".md")),
    }
    if out_prefix.with_suffix(".json").exists():
        result["parsed"] = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    return result


def polar_error_ratio(ids: List[str], pred_files: Dict[str, Path], gt_files: Dict[str, Path]) -> float | str:
    wrong = 0
    total = 0
    for mid in ids:
        pred = load_mask(pred_files[mid]).astype(np.int64)
        gt = map_densepass13(load_mask(gt_files[mid])).astype(np.int64)
        if pred.shape != gt.shape:
            continue
        region = build_region_mask(gt.shape[0], gt.shape[1], seam_width=SEAM_WIDTH, polar_band_ratio=POLAR_BAND_RATIO)
        valid = (gt != IGNORE_INDEX) & np.isin(region, [REGION_POLAR, REGION_SEAM_POLAR])
        total += int(valid.sum())
        if valid.any():
            wrong += int((pred[valid] != gt[valid]).sum())
    return float(wrong / total) if total else "N/A"


def label_range_report(ids: List[str], pred_files: Dict[str, Path], gt_files: Dict[str, Path]) -> Dict:
    pred_values = set()
    gt_raw_values = set()
    gt_mapped_values = set()
    pred_oor_on_valid_gt = 0
    valid_gt_pixels = 0
    ignore_pixels = 0
    for mid in ids:
        pred = load_mask(pred_files[mid]).astype(np.int64)
        gt_raw = load_mask(gt_files[mid]).astype(np.int64)
        gt = map_densepass13(gt_raw).astype(np.int64)
        if pred.shape != gt.shape:
            continue
        pred_values.update(int(x) for x in np.unique(pred))
        gt_raw_values.update(int(x) for x in np.unique(gt_raw))
        gt_mapped_values.update(int(x) for x in np.unique(gt))
        valid = (gt >= 0) & (gt < NUM_CLASSES)
        valid_gt_pixels += int(valid.sum())
        ignore_pixels += int((gt == IGNORE_INDEX).sum())
        if valid.any():
            pv = pred[valid]
            pred_oor_on_valid_gt += int(((pv < 0) | (pv >= NUM_CLASSES)).sum())
    def summarize(vals):
        vals = sorted(vals)
        return {"unique": vals, "min": vals[0] if vals else "N/A", "max": vals[-1] if vals else "N/A"}
    return {
        "prediction": summarize(pred_values),
        "gt_raw_19class": summarize(gt_raw_values),
        "gt_mapped_13class": summarize(gt_mapped_values),
        "valid_gt_pixels": int(valid_gt_pixels),
        "ignore_pixels": int(ignore_pixels),
        "pred_out_of_range_on_valid_gt_pixels": int(pred_oor_on_valid_gt),
        "class_mapping": CLASS_MAPPING_NOTE,
    }


def pct(x):
    if isinstance(x, (int, float)):
        return f"{100.0 * float(x):.2f}"
    return "N/A"


def extract_region(eval_result: Dict, region: str) -> float | str:
    try:
        v = eval_result["parsed"]["directory_eval"]["regions"][region]["mean_miou"]
        return v
    except Exception:
        return "N/A"


def main() -> int:
    gt_files = list_masks(GT_DIR, "gt")
    output = {
        "status": "ok",
        "run_level": "L2 diagnosis",
        "training_run": False,
        "model_changed": False,
        "memory_selection_changed": False,
        "loss_changed": False,
        "gt_dir": str(GT_DIR),
        "gt_mask_count": len(gt_files),
        "ignore_index": IGNORE_INDEX,
        "num_classes": NUM_CLASSES,
        "class_mapping": CLASS_MAPPING_NOTE,
        "methods": {},
    }
    table_rows = []

    for method in METHODS:
        key = method["key"]
        pred_dir, pred_status = choose_pred_dir(method["pred_candidates"])
        pred_files = list_masks(pred_dir, "pred") if pred_dir else {}
        pred_ids = set(pred_files)
        gt_ids = set(gt_files)
        matched = sorted(pred_ids & gt_ids, key=natural_key)
        missing_pred = sorted(gt_ids - pred_ids, key=natural_key)
        missing_gt = sorted(pred_ids - gt_ids, key=natural_key)
        size_mismatches = []
        same_size_ids = []
        for mid in matched:
            pred_shape = tuple(load_mask(pred_files[mid]).shape)
            gt_shape = tuple(load_mask(gt_files[mid]).shape)
            if pred_shape != gt_shape:
                size_mismatches.append({"id": mid, "pred_shape": pred_shape, "gt_shape": gt_shape})
            else:
                same_size_ids.append(mid)
        rec = {
            "method": method["table_name"],
            "prediction_candidates": [str(p) for p in method["pred_candidates"]],
            "selected_prediction_dir": str(pred_dir) if pred_dir else "N/A",
            "prediction_status": pred_status,
            "gt_dir": str(GT_DIR),
            "matched": len(matched),
            "missing_pred": len(missing_pred),
            "missing_gt": len(missing_gt),
            "size_mismatch": len(size_mismatches),
            "matched_ids": matched,
            "missing_pred_sample": missing_pred[:20],
            "missing_gt_sample": missing_gt[:20],
            "size_mismatch_sample": size_mismatches[:20],
            "dry_run": "N/A",
            "full_eval": "N/A",
            "polar_error_ratio": "N/A",
            "label_range_check": "N/A",
        }

        source_path = f"pred={pred_dir if pred_dir else 'N/A'}; gt={GT_DIR}"
        if pred_dir is None:
            source_path = f"missing prediction path(s): {', '.join(str(p) for p in method['pred_candidates'])}; gt={GT_DIR}"
            table_rows.append([method["table_name"], "N/A", "N/A", "N/A", "N/A", "N/A", source_path])
            output["methods"][key] = rec
            continue
        if not same_size_ids:
            table_rows.append([method["table_name"], "N/A", "N/A", "N/A", "N/A", "N/A", source_path + "; no same-size matched masks"])
            output["methods"][key] = rec
            continue

        rec["label_range_check"] = label_range_report(same_size_ids, pred_files, gt_files)
        dry_ids = same_size_ids[:DRY_RUN_N]
        dry_pred, dry_gt = copy_normalized_inputs(key, dry_ids, pred_files, gt_files, "dry_run")
        dry_eval = run_eval_region_metrics(dry_pred, dry_gt, RAW_ROOT / key / "dry_run_region_metrics")
        rec["dry_run"] = {"ids": dry_ids, **dry_eval}
        dry_ok = dry_eval.get("returncode") == 0 and dry_eval.get("parsed", {}).get("directory_eval", {}).get("status") == "ok"
        dry_failures = dry_eval.get("parsed", {}).get("directory_eval", {}).get("failed_masks", [])
        if not dry_ok or dry_failures:
            table_rows.append([method["table_name"], "N/A", "N/A", "N/A", "N/A", "N/A", source_path + "; dry-run failed"])
            output["methods"][key] = rec
            continue

        full_pred, full_gt = copy_normalized_inputs(key, same_size_ids, pred_files, gt_files, "full_available")
        full_eval = run_eval_region_metrics(full_pred, full_gt, RAW_ROOT / key / "full_region_metrics")
        rec["full_eval"] = {"ids": same_size_ids, **full_eval}
        rec["polar_error_ratio"] = polar_error_ratio(same_size_ids, pred_files, gt_files)
        overall = extract_region(full_eval, "overall")
        seam = extract_region(full_eval, "seam")
        polar = extract_region(full_eval, "polar")
        equator = extract_region(full_eval, "equator")
        table_rows.append([
            method["table_name"], pct(overall), pct(seam), pct(polar), pct(equator), pct(rec["polar_error_ratio"]),
            f"pred={pred_dir}; gt={GT_DIR}; evaluated={len(same_size_ids)} matched masks; missing_pred={len(missing_pred)}",
        ])
        output["methods"][key] = rec

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Existing Result Region Metrics",
        "",
        "Values are percentages. GT uses DensePASS13 mapping before evaluation; ignore_index=255 is excluded.",
        "",
        "| Method | Overall mIoU | Seam mIoU | Polar mIoU | Equator mIoU | Polar Error | Source Path |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    lines += [
        "",
        "## Input Inventory",
        "",
        "| Method | Selected Prediction Dir | GT Dir | matched | missing_pred | missing_gt | size_mismatch | Dry-run | Full Eval |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for key, rec in output["methods"].items():
        dry = rec["dry_run"]
        full = rec["full_eval"]
        dry_status = "N/A" if dry == "N/A" else f"rc={dry.get('returncode')}, status={dry.get('parsed', {}).get('directory_eval', {}).get('status')}"
        full_status = "N/A" if full == "N/A" else f"rc={full.get('returncode')}, status={full.get('parsed', {}).get('directory_eval', {}).get('status')}"
        lines.append(
            f"| {rec['method']} | {rec['selected_prediction_dir']} | {rec['gt_dir']} | {rec['matched']} | {rec['missing_pred']} | {rec['missing_gt']} | {rec['size_mismatch']} | {dry_status} | {full_status} |"
        )
    OUT_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    note = [
        "# Step 01-4 Existing Region Metrics Diagnosis",
        "",
        "## 本次做了什么",
        "",
        "- 先完成 input inventory，再对可用 matched 样本做 dry-run，dry-run 通过后才运行该 method 的 available-mask region metrics。",
        "- 扫描 GM0 baseline_omnisam 与 GM1 tecmem_base / A_Dtrain_Etrain 的 prediction mask 路径，并使用 DensePASS GT。",
        "- 对每个 method 统计 matched / missing_pred / missing_gt / size_mismatch。",
        "- 检查 label id 范围、ignore_index、DensePASS13 class mapping 与原 OmniSAM validation 保持一致。",
        "",
        "## 输出文件",
        "",
        f"- `{OUT_JSON}`",
        f"- `{OUT_TABLE}`",
        f"- `{OUT_NOTE}`",
        "- dry-run/full raw outputs under `experiments/geo_memory/raw_results/GM*/`",
        "",
        "## Class mapping / ignore_index",
        "",
        f"- {CLASS_MAPPING_NOTE}",
        "- Raw DensePASS GT `labelTrainIds` is mapped to 13 classes before region metrics; using raw 19-class GT directly would be inconsistent.",
        "- `ignore_index=255` pixels are excluded from intersection/union and polar error.",
        "",
        "## 关键结论",
        "",
    ]
    gm0 = output["methods"].get("GM0_baseline_omnisam", {})
    gm1 = output["methods"].get("GM1_A_Dtrain_Etrain", {})
    if gm0.get("matched", 0) == 0:
        note.append("- GM0 baseline_omnisam prediction masks were not found in the scanned final/confirmed prediction paths, so region metrics are recorded as N/A rather than fabricated.")
    if gm1.get("matched", 0) > 0:
        note.append(f"- GM1 A_Dtrain_Etrain has {gm1.get('matched')} matched confirmed prediction masks; {gm1.get('missing_pred')} GT masks have no corresponding confirmed prediction mask, so metrics are diagnostic on available masks only.")
    note += [
        "- 因 GM0 缺少 comparable prediction masks，当前不能可靠回答 memory 压缩后 seam/polar 是否相对 GM0 退化；该项标记为证据不足。",
        "- GM1 available-mask diagnostics can motivate Geo-TeCMA region-wise analysis, but claims must be limited to the matched subset unless GM0/full predictions are exported later.",
        "",
        "## 禁止项确认",
        "",
        "- training_run: false",
        "- full_training_run: false",
        "- model_changed: false",
        "- memory_selection_changed: false",
        "- loss_changed: false",
        "- inference_graph_changed: false",
    ]
    OUT_NOTE.write_text("\n".join(note) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "outputs": [str(OUT_JSON), str(OUT_TABLE), str(OUT_NOTE)], "methods": {k: {kk: vv for kk, vv in v.items() if kk in ['method','matched','missing_pred','missing_gt','size_mismatch','selected_prediction_dir']} for k, v in output['methods'].items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
