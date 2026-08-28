#!/usr/bin/env python3
"""Region metrics for Geo-Memory diagnostics.

Standalone evaluation script. It does not import training/model code and does
not modify memory selection, sparse retrieval, attention, or loss.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.geo_memory.geometry_ops import (  # noqa: E402
    REGION_EQUATOR,
    REGION_ID_TO_NAME,
    REGION_POLAR,
    REGION_SEAM,
    REGION_SEAM_POLAR,
    build_region_mask,
)

REGION_GROUPS = {
    "overall": None,
    "seam": (REGION_SEAM, REGION_SEAM_POLAR),
    "polar": (REGION_POLAR, REGION_SEAM_POLAR),
    "equator": (REGION_EQUATOR,),
}


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.int64)


def list_mask_files(root: Path) -> Dict[str, Path]:
    if not root.exists() or not root.is_dir():
        return {}
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def load_any_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.int64)
    return load_mask(path)


def compute_iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, num_classes: Optional[int] = None) -> Dict:
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: pred={pred.shape}, gt={gt.shape}")
    valid = valid.astype(bool)
    if valid.shape != gt.shape:
        raise ValueError(f"valid shape mismatch: valid={valid.shape}, gt={gt.shape}")
    if not valid.any():
        return {"miou": "N/A", "per_class_iou": {}, "valid_pixels": 0}

    pred_v = pred[valid].astype(np.int64)
    gt_v = gt[valid].astype(np.int64)
    max_label = int(max(pred_v.max(initial=0), gt_v.max(initial=0)))
    if num_classes is None:
        num_classes = max_label + 1
    num_classes = int(num_classes)

    per_class = {}
    ious = []
    for cls in range(num_classes):
        pred_c = pred_v == cls
        gt_c = gt_v == cls
        inter = int(np.logical_and(pred_c, gt_c).sum())
        union = int(np.logical_or(pred_c, gt_c).sum())
        if union == 0:
            per_class[str(cls)] = "N/A"
            continue
        iou = inter / union
        per_class[str(cls)] = float(iou)
        ious.append(iou)
    return {
        "miou": float(np.mean(ious)) if ious else "N/A",
        "per_class_iou": per_class,
        "valid_pixels": int(valid.sum()),
    }


def compute_region_metrics_for_arrays(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_index: int = 255,
    seam_width: int = 32,
    polar_band_ratio: float = 0.15,
    num_classes: Optional[int] = None,
) -> Dict:
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: pred={pred.shape}, gt={gt.shape}")
    H, W = gt.shape
    region_mask = build_region_mask(H, W, seam_width=seam_width, polar_band_ratio=polar_band_ratio)
    valid_base = gt != int(ignore_index)
    out = {
        "image_shape": [int(H), int(W)],
        "ignore_index": int(ignore_index),
        "region_definition": "experiments.geo_memory.geometry_ops.build_region_mask",
        "regions": {},
    }
    for region_name, region_ids in REGION_GROUPS.items():
        if region_ids is None:
            valid = valid_base
        else:
            region_valid = np.isin(region_mask, list(region_ids))
            valid = valid_base & region_valid
        out["regions"][region_name] = compute_iou(pred, gt, valid, num_classes=num_classes)
    out["per_class_iou_by_region_status"] = "implemented"
    return out


def aggregate_region_metrics(items: List[Dict]) -> Dict:
    if not items:
        return {"status": "N/A", "reason": "no matched prediction/GT masks", "regions": {}}
    regions = {}
    for region in REGION_GROUPS:
        vals = []
        valid_pixels = 0
        for item in items:
            r = item["regions"][region]
            valid_pixels += int(r.get("valid_pixels", 0))
            if isinstance(r.get("miou"), (int, float)):
                vals.append(float(r["miou"]))
        regions[region] = {
            "mean_miou": float(np.mean(vals)) if vals else "N/A",
            "valid_pixels": int(valid_pixels),
            "num_images_with_metric": int(len(vals)),
        }
    return {"status": "ok", "num_images": len(items), "regions": regions}


def evaluate_dirs(
    pred_dir: Optional[Path],
    gt_dir: Optional[Path],
    ignore_index: int,
    seam_width: int,
    polar_band_ratio: float,
    num_classes: Optional[int],
) -> Dict:
    if pred_dir is None or gt_dir is None:
        return {"status": "N/A", "reason": "prediction or GT directory was not provided"}
    pred_files = list_mask_files(pred_dir)
    gt_files = list_mask_files(gt_dir)
    if not pred_files:
        return {"status": "N/A", "reason": f"no prediction masks found in {pred_dir}"}
    if not gt_files:
        return {"status": "N/A", "reason": f"no GT masks found in {gt_dir}"}
    keys = sorted(set(pred_files) & set(gt_files))
    if not keys:
        return {
            "status": "N/A",
            "reason": "prediction and GT directories have no matching mask stems",
            "num_pred_masks": len(pred_files),
            "num_gt_masks": len(gt_files),
        }
    items = []
    failures = []
    for key in keys:
        try:
            pred = load_any_mask(pred_files[key])
            gt = load_any_mask(gt_files[key])
            items.append(compute_region_metrics_for_arrays(
                pred=pred,
                gt=gt,
                ignore_index=ignore_index,
                seam_width=seam_width,
                polar_band_ratio=polar_band_ratio,
                num_classes=num_classes,
            ))
        except Exception as exc:
            failures.append({"key": key, "error": str(exc)})
    agg = aggregate_region_metrics(items)
    agg.update({"matched_masks": len(keys), "failed_masks": failures})
    return agg


def synthetic_self_test() -> Dict:
    H, W = 8, 8
    gt = np.zeros((H, W), dtype=np.int64)
    pred = np.zeros((H, W), dtype=np.int64)
    gt[2:6, 2:6] = 1
    pred[2:6, 2:6] = 1
    pred[3, 3] = 0
    gt[0, 4] = 1
    pred[0, 4] = 0
    gt[4, 0] = 1
    pred[4, 0] = 1
    gt[7, 7] = 255
    pred[7, 7] = 1

    metrics = compute_region_metrics_for_arrays(
        pred=pred,
        gt=gt,
        ignore_index=255,
        seam_width=1,
        polar_band_ratio=0.25,
        num_classes=2,
    )
    # Expected values are derived from the explicit synthetic masks above.
    # gt[7, 7] is ignore_index, so pred[7, 7] is excluded from all unions.
    expected = {
        "overall_miou": 0.9231678486997636,
        "seam_miou": 1.0,
        "polar_miou": 0.4838709677419355,
        "equator_miou": 0.9131944444444444,
    }
    actual = {
        "overall_miou": metrics["regions"]["overall"]["miou"],
        "seam_miou": metrics["regions"]["seam"]["miou"],
        "polar_miou": metrics["regions"]["polar"]["miou"],
        "equator_miou": metrics["regions"]["equator"]["miou"],
    }
    checks = {}
    for key, exp in expected.items():
        act = actual[key]
        checks[key] = bool(abs(float(act) - exp) < 1e-9)
    checks["ignore_index_excluded"] = metrics["regions"]["overall"]["valid_pixels"] == 63
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "failed",
        "checks": checks,
        "expected": expected,
        "actual": actual,
        "metrics": metrics,
    }


def write_outputs(result: Dict, json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Region Metrics Sample",
        "",
        f"Status: `{result.get('status', 'unknown')}`",
        "",
        "## Synthetic Self-Test",
        "",
        f"Status: `{result.get('synthetic_self_test', {}).get('status', 'N/A')}`",
        "",
        "| Region | mIoU | Valid Pixels |",
        "|---|---:|---:|",
    ]
    syn_regions = result.get("synthetic_self_test", {}).get("metrics", {}).get("regions", {})
    for region in ["overall", "seam", "polar", "equator"]:
        r = syn_regions.get(region, {})
        lines.append(f"| {region} | {r.get('miou', 'N/A')} | {r.get('valid_pixels', 'N/A')} |")
    lines.extend([
        "",
        "## Directory Evaluation",
        "",
        f"Status: `{result.get('directory_eval', {}).get('status', 'N/A')}`",
        f"Reason: `{result.get('directory_eval', {}).get('reason', '')}`",
        "",
        "## Implementation Notes",
        "",
        "- Region masks reuse `experiments.geo_memory.geometry_ops.build_region_mask`.",
        "- `ignore_index` pixels are excluded from intersection and union before IoU computation.",
        "- Overall region mIoU and per-class IoU by region are implemented for matched masks.",
        "- Missing prediction / GT inputs return `N/A` instead of failing or inventing metrics.",
        "- No training, model, memory selection, sparse retrieval, attention, or loss code is touched.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate seam/polar/equator region metrics.")
    parser.add_argument("--pred-dir", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--seam-width", type=int, default=32)
    parser.add_argument("--polar-band-ratio", type=float, default=0.15)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("experiments/geo_memory/results/region_metrics_sample.json"))
    parser.add_argument("--output-md", type=Path, default=Path("experiments/geo_memory/tables/region_metrics_sample.md"))
    parser.add_argument("--self-test", action="store_true", help="Run synthetic self-test and fail if it does not pass.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    synthetic = synthetic_self_test()
    directory_eval = evaluate_dirs(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        ignore_index=args.ignore_index,
        seam_width=args.seam_width,
        polar_band_ratio=args.polar_band_ratio,
        num_classes=args.num_classes,
    )
    result = {
        "status": "ok" if synthetic["status"] == "passed" else "failed",
        "synthetic_self_test": synthetic,
        "directory_eval": directory_eval,
        "outputs": {"json": str(args.output_json), "md": str(args.output_md)},
        "per_class_iou_by_region_status": "implemented",
        "training_run": False,
        "model_changed": False,
        "memory_selection_changed": False,
        "loss_changed": False,
    }
    write_outputs(result, args.output_json, args.output_md)
    print(json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2))
    if args.self_test and synthetic["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
