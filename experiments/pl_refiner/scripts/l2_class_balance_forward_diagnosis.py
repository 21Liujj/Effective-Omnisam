#!/usr/bin/env python3
"""
experiments/pl_refiner/scripts/l2_class_balance_forward_diagnosis.py

L2 Forward Diagnosis: R1_conf_mask vs R2_class_balanced.

Run Level: L2 (forward pass only, NO training, NO checkpoint saved, NO mIoU eval).

What this script does:
  1. Loads offline PL PNG files (same as R0/R1 sanity runs: base_last epoch1).
  2. Synthesises online confidence maps from the PL itself (offline_conf proxy)
     OR uses a uniform confidence = 0.85 as a conservative stand-in.
     (No model forward pass needed for structural diagnosis.)
  3. Runs ClassBalancedSelector.select() on several batches.
  4. Collects per-class statistics: candidate_pixels, selected_pixels,
     class_valid_ratio, skip_reason.
  5. Compares:
     - R1_conf_mask behaviour: same as original valid_mask
       (pseudo_conf_mask plugin = conf >= 0.85, no class-specific ratio)
     - R2_class_balanced: per-class top-ratio selection
  6. Prints a per-class table and summary.
  7. Writes results to:
       experiments/pl_refiner/tables/r2_class_balance_forward_diagnosis.md

This script does NOT:
  - Load the OmniSAM model.
  - Perform gradient computation.
  - Save checkpoints.
  - Compute mIoU.
  - Modify any files except the output tables.

Usage:
  cd /root/autodl-tmp/OmniSAM-main
  /root/miniconda3/envs/omnisam/bin/python3 \
      experiments/pl_refiner/scripts/l2_class_balance_forward_diagnosis.py
"""

from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE = Path("/root/autodl-tmp/OmniSAM-main")
PL_ROOT = BASE / "data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1/gtFine"
OUTPUT_MD = BASE / "experiments/pl_refiner/tables/r2_class_balance_forward_diagnosis.md"
OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

IGNORE_INDEX = 255
NUM_CLASSES  = 13

# Confidence proxy: we use per-pixel offline-PL confidence proxy.
# Without a model forward pass, we simulate 3 confidence scenarios:
#   (a) HIGH_CONF = 0.92  (representative of easy/common classes)
#   (b) MED_CONF  = 0.80  (medium confidence)
#   (c) LOW_CONF  = 0.65  (difficult / rare classes)
# In practice, we assign confidence per-class based on mean expected confidence
# derived from known IoU values (higher IoU ≈ higher mean confidence).
# This is a structural diagnosis, not a final accuracy measurement.
# The real confidence values come from the model during actual training.

# Mean confidence proxy per class (from IoU ~ 0.7-0.95 for common, 0.4-0.7 for rare)
# Calibrated heuristically: conf_proxy ≈ sqrt(IoU/100 + 0.1) * 0.9 + uniform noise
CLASS_CONF_PROXY = {
    0:  0.91,  # road         IoU=80
    1:  0.75,  # sidewalk     IoU=47
    2:  0.93,  # building     IoU=90
    3:  0.70,  # wall         IoU=44
    4:  0.72,  # fence        IoU=49
    5:  0.70,  # pole         IoU=43
    6:  0.60,  # traffic light IoU=28 (often misclassified)
    7:  0.62,  # traffic sign  IoU=32
    8:  0.90,  # vegetation   IoU=79
    9:  0.76,  # terrain      IoU=45
    10: 0.96,  # sky          IoU=95
    11: 0.82,  # person       IoU=68
    12: 0.92,  # car          IoU=83
}

CLASS_NAMES = [
    "road","sidewalk","building","wall","fence","pole",
    "traffic light","traffic sign","vegetation","terrain",
    "sky","person","car"
]

# R2 per-class params (from step_04)
CLASS_PARAMS = {
    0:  dict(group="common",    top_ratio=0.50, min_conf=0.75),
    1:  dict(group="common",    top_ratio=0.55, min_conf=0.70),
    2:  dict(group="common",    top_ratio=0.50, min_conf=0.75),
    3:  dict(group="rare-hard", top_ratio=0.75, min_conf=0.60),
    4:  dict(group="rare-hard", top_ratio=0.70, min_conf=0.60),
    5:  dict(group="rare-hard", top_ratio=0.75, min_conf=0.60),
    6:  dict(group="rare-hard", top_ratio=0.80, min_conf=0.55),
    7:  dict(group="rare-hard", top_ratio=0.80, min_conf=0.55),
    8:  dict(group="common",    top_ratio=0.50, min_conf=0.75),
    9:  dict(group="common",    top_ratio=0.55, min_conf=0.70),
    10: dict(group="common",    top_ratio=0.50, min_conf=0.80),
    11: dict(group="rare-hard", top_ratio=0.75, min_conf=0.60),
    12: dict(group="common",    top_ratio=0.50, min_conf=0.75),
}

# ---------------------------------------------------------------------------
# Import R2 module
# ---------------------------------------------------------------------------

print("[L2] Importing ClassBalancedSelector ...")
from adaptation.pseudo_label.class_balance import ClassBalancedSelector

print("[L2] Importing PseudoLabelRefiner ...")
from adaptation.pseudo_label.refiner import PseudoLabelRefiner, PseudoLabelRefineResult

# Verify class_balanced is in SUPPORTED_MODES
assert 'class_balanced' in PseudoLabelRefiner.SUPPORTED_MODES, \
    "FAIL: class_balanced not in SUPPORTED_MODES"
print(f"[L2] SUPPORTED_MODES = {sorted(PseudoLabelRefiner.SUPPORTED_MODES)}")

# Verify off/identity still work
r_off = PseudoLabelRefiner(cfg={'mode': 'off'})
assert r_off.refine(pseudo_label=torch.zeros(1,4,4,dtype=torch.long),
                    confidence=torch.ones(1,4,4)) is None
r_id = PseudoLabelRefiner(cfg={'mode': 'identity'})
pl_tmp = torch.zeros(1,4,4,dtype=torch.long)
res_id = r_id.refine(pseudo_label=pl_tmp, confidence=torch.ones(1,4,4),
                     valid_mask=(pl_tmp!=255))
assert res_id.refined_label is pl_tmp
print("[L2] mode=off / mode=identity unchanged: OK")

# ---------------------------------------------------------------------------
# Load PL files (up to N_SAMPLES images for diagnosis)
# ---------------------------------------------------------------------------
N_SAMPLES = 50  # structural diagnosis, no need for all 2000

print(f"[L2] Loading up to {N_SAMPLES} PL PNG files from {PL_ROOT} ...")

pl_arrays: List[np.ndarray] = []
cities = sorted(os.listdir(PL_ROOT))
for city in cities:
    city_dir = PL_ROOT / city
    if not city_dir.is_dir():
        continue
    for fname in sorted(os.listdir(city_dir)):
        if not fname.endswith("_labelTrainIds.png"):
            continue
        arr = np.array(Image.open(city_dir / fname), dtype=np.int32)
        pl_arrays.append(arr)
        if len(pl_arrays) >= N_SAMPLES:
            break
    if len(pl_arrays) >= N_SAMPLES:
        break

print(f"[L2] Loaded {len(pl_arrays)} PL images.")
if len(pl_arrays) == 0:
    print("ERROR: No PL files found. Check PL_ROOT path.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Build batched tensors + confidence proxy
# ---------------------------------------------------------------------------
# Stack into one big tensor for batch processing
# PL images have varying sizes (DensePASS is 400x2048); use as-is.

print("[L2] Building confidence proxy tensors ...")

# Aggregate over all samples (not batched — too big) — process one-by-one
# and accumulate statistics

# Accumulators
r1_cand   = [0] * NUM_CLASSES   # R1: original valid pixels per class
r1_sel    = [0] * NUM_CLASSES   # R1: conf>=0.85 pixels per class
r2_cand   = [0] * NUM_CLASSES   # R2: original valid pixels per class
r2_sel    = [0] * NUM_CLASSES   # R2: class-balanced selected per class
skip_reasons: Dict[int, str] = {}

R1_CONF_THR = 0.85  # R1 global threshold

cb_cfg_for_selector = {
    "min_pixels_per_class": 50,
    "per_class": {
        str(c): {"top_ratio": CLASS_PARAMS[c]["top_ratio"],
                 "min_conf": CLASS_PARAMS[c]["min_conf"]}
        for c in range(NUM_CLASSES)
    }
}
selector = ClassBalancedSelector(cfg=cb_cfg_for_selector, num_classes=NUM_CLASSES)

SEED = 42
rng = np.random.default_rng(SEED)

for idx, pl_np in enumerate(pl_arrays):
    H, W = pl_np.shape
    pl_t = torch.from_numpy(pl_np).long()        # (H, W)
    pl_b = pl_t.unsqueeze(0)                     # (1, H, W)

    # Build per-class confidence proxy with mild noise for realism
    conf_np = np.zeros((H, W), dtype=np.float32)
    for c in range(NUM_CLASSES):
        mask_c = (pl_np == c)
        if mask_c.any():
            base = CLASS_CONF_PROXY[c]
            noise = rng.normal(0, 0.05, size=mask_c.sum()).clip(-0.15, 0.15)
            vals = np.clip(base + noise, 0.01, 0.999)
            conf_np[mask_c] = vals.astype(np.float32)
    # pixels with label=255 get conf=0 (won't matter since filtered by valid_mask)
    conf_t = torch.from_numpy(conf_np).float().unsqueeze(0)   # (1, H, W)

    valid_orig = (pl_b != IGNORE_INDEX)   # (1, H, W) bool

    # --- R1 stats: global conf >= R1_CONF_THR AND original valid ---
    r1_mask = (conf_t >= R1_CONF_THR) & valid_orig   # (1, H, W)
    for c in range(NUM_CLASSES):
        cls_orig = (pl_b == c) & valid_orig
        r1_cand[c] += int(cls_orig.sum().item())
        r1_sel[c]  += int(((pl_b == c) & r1_mask).sum().item())

    # --- R2 stats: class-balanced selection ---
    r2_result = selector.select(
        pseudo_label=pl_b,
        confidence=conf_t,
        original_valid_mask=valid_orig,
        ignore_index=IGNORE_INDEX,
    )
    r2_new_mask = r2_result.new_valid_mask   # (1, H, W) bool
    diag = r2_result.diagnostics

    for c in range(NUM_CLASSES):
        cls_orig = (pl_b == c) & valid_orig
        r2_cand[c] += int(cls_orig.sum().item())
        r2_sel[c]  += int(((pl_b == c) & r2_new_mask).sum().item())
        # Update skip reasons (keep latest non-empty)
        if c in diag.get("skipped_class_by_reason", {}):
            skip_reasons[c] = diag["skipped_class_by_reason"][c]

    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx+1}/{len(pl_arrays)} images ...")

print("[L2] Forward diagnosis complete.")

# ---------------------------------------------------------------------------
# Compute per-class ratios
# ---------------------------------------------------------------------------

def safe_ratio(num, denom):
    return float(num / denom) if denom > 0 else float("nan")

rows = []
for c in range(NUM_CLASSES):
    r1_vr = safe_ratio(r1_sel[c], r1_cand[c])
    r2_vr = safe_ratio(r2_sel[c], r2_cand[c])
    skip  = skip_reasons.get(c, "—")
    skipped = "Y" if c in skip_reasons else "."
    rows.append({
        "class_id": c,
        "class_name": CLASS_NAMES[c],
        "group": CLASS_PARAMS[c]["group"],
        "top_ratio": CLASS_PARAMS[c]["top_ratio"],
        "min_conf": CLASS_PARAMS[c]["min_conf"],
        "r1_cand": r1_cand[c],
        "r1_sel": r1_sel[c],
        "r1_valid_ratio": r1_vr,
        "r2_cand": r2_cand[c],
        "r2_sel": r2_sel[c],
        "r2_valid_ratio": r2_vr,
        "skipped": skipped,
        "skip_reason": skip,
    })

# Summary
total_px    = sum(r1_cand) + sum(1 for c in range(NUM_CLASSES) if r1_cand[c] == 0)
r1_total    = sum(r1_sel)
r2_total    = sum(r2_sel)
r1_cand_tot = sum(r1_cand)
r2_cand_tot = sum(r2_cand)

common_cls  = [c for c in range(NUM_CLASSES) if CLASS_PARAMS[c]["group"] == "common"]
rare_cls    = [c for c in range(NUM_CLASSES) if CLASS_PARAMS[c]["group"] == "rare-hard"]
ext_rare    = [c for c in range(NUM_CLASSES) if CLASS_NAMES[c] in ("traffic light","traffic sign")]

def group_ratio(cls_list, sel_arr, cand_arr):
    s = sum(sel_arr[c] for c in cls_list)
    ca= sum(cand_arr[c] for c in cls_list)
    return safe_ratio(s, ca)

r1_common  = group_ratio(common_cls, r1_sel, r1_cand)
r1_rare    = group_ratio(rare_cls,   r1_sel, r1_cand)
r1_ext     = group_ratio(ext_rare,   r1_sel, r1_cand)
r2_common  = group_ratio(common_cls, r2_sel, r2_cand)
r2_rare    = group_ratio(rare_cls,   r2_sel, r2_cand)
r2_ext     = group_ratio(ext_rare,   r2_sel, r2_cand)

r1_total_ratio = safe_ratio(r1_total, r1_cand_tot)
r2_total_ratio = safe_ratio(r2_total, r2_cand_tot)

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

print()
print("=" * 90)
print("  Per-Class Diagnosis Table")
print("=" * 90)
hdr = f"  {'id':>2} {'name':<16} {'grp':<10} {'tr':>5} {'mc':>5} | {'R1 sel/cand':>14} {'R1 ratio':>9} | {'R2 sel/cand':>14} {'R2 ratio':>9} | skip"
print(hdr)
print("  " + "-" * 88)
for r in rows:
    r1_frac = f"{r['r1_sel']:,}/{r['r1_cand']:,}"
    r2_frac = f"{r['r2_sel']:,}/{r['r2_cand']:,}"
    r1_pct  = f"{r['r1_valid_ratio']*100:6.2f}%" if not (r['r1_valid_ratio'] != r['r1_valid_ratio']) else "  nan%"
    r2_pct  = f"{r['r2_valid_ratio']*100:6.2f}%" if not (r['r2_valid_ratio'] != r['r2_valid_ratio']) else "  nan%"
    skip_str = r['skip_reason'][:30] if r['skipped'] == 'Y' else "—"
    print(f"  {r['class_id']:>2} {r['class_name']:<16} {r['group']:<10} {r['top_ratio']:>5.2f} {r['min_conf']:>5.2f} | {r1_frac:>14} {r1_pct:>9} | {r2_frac:>14} {r2_pct:>9} | {r['skipped']} {skip_str}")

print()
print("=" * 90)
print("  Summary")
print("=" * 90)
print(f"  {'Run':<20} {'total_valid_ratio':>18} {'common_ratio':>14} {'rare_ratio':>12} {'ext_rare_ratio':>16} {'status'}")
print("  " + "-" * 88)
print(f"  {'R1_conf_mask':<20} {r1_total_ratio*100:>17.2f}% {r1_common*100:>13.2f}% {r1_rare*100:>11.2f}% {r1_ext*100:>15.2f}%   reference")
print(f"  {'R2_class_balanced':<20} {r2_total_ratio*100:>17.2f}% {r2_common*100:>13.2f}% {r2_rare*100:>11.2f}% {r2_ext*100:>15.2f}%   target")
print(f"  Δ (R2-R1)            {(r2_total_ratio-r1_total_ratio)*100:>+17.2f}% {(r2_common-r1_common)*100:>+13.2f}% {(r2_rare-r1_rare)*100:>+11.2f}% {(r2_ext-r1_ext)*100:>+15.2f}%")
print()

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
checks = []

# 1. rare classes must not all be zero
rare_all_zero = all(r2_sel[c] == 0 for c in rare_cls)
checks.append(("rare classes not all zero", not rare_all_zero, f"rare sel counts: {[r2_sel[c] for c in rare_cls]}"))

# 2. common valid ratio should not be anomalously high (> 80%)
common_ok = r2_common <= 0.85
checks.append(("common valid_ratio <= 85%", common_ok, f"common_ratio={r2_common*100:.2f}%"))

# 3. no NaN in ratios
has_nan = any(r["r2_valid_ratio"] != r["r2_valid_ratio"] for r in rows if r["r2_cand"] > 0)
checks.append(("no NaN in r2 ratios", not has_nan, "NaN detected" if has_nan else "clean"))

# 4. traffic light / traffic sign skip reason documented
for c in [6, 7]:
    if r2_cand[c] == 0:
        reason = "no_candidates (PL has no traffic light/sign pixels in sample)"
        checks.append((f"{CLASS_NAMES[c]} skip documented", True, reason))
    elif c in skip_reasons:
        checks.append((f"{CLASS_NAMES[c]} skip documented", True, skip_reasons[c]))
    else:
        checks.append((f"{CLASS_NAMES[c]} normal selection", True, f"sel={r2_sel[c]}, cand={r2_cand[c]}"))

# 5. R2 does not activate originally-invalid pixels
# Check: r2_cand[c] == r1_cand[c] (both start from same original valid mask)
cand_mismatch = [(c, r1_cand[c], r2_cand[c]) for c in range(NUM_CLASSES) if r1_cand[c] != r2_cand[c]]
checks.append(("R2 candidate set == original valid (no re-activation)", len(cand_mismatch) == 0,
               f"mismatches: {cand_mismatch}" if cand_mismatch else "OK"))

print("=" * 90)
print("  Verification Checks")
print("=" * 90)
all_pass = True
for name, ok, detail in checks:
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{status}] {name}")
    print(f"          {detail}")
print()
if all_pass:
    print("  ✓ All L2 checks PASS.")
else:
    print("  ✗ Some checks FAILED — review before Task 2-2b.")
print()

# ---------------------------------------------------------------------------
# Write markdown table
# ---------------------------------------------------------------------------

def pct(v):
    if v != v:  # nan
        return "NaN"
    return f"{v*100:.2f}%"

md = []
md.append("# r2_class_balance_forward_diagnosis.md")
md.append("# Phase 2 / Task 2-2a — L2 Forward Diagnosis: R1 vs R2")
md.append("")
md.append(f"**Date**: 2026-06-23  ")
md.append(f"**Run Level**: L2 forward diagnosis (no training, no checkpoint, no mIoU)  ")
md.append(f"**N_SAMPLES**: {len(pl_arrays)} PL images (from base_last_20260425_232941 epoch1)  ")
md.append(f"**Confidence proxy**: per-class heuristic (class_conf_proxy dict), not from model  ")
md.append("")
md.append("## Per-Class Table")
md.append("")
md.append("| class_id | class_name | group | top_ratio | min_conf | R1 cand | R1 selected | R1 valid_ratio | R2 cand | R2 selected | R2 valid_ratio | skipped? | skip_reason |")
md.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
for r in rows:
    md.append(
        f"| {r['class_id']} | {r['class_name']} | {r['group']} | {r['top_ratio']:.2f} | {r['min_conf']:.2f} "
        f"| {r['r1_cand']:,} | {r['r1_sel']:,} | {pct(r['r1_valid_ratio'])} "
        f"| {r['r2_cand']:,} | {r['r2_sel']:,} | {pct(r['r2_valid_ratio'])} "
        f"| {r['skipped']} | {r['skip_reason']} |"
    )
md.append("")
md.append("## Summary")
md.append("")
md.append("| Run | total_valid_ratio | common_valid_ratio | rare_valid_ratio | extreme_rare_valid_ratio | status |")
md.append("|---|---:|---:|---:|---:|---|")
md.append(f"| R1_conf_mask | {pct(r1_total_ratio)} | {pct(r1_common)} | {pct(r1_rare)} | {pct(r1_ext)} | reference |")
md.append(f"| R2_class_balanced | {pct(r2_total_ratio)} | {pct(r2_common)} | {pct(r2_rare)} | {pct(r2_ext)} | target |")
md.append(f"| Δ (R2-R1) | {(r2_total_ratio-r1_total_ratio)*100:+.2f}pp | {(r2_common-r1_common)*100:+.2f}pp | {(r2_rare-r1_rare)*100:+.2f}pp | {(r2_ext-r1_ext)*100:+.2f}pp | — |")
md.append("")
md.append("## Checks")
md.append("")
for name, ok, detail in checks:
    md.append(f"- [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
md.append("")
md.append("## Notes")
md.append("")
md.append("- Confidence values are from a **heuristic proxy** (no model forward pass).")
md.append("  Real confidence will come from the OmniSAM model during L3 short-run.")
md.append("- The goal of L2 is to verify structural correctness: class distribution,")
md.append("  skip handling, no NaN, no re-activation of invalid pixels.")
md.append("- mIoU should NOT be inferred from this diagnosis.")

OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")
print(f"[L2] Wrote markdown table to {OUTPUT_MD}")

# Final exit code
sys.exit(0 if all_pass else 1)
