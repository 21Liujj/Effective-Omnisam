#!/usr/bin/env python3
"""Unit tests for Geo-Memory geometry ops.

Run level: L1 unit-test only. This file does not import or call training,
memory selection, sparse retrieval, attention, or loss code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.geo_memory.geometry_ops import (  # noqa: E402
    REGION_EQUATOR,
    REGION_ID_TO_NAME,
    REGION_POLAR,
    REGION_SEAM,
    REGION_SEAM_POLAR,
    assign_region_from_box,
    assign_region_from_xy,
    build_region_mask,
    compute_latitude_prior,
    compute_seam_prior,
    compute_token_row_region_counts,
    resize_region_map_to_token_grid,
)


def assert_true(name: str, cond: bool):
    if not cond:
        raise AssertionError(name)


def count_regions(mask):
    vals, cnts = np.unique(mask, return_counts=True)
    return {REGION_ID_TO_NAME[int(v)]: int(c) for v, c in zip(vals, cnts)}


def main():
    report = {"checks": []}

    lat = compute_latitude_prior(64, 64)
    assert_true("top rows high latitude", float(lat[0].mean()) > float(lat[32].mean()))
    assert_true("bottom rows high latitude", float(lat[-1].mean()) > float(lat[32].mean()))
    assert_true("middle rows equator low latitude", float(lat[31:33].mean()) < float(lat[:4].mean()))
    report["checks"].append("top/bottom rows marked high-latitude relative to middle rows")

    seam_prior = compute_seam_prior(64, 64, seam_width=4, tau=2)
    assert_true("left seam prior high", float(seam_prior[:, 0].mean()) > float(seam_prior[:, 32].mean()))
    assert_true("right seam prior high", float(seam_prior[:, -1].mean()) > float(seam_prior[:, 32].mean()))
    report["checks"].append("left/right columns have stronger seam prior than middle columns")

    mask64 = build_region_mask(64, 64, seam_width=4, polar_band_ratio=0.125)
    counts64 = count_regions(mask64)
    assert_true("corner seam_polar", mask64[0, 0] == REGION_SEAM_POLAR)
    assert_true("top non-seam polar", mask64[0, 32] == REGION_POLAR)
    assert_true("middle left seam", mask64[32, 0] == REGION_SEAM)
    assert_true("middle center equator", mask64[32, 32] == REGION_EQUATOR)
    assert_true("region sum equals 4096", int(sum(counts64.values())) == 4096)
    assert_true("no unknown in valid token grid", "unknown" not in counts64)
    report["checks"].append("64x64 region ids have clear priority and sum to 4096")

    mask1024 = build_region_mask(1024, 1024, seam_width=64, polar_band_ratio=0.125)
    token_mask = resize_region_map_to_token_grid(mask1024, 64, 64)
    token_counts = count_regions(token_mask)
    assert_true("resize shape 64x64", token_mask.shape == (64, 64))
    assert_true("resize no nan", not np.isnan(token_mask.astype(np.float32)).any())
    assert_true("resized token sum equals 4096", int(sum(token_counts.values())) == 4096)
    assert_true("resized corner seam_polar", token_mask[0, 0] == REGION_SEAM_POLAR)
    assert_true("resized top polar", token_mask[0, 32] == REGION_POLAR)
    assert_true("resized left seam", token_mask[32, 0] == REGION_SEAM)
    assert_true("resized equator", token_mask[32, 32] == REGION_EQUATOR)
    report["checks"].append("1024x1024 -> 64x64 categorical resize preserves expected regions")

    assert_true("assign xy corner seam_polar", assign_region_from_xy(0, 0, 64, 64, 4, 0.125) == "seam_polar")
    assert_true("assign xy center equator", assign_region_from_xy(32, 32, 64, 64, 4, 0.125) == "equator")
    assert_true("assign xy out of bounds unknown", assign_region_from_xy(-1, 32, 64, 64, 4, 0.125) == "unknown")
    assert_true("assign box seam", assign_region_from_box((0, 30, 2, 34), 64, 64, 4, 0.125) == "seam")
    assert_true("assign box polar", assign_region_from_box((30, 0, 34, 2), 64, 64, 4, 0.125) == "polar")
    assert_true("assign box seam_polar", assign_region_from_box((0, 0, 2, 2), 64, 64, 4, 0.125) == "seam_polar")
    assert_true("assign box equator", assign_region_from_box((30, 30, 34, 34), 64, 64, 4, 0.125) == "equator")
    assert_true("full-height middle box not polar/equator", assign_region_from_box((30, 0, 34, 64), 64, 64, 4, 0.125) == "unknown")
    assert_true("full-height seam box keeps seam only", assign_region_from_box((0, 0, 2, 64), 64, 64, 4, 0.125) == "seam")
    report["checks"].append("assign_region_from_xy/box returns seam, polar, seam_polar, equator, unknown")
    report["checks"].append("full-height boxes are not collapsed to polar/equator; non-seam full-height returns unknown")

    row_counts = compute_token_row_region_counts(64, 64, polar_band_ratio=0.125)
    assert_true("top middle bottom sum", sum(row_counts.values()) == 4096)
    assert_true("top count positive", row_counts["top"] > 0)
    assert_true("middle count positive", row_counts["middle"] > 0)
    assert_true("bottom count positive", row_counts["bottom"] > 0)
    report["checks"].append("token-row top/middle/bottom counts sum to 4096")

    report.update({
        "status": "passed",
        "mask64_counts": counts64,
        "resized_1024_to_64_counts": token_counts,
        "token_row_counts_64x64": row_counts,
        "notes": [
            "Latitude/polar/equator token diagnostics depend on token_row only.",
            "Seam diagnostics depend on horizontal coordinate and must be gated by reliable true ERP x-range.",
            "Box-level polar/equator assignment is coarse and unreliable for full-height patches; token-row labels are preferred.",
            "Task 1-2 adds explicit full-height box tests: non-seam full-height boxes return unknown, seam full-height boxes keep seam only.",
        ],
        "training_run": False,
        "memory_selection_changed": False,
        "loss_changed": False,
    })

    out_dir = Path("experiments/geo_memory/unit_tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_geometry_ops_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Geometry Ops Unit Test Report",
        "",
        "Status: passed",
        "",
        "## Key Checks",
    ]
    md.extend(f"- {c}" for c in report["checks"])
    md.extend([
        "",
        "## 64x64 Region Counts",
        "",
        json.dumps(counts64, ensure_ascii=False, indent=2),
        "",
        "## 1024x1024 -> 64x64 Region Counts",
        "",
        json.dumps(token_counts, ensure_ascii=False, indent=2),
        "",
        "## Token Row Counts",
        "",
        json.dumps(row_counts, ensure_ascii=False, indent=2),
        "",
        "No training, memory selection, sparse retrieval, attention, or loss code was run.",
    ])
    (out_dir / "test_geometry_ops_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
