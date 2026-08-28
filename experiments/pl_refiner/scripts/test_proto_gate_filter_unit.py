#!/usr/bin/env python3
"""Unit tests for R4 class-wise percentile proto-gate filtering helpers.

L1 unit-test only: synthetic tensors, no training, no run_adaptation, no checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptation.pseudo_label.prototype_gate import (  # noqa: E402
    apply_proto_gate_to_valid_mask,
    compute_classwise_percentile_keep_mask,
    upsample_keep_mask_to_label_resolution,
)

COMMON = [0, 1, 2, 8, 9, 10, 12]
RARE = [3, 4, 5, 11]
EXTREME = [6, 7]
REQUIRED_DIAG_KEYS = (
    "proto_gate_filter_ratio_total",
    "proto_gate_keep_ratio_total",
    "proto_gate_filter_ratio_common",
    "proto_gate_filter_ratio_rare",
    "proto_gate_filter_ratio_extreme_rare",
    "proto_gate_filtered_pixels_by_class",
    "proto_gate_kept_pixels_by_class",
    "proto_gate_total_pixels_by_class",
    "proto_gate_threshold_by_class",
    "proto_gate_fallback_classes",
    "proto_gate_fallback_reasons",
    "proto_gate_common_filter_percentile",
    "proto_gate_rare_filter_percentile",
    "proto_gate_extreme_rare_filter_percentile",
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _base_tensors():
    # 2*16*32 = 1024 locations.  Build deterministic class blocks.
    sim_map = torch.full((2, 16, 32), 0.9, dtype=torch.float32)
    sim_valid = torch.ones((2, 16, 32), dtype=torch.bool)
    label_f = torch.full((2, 16, 32), 12, dtype=torch.long)
    flat_sim = sim_map.reshape(-1)
    flat_label = label_f.reshape(-1)
    flat_valid = sim_valid.reshape(-1)

    # common class 0: 200 pixels, should filter ~10%.
    flat_label[0:200] = 0
    flat_sim[0:200] = torch.linspace(0.50, 0.99, 200)
    # rare class 3: 200 pixels, should filter ~5%.
    flat_label[200:400] = 3
    flat_sim[200:400] = torch.linspace(0.45, 0.98, 200)
    # extreme rare class 6: 150 pixels, should not filter.
    flat_label[400:550] = 6
    flat_sim[400:550] = torch.linspace(0.40, 0.97, 150)
    # proto missing class 4: 120 pixels, should fallback keep.
    flat_label[550:670] = 4
    flat_sim[550:670] = torch.linspace(0.30, 0.96, 120)
    # too few class 5: 50 pixels, should fallback keep below min_filter_pixels.
    flat_label[670:720] = 5
    flat_sim[670:720] = torch.linspace(0.20, 0.95, 50)
    # class 1: 105 pixels, common, but min_keep guard can be tested with high min_keep.
    flat_label[720:825] = 1
    flat_sim[720:825] = torch.linspace(0.10, 0.90, 105)
    # remaining locations are common class 12, with enough pixels to filter normally.
    flat_label[825:850] = 2
    flat_sim[825:850] = 0.0
    flat_valid[825:850] = False

    proto_valid = torch.ones(13, dtype=torch.bool)
    proto_valid[4] = False
    return sim_map, sim_valid, label_f, proto_valid


def test_classwise_filter_core() -> None:
    sim_map, sim_valid, label_f, proto_valid = _base_tensors()
    keep_f, filter_f, diag = compute_classwise_percentile_keep_mask(
        sim_map, sim_valid, label_f, proto_valid,
        num_classes=13, common_classes=COMMON, rare_classes=RARE, extreme_rare_classes=EXTREME,
        min_filter_pixels_per_class=100, min_keep_pixels_per_class=10,
    )
    _assert(keep_f.shape == (2, 16, 32), "keep_f shape mismatch")
    _assert(filter_f.shape == (2, 16, 32), "filter_f shape mismatch")
    _assert(keep_f.dtype == torch.bool and filter_f.dtype == torch.bool, "filter masks must be bool")
    _assert(not torch.isnan(sim_map).any().item(), "sim_map contains NaN")

    # Common class 0 filters bottom 10%; quantile inclusive may filter ~20-21.
    _assert(diag["proto_gate_filtered_pixels_by_class"][0] > 0, "common class 0 should filter")
    _assert(15 <= diag["proto_gate_filtered_pixels_by_class"][0] <= 25, "common class 0 should filter about 10%")
    # Rare class 3 filters bottom 5%.
    _assert(diag["proto_gate_filtered_pixels_by_class"][3] > 0, "rare class 3 should filter")
    _assert(8 <= diag["proto_gate_filtered_pixels_by_class"][3] <= 12, "rare class 3 should filter about 5%")
    # Extreme rare class 6 fallback keep.
    _assert(diag["proto_gate_filtered_pixels_by_class"][6] == 0, "extreme rare should not filter")
    _assert(diag["proto_gate_fallback_reasons"][6] == "extreme_rare_keep", "extreme rare fallback reason mismatch")
    # proto_valid False class 4 fallback keep.
    _assert(diag["proto_gate_filtered_pixels_by_class"][4] == 0, "proto missing class should not filter")
    _assert(diag["proto_gate_fallback_reasons"][4] == "proto_missing", "proto missing fallback reason mismatch")
    # too few valid pixels class 5 fallback keep.
    _assert(diag["proto_gate_filtered_pixels_by_class"][5] == 0, "small class should not filter")
    _assert(str(diag["proto_gate_fallback_reasons"][5]).startswith("below_min_filter_pixels"), "small class fallback reason mismatch")
    # sim_valid false positions cannot be filtered and should be kept.
    _assert(not filter_f[~sim_valid].any().item(), "sim_valid=False locations must not be filtered")
    _assert(keep_f[~sim_valid].all().item(), "sim_valid=False locations should fallback keep")
    for key in REQUIRED_DIAG_KEYS:
        _assert(key in diag, f"missing diag key {key}")


def test_min_keep_fallback_and_apply_highres() -> None:
    sim_map, sim_valid, label_f, proto_valid = _base_tensors()
    keep_f, filter_f, diag = compute_classwise_percentile_keep_mask(
        sim_map, sim_valid, label_f, proto_valid,
        num_classes=13, common_classes=COMMON, rare_classes=RARE, extreme_rare_classes=EXTREME,
        min_filter_pixels_per_class=100, min_keep_pixels_per_class=100,
    )
    # class 1 has 105 pixels; bottom 10% would keep ~94 < 100, so fallback keep.
    _assert(diag["proto_gate_filtered_pixels_by_class"][1] == 0, "min_keep class should fallback keep")
    _assert(str(diag["proto_gate_fallback_reasons"][1]).startswith("below_min_keep_pixels"), "min_keep fallback reason mismatch")

    keep_highres = upsample_keep_mask_to_label_resolution(keep_f, (64, 128))
    _assert(keep_highres.shape == (2, 64, 128), "keep_highres shape mismatch")
    _assert(keep_highres.dtype == torch.bool, "keep_highres dtype must be bool")
    r2_valid_mask = torch.ones(2, 64, 128, dtype=torch.bool)
    r2_valid_mask[:, 0:2, :] = False
    final_valid_mask = apply_proto_gate_to_valid_mask(r2_valid_mask, keep_highres)
    _assert(final_valid_mask.shape == (2, 64, 128), "final_valid_mask shape mismatch")
    _assert(final_valid_mask.dtype == torch.bool, "final_valid_mask dtype must be bool")
    _assert(((final_valid_mask & ~r2_valid_mask).sum().item() == 0), "final_valid_mask must be <= r2_valid_mask")


def main() -> None:
    tests = [
        ("classwise_filter_core", test_classwise_filter_core),
        ("min_keep_fallback_and_apply_highres", test_min_keep_fallback_and_apply_highres),
    ]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print("ALL_PROTO_GATE_FILTER_UNIT_TESTS_PASSED")


if __name__ == "__main__":
    main()
