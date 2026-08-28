#!/usr/bin/env python3
"""Unit tests for R4 current-batch prototype similarity helpers.

L1 unit-test only: synthetic tensors, no training, no run_adaptation, no checkpoint.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptation.pseudo_label.prototype_gate import (  # noqa: E402
    compute_current_batch_prototypes,
    compute_proto_similarity_map,
    downsample_label_and_mask_to_feature,
    summarize_proto_similarity,
)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_shape_normal_and_outputs() -> None:
    torch.manual_seed(7)
    feat = torch.randn(2, 256, 16, 32)
    pseudo_label = torch.randint(0, 13, (2, 64, 128), dtype=torch.long)
    valid_mask = torch.rand(2, 64, 128) > 0.2
    pseudo_label[:, :2, :2] = 255

    label_f, mask_f = downsample_label_and_mask_to_feature(pseudo_label, valid_mask, (16, 32), ignore_index=255)
    _assert(label_f.shape == (2, 16, 32), "label_f shape mismatch")
    _assert(mask_f.shape == (2, 16, 32), "mask_f shape mismatch")
    _assert(label_f.dtype == torch.long, "label_f dtype must be long")
    _assert(mask_f.dtype == torch.bool, "mask_f dtype must be bool")
    _assert(not mask_f[label_f == 255].any().item(), "ignore pixels must not be valid")

    prototypes, proto_valid, proto_counts = compute_current_batch_prototypes(
        feat, label_f, mask_f, num_classes=13, min_pixels_per_class=10
    )
    _assert(prototypes.shape == (13, 256), "prototypes shape mismatch")
    _assert(proto_valid.shape == (13,), "proto_valid shape mismatch")
    _assert(proto_counts.shape == (13,), "proto_counts shape mismatch")
    _assert(proto_valid.dtype == torch.bool, "proto_valid dtype must be bool")
    _assert(proto_counts.dtype == torch.long, "proto_counts dtype must be long")

    sim_map, sim_valid = compute_proto_similarity_map(feat, label_f, mask_f, prototypes, proto_valid, ignore_index=255)
    _assert(sim_map.shape == (2, 16, 32), "sim_map shape mismatch")
    _assert(sim_valid.shape == (2, 16, 32), "sim_valid shape mismatch")
    _assert(sim_valid.dtype == torch.bool, "sim_valid dtype must be bool")
    _assert(not torch.isnan(sim_map).any().item(), "sim_map must not contain NaN")
    _assert(float(sim_map.min().item()) >= -1.000001, "similarity below -1")
    _assert(float(sim_map.max().item()) <= 1.000001, "similarity above 1")

    summary = summarize_proto_similarity(sim_map, sim_valid, label_f, 13, proto_valid=proto_valid, proto_counts=proto_counts)
    for key in (
        "sim_mean", "sim_std", "sim_min", "sim_max", "sim_valid_ratio",
        "proto_valid_classes", "proto_missing_classes", "proto_counts_by_class",
        "sim_mean_by_class", "low_sim_ratio_at_0.2", "low_sim_ratio_at_0.4", "low_sim_ratio_at_0.6",
    ):
        _assert(key in summary, f"summary missing {key}")


def test_b1hw_label_supported() -> None:
    torch.manual_seed(11)
    feat = torch.randn(2, 256, 16, 32)
    pseudo_label = torch.randint(0, 13, (2, 1, 64, 128), dtype=torch.long)
    valid_mask = torch.ones(2, 1, 64, 128, dtype=torch.bool)
    label_f, mask_f = downsample_label_and_mask_to_feature(pseudo_label, valid_mask, (16, 32), ignore_index=255)
    _assert(label_f.shape == (2, 16, 32), "B1HW label squeeze failed")
    _assert(mask_f.shape == (2, 16, 32), "B1HW mask squeeze failed")
    prototypes, proto_valid, _ = compute_current_batch_prototypes(feat, label_f, mask_f, 13, 10)
    sim_map, sim_valid = compute_proto_similarity_map(feat, label_f, mask_f, prototypes, proto_valid)
    _assert(sim_map.shape == (2, 16, 32), "B1HW sim_map shape mismatch")
    _assert(sim_valid.shape == (2, 16, 32), "B1HW sim_valid shape mismatch")


def test_missing_class_and_ignore() -> None:
    torch.manual_seed(13)
    feat = torch.randn(2, 256, 16, 32)
    label_f = torch.randint(1, 13, (2, 16, 32), dtype=torch.long)  # class 0 absent initially
    mask_f = torch.ones(2, 16, 32, dtype=torch.bool)
    label_f[0, 0, 0] = 0
    label_f[0, 0, 1] = 0
    label_f[0, 0, 2] = 255
    mask_f[0, 0, 2] = True

    prototypes, proto_valid, proto_counts = compute_current_batch_prototypes(
        feat, label_f, mask_f, num_classes=13, min_pixels_per_class=10
    )
    _assert(int(proto_counts[0].item()) == 2, "class 0 count should ignore 255 and count two pixels")
    _assert(not bool(proto_valid[0].item()), "class with fewer than min pixels must be invalid")
    _assert(torch.allclose(prototypes[0], torch.zeros_like(prototypes[0])), "invalid prototype must be zero")

    sim_map, sim_valid = compute_proto_similarity_map(feat, label_f, mask_f, prototypes, proto_valid, ignore_index=255)
    _assert(not sim_valid[label_f == 0].any().item(), "missing class locations must have sim_valid=False")
    _assert(not sim_valid[label_f == 255].any().item(), "ignore_index locations must have sim_valid=False")
    _assert(not torch.isnan(sim_map).any().item(), "sim_map must not contain NaN for missing/ignore")


def test_all_false_valid_mask() -> None:
    torch.manual_seed(17)
    feat = torch.randn(2, 256, 16, 32)
    pseudo_label = torch.randint(0, 13, (2, 64, 128), dtype=torch.long)
    valid_mask = torch.zeros(2, 64, 128, dtype=torch.bool)
    label_f, mask_f = downsample_label_and_mask_to_feature(pseudo_label, valid_mask, (16, 32), ignore_index=255)
    prototypes, proto_valid, proto_counts = compute_current_batch_prototypes(feat, label_f, mask_f, 13, 10)
    _assert(not proto_valid.any().item(), "all-false mask should produce no valid prototypes")
    _assert(int(proto_counts.sum().item()) == 0, "all-false mask should produce zero counts")
    sim_map, sim_valid = compute_proto_similarity_map(feat, label_f, mask_f, prototypes, proto_valid)
    _assert(not sim_valid.any().item(), "all-false mask should produce no valid similarities")
    _assert(not torch.isnan(sim_map).any().item(), "all-false sim_map must not contain NaN")
    summary = summarize_proto_similarity(sim_map, sim_valid, label_f, 13, proto_valid=proto_valid, proto_counts=proto_counts)
    _assert(summary["sim_valid_ratio"] == 0.0, "all-false sim_valid_ratio should be 0")


def main() -> None:
    tests = [
        ("shape_normal", test_shape_normal_and_outputs),
        ("b1hw_label", test_b1hw_label_supported),
        ("missing_class_and_ignore", test_missing_class_and_ignore),
        ("all_false_valid_mask", test_all_false_valid_mask),
    ]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print("ALL_CURRENT_BATCH_PROTO_GATE_UNIT_TESTS_PASSED")


if __name__ == "__main__":
    main()
