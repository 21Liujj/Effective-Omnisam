from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _clean_stats(stats: Any) -> Dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
            out[key] = value
        else:
            try:
                out[key] = float(value)
            except Exception:
                out[key] = str(value)
    return out


def _score_stats(values: List[Any]) -> Dict[str, Any]:
    vals: List[float] = []
    nan_count = 0
    for value in values:
        try:
            fval = float(value)
        except Exception:
            nan_count += 1
            continue
        if math.isnan(fval):
            nan_count += 1
        else:
            vals.append(fval)
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0, "nan_count": int(nan_count)}
    mean = sum(vals) / float(len(vals))
    var = sum((v - mean) * (v - mean) for v in vals) / float(len(vals))
    return {
        "mean": float(mean),
        "std": float(var ** 0.5),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "count": int(len(vals)),
        "nan_count": int(nan_count),
    }


def _token_region(context: Dict[str, Any], token_id: int, mem_h: int, mem_w: int, geo_cfg: Dict[str, Any]) -> Tuple[int, int, Any, float, str, str, str, float, float]:
    token_id = int(token_id)
    mem_h = int(mem_h)
    mem_w = int(mem_w)
    token_row = token_id // max(1, mem_w)
    token_col = token_id % max(1, mem_w)
    if mem_h <= 0 or mem_w <= 0 or token_id < 0 or token_id >= mem_h * mem_w:
        return token_row, token_col, None, 0.0, "unknown", "unknown", "unknown", None, None
    polar_band_ratio = float(geo_cfg.get("polar_band_ratio", 0.15))
    polar_band = max(0, min(mem_h // 2, int(round(mem_h * polar_band_ratio))))
    if token_row < polar_band:
        latitude_band = "top"
    elif token_row >= mem_h - polar_band:
        latitude_band = "bottom"
    else:
        latitude_band = "middle"
    latitude_rad = math.pi * (0.5 - (float(token_row) + 0.5) / float(mem_h))
    latitude_prior = abs(math.sin(latitude_rad))
    token_x = None
    seam_region = "unknown"
    true_range = context.get("true_erp_x_range")
    if bool(context.get("true_erp_x_range_reliable", False)) and isinstance(true_range, (list, tuple)) and len(true_range) >= 2:
        erp_w = int(geo_cfg.get("erp_width", 4096))
        seam_margin = int(geo_cfg.get("seam_margin", geo_cfg.get("stride", 384)))
        x0, x1 = float(true_range[0]), float(true_range[1])
        token_x = x0 + (float(token_col) + 0.5) * max(0.0, x1 - x0) / float(mem_w)
        if token_x <= float(seam_margin):
            seam_region = "left_seam"
        elif token_x >= float(erp_w - seam_margin):
            seam_region = "right_seam"
        else:
            seam_region = "non_seam"
    erp_h = int(geo_cfg.get("erp_height", 1024))
    token_y = (float(token_row) + 0.5) * float(erp_h) / float(mem_h)
    is_polar = latitude_band in {"top", "bottom"}
    is_seam = seam_region in {"left_seam", "right_seam"}
    if is_seam and is_polar:
        assigned_region = "seam_polar"
    elif is_seam:
        assigned_region = "seam"
    elif is_polar:
        assigned_region = "polar"
    else:
        assigned_region = "equator"
    return token_row, token_col, token_x, token_y, latitude_band, seam_region, assigned_region, latitude_rad, latitude_prior


def build_geo_token_records(args: Any, global_step: int, context: Dict[str, Any], memory_diag: Dict[str, Any], geo_cfg: Dict[str, Any]):
    writer_cfg = geo_cfg.get("token_writer", {}) if isinstance(geo_cfg, dict) else {}
    if not bool(writer_cfg.get("enabled", False)):
        return None, []
    mem_hw = memory_diag.get("mem_hw") if isinstance(memory_diag, dict) else None
    if isinstance(mem_hw, tuple):
        mem_hw = list(mem_hw)
    mem_h = int(mem_hw[0]) if isinstance(mem_hw, (list, tuple)) and len(mem_hw) >= 2 else 0
    mem_w = int(mem_hw[1]) if isinstance(mem_hw, (list, tuple)) and len(mem_hw) >= 2 else 0
    memory_num_tokens = int(memory_diag.get("num_tokens", 0)) if isinstance(memory_diag, dict) else 0
    expected_tokens = int(mem_h * mem_w) if mem_h > 0 and mem_w > 0 else 0
    token_grid_valid = bool(expected_tokens > 0 and memory_num_tokens == expected_tokens)
    token_total = expected_tokens if token_grid_valid else max(0, memory_num_tokens)
    max_records = int(max(0, writer_cfg.get("max_records", 5)))
    sample_rate = max(0.0, min(1.0, float(writer_cfg.get("sample_rate", 1.0))))
    region_names = ("seam", "polar", "seam_polar", "equator", "unknown")
    band_names = ("top", "middle", "bottom", "unknown")
    region_counts = {name: 0 for name in region_names}
    latitude_band_counts = {name: 0 for name in band_names}
    geo_scores = {name: [] for name in region_names}
    token_records: List[Dict[str, Any]] = []
    compression_stats = _clean_stats(memory_diag.get("compression_stats", {}) if isinstance(memory_diag, dict) else {})
    sparse_stats = _clean_stats(memory_diag.get("sparse_stats", {}) if isinstance(memory_diag, dict) else {})
    for token_id in range(token_total):
        token_row, token_col, token_x, token_y, latitude_band, seam_region, assigned_region, latitude_rad, latitude_prior = _token_region(context, token_id, mem_h, mem_w, geo_cfg)
        if assigned_region not in region_counts:
            assigned_region = "unknown"
        if latitude_band not in latitude_band_counts:
            latitude_band = "unknown"
        region_counts[assigned_region] += 1
        latitude_band_counts[latitude_band] += 1
        geo_scores[assigned_region].append(latitude_prior)
        take_sample = len(token_records) < max_records
        if sample_rate < 1.0 and token_total > 0:
            take_sample = take_sample and ((float(token_id + 1) / float(token_total)) <= sample_rate or token_id == 0)
        if not take_sample:
            continue
        reason_unavailable = {
            "image_id": "dataset metadata not provided to strategy yet",
            "patch_id": "dataset patch metadata not provided to strategy yet",
            "view_id": "dataset view metadata not provided to strategy yet",
            "source_box": "token footprint writer not implemented in Task 2-2",
            "base_score": "per-token selection score is not exposed by current memory path",
            "final_score": "diagnosis writer does not change selection scores",
            "kept_or_dropped": "write_new_memory stage does not expose per-token keep/drop state",
            "selection_rank": "per-token top-k rank is not exposed by current memory path",
            "keep_ratio": "per-token keep ratio is not applicable at write_new_memory stage",
        }
        if not bool(context.get("true_erp_x_range_reliable", False)):
            reason_unavailable["true_erp_x_range"] = "tensor-order horizontal inference is disabled or unsafe"
            reason_unavailable["token_x"] = "true ERP x range is unavailable or unreliable"
        token_records.append({
            "schema_version": "geo_memory_token_v1", "run_level": "L2_dry_run", "stage": str(context.get("stage", "train")),
            "global_step": int(global_step) if global_step is not None else None, "epoch": None, "memory_event": "write_new_memory",
            "mem_type": str(context.get("mem_type", "target")), "batch_id": 0, "image_id": None, "patch_id": None, "view_id": None,
            "tensor_order": int(context.get("tensor_order", -1)), "frame_idx": int(context.get("frame_idx", -1)), "frame_cnt": int(context.get("frame_cnt", 0)),
            "random_reverse_possible": bool(context.get("random_reverse_possible", False)), "tensor_order_inference_allowed": bool(context.get("tensor_order_inference_allowed", False)),
            "true_erp_x_range": context.get("true_erp_x_range"), "true_erp_y_range": context.get("true_erp_y_range"),
            "true_erp_x_range_reliable": bool(context.get("true_erp_x_range_reliable", False)),
            "inferred_erp_x_range_from_tensor_order": context.get("inferred_erp_x_range_from_tensor_order"),
            "erp_width": int(geo_cfg.get("erp_width", 4096)), "erp_height": int(geo_cfg.get("erp_height", 1024)),
            "window_size": int(geo_cfg.get("window_size", 1024)), "stride": int(geo_cfg.get("stride", 384)), "mem_hw": [int(mem_h), int(mem_w)],
            "memory_num_tokens": int(memory_num_tokens), "token_id": int(token_id), "token_row": int(token_row), "token_col": int(token_col),
            "token_x": token_x, "token_y": token_y, "source_box": None, "latitude_rad": latitude_rad, "latitude_prior": latitude_prior,
            "latitude_band": str(latitude_band), "seam_region": str(seam_region), "assigned_region": str(assigned_region),
            "region_priority": "seam_polar > seam > polar > equator", "base_score": None, "base_score_source": "none",
            "geo_score": latitude_prior, "geo_score_source": "latitude_prior", "final_score": None, "kept_or_dropped": "not_applicable",
            "drop_reason": None, "selection_rank": None, "keep_ratio": None,
            "compression_stage": str(compression_stats.get("compression_location", "none") or "none"),
            "compression_mode": str(compression_stats.get("compression_mode", memory_diag.get("memory_compress_mode", "none") if isinstance(memory_diag, dict) else "none") or "none"),
            "original_hw": compression_stats.get("original_hw"), "compressed_hw": compression_stats.get("compressed_hw"),
            "original_num_tokens": compression_stats.get("original_num_tokens"), "compressed_num_tokens": compression_stats.get("compressed_num_tokens"),
            "compression_ratio": compression_stats.get("compression_ratio"), "sparse_tokens_before": memory_diag.get("last_sparse_tokens_before") if isinstance(memory_diag, dict) else None,
            "sparse_tokens_after": memory_diag.get("last_sparse_tokens_after") if isinstance(memory_diag, dict) else None,
            "sparse_keep_ratio": memory_diag.get("last_sparse_keep_ratio") if isinstance(memory_diag, dict) else None,
            "reason_unavailable": reason_unavailable, "selection_effect": "diagnosis_only",
        })
    denom = float(max(1, token_total))
    summary = {
        "schema_version": "geo_memory_summary_v1", "stage": str(context.get("stage", "train")),
        "global_step": int(global_step) if global_step is not None else None, "mem_type": str(context.get("mem_type", "target")),
        "batch_id": 0, "image_id": None, "tensor_order": int(context.get("tensor_order", -1)), "frame_idx": int(context.get("frame_idx", -1)),
        "frame_cnt": int(context.get("frame_cnt", 0)), "true_erp_x_range_reliable": bool(context.get("true_erp_x_range_reliable", False)),
        "random_reverse_possible": bool(context.get("random_reverse_possible", False)), "mem_hw": [int(mem_h), int(mem_w)],
        "memory_num_tokens": int(memory_num_tokens), "token_total_checked": int(token_total), "token_grid_valid": bool(token_grid_valid),
        "region_counts": region_counts, "region_ratios": {name: float(region_counts[name] / denom) for name in region_names},
        "latitude_band_counts": latitude_band_counts, "kept_counts_by_region": None, "dropped_counts_by_region": None,
        "base_score_stats_by_region": None, "geo_score_stats_by_region": {name: _score_stats(geo_scores[name]) for name in region_names},
        "compression_stats": compression_stats, "sparse_stats": sparse_stats, "selection_effect": "diagnosis_only",
    }
    return summary, token_records
