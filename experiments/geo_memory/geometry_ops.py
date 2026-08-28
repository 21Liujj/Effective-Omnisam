"""Geo-Memory geometry utilities for ERP region diagnostics.

These ops are standalone and diagnosis/unit-test only. They are not connected to
training, memory selection, sparse retrieval, attention, or loss.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

REGION_UNKNOWN = 0
REGION_SEAM = 1
REGION_POLAR = 2
REGION_SEAM_POLAR = 3
REGION_EQUATOR = 4

REGION_NAME_TO_ID = {
    "unknown": REGION_UNKNOWN,
    "seam": REGION_SEAM,
    "polar": REGION_POLAR,
    "seam_polar": REGION_SEAM_POLAR,
    "equator": REGION_EQUATOR,
}

REGION_ID_TO_NAME = {v: k for k, v in REGION_NAME_TO_ID.items()}


def _validate_hw(H: int, W: int) -> Tuple[int, int]:
    H, W = int(H), int(W)
    if H <= 0 or W <= 0:
        raise ValueError(f"H and W must be positive, got H={H}, W={W}")
    return H, W


def compute_latitude_map(H: int, W: int) -> np.ndarray:
    """Return ERP latitude in radians for each grid cell center.

    Works for both image grids and token grids. For a memory token grid, pass
    ``H=token_H`` and ``W=token_W`` so latitude depends only on token row.
    """
    H, W = _validate_hw(H, W)
    y = np.arange(H, dtype=np.float32)[:, None]
    lat = np.pi * (0.5 - (y + 0.5) / float(H))
    return np.repeat(lat, W, axis=1).astype(np.float32)


def compute_latitude_prior(H: int, W: int) -> np.ndarray:
    """Return high-latitude prior ``abs(sin(latitude))``.

    Top and bottom rows have larger values than equator rows. This function is
    row-only and does not depend on tensor order or horizontal ERP position.
    """
    return np.abs(np.sin(compute_latitude_map(H, W))).astype(np.float32)


def compute_seam_distance_map(H: int, W: int) -> np.ndarray:
    """Return horizontal distance to the left/right ERP seam for each cell."""
    H, W = _validate_hw(H, W)
    x = np.arange(W, dtype=np.float32)[None, :]
    dist = np.minimum(x, float(W - 1) - x)
    return np.repeat(dist, H, axis=0).astype(np.float32)


def compute_seam_prior(H: int, W: int, seam_width: int = 32, tau: float | None = None) -> np.ndarray:
    """Return seam prior ``exp(-d_seam / tau)``.

    ``seam_width`` is kept as an explicit geometry parameter; if ``tau`` is not
    provided, ``tau = max(1, seam_width / 2)``.
    """
    H, W = _validate_hw(H, W)
    seam_width = int(seam_width)
    if seam_width <= 0:
        raise ValueError(f"seam_width must be positive, got {seam_width}")
    tau = float(tau if tau is not None else max(1.0, seam_width / 2.0))
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return np.exp(-compute_seam_distance_map(H, W) / tau).astype(np.float32)


def build_region_mask(H: int, W: int, seam_width: int = 32, polar_band_ratio: float = 0.15) -> np.ndarray:
    """Build mutually exclusive ERP region ids with explicit priority.

    Priority is:
    1. ``seam_polar`` for cells that are both seam and polar.
    2. ``seam`` for seam-only cells.
    3. ``polar`` for polar-only cells.
    4. ``equator`` for all remaining valid cells.

    ``unknown`` is reserved for coordinates that cannot be mapped and is not
    assigned to valid grid cells by this function.
    """
    H, W = _validate_hw(H, W)
    seam_width = int(seam_width)
    if seam_width < 0:
        raise ValueError(f"seam_width must be non-negative, got {seam_width}")
    polar_band_ratio = float(polar_band_ratio)
    if not 0.0 <= polar_band_ratio < 0.5:
        raise ValueError(f"polar_band_ratio must be in [0, 0.5), got {polar_band_ratio}")

    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    seam = (x < seam_width) | (x >= max(0, W - seam_width))
    polar_band = int(round(H * polar_band_ratio))
    polar = (y < polar_band) | (y >= max(0, H - polar_band))

    mask = np.full((H, W), REGION_EQUATOR, dtype=np.uint8)
    mask[np.repeat(seam, H, axis=0)] = REGION_SEAM
    mask[np.repeat(polar, W, axis=1)] = REGION_POLAR
    mask[np.repeat(seam, H, axis=0) & np.repeat(polar, W, axis=1)] = REGION_SEAM_POLAR
    return mask


def assign_region_from_xy(
    x: int,
    y: int,
    H: int,
    W: int,
    seam_width: int = 32,
    polar_band_ratio: float = 0.15,
) -> str:
    """Assign a single coordinate to a region name.

    This is reliable for point coordinates. For full-height panoramic patches,
    point/box-level polar-equator decisions are insufficient for latitude prior;
    use token-row geometry for memory tokens.
    """
    H, W = _validate_hw(H, W)
    x, y = int(x), int(y)
    if x < 0 or x >= W or y < 0 or y >= H:
        return "unknown"
    mask = build_region_mask(H, W, seam_width=seam_width, polar_band_ratio=polar_band_ratio)
    return REGION_ID_TO_NAME[int(mask[y, x])]


def assign_region_from_box(
    box,
    H: int,
    W: int,
    seam_width: int = 32,
    polar_band_ratio: float = 0.15,
) -> str:
    """Assign a box to a coarse region by center point.

    ``box`` is ``(x0, y0, x1, y1)``. This center-point assignment is a coarse
    diagnostic helper. When a panoramic patch/box covers full height, box-level
    polar/equator judgment is unreliable because it mixes polar and equator rows;
    later latitude prior must use token-row labels instead.
    """
    H, W = _validate_hw(H, W)
    if box is None or len(box) != 4:
        return "unknown"
    x0, y0, x1, y1 = [float(v) for v in box]
    if x1 <= x0 or y1 <= y0:
        return "unknown"
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        return "unknown"
    # Full-height panoramic boxes/patches mix polar, equator, and bottom rows.
    # Do not collapse them to a single polar/equator label by center y. Horizontal
    # seam is still a coarse x-only diagnostic; otherwise return unknown and let
    # downstream latitude prior use token-row labels.
    covers_full_height = y0 <= 0 and y1 >= H
    x = int(np.clip((x0 + x1 - 1.0) / 2.0, 0, W - 1))
    y = int(np.clip((y0 + y1 - 1.0) / 2.0, 0, H - 1))
    if covers_full_height:
        seam_width_i = int(seam_width)
        if x < seam_width_i or x >= max(0, W - seam_width_i):
            return "seam"
        return "unknown"
    return assign_region_from_xy(x, y, H, W, seam_width=seam_width, polar_band_ratio=polar_band_ratio)


def resize_region_map_to_token_grid(region_map, token_H: int, token_W: int) -> np.ndarray:
    """Resize a region id map to token grid by nearest-neighbor cell centers.

    This preserves categorical ids and is intended for cases such as
    ``1024 x 1024 -> 64 x 64`` memory-token grids.
    """
    arr = np.asarray(region_map)
    if arr.ndim != 2:
        raise ValueError(f"region_map must be 2D, got shape={arr.shape}")
    src_H, src_W = _validate_hw(arr.shape[0], arr.shape[1])
    token_H, token_W = _validate_hw(token_H, token_W)
    row_idx = np.floor((np.arange(token_H, dtype=np.float64) + 0.5) * src_H / token_H).astype(np.int64)
    col_idx = np.floor((np.arange(token_W, dtype=np.float64) + 0.5) * src_W / token_W).astype(np.int64)
    row_idx = np.clip(row_idx, 0, src_H - 1)
    col_idx = np.clip(col_idx, 0, src_W - 1)
    return arr[row_idx[:, None], col_idx[None, :]].astype(arr.dtype, copy=False)


def compute_token_row_region_counts(
    token_H: int,
    token_W: int,
    polar_band_ratio: float = 0.15,
) -> Dict[str, int]:
    """Count top/middle/bottom token rows for latitude diagnostics only."""
    token_H, token_W = _validate_hw(token_H, token_W)
    polar_band = int(round(token_H * float(polar_band_ratio)))
    polar_band = max(0, min(token_H // 2, polar_band))
    top = polar_band * token_W
    bottom = polar_band * token_W
    middle = token_H * token_W - top - bottom
    return {"top": int(top), "middle": int(middle), "bottom": int(bottom)}
