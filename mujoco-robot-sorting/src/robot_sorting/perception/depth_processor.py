"""Depth image filtering and object depth extraction."""

from __future__ import annotations

import numpy as np


def get_object_depth_from_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    min_valid_depth: float,
    max_valid_depth: float,
) -> float | None:
    """Return robust median depth for a segmented object mask."""

    if depth.shape[:2] != mask.shape[:2]:
        raise ValueError("Depth and mask dimensions must match")
    masked = depth[mask > 0].astype(float)
    valid = masked[
        np.isfinite(masked)
        & (masked > 0.0)
        & (masked >= min_valid_depth)
        & (masked <= max_valid_depth)
    ]
    if valid.size < 5:
        return None
    return float(np.median(valid))


def valid_depth_ratio(
    depth: np.ndarray,
    mask: np.ndarray,
    min_valid_depth: float,
    max_valid_depth: float,
) -> float:
    """Return valid-depth pixel ratio inside a segmentation mask."""

    if depth.shape[:2] != mask.shape[:2]:
        raise ValueError("Depth and mask dimensions must match")
    mask_pixels = int(np.count_nonzero(mask > 0))
    if mask_pixels == 0:
        return 0.0
    masked = depth[mask > 0].astype(float)
    valid_count = int(
        np.count_nonzero(
            np.isfinite(masked)
            & (masked > 0.0)
            & (masked >= min_valid_depth)
            & (masked <= max_valid_depth)
        )
    )
    return valid_count / mask_pixels

