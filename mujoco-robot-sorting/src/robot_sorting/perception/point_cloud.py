"""Point cloud generation from masked depth images."""

from __future__ import annotations

import numpy as np

from robot_sorting.perception.camera_calibration import pixel_depth_to_world_point
from robot_sorting.schemas import CameraCalibration


def mask_to_point_cloud(
    depth: np.ndarray,
    mask: np.ndarray,
    calibration: CameraCalibration,
    stride: int = 2,
) -> np.ndarray:
    """Convert masked depth pixels into an N x 3 world-space point cloud."""

    if depth.shape[:2] != mask.shape[:2]:
        raise ValueError("Depth and mask dimensions must match")
    safe_stride = max(1, stride)
    points: list[tuple[float, float, float]] = []
    rows, cols = np.nonzero(mask > 0)
    for row, col in zip(rows[::safe_stride], cols[::safe_stride], strict=False):
        value = float(depth[row, col])
        if not np.isfinite(value) or value <= 0.0:
            continue
        try:
            points.append(pixel_depth_to_world_point((int(col), int(row)), value, calibration))
        except ValueError:
            continue
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float).reshape((-1, 3))

