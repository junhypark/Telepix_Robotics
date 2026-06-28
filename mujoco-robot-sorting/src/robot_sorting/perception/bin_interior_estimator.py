"""Estimate free placement space inside detected bins."""

from __future__ import annotations

import cv2
import numpy as np

from robot_sorting.perception.camera_calibration import pixel_depth_to_world_point
from robot_sorting.schemas import BinInteriorEstimate, CameraCalibration, DetectedBin


def estimate_bin_interior_free_space(
    rgb: np.ndarray,
    depth: np.ndarray,
    detected_bin: DetectedBin,
    calibration: CameraCalibration,
    object_size_xyz: tuple[float, float, float],
) -> BinInteriorEstimate:
    """Estimate candidate placement pixels and world positions inside a detected bin."""

    del rgb
    if depth.ndim != 2:
        raise ValueError("Depth must be a two-dimensional array")

    inner_bounds = _inner_pixel_bounds(detected_bin, calibration, object_size_xyz, depth.shape)
    left, top, right, bottom = inner_bounds
    inner_polygon = [(left, top), (right, top), (right, bottom), (left, bottom)]
    occupied_regions = _occupied_regions(depth, inner_bounds)
    candidates = _candidate_pixels(inner_bounds, object_size_xyz, calibration)
    candidate_positions = [
        _pixel_to_world(pixel, depth, detected_bin, calibration)
        for pixel in candidates
        if not _pixel_inside_regions(pixel, occupied_regions)
    ]
    candidate_pixels = [
        pixel
        for pixel in candidates
        if not _pixel_inside_regions(pixel, occupied_regions)
    ]
    return BinInteriorEstimate(
        bin_id=detected_bin.bin_id,
        bin_label=detected_bin.label,
        inner_polygon_pixels=inner_polygon,
        candidate_place_pixels=candidate_pixels,
        candidate_place_positions=candidate_positions,
        occupied_regions=occupied_regions,
        confidence=detected_bin.confidence,
    )


def _inner_pixel_bounds(
    detected_bin: DetectedBin,
    calibration: CameraCalibration,
    object_size_xyz: tuple[float, float, float],
    depth_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    center_u, center_v = detected_bin.pixel_center
    half_width_px = max(1, int((detected_bin.size_xyz[0] / 2.0) * calibration.intrinsics.fx))
    half_height_px = max(1, int((detected_bin.size_xyz[1] / 2.0) * calibration.intrinsics.fy))
    object_margin_x = max(1, int((object_size_xyz[0] / 2.0) * calibration.intrinsics.fx))
    object_margin_y = max(1, int((object_size_xyz[1] / 2.0) * calibration.intrinsics.fy))
    height, width = depth_shape
    left = max(0, center_u - half_width_px + object_margin_x)
    right = min(width - 1, center_u + half_width_px - object_margin_x)
    top = max(0, center_v - half_height_px + object_margin_y)
    bottom = min(height - 1, center_v + half_height_px - object_margin_y)
    if left > right or top > bottom:
        return (center_u, center_v, center_u, center_v)
    return (left, top, right, bottom)


def _occupied_regions(depth: np.ndarray, bounds: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    left, top, right, bottom = bounds
    region = depth[top : bottom + 1, left : right + 1]
    if region.size == 0:
        return []
    finite = region[np.isfinite(region)]
    if finite.size == 0:
        return []
    baseline = float(np.median(finite))
    occupied_mask = np.zeros(region.shape, dtype=np.uint8)
    occupied_mask[np.isfinite(region) & (region < baseline - 0.015)] = 255
    contours, _ = cv2.findContours(occupied_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    occupied: list[list[tuple[int, int]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 4.0:
            continue
        points = [(int(point[0][0] + left), int(point[0][1] + top)) for point in contour]
        occupied.append(points)
    return occupied


def _candidate_pixels(
    bounds: tuple[int, int, int, int],
    object_size_xyz: tuple[float, float, float],
    calibration: CameraCalibration,
) -> list[tuple[int, int]]:
    left, top, right, bottom = bounds
    step_x = max(1, int(object_size_xyz[0] * calibration.intrinsics.fx))
    step_y = max(1, int(object_size_xyz[1] * calibration.intrinsics.fy))
    center_u = (left + right) // 2
    center_v = (top + bottom) // 2
    candidates = [
        (u, v)
        for u in range(left, right + 1, step_x)
        for v in range(top, bottom + 1, step_y)
    ]
    candidates.append((center_u, center_v))
    return sorted(
        set(candidates),
        key=lambda item: (abs(item[0] - center_u) + abs(item[1] - center_v), item[0], item[1]),
    )


def _pixel_to_world(
    pixel: tuple[int, int],
    depth: np.ndarray,
    detected_bin: DetectedBin,
    calibration: CameraCalibration,
) -> tuple[float, float, float]:
    value = float(depth[pixel[1], pixel[0]])
    if not np.isfinite(value) or value <= 0.0:
        return detected_bin.world_position
    try:
        world = pixel_depth_to_world_point(pixel, value, calibration)
    except ValueError:
        return detected_bin.world_position
    return (float(world[0]), float(world[1]), float(detected_bin.world_position[2]))


def _pixel_inside_regions(pixel: tuple[int, int], regions: list[list[tuple[int, int]]]) -> bool:
    for region in regions:
        if len(region) < 3:
            continue
        contour = np.array(region, dtype=np.int32).reshape((-1, 1, 2))
        if cv2.pointPolygonTest(contour, pixel, measureDist=False) >= 0:
            return True
    return False
