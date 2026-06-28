"""3D object pose estimation from RGB-D detections."""

from __future__ import annotations

import math

import numpy as np

from robot_sorting.modules.vision_module import MaskedDetection
from robot_sorting.perception.camera_calibration import pixel_depth_to_world_point
from robot_sorting.perception.depth_processor import get_object_depth_from_mask, valid_depth_ratio
from robot_sorting.perception.point_cloud import mask_to_point_cloud
from robot_sorting.schemas import CameraCalibration, DetectedObject, ObjectPose3D


def estimate_object_pose_3d(
    detected_object: DetectedObject,
    mask: np.ndarray,
    depth: np.ndarray,
    calibration: CameraCalibration,
) -> ObjectPose3D | None:
    """Estimate table-top object 3D pose from a color mask and depth image."""

    depth_value = get_object_depth_from_mask(depth, mask, min_valid_depth=0.01, max_valid_depth=10.0)
    if depth_value is None:
        return None
    depth_ratio = valid_depth_ratio(depth, mask, min_valid_depth=0.01, max_valid_depth=10.0)
    if depth_ratio < 0.20:
        return None
    cloud = mask_to_point_cloud(depth, mask, calibration, stride=2)
    if cloud.shape[0] < 6:
        return None

    center = pixel_depth_to_world_point(detected_object.pixel_center, depth_value, calibration)
    bounds_min = np.min(cloud, axis=0)
    bounds_max = np.max(cloud, axis=0)
    size = np.maximum(bounds_max - bounds_min, np.array([0.005, 0.005, 0.005]))
    yaw = _estimate_yaw_from_mask(mask)
    cloud_score = min(1.0, cloud.shape[0] / 120.0)
    confidence = float(np.clip(detected_object.confidence * (0.45 + 0.35 * depth_ratio + 0.20 * cloud_score), 0.0, 1.0))
    return ObjectPose3D(
        object_id=detected_object.object_id,
        label=detected_object.label,
        position=(float(center[0]), float(center[1]), float(center[2])),
        orientation_rpy=(0.0, 0.0, yaw),
        size_xyz=(float(size[0]), float(size[1]), float(size[2])),
        confidence=confidence,
    )


def _estimate_yaw_from_mask(mask: np.ndarray) -> float:
    rows, cols = np.nonzero(mask > 0)
    if rows.size < 2:
        return 0.0
    points = np.column_stack((cols.astype(float), rows.astype(float)))
    points -= np.mean(points, axis=0)
    covariance = np.cov(points, rowvar=False)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    yaw = math.atan2(float(principal[1]), float(principal[0]))
    if not math.isfinite(yaw):
        return 0.0
    return float(yaw)


def masks_from_rgb_detections(rgb: np.ndarray) -> list[MaskedDetection]:
    """Detect colored objects and return one binary mask per contour."""

    from robot_sorting.config import create_simulation_config
    from robot_sorting.modules.vision_module import VisionModule

    config = create_simulation_config(width=rgb.shape[1], height=rgb.shape[0])
    return VisionModule(config).detect_with_masks(rgb)
