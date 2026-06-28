"""Unit tests for 3D pose estimation."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from robot_sorting.perception.camera_calibration import build_top_down_workspace_calibration
from robot_sorting.perception.pose_estimator import estimate_object_pose_3d
from robot_sorting.schemas import DetectedObject, WorkspaceBounds

pytestmark = pytest.mark.unit


def _mask() -> np.ndarray:
    mask = np.zeros((80, 100), dtype=np.uint8)
    cv2.rectangle(mask, (40, 30), (60, 50), 255, thickness=-1)
    return mask


def _detected(confidence: float = 0.9) -> DetectedObject:
    return DetectedObject(
        object_id="object_0",
        label="normal",
        pixel_center=(50, 40),
        world_position=(0.0, 0.0, 0.05),
        confidence=confidence,
    )


def test_object_pose_is_estimated_from_depth_and_mask() -> None:
    calibration = build_top_down_workspace_calibration(100, 80, WorkspaceBounds())
    depth = np.ones((80, 100), dtype=np.float32) * 0.95

    pose = estimate_object_pose_3d(_detected(), _mask(), depth, calibration)

    assert pose is not None
    assert pose.position[2] == pytest.approx(0.05, abs=0.02)
    assert all(value > 0 for value in pose.size_xyz)
    assert math.isfinite(pose.orientation_rpy[2])


def test_low_valid_depth_ratio_returns_none() -> None:
    calibration = build_top_down_workspace_calibration(100, 80, WorkspaceBounds())
    depth = np.zeros((80, 100), dtype=np.float32)
    depth[40, 50] = 0.95

    assert estimate_object_pose_3d(_detected(), _mask(), depth, calibration) is None


def test_small_point_cloud_returns_none() -> None:
    calibration = build_top_down_workspace_calibration(100, 80, WorkspaceBounds())
    depth = np.ones((80, 100), dtype=np.float32) * 0.95
    tiny_mask = np.zeros((80, 100), dtype=np.uint8)
    tiny_mask[40, 50] = 255

    assert estimate_object_pose_3d(_detected(), tiny_mask, depth, calibration) is None


def test_confidence_decreases_when_color_confidence_is_lower() -> None:
    calibration = build_top_down_workspace_calibration(100, 80, WorkspaceBounds())
    depth = np.ones((80, 100), dtype=np.float32) * 0.95

    high = estimate_object_pose_3d(_detected(0.9), _mask(), depth, calibration)
    low = estimate_object_pose_3d(_detected(0.4), _mask(), depth, calibration)

    assert high is not None and low is not None
    assert low.confidence < high.confidence

