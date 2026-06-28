"""Unit tests for point cloud generation."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sorting.perception.camera_calibration import build_top_down_workspace_calibration
from robot_sorting.perception.point_cloud import mask_to_point_cloud
from robot_sorting.schemas import WorkspaceBounds

pytestmark = pytest.mark.unit


def test_masked_depth_converts_to_point_cloud() -> None:
    workspace = WorkspaceBounds()
    calibration = build_top_down_workspace_calibration(20, 20, workspace)
    depth = np.ones((20, 20), dtype=np.float32) * 0.9
    mask = np.ones((20, 20), dtype=np.uint8) * 255

    cloud = mask_to_point_cloud(depth, mask, calibration, stride=2)

    assert cloud.shape[1] == 3
    assert cloud.shape[0] > 0
    assert np.all(np.isfinite(cloud))


def test_empty_mask_returns_empty_cloud() -> None:
    calibration = build_top_down_workspace_calibration(20, 20, WorkspaceBounds())
    cloud = mask_to_point_cloud(
        np.ones((20, 20), dtype=np.float32),
        np.zeros((20, 20), dtype=np.uint8),
        calibration,
    )

    assert cloud.shape == (0, 3)


def test_invalid_depth_values_are_ignored_and_stride_reduces_count() -> None:
    calibration = build_top_down_workspace_calibration(20, 20, WorkspaceBounds())
    depth = np.ones((20, 20), dtype=np.float32)
    depth[::2, ::2] = np.nan
    mask = np.ones((20, 20), dtype=np.uint8) * 255

    dense = mask_to_point_cloud(depth, mask, calibration, stride=1)
    sparse = mask_to_point_cloud(depth, mask, calibration, stride=4)

    assert sparse.shape[0] < dense.shape[0]


def test_points_are_inside_expected_workspace_bounds() -> None:
    workspace = WorkspaceBounds()
    calibration = build_top_down_workspace_calibration(20, 20, workspace)
    depth = np.ones((20, 20), dtype=np.float32) * 0.95
    mask = np.ones((20, 20), dtype=np.uint8) * 255

    cloud = mask_to_point_cloud(depth, mask, calibration, stride=3)

    assert np.all(cloud[:, 0] >= workspace.x_min - 0.01)
    assert np.all(cloud[:, 0] <= workspace.x_max + 0.01)
    assert np.all(cloud[:, 1] >= workspace.y_min - 0.01)
    assert np.all(cloud[:, 1] <= workspace.y_max + 0.01)

