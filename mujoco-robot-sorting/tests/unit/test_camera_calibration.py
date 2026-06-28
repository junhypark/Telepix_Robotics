"""Unit tests for RGB-D camera calibration helpers."""

from __future__ import annotations

import math

import pytest

from robot_sorting.perception.camera_calibration import (
    camera_to_world_point,
    pixel_depth_to_world_point,
    pixel_to_camera_point,
)
from robot_sorting.schemas import CameraCalibration, CameraExtrinsics, CameraIntrinsics

pytestmark = pytest.mark.unit


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, width=100, height=80),
        extrinsics=CameraExtrinsics(
            rotation_world_from_camera=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation_world_from_camera=(0.1, 0.2, 0.3),
        ),
    )


def test_center_pixel_maps_to_camera_z_axis() -> None:
    point = pixel_to_camera_point((50, 40), 2.0, _calibration().intrinsics)

    assert point == (0.0, 0.0, 2.0)


def test_greater_u_maps_to_positive_camera_x() -> None:
    point = pixel_to_camera_point((60, 40), 2.0, _calibration().intrinsics)

    assert point[0] > 0


def test_greater_v_maps_to_positive_camera_y_convention() -> None:
    point = pixel_to_camera_point((50, 50), 2.0, _calibration().intrinsics)

    assert point[1] > 0


def test_camera_to_world_transform_returns_finite_values() -> None:
    world = camera_to_world_point((1.0, 2.0, 3.0), _calibration().extrinsics)

    assert all(math.isfinite(value) for value in world)
    assert world == (1.1, 2.2, 3.3)


def test_invalid_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="Depth"):
        pixel_depth_to_world_point((50, 40), float("nan"), _calibration())

