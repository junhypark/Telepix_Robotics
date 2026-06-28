"""Unit tests for top-down grasp pose generation."""

from __future__ import annotations

import pytest

from robot_sorting.perception.grasp_pose_generator import generate_top_down_grasp_pose
from robot_sorting.schemas import ObjectPose3D, WorkspaceBounds

pytestmark = pytest.mark.unit


def _pose(position: tuple[float, float, float] = (0.25, 0.1, 0.05)) -> ObjectPose3D:
    return ObjectPose3D(
        object_id="object_0",
        label="normal",
        position=position,
        orientation_rpy=(0.0, 0.0, 1.2),
        size_xyz=(0.04, 0.04, 0.03),
        confidence=0.9,
    )


def test_top_down_grasp_pose_is_generated() -> None:
    grasp = generate_top_down_grasp_pose(_pose(), 0.005, 0.10, 0.12, WorkspaceBounds())

    assert grasp is not None
    assert grasp.pre_grasp_position[2] > grasp.grasp_position[2]
    assert grasp.retreat_position[2] > grasp.grasp_position[2]
    assert grasp.approach_vector == (0.0, 0.0, -1.0)
    assert grasp.gripper_yaw == pytest.approx(1.2)


def test_pose_outside_workspace_returns_none() -> None:
    assert generate_top_down_grasp_pose(_pose((2.0, 0.0, 0.05)), 0.005, 0.10, 0.12, WorkspaceBounds()) is None


def test_pose_inside_base_exclusion_zone_returns_none() -> None:
    assert generate_top_down_grasp_pose(_pose((0.01, 0.01, 0.05)), 0.005, 0.10, 0.12, WorkspaceBounds()) is None

