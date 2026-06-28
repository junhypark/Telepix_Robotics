"""Unit tests for collision-aware trajectory planning."""

from __future__ import annotations

import math

import pytest

from robot_sorting.planning.trajectory_planner import plan_pick_place_trajectory
from robot_sorting.schemas import GraspPose, WorkspaceBounds

pytestmark = pytest.mark.unit


def _grasp(position: tuple[float, float, float] = (0.25, 0.12, 0.06)) -> GraspPose:
    x, y, z = position
    return GraspPose(
        object_id="object_0",
        pre_grasp_position=(x, y, z + 0.10),
        grasp_position=position,
        retreat_position=(x, y, z + 0.12),
        approach_vector=(0.0, 0.0, -1.0),
        gripper_yaw=0.0,
        confidence=0.9,
    )


def test_valid_grasp_and_place_position_produces_safe_trajectory() -> None:
    trajectory = plan_pick_place_trajectory(_grasp(), (0.45, 0.28, 0.055), WorkspaceBounds(), 0.11, 0.12, 0.025)

    assert trajectory.is_safe
    assert len(trajectory.waypoints) == 8
    assert all(all(math.isfinite(value) for value in waypoint.position) for waypoint in trajectory.waypoints)


def test_unsafe_waypoint_inside_base_zone_fails() -> None:
    trajectory = plan_pick_place_trajectory(
        _grasp((0.01, 0.01, 0.06)),
        (0.45, 0.28, 0.055),
        WorkspaceBounds(),
        0.11,
        0.12,
        0.025,
    )

    assert not trajectory.is_safe
    assert trajectory.failure_reason in {"self_collision_risk", "trajectory_collision_risk"}


def test_segment_crossing_base_zone_fails_with_explicit_reason() -> None:
    trajectory = plan_pick_place_trajectory(
        _grasp((-0.25, 0.0, 0.06)),
        (0.25, 0.0, 0.06),
        WorkspaceBounds(),
        0.11,
        0.20,
        0.025,
    )

    assert not trajectory.is_safe
    assert trajectory.failure_reason == "trajectory_collision_risk"
