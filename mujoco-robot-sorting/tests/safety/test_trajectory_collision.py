"""Safety tests for 3D trajectory collision checks."""

from __future__ import annotations

import pytest

from robot_sorting.planning.collision_checker import (
    segment_intersects_base_exclusion_zone,
    trajectory_has_collision_risk,
)
from robot_sorting.schemas import TrajectoryWaypoint

pytestmark = pytest.mark.safety


def test_path_segment_crossing_robot_base_is_rejected() -> None:
    assert segment_intersects_base_exclusion_zone((-0.3, 0.0, 0.08), (0.3, 0.0, 0.08), 0.11, 0.12, 0.025)


def test_path_segment_outside_robot_base_is_accepted() -> None:
    assert not segment_intersects_base_exclusion_zone((0.3, 0.25, 0.2), (0.45, 0.25, 0.2), 0.11, 0.12, 0.025)


def test_waypoint_inside_base_exclusion_zone_is_rejected() -> None:
    waypoints = [TrajectoryWaypoint(position=(0.01, 0.01, 0.08), gripper_state="open", duration_seconds=0.1)]

    assert trajectory_has_collision_risk(waypoints, 0.11, 0.12, 0.025)


def test_full_trajectory_around_base_is_accepted() -> None:
    waypoints = [
        TrajectoryWaypoint(position=(0.25, 0.25, 0.18), gripper_state="open", duration_seconds=0.1),
        TrajectoryWaypoint(position=(0.45, 0.25, 0.18), gripper_state="closed", duration_seconds=0.1),
    ]

    assert not trajectory_has_collision_risk(waypoints, 0.11, 0.12, 0.025)


def test_full_trajectory_through_base_is_rejected() -> None:
    waypoints = [
        TrajectoryWaypoint(position=(-0.25, 0.0, 0.08), gripper_state="open", duration_seconds=0.1),
        TrajectoryWaypoint(position=(0.25, 0.0, 0.08), gripper_state="closed", duration_seconds=0.1),
    ]

    assert trajectory_has_collision_risk(waypoints, 0.11, 0.12, 0.025)

