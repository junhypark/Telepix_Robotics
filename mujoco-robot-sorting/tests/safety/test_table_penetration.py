"""Table penetration safety tests for robot arm links."""

from __future__ import annotations

import pytest

from robot_sorting.robot.safety import (
    check_arm_path_table_clearance,
    check_link_table_clearance,
    sample_link_segment_points,
)

pytestmark = pytest.mark.safety


def test_link_point_below_table_safety_height_is_rejected() -> None:
    assert not check_link_table_clearance(
        link_points=[(0.2, 0.1, -0.01)],
        table_top_z=0.0,
        min_clearance_meters=0.03,
    )


@pytest.mark.parametrize("joint_name", ["shoulder", "elbow", "wrist", "end_effector"])
def test_major_joint_below_table_safety_height_is_rejected(joint_name: str) -> None:
    positions = {
        "shoulder": (0.0, 0.0, 0.06),
        "elbow": (0.2, 0.0, 0.06),
        "wrist": (0.3, 0.0, 0.06),
        "end_effector": (0.35, 0.0, 0.06),
    }
    positions[joint_name] = (positions[joint_name][0], positions[joint_name][1], 0.01)

    assert not check_arm_path_table_clearance([positions], 0.0, 0.03)


def test_safe_link_points_above_table_are_accepted() -> None:
    assert check_link_table_clearance(
        link_points=[(0.2, 0.1, 0.05), (0.3, 0.2, 0.07)],
        table_top_z=0.0,
        min_clearance_meters=0.03,
    )


def test_path_with_intermediate_link_sample_below_table_is_rejected() -> None:
    link_points = sample_link_segment_points((0.2, 0.1, 0.05), (0.3, 0.1, 0.01), sample_count=5)

    assert not check_link_table_clearance(link_points, 0.0, 0.03)
