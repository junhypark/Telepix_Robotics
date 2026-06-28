"""Integration tests for table-clear motion planning and execution."""

from __future__ import annotations

import pytest

from robot_sorting.perception.grasp_pose_generator import generate_top_down_grasp_pose
from robot_sorting.planning.trajectory_planner import plan_pick_place_trajectory
from robot_sorting.robot.kinematics import forward_link_positions, solve_ik
from robot_sorting.robot.safety import sample_link_segment_points
from robot_sorting.schemas import ObjectPose3D, SimulationConfig, TableSafetyConfig

pytestmark = pytest.mark.integration


def test_planned_motion_keeps_arm_links_above_table(two_object_config: SimulationConfig) -> None:
    object_pose = ObjectPose3D(
        object_id="object_0",
        label="normal",
        position=(0.24, 0.10, two_object_config.workspace.z_min),
        orientation_rpy=(0.0, 0.0, 0.0),
        size_xyz=(0.04, 0.04, 0.03),
        confidence=1.0,
    )
    grasp = generate_top_down_grasp_pose(
        object_pose,
        grasp_height_offset=0.005,
        pre_grasp_z_offset=two_object_config.approach_height,
        retreat_z_offset=two_object_config.approach_height + 0.02,
        workspace=two_object_config.workspace,
    )
    assert grasp is not None

    trajectory = plan_pick_place_trajectory(
        grasp,
        (0.42, 0.20, two_object_config.workspace.z_min),
        two_object_config.workspace,
        two_object_config.base_radius,
        two_object_config.base_height,
        two_object_config.safety_margin,
        two_object_config.table_safety,
        two_object_config.link_1,
        two_object_config.link_2,
    )

    min_required_link_z = (
        two_object_config.table_safety.table_top_z + two_object_config.table_safety.min_link_clearance_meters
    )
    min_observed_link_z = float("inf")
    for waypoint in trajectory.waypoints:
        ik_result = solve_ik(
            waypoint.position,
            two_object_config.link_1,
            two_object_config.link_2,
            two_object_config.base_height,
        )
        positions = forward_link_positions(
            ik_result.angles,
            two_object_config.link_1,
            two_object_config.link_2,
            two_object_config.base_height,
        )
        link_points = [
            *sample_link_segment_points(positions["shoulder"], positions["elbow"]),
            *sample_link_segment_points(positions["elbow"], positions["wrist"]),
            *sample_link_segment_points(positions["wrist"], positions["end_effector"]),
        ]
        min_observed_link_z = min(min_observed_link_z, *(point[2] for point in link_points))

    assert trajectory.is_safe is True
    assert min_observed_link_z >= min_required_link_z
    assert trajectory.waypoints[3].position[2] > trajectory.waypoints[2].position[2]
    assert trajectory.waypoints[4].position[2] == pytest.approx(trajectory.waypoints[3].position[2])


def test_unsafe_table_clearance_requirement_rejects_trajectory(two_object_config: SimulationConfig) -> None:
    object_pose = ObjectPose3D(
        object_id="object_0",
        label="normal",
        position=(0.24, 0.10, two_object_config.workspace.z_min),
        orientation_rpy=(0.0, 0.0, 0.0),
        size_xyz=(0.04, 0.04, 0.03),
        confidence=1.0,
    )
    grasp = generate_top_down_grasp_pose(object_pose, 0.005, 0.11, 0.13, two_object_config.workspace)
    assert grasp is not None
    strict_table_safety = TableSafetyConfig(min_link_clearance_meters=0.50)

    unsafe_trajectory = plan_pick_place_trajectory(
        grasp,
        (0.42, 0.20, two_object_config.workspace.z_min),
        two_object_config.workspace,
        two_object_config.base_radius,
        two_object_config.base_height,
        two_object_config.safety_margin,
        strict_table_safety,
        two_object_config.link_1,
        two_object_config.link_2,
    )

    assert unsafe_trajectory.is_safe is False
    assert unsafe_trajectory.failure_reason == "link_table_penetration_risk"
