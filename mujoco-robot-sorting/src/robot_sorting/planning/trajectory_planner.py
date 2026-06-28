"""Collision-aware pick-and-place trajectory planner."""

from __future__ import annotations

import math

from robot_sorting.planning.collision_checker import trajectory_has_collision_risk
from robot_sorting.robot.kinematics import forward_link_positions, solve_ik
from robot_sorting.robot.safety import is_inside_base_exclusion_zone, is_inside_workspace, sample_link_segment_points
from robot_sorting.schemas import GraspPose, PlannedTrajectory, TableSafetyConfig, TrajectoryWaypoint, WorkspaceBounds


def plan_pick_place_trajectory(
    grasp_pose: GraspPose,
    place_position: tuple[float, float, float],
    workspace: WorkspaceBounds,
    base_radius: float,
    base_height: float,
    safety_margin: float,
    table_safety: TableSafetyConfig | None = None,
    link_1: float | None = None,
    link_2: float | None = None,
) -> PlannedTrajectory:
    """Plan a safe top-down pick-and-place trajectory."""

    safe_lift_z = _safe_lift_z(grasp_pose, workspace, table_safety)
    pre_grasp = _with_z(grasp_pose.pre_grasp_position, safe_lift_z)
    retreat = _with_z(grasp_pose.retreat_position, safe_lift_z)
    above_place = (
        place_position[0],
        place_position[1],
        safe_lift_z,
    )
    waypoints = [
        TrajectoryWaypoint(position=pre_grasp, gripper_state="open", duration_seconds=0.10),
        TrajectoryWaypoint(position=grasp_pose.grasp_position, gripper_state="open", duration_seconds=0.08),
        TrajectoryWaypoint(position=grasp_pose.grasp_position, gripper_state="closed", duration_seconds=0.03),
        TrajectoryWaypoint(position=retreat, gripper_state="closed", duration_seconds=0.10),
        TrajectoryWaypoint(position=above_place, gripper_state="closed", duration_seconds=0.14),
        TrajectoryWaypoint(position=place_position, gripper_state="closed", duration_seconds=0.08),
        TrajectoryWaypoint(position=place_position, gripper_state="open", duration_seconds=0.03),
        TrajectoryWaypoint(position=above_place, gripper_state="open", duration_seconds=0.10),
    ]

    failure_reason = _validate_waypoints(waypoints, workspace, base_radius, base_height, safety_margin)
    if failure_reason is None and table_safety is not None and link_1 is not None and link_2 is not None:
        failure_reason = _validate_link_table_clearance(waypoints, table_safety, link_1, link_2, base_height)
    if failure_reason is not None:
        return PlannedTrajectory(
            object_id=grasp_pose.object_id,
            waypoints=waypoints,
            is_safe=False,
            failure_reason=failure_reason,
        )
    return PlannedTrajectory(
        object_id=grasp_pose.object_id,
        waypoints=waypoints,
        is_safe=True,
        failure_reason=None,
    )


def _validate_waypoints(
    waypoints: list[TrajectoryWaypoint],
    workspace: WorkspaceBounds,
    base_radius: float,
    base_height: float,
    safety_margin: float,
) -> str | None:
    for waypoint in waypoints:
        if not all(math.isfinite(value) for value in waypoint.position):
            return "non_finite_waypoint"
        if not is_inside_workspace(waypoint.position, workspace):
            return "workspace_limit"
        if is_inside_base_exclusion_zone(
            waypoint.position,
            base_radius + safety_margin,
            0.0,
            base_height + safety_margin,
        ):
            return "self_collision_risk"
    if trajectory_has_collision_risk(waypoints, base_radius, base_height, safety_margin):
        return "trajectory_collision_risk"
    return None


def _lift_height(grasp_pose: GraspPose) -> float:
    return max(0.10, grasp_pose.retreat_position[2] - grasp_pose.grasp_position[2])


def _safe_lift_z(
    grasp_pose: GraspPose,
    workspace: WorkspaceBounds,
    table_safety: TableSafetyConfig | None,
) -> float:
    requested = max(
        grasp_pose.pre_grasp_position[2],
        grasp_pose.retreat_position[2],
        grasp_pose.grasp_position[2] + _lift_height(grasp_pose),
    )
    if table_safety is not None:
        requested = max(requested, table_safety.vertical_escape_height)
    return min(workspace.z_max, requested)


def _with_z(position: tuple[float, float, float], z: float) -> tuple[float, float, float]:
    return (position[0], position[1], z)


def _validate_link_table_clearance(
    waypoints: list[TrajectoryWaypoint],
    table_safety: TableSafetyConfig,
    link_1: float,
    link_2: float,
    base_height: float,
) -> str | None:
    min_safe_z = table_safety.table_top_z + table_safety.min_link_clearance_meters
    for waypoint in waypoints:
        ik_result = solve_ik(waypoint.position, link_1, link_2, base_height)
        if ik_result.status == "invalid":
            return "link_table_penetration_risk"
        positions = forward_link_positions(ik_result.angles, link_1, link_2, base_height)
        base = (0.0, 0.0, base_height)
        link_points = [
            *sample_link_segment_points(base, positions["shoulder"]),
            *sample_link_segment_points(positions["shoulder"], positions["elbow"]),
            *sample_link_segment_points(positions["elbow"], positions["wrist"]),
            *sample_link_segment_points(positions["wrist"], positions["end_effector"]),
        ]
        min_link_z = min(point[2] for point in link_points)
        if min_link_z < min_safe_z:
            return "link_table_penetration_risk"
    return None
