"""Collision-aware pick-and-place trajectory planner."""

from __future__ import annotations

import math

from robot_sorting.planning.collision_checker import trajectory_has_collision_risk
from robot_sorting.robot.safety import is_inside_base_exclusion_zone, is_inside_workspace
from robot_sorting.schemas import GraspPose, PlannedTrajectory, TrajectoryWaypoint, WorkspaceBounds


def plan_pick_place_trajectory(
    grasp_pose: GraspPose,
    place_position: tuple[float, float, float],
    workspace: WorkspaceBounds,
    base_radius: float,
    base_height: float,
    safety_margin: float,
) -> PlannedTrajectory:
    """Plan a safe top-down pick-and-place trajectory."""

    above_place = (
        place_position[0],
        place_position[1],
        min(workspace.z_max, place_position[2] + _lift_height(grasp_pose)),
    )
    waypoints = [
        TrajectoryWaypoint(position=grasp_pose.pre_grasp_position, gripper_state="open", duration_seconds=0.10),
        TrajectoryWaypoint(position=grasp_pose.grasp_position, gripper_state="open", duration_seconds=0.08),
        TrajectoryWaypoint(position=grasp_pose.grasp_position, gripper_state="closed", duration_seconds=0.03),
        TrajectoryWaypoint(position=grasp_pose.retreat_position, gripper_state="closed", duration_seconds=0.10),
        TrajectoryWaypoint(position=above_place, gripper_state="closed", duration_seconds=0.14),
        TrajectoryWaypoint(position=place_position, gripper_state="closed", duration_seconds=0.08),
        TrajectoryWaypoint(position=place_position, gripper_state="open", duration_seconds=0.03),
        TrajectoryWaypoint(position=above_place, gripper_state="open", duration_seconds=0.10),
    ]

    failure_reason = _validate_waypoints(waypoints, workspace, base_radius, base_height, safety_margin)
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

