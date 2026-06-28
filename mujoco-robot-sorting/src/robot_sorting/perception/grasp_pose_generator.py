"""Top-down grasp pose generation from estimated object poses."""

from __future__ import annotations

import math

from robot_sorting.robot.safety import is_inside_base_exclusion_zone, is_inside_workspace
from robot_sorting.schemas import GraspPose, ObjectPose3D, WorkspaceBounds

DEFAULT_BASE_RADIUS = 0.11
DEFAULT_BASE_HEIGHT = 0.12


def generate_top_down_grasp_pose(
    object_pose: ObjectPose3D,
    grasp_height_offset: float,
    pre_grasp_z_offset: float,
    retreat_z_offset: float,
    workspace: WorkspaceBounds,
) -> GraspPose | None:
    """Generate a conservative top-down suction grasp pose."""

    x, y, z = object_pose.position
    object_top_z = z + max(object_pose.size_xyz[2] / 2.0, 0.0)
    grasp_position = (x, y, object_top_z + grasp_height_offset)
    pre_grasp_position = (x, y, grasp_position[2] + pre_grasp_z_offset)
    retreat_position = (x, y, grasp_position[2] + retreat_z_offset)
    for point in (grasp_position, pre_grasp_position, retreat_position):
        if not is_inside_workspace(point, workspace):
            return None
        if is_inside_base_exclusion_zone(point, DEFAULT_BASE_RADIUS, 0.0, DEFAULT_BASE_HEIGHT):
            return None
    yaw = object_pose.orientation_rpy[2]
    if not math.isfinite(yaw):
        yaw = 0.0
    return GraspPose(
        object_id=object_pose.object_id,
        pre_grasp_position=pre_grasp_position,
        grasp_position=grasp_position,
        retreat_position=retreat_position,
        approach_vector=(0.0, 0.0, -1.0),
        gripper_yaw=yaw,
        confidence=object_pose.confidence,
    )

