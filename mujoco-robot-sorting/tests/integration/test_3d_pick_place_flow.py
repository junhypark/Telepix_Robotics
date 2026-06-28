"""End-to-end 3D pick-and-place flow without waiting on physical motion timing."""

from __future__ import annotations

import time

import pytest

from robot_sorting.modules.command_queue import create_robot_commands
from robot_sorting.modules.task_planner import TaskPlanner
from robot_sorting.modules.vision_module import VisionModule
from robot_sorting.perception.camera_calibration import build_top_down_workspace_calibration
from robot_sorting.perception.grasp_pose_generator import generate_top_down_grasp_pose
from robot_sorting.perception.pose_estimator import estimate_object_pose_3d
from robot_sorting.planning.trajectory_planner import plan_pick_place_trajectory
from robot_sorting.robot.controller import RobotController
from robot_sorting.schemas import ExternalInspectionResult, SimulationConfig
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv
from robot_sorting.simulation.renderer import MujocoRenderer

pytestmark = pytest.mark.integration


def test_3d_pick_place_flow_queues_and_executes(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)
    frame = MujocoRenderer(env).render_ground_truth_rgbd()
    calibration = build_top_down_workspace_calibration(
        two_object_config.width,
        two_object_config.height,
        two_object_config.workspace,
    )
    masked = VisionModule(two_object_config).detect_with_masks(frame.rgb)[0]
    pose = estimate_object_pose_3d(masked.detected_object, masked.mask, frame.depth, calibration)
    assert pose is not None
    grasp = generate_top_down_grasp_pose(pose, 0.005, 0.11, 0.13, two_object_config.workspace)
    assert grasp is not None

    api_response_received_at = time.perf_counter()
    inspections = [
        ExternalInspectionResult(
            object_id=pose.object_id,
            label=pose.label,
            world_position=grasp.grasp_position,
            confidence=pose.confidence,
            inspected_at=api_response_received_at,
        )
    ]
    tasks = TaskPlanner(two_object_config).plan(inspections)
    trajectory = plan_pick_place_trajectory(
        grasp,
        tasks[0].place_position,
        two_object_config.workspace,
        two_object_config.base_radius,
        two_object_config.base_height,
        two_object_config.safety_margin,
    )
    commands = create_robot_commands(
        tasks,
        trajectories={pose.object_id: trajectory},
        object_poses={pose.object_id: pose},
        grasp_poses={pose.object_id: grasp},
    )
    elapsed_after_api_response = time.perf_counter() - api_response_received_at

    assert commands
    assert elapsed_after_api_response <= 0.5
    result = RobotController(env, two_object_config).execute_command(commands[0])
    assert result.status == "completed"
    assert result.trajectory_safe is True

