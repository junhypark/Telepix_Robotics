"""Integration tests for robot movement and safety during execution."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sorting.modules.command_queue import create_robot_commands
from robot_sorting.robot.controller import RobotController
from robot_sorting.robot.safety import is_inside_workspace
from robot_sorting.schemas import PickPlaceTask, SimulationConfig
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv

pytestmark = pytest.mark.integration


def _tasks(config: SimulationConfig) -> list[PickPlaceTask]:
    return [
        PickPlaceTask(
            object_id="object_0",
            label="normal",
            pick_position=(0.22, 0.12, config.workspace.z_min),
            place_position=config.normal_bin_position,
            target_bin="normal_bin",
        ),
        PickPlaceTask(
            object_id="object_1",
            label="defect",
            pick_position=(0.24, -0.12, config.workspace.z_min),
            place_position=config.defect_bin_position,
            target_bin="defect_bin",
        ),
    ]


def test_move_to_basic_pick_and_place_positions(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)
    controller = RobotController(env, two_object_config)

    for target in [
        (0.22, 0.12, 0.15),
        (0.24, -0.12, 0.15),
        (two_object_config.normal_bin_position[0], two_object_config.normal_bin_position[1], 0.16),
        (two_object_config.defect_bin_position[0], two_object_config.defect_bin_position[1], 0.16),
    ]:
        final_pos = controller.move_end_effector_to(target)
        assert is_inside_workspace(final_pos, two_object_config.workspace)


def test_execute_full_and_multiple_pick_place_tasks(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)
    controller = RobotController(env, two_object_config)
    commands = create_robot_commands(_tasks(two_object_config))

    results = controller.execute_commands(commands)

    assert [result.status for result in results] == ["completed", "completed"]
    assert np.all(np.isfinite(env.data.qpos))
    assert np.all(np.isfinite(env.data.qvel))
    ee_pos = env.get_end_effector_position()
    assert two_object_config.workspace.x_min <= ee_pos[0] <= two_object_config.workspace.x_max
    assert two_object_config.workspace.y_min <= ee_pos[1] <= two_object_config.workspace.y_max
    assert two_object_config.workspace.z_min <= ee_pos[2] <= two_object_config.workspace.z_max


def test_unsafe_base_target_is_rejected(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)
    controller = RobotController(env, two_object_config)
    unsafe_task = PickPlaceTask(
        object_id="object_0",
        label="normal",
        pick_position=(0.01, 0.01, 0.08),
        place_position=two_object_config.normal_bin_position,
        target_bin="normal_bin",
    )

    result = controller.execute_command(create_robot_commands([unsafe_task])[0])

    assert result.status == "failed"
    assert result.failure_reason == "self_collision_risk"

