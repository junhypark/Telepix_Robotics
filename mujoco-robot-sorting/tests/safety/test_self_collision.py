"""Safety tests for base/body self-collision prevention."""

from __future__ import annotations

import pytest

from robot_sorting.modules.command_queue import create_robot_commands
from robot_sorting.robot.controller import RobotController
from robot_sorting.robot.safety import (
    check_self_collision_risk,
    is_inside_base_exclusion_zone,
    sample_line_segment,
)
from robot_sorting.schemas import PickPlaceTask, SimulationConfig
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv

pytestmark = pytest.mark.safety


def test_base_exclusion_zone_geometry(config: SimulationConfig) -> None:
    base_radius = config.base_radius
    base_height = config.base_height

    assert is_inside_base_exclusion_zone((0.01, 0.01, 0.10), base_radius, 0.0, base_height)
    assert not is_inside_base_exclusion_zone((0.40, 0.20, 0.20), base_radius, 0.0, base_height)


def test_self_collision_risk_for_unsafe_and_safe_points(config: SimulationConfig) -> None:
    unsafe_link_points = [(0.01, 0.01, 0.10), (0.20, 0.10, 0.20)]
    safe_link_points = [(0.40, 0.20, 0.20), (0.45, 0.25, 0.18)]

    assert check_self_collision_risk(
        unsafe_link_points,
        config.base_radius,
        config.base_height,
        config.safety_margin,
    )
    assert not check_self_collision_risk(
        safe_link_points,
        config.base_radius,
        config.base_height,
        config.safety_margin,
    )


def test_planned_path_crossing_base_exclusion_zone_is_rejected(config: SimulationConfig) -> None:
    path = sample_line_segment((-0.30, 0.0, 0.08), (0.30, 0.0, 0.08), samples=20)

    assert check_self_collision_risk(path, config.base_radius, config.base_height, config.safety_margin)


def test_controller_marks_unsafe_command_as_self_collision(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)
    controller = RobotController(env, two_object_config)
    task = PickPlaceTask(
        object_id="object_0",
        label="normal",
        pick_position=(0.01, 0.01, 0.10),
        place_position=two_object_config.normal_bin_position,
        target_bin="normal_bin",
    )

    result = controller.execute_command(create_robot_commands([task])[0])

    assert result.status == "failed"
    assert result.failure_reason == "self_collision_risk"


def test_full_pick_and_place_path_stays_outside_base(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)
    controller = RobotController(env, two_object_config)
    task = PickPlaceTask(
        object_id="object_0",
        label="normal",
        pick_position=(0.24, 0.16, two_object_config.workspace.z_min),
        place_position=two_object_config.normal_bin_position,
        target_bin="normal_bin",
    )

    result = controller.execute_command(create_robot_commands([task])[0])

    assert result.status == "completed"

