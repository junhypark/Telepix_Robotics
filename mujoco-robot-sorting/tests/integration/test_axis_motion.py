"""Integration tests for X, Y, and Z end-effector motion direction."""

from __future__ import annotations

import pytest

from robot_sorting.robot.controller import RobotController
from robot_sorting.schemas import SimulationConfig
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv

pytestmark = pytest.mark.integration

AXIS_TOLERANCE = 0.03


def test_x_axis_increase(two_object_config: SimulationConfig) -> None:
    controller = RobotController(MujocoSortingEnv(two_object_config), two_object_config)
    initial = controller.move_end_effector_to((0.22, 0.18, 0.18))
    final = controller.move_end_effector_to((0.32, 0.18, 0.18))

    assert final[0] > initial[0]
    assert abs(final[1] - initial[1]) <= AXIS_TOLERANCE
    assert abs(final[2] - initial[2]) <= AXIS_TOLERANCE


def test_y_axis_increase(two_object_config: SimulationConfig) -> None:
    controller = RobotController(MujocoSortingEnv(two_object_config), two_object_config)
    initial = controller.move_end_effector_to((0.28, -0.10, 0.18))
    final = controller.move_end_effector_to((0.28, 0.08, 0.18))

    assert final[1] > initial[1]
    assert abs(final[0] - initial[0]) <= AXIS_TOLERANCE
    assert abs(final[2] - initial[2]) <= AXIS_TOLERANCE


def test_z_axis_increase(two_object_config: SimulationConfig) -> None:
    controller = RobotController(MujocoSortingEnv(two_object_config), two_object_config)
    initial = controller.move_end_effector_to((0.30, 0.12, 0.12))
    final = controller.move_end_effector_to((0.30, 0.12, 0.22))

    assert final[2] > initial[2]
    assert abs(final[0] - initial[0]) <= AXIS_TOLERANCE
    assert abs(final[1] - initial[1]) <= AXIS_TOLERANCE


def test_negative_x_and_y_movements(two_object_config: SimulationConfig) -> None:
    controller = RobotController(MujocoSortingEnv(two_object_config), two_object_config)
    initial = controller.move_end_effector_to((-0.22, -0.18, 0.18))
    final = controller.move_end_effector_to((-0.32, -0.25, 0.18))

    assert final[0] < initial[0]
    assert final[1] < initial[1]

