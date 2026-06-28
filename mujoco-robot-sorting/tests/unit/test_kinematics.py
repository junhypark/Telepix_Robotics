"""Unit tests for analytical inverse kinematics."""

from __future__ import annotations

import math

import pytest

from robot_sorting.robot.kinematics import DEFAULT_JOINT_LIMITS, solve_ik
from robot_sorting.schemas import SimulationConfig

pytestmark = pytest.mark.unit


def _assert_joint_limits(result) -> None:
    joint_limits = DEFAULT_JOINT_LIMITS
    assert math.isfinite(result.angles.base_yaw)
    assert math.isfinite(result.angles.shoulder)
    assert math.isfinite(result.angles.elbow)
    assert joint_limits.base_min <= result.angles.base_yaw <= joint_limits.base_max
    assert joint_limits.shoulder_min <= result.angles.shoulder <= joint_limits.shoulder_max
    assert joint_limits.elbow_min <= result.angles.elbow <= joint_limits.elbow_max


def test_reachable_target_returns_finite_joint_angles(config: SimulationConfig) -> None:
    result = solve_ik((0.3, 0.1, 0.12), config.link_1, config.link_2, config.base_height)

    assert result.status == "reachable"
    _assert_joint_limits(result)


def test_unreachable_target_is_clamped_without_nan(config: SimulationConfig) -> None:
    result = solve_ik((2.0, 0.0, 1.0), config.link_1, config.link_2, config.base_height)

    assert result.status == "clamped"
    _assert_joint_limits(result)


def test_symmetric_targets_produce_correct_base_yaw_signs(config: SimulationConfig) -> None:
    positive = solve_ik((0.25, 0.25, 0.12), config.link_1, config.link_2, config.base_height)
    negative = solve_ik((0.25, -0.25, 0.12), config.link_1, config.link_2, config.base_height)

    assert positive.angles.base_yaw > 0
    assert negative.angles.base_yaw < 0


@pytest.mark.parametrize("target", [(0.001, 0.0, 0.12), (0.57, 0.0, 0.12)])
def test_minimum_and_maximum_radius_do_not_crash(
    config: SimulationConfig,
    target: tuple[float, float, float],
) -> None:
    result = solve_ik(target, config.link_1, config.link_2, config.base_height)

    assert result.status in {"reachable", "clamped", "invalid"}
    _assert_joint_limits(result)


def test_target_below_table_is_clamped_or_rejected_safely(config: SimulationConfig) -> None:
    result = solve_ik((0.2, 0.0, -0.5), config.link_1, config.link_2, config.base_height)

    assert result.status in {"clamped", "invalid"}
    _assert_joint_limits(result)


def test_target_above_maximum_reach_is_clamped_or_rejected(config: SimulationConfig) -> None:
    result = solve_ik((0.1, 0.0, 3.0), config.link_1, config.link_2, config.base_height)

    assert result.status in {"clamped", "invalid"}
    _assert_joint_limits(result)


def test_ik_solver_returns_status_field(config: SimulationConfig) -> None:
    result = solve_ik((0.2, 0.0, 0.1), config.link_1, config.link_2, config.base_height)

    assert result.status in {"reachable", "clamped", "invalid"}

