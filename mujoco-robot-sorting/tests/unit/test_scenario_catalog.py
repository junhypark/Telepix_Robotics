"""Unit tests for deterministic production scenarios."""

from __future__ import annotations

import pytest

from robot_sorting.cli import _build_viewer_demo_commands
from robot_sorting.scenarios import SCENARIOS, create_config_for_scenario, get_scenario, scenario_ids
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv

pytestmark = pytest.mark.unit


def test_scenario_ids_are_unique() -> None:
    ids = scenario_ids()

    assert len(ids) == len(set(ids))
    assert len(ids) >= 6


def test_each_scenario_creates_matching_config(tmp_path) -> None:
    for scenario in SCENARIOS:
        config = create_config_for_scenario(
            scenario.scenario_id,
            output_dir=tmp_path / scenario.scenario_id,
            save_images=False,
        )

        assert config.scenario_id == scenario.scenario_id
        assert config.objects == len(scenario.labels)
        assert config.object_label_sequence == scenario.labels
        assert config.object_spawn_positions == scenario.object_positions


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(KeyError):
        get_scenario("missing")


def test_each_scenario_scene_loads() -> None:
    for scenario in SCENARIOS:
        config = create_config_for_scenario(scenario.scenario_id, save_images=False)
        env = MujocoSortingEnv(config)

        assert len(env.object_specs) == config.objects
        assert len(env.bin_specs) == 2


def test_viewer_demo_commands_follow_scenario_config() -> None:
    config = create_config_for_scenario("small_batch_single_defect", headless=False, save_images=False)
    env = MujocoSortingEnv(config)

    commands = _build_viewer_demo_commands(env, config)

    assert len(commands) == config.objects
    assert {command.task.target_bin for command in commands} == {"normal_bin", "defect_bin"}
    assert all(command.trajectory is not None for command in commands)
