"""Integration test for the Typer CLI."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from robot_sorting.cli import _build_viewer_demo_commands, app
from robot_sorting.schemas import SimulationConfig
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv

pytestmark = pytest.mark.integration


def test_cli_run_creates_required_output_files(tmp_path) -> None:
    output_dir = tmp_path / "run"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--headless",
            "--objects",
            "2",
            "--seed",
            "1",
            "--output",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    for filename in ["result_log.csv", "detected_objects.json", "planned_tasks.json", "summary.json"]:
        assert (output_dir / filename).exists()

    with (output_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    for key in [
        "total_objects",
        "normal_count",
        "defect_count",
        "placed_count",
        "failed_count",
        "success_rate",
        "average_command_latency_seconds",
        "max_command_latency_seconds",
    ]:
        assert key in summary
    assert summary["max_command_latency_seconds"] <= 0.5


def test_viewer_demo_commands_include_planned_trajectories(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)

    commands = _build_viewer_demo_commands(env, two_object_config)

    assert len(commands) == two_object_config.objects
    assert all(command.trajectory is not None for command in commands)
