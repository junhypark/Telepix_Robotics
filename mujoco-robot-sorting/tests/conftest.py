"""Pytest fixtures for robot sorting tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sorting.config import create_simulation_config
from robot_sorting.schemas import SimulationConfig


@pytest.fixture()
def config(tmp_path: Path) -> SimulationConfig:
    """Return a small deterministic simulation config for tests."""

    return create_simulation_config(objects=3, seed=7, output_dir=tmp_path / "run")


@pytest.fixture()
def two_object_config(tmp_path: Path) -> SimulationConfig:
    """Return a deterministic config with two scene objects."""

    return create_simulation_config(objects=2, seed=1, output_dir=tmp_path / "run")

