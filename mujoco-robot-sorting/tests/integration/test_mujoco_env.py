"""Integration tests for the MuJoCo environment wrapper."""

from __future__ import annotations

import pytest

from robot_sorting.schemas import SimulationConfig
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv
from robot_sorting.simulation.renderer import MujocoCameraRenderer

pytestmark = pytest.mark.integration


def test_mujoco_model_loads_successfully(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)

    assert env.model is not None
    assert env.data is not None
    assert env.model.nbody > 0
    assert env.model.ngeom > 0


def test_simulation_can_step_headless(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)

    env.step(steps=10)

    assert two_object_config.headless
    assert env.data is not None


def test_scene_contains_required_named_entities(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)

    for name in [
        "robot_base",
        "robot_upper_link",
        "robot_forearm_link",
        "end_effector",
        "table",
        "normal_bin",
        "defect_bin",
        "object_0",
    ]:
        assert env.get_body_id(name) >= 0
    assert env.get_camera_id(two_object_config.renderer_camera) >= 0


@pytest.mark.rendering
def test_camera_rendering_works_or_skips(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)
    image = MujocoCameraRenderer(env).render_rgb()
    if image is None:
        pytest.skip("MuJoCo rendering unavailable in this environment")

    assert image.shape == (two_object_config.height, two_object_config.width, 3)

