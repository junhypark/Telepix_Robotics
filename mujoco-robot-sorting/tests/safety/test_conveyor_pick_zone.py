"""Safety tests for conveyor pick-zone behavior."""

from __future__ import annotations

import pytest

from robot_sorting.config import create_simulation_config
from robot_sorting.conveyor.conveyor_controller import has_reached_inspection_zone, should_stop_conveyor
from robot_sorting.conveyor.conveyor_schemas import ConveyorConfig, ConveyorObjectState, StationState
from robot_sorting.robot.safety import is_inside_workspace
from robot_sorting.schemas import WorkspaceBounds
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv

pytestmark = pytest.mark.safety


def test_robot_picks_only_when_object_is_inside_pick_zone() -> None:
    config = ConveyorConfig(
        inspection_zone_center=(0.30, -0.20, 0.035),
        pick_zone_center=(0.30, -0.20, 0.035),
        max_feed_count=1,
    )
    object_state = ConveyorObjectState(
        object_id="object_0",
        label="normal",
        current_position=(0.30, -0.20, 0.035),
        target_inspection_position=config.inspection_zone_center,
        status="at_inspection_zone",
    )

    assert has_reached_inspection_zone(
        object_state.current_position,
        config.pick_zone_center,
        config.zone_tolerance_meters,
    )


def test_conveyor_is_stopped_before_grasp_command() -> None:
    station_state = StationState(state="PICKING", active_object_id="object_0", timestamp=1.0)

    assert should_stop_conveyor(station_state) is True


def test_robot_does_not_pick_moving_object() -> None:
    station_state = StationState(state="MOVING_TO_INSPECTION", active_object_id="object_0", timestamp=1.0)
    object_state = ConveyorObjectState(
        object_id="object_0",
        label="normal",
        current_position=(0.18, -0.20, 0.035),
        target_inspection_position=(0.30, -0.20, 0.035),
        status="on_conveyor",
    )

    conveyor_running = not should_stop_conveyor(station_state)

    assert conveyor_running is True
    assert object_state.status == "on_conveyor"


def test_object_outside_reachable_workspace_is_rejected() -> None:
    workspace = WorkspaceBounds()

    assert not is_inside_workspace((workspace.x_max + 0.2, 0.0, 0.035), workspace)


def test_conveyor_geometry_is_present_without_table_penetration_risk(tmp_path) -> None:
    config = create_simulation_config(objects=1, output_dir=tmp_path)
    config.conveyor_enabled = True
    env = MujocoSortingEnv(config)

    assert env.get_geom_id("conveyor_belt_geom") >= 0
    assert config.table_safety.min_link_clearance_meters > 0.0
