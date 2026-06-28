"""Unit tests for the kinematic conveyor controller."""

from __future__ import annotations

import pytest

from robot_sorting.conveyor.conveyor_controller import (
    has_reached_inspection_zone,
    should_stop_conveyor,
    transition_station_state,
    update_conveyor_object_position,
)
from robot_sorting.conveyor.conveyor_schemas import ConveyorConfig, ConveyorObjectState, StationState

pytestmark = pytest.mark.unit


def _config() -> ConveyorConfig:
    return ConveyorConfig(
        conveyor_speed_mps=0.05,
        entry_position=(0.10, -0.20, 0.035),
        inspection_zone_center=(0.30, -0.20, 0.035),
        pick_zone_center=(0.30, -0.20, 0.035),
        max_feed_count=2,
    )


def _state() -> ConveyorObjectState:
    return ConveyorObjectState(
        object_id="object_0",
        label="normal",
        current_position=(0.10, -0.20, 0.035),
        target_inspection_position=(0.30, -0.20, 0.035),
        status="on_conveyor",
    )


def test_conveyor_moves_object_along_axis() -> None:
    previous = _state()

    updated = update_conveyor_object_position(previous, _config(), dt=1.0)

    assert updated.current_position[0] > previous.current_position[0]


def test_conveyor_stops_at_inspection_zone() -> None:
    state = _state()
    config = _config()

    for _ in range(20):
        state = update_conveyor_object_position(state, config, dt=0.5)

    assert state.status == "at_inspection_zone"
    assert has_reached_inspection_zone(state.current_position, config.inspection_zone_center, 0.03)


def test_conveyor_stop_states_cover_inspection_and_pick() -> None:
    assert should_stop_conveyor(StationState(state="INSPECTING", active_object_id="object_0", timestamp=1.0))
    assert should_stop_conveyor(StationState(state="PICKING", active_object_id="object_0", timestamp=1.0))
    moving = StationState(state="MOVING_TO_INSPECTION", active_object_id="object_0", timestamp=1.0)
    assert not should_stop_conveyor(moving)


def test_station_transitions_are_monotonic() -> None:
    current = StationState(state="FEEDING", active_object_id="object_0", timestamp=1.0)
    next_state = StationState(state="INSPECTING", active_object_id="object_0", timestamp=2.0)

    assert transition_station_state(current, next_state) == next_state

    with pytest.raises(ValueError):
        transition_station_state(next_state, current)
