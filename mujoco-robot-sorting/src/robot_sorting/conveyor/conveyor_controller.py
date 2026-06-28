"""Kinematic conveyor controller used by the automation-cell run flow."""

from __future__ import annotations

import math

from robot_sorting.conveyor.conveyor_schemas import ConveyorConfig, ConveyorObjectState, StationState

STOP_STATES = {"INSPECTING", "WAITING_FOR_PICK", "PICKING", "PLACING"}


def update_conveyor_object_position(
    state: ConveyorObjectState,
    config: ConveyorConfig,
    dt: float,
) -> ConveyorObjectState:
    """Move one object along the configured conveyor axis."""

    if state.status != "on_conveyor" or dt <= 0.0:
        return state
    index = 0 if config.conveyor_axis == "x" else 1
    current = list(state.current_position)
    target = config.inspection_zone_center[index]
    direction = 1.0 if target >= current[index] else -1.0
    next_axis_value = current[index] + direction * config.conveyor_speed_mps * dt
    if (direction > 0.0 and next_axis_value >= target) or (direction < 0.0 and next_axis_value <= target):
        current[index] = target
    else:
        current[index] = next_axis_value
    next_position = (float(current[0]), float(current[1]), float(current[2]))
    if has_reached_inspection_zone(
        next_position,
        config.inspection_zone_center,
        config.zone_tolerance_meters,
    ):
        return state.model_copy(
            update={
                "current_position": config.pick_zone_center,
                "target_inspection_position": config.inspection_zone_center,
                "status": "at_inspection_zone",
            }
        )
    return state.model_copy(update={"current_position": next_position})


def has_reached_inspection_zone(
    position: tuple[float, float, float],
    inspection_zone_center: tuple[float, float, float],
    tolerance_meters: float,
) -> bool:
    """Return whether a product is inside the inspection-zone tolerance."""

    return math.dist(position[:2], inspection_zone_center[:2]) <= tolerance_meters


def should_stop_conveyor(station_state: StationState) -> bool:
    """Return whether the conveyor should be stopped for the station state."""

    return station_state.state in STOP_STATES


def transition_station_state(
    current: StationState,
    next_state: StationState,
) -> StationState:
    """Return a deterministic station transition target."""

    if next_state.timestamp < current.timestamp:
        raise ValueError("Station state timestamps must be monotonic")
    return next_state
