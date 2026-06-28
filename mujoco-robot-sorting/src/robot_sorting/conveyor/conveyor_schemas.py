"""Schemas for deterministic conveyor-cell state."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from robot_sorting.schemas import ObjectLabel, StrictBaseModel

ConveyorAxis = Literal["x", "y"]
ConveyorObjectStatus = Literal[
    "queued",
    "on_conveyor",
    "at_inspection_zone",
    "inspected",
    "waiting_for_pick",
    "picked",
    "placed",
    "failed",
]
StationStateName = Literal[
    "IDLE",
    "FEEDING",
    "MOVING_TO_INSPECTION",
    "INSPECTING",
    "WAITING_FOR_PICK",
    "PICKING",
    "PLACING",
    "COMPLETED",
    "FAILED",
]


class ConveyorConfig(StrictBaseModel):
    """Configuration for a stable kinematic conveyor abstraction."""

    conveyor_speed_mps: float = Field(default=0.05, gt=0.0)
    conveyor_axis: ConveyorAxis = "x"
    entry_position: tuple[float, float, float] = (0.12, -0.22, 0.035)
    inspection_zone_center: tuple[float, float, float] = (0.30, -0.22, 0.035)
    pick_zone_center: tuple[float, float, float] = (0.30, -0.22, 0.035)
    zone_tolerance_meters: float = Field(default=0.03, gt=0.0)
    stop_during_inspection: bool = True
    stop_during_pick: bool = True
    max_feed_count: int = Field(default=1, ge=0)


class ConveyorObjectState(StrictBaseModel):
    """Runtime state for one product travelling through the conveyor cell."""

    object_id: str
    label: ObjectLabel
    current_position: tuple[float, float, float]
    target_inspection_position: tuple[float, float, float]
    status: ConveyorObjectStatus


class StationState(StrictBaseModel):
    """High-level state of the conveyor inspection and robot pick station."""

    state: StationStateName
    active_object_id: str | None
    timestamp: float
    reason: str | None = None
