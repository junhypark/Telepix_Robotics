"""Schemas used by the run-result dashboard."""

from __future__ import annotations

from typing import Literal

from robot_sorting.schemas import ObjectLabel, StrictBaseModel

TransportMode = Literal["overhead_rotate", "level_parallel"]


class DashboardSummary(StrictBaseModel):
    """Dashboard card values for one run."""

    run_id: str
    total_objects: int
    normal_count: int
    defect_count: int
    placed_count: int
    failed_count: int
    success_rate: float
    sla_passed: bool
    average_command_latency_seconds: float
    max_command_latency_seconds: float
    conveyor_enabled: bool
    conveyor_stop_count: int
    overhead_rotate_count: int
    level_parallel_count: int
    normal_bin_detected: bool
    defect_bin_detected: bool
    table_penetration_failure_count: int = 0
    collision_failure_count: int = 0
    no_free_space_inside_bin_count: int = 0


class DashboardObjectRow(StrictBaseModel):
    """Per-object row displayed by the dashboard."""

    object_id: str
    label: ObjectLabel
    status: str
    target_bin: str | None
    transport_mode: str | None
    failure_reason: str | None
    command_latency_seconds: float | None
    station_cycle_time_seconds: float | None


class DashboardData(StrictBaseModel):
    """Complete dashboard data payload."""

    summary: DashboardSummary
    objects: list[DashboardObjectRow]
    timeline: list[dict[str, object]]
    events: list[dict[str, object]]
    images: dict[str, str]
