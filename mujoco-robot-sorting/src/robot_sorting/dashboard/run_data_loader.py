"""Load run outputs and build dashboard data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

from robot_sorting.dashboard.dashboard_schemas import DashboardData, DashboardObjectRow, DashboardSummary


def build_dashboard_data(output_dir: Path) -> DashboardData:
    """Build dashboard data from run artifacts in an output directory."""

    summary = _read_json_object(output_dir / "summary.json")
    result_rows = _read_csv(output_dir / "result_log.csv")
    timeline_rows = _read_csv(output_dir / "station_timeline.csv")
    events_payload = _read_json_object(output_dir / "conveyor_events.json")
    raw_events = events_payload.get("events", [])
    events = [item for item in raw_events if isinstance(item, dict)] if isinstance(raw_events, list) else []
    transport_by_object = _transport_modes_by_object(events)
    cycle_times = _cycle_times_by_object(timeline_rows)
    objects = [
        DashboardObjectRow(
            object_id=row.get("object_id", ""),
            label=row.get("label", "normal") if row.get("label") in {"normal", "defect"} else "normal",
            status=row.get("status", "unknown"),
            target_bin=row.get("target_bin") or None,
            transport_mode=transport_by_object.get(row.get("object_id", "")),
            failure_reason=row.get("failure_reason") or None,
            command_latency_seconds=_optional_float(row.get("command_latency_seconds")),
            station_cycle_time_seconds=cycle_times.get(row.get("object_id", "")),
        )
        for row in result_rows
    ]
    target_bins = {row.target_bin for row in objects if row.target_bin}
    max_latency = _to_float(summary.get("max_command_latency_seconds"), 0.0)
    dashboard_summary = DashboardSummary(
        run_id=output_dir.name or "run",
        total_objects=_to_int(summary.get("total_objects"), len(objects)),
        normal_count=_to_int(summary.get("normal_count"), 0),
        defect_count=_to_int(summary.get("defect_count"), 0),
        placed_count=_to_int(summary.get("placed_count"), 0),
        failed_count=_to_int(summary.get("failed_count"), 0),
        success_rate=_to_float(summary.get("success_rate"), 0.0),
        sla_passed=max_latency <= 0.5,
        average_command_latency_seconds=_to_float(summary.get("average_command_latency_seconds"), 0.0),
        max_command_latency_seconds=max_latency,
        conveyor_enabled=bool(summary.get("conveyor_enabled", False)),
        conveyor_stop_count=_to_int(summary.get("conveyor_stop_count"), 0),
        overhead_rotate_count=_to_int(summary.get("overhead_rotate_count"), 0),
        level_parallel_count=_to_int(summary.get("level_parallel_count"), 0),
        normal_bin_detected="normal_bin" in target_bins or _to_int(summary.get("normal_count"), 0) > 0,
        defect_bin_detected="defect_bin" in target_bins or _to_int(summary.get("defect_count"), 0) > 0,
        table_penetration_failure_count=_to_int(summary.get("table_penetration_failures"), 0),
        collision_failure_count=_to_int(summary.get("trajectory_collision_failures"), 0),
        no_free_space_inside_bin_count=_to_int(summary.get("no_free_space_inside_bin_count"), 0),
    )
    images = {
        "annotated_detection": "annotated_detection.png",
        "trajectory_preview": "trajectory_preview.png",
    }
    return DashboardData(
        summary=dashboard_summary,
        objects=objects,
        timeline=timeline_rows,
        events=events,
        images=images,
    )


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    return cast(dict[str, object], loaded)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, int | float | str):
        return float(value)
    return None


def _to_float(value: object, default: float) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    return default


def _to_int(value: object, default: int) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    return default


def _transport_modes_by_object(events: list[dict[str, object]]) -> dict[str, str]:
    modes: dict[str, str] = {}
    for event in events:
        mode = event.get("transport_mode")
        object_id = event.get("object_id")
        if isinstance(object_id, str) and isinstance(mode, str):
            modes[object_id] = mode
    return modes


def _cycle_times_by_object(rows: list[dict[str, str]]) -> dict[str, float]:
    bounds: dict[str, list[float]] = {}
    for row in rows:
        object_id = row.get("active_object_id")
        if not object_id:
            continue
        timestamp = float(row.get("timestamp", "0.0") or "0.0")
        bounds.setdefault(str(object_id), []).append(timestamp)
    return {
        object_id: round(max(timestamps) - min(timestamps), 6)
        for object_id, timestamps in bounds.items()
        if timestamps
    }
