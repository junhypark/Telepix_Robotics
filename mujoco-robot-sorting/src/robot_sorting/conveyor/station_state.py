"""Timeline and event helpers for conveyor station logging."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_sorting.conveyor.conveyor_schemas import ConveyorObjectState, StationState, StationStateName

STATION_TIMELINE_FIELDS = [
    "timestamp",
    "state",
    "active_object_id",
    "object_label",
    "object_x",
    "object_y",
    "object_z",
    "conveyor_running",
    "event",
    "reason",
]


@dataclass
class StationTimelineRecorder:
    """Collect station timeline rows and conveyor events."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        timestamp: float,
        state: StationStateName,
        event: str,
        conveyor_running: bool,
        object_state: ConveyorObjectState | None = None,
        reason: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> StationState:
        """Record a station row and matching event entry."""

        object_id = object_state.object_id if object_state is not None else None
        label = object_state.label if object_state is not None else None
        position = object_state.current_position if object_state is not None else (None, None, None)
        row = {
            "timestamp": round(timestamp, 6),
            "state": state,
            "active_object_id": object_id,
            "object_label": label,
            "object_x": position[0],
            "object_y": position[1],
            "object_z": position[2],
            "conveyor_running": conveyor_running,
            "event": event,
            "reason": reason,
        }
        self.rows.append(row)
        event_row = {
            "timestamp": round(timestamp, 6),
            "state": state,
            "object_id": object_id,
            "event": event,
            "reason": reason,
        }
        if extra:
            event_row.update(extra)
        self.events.append(event_row)
        return StationState(state=state, active_object_id=object_id, timestamp=timestamp, reason=reason)

    def write(self, output_dir: Path) -> tuple[Path, Path]:
        """Write station timeline CSV and conveyor events JSON."""

        output_dir.mkdir(parents=True, exist_ok=True)
        timeline_path = output_dir / "station_timeline.csv"
        with timeline_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=STATION_TIMELINE_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        events_path = output_dir / "conveyor_events.json"
        with events_path.open("w", encoding="utf-8") as handle:
            json.dump({"events": self.events}, handle, indent=2, ensure_ascii=False)
        return timeline_path, events_path
