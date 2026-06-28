"""Result serialization for robot sorting runs."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from robot_sorting.schemas import (
    DetectedObject,
    PickPlaceTask,
    RunSummary,
    TaskExecutionResult,
)

RESULT_LOG_FIELDS = [
    "object_id",
    "label",
    "pick_x",
    "pick_y",
    "pick_z",
    "place_x",
    "place_y",
    "place_z",
    "target_bin",
    "status",
    "failure_reason",
    "command_latency_seconds",
    "self_collision_checked",
    "workspace_checked",
]


class ResultLogger:
    """Write run outputs as CSV and JSON artifacts."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_detected_objects(self, detections: list[DetectedObject]) -> Path:
        """Write detected objects to JSON."""

        path = self.output_dir / "detected_objects.json"
        self._write_json(path, detections)
        return path

    def write_planned_tasks(self, tasks: list[PickPlaceTask]) -> Path:
        """Write planned tasks to JSON."""

        path = self.output_dir / "planned_tasks.json"
        self._write_json(path, tasks)
        return path

    def write_result_log(self, results: list[TaskExecutionResult]) -> Path:
        """Write per-task execution results to CSV."""

        path = self.output_dir / "result_log.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_LOG_FIELDS)
            writer.writeheader()
            for result in results:
                writer.writerow(self._result_row(result))
        return path

    def write_summary(
        self,
        detections: list[DetectedObject],
        results: list[TaskExecutionResult],
        *,
        min_confidence: float,
    ) -> Path:
        """Write aggregate run summary to JSON."""

        summary = self.build_summary(detections, results, min_confidence=min_confidence)
        path = self.output_dir / "summary.json"
        self._write_json(path, summary)
        return path

    def write_all(
        self,
        detections: list[DetectedObject],
        tasks: list[PickPlaceTask],
        results: list[TaskExecutionResult],
        *,
        min_confidence: float,
    ) -> RunSummary:
        """Write every required output file and return the summary."""

        self.write_detected_objects(detections)
        self.write_planned_tasks(tasks)
        self.write_result_log(results)
        summary = self.build_summary(detections, results, min_confidence=min_confidence)
        self._write_json(self.output_dir / "summary.json", summary)
        return summary

    @staticmethod
    def build_summary(
        detections: list[DetectedObject],
        results: list[TaskExecutionResult],
        *,
        min_confidence: float,
    ) -> RunSummary:
        """Build an aggregate run summary."""

        latencies = [result.command_latency_seconds for result in results]
        placed_count = sum(1 for result in results if result.status == "completed")
        failed_count = sum(1 for result in results if result.status == "failed")
        total_objects = len(detections)
        return RunSummary(
            total_objects=total_objects,
            normal_count=sum(1 for item in detections if item.label == "normal"),
            defect_count=sum(1 for item in detections if item.label == "defect"),
            placed_count=placed_count,
            failed_count=failed_count,
            success_rate=placed_count / total_objects if total_objects else 0.0,
            average_command_latency_seconds=sum(latencies) / len(latencies) if latencies else 0.0,
            max_command_latency_seconds=max(latencies) if latencies else 0.0,
            self_collision_failures=sum(1 for item in results if item.failure_reason == "self_collision_risk"),
            workspace_failures=sum(1 for item in results if item.failure_reason == "workspace_limit"),
            vision_low_confidence_count=sum(1 for item in detections if item.confidence < min_confidence),
        )

    @staticmethod
    def _result_row(result: TaskExecutionResult) -> dict[str, Any]:
        pick_x, pick_y, pick_z = result.pick_position
        place_x, place_y, place_z = result.place_position
        return {
            "object_id": result.object_id,
            "label": result.label,
            "pick_x": pick_x,
            "pick_y": pick_y,
            "pick_z": pick_z,
            "place_x": place_x,
            "place_y": place_y,
            "place_z": place_z,
            "target_bin": result.target_bin,
            "status": result.status,
            "failure_reason": result.failure_reason,
            "command_latency_seconds": result.command_latency_seconds,
            "self_collision_checked": result.self_collision_checked,
            "workspace_checked": result.workspace_checked,
        }

    @staticmethod
    def _write_json(path: Path, data: BaseModel | Sequence[BaseModel]) -> None:
        serializable: Any
        if isinstance(data, BaseModel):
            serializable = data.model_dump(mode="json")
        else:
            serializable = [item.model_dump(mode="json") for item in data]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, ensure_ascii=False)
