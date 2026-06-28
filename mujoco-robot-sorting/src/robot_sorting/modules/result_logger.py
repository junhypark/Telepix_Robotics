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
    PlacedObjectRecord,
    RunSummary,
    TaskExecutionResult,
)

RESULT_LOG_FIELDS = [
    "object_id",
    "label",
    "pick_x",
    "pick_y",
    "pick_z",
    "object_x",
    "object_y",
    "object_z",
    "object_roll",
    "object_pitch",
    "object_yaw",
    "grasp_x",
    "grasp_y",
    "grasp_z",
    "pre_grasp_x",
    "pre_grasp_y",
    "pre_grasp_z",
    "retreat_x",
    "retreat_y",
    "retreat_z",
    "place_x",
    "place_y",
    "place_z",
    "target_bin",
    "target_bin_id",
    "placement_strategy",
    "placement_confidence",
    "status",
    "failure_reason",
    "trajectory_safe",
    "collision_checked",
    "workspace_checked",
    "command_latency_seconds",
    "self_collision_checked",
    "table_clearance_checked",
    "min_observed_link_z",
    "min_required_link_z",
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

    def write_placed_objects(self, placed_objects: list[PlacedObjectRecord]) -> Path:
        """Write placed product state to JSON."""

        path = self.output_dir / "placed_objects.json"
        self._write_json(path, placed_objects)
        return path

    def write_summary(
        self,
        detections: list[DetectedObject],
        results: list[TaskExecutionResult],
        *,
        min_confidence: float,
        rgbd_used: bool = False,
        depth_fallback_used: bool = False,
        ground_truth_fallback_used: bool = False,
    ) -> Path:
        """Write aggregate run summary to JSON."""

        summary = self.build_summary(
            detections,
            results,
            min_confidence=min_confidence,
            rgbd_used=rgbd_used,
            depth_fallback_used=depth_fallback_used,
            ground_truth_fallback_used=ground_truth_fallback_used,
        )
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
        rgbd_used: bool = False,
        depth_fallback_used: bool = False,
        ground_truth_fallback_used: bool = False,
        placed_objects: list[PlacedObjectRecord] | None = None,
    ) -> RunSummary:
        """Write every required output file and return the summary."""

        self.write_detected_objects(detections)
        self.write_planned_tasks(tasks)
        self.write_result_log(results)
        self.write_placed_objects(placed_objects or [])
        summary = self.build_summary(
            detections,
            results,
            min_confidence=min_confidence,
            rgbd_used=rgbd_used,
            depth_fallback_used=depth_fallback_used,
            ground_truth_fallback_used=ground_truth_fallback_used,
        )
        self._write_json(self.output_dir / "summary.json", summary)
        return summary

    @staticmethod
    def build_summary(
        detections: list[DetectedObject],
        results: list[TaskExecutionResult],
        *,
        min_confidence: float,
        rgbd_used: bool = False,
        depth_fallback_used: bool = False,
        ground_truth_fallback_used: bool = False,
    ) -> RunSummary:
        """Build an aggregate run summary."""

        latencies = [result.command_latency_seconds for result in results]
        placed_count = sum(1 for result in results if result.status == "completed")
        failed_count = sum(1 for result in results if result.status == "failed")
        total_objects = len(detections)
        pose_success = sum(1 for result in results if result.object_pose is not None)
        grasp_success = sum(1 for result in results if result.grasp_pose is not None)
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
            rgbd_used=rgbd_used,
            depth_fallback_used=depth_fallback_used,
            ground_truth_fallback_used=ground_truth_fallback_used,
            pose_estimation_success_count=pose_success,
            pose_estimation_failure_count=max(0, total_objects - pose_success) if rgbd_used else 0,
            grasp_generation_success_count=grasp_success,
            grasp_generation_failure_count=max(0, total_objects - grasp_success) if rgbd_used else 0,
            trajectory_collision_failures=sum(
                1 for item in results if item.failure_reason == "trajectory_collision_risk"
            ),
            table_penetration_failures=sum(
                1 for item in results if item.failure_reason == "link_table_penetration_risk"
            ),
            min_observed_link_z=min(
                (item.min_observed_link_z for item in results if item.min_observed_link_z is not None),
                default=0.0,
            ),
            min_required_link_z=max(
                (item.min_required_link_z for item in results if item.min_required_link_z is not None),
                default=0.0,
            ),
        )

    @staticmethod
    def _result_row(result: TaskExecutionResult) -> dict[str, Any]:
        pick_x, pick_y, pick_z = result.pick_position
        place_x, place_y, place_z = result.place_position
        object_position = result.object_pose.position if result.object_pose is not None else result.pick_position
        object_orientation = (
            result.object_pose.orientation_rpy if result.object_pose is not None else (0.0, 0.0, 0.0)
        )
        grasp_position = result.grasp_pose.grasp_position if result.grasp_pose is not None else result.pick_position
        pre_grasp = result.grasp_pose.pre_grasp_position if result.grasp_pose is not None else result.pick_position
        retreat = result.grasp_pose.retreat_position if result.grasp_pose is not None else result.pick_position
        return {
            "object_id": result.object_id,
            "label": result.label,
            "pick_x": pick_x,
            "pick_y": pick_y,
            "pick_z": pick_z,
            "object_x": object_position[0],
            "object_y": object_position[1],
            "object_z": object_position[2],
            "object_roll": object_orientation[0],
            "object_pitch": object_orientation[1],
            "object_yaw": object_orientation[2],
            "grasp_x": grasp_position[0],
            "grasp_y": grasp_position[1],
            "grasp_z": grasp_position[2],
            "pre_grasp_x": pre_grasp[0],
            "pre_grasp_y": pre_grasp[1],
            "pre_grasp_z": pre_grasp[2],
            "retreat_x": retreat[0],
            "retreat_y": retreat[1],
            "retreat_z": retreat[2],
            "place_x": place_x,
            "place_y": place_y,
            "place_z": place_z,
            "target_bin": result.target_bin,
            "target_bin_id": result.target_bin_id,
            "placement_strategy": result.task_placement_strategy,
            "placement_confidence": result.task_placement_confidence,
            "status": result.status,
            "failure_reason": result.failure_reason,
            "trajectory_safe": result.trajectory_safe,
            "collision_checked": result.collision_checked,
            "workspace_checked": result.workspace_checked,
            "command_latency_seconds": result.command_latency_seconds,
            "self_collision_checked": result.self_collision_checked,
            "table_clearance_checked": result.table_clearance_checked,
            "min_observed_link_z": result.min_observed_link_z,
            "min_required_link_z": result.min_required_link_z,
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
