"""Unit tests for result output files."""

from __future__ import annotations

import csv
import json

import pytest

from robot_sorting.modules.result_logger import ResultLogger
from robot_sorting.schemas import DetectedObject, PickPlaceTask, TaskExecutionResult

pytestmark = pytest.mark.unit


def _detections() -> list[DetectedObject]:
    return [
        DetectedObject(
            object_id="object_0",
            label="normal",
            pixel_center=(1, 2),
            world_position=(0.2, 0.1, 0.04),
            confidence=0.9,
        ),
        DetectedObject(
            object_id="object_1",
            label="defect",
            pixel_center=(3, 4),
            world_position=(0.3, -0.1, 0.04),
            confidence=0.9,
        ),
    ]


def _tasks() -> list[PickPlaceTask]:
    return [
        PickPlaceTask(
            object_id="object_0",
            label="normal",
            pick_position=(0.2, 0.1, 0.04),
            place_position=(0.45, 0.28, 0.055),
            target_bin="normal_bin",
        )
    ]


def _results() -> list[TaskExecutionResult]:
    return [
        TaskExecutionResult(
            object_id="object_0",
            label="normal",
            pick_position=(0.2, 0.1, 0.04),
            place_position=(0.45, 0.28, 0.055),
            target_bin="normal_bin",
            status="completed",
            command_latency_seconds=0.01,
            self_collision_checked=True,
            workspace_checked=True,
        ),
        TaskExecutionResult(
            object_id="object_1",
            label="defect",
            pick_position=(0.01, 0.01, 0.04),
            place_position=(0.45, -0.28, 0.055),
            target_bin="defect_bin",
            status="failed",
            failure_reason="self_collision_risk",
            command_latency_seconds=0.02,
            self_collision_checked=True,
            workspace_checked=True,
        ),
    ]


def test_logger_creates_output_directory(tmp_path) -> None:
    output_dir = tmp_path / "outputs" / "run"

    ResultLogger(output_dir)

    assert output_dir.exists()


def test_logger_writes_csv_and_json_outputs(tmp_path) -> None:
    logger = ResultLogger(tmp_path)
    summary = logger.write_all(_detections(), _tasks(), _results(), min_confidence=0.55)

    assert (tmp_path / "result_log.csv").exists()
    assert (tmp_path / "detected_objects.json").exists()
    assert (tmp_path / "planned_tasks.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert summary.total_objects == 2
    assert summary.normal_count == 1
    assert summary.defect_count == 1
    assert summary.placed_count == 1
    assert summary.failed_count == 1
    assert summary.max_command_latency_seconds == 0.02
    assert summary.self_collision_failures == 1


def test_result_csv_contains_required_safety_and_latency_fields(tmp_path) -> None:
    logger = ResultLogger(tmp_path)
    logger.write_result_log(_results())

    with (tmp_path / "result_log.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["failure_reason"] == ""
    assert "command_latency_seconds" in rows[0]
    assert "self_collision_checked" in rows[0]
    assert "workspace_checked" in rows[0]


def test_summary_json_contains_required_fields(tmp_path) -> None:
    logger = ResultLogger(tmp_path)
    logger.write_summary(_detections(), _results(), min_confidence=0.55)

    with (tmp_path / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)

    assert "average_command_latency_seconds" in summary
    assert "max_command_latency_seconds" in summary
    assert "workspace_failures" in summary
    assert "vision_low_confidence_count" in summary

