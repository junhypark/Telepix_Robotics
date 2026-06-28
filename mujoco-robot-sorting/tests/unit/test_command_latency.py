"""Latency tests for inspection-result to robot-command generation."""

from __future__ import annotations

import time

import pytest

from robot_sorting.modules.command_queue import RobotCommandQueue, create_robot_commands
from robot_sorting.modules.inspection_module import InspectionModule
from robot_sorting.modules.task_planner import TaskPlanner
from robot_sorting.schemas import DetectedObject, ExternalInspectionResult, SimulationConfig

pytestmark = [pytest.mark.unit, pytest.mark.latency]


def _detected(index: int = 0) -> DetectedObject:
    return DetectedObject(
        object_id=f"object_{index}",
        label="normal" if index % 2 == 0 else "defect",
        pixel_center=(100 + index, 120),
        world_position=(0.2 + index * 0.01, 0.1, 0.04),
        confidence=0.95,
    )


def test_single_inspection_result_to_command_within_sla(config: SimulationConfig) -> None:
    detected = _detected()
    inspection_result = InspectionModule().inspect(detected)

    start = time.perf_counter()
    tasks = TaskPlanner(config).plan([inspection_result])
    commands = create_robot_commands(tasks)
    queue = RobotCommandQueue()
    queue.extend(commands)
    elapsed_seconds = time.perf_counter() - start

    assert elapsed_seconds <= 0.5
    assert len(queue.pop_all()) == 1


def test_ten_inspection_results_to_commands_within_sla(config: SimulationConfig) -> None:
    inspection_results: list[ExternalInspectionResult] = [
        InspectionModule().inspect(_detected(index)) for index in range(10)
    ]

    start = time.perf_counter()
    tasks = TaskPlanner(config).plan(inspection_results)
    commands = create_robot_commands(tasks)
    elapsed_seconds = time.perf_counter() - start

    assert len(commands) == 10
    assert elapsed_seconds <= 0.5

