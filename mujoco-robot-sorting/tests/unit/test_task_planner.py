"""Unit tests for the task planner."""

from __future__ import annotations

import math

import pytest

from robot_sorting.modules.task_planner import TaskPlanner
from robot_sorting.schemas import ExternalInspectionResult, SimulationConfig

pytestmark = pytest.mark.unit


def _inspection(
    object_id: str,
    label: str,
    position: tuple[float, float, float],
    confidence: float = 0.9,
    inspected_at: float = 1.0,
) -> ExternalInspectionResult:
    return ExternalInspectionResult(
        object_id=object_id,
        label=label,  # type: ignore[arg-type]
        world_position=position,
        confidence=confidence,
        inspected_at=inspected_at,
    )


def test_normal_object_maps_to_normal_bin(config: SimulationConfig) -> None:
    task = TaskPlanner(config).plan([_inspection("a", "normal", (0.2, 0.1, 0.04))])[0]

    assert task.target_bin == "normal_bin"
    assert task.pick_position != task.place_position


def test_defect_object_maps_to_defect_bin(config: SimulationConfig) -> None:
    task = TaskPlanner(config).plan([_inspection("a", "defect", (0.2, -0.1, 0.04))])[0]

    assert task.target_bin == "defect_bin"
    assert task.pick_position != task.place_position


def test_planner_sorts_objects_by_nearest_distance(config: SimulationConfig) -> None:
    tasks = TaskPlanner(config).plan(
        [
            _inspection("far", "normal", (0.4, 0.0, 0.04)),
            _inspection("near", "defect", (0.2, 0.0, 0.04)),
        ]
    )

    assert [task.object_id for task in tasks] == ["near", "far"]


def test_empty_input_returns_empty_task_list(config: SimulationConfig) -> None:
    assert TaskPlanner(config).plan([]) == []


def test_low_confidence_objects_are_ignored(config: SimulationConfig) -> None:
    assert TaskPlanner(config).plan([_inspection("low", "normal", (0.2, 0.1, 0.04), 0.1)]) == []


def test_objects_outside_workspace_are_rejected(config: SimulationConfig) -> None:
    outside = (config.workspace.x_max + 1.0, 0.0, 0.04)

    assert TaskPlanner(config).plan([_inspection("outside", "normal", outside)]) == []


def test_duplicate_object_ids_are_handled_deterministically(config: SimulationConfig) -> None:
    tasks = TaskPlanner(config).plan(
        [
            _inspection("dup", "normal", (0.25, 0.0, 0.04), 0.7, 3.0),
            _inspection("dup", "defect", (0.20, 0.0, 0.04), 0.9, 2.0),
        ]
    )

    assert len(tasks) == 1
    assert tasks[0].label == "defect"


def test_planner_output_contains_finite_coordinates(config: SimulationConfig) -> None:
    tasks = TaskPlanner(config).plan([_inspection("a", "normal", (0.2, 0.1, 0.04))])

    assert tasks
    for task in tasks:
        assert all(math.isfinite(v) for v in task.pick_position)
        assert all(math.isfinite(v) for v in task.place_position)

