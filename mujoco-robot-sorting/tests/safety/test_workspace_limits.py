"""Safety tests for workspace boundaries."""

from __future__ import annotations

import pytest

from robot_sorting.modules.inspection_module import InspectionModule
from robot_sorting.modules.task_planner import TaskPlanner
from robot_sorting.robot.safety import is_inside_workspace
from robot_sorting.schemas import DetectedObject, SimulationConfig

pytestmark = pytest.mark.safety


def test_workspace_accepts_valid_and_rejects_invalid_points(config: SimulationConfig) -> None:
    valid_point = (config.workspace.x_min, config.workspace.y_max, config.workspace.z_min)
    invalid_point = (config.workspace.x_max + 0.01, 0.0, config.workspace.z_min)

    assert is_inside_workspace(valid_point, config.workspace)
    assert not is_inside_workspace(invalid_point, config.workspace)


@pytest.mark.parametrize(
    "point",
    [
        (0.2, 0.0, 0.0),
        (0.2, 0.0, 0.9),
        (2.0, 0.0, 0.1),
        (0.0, -2.0, 0.1),
    ],
)
def test_invalid_workspace_targets_are_rejected(
    config: SimulationConfig,
    point: tuple[float, float, float],
) -> None:
    assert not is_inside_workspace(point, config.workspace)


def test_boundary_targets_are_accepted(config: SimulationConfig) -> None:
    for point in [
        (config.workspace.x_min, 0.0, config.workspace.z_min),
        (config.workspace.x_max, 0.0, config.workspace.z_max),
        (0.0, config.workspace.y_min, config.workspace.z_min),
        (0.0, config.workspace.y_max, config.workspace.z_max),
    ]:
        assert is_inside_workspace(point, config.workspace)


def test_planner_generates_only_workspace_safe_tasks(config: SimulationConfig) -> None:
    detections = [
        DetectedObject(
            object_id="object_0",
            label="normal",
            pixel_center=(1, 1),
            world_position=(0.2, 0.1, config.workspace.z_min),
            confidence=0.95,
        ),
        DetectedObject(
            object_id="object_1",
            label="defect",
            pixel_center=(1, 1),
            world_position=(config.workspace.x_max + 1.0, 0.0, config.workspace.z_min),
            confidence=0.95,
        ),
    ]
    inspections = InspectionModule().inspect_all(detections)

    tasks = TaskPlanner(config).plan(inspections)

    assert len(tasks) == 1
    assert all(is_inside_workspace(task.pick_position, config.workspace) for task in tasks)
    assert all(is_inside_workspace(task.place_position, config.workspace) for task in tasks)

