"""Task planner for sorting inspected objects into bins."""

from __future__ import annotations

import math

import numpy as np

from robot_sorting.robot.safety import is_inside_base_exclusion_zone, is_inside_workspace
from robot_sorting.schemas import ExternalInspectionResult, PickPlaceTask, SimulationConfig, TargetBin


class TaskPlanner:
    """Create deterministic pick-and-place tasks from inspection results."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def plan(self, inspection_results: list[ExternalInspectionResult]) -> list[PickPlaceTask]:
        """Plan safe, deterministic sorting tasks."""

        unique_results = self._deduplicate(inspection_results)
        candidates = [
            result
            for result in unique_results
            if result.confidence >= self.config.min_confidence
            and is_inside_workspace(result.world_position, self.config.workspace)
            and not is_inside_base_exclusion_zone(
                result.world_position,
                self.config.base_radius,
                0.0,
                self.config.base_height + self.config.safety_margin,
            )
        ]
        candidates.sort(key=lambda item: (self._base_distance(item.world_position), item.object_id, item.label))
        return [self._to_task(result) for result in candidates]

    def _deduplicate(self, inspection_results: list[ExternalInspectionResult]) -> list[ExternalInspectionResult]:
        by_id: dict[str, ExternalInspectionResult] = {}
        for result in sorted(
            inspection_results,
            key=lambda item: (item.object_id, -item.confidence, item.inspected_at),
        ):
            by_id.setdefault(result.object_id, result)
        return list(by_id.values())

    @staticmethod
    def _base_distance(position: tuple[float, float, float]) -> float:
        return float(math.hypot(position[0], position[1]))

    def _to_task(self, result: ExternalInspectionResult) -> PickPlaceTask:
        target_bin: TargetBin = "normal_bin" if result.label == "normal" else "defect_bin"
        place = self.config.normal_bin_position if target_bin == "normal_bin" else self.config.defect_bin_position
        pick = tuple(float(v) for v in result.world_position)
        if not all(math.isfinite(v) for v in (*pick, *place)):
            raise ValueError(f"Invalid non-finite task coordinate for {result.object_id}")
        if np.allclose(np.array(pick), np.array(place)):
            raise ValueError(f"Pick and place positions are identical for {result.object_id}")
        return PickPlaceTask(
            object_id=result.object_id,
            label=result.label,
            pick_position=pick,
            place_position=place,
            target_bin=target_bin,
            inspection_at=result.inspected_at,
        )

