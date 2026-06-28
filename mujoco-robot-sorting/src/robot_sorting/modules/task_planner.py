"""Task planner for sorting inspected objects into bins."""

from __future__ import annotations

import math

import numpy as np

from robot_sorting.robot.safety import is_inside_base_exclusion_zone, is_inside_workspace
from robot_sorting.schemas import (
    BinPlacementDecision,
    DetectedBin,
    ExternalInspectionResult,
    PickPlaceTask,
    SimulationConfig,
    TargetBin,
)


class TaskPlanner:
    """Create deterministic pick-and-place tasks from inspection results."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def plan(
        self,
        inspection_results: list[ExternalInspectionResult],
        detected_bins: list[DetectedBin] | None = None,
        bin_placement_decisions: list[BinPlacementDecision] | None = None,
    ) -> list[PickPlaceTask]:
        """Plan safe, deterministic sorting tasks."""

        unique_results = self._deduplicate(inspection_results)
        placement_by_object_id = {
            decision.object_id: decision
            for decision in bin_placement_decisions or []
        }
        bin_by_label = self._valid_bins_by_label(detected_bins or [])
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
        tasks: list[PickPlaceTask] = []
        for result in candidates:
            task = self._to_task(result, bin_by_label, placement_by_object_id.get(result.object_id))
            if task is not None:
                tasks.append(task)
        return tasks

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

    def _to_task(
        self,
        result: ExternalInspectionResult,
        bin_by_label: dict[TargetBin, DetectedBin],
        placement_decision: BinPlacementDecision | None,
    ) -> PickPlaceTask | None:
        target_bin: TargetBin = "normal_bin" if result.label == "normal" else "defect_bin"
        target_bin_id: str | None = None
        placement_strategy = None
        placement_confidence = None
        if placement_decision is not None:
            if placement_decision.failure_reason is not None:
                return None
            place = placement_decision.placement_position
            target_bin_id = placement_decision.target_bin_id
            placement_strategy = placement_decision.placement_strategy
            placement_confidence = placement_decision.confidence
        elif target_bin in bin_by_label:
            detected_bin = bin_by_label[target_bin]
            place = detected_bin.world_position
            target_bin_id = detected_bin.bin_id
            placement_strategy = "fallback_center"
            placement_confidence = detected_bin.confidence
        else:
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
            target_bin_id=target_bin_id,
            placement_strategy=placement_strategy,
            placement_confidence=placement_confidence,
            inspection_at=result.inspected_at,
        )

    def _valid_bins_by_label(self, detected_bins: list[DetectedBin]) -> dict[TargetBin, DetectedBin]:
        by_label: dict[TargetBin, DetectedBin] = {}
        for detected_bin in sorted(detected_bins, key=lambda item: (-item.confidence, item.bin_id)):
            if detected_bin.confidence < self.config.min_confidence:
                continue
            if not is_inside_workspace(detected_bin.world_position, self.config.workspace):
                continue
            by_label.setdefault(detected_bin.label, detected_bin)
        return by_label
