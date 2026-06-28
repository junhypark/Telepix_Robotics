"""Vision-based in-bin placement planning."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from robot_sorting.robot.safety import is_inside_workspace
from robot_sorting.schemas import (
    BinPlacementConfig,
    BinPlacementDecision,
    DetectedBin,
    ObjectPose3D,
    PlacedObjectRecord,
    PlacementStrategy,
    WorkspaceBounds,
)

PLACEMENT_FAILURE_REASON = "no_free_space_inside_bin"


def decide_in_bin_placement_position(
    object_id: str,
    object_size_xyz: tuple[float, float, float],
    detected_bin: DetectedBin,
    bin_depth_region: np.ndarray | None,
    existing_placed_objects: list[ObjectPose3D],
    workspace: WorkspaceBounds,
    safety_margin: float,
) -> BinPlacementDecision:
    """Choose a non-overlapping placement position inside the detected bin."""

    placed = [
        PlacedObjectRecord(
            object_id=item.object_id,
            target_bin_id=detected_bin.bin_id,
            position=item.position,
            size_xyz=item.size_xyz,
        )
        for item in existing_placed_objects
    ]
    config = BinPlacementConfig(object_spacing_margin_meters=max(0.0, safety_margin))
    strategy: PlacementStrategy = "depth_lowest_free_region" if bin_depth_region is not None else "grid_free_slot"
    return _find_slot(
        target_bin=detected_bin,
        object_size_xyz=object_size_xyz,
        placed_objects=placed,
        config=config,
        workspace=workspace,
        object_id=object_id,
        preferred_strategy=strategy,
    )


def find_non_overlapping_bin_slot(
    target_bin: DetectedBin,
    object_size_xyz: tuple[float, float, float],
    placed_objects: list[PlacedObjectRecord],
    config: BinPlacementConfig,
    workspace: WorkspaceBounds,
    *,
    object_id: str = "unknown",
) -> BinPlacementDecision:
    """Find a free grid slot inside a detected bin footprint."""

    return _find_slot(
        target_bin=target_bin,
        object_size_xyz=object_size_xyz,
        placed_objects=placed_objects,
        config=config,
        workspace=workspace,
        object_id=object_id,
        preferred_strategy="grid_free_slot",
    )


def objects_overlap_2d(
    pos_a: tuple[float, float, float],
    size_a: tuple[float, float, float],
    pos_b: tuple[float, float, float],
    size_b: tuple[float, float, float],
    margin_meters: float,
) -> bool:
    """Return whether two conservative axis-aligned footprints overlap in XY."""

    half_a_x = size_a[0] / 2.0 + margin_meters
    half_a_y = size_a[1] / 2.0 + margin_meters
    half_b_x = size_b[0] / 2.0
    half_b_y = size_b[1] / 2.0
    return abs(pos_a[0] - pos_b[0]) <= half_a_x + half_b_x and abs(pos_a[1] - pos_b[1]) <= half_a_y + half_b_y


def _find_slot(
    *,
    target_bin: DetectedBin,
    object_size_xyz: tuple[float, float, float],
    placed_objects: list[PlacedObjectRecord],
    config: BinPlacementConfig,
    workspace: WorkspaceBounds,
    object_id: str,
    preferred_strategy: PlacementStrategy,
) -> BinPlacementDecision:
    candidates = _candidate_positions(target_bin, object_size_xyz, config)
    ordered_candidates = _deterministic_order(candidates, object_id)
    for candidate in ordered_candidates:
        if not is_inside_workspace(candidate, workspace):
            continue
        if any(
            objects_overlap_2d(
                candidate,
                object_size_xyz,
                placed.position,
                placed.size_xyz,
                margin_meters=config.object_spacing_margin_meters,
            )
            for placed in placed_objects
            if placed.target_bin_id == target_bin.bin_id
        ):
            continue
        return BinPlacementDecision(
            object_id=object_id,
            target_bin_id=target_bin.bin_id,
            target_bin=target_bin.label,
            placement_position=candidate,
            placement_pixel=None,
            placement_strategy=preferred_strategy,
            confidence=min(1.0, target_bin.confidence),
            reason="selected_non_overlapping_grid_slot",
            failure_reason=None,
        )
    return BinPlacementDecision(
        object_id=object_id,
        target_bin_id=target_bin.bin_id,
        target_bin=target_bin.label,
        placement_position=target_bin.world_position,
        placement_pixel=target_bin.pixel_center,
        placement_strategy="fallback_center",
        confidence=0.0,
        reason=PLACEMENT_FAILURE_REASON,
        failure_reason=PLACEMENT_FAILURE_REASON,
    )


def _candidate_positions(
    target_bin: DetectedBin,
    object_size_xyz: tuple[float, float, float],
    config: BinPlacementConfig,
) -> list[tuple[float, float, float]]:
    center_x, center_y, center_z = target_bin.world_position
    half_bin_x = target_bin.size_xyz[0] / 2.0
    half_bin_y = target_bin.size_xyz[1] / 2.0
    half_object_x = object_size_xyz[0] / 2.0
    half_object_y = object_size_xyz[1] / 2.0
    margin = config.bin_wall_margin_meters
    min_x = center_x - half_bin_x + margin + half_object_x
    max_x = center_x + half_bin_x - margin - half_object_x
    min_y = center_y - half_bin_y + margin + half_object_y
    max_y = center_y + half_bin_y - margin - half_object_y
    if min_x > max_x or min_y > max_y:
        return []

    xs = _grid_axis(min_x, max_x, config.grid_resolution_meters)
    ys = _grid_axis(min_y, max_y, config.grid_resolution_meters)
    z = center_z
    candidates = [(x, y, z) for x in xs for y in ys]
    return sorted(candidates, key=lambda item: (math.hypot(item[0] - center_x, item[1] - center_y), item[0], item[1]))


def _grid_axis(start: float, end: float, resolution: float) -> list[float]:
    if math.isclose(start, end, rel_tol=1e-9, abs_tol=1e-9):
        return [float(start)]
    safe_resolution = max(resolution, 1e-6)
    values = np.arange(start, end + safe_resolution * 0.5, safe_resolution, dtype=float)
    values = values[(values >= start - 1e-9) & (values <= end + 1e-9)]
    if values.size == 0:
        return [float((start + end) / 2.0)]
    return [float(value) for value in values]


def _deterministic_order(
    candidates: list[tuple[float, float, float]],
    object_id: str,
) -> list[tuple[float, float, float]]:
    if not candidates:
        return []
    digest = hashlib.sha256(object_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % len(candidates)
    return [*candidates[offset:], *candidates[:offset]]
