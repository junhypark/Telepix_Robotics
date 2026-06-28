"""Unit tests for non-overlapping in-bin placement."""

from __future__ import annotations

import pytest

from robot_sorting.planning.bin_placement_planner import find_non_overlapping_bin_slot, objects_overlap_2d
from robot_sorting.schemas import BinPlacementConfig, DetectedBin, PlacedObjectRecord, WorkspaceBounds

pytestmark = pytest.mark.unit


def _bin(size: tuple[float, float, float] = (0.18, 0.15, 0.04)) -> DetectedBin:
    return DetectedBin(
        bin_id="normal_bin",
        label="normal_bin",
        pixel_center=(320, 240),
        world_position=(0.36, 0.10, 0.04),
        orientation_rpy=(0.0, 0.0, 0.0),
        size_xyz=size,
        confidence=0.98,
    )


def test_first_object_can_be_placed_inside_empty_bin() -> None:
    decision = find_non_overlapping_bin_slot(
        _bin(),
        (0.04, 0.04, 0.03),
        [],
        BinPlacementConfig(),
        WorkspaceBounds(),
        object_id="object_0",
    )

    assert decision.failure_reason is None
    assert decision.placement_position is not None


def test_second_object_is_placed_at_non_overlapping_slot() -> None:
    config = BinPlacementConfig()
    object_size = (0.04, 0.04, 0.03)
    decision_1 = find_non_overlapping_bin_slot(_bin(), object_size, [], config, WorkspaceBounds(), object_id="a")
    placed = [
        PlacedObjectRecord(
            object_id="a",
            target_bin_id="normal_bin",
            position=decision_1.placement_position,
            size_xyz=object_size,
        )
    ]

    decision_2 = find_non_overlapping_bin_slot(_bin(), object_size, placed, config, WorkspaceBounds(), object_id="b")

    assert decision_2.failure_reason is None
    assert not objects_overlap_2d(
        decision_1.placement_position,
        object_size,
        decision_2.placement_position,
        object_size,
        margin_meters=config.object_spacing_margin_meters,
    )


def test_candidate_overlapping_with_existing_object_is_rejected() -> None:
    config = BinPlacementConfig(grid_resolution_meters=0.03)
    object_size = (0.04, 0.04, 0.03)
    placed = [
        PlacedObjectRecord(
            object_id="existing",
            target_bin_id="normal_bin",
            position=(0.36, 0.10, 0.04),
            size_xyz=(0.12, 0.10, 0.03),
        )
    ]

    decision = find_non_overlapping_bin_slot(
        _bin(size=(0.34, 0.28, 0.04)),
        object_size,
        placed,
        config,
        WorkspaceBounds(),
        object_id="new",
    )

    assert decision.failure_reason is None
    assert not objects_overlap_2d(
        decision.placement_position,
        object_size,
        placed[0].position,
        placed[0].size_xyz,
        margin_meters=config.object_spacing_margin_meters,
    )


def test_candidate_too_close_to_bin_wall_is_rejected() -> None:
    decision = find_non_overlapping_bin_slot(
        _bin(size=(0.07, 0.07, 0.04)),
        (0.06, 0.06, 0.03),
        [],
        BinPlacementConfig(bin_wall_margin_meters=0.02),
        WorkspaceBounds(),
        object_id="object_0",
    )

    assert decision.failure_reason == "no_free_space_inside_bin"


def test_full_bin_returns_no_free_space() -> None:
    target_bin = _bin(size=(0.08, 0.08, 0.04))
    object_size = (0.04, 0.04, 0.03)
    placed = [
        PlacedObjectRecord(
            object_id="existing",
            target_bin_id="normal_bin",
            position=target_bin.world_position,
            size_xyz=(0.07, 0.07, 0.03),
        )
    ]

    decision = find_non_overlapping_bin_slot(target_bin, object_size, placed, BinPlacementConfig(), WorkspaceBounds())

    assert decision.failure_reason == "no_free_space_inside_bin"


def test_placement_position_is_inside_detected_bin_footprint() -> None:
    target_bin = _bin()
    decision = find_non_overlapping_bin_slot(
        target_bin,
        (0.04, 0.04, 0.03),
        [],
        BinPlacementConfig(),
        WorkspaceBounds(),
        object_id="inside",
    )

    assert abs(decision.placement_position[0] - target_bin.world_position[0]) < target_bin.size_xyz[0] / 2.0
    assert abs(decision.placement_position[1] - target_bin.world_position[1]) < target_bin.size_xyz[1] / 2.0


def test_placement_decision_is_deterministic_by_object_id() -> None:
    args = (_bin(), (0.04, 0.04, 0.03), [], BinPlacementConfig(), WorkspaceBounds())

    first = find_non_overlapping_bin_slot(*args, object_id="stable_object")
    second = find_non_overlapping_bin_slot(*args, object_id="stable_object")

    assert first.placement_position == second.placement_position
