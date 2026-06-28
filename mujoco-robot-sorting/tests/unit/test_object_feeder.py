"""Unit tests for deterministic conveyor object feeding."""

from __future__ import annotations

import pytest

from robot_sorting.conveyor.object_feeder import (
    can_spawn_next_object,
    create_feed_queue,
    get_next_object_to_feed,
    spawn_object_on_conveyor,
)
from robot_sorting.schemas import GeneratedObject, GeneratedScenario

pytestmark = pytest.mark.unit


def _scenario(seed: int = 7) -> GeneratedScenario:
    objects = [
        GeneratedObject(
            object_id="object_1",
            label="defect",
            target_spawn_position=(0.30, -0.20, 0.035),
            actual_spawn_position=(0.10, -0.20, 0.035),
            size_xyz=(0.05, 0.05, 0.03),
        ),
        GeneratedObject(
            object_id="object_0",
            label="normal",
            target_spawn_position=(0.30, -0.20, 0.035),
            actual_spawn_position=(0.10, -0.20, 0.035),
            size_xyz=(0.05, 0.05, 0.03),
        ),
    ]
    return GeneratedScenario(seed=seed, objects=objects, bins=[], normal_count=1, defect_count=1)


def test_same_seed_creates_same_feed_queue() -> None:
    first = create_feed_queue(_scenario(seed=42))
    second = create_feed_queue(_scenario(seed=42))

    assert [item.object_id for item in first] == [item.object_id for item in second]


def test_feed_queue_includes_normal_and_defective_products() -> None:
    queue = create_feed_queue(_scenario())

    assert {item.label for item in queue} == {"normal", "defect"}


def test_feeder_returns_products_in_deterministic_order() -> None:
    queue = create_feed_queue(_scenario())

    first = get_next_object_to_feed(queue)
    second = get_next_object_to_feed(queue)

    assert first is not None
    assert second is not None
    assert first.object_id == "object_0"
    assert second.object_id == "object_1"
    assert get_next_object_to_feed(queue) is None


def test_spawned_object_starts_at_entry_position() -> None:
    obj = create_feed_queue(_scenario())[0]
    state = spawn_object_on_conveyor(obj, (0.12, -0.20, 0.035))

    assert state.current_position == (0.12, -0.20, 0.035)
    assert state.status == "on_conveyor"


def test_feeder_blocks_spawn_while_active_object_is_on_conveyor() -> None:
    obj = create_feed_queue(_scenario())[0]
    active = spawn_object_on_conveyor(obj, (0.12, -0.20, 0.035))

    assert not can_spawn_next_object(active)
    assert can_spawn_next_object(active.model_copy(update={"status": "placed"}))
