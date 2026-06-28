"""Deterministic product feeder for conveyor-based sorting runs."""

from __future__ import annotations

from robot_sorting.conveyor.conveyor_schemas import ConveyorObjectState
from robot_sorting.schemas import GeneratedObject, GeneratedScenario


def create_feed_queue(scenario: GeneratedScenario) -> list[GeneratedObject]:
    """Return products in deterministic feeder order."""

    return sorted(scenario.objects, key=lambda item: item.object_id)


def get_next_object_to_feed(queue: list[GeneratedObject]) -> GeneratedObject | None:
    """Pop the next product to feed, if one is available."""

    if not queue:
        return None
    return queue.pop(0)


def can_spawn_next_object(active_state: ConveyorObjectState | None) -> bool:
    """Return whether the feeder can introduce a new object."""

    return active_state is None or active_state.status in {"placed", "failed"}


def spawn_object_on_conveyor(
    obj: GeneratedObject,
    entry_position: tuple[float, float, float],
) -> ConveyorObjectState:
    """Create the runtime conveyor state for a product at the entry zone."""

    return ConveyorObjectState(
        object_id=obj.object_id,
        label=obj.label,
        current_position=entry_position,
        target_inspection_position=obj.target_spawn_position,
        status="on_conveyor",
    )
