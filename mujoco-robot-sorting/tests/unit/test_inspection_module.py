"""Unit tests for the external inspection module."""

from __future__ import annotations

import pytest

from robot_sorting.modules.inspection_module import InspectionModule
from robot_sorting.schemas import DetectedObject

pytestmark = pytest.mark.unit


def _detected(label: str = "normal", confidence: float = 0.9) -> DetectedObject:
    return DetectedObject(
        object_id=f"{label}_1",
        label=label,  # type: ignore[arg-type]
        pixel_center=(10, 20),
        world_position=(0.2, 0.1, 0.04),
        confidence=confidence,
    )


def test_normal_detected_object_returns_normal_result() -> None:
    result = InspectionModule().inspect(_detected("normal"))

    assert result.label == "normal"


def test_defect_detected_object_returns_defect_result() -> None:
    result = InspectionModule().inspect(_detected("defect"))

    assert result.label == "defect"


def test_low_confidence_object_is_handled_gracefully() -> None:
    result = InspectionModule().inspect(_detected("normal", confidence=0.1))

    assert result.confidence == 0.1


def test_inspection_result_includes_timestamp() -> None:
    result = InspectionModule().inspect(_detected("normal"))

    assert result.inspected_at > 0


def test_inspection_preserves_object_id_and_world_position() -> None:
    detected = _detected("defect")
    result = InspectionModule().inspect(detected)

    assert result.object_id == detected.object_id
    assert result.world_position == detected.world_position

