"""Unit tests for the OpenCV vision module."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from robot_sorting.modules.vision_module import VisionModule
from robot_sorting.schemas import DetectedObject, SimulationConfig

pytestmark = pytest.mark.unit


def _image(background: tuple[int, int, int] = (190, 190, 190)) -> np.ndarray:
    return np.full((240, 320, 3), background, dtype=np.uint8)


def _draw_circle(image: np.ndarray, center: tuple[int, int], color: tuple[int, int, int], radius: int = 18) -> None:
    cv2.circle(image, center, radius, color, thickness=-1)


def _assert_inside_workspace(obj: DetectedObject, config: SimulationConfig) -> None:
    workspace = config.workspace
    assert obj.confidence >= config.min_confidence
    assert obj.label in {"normal", "defect"}
    assert workspace.x_min <= obj.world_position[0] <= workspace.x_max
    assert workspace.y_min <= obj.world_position[1] <= workspace.y_max
    assert workspace.z_min <= obj.world_position[2] <= workspace.z_max


def test_detects_blue_normal_and_red_defect(config: SimulationConfig) -> None:
    image = _image()
    _draw_circle(image, (80, 120), (0, 0, 255))
    _draw_circle(image, (230, 100), (255, 0, 0))

    detections = VisionModule(config).detect(image)

    assert {obj.label for obj in detections} == {"normal", "defect"}
    blue = next(obj for obj in detections if obj.label == "normal")
    red = next(obj for obj in detections if obj.label == "defect")
    assert abs(blue.pixel_center[0] - 80) <= 2
    assert abs(blue.pixel_center[1] - 120) <= 2
    assert abs(red.pixel_center[0] - 230) <= 2
    assert abs(red.pixel_center[1] - 100) <= 2
    for obj in detections:
        _assert_inside_workspace(obj, config)


@pytest.mark.parametrize(
    ("background", "draw_color", "expected_label"),
    [
        ((25, 45, 120), (0, 0, 255), "normal"),
        ((120, 45, 45), (255, 0, 0), "defect"),
    ],
)
def test_background_similar_to_object_is_handled(
    config: SimulationConfig,
    background: tuple[int, int, int],
    draw_color: tuple[int, int, int],
    expected_label: str,
) -> None:
    image = _image(background)
    _draw_circle(image, (160, 120), draw_color)

    detections = VisionModule(config).detect(image)

    assert any(obj.label == expected_label for obj in detections)


def test_low_saturation_object_is_ignored_gracefully(config: SimulationConfig) -> None:
    image = _image()
    _draw_circle(image, (160, 120), (120, 120, 150))

    assert VisionModule(config).detect(image) == []


def test_lighting_noise_does_not_crash_detection(config: SimulationConfig) -> None:
    rng = np.random.default_rng(3)
    image = _image()
    noise = rng.integers(0, 35, size=image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    _draw_circle(image, (160, 120), (255, 0, 0))

    detections = VisionModule(config).detect(image)

    assert any(obj.label == "defect" for obj in detections)


def test_very_small_object_below_min_area_is_ignored(config: SimulationConfig) -> None:
    image = _image()
    _draw_circle(image, (160, 120), (0, 0, 255), radius=2)

    assert VisionModule(config).detect(image) == []


def test_partially_visible_object_keeps_valid_world_coordinates(config: SimulationConfig) -> None:
    image = _image()
    _draw_circle(image, (5, 5), (0, 0, 255), radius=18)

    detections = VisionModule(config).detect(image)

    assert detections
    for obj in detections:
        _assert_inside_workspace(obj, config)


def test_overlapping_red_blue_objects_are_deterministic(config: SimulationConfig) -> None:
    image = _image()
    _draw_circle(image, (150, 120), (0, 0, 255), radius=24)
    _draw_circle(image, (165, 120), (255, 0, 0), radius=24)

    first = VisionModule(config).detect(image)
    second = VisionModule(config).detect(image)

    assert [obj.model_dump() for obj in first] == [obj.model_dump() for obj in second]
    assert {obj.label for obj in first}.issubset({"normal", "defect"})


def test_plain_background_returns_empty_list(config: SimulationConfig) -> None:
    assert VisionModule(config).detect(_image()) == []


def test_ambiguous_purple_is_not_confidently_classified(config: SimulationConfig) -> None:
    image = _image()
    _draw_circle(image, (160, 120), (140, 0, 140))

    assert VisionModule(config).detect(image) == []

