"""Unit tests for depth processing utilities."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sorting.perception.depth_processor import get_object_depth_from_mask

pytestmark = pytest.mark.unit


def test_median_depth_from_valid_masked_pixels() -> None:
    depth = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], dtype=np.float32)
    mask = np.ones_like(depth, dtype=np.uint8) * 255

    assert get_object_depth_from_mask(depth, mask, 0.1, 2.0) == pytest.approx(0.75)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, 0.0, 9.0])
def test_invalid_depth_values_are_ignored(bad_value: float) -> None:
    depth = np.array([[0.5, 0.6, bad_value], [0.7, 0.8, 0.9]], dtype=np.float32)
    mask = np.ones_like(depth, dtype=np.uint8) * 255

    assert get_object_depth_from_mask(depth, mask, 0.1, 2.0) == pytest.approx(0.7)


def test_empty_mask_returns_none() -> None:
    depth = np.ones((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint8)

    assert get_object_depth_from_mask(depth, mask, 0.1, 2.0) is None


def test_too_few_valid_pixels_returns_none() -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[0, :4] = 0.5
    mask = np.ones((4, 4), dtype=np.uint8) * 255

    assert get_object_depth_from_mask(depth, mask, 0.1, 2.0) is None

