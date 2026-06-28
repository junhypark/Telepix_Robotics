"""Multipart image upload API tests."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from robot_sorting.api import app
from robot_sorting.config import create_simulation_config

pytestmark = pytest.mark.api


def _png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


def _rgb_image() -> np.ndarray:
    image = np.full((160, 220, 3), 190, dtype=np.uint8)
    cv2.circle(image, (60, 80), 18, (0, 0, 255), thickness=-1)
    cv2.circle(image, (150, 80), 18, (255, 0, 0), thickness=-1)
    return image


def test_inspect_image_upload_returns_detected_objects() -> None:
    config = create_simulation_config(width=220, height=160)
    response = TestClient(app).post(
        "/inspect-image",
        files={"image": ("rgb.png", _png_bytes(_rgb_image()), "image/png")},
        data={"config_json": config.model_dump_json()},
    )

    payload = response.json()
    assert response.status_code == 200
    assert "objects" in payload
    assert "processing_time_seconds" in payload
    assert payload["processing_time_seconds"] >= 0.0
    assert {item["label"] for item in payload["objects"]} == {"normal", "defect"}
    for item in payload["objects"]:
        assert {"object_id", "label", "pixel_center", "world_position", "confidence"} <= set(item)


def test_empty_image_returns_empty_object_list() -> None:
    image = np.full((160, 220, 3), 190, dtype=np.uint8)
    response = TestClient(app).post(
        "/inspect-image",
        files={"image": ("empty.png", _png_bytes(image), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["objects"] == []


def test_invalid_image_upload_returns_400() -> None:
    response = TestClient(app).post(
        "/inspect-image",
        files={"image": ("broken.png", BytesIO(b"not an image"), "image/png")},
    )

    assert response.status_code == 400
    assert "detail" in response.json()

