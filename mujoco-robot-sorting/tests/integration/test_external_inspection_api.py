"""Integration tests for the FastAPI external inspection service."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from robot_sorting.api import app
from robot_sorting.config import create_simulation_config
from robot_sorting.schemas import DetectedObject

pytestmark = pytest.mark.integration


def _encoded_test_image() -> str:
    image = np.full((240, 320, 3), (190, 190, 190), dtype=np.uint8)
    cv2.circle(image, (80, 120), 18, (0, 0, 255), thickness=-1)
    cv2.circle(image, (230, 120), 18, (255, 0, 0), thickness=-1)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_inspect_image_endpoint_detects_and_inspects_objects() -> None:
    config = create_simulation_config(objects=2, width=320, height=240)
    response = TestClient(app).post(
        "/inspect-image",
        json={
            "image_base64": _encoded_test_image(),
            "config": config.model_dump(mode="json"),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert {item["label"] for item in body["detected_objects"]} == {"normal", "defect"}
    assert {item["label"] for item in body["inspection_results"]} == {"normal", "defect"}


def test_inspect_detections_endpoint_preserves_fallback_detections() -> None:
    detected = DetectedObject(
        object_id="object_0",
        label="normal",
        pixel_center=(10, 10),
        world_position=(0.2, 0.1, 0.04),
        confidence=0.95,
    )

    response = TestClient(app).post(
        "/inspect-detections",
        json={"detected_objects": [detected.model_dump(mode="json")]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["detected_objects"][0]["object_id"] == "object_0"
    assert body["inspection_results"][0]["label"] == "normal"

