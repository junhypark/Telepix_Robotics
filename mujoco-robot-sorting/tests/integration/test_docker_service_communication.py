"""Lightweight Docker service communication checks."""

from __future__ import annotations

import os
from io import BytesIO

import cv2
import httpx
import numpy as np
import pytest

from robot_sorting.perception.camera_calibration import build_top_down_workspace_calibration
from robot_sorting.schemas import WorkspaceBounds

pytestmark = pytest.mark.integration


def test_docker_service_communication_when_api_url_is_available() -> None:
    api_url = os.getenv("INSPECTION_API_URL")
    if not api_url:
        pytest.skip("INSPECTION_API_URL is not set for Docker service communication test")

    image, depth, calibration_json, workspace_json = _payload()
    health_response = httpx.get(f"{api_url}/health", timeout=5.0)
    inspect_response = httpx.post(
        f"{api_url}/inspect-rgbd",
        files={
            "image": ("rgb.png", image, "image/png"),
            "depth": ("depth.npy", depth, "application/octet-stream"),
        },
        data={"calibration_json": calibration_json, "workspace_json": workspace_json},
        timeout=5.0,
    )

    assert health_response.status_code == 200
    assert inspect_response.status_code == 200
    assert "objects" in inspect_response.json()


def _payload() -> tuple[bytes, bytes, str, str]:
    workspace = WorkspaceBounds()
    image = np.full((100, 120, 3), 190, dtype=np.uint8)
    cv2.circle(image, (60, 50), 14, (0, 0, 255), thickness=-1)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    depth = np.zeros((100, 120), dtype=np.float32)
    cv2.circle(depth, (60, 50), 14, 0.95, thickness=-1)
    buffer = BytesIO()
    np.save(buffer, depth)
    calibration = build_top_down_workspace_calibration(120, 100, workspace)
    return encoded.tobytes(), buffer.getvalue(), calibration.model_dump_json(), workspace.model_dump_json()

