"""API user-input error handling tests."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from robot_sorting.api import app
from robot_sorting.perception.camera_calibration import build_top_down_workspace_calibration
from robot_sorting.schemas import WorkspaceBounds

pytestmark = pytest.mark.api


def _valid_image_bytes() -> bytes:
    image = np.full((80, 100, 3), 190, dtype=np.uint8)
    cv2.circle(image, (50, 40), 12, (0, 0, 255), thickness=-1)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    return encoded.tobytes()


def _valid_depth_bytes() -> bytes:
    depth = np.ones((80, 100), dtype=np.float32) * 0.95
    buffer = BytesIO()
    np.save(buffer, depth)
    return buffer.getvalue()


def _valid_form() -> dict[str, str]:
    workspace = WorkspaceBounds()
    calibration = build_top_down_workspace_calibration(100, 80, workspace)
    return {"calibration_json": calibration.model_dump_json(), "workspace_json": workspace.model_dump_json()}


@pytest.mark.parametrize(
    "files,data",
    [
        ({"image": ("bad.png", b"bad", "image/png")}, {}),
        ({}, {}),
        (
            {"image": ("rgb.png", _valid_image_bytes(), "image/png")},
            _valid_form(),
        ),
        (
            {
                "image": ("rgb.png", _valid_image_bytes(), "image/png"),
                "depth": ("depth.npy", b"bad", "application/octet-stream"),
            },
            _valid_form(),
        ),
        (
            {
                "image": ("rgb.png", _valid_image_bytes(), "image/png"),
                "depth": ("depth.npy", _valid_depth_bytes(), "application/octet-stream"),
            },
            {"calibration_json": "{bad", "workspace_json": WorkspaceBounds().model_dump_json()},
        ),
        (
            {
                "image": ("rgb.png", _valid_image_bytes(), "image/png"),
                "depth": ("depth.npy", _valid_depth_bytes(), "application/octet-stream"),
            },
            {
                "calibration_json": build_top_down_workspace_calibration(
                    100,
                    80,
                    WorkspaceBounds(),
                ).model_dump_json(),
                "workspace_json": "{bad",
            },
        ),
    ],
)
def test_api_user_input_errors_are_controlled(files, data) -> None:
    path = "/inspect-image" if "depth" not in files and data == {} else "/inspect-rgbd"

    response = TestClient(app).post(path, files=files, data=data)

    assert response.status_code in {400, 422}
    assert "detail" in response.json()
    assert response.status_code != 500

