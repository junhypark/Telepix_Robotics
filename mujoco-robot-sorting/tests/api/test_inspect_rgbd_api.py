"""Multipart RGB-D upload API tests."""

from __future__ import annotations

import math
from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from robot_sorting.api import app
from robot_sorting.perception.camera_calibration import build_top_down_workspace_calibration
from robot_sorting.schemas import WorkspaceBounds

pytestmark = pytest.mark.api


def _rgb_depth_payload() -> tuple[bytes, bytes, str, str]:
    workspace = WorkspaceBounds()
    image = np.full((160, 220, 3), 190, dtype=np.uint8)
    cv2.circle(image, (150, 80), 20, (0, 0, 255), thickness=-1)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    depth = np.zeros((160, 220), dtype=np.float32)
    cv2.circle(depth, (150, 80), 20, 0.95, thickness=-1)
    buffer = BytesIO()
    np.save(buffer, depth)
    calibration = build_top_down_workspace_calibration(220, 160, workspace)
    return encoded.tobytes(), buffer.getvalue(), calibration.model_dump_json(), workspace.model_dump_json()


def test_inspect_rgbd_upload_returns_pose_and_grasp() -> None:
    image, depth, calibration_json, workspace_json = _rgb_depth_payload()
    response = TestClient(app).post(
        "/inspect-rgbd",
        files={
            "image": ("rgb.png", image, "image/png"),
            "depth": ("depth.npy", depth, "application/octet-stream"),
        },
        data={"calibration_json": calibration_json, "workspace_json": workspace_json},
    )

    payload = response.json()
    assert response.status_code == 200
    assert len(payload["objects"]) >= 1
    obj = payload["objects"][0]
    assert all(math.isfinite(v) for v in obj["position"])
    assert all(math.isfinite(v) for v in obj["orientation_rpy"])
    assert all(math.isfinite(v) for v in obj["size_xyz"])
    grasp = obj["grasp_pose"]
    assert grasp["pre_grasp_position"][2] > grasp["grasp_position"][2]
    assert grasp["retreat_position"][2] > grasp["grasp_position"][2]
    assert payload["processing_time_seconds"] >= 0.0


def test_invalid_depth_file_returns_400() -> None:
    image, _, calibration_json, workspace_json = _rgb_depth_payload()
    response = TestClient(app).post(
        "/inspect-rgbd",
        files={
            "image": ("rgb.png", image, "image/png"),
            "depth": ("depth.npy", b"bad depth", "application/octet-stream"),
        },
        data={"calibration_json": calibration_json, "workspace_json": workspace_json},
    )

    assert response.status_code == 400
    assert "detail" in response.json()


def test_mismatched_rgb_depth_size_returns_400() -> None:
    image, _, calibration_json, workspace_json = _rgb_depth_payload()
    buffer = BytesIO()
    np.save(buffer, np.ones((10, 10), dtype=np.float32))
    response = TestClient(app).post(
        "/inspect-rgbd",
        files={
            "image": ("rgb.png", image, "image/png"),
            "depth": ("depth.npy", buffer.getvalue(), "application/octet-stream"),
        },
        data={"calibration_json": calibration_json, "workspace_json": workspace_json},
    )

    assert response.status_code == 400
    assert "detail" in response.json()


def test_missing_form_fields_return_controlled_error() -> None:
    image, depth, _, _ = _rgb_depth_payload()
    response = TestClient(app).post(
        "/inspect-rgbd",
        files={
            "image": ("rgb.png", image, "image/png"),
            "depth": ("depth.npy", depth, "application/octet-stream"),
        },
    )

    assert response.status_code in {400, 422}
    assert "detail" in response.json()
