"""Camera calibration helpers for RGB-D perception."""

from __future__ import annotations

import math

import numpy as np

from robot_sorting.schemas import CameraCalibration, CameraExtrinsics, CameraIntrinsics, WorkspaceBounds


def pixel_to_camera_point(
    pixel: tuple[int, int],
    depth: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float, float]:
    """Project a pixel and depth into camera coordinates using a pinhole model."""

    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError("Depth must be a positive finite value")
    u, v = pixel
    x_camera = (u - intrinsics.cx) * depth / intrinsics.fx
    y_camera = (v - intrinsics.cy) * depth / intrinsics.fy
    return (float(x_camera), float(y_camera), float(depth))


def camera_to_world_point(
    camera_point: tuple[float, float, float],
    extrinsics: CameraExtrinsics,
) -> tuple[float, float, float]:
    """Transform a camera-frame point into world coordinates."""

    rotation = np.asarray(extrinsics.rotation_world_from_camera, dtype=float)
    translation = np.asarray(extrinsics.translation_world_from_camera, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("Camera rotation must be a 3x3 matrix")
    point = np.asarray(camera_point, dtype=float)
    world = rotation @ point + translation
    if not np.all(np.isfinite(world)):
        raise ValueError("World point contains non-finite values")
    return (float(world[0]), float(world[1]), float(world[2]))


def pixel_depth_to_world_point(
    pixel: tuple[int, int],
    depth: float,
    calibration: CameraCalibration,
) -> tuple[float, float, float]:
    """Project a pixel-depth observation directly into world coordinates."""

    camera_point = pixel_to_camera_point(pixel, depth, calibration.intrinsics)
    return camera_to_world_point(camera_point, calibration.extrinsics)


def build_top_down_workspace_calibration(
    width: int,
    height: int,
    workspace: WorkspaceBounds,
    *,
    camera_z: float = 1.0,
) -> CameraCalibration:
    """Build a deterministic top-down calibration for tests and fallback RGB-D frames."""

    fx = width / max(workspace.x_max - workspace.x_min, 1e-9)
    fy = height / max(workspace.y_max - workspace.y_min, 1e-9)
    cx = -workspace.x_min * fx
    cy = workspace.y_max * fy
    return CameraCalibration(
        intrinsics=CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height),
        extrinsics=CameraExtrinsics(
            rotation_world_from_camera=[
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            translation_world_from_camera=(0.0, 0.0, camera_z),
        ),
    )

