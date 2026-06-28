"""MuJoCo camera renderer with graceful headless fallback."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from robot_sorting.perception.camera_calibration import build_top_down_workspace_calibration
from robot_sorting.schemas import RGBDFrame
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv

LOGGER = logging.getLogger(__name__)


class RendererUnavailableError(RuntimeError):
    """Raised when MuJoCo rendering cannot be initialized."""


class MujocoRenderer:
    """Render RGB and depth frames from the MuJoCo top-down camera."""

    def __init__(self, env: MujocoSortingEnv) -> None:
        self.env = env

    def render_rgb(self) -> np.ndarray | None:
        """Render one RGB image, or return None when rendering is unavailable."""

        try:
            import mujoco

            renderer = mujoco.Renderer(
                self.env.model,
                height=self.env.config.height,
                width=self.env.config.width,
            )
            renderer.update_scene(self.env.data, camera=self.env.config.renderer_camera)
            image = renderer.render()
            renderer.close()
            return np.asarray(image, dtype=np.uint8)
        except Exception as exc:
            LOGGER.warning(
                "Renderer unavailable. Falling back to simulation ground-truth object positions.",
                exc_info=exc,
            )
            return None

    def render_depth(self) -> np.ndarray | None:
        """Render one depth image, or return None when depth rendering is unavailable."""

        try:
            import mujoco

            renderer = mujoco.Renderer(
                self.env.model,
                height=self.env.config.height,
                width=self.env.config.width,
            )
            renderer.enable_depth_rendering()
            renderer.update_scene(self.env.data, camera=self.env.config.renderer_camera)
            depth = renderer.render()
            renderer.close()
            return np.asarray(depth, dtype=np.float32)
        except Exception as exc:
            LOGGER.warning(
                "Depth rendering unavailable. Falling back to MuJoCo ground-truth object poses.",
                exc_info=exc,
            )
            return None

    def render_rgbd(self) -> RGBDFrame | None:
        """Render RGB-D, returning None when RGB rendering is unavailable."""

        rgb = self.render_rgb()
        if rgb is None:
            return None
        depth = self.render_depth()
        if depth is None:
            return self.render_ground_truth_rgbd(rgb=rgb)
        return RGBDFrame(
            rgb=rgb,
            depth=depth,
            width=self.env.config.width,
            height=self.env.config.height,
            camera_name=self.env.config.renderer_camera,
            captured_at=time.time(),
        )

    def render_ground_truth_rgbd(self, rgb: np.ndarray | None = None) -> RGBDFrame:
        """Create a deterministic RGB-D frame from MuJoCo ground-truth object poses."""

        config = self.env.config
        image = (
            np.full((config.height, config.width, 3), 190, dtype=np.uint8)
            if rgb is None
            else np.asarray(rgb, dtype=np.uint8).copy()
        )
        calibration = build_top_down_workspace_calibration(config.width, config.height, config.workspace)
        camera_z = calibration.extrinsics.translation_world_from_camera[2]
        depth = np.full((config.height, config.width), camera_z - config.table_height, dtype=np.float32)
        for obj in self.env.object_specs:
            px, py = self.env.world_to_pixel(obj.position)
            color = (0, 0, 255) if obj.label == "normal" else (255, 0, 0)
            half_size = max(3, int(config.object_radius * config.width * 0.7))
            top_left = (max(0, px - half_size), max(0, py - half_size))
            bottom_right = (min(config.width - 1, px + half_size), min(config.height - 1, py + half_size))
            cv2.rectangle(image, top_left, bottom_right, color, thickness=-1)
            depth[top_left[1] : bottom_right[1] + 1, top_left[0] : bottom_right[0] + 1] = camera_z - obj.position[2]
        return RGBDFrame(
            rgb=image,
            depth=depth,
            width=config.width,
            height=config.height,
            camera_name=config.renderer_camera,
            captured_at=time.time(),
        )

    def save_rgb(self, image: np.ndarray, output_path: Path) -> None:
        """Save an RGB image as a PNG file."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)

    def save_depth(self, depth: np.ndarray, output_path: Path) -> None:
        """Save a depth array as .npy."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, depth)


MujocoCameraRenderer = MujocoRenderer
