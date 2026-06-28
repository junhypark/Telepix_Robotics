"""MuJoCo camera renderer with graceful headless fallback."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from robot_sorting.simulation.mujoco_env import MujocoSortingEnv

LOGGER = logging.getLogger(__name__)


class RendererUnavailableError(RuntimeError):
    """Raised when MuJoCo rendering cannot be initialized."""


class MujocoCameraRenderer:
    """Render RGB frames from the MuJoCo top-down camera."""

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

    def save_rgb(self, image: np.ndarray, output_path: Path) -> None:
        """Save an RGB image as a PNG file."""

        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)

