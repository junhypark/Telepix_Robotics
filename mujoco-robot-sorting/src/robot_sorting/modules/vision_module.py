"""External OpenCV-based color vision module."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from robot_sorting.robot.safety import is_inside_workspace
from robot_sorting.schemas import DetectedObject, ObjectLabel, SimulationConfig, WorkspaceBounds

LOGGER = logging.getLogger(__name__)


class VisionModule:
    """Detect red defect objects and blue normal objects from RGB images."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def detect(self, rgb_image: np.ndarray) -> list[DetectedObject]:
        """Detect colored objects in an RGB image without depending on robot control."""

        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            LOGGER.warning("Vision input is not an RGB image")
            return []

        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        blue_mask = self._mask_blue(hsv)
        red_mask = self._mask_red(hsv)
        detections = [
            *self._detections_from_mask(blue_mask, "normal", rgb_image.shape),
            *self._detections_from_mask(red_mask, "defect", rgb_image.shape),
        ]
        return sorted(detections, key=lambda item: (item.world_position[0], item.world_position[1], item.label))

    def pixel_to_world(
        self,
        pixel_center: tuple[int, int],
        image_shape: tuple[int, ...],
        workspace: WorkspaceBounds | None = None,
    ) -> tuple[float, float, float]:
        """Convert an image pixel to an approximate top-down world coordinate."""

        bounds = workspace or self.config.workspace
        height, width = image_shape[:2]
        px, py = pixel_center
        x_norm = px / max(width - 1, 1)
        y_norm = py / max(height - 1, 1)
        x = bounds.x_min + x_norm * (bounds.x_max - bounds.x_min)
        y = bounds.y_max - y_norm * (bounds.y_max - bounds.y_min)
        z = bounds.z_min
        world = (float(x), float(y), float(z))
        if not is_inside_workspace(world, bounds):
            return (
                float(np.clip(world[0], bounds.x_min, bounds.x_max)),
                float(np.clip(world[1], bounds.y_min, bounds.y_max)),
                float(np.clip(world[2], bounds.z_min, bounds.z_max)),
            )
        return world

    def _mask_blue(self, hsv: np.ndarray) -> np.ndarray:
        lower = np.array([95, self.config.low_saturation_threshold, 45], dtype=np.uint8)
        upper = np.array([130, 255, 255], dtype=np.uint8)
        return self._clean_mask(cv2.inRange(hsv, lower, upper))

    def _mask_red(self, hsv: np.ndarray) -> np.ndarray:
        lower_one = np.array([0, self.config.low_saturation_threshold, 45], dtype=np.uint8)
        upper_one = np.array([10, 255, 255], dtype=np.uint8)
        lower_two = np.array([170, self.config.low_saturation_threshold, 45], dtype=np.uint8)
        upper_two = np.array([179, 255, 255], dtype=np.uint8)
        first_red_range = cv2.inRange(hsv, lower_one, upper_one)
        second_red_range = cv2.inRange(hsv, lower_two, upper_two)
        return self._clean_mask(cv2.bitwise_or(first_red_range, second_red_range))

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), dtype=np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    def _detections_from_mask(
        self,
        mask: np.ndarray,
        label: ObjectLabel,
        image_shape: tuple[int, ...],
    ) -> list[DetectedObject]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[DetectedObject] = []
        sorted_contours = sorted(contours, key=self._contour_sort_key)
        for index, contour in enumerate(sorted_contours):
            area = float(cv2.contourArea(contour))
            if area < self.config.min_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center = (
                round(moments["m10"] / moments["m00"]),
                round(moments["m01"] / moments["m00"]),
            )
            world = self.pixel_to_world(center, image_shape)
            confidence = self._confidence(area, contour)
            if confidence < self.config.min_confidence:
                LOGGER.debug("Detected low-confidence %s candidate at %s", label, center)
            detections.append(
                DetectedObject(
                    object_id=f"{label}_{index}",
                    label=label,
                    pixel_center=center,
                    world_position=world,
                    confidence=confidence,
                )
            )
        return detections

    def _confidence(self, area: float, contour: np.ndarray) -> float:
        x, y, width, height = cv2.boundingRect(contour)
        del x, y
        fill_ratio = area / max(float(width * height), 1.0)
        area_score = min(1.0, area / max(self.config.min_area * 5.0, 1.0))
        confidence = 0.55 + 0.25 * area_score + 0.20 * min(fill_ratio, 1.0)
        return float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _contour_sort_key(contour: np.ndarray) -> tuple[int, int]:
        x, y, _, _ = cv2.boundingRect(contour)
        return (x, y)
