"""HTTP client for the FastAPI external inspection service."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import httpx
import numpy as np

from robot_sorting.schemas import (
    DetectedObject,
    DetectionInspectionRequest,
    ImageInspectionRequest,
    InspectionPipelineResponse,
    SimulationConfig,
)


class InspectionApiUnavailableError(RuntimeError):
    """Raised when the external inspection API cannot be reached."""


class ExternalInspectionApiClient:
    """Client used by the simulation container to call the inspection service."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def inspect_image(
        self,
        rgb_image: np.ndarray,
        config: SimulationConfig,
    ) -> InspectionPipelineResponse:
        """Send an RGB image to the external API for color classification and inspection."""

        payload = ImageInspectionRequest(
            image_base64=self._encode_png_base64(rgb_image),
            config=config,
        )
        return self._post("/inspect-image", payload.model_dump(mode="json"))

    def inspect_detections(
        self,
        detected_objects: list[DetectedObject],
    ) -> InspectionPipelineResponse:
        """Send fallback detections to the external API for inspection."""

        payload = DetectionInspectionRequest(detected_objects=detected_objects)
        return self._post("/inspect-detections", payload.model_dump(mode="json"))

    def _post(self, path: str, payload: dict[str, Any]) -> InspectionPipelineResponse:
        try:
            response = httpx.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InspectionApiUnavailableError(str(exc)) from exc
        return InspectionPipelineResponse.model_validate(response.json())

    @staticmethod
    def _encode_png_base64(rgb_image: np.ndarray) -> str:
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", bgr)
        if not ok:
            raise InspectionApiUnavailableError("Could not encode RGB image as PNG")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

