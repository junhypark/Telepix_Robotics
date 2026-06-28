"""HTTP client for the FastAPI external inspection service."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import cv2
import httpx
import numpy as np

from robot_sorting.schemas import (
    CameraCalibration,
    DetectedObject,
    DetectionInspectionRequest,
    ImageInspectionRequest,
    InspectionFallbackResult,
    InspectionPipelineResponse,
    RGBDFrame,
    RGBDInspectionResponse,
    SimulationConfig,
    WorkspaceBounds,
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
    ) -> InspectionFallbackResult:
        """Upload an RGB image to the external API for color classification and inspection."""

        encoded = self._encode_png_bytes(rgb_image)
        files = {"image": ("rgb.png", encoded, "image/png")}
        data = {"config_json": json.dumps(config.model_dump(mode="json"))}
        body = self._post_multipart("/inspect-image", files=files, data=data)
        detected_objects = [DetectedObject.model_validate(item) for item in body.get("detected_objects", [])]
        if not detected_objects:
            detected_objects = [
                DetectedObject(
                    object_id=item["object_id"],
                    label=item["label"],
                    pixel_center=tuple(item["pixel_center"]),
                    world_position=tuple(item["world_position"]),
                    confidence=item["confidence"],
                )
                for item in body.get("objects", [])
            ]
        response = InspectionPipelineResponse(
            detected_objects=detected_objects,
            inspection_results=[
                item
                for item in InspectionPipelineResponse.model_validate(
                    {
                        "detected_objects": detected_objects,
                        "inspection_results": body.get("inspection_results", []),
                    }
                ).inspection_results
            ],
        )
        return InspectionFallbackResult(
            detected_objects=response.detected_objects,
            inspection_results=response.inspection_results,
            used_image_fallback=True,
        )

    def inspect_image_json(
        self,
        rgb_image: np.ndarray,
        config: SimulationConfig,
    ) -> InspectionPipelineResponse:
        """Send legacy JSON image payload to the external API."""

        payload = ImageInspectionRequest(image_base64=self._encode_png_base64(rgb_image), config=config)
        return self._post_json("/inspect-image", payload.model_dump(mode="json"))

    def inspect_rgbd(
        self,
        frame: RGBDFrame,
        calibration: CameraCalibration,
        workspace: WorkspaceBounds,
    ) -> RGBDInspectionResponse:
        """Upload RGB-D data to the external API."""

        depth_buffer = BytesIO()
        np.save(depth_buffer, frame.depth.astype(np.float32))
        files = {
            "image": ("rgb.png", self._encode_png_bytes(frame.rgb), "image/png"),
            "depth": ("depth.npy", depth_buffer.getvalue(), "application/octet-stream"),
        }
        data = {
            "calibration_json": json.dumps(calibration.model_dump(mode="json")),
            "workspace_json": json.dumps(workspace.model_dump(mode="json")),
        }
        body = self._post_multipart("/inspect-rgbd", files=files, data=data)
        return RGBDInspectionResponse.model_validate(body)

    def inspect_detections(
        self,
        detected_objects: list[DetectedObject],
    ) -> InspectionPipelineResponse:
        """Send fallback detections to the external API for inspection."""

        payload = DetectionInspectionRequest(detected_objects=detected_objects)
        return self._post("/inspect-detections", payload.model_dump(mode="json"))

    def _post(self, path: str, payload: dict[str, Any]) -> InspectionPipelineResponse:
        return self._post_json(path, payload)

    def _post_json(self, path: str, payload: dict[str, Any]) -> InspectionPipelineResponse:
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

    def _post_multipart(
        self,
        path: str,
        *,
        files: dict[str, tuple[str, bytes, str]],
        data: dict[str, str],
    ) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{self.base_url}{path}",
                files=files,
                data=data,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InspectionApiUnavailableError(str(exc)) from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise InspectionApiUnavailableError("Inspection API returned a non-object response")
        return payload

    @staticmethod
    def _encode_png_base64(rgb_image: np.ndarray) -> str:
        return base64.b64encode(ExternalInspectionApiClient._encode_png_bytes(rgb_image)).decode("ascii")

    @staticmethod
    def _encode_png_bytes(rgb_image: np.ndarray) -> bytes:
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", bgr)
        if not ok:
            raise InspectionApiUnavailableError("Could not encode RGB image as PNG")
        return encoded.tobytes()
