"""FastAPI app exposing external color classification and inspection."""

from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException

from robot_sorting.modules.inspection_module import InspectionModule
from robot_sorting.modules.vision_module import VisionModule
from robot_sorting.schemas import (
    DetectionInspectionRequest,
    ImageInspectionRequest,
    InspectionPipelineResponse,
)

app = FastAPI(
    title="Robot Sorting External Inspection API",
    version="0.1.0",
    description="Color classification and defect inspection service for the MuJoCo sorting sim.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return service health for Docker Compose health checks."""

    return {"status": "ok"}


@app.post("/inspect-image")
def inspect_image(request: ImageInspectionRequest) -> InspectionPipelineResponse:
    """Detect colored objects from an RGB image and inspect them."""

    image = _decode_rgb_image(request.image_base64)
    detected_objects = VisionModule(request.config).detect(image)
    inspection_results = InspectionModule().inspect_all(detected_objects)
    return InspectionPipelineResponse(
        detected_objects=detected_objects,
        inspection_results=inspection_results,
    )


@app.post("/inspect-detections")
def inspect_detections(request: DetectionInspectionRequest) -> InspectionPipelineResponse:
    """Inspect detections supplied by a simulation fallback path."""

    inspection_results = InspectionModule().inspect_all(request.detected_objects)
    return InspectionPipelineResponse(
        detected_objects=request.detected_objects,
        inspection_results=inspection_results,
    )


def _decode_rgb_image(image_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc

    encoded = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Image payload could not be decoded")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

