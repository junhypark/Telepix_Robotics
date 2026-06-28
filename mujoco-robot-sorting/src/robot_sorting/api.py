"""FastAPI app exposing external color classification and inspection."""

from __future__ import annotations

import base64
import binascii
import json
import time
from io import BytesIO
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from robot_sorting.config import create_simulation_config
from robot_sorting.modules.inspection_module import InspectionModule
from robot_sorting.modules.vision_module import VisionModule
from robot_sorting.perception.grasp_pose_generator import generate_top_down_grasp_pose
from robot_sorting.perception.pose_estimator import estimate_object_pose_3d
from robot_sorting.schemas import (
    CameraCalibration,
    DetectionInspectionRequest,
    ImageInspectionRequest,
    ImageUploadInspectionObject,
    InspectionPipelineResponse,
    RGBDInspectionObject,
    RGBDInspectionResponse,
    SimulationConfig,
    WorkspaceBounds,
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
async def inspect_image(
    request: Request,
    image: Annotated[UploadFile | None, File()] = None,
    config_json: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    """Detect colored objects from JSON or multipart RGB image input."""

    started = time.perf_counter()
    config: SimulationConfig
    if _is_json_request(request):
        payload = ImageInspectionRequest.model_validate(await request.json())
        rgb = _decode_rgb_image(payload.image_base64)
        config = payload.config
    else:
        if image is None:
            raise HTTPException(status_code=400, detail="Missing RGB image upload")
        rgb = _decode_uploaded_rgb_image(await image.read())
        config = _parse_config_json(config_json, rgb.shape)

    detected_objects = VisionModule(config).detect(rgb)
    inspection_results = InspectionModule().inspect_all(detected_objects)
    objects = [
        ImageUploadInspectionObject(
            object_id=item.object_id,
            label=item.label,
            pixel_center=item.pixel_center,
            world_position=item.world_position,
            confidence=item.confidence,
        )
        for item in detected_objects
    ]
    return {
        "objects": [item.model_dump(mode="json") for item in objects],
        "detected_objects": [item.model_dump(mode="json") for item in detected_objects],
        "inspection_results": [item.model_dump(mode="json") for item in inspection_results],
        "processing_time_seconds": time.perf_counter() - started,
    }


@app.post("/inspect-rgbd")
async def inspect_rgbd(
    image: Annotated[UploadFile, File()],
    depth: Annotated[UploadFile, File()],
    calibration_json: Annotated[str, Form()],
    workspace_json: Annotated[str, Form()],
) -> RGBDInspectionResponse:
    """Estimate 3D object poses and grasp poses from multipart RGB-D input."""

    started = time.perf_counter()
    rgb = _decode_uploaded_rgb_image(await image.read())
    depth_array = _decode_depth_npy(await depth.read())
    if depth_array.shape[:2] != rgb.shape[:2]:
        raise HTTPException(status_code=400, detail="RGB and depth dimensions must match")
    calibration = _parse_json_model(calibration_json, CameraCalibration, "Invalid calibration JSON")
    workspace = _parse_json_model(workspace_json, WorkspaceBounds, "Invalid workspace JSON")
    config = create_simulation_config(width=rgb.shape[1], height=rgb.shape[0])
    config.workspace = workspace

    objects: list[RGBDInspectionObject] = []
    masked_detections = VisionModule(config).detect_with_masks(rgb)
    for masked in masked_detections:
        pose = estimate_object_pose_3d(masked.detected_object, masked.mask, depth_array, calibration)
        if pose is None:
            continue
        grasp_pose = generate_top_down_grasp_pose(
            pose,
            grasp_height_offset=0.005,
            pre_grasp_z_offset=0.11,
            retreat_z_offset=0.13,
            workspace=workspace,
        )
        if grasp_pose is None:
            continue
        objects.append(
            RGBDInspectionObject(
                object_id=pose.object_id,
                label=pose.label,
                position=pose.position,
                orientation_rpy=pose.orientation_rpy,
                size_xyz=pose.size_xyz,
                confidence=pose.confidence,
                grasp_pose=grasp_pose,
            )
        )
    return RGBDInspectionResponse(objects=objects, processing_time_seconds=time.perf_counter() - started)


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


def _decode_uploaded_rgb_image(raw: bytes) -> np.ndarray:
    encoded = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Image upload could not be decoded")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_depth_npy(raw: bytes) -> np.ndarray:
    try:
        depth = np.load(BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Depth upload must be a valid .npy array") from exc
    if depth.ndim != 2:
        raise HTTPException(status_code=400, detail="Depth array must be two-dimensional")
    return np.asarray(depth, dtype=np.float32)


def _parse_config_json(config_json: str | None, image_shape: tuple[int, ...]) -> SimulationConfig:
    if config_json is None:
        return create_simulation_config(width=image_shape[1], height=image_shape[0])
    return _parse_json_model(config_json, SimulationConfig, "Invalid config JSON")


def _parse_json_model[ModelT: BaseModel](
    raw: str,
    model_type: type[ModelT],
    error_detail: str,
) -> ModelT:
    try:
        payload = json.loads(raw)
        return model_type.model_validate(payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=error_detail) from exc


def _is_json_request(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    return "application/json" in content_type.lower()
