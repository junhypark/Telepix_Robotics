"""Shared data schemas for the robot sorting simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ObjectLabel = Literal["normal", "defect"]
TargetBin = Literal["normal_bin", "defect_bin"]
ExecutionStatus = Literal["completed", "failed"]


class StrictBaseModel(BaseModel):
    """Base model with production-friendly validation defaults."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class WorkspaceBounds(StrictBaseModel):
    """Axis-aligned workspace accepted by the planner and controller."""

    x_min: float = -0.45
    x_max: float = 0.58
    y_min: float = -0.40
    y_max: float = 0.40
    z_min: float = 0.035
    z_max: float = 0.55


class JointLimits(StrictBaseModel):
    """Robot joint limits in radians."""

    base_min: float = -3.141592653589793
    base_max: float = 3.141592653589793
    shoulder_min: float = -2.4
    shoulder_max: float = 2.4
    elbow_min: float = -2.6
    elbow_max: float = 2.6


class SimulationConfig(StrictBaseModel):
    """Top-level configuration for the simulation run."""

    headless: bool = True
    objects: int = Field(default=5, ge=0, le=50)
    seed: int = 42
    width: int = Field(default=640, ge=64)
    height: int = Field(default=480, ge=64)
    output_dir: Path = Path("outputs/run")
    save_images: bool = True
    workspace: WorkspaceBounds = Field(default_factory=WorkspaceBounds)
    joint_limits: JointLimits = Field(default_factory=JointLimits)
    base_radius: float = 0.11
    base_height: float = 0.12
    safety_margin: float = 0.025
    table_height: float = 0.02
    object_half_height: float = 0.015
    object_radius: float = 0.025
    link_1: float = 0.30
    link_2: float = 0.27
    approach_height: float = 0.11
    normal_bin_position: tuple[float, float, float] = (0.45, 0.28, 0.055)
    defect_bin_position: tuple[float, float, float] = (0.45, -0.28, 0.055)
    min_area: float = 80.0
    min_confidence: float = 0.55
    low_saturation_threshold: int = 75
    renderer_camera: str = "top_camera"
    control_steps_per_move: int = 20


class DetectedObject(StrictBaseModel):
    """Object detected by the external vision module."""

    object_id: str
    label: ObjectLabel
    pixel_center: tuple[int, int]
    world_position: tuple[float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)


class ExternalInspectionResult(StrictBaseModel):
    """Inspection result produced outside the robot controller."""

    object_id: str
    label: ObjectLabel
    world_position: tuple[float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    inspected_at: float


class PickPlaceTask(StrictBaseModel):
    """Planned pick-and-place task created from an inspection result."""

    object_id: str
    label: ObjectLabel
    pick_position: tuple[float, float, float]
    place_position: tuple[float, float, float]
    target_bin: TargetBin
    inspection_at: float | None = None


class RobotCommand(StrictBaseModel):
    """Command queued for the robot controller."""

    object_id: str
    task: PickPlaceTask
    queued_at: float
    command_type: Literal["pick_and_place"] = "pick_and_place"
    inspection_at: float | None = None
    command_latency_seconds: float = Field(default=0.0, ge=0.0)


class JointAngles(StrictBaseModel):
    """Yaw, shoulder, and elbow angles for the simplified arm."""

    base_yaw: float
    shoulder: float
    elbow: float


class IKResult(StrictBaseModel):
    """Result of analytical inverse kinematics."""

    angles: JointAngles
    status: Literal["reachable", "clamped", "invalid"]
    clamped_target: tuple[float, float, float]


class TaskExecutionResult(StrictBaseModel):
    """Status emitted after the controller attempts a command."""

    object_id: str
    label: ObjectLabel
    pick_position: tuple[float, float, float]
    place_position: tuple[float, float, float]
    target_bin: TargetBin
    status: ExecutionStatus
    failure_reason: str = ""
    command_latency_seconds: float = Field(default=0.0, ge=0.0)
    self_collision_checked: bool = False
    workspace_checked: bool = False


class RunSummary(StrictBaseModel):
    """Aggregate run statistics saved as JSON."""

    total_objects: int = 0
    normal_count: int = 0
    defect_count: int = 0
    placed_count: int = 0
    failed_count: int = 0
    success_rate: float = 0.0
    average_command_latency_seconds: float = 0.0
    max_command_latency_seconds: float = 0.0
    self_collision_failures: int = 0
    workspace_failures: int = 0
    vision_low_confidence_count: int = 0


class ImageInspectionRequest(StrictBaseModel):
    """HTTP request body for image-based external inspection."""

    image_base64: str
    config: SimulationConfig = Field(default_factory=SimulationConfig)


class DetectionInspectionRequest(StrictBaseModel):
    """HTTP request body for inspecting known detections."""

    detected_objects: list[DetectedObject]


class InspectionPipelineResponse(StrictBaseModel):
    """HTTP response body returned by the external inspection service."""

    detected_objects: list[DetectedObject]
    inspection_results: list[ExternalInspectionResult]

