"""Shared data schemas for the robot sorting simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

ObjectLabel = Literal["normal", "defect"]
TargetBin = Literal["normal_bin", "defect_bin"]
ExecutionStatus = Literal["completed", "failed"]
PlacementStrategy = Literal["center_free_slot", "grid_free_slot", "depth_lowest_free_region", "fallback_center"]


class StrictBaseModel(BaseModel):
    """Base model with production-friendly validation defaults."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)


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


class TableSafetyConfig(StrictBaseModel):
    """Minimum clearance constraints for robot links above the table."""

    table_top_z: float = 0.0
    min_link_clearance_meters: float = 0.03
    min_end_effector_clearance_meters: float = 0.02
    vertical_escape_height: float = 0.18


class BinPlacementConfig(StrictBaseModel):
    """Grid and margin settings for non-overlapping bin placement."""

    bin_wall_margin_meters: float = 0.02
    object_spacing_margin_meters: float = 0.015
    grid_resolution_meters: float = 0.03


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
    bin_size_xyz: tuple[float, float, float] = (0.22, 0.18, 0.04)
    table_safety: TableSafetyConfig = Field(default_factory=TableSafetyConfig)
    bin_placement: BinPlacementConfig = Field(default_factory=BinPlacementConfig)
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


class DetectedBin(StrictBaseModel):
    """Target bin detected by the external vision module."""

    bin_id: str
    label: TargetBin
    pixel_center: tuple[int, int]
    world_position: tuple[float, float, float]
    orientation_rpy: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
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
    target_bin_id: str | None = None
    placement_strategy: PlacementStrategy | None = None
    placement_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    inspection_at: float | None = None


class RobotCommand(StrictBaseModel):
    """Command queued for the robot controller."""

    object_id: str
    task: PickPlaceTask
    queued_at: float
    command_type: Literal["pick_and_place"] = "pick_and_place"
    inspection_at: float | None = None
    command_latency_seconds: float = Field(default=0.0, ge=0.0)
    trajectory: PlannedTrajectory | None = None
    object_pose: ObjectPose3D | None = None
    grasp_pose: GraspPose | None = None


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
    target_bin_id: str | None = None
    task_placement_strategy: PlacementStrategy | None = None
    task_placement_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: ExecutionStatus
    failure_reason: str = ""
    command_latency_seconds: float = Field(default=0.0, ge=0.0)
    self_collision_checked: bool = False
    workspace_checked: bool = False
    object_pose: ObjectPose3D | None = None
    grasp_pose: GraspPose | None = None
    trajectory_safe: bool | None = None
    collision_checked: bool = False
    table_clearance_checked: bool = False
    min_observed_link_z: float | None = None
    min_required_link_z: float | None = None


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
    rgbd_used: bool = False
    depth_fallback_used: bool = False
    ground_truth_fallback_used: bool = False
    pose_estimation_success_count: int = 0
    pose_estimation_failure_count: int = 0
    grasp_generation_success_count: int = 0
    grasp_generation_failure_count: int = 0
    trajectory_collision_failures: int = 0
    table_penetration_failures: int = 0
    min_observed_link_z: float = 0.0
    min_required_link_z: float = 0.0


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
    detected_bins: list[DetectedBin] = Field(default_factory=list)


class RGBDFrame(StrictBaseModel):
    """RGB-D frame captured from a MuJoCo camera."""

    rgb: np.ndarray
    depth: np.ndarray
    width: int
    height: int
    camera_name: str
    captured_at: float


class CameraIntrinsics(StrictBaseModel):
    """Pinhole camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class CameraExtrinsics(StrictBaseModel):
    """Camera-to-world transform."""

    rotation_world_from_camera: list[list[float]]
    translation_world_from_camera: tuple[float, float, float]


class CameraCalibration(StrictBaseModel):
    """Full camera calibration for pixel-depth projection."""

    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics


class ObjectPose3D(StrictBaseModel):
    """Estimated 3D object pose and size."""

    object_id: str
    label: ObjectLabel
    position: tuple[float, float, float]
    orientation_rpy: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)


class GeneratedObject(StrictBaseModel):
    """Seed-generated product used by scenario builders."""

    object_id: str
    label: ObjectLabel
    target_spawn_position: tuple[float, float, float]
    actual_spawn_position: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    orientation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


class GeneratedBin(StrictBaseModel):
    """Seed-generated target bin specification."""

    bin_id: str
    label: TargetBin
    color_rgb: tuple[int, int, int]
    target_spawn_position: tuple[float, float, float]
    actual_spawn_position: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    orientation_rpy: tuple[float, float, float]


class ObstacleSpec(StrictBaseModel):
    """Optional obstacle reserved for scenario generation extensions."""

    obstacle_id: str
    position: tuple[float, float, float]
    size_xyz: tuple[float, float, float]


class GeneratedScenario(StrictBaseModel):
    """Complete deterministic scenario specification."""

    seed: int
    objects: list[GeneratedObject]
    bins: list[GeneratedBin]
    obstacles: list[ObstacleSpec] = Field(default_factory=list)
    normal_count: int
    defect_count: int


class GraspPose(StrictBaseModel):
    """Top-down grasp pose for a table-top object."""

    object_id: str
    pre_grasp_position: tuple[float, float, float]
    grasp_position: tuple[float, float, float]
    retreat_position: tuple[float, float, float]
    approach_vector: tuple[float, float, float]
    gripper_yaw: float
    confidence: float = Field(ge=0.0, le=1.0)


class TrajectoryWaypoint(StrictBaseModel):
    """One collision-checked trajectory waypoint."""

    position: tuple[float, float, float]
    gripper_state: Literal["open", "closed"]
    duration_seconds: float = Field(gt=0.0)


class PlannedTrajectory(StrictBaseModel):
    """Collision-aware pick-and-place waypoint sequence."""

    object_id: str
    waypoints: list[TrajectoryWaypoint]
    is_safe: bool
    failure_reason: str | None = None


class RGBDInspectionObject(StrictBaseModel):
    """Object returned by the RGB-D inspection endpoint."""

    object_id: str
    label: ObjectLabel
    position: tuple[float, float, float]
    orientation_rpy: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    grasp_pose: GraspPose


class BinPlacementDecision(StrictBaseModel):
    """Vision-derived placement target inside a detected bin."""

    object_id: str
    target_bin_id: str
    target_bin: TargetBin
    placement_position: tuple[float, float, float]
    placement_pixel: tuple[int, int] | None = None
    placement_strategy: PlacementStrategy
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    failure_reason: str | None = None


class PlacedObjectRecord(StrictBaseModel):
    """Placed product state used to avoid overlapping future placements."""

    object_id: str
    target_bin_id: str
    position: tuple[float, float, float]
    size_xyz: tuple[float, float, float]


class BinInteriorEstimate(StrictBaseModel):
    """Estimated free-space candidates inside a detected target bin."""

    bin_id: str
    bin_label: TargetBin
    inner_polygon_pixels: list[tuple[int, int]]
    candidate_place_pixels: list[tuple[int, int]]
    candidate_place_positions: list[tuple[float, float, float]]
    occupied_regions: list[list[tuple[int, int]]]
    confidence: float = Field(ge=0.0, le=1.0)


class RGBDInspectionResponse(StrictBaseModel):
    """Response for advanced RGB-D inspection."""

    objects: list[RGBDInspectionObject]
    bins: list[DetectedBin] = Field(default_factory=list)
    processing_time_seconds: float = Field(ge=0.0)


class ImageUploadInspectionObject(StrictBaseModel):
    """Object returned by the multipart image inspection endpoint."""

    object_id: str
    label: ObjectLabel
    pixel_center: tuple[int, int]
    world_position: tuple[float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)


class ImageUploadInspectionResponse(StrictBaseModel):
    """Response for multipart RGB image inspection."""

    objects: list[ImageUploadInspectionObject]
    bins: list[DetectedBin] = Field(default_factory=list)
    processing_time_seconds: float = Field(ge=0.0)


class InspectionFallbackResult(StrictBaseModel):
    """Simulation-side inspection result with fallback metadata."""

    detected_objects: list[DetectedObject]
    inspection_results: list[ExternalInspectionResult]
    rgbd_objects: list[RGBDInspectionObject] = Field(default_factory=list)
    detected_bins: list[DetectedBin] = Field(default_factory=list)
    used_rgbd: bool = False
    used_image_fallback: bool = False
    used_ground_truth_fallback: bool = False
    depth_fallback_used: bool = False
