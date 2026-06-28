"""Typer CLI for the robot sorting simulation."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Annotated, Protocol

import numpy as np
import typer
from rich.console import Console

from robot_sorting.config import create_simulation_config
from robot_sorting.modules.command_queue import RobotCommandQueue, create_robot_commands
from robot_sorting.modules.inspection_client import (
    ExternalInspectionApiClient,
    InspectionApiUnavailableError,
)
from robot_sorting.modules.inspection_module import InspectionModule
from robot_sorting.modules.result_logger import ResultLogger
from robot_sorting.modules.task_planner import TaskPlanner
from robot_sorting.modules.vision_module import VisionModule
from robot_sorting.perception.camera_calibration import build_top_down_workspace_calibration
from robot_sorting.perception.grasp_pose_generator import generate_top_down_grasp_pose
from robot_sorting.perception.pose_estimator import estimate_object_pose_3d
from robot_sorting.planning.trajectory_planner import plan_pick_place_trajectory
from robot_sorting.robot.controller import RobotController
from robot_sorting.schemas import (
    CameraCalibration,
    DetectedObject,
    ExternalInspectionResult,
    GraspPose,
    ObjectPose3D,
    PickPlaceTask,
    PlannedTrajectory,
    RGBDFrame,
    RGBDInspectionObject,
    RobotCommand,
    SimulationConfig,
    TaskExecutionResult,
)
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv
from robot_sorting.simulation.renderer import MujocoRenderer

app = typer.Typer(help="MuJoCo robot arm sorting simulation")
console = Console()
DEFAULT_OUTPUT_DIR = Path("outputs/run")
DEFAULT_VIEWER_DELAY_SECONDS = 0.03


class ViewerHandle(Protocol):
    """Minimal MuJoCo passive viewer contract used by CLI playback."""

    def sync(self) -> None:
        """Synchronize the viewer with current MuJoCo model/data state."""

    def is_running(self) -> bool:
        """Return whether the viewer window is still open."""


@app.callback()
def _callback() -> None:
    """Robot sorting CLI."""


@app.command()
def run(
    headless: Annotated[bool, typer.Option("--headless/--viewer", help="Run without opening a viewer.")] = True,
    objects: Annotated[int, typer.Option("--objects", min=0, help="Number of objects to sort.")] = 5,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic scene seed.")] = 42,
    width: Annotated[int, typer.Option("--width", min=64, help="Render width.")] = 640,
    height: Annotated[int, typer.Option("--height", min=64, help="Render height.")] = 480,
    output: Annotated[Path, typer.Option("--output", help="Output directory.")] = DEFAULT_OUTPUT_DIR,
    save_images: Annotated[
        bool,
        typer.Option("--save-images/--no-save-images", help="Save camera image."),
    ] = True,
    inspection_api_url: Annotated[
        str | None,
        typer.Option(
            "--inspection-api-url",
            help="FastAPI external inspection service URL. Defaults to INSPECTION_API_URL.",
        ),
    ] = None,
    viewer_delay: Annotated[
        float,
        typer.Option("--viewer-delay", min=0.0, help="Frame delay in seconds when --viewer is used."),
    ] = DEFAULT_VIEWER_DELAY_SECONDS,
) -> None:
    """Run the full external-inspection sorting pipeline."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    config = create_simulation_config(
        headless=headless,
        objects=objects,
        seed=seed,
        width=width,
        height=height,
        output_dir=output,
        save_images=save_images,
    )

    env = MujocoSortingEnv(config)
    renderer = MujocoRenderer(env)
    rgbd_frame = renderer.render_rgbd()
    api_url = inspection_api_url or os.getenv("INSPECTION_API_URL")
    api_client = ExternalInspectionApiClient(api_url) if api_url else None
    calibration = build_top_down_workspace_calibration(config.width, config.height, config.workspace)
    rgbd_used = False
    ground_truth_fallback_used = False
    rgbd_objects: list[RGBDInspectionObject] = []

    if rgbd_frame is not None:
        if config.save_images:
            renderer.save_rgb(rgbd_frame.rgb, config.output_dir / "camera_rgb.png")
            renderer.save_depth(rgbd_frame.depth, config.output_dir / "camera_depth.npy")
        rgbd_objects = _inspect_rgbd(rgbd_frame, calibration, config, api_client)
        if config.objects > 0 and len(rgbd_objects) < config.objects:
            console.print("Depth rendering unavailable. Falling back to MuJoCo ground-truth object poses.")
            renderer.depth_fallback_used = True
            rgbd_frame = renderer.render_ground_truth_rgbd()
            if config.save_images:
                renderer.save_rgb(rgbd_frame.rgb, config.output_dir / "camera_rgb_fallback.png")
                renderer.save_depth(rgbd_frame.depth, config.output_dir / "camera_depth_fallback.npy")
            rgbd_objects = _inspect_rgbd(rgbd_frame, calibration, config, api_client)
        if rgbd_objects:
            rgbd_used = True
            detections, inspections = _detections_from_rgbd_objects(rgbd_objects, env)
        else:
            console.print("RGB-D inspection API failed. Falling back to image inspection.")
            detections, inspections, ground_truth_fallback_used = _inspect_image_or_ground_truth(
                rgbd_frame.rgb,
                config,
                env,
                api_client,
            )
    else:
        image = renderer.render_rgb()
        if image is None:
            console.print("Renderer unavailable. Falling back to simulation ground-truth object positions.")
            detections = env.get_ground_truth_detections()
            inspections = _fresh_inspections(detections)
            ground_truth_fallback_used = True
        else:
            if config.save_images:
                renderer.save_rgb(image, config.output_dir / "camera_rgb.png")
            detections, inspections, ground_truth_fallback_used = _inspect_image_or_ground_truth(
                image,
                config,
                env,
                api_client,
            )

    tasks = TaskPlanner(config).plan(inspections)
    object_pose_map = _object_pose_map(rgbd_objects)
    grasp_pose_map = _grasp_pose_map(rgbd_objects)
    trajectories = _plan_trajectories(tasks, grasp_pose_map, config)
    commands = create_robot_commands(
        tasks,
        trajectories=trajectories,
        object_poses=object_pose_map,
        grasp_poses=grasp_pose_map,
    )
    command_queue = RobotCommandQueue()
    command_queue.extend(commands)
    queued_commands = command_queue.pop_all()
    if config.headless:
        controller = RobotController(env, config)
        results = controller.execute_commands(queued_commands)
    else:
        results = _play_viewer_commands(env, config, queued_commands, viewer_delay)

    logger = ResultLogger(config.output_dir)
    summary = logger.write_all(
        detections,
        tasks,
        results,
        min_confidence=config.min_confidence,
        rgbd_used=rgbd_used,
        depth_fallback_used=renderer.depth_fallback_used,
        ground_truth_fallback_used=ground_truth_fallback_used,
    )
    console.print(f"Run complete. Outputs: {config.output_dir}")
    console.print(
        f"Placed {summary.placed_count}/{summary.total_objects}, "
        f"max command latency={summary.max_command_latency_seconds:.6f}s"
    )


@app.command()
def view(
    objects: Annotated[int, typer.Option("--objects", min=0, help="Number of objects to display.")] = 5,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic scene seed.")] = 42,
    animate: Annotated[
        bool,
        typer.Option("--animate/--static", help="Play the sorting trajectory instead of opening a static viewer."),
    ] = True,
    delay: Annotated[
        float,
        typer.Option("--delay", min=0.0, help="Frame delay in seconds during viewer playback."),
    ] = DEFAULT_VIEWER_DELAY_SECONDS,
) -> None:
    """Open the MuJoCo viewer for local visual inspection."""

    config = create_simulation_config(headless=False, objects=objects, seed=seed)
    env = MujocoSortingEnv(config)

    if not animate:
        _launch_static_viewer(env)
        return

    commands = _build_viewer_demo_commands(env, config)
    if not commands:
        console.print("No sorting commands were generated. Opening the scene for static inspection.")
        _launch_static_viewer(env)
        return
    _play_viewer_commands(env, config, commands, delay)


def _launch_static_viewer(env: MujocoSortingEnv) -> None:
    import mujoco.viewer

    mujoco.viewer.launch(env.model, env.data)


def _build_viewer_demo_commands(env: MujocoSortingEnv, config: SimulationConfig) -> list[RobotCommand]:
    frame = MujocoRenderer(env).render_ground_truth_rgbd()
    calibration = build_top_down_workspace_calibration(config.width, config.height, config.workspace)
    rgbd_objects = _inspect_rgbd(frame, calibration, config, api_client=None)
    if not rgbd_objects:
        return []

    _, inspections = _detections_from_rgbd_objects(rgbd_objects, env)
    tasks = TaskPlanner(config).plan(inspections)
    return create_robot_commands(
        tasks,
        trajectories=_plan_trajectories(tasks, _grasp_pose_map(rgbd_objects), config),
        object_poses=_object_pose_map(rgbd_objects),
        grasp_poses=_grasp_pose_map(rgbd_objects),
    )


def _play_viewer_commands(
    env: MujocoSortingEnv,
    config: SimulationConfig,
    commands: list[RobotCommand],
    frame_delay: float,
) -> list[TaskExecutionResult]:
    import mujoco.viewer

    results: list[TaskExecutionResult] = []
    viewer_context = mujoco.viewer.launch_passive(env.model, env.data)
    with viewer_context as viewer_handle:
        viewer: ViewerHandle = viewer_handle

        def sync_motion_step(_position: tuple[float, float, float]) -> None:
            if not _viewer_is_running(viewer):
                return
            viewer.sync()
            if frame_delay > 0:
                time.sleep(frame_delay)

        controller = RobotController(env, config, step_callback=sync_motion_step)
        viewer.sync()
        console.print("Viewer playback started. Close the MuJoCo viewer window to exit after playback.")
        for command in commands:
            if not _viewer_is_running(viewer):
                break
            result = controller.execute_command(command)
            results.append(result)
            console.print(f"{command.object_id}: {result.status}")
        viewer.sync()
        while _viewer_is_running(viewer):
            viewer.sync()
            time.sleep(max(frame_delay, 0.05))
    return results


def _viewer_is_running(viewer: ViewerHandle) -> bool:
    return viewer.is_running()


def _inspect_detections(
    detections: list[DetectedObject],
    api_client: ExternalInspectionApiClient | None,
) -> list[ExternalInspectionResult]:
    if api_client is None:
        return InspectionModule().inspect_all(detections)
    try:
        return api_client.inspect_detections(detections).inspection_results
    except InspectionApiUnavailableError as exc:
        console.print(f"Inspection API unavailable, using local modules: {exc}")
        return InspectionModule().inspect_all(detections)


def _inspect_rgbd(
    frame: RGBDFrame,
    calibration: CameraCalibration,
    config: SimulationConfig,
    api_client: ExternalInspectionApiClient | None,
) -> list[RGBDInspectionObject]:
    if api_client is not None:
        try:
            return api_client.inspect_rgbd(frame, calibration, config.workspace).objects
        except InspectionApiUnavailableError:
            return []

    objects: list[RGBDInspectionObject] = []
    for masked in VisionModule(config).detect_with_masks(frame.rgb):
        pose = estimate_object_pose_3d(masked.detected_object, masked.mask, frame.depth, calibration)
        if pose is None:
            continue
        grasp_pose = generate_top_down_grasp_pose(
            pose,
            grasp_height_offset=0.005,
            pre_grasp_z_offset=config.approach_height,
            retreat_z_offset=config.approach_height + 0.02,
            workspace=config.workspace,
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
    return objects


def _inspect_image_or_ground_truth(
    image: np.ndarray,
    config: SimulationConfig,
    env: MujocoSortingEnv,
    api_client: ExternalInspectionApiClient | None,
) -> tuple[list[DetectedObject], list[ExternalInspectionResult], bool]:
    if api_client is not None:
        try:
            response = api_client.inspect_image(image, config)
            return response.detected_objects, _fresh_inspections(response.detected_objects), False
        except InspectionApiUnavailableError:
            console.print("Image inspection API failed. Falling back to MuJoCo ground-truth object poses.")
            detections = env.get_ground_truth_detections()
            return detections, _fresh_inspections(detections), True

    detections = VisionModule(config).detect(image)
    if not detections:
        console.print("Image inspection produced no detections. Falling back to MuJoCo ground-truth object poses.")
        detections = env.get_ground_truth_detections()
        return detections, _fresh_inspections(detections), True
    return detections, _fresh_inspections(detections), False


def _detections_from_rgbd_objects(
    objects: list[RGBDInspectionObject],
    env: MujocoSortingEnv,
) -> tuple[list[DetectedObject], list[ExternalInspectionResult]]:
    detections = [
        DetectedObject(
            object_id=item.object_id,
            label=item.label,
            pixel_center=env.world_to_pixel(item.position),
            world_position=item.grasp_pose.grasp_position,
            confidence=item.confidence,
        )
        for item in objects
    ]
    return detections, _fresh_inspections(detections)


def _fresh_inspections(detections: list[DetectedObject]) -> list[ExternalInspectionResult]:
    inspected_at = time.perf_counter()
    return [
        ExternalInspectionResult(
            object_id=item.object_id,
            label=item.label,
            world_position=item.world_position,
            confidence=item.confidence,
            inspected_at=inspected_at,
        )
        for item in detections
    ]


def _object_pose_map(objects: list[RGBDInspectionObject]) -> dict[str, ObjectPose3D]:
    return {
        item.object_id: ObjectPose3D(
            object_id=item.object_id,
            label=item.label,
            position=item.position,
            orientation_rpy=item.orientation_rpy,
            size_xyz=item.size_xyz,
            confidence=item.confidence,
        )
        for item in objects
    }


def _grasp_pose_map(objects: list[RGBDInspectionObject]) -> dict[str, GraspPose]:
    return {item.object_id: item.grasp_pose for item in objects}


def _plan_trajectories(
    tasks: list[PickPlaceTask],
    grasp_pose_map: dict[str, GraspPose],
    config: SimulationConfig,
) -> dict[str, PlannedTrajectory]:
    trajectories: dict[str, PlannedTrajectory] = {}
    for task in tasks:
        grasp_pose = grasp_pose_map.get(task.object_id)
        if grasp_pose is None:
            continue
        trajectories[task.object_id] = plan_pick_place_trajectory(
            grasp_pose,
            task.place_position,
            config.workspace,
            config.base_radius,
            config.base_height,
            config.safety_margin,
        )
    return trajectories


if __name__ == "__main__":
    app()
