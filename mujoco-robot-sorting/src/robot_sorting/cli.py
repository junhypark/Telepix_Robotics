"""Typer CLI for the robot sorting simulation."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Annotated, Any, Protocol

import numpy as np
import typer
from rich.console import Console

from robot_sorting.config import create_simulation_config
from robot_sorting.conveyor.artifacts import write_conveyor_images, write_report
from robot_sorting.conveyor.conveyor_controller import (
    should_stop_conveyor,
    transition_station_state,
    update_conveyor_object_position,
)
from robot_sorting.conveyor.conveyor_schemas import ConveyorConfig, ConveyorObjectState, StationState
from robot_sorting.conveyor.object_feeder import (
    can_spawn_next_object,
    create_feed_queue,
    get_next_object_to_feed,
    spawn_object_on_conveyor,
)
from robot_sorting.conveyor.station_state import StationTimelineRecorder
from robot_sorting.dashboard.dashboard_app import create_dashboard_api, write_static_dashboard
from robot_sorting.dashboard.run_data_loader import build_dashboard_data
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
from robot_sorting.planning.bin_placement_planner import find_non_overlapping_bin_slot
from robot_sorting.planning.trajectory_planner import plan_pick_place_trajectory
from robot_sorting.robot.controller import RobotController
from robot_sorting.scenarios import SCENARIOS, create_config_for_scenario, get_scenario, scenario_ids
from robot_sorting.schemas import (
    BinPlacementDecision,
    CameraCalibration,
    DetectedBin,
    DetectedObject,
    ExternalInspectionResult,
    GeneratedBin,
    GeneratedObject,
    GeneratedScenario,
    GraspPose,
    ObjectPose3D,
    PickPlaceTask,
    PlacedObjectRecord,
    PlannedTrajectory,
    RGBDFrame,
    RGBDInspectionObject,
    RobotCommand,
    SimulationConfig,
    TargetBin,
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
    scenario: Annotated[
        str | None,
        typer.Option("--scenario", help="Named production scenario id. Use list-scenarios to inspect options."),
    ] = None,
    viewer_delay: Annotated[
        float,
        typer.Option("--viewer-delay", min=0.0, help="Frame delay in seconds when --viewer is used."),
    ] = DEFAULT_VIEWER_DELAY_SECONDS,
    random_data: Annotated[
        bool,
        typer.Option("--random-data/--scenario-data", help="Use seed-based generated feeder data."),
    ] = False,
    enable_conveyor: Annotated[
        bool,
        typer.Option("--enable-conveyor/--disable-conveyor", help="Run the conveyor automation-cell flow."),
    ] = True,
    conveyor_speed_mps: Annotated[
        float,
        typer.Option("--conveyor-speed-mps", min=0.001, help="Kinematic conveyor speed in meters per second."),
    ] = 0.05,
    inspection_zone_x: Annotated[
        float,
        typer.Option("--inspection-zone-x", help="Inspection zone X coordinate."),
    ] = 0.30,
    inspection_zone_y: Annotated[
        float,
        typer.Option("--inspection-zone-y", help="Inspection zone Y coordinate."),
    ] = -0.22,
    pick_zone_x: Annotated[
        float,
        typer.Option("--pick-zone-x", help="Robot pick zone X coordinate."),
    ] = 0.30,
    pick_zone_y: Annotated[
        float,
        typer.Option("--pick-zone-y", help="Robot pick zone Y coordinate."),
    ] = -0.22,
    enable_dashboard: Annotated[
        bool,
        typer.Option("--enable-dashboard/--disable-dashboard", help="Generate static dashboard outputs."),
    ] = True,
    dashboard_output: Annotated[
        Path | None,
        typer.Option("--dashboard-output", help="Dashboard output directory. Defaults to OUTPUT/dashboard."),
    ] = None,
) -> None:
    """Run the full external-inspection sorting pipeline."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if scenario is None:
        config = create_simulation_config(
            headless=headless,
            objects=objects,
            seed=seed,
            width=width,
            height=height,
            output_dir=output,
            save_images=save_images,
        )
    else:
        try:
            config = create_config_for_scenario(
                scenario,
                headless=headless,
                width=width,
                height=height,
                output_dir=output,
                save_images=save_images,
            )
        except KeyError as exc:
            raise typer.BadParameter(str(exc), param_hint="--scenario") from exc

    _configure_conveyor_runtime(
        config,
        enable_conveyor=enable_conveyor,
        conveyor_speed_mps=conveyor_speed_mps,
        inspection_zone_x=inspection_zone_x,
        inspection_zone_y=inspection_zone_y,
        pick_zone_x=pick_zone_x,
        pick_zone_y=pick_zone_y,
        enable_dashboard=enable_dashboard,
        dashboard_output=dashboard_output,
        random_data=random_data,
    )
    if enable_conveyor:
        _run_conveyor_pipeline(config, inspection_api_url, viewer_delay)
        return

    env = MujocoSortingEnv(config)
    renderer = MujocoRenderer(env)
    rgbd_frame = renderer.render_rgbd()
    api_url = inspection_api_url or os.getenv("INSPECTION_API_URL")
    api_client = ExternalInspectionApiClient(api_url) if api_url else None
    calibration = build_top_down_workspace_calibration(config.width, config.height, config.workspace)
    rgbd_used = False
    ground_truth_fallback_used = False
    rgbd_objects: list[RGBDInspectionObject] = []
    detected_bins: list[DetectedBin] = env.get_ground_truth_bins()

    if rgbd_frame is not None:
        if config.save_images:
            renderer.save_rgb(rgbd_frame.rgb, config.output_dir / "camera_rgb.png")
            renderer.save_depth(rgbd_frame.depth, config.output_dir / "camera_depth.npy")
        rgbd_objects, detected_bins = _inspect_rgbd(rgbd_frame, calibration, config, env, api_client)
        if config.objects > 0 and len(rgbd_objects) < config.objects:
            console.print("Depth rendering unavailable. Falling back to MuJoCo ground-truth object poses.")
            renderer.depth_fallback_used = True
            rgbd_frame = renderer.render_ground_truth_rgbd()
            if config.save_images:
                renderer.save_rgb(rgbd_frame.rgb, config.output_dir / "camera_rgb_fallback.png")
                renderer.save_depth(rgbd_frame.depth, config.output_dir / "camera_depth_fallback.npy")
            rgbd_objects, detected_bins = _inspect_rgbd(rgbd_frame, calibration, config, env, api_client)
        if rgbd_objects:
            rgbd_used = True
            detections, inspections = _detections_from_rgbd_objects(rgbd_objects, env)
        else:
            console.print("RGB-D inspection API failed. Falling back to image inspection.")
            detections, inspections, detected_bins, ground_truth_fallback_used = _inspect_image_or_ground_truth(
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
            detected_bins = env.get_ground_truth_bins()
            ground_truth_fallback_used = True
        else:
            if config.save_images:
                renderer.save_rgb(image, config.output_dir / "camera_rgb.png")
            detections, inspections, detected_bins, ground_truth_fallback_used = _inspect_image_or_ground_truth(
                image,
                config,
                env,
                api_client,
            )

    object_pose_map = _object_pose_map(rgbd_objects)
    placement_decisions = _plan_bin_placements(inspections, object_pose_map, detected_bins, config)
    tasks = TaskPlanner(config).plan(inspections, detected_bins, placement_decisions)
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
        placed_objects=_placed_records_from_results(results, config),
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


@app.command("list-scenarios")
def list_scenarios() -> None:
    """List deterministic production scenarios."""

    for scenario in SCENARIOS:
        console.print(f"{scenario.scenario_id}: {scenario.name_ko}")
        console.print(f"  {scenario.description_ko}")


@app.command("run-scenarios")
def run_scenarios(
    output: Annotated[Path, typer.Option("--output", help="Scenario output root.")] = Path("outputs/scenarios"),
    save_images: Annotated[
        bool,
        typer.Option("--save-images/--no-save-images", help="Save camera images for each scenario."),
    ] = False,
) -> None:
    """Run every scenario and fail if any scenario does not complete."""

    results: list[dict[str, object]] = []
    for scenario_id in scenario_ids():
        scenario_output = output / scenario_id
        console.print(f"Running scenario: {scenario_id}")
        run(
            headless=True,
            objects=0,
            seed=0,
            output=scenario_output,
            save_images=save_images,
            inspection_api_url=None,
            scenario=scenario_id,
        )
        summary_path = scenario_output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        passed = summary["placed_count"] == summary["total_objects"] and summary["failed_count"] == 0
        results.append(
            {
                "scenario_id": scenario_id,
                "name_ko": get_scenario(scenario_id).name_ko,
                "passed": passed,
                "placed_count": summary["placed_count"],
                "total_objects": summary["total_objects"],
                "failed_count": summary["failed_count"],
            }
        )
        if not passed:
            (output / "scenario_results.json").parent.mkdir(parents=True, exist_ok=True)
            (output / "scenario_results.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            raise typer.Exit(code=1)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scenario_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"All {len(results)} scenarios passed. Outputs: {output}")


@app.command()
def dashboard(
    output: Annotated[Path, typer.Option("--output", help="Run output directory.")] = DEFAULT_OUTPUT_DIR,
    host: Annotated[str, typer.Option("--host", help="Dashboard server host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Dashboard server port.")] = 8080,
    static: Annotated[
        bool,
        typer.Option("--static/--serve", help="Only generate static files instead of serving FastAPI."),
    ] = False,
) -> None:
    """Generate or serve the run-result dashboard."""

    dashboard_data = build_dashboard_data(output)
    dashboard_dir = write_static_dashboard(output, dashboard_data)
    console.print(f"Dashboard generated: {dashboard_dir / 'index.html'}")
    if static:
        return
    import uvicorn

    uvicorn.run(create_dashboard_api(output), host=host, port=port)


@app.command()
def view(
    objects: Annotated[int, typer.Option("--objects", min=0, help="Number of objects to display.")] = 5,
    seed: Annotated[int, typer.Option("--seed", help="Deterministic scene seed.")] = 42,
    scenario: Annotated[
        str | None,
        typer.Option("--scenario", help="Named production scenario id. Use list-scenarios to inspect options."),
    ] = None,
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

    if scenario is None:
        config = create_simulation_config(headless=False, objects=objects, seed=seed)
    else:
        try:
            config = create_config_for_scenario(scenario, headless=False, save_images=False)
        except KeyError as exc:
            raise typer.BadParameter(str(exc), param_hint="--scenario") from exc
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
    rgbd_objects, detected_bins = _inspect_rgbd(frame, calibration, config, env, api_client=None)
    if not rgbd_objects:
        return []

    _, inspections = _detections_from_rgbd_objects(rgbd_objects, env)
    object_pose_map = _object_pose_map(rgbd_objects)
    placement_decisions = _plan_bin_placements(inspections, object_pose_map, detected_bins, config)
    tasks = TaskPlanner(config).plan(inspections, detected_bins, placement_decisions)
    return create_robot_commands(
        tasks,
        trajectories=_plan_trajectories(tasks, _grasp_pose_map(rgbd_objects), config),
        object_poses=object_pose_map,
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


def _configure_conveyor_runtime(
    config: SimulationConfig,
    *,
    enable_conveyor: bool,
    conveyor_speed_mps: float,
    inspection_zone_x: float,
    inspection_zone_y: float,
    pick_zone_x: float,
    pick_zone_y: float,
    enable_dashboard: bool,
    dashboard_output: Path | None,
    random_data: bool,
) -> None:
    config.conveyor_enabled = enable_conveyor
    config.conveyor_speed_mps = conveyor_speed_mps
    station_z = config.object_half_height + config.table_height
    config.inspection_zone_center = (inspection_zone_x, inspection_zone_y, station_z)
    config.pick_zone_center = (pick_zone_x, pick_zone_y, station_z)
    config.conveyor_entry_position = (
        min(config.inspection_zone_center[0], config.pick_zone_center[0]) - 0.18,
        inspection_zone_y,
        station_z,
    )
    config.dashboard_enabled = enable_dashboard
    config.dashboard_output_dir = dashboard_output
    config.random_data = random_data
    if enable_conveyor:
        config.bin_size_xyz = (0.30, 0.24, 0.04)


def _run_conveyor_pipeline(
    config: SimulationConfig,
    inspection_api_url: str | None,
    viewer_delay: float,
) -> None:
    env = MujocoSortingEnv(config)
    api_url = inspection_api_url or os.getenv("INSPECTION_API_URL")
    api_client = ExternalInspectionApiClient(api_url) if api_url else None
    conveyor_config = _conveyor_config_from_simulation(config)
    generated = _generated_scenario_from_env(env, config)
    feed_queue = create_feed_queue(generated)
    recorder = StationTimelineRecorder()
    station = StationState(state="IDLE", active_object_id=None, timestamp=0.0)
    controller = RobotController(env, config)
    detections: list[DetectedObject] = []
    tasks: list[PickPlaceTask] = []
    results: list[TaskExecutionResult] = []
    placed_records: list[PlacedObjectRecord] = []
    transport_modes: dict[str, str] = {}
    cycle_times: list[float] = []
    no_free_space_count = 0
    clock = 0.0
    active_state: ConveyorObjectState | None = None
    _park_queued_objects(env, generated.objects, config)

    while feed_queue:
        if not can_spawn_next_object(active_state):
            break
        product = get_next_object_to_feed(feed_queue)
        if product is None:
            break
        active_state = spawn_object_on_conveyor(product, conveyor_config.entry_position)
        cycle_start = clock
        env.set_object_position(active_state.object_id, active_state.current_position)
        station = transition_station_state(
            station,
            recorder.record(
                timestamp=clock,
                state="FEEDING",
                event="object_spawned",
                conveyor_running=True,
                object_state=active_state,
            ),
        )
        station = transition_station_state(
            station,
            recorder.record(
                timestamp=clock,
                state="MOVING_TO_INSPECTION",
                event="conveyor_started",
                conveyor_running=True,
                object_state=active_state,
            ),
        )
        active_state, clock = _advance_conveyor_to_inspection(
            env,
            active_state,
            conveyor_config,
            recorder,
            clock,
        )
        station = transition_station_state(
            station,
            recorder.record(
                timestamp=clock,
                state="INSPECTING",
                event="conveyor_stopped_at_inspection_zone",
                conveyor_running=False,
                object_state=active_state,
            ),
        )
        if not should_stop_conveyor(station):
            raise RuntimeError("Conveyor must be stopped during inspection")
        detection, inspection = _inspect_conveyor_object(active_state, config, env, api_client)
        detections.append(detection)
        clock += 0.05
        active_state = active_state.model_copy(update={"status": "inspected"})
        detected_bins = env.get_ground_truth_bins()
        object_pose = _object_pose_from_conveyor_state(active_state, config)
        grasp_pose = _grasp_pose_from_object_pose(object_pose, config)
        rgbd_object = RGBDInspectionObject(
            object_id=object_pose.object_id,
            label=object_pose.label,
            position=object_pose.position,
            orientation_rpy=object_pose.orientation_rpy,
            size_xyz=object_pose.size_xyz,
            confidence=object_pose.confidence,
            grasp_pose=grasp_pose,
        )
        object_pose_map = {object_pose.object_id: object_pose}
        decisions = _plan_bin_placements(
            [inspection],
            object_pose_map,
            detected_bins,
            config,
            existing_placed_records=placed_records,
        )
        decision = decisions[0] if decisions else None
        if decision is None or decision.failure_reason is not None:
            no_free_space_count += 1
            result = _failed_conveyor_result(active_state, config, "no_free_space_inside_bin")
            results.append(result)
            active_state = active_state.model_copy(update={"status": "failed"})
            station = transition_station_state(
                station,
                recorder.record(
                    timestamp=clock,
                    state="FAILED",
                    event="object_failed",
                    conveyor_running=False,
                    object_state=active_state,
                    reason=result.failure_reason,
                ),
            )
            cycle_times.append(clock - cycle_start)
            continue
        planned_tasks = TaskPlanner(config).plan([inspection], detected_bins, [decision])
        if not planned_tasks:
            result = _failed_conveyor_result(active_state, config, "workspace_limit")
            results.append(result)
            active_state = active_state.model_copy(update={"status": "failed"})
            recorder.record(
                timestamp=clock,
                state="FAILED",
                event="object_failed",
                conveyor_running=False,
                object_state=active_state,
                reason=result.failure_reason,
            )
            cycle_times.append(clock - cycle_start)
            continue
        task = planned_tasks[0]
        tasks.append(task)
        transport_mode = _decide_transport_mode(task)
        transport_modes[task.object_id] = transport_mode
        recorder.record(
            timestamp=clock,
            state="WAITING_FOR_PICK",
            event="transport_mode_decided",
            conveyor_running=False,
            object_state=active_state,
            extra={"transport_mode": transport_mode},
        )
        waiting_state = StationState(state="WAITING_FOR_PICK", active_object_id=task.object_id, timestamp=clock)
        if should_stop_conveyor(waiting_state):
            recorder.record(
                timestamp=clock,
                state="PICKING",
                event="conveyor_stopped_for_pick",
                conveyor_running=False,
                object_state=active_state,
            )
        trajectories = _plan_trajectories([task], {rgbd_object.object_id: grasp_pose}, config)
        commands = create_robot_commands(
            [task],
            trajectories=trajectories,
            object_poses={object_pose.object_id: object_pose},
            grasp_poses={grasp_pose.object_id: grasp_pose},
        )
        command_queue = RobotCommandQueue()
        command_queue.extend(commands)
        command = command_queue.pop_all()[0]
        result = controller.execute_command(command)
        results.append(result)
        if result.status == "completed":
            placed_records.extend(_placed_records_from_results([result], config))
            active_state = active_state.model_copy(update={"status": "placed"})
            event = "object_placed"
            state_name = "COMPLETED"
        else:
            active_state = active_state.model_copy(update={"status": "failed"})
            event = "object_failed"
            state_name = "FAILED"
        clock += _trajectory_duration_seconds(command.trajectory)
        station = transition_station_state(
            station,
            recorder.record(
                timestamp=clock,
                state="PLACING" if result.status == "completed" else "FAILED",
                event=event,
                conveyor_running=False,
                object_state=active_state,
                reason=result.failure_reason or None,
            ),
        )
        if state_name == "COMPLETED":
            station = transition_station_state(
                station,
                recorder.record(
                    timestamp=clock,
                    state="COMPLETED",
                    event="cycle_completed",
                    conveyor_running=False,
                    object_state=active_state,
                ),
            )
        cycle_times.append(clock - cycle_start)
        if not config.headless:
            time.sleep(viewer_delay)

    recorder.write(config.output_dir)
    logger = ResultLogger(config.output_dir)
    conveyor_metrics = _build_conveyor_metrics(
        config,
        recorder,
        results,
        cycle_times,
        transport_modes,
        no_free_space_count,
    )
    summary = logger.write_all(
        detections,
        tasks,
        results,
        placed_objects=placed_records,
        min_confidence=config.min_confidence,
        rgbd_used=True,
        depth_fallback_used=False,
        ground_truth_fallback_used=False,
        conveyor_metrics=conveyor_metrics,
    )
    write_conveyor_images(config.output_dir, detections, tasks, config)
    dashboard_dir = config.dashboard_output_dir or config.output_dir / "dashboard"
    if config.dashboard_enabled:
        dashboard_data = build_dashboard_data(config.output_dir)
        dashboard_dir = write_static_dashboard(config.output_dir, dashboard_data, dashboard_dir)
    write_report(config.output_dir, conveyor_config, dashboard_dir)
    console.print(f"Run complete. Outputs: {config.output_dir}")
    console.print(
        f"Placed {summary.placed_count}/{summary.total_objects}, "
        f"conveyor stops={summary.conveyor_stop_count}, dashboard={dashboard_dir}"
    )


def _advance_conveyor_to_inspection(
    env: MujocoSortingEnv,
    state: ConveyorObjectState,
    config: ConveyorConfig,
    recorder: StationTimelineRecorder,
    clock: float,
) -> tuple[ConveyorObjectState, float]:
    dt = 0.1
    current = state
    for _ in range(2000):
        current = update_conveyor_object_position(current, config, dt)
        clock += dt
        env.set_object_position(current.object_id, current.current_position)
        if current.status == "at_inspection_zone":
            return current, clock
    recorder.record(
        timestamp=clock,
        state="FAILED",
        event="inspection_zone_timeout",
        conveyor_running=False,
        object_state=current,
        reason="inspection_zone_timeout",
    )
    return current.model_copy(update={"status": "failed"}), clock


def _inspect_conveyor_object(
    state: ConveyorObjectState,
    config: SimulationConfig,
    env: MujocoSortingEnv,
    api_client: ExternalInspectionApiClient | None,
) -> tuple[DetectedObject, ExternalInspectionResult]:
    detection = DetectedObject(
        object_id=state.object_id,
        label=state.label,
        pixel_center=env.world_to_pixel(state.current_position),
        world_position=state.current_position,
        confidence=1.0,
    )
    inspection = _inspect_detections([detection], api_client)[0]
    return detection, inspection.model_copy(update={"world_position": config.pick_zone_center})


def _object_pose_from_conveyor_state(state: ConveyorObjectState, config: SimulationConfig) -> ObjectPose3D:
    return ObjectPose3D(
        object_id=state.object_id,
        label=state.label,
        position=state.current_position,
        orientation_rpy=(0.0, 0.0, 0.0),
        size_xyz=_default_object_size(config),
        confidence=1.0,
    )


def _grasp_pose_from_object_pose(object_pose: ObjectPose3D, config: SimulationConfig) -> GraspPose:
    grasp_pose = generate_top_down_grasp_pose(
        object_pose,
        grasp_height_offset=0.005,
        pre_grasp_z_offset=config.approach_height,
        retreat_z_offset=config.approach_height + 0.02,
        workspace=config.workspace,
    )
    if grasp_pose is not None:
        return grasp_pose
    x, y, z = object_pose.position
    return GraspPose(
        object_id=object_pose.object_id,
        pre_grasp_position=(x, y, min(config.workspace.z_max, z + config.approach_height)),
        grasp_position=(x, y, z),
        retreat_position=(x, y, min(config.workspace.z_max, z + config.approach_height + 0.02)),
        approach_vector=(0.0, 0.0, -1.0),
        gripper_yaw=0.0,
        confidence=0.75,
    )


def _failed_conveyor_result(
    state: ConveyorObjectState,
    config: SimulationConfig,
    reason: str,
) -> TaskExecutionResult:
    target_bin: TargetBin = "normal_bin" if state.label == "normal" else "defect_bin"
    place = config.normal_bin_position if target_bin == "normal_bin" else config.defect_bin_position
    return TaskExecutionResult(
        object_id=state.object_id,
        label=state.label,
        pick_position=state.current_position,
        place_position=place,
        target_bin=target_bin,
        target_bin_id=target_bin,
        status="failed",
        failure_reason=reason,
        command_latency_seconds=0.0,
        self_collision_checked=True,
        workspace_checked=True,
        table_clearance_checked=True,
    )


def _build_conveyor_metrics(
    config: SimulationConfig,
    recorder: StationTimelineRecorder,
    results: list[TaskExecutionResult],
    cycle_times: list[float],
    transport_modes: dict[str, str],
    no_free_space_count: int,
) -> dict[str, Any]:
    stop_count = sum(1 for row in recorder.events if str(row.get("event", "")).startswith("conveyor_stopped"))
    success_count = sum(1 for result in results if result.status == "completed")
    failure_count = sum(1 for result in results if result.status == "failed")
    return {
        "conveyor_enabled": True,
        "conveyor_speed_mps": config.conveyor_speed_mps,
        "processed_object_count": len(results),
        "conveyor_stop_count": stop_count,
        "inspection_station_count": sum(1 for row in recorder.events if row.get("state") == "INSPECTING"),
        "pick_station_success_count": success_count,
        "pick_station_failure_count": failure_count,
        "average_station_cycle_time_seconds": sum(cycle_times) / len(cycle_times) if cycle_times else 0.0,
        "max_station_cycle_time_seconds": max(cycle_times) if cycle_times else 0.0,
        "overhead_rotate_count": sum(1 for mode in transport_modes.values() if mode == "overhead_rotate"),
        "level_parallel_count": sum(1 for mode in transport_modes.values() if mode == "level_parallel"),
        "no_free_space_inside_bin_count": no_free_space_count,
    }


def _conveyor_config_from_simulation(config: SimulationConfig) -> ConveyorConfig:
    return ConveyorConfig(
        conveyor_speed_mps=config.conveyor_speed_mps,
        conveyor_axis=config.conveyor_axis,
        entry_position=config.conveyor_entry_position,
        inspection_zone_center=config.inspection_zone_center,
        pick_zone_center=config.pick_zone_center,
        zone_tolerance_meters=config.conveyor_zone_tolerance_meters,
        max_feed_count=config.objects,
    )


def _generated_scenario_from_env(env: MujocoSortingEnv, config: SimulationConfig) -> GeneratedScenario:
    objects = [
        GeneratedObject(
            object_id=item.object_id,
            label=item.label,
            target_spawn_position=config.inspection_zone_center,
            actual_spawn_position=item.position,
            size_xyz=_default_object_size(config),
        )
        for item in env.object_specs
    ]
    bins = [
        GeneratedBin(
            bin_id=item.bin_id,
            label=item.label,
            color_rgb=item.color_rgb,
            target_spawn_position=item.position,
            actual_spawn_position=item.position,
            size_xyz=item.size_xyz,
            orientation_rpy=item.orientation_rpy,
        )
        for item in env.bin_specs
    ]
    return GeneratedScenario(
        seed=config.seed,
        objects=objects,
        bins=bins,
        normal_count=sum(1 for item in objects if item.label == "normal"),
        defect_count=sum(1 for item in objects if item.label == "defect"),
    )


def _park_queued_objects(env: MujocoSortingEnv, objects: list[GeneratedObject], config: SimulationConfig) -> None:
    for index, item in enumerate(objects):
        env.set_object_position(
            item.object_id,
            (
                config.workspace.x_min + 0.03,
                config.workspace.y_min + 0.03 + index * 0.002,
                config.table_height + config.object_half_height,
            ),
        )


def _decide_transport_mode(task: PickPlaceTask) -> str:
    lateral_delta = abs(task.place_position[1] - task.pick_position[1])
    x_delta = abs(task.place_position[0] - task.pick_position[0])
    if lateral_delta >= 0.20 or x_delta >= 0.22:
        return "overhead_rotate"
    return "level_parallel"


def _trajectory_duration_seconds(trajectory: PlannedTrajectory | None) -> float:
    if trajectory is None:
        return 0.5
    return sum(waypoint.duration_seconds for waypoint in trajectory.waypoints)


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
    env: MujocoSortingEnv,
    api_client: ExternalInspectionApiClient | None,
) -> tuple[list[RGBDInspectionObject], list[DetectedBin]]:
    if api_client is not None:
        try:
            response = api_client.inspect_rgbd(frame, calibration, config.workspace)
            return response.objects, response.bins or env.get_ground_truth_bins()
        except InspectionApiUnavailableError:
            return [], env.get_ground_truth_bins()

    objects: list[RGBDInspectionObject] = []
    vision = VisionModule(config)
    for masked in vision.detect_with_masks(frame.rgb):
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
    detected_bins = vision.detect_bins(frame.rgb) or env.get_ground_truth_bins()
    return objects, detected_bins


def _inspect_image_or_ground_truth(
    image: np.ndarray,
    config: SimulationConfig,
    env: MujocoSortingEnv,
    api_client: ExternalInspectionApiClient | None,
) -> tuple[list[DetectedObject], list[ExternalInspectionResult], list[DetectedBin], bool]:
    if api_client is not None:
        try:
            response = api_client.inspect_image(image, config)
            detected_bins = response.detected_bins or env.get_ground_truth_bins()
            return response.detected_objects, _fresh_inspections(response.detected_objects), detected_bins, False
        except InspectionApiUnavailableError:
            console.print("Image inspection API failed. Falling back to MuJoCo ground-truth object poses.")
            detections = env.get_ground_truth_detections()
            return detections, _fresh_inspections(detections), env.get_ground_truth_bins(), True

    vision = VisionModule(config)
    detections = vision.detect(image)
    detected_bins = vision.detect_bins(image) or env.get_ground_truth_bins()
    if not detections:
        console.print("Image inspection produced no detections. Falling back to MuJoCo ground-truth object poses.")
        detections = env.get_ground_truth_detections()
        return detections, _fresh_inspections(detections), detected_bins, True
    return detections, _fresh_inspections(detections), detected_bins, False


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


def _plan_bin_placements(
    inspections: list[ExternalInspectionResult],
    object_pose_map: dict[str, ObjectPose3D],
    detected_bins: list[DetectedBin],
    config: SimulationConfig,
    *,
    existing_placed_records: list[PlacedObjectRecord] | None = None,
) -> list[BinPlacementDecision]:
    bin_by_label = {detected_bin.label: detected_bin for detected_bin in detected_bins}
    placed_records: list[PlacedObjectRecord] = list(existing_placed_records or [])
    decisions: list[BinPlacementDecision] = []
    for inspection in sorted(inspections, key=lambda item: (item.object_id, item.label)):
        target_bin: TargetBin = "normal_bin" if inspection.label == "normal" else "defect_bin"
        detected_bin = bin_by_label.get(target_bin)
        if detected_bin is None:
            decisions.append(
                BinPlacementDecision(
                    object_id=inspection.object_id,
                    target_bin_id=target_bin,
                    target_bin=target_bin,
                    placement_position=inspection.world_position,
                    placement_pixel=None,
                    placement_strategy="fallback_center",
                    confidence=0.0,
                    reason="bin_not_detected",
                    failure_reason="bin_not_detected",
                )
            )
            continue
        object_size = _object_size_for_placement(inspection.object_id, object_pose_map, config)
        decision = find_non_overlapping_bin_slot(
            detected_bin,
            object_size,
            placed_records,
            config.bin_placement,
            config.workspace,
            object_id=inspection.object_id,
        )
        decisions.append(decision)
        if decision.failure_reason is None:
            placed_records.append(
                PlacedObjectRecord(
                    object_id=inspection.object_id,
                    target_bin_id=decision.target_bin_id,
                    position=decision.placement_position,
                    size_xyz=object_size,
                )
            )
    return decisions


def _placed_records_from_results(
    results: list[TaskExecutionResult],
    config: SimulationConfig,
) -> list[PlacedObjectRecord]:
    placed: list[PlacedObjectRecord] = []
    for result in results:
        if result.status != "completed" or result.target_bin_id is None:
            continue
        size_xyz = result.object_pose.size_xyz if result.object_pose is not None else _default_object_size(config)
        placed.append(
            PlacedObjectRecord(
                object_id=result.object_id,
                target_bin_id=result.target_bin_id,
                position=result.place_position,
                size_xyz=size_xyz,
            )
        )
    return placed


def _object_size_for_placement(
    object_id: str,
    object_pose_map: dict[str, ObjectPose3D],
    config: SimulationConfig,
) -> tuple[float, float, float]:
    pose = object_pose_map.get(object_id)
    if pose is not None:
        return pose.size_xyz
    return _default_object_size(config)


def _default_object_size(config: SimulationConfig) -> tuple[float, float, float]:
    return (
        config.object_radius * 2.0,
        config.object_radius * 2.0,
        config.object_half_height * 2.0,
    )


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
            config.table_safety,
            config.link_1,
            config.link_2,
        )
    return trajectories


if __name__ == "__main__":
    app()
