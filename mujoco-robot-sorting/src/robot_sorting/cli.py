"""Typer CLI for the robot sorting simulation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

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
from robot_sorting.robot.controller import RobotController
from robot_sorting.schemas import DetectedObject, ExternalInspectionResult
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv
from robot_sorting.simulation.renderer import MujocoCameraRenderer

app = typer.Typer(help="MuJoCo robot arm sorting simulation")
console = Console()
DEFAULT_OUTPUT_DIR = Path("outputs/run")


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
    renderer = MujocoCameraRenderer(env)
    image = renderer.render_rgb()
    api_url = inspection_api_url or os.getenv("INSPECTION_API_URL")
    api_client = ExternalInspectionApiClient(api_url) if api_url else None

    if image is None:
        console.print("Renderer unavailable. Falling back to simulation ground-truth object positions.")
        detections = env.get_ground_truth_detections()
        inspections = _inspect_detections(detections, api_client)
    else:
        if config.save_images:
            renderer.save_rgb(image, config.output_dir / "camera_rgb.png")
        if api_client is not None:
            try:
                response = api_client.inspect_image(image, config)
                detections = response.detected_objects
                inspections = response.inspection_results
            except InspectionApiUnavailableError as exc:
                console.print(f"Inspection API unavailable, using local modules: {exc}")
                detections = VisionModule(config).detect(image)
                inspections = InspectionModule().inspect_all(detections)
        else:
            detections = VisionModule(config).detect(image)
            inspections = InspectionModule().inspect_all(detections)
        if not detections:
            console.print("Renderer produced no detections. Falling back to simulation ground-truth object positions.")
            detections = env.get_ground_truth_detections()
            inspections = _inspect_detections(detections, api_client)

    tasks = TaskPlanner(config).plan(inspections)
    commands = create_robot_commands(tasks)
    command_queue = RobotCommandQueue()
    command_queue.extend(commands)
    controller = RobotController(env, config)
    results = controller.execute_commands(command_queue.pop_all())

    logger = ResultLogger(config.output_dir)
    summary = logger.write_all(
        detections,
        tasks,
        results,
        min_confidence=config.min_confidence,
    )
    console.print(f"Run complete. Outputs: {config.output_dir}")
    console.print(
        f"Placed {summary.placed_count}/{summary.total_objects}, "
        f"max command latency={summary.max_command_latency_seconds:.6f}s"
    )


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


if __name__ == "__main__":
    app()
