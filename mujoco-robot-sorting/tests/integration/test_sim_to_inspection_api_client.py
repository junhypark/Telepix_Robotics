"""Real HTTP tests for simulation client to FastAPI inspection API."""

from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn

from robot_sorting.api import app
from robot_sorting.modules.command_queue import create_robot_commands
from robot_sorting.modules.inspection_client import ExternalInspectionApiClient
from robot_sorting.modules.task_planner import TaskPlanner
from robot_sorting.perception.camera_calibration import build_top_down_workspace_calibration
from robot_sorting.planning.trajectory_planner import plan_pick_place_trajectory
from robot_sorting.schemas import ExternalInspectionResult, RGBDFrame
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv
from robot_sorting.simulation.renderer import MujocoRenderer

pytestmark = pytest.mark.integration


def test_simulation_client_uploads_rgbd_to_fastapi(two_object_config) -> None:
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server(port)
    try:
        env = MujocoSortingEnv(two_object_config)
        frame: RGBDFrame = MujocoRenderer(env).render_ground_truth_rgbd()
        calibration = build_top_down_workspace_calibration(frame.width, frame.height, two_object_config.workspace)
        client = ExternalInspectionApiClient(f"http://127.0.0.1:{port}")

        inspection_response = client.inspect_rgbd(frame, calibration, two_object_config.workspace)
        api_response_received_at = time.perf_counter()
        obj = inspection_response.objects[0]
        inspections = [
            ExternalInspectionResult(
                object_id=obj.object_id,
                label=obj.label,
                world_position=obj.grasp_pose.grasp_position,
                confidence=obj.confidence,
                inspected_at=api_response_received_at,
            )
        ]
        tasks = TaskPlanner(two_object_config).plan(inspections)
        trajectory = plan_pick_place_trajectory(
            obj.grasp_pose,
            tasks[0].place_position,
            two_object_config.workspace,
            two_object_config.base_radius,
            two_object_config.base_height,
            two_object_config.safety_margin,
        )
        commands = create_robot_commands(tasks, trajectories={obj.object_id: trajectory})
        elapsed_after_api_response = time.perf_counter() - api_response_received_at

        assert inspection_response.objects
        assert tasks[0] is not None
        assert trajectory.is_safe is True
        assert len(commands) >= 1
        assert elapsed_after_api_response <= 0.5
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(port: int) -> None:
    import httpx

    for _ in range(50):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=0.2).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise RuntimeError("FastAPI test server did not start")
