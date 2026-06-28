"""Integration tests for RGB-D perception pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_sorting.perception.camera_calibration import build_top_down_workspace_calibration
from robot_sorting.perception.grasp_pose_generator import generate_top_down_grasp_pose
from robot_sorting.perception.pose_estimator import estimate_object_pose_3d
from robot_sorting.schemas import SimulationConfig
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv
from robot_sorting.simulation.renderer import MujocoRenderer

pytestmark = pytest.mark.integration


def test_rgbd_pipeline_produces_pose_and_grasp(two_object_config: SimulationConfig) -> None:
    env = MujocoSortingEnv(two_object_config)
    renderer = MujocoRenderer(env)
    frame = renderer.render_ground_truth_rgbd()
    calibration = build_top_down_workspace_calibration(
        two_object_config.width,
        two_object_config.height,
        two_object_config.workspace,
    )

    assert frame.rgb.shape[:2] == frame.depth.shape
    masked = __import__("robot_sorting.modules.vision_module", fromlist=["VisionModule"]).VisionModule(
        two_object_config
    ).detect_with_masks(frame.rgb)
    assert masked
    pose = estimate_object_pose_3d(masked[0].detected_object, masked[0].mask, frame.depth, calibration)
    assert pose is not None
    grasp = generate_top_down_grasp_pose(pose, 0.005, 0.11, 0.13, two_object_config.workspace)
    assert grasp is not None
    assert all(math.isfinite(value) for value in pose.position)
    assert np.all(np.isfinite(frame.depth))

