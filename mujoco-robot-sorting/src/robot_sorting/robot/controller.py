"""Robot controller for executing queued pick-and-place commands."""

from __future__ import annotations

import logging
import math
import time

from robot_sorting.robot.kinematics import solve_ik
from robot_sorting.robot.safety import (
    check_self_collision_risk,
    is_inside_base_exclusion_zone,
    is_inside_workspace,
    sample_line_segment,
)
from robot_sorting.schemas import RobotCommand, SimulationConfig, TaskExecutionResult
from robot_sorting.simulation.mujoco_env import MujocoSortingEnv

LOGGER = logging.getLogger(__name__)


class UnsafeMotionError(RuntimeError):
    """Raised when a requested motion fails safety validation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class RobotController:
    """Execute RobotCommand objects against a MuJoCoSortingEnv."""

    def __init__(self, env: MujocoSortingEnv, config: SimulationConfig) -> None:
        self.env = env
        self.config = config
        self.step_logs: list[dict[str, float | str]] = []

    def move_end_effector_to(self, target: tuple[float, float, float]) -> tuple[float, float, float]:
        """Move the end effector to a target if it passes safety checks."""

        self._validate_target(target)
        start = self.env.get_end_effector_position()
        self._validate_path(start, target)
        ik_result = None
        waypoints = sample_line_segment(
            start,
            target,
            samples=max(2, self.config.control_steps_per_move),
        )
        for waypoint in waypoints[1:]:
            ik_result = solve_ik(
                waypoint,
                self.config.link_1,
                self.config.link_2,
                self.config.base_height,
            )
            if ik_result.status == "invalid":
                raise UnsafeMotionError("ik_invalid")
            self.env.set_arm_joint_angles(ik_result.angles)
        final_pos = self.env.get_end_effector_position()
        ik_status = ik_result.status if ik_result is not None else "invalid"
        self.step_logs.append(
            {
                "timestamp": time.time(),
                "target_x": target[0],
                "target_y": target[1],
                "target_z": target[2],
                "final_x": final_pos[0],
                "final_y": final_pos[1],
                "final_z": final_pos[2],
                "ik_status": ik_status,
            }
        )
        return final_pos

    def execute_command(self, command: RobotCommand) -> TaskExecutionResult:
        """Execute one pick-and-place command and return a structured result."""

        task = command.task
        workspace_checked = True
        self_collision_checked = True
        try:
            self._validate_task_workspace(command)
            self._validate_task_self_collision(command)
            above_pick = self._above(task.pick_position)
            above_place = self._above(task.place_position)
            self.move_end_effector_to(above_pick)
            self.move_end_effector_to(task.pick_position)
            self.env.attach_nearest_object(task.pick_position, preferred_object_id=command.object_id)
            self.move_end_effector_to(above_pick)
            self.move_end_effector_to(above_place)
            self.move_end_effector_to(task.place_position)
            self.env.release_attached_object(task.place_position)
            self.move_end_effector_to(above_place)
            status = "completed"
            failure_reason = ""
        except UnsafeMotionError as exc:
            LOGGER.warning("Command for %s rejected: %s", command.object_id, exc.reason)
            status = "failed"
            failure_reason = exc.reason
        return TaskExecutionResult(
            object_id=task.object_id,
            label=task.label,
            pick_position=task.pick_position,
            place_position=task.place_position,
            target_bin=task.target_bin,
            status=status,
            failure_reason=failure_reason,
            command_latency_seconds=command.command_latency_seconds,
            self_collision_checked=self_collision_checked,
            workspace_checked=workspace_checked,
        )

    def execute_commands(self, commands: list[RobotCommand]) -> list[TaskExecutionResult]:
        """Execute commands sequentially."""

        return [self.execute_command(command) for command in commands]

    def _validate_task_workspace(self, command: RobotCommand) -> None:
        task_points = (
            command.task.pick_position,
            command.task.place_position,
            self._above(command.task.pick_position),
            self._above(command.task.place_position),
        )
        for point in task_points:
            if not is_inside_workspace(point, self.config.workspace):
                raise UnsafeMotionError("workspace_limit")

    def _validate_task_self_collision(self, command: RobotCommand) -> None:
        points = [
            command.task.pick_position,
            command.task.place_position,
            self._above(command.task.pick_position),
            self._above(command.task.place_position),
        ]
        for point in points:
            if is_inside_base_exclusion_zone(
                point,
                self.config.base_radius,
                0.0,
                self.config.base_height + self.config.safety_margin,
            ):
                raise UnsafeMotionError("self_collision_risk")
        current = self.env.get_end_effector_position()
        above_pick = self._above(command.task.pick_position)
        above_place = self._above(command.task.place_position)
        path_segments = [
            (current, above_pick),
            (above_pick, command.task.pick_position),
            (command.task.pick_position, above_pick),
            (above_pick, above_place),
            (above_place, command.task.place_position),
        ]
        path_points: list[tuple[float, float, float]] = []
        for start, end in path_segments:
            path_points.extend(sample_line_segment(start, end))
        if check_self_collision_risk(
            path_points,
            self.config.base_radius,
            self.config.base_height,
            self.config.safety_margin,
        ):
            raise UnsafeMotionError("self_collision_risk")

    def _validate_target(self, target: tuple[float, float, float]) -> None:
        if not all(math.isfinite(value) for value in target):
            raise UnsafeMotionError("workspace_limit")
        if not is_inside_workspace(target, self.config.workspace):
            raise UnsafeMotionError("workspace_limit")
        if is_inside_base_exclusion_zone(
            target,
            self.config.base_radius,
            0.0,
            self.config.base_height + self.config.safety_margin,
        ):
            raise UnsafeMotionError("self_collision_risk")

    def _validate_path(
        self,
        start: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> None:
        path_points = sample_line_segment(start, target)
        if check_self_collision_risk(
            path_points,
            self.config.base_radius,
            self.config.base_height,
            self.config.safety_margin,
        ):
            raise UnsafeMotionError("self_collision_risk")

    def _above(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            point[0],
            point[1],
            min(self.config.workspace.z_max, point[2] + self.config.approach_height),
        )
