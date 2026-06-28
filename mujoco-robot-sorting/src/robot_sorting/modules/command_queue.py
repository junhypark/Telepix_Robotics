"""Command queue with command-generation latency tracking."""

from __future__ import annotations

import time
from collections import deque

from robot_sorting.schemas import PickPlaceTask, RobotCommand


def create_robot_commands(tasks: list[PickPlaceTask]) -> list[RobotCommand]:
    """Create robot commands from planned tasks."""

    commands: list[RobotCommand] = []
    for task in tasks:
        queued_at = time.perf_counter()
        inspection_at = task.inspection_at
        latency = max(0.0, queued_at - inspection_at) if inspection_at is not None else 0.0
        # SLA: Once the external inspection result is available,
        # a robot command must be generated and queued within 0.5 seconds.
        commands.append(
            RobotCommand(
                object_id=task.object_id,
                task=task,
                queued_at=queued_at,
                inspection_at=inspection_at,
                command_latency_seconds=latency,
            )
        )
    return commands


class RobotCommandQueue:
    """Simple FIFO queue used to deliver commands to the controller."""

    def __init__(self) -> None:
        self._queue: deque[RobotCommand] = deque()

    def extend(self, commands: list[RobotCommand]) -> None:
        """Append robot commands in planner order."""

        self._queue.extend(commands)

    def pop_all(self) -> list[RobotCommand]:
        """Drain and return all queued commands."""

        commands = list(self._queue)
        self._queue.clear()
        return commands

    def __len__(self) -> int:
        return len(self._queue)

