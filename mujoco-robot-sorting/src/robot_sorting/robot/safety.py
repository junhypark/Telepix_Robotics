"""Workspace and self-collision safety checks."""

from __future__ import annotations

import math

from robot_sorting.schemas import WorkspaceBounds


def is_inside_workspace(
    point: tuple[float, float, float],
    workspace: WorkspaceBounds,
) -> bool:
    """Return whether a point is inside inclusive workspace limits."""

    x, y, z = point
    return (
        math.isfinite(x)
        and math.isfinite(y)
        and math.isfinite(z)
        and workspace.x_min <= x <= workspace.x_max
        and workspace.y_min <= y <= workspace.y_max
        and workspace.z_min <= z <= workspace.z_max
    )


def is_inside_base_exclusion_zone(
    point: tuple[float, float, float],
    base_radius: float,
    min_z: float,
    max_z: float,
) -> bool:
    """Return whether a point enters the robot base/body exclusion cylinder."""

    x, y, z = point
    if not all(math.isfinite(value) for value in point):
        return True
    radial = math.hypot(x, y)
    return radial <= base_radius and min_z <= z <= max_z


def check_self_collision_risk(
    link_points: list[tuple[float, float, float]],
    base_radius: float,
    base_height: float,
    safety_margin: float,
) -> bool:
    """Check whether sampled path/link points enter the base exclusion volume."""

    inflated_radius = base_radius + safety_margin
    inflated_height = base_height + safety_margin
    return any(
        is_inside_base_exclusion_zone(point, inflated_radius, 0.0, inflated_height)
        for point in link_points
    )


def sample_line_segment(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    samples: int = 16,
) -> list[tuple[float, float, float]]:
    """Sample points along a straight segment for conservative safety checks."""

    if samples <= 1:
        return [end]
    points: list[tuple[float, float, float]] = []
    for index in range(samples):
        alpha = index / (samples - 1)
        points.append(
            (
                start[0] + (end[0] - start[0]) * alpha,
                start[1] + (end[1] - start[1]) * alpha,
                start[2] + (end[2] - start[2]) * alpha,
            )
        )
    return points

