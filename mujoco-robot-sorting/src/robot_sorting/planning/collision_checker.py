"""Conservative 3D collision checks for planned trajectories."""

from __future__ import annotations

import math
from itertools import pairwise

from robot_sorting.robot.safety import check_self_collision_risk, is_inside_base_exclusion_zone, sample_line_segment
from robot_sorting.schemas import TrajectoryWaypoint


def segment_intersects_base_exclusion_zone(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    base_radius: float,
    base_height: float,
    safety_margin: float,
) -> bool:
    """Return whether a path segment intersects the inflated robot base cylinder."""

    if not all(math.isfinite(value) for value in (*start, *end)):
        return True
    inflated_radius = base_radius + safety_margin
    inflated_height = base_height + safety_margin
    if is_inside_base_exclusion_zone(start, inflated_radius, 0.0, inflated_height):
        return True
    if is_inside_base_exclusion_zone(end, inflated_radius, 0.0, inflated_height):
        return True
    return check_self_collision_risk(
        sample_line_segment(start, end, samples=24),
        base_radius,
        base_height,
        safety_margin,
    )


def trajectory_has_collision_risk(
    waypoints: list[TrajectoryWaypoint],
    base_radius: float,
    base_height: float,
    safety_margin: float,
) -> bool:
    """Return whether any waypoint or segment risks colliding with the base/body."""

    if not waypoints:
        return False
    inflated_radius = base_radius + safety_margin
    inflated_height = base_height + safety_margin
    for waypoint in waypoints:
        if is_inside_base_exclusion_zone(waypoint.position, inflated_radius, 0.0, inflated_height):
            return True
    for start, end in pairwise(waypoints):
        if segment_intersects_base_exclusion_zone(
            start.position,
            end.position,
            base_radius,
            base_height,
            safety_margin,
        ):
            return True
    return False
