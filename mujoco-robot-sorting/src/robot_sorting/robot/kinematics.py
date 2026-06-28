"""Analytical inverse kinematics for the educational robot arm."""

from __future__ import annotations

import math

from robot_sorting.schemas import IKResult, JointAngles, JointLimits

DEFAULT_JOINT_LIMITS = JointLimits()


def solve_ik(
    target: tuple[float, float, float],
    link_1: float,
    link_2: float,
    base_height: float,
) -> IKResult:
    """Solve yaw + shoulder + elbow inverse kinematics for a 2-link arm."""

    if link_1 <= 0 or link_2 <= 0 or not all(math.isfinite(value) for value in target):
        return IKResult(
            angles=JointAngles(base_yaw=0.0, shoulder=0.0, elbow=0.0),
            status="invalid",
            clamped_target=(0.0, 0.0, base_height),
        )

    x, y, z = target
    yaw = math.atan2(y, x)
    radial = math.hypot(x, y)
    vertical = z - base_height
    requested_distance = math.hypot(radial, vertical)

    min_reach = max(abs(link_1 - link_2) + 1e-6, 1e-6)
    max_reach = link_1 + link_2 - 1e-6
    clamped_distance = min(max(requested_distance, min_reach), max_reach)
    status = "reachable"
    if not math.isclose(requested_distance, clamped_distance, rel_tol=1e-9, abs_tol=1e-9):
        status = "clamped"
        scale = clamped_distance / requested_distance if requested_distance > 1e-9 else 1.0
        radial *= scale
        vertical *= scale

    cos_elbow = (radial**2 + vertical**2 - link_1**2 - link_2**2) / (2.0 * link_1 * link_2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow_standard = math.acos(cos_elbow)
    shoulder_standard = math.atan2(vertical, radial) - math.atan2(
        link_2 * math.sin(elbow_standard),
        link_1 + link_2 * math.cos(elbow_standard),
    )

    angles = JointAngles(
        base_yaw=_clamp(yaw, DEFAULT_JOINT_LIMITS.base_min, DEFAULT_JOINT_LIMITS.base_max),
        shoulder=_clamp(-shoulder_standard, DEFAULT_JOINT_LIMITS.shoulder_min, DEFAULT_JOINT_LIMITS.shoulder_max),
        elbow=_clamp(-elbow_standard, DEFAULT_JOINT_LIMITS.elbow_min, DEFAULT_JOINT_LIMITS.elbow_max),
    )
    if not _angles_equal_to_raw(angles, yaw, -shoulder_standard, -elbow_standard):
        status = "clamped"

    clamped_target = (
        math.cos(yaw) * radial,
        math.sin(yaw) * radial,
        base_height + vertical,
    )
    if not all(math.isfinite(value) for value in (*angles.model_dump().values(), *clamped_target)):
        return IKResult(
            angles=JointAngles(base_yaw=0.0, shoulder=0.0, elbow=0.0),
            status="invalid",
            clamped_target=(0.0, 0.0, base_height),
        )
    return IKResult(angles=angles, status=status, clamped_target=clamped_target)


def forward_kinematics(
    angles: JointAngles,
    link_1: float,
    link_2: float,
    base_height: float,
) -> tuple[float, float, float]:
    """Compute the end-effector position for the MuJoCo joint convention."""

    planar_x = link_1 * math.cos(angles.shoulder) + link_2 * math.cos(angles.shoulder + angles.elbow)
    z = base_height - link_1 * math.sin(angles.shoulder) - link_2 * math.sin(angles.shoulder + angles.elbow)
    x = planar_x * math.cos(angles.base_yaw)
    y = planar_x * math.sin(angles.base_yaw)
    return (float(x), float(y), float(z))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _angles_equal_to_raw(angles: JointAngles, yaw: float, shoulder: float, elbow: float) -> bool:
    return (
        math.isclose(angles.base_yaw, yaw, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(angles.shoulder, shoulder, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(angles.elbow, elbow, rel_tol=1e-9, abs_tol=1e-9)
    )

