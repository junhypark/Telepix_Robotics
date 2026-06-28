"""Headless MuJoCo environment wrapper for the sorting task."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from robot_sorting.schemas import DetectedBin, DetectedObject, JointAngles, SimulationConfig
from robot_sorting.simulation.scene_builder import SceneBin, SceneBuilder, SceneObject

LOGGER = logging.getLogger(__name__)


class MujocoSortingEnv:
    """Load and control the deterministic MuJoCo sorting scene."""

    arm_joint_names = ("base_yaw_joint", "shoulder_joint", "elbow_joint")

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.scene_builder = SceneBuilder(config)
        self.scene_xml = self.scene_builder.build_xml()
        self.object_specs = self.scene_builder.objects
        self.bin_specs = self.scene_builder.bins
        self.model: Any = None
        self.data: Any = None
        self._mujoco: Any = None
        self._attached_object_id: str | None = None
        self.reset()

    def reset(self) -> None:
        """Load a fresh MuJoCo model and data object."""

        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(self.scene_xml)
        self.data = mujoco.MjData(self.model)
        for obj in self.object_specs:
            self.set_object_position(obj.object_id, obj.position)
        mujoco.mj_forward(self.model, self.data)

    def step(self, steps: int = 1) -> None:
        """Advance the simulation by a fixed number of steps."""

        self._require_loaded()
        for _ in range(max(0, steps)):
            self._mujoco.mj_step(self.model, self.data)

    def get_body_id(self, name: str) -> int:
        """Return a MuJoCo body id by name."""

        return self._name_to_id(self._mujoco.mjtObj.mjOBJ_BODY, name)

    def get_geom_id(self, name: str) -> int:
        """Return a MuJoCo geom id by name."""

        return self._name_to_id(self._mujoco.mjtObj.mjOBJ_GEOM, name)

    def get_joint_id(self, name: str) -> int:
        """Return a MuJoCo joint id by name."""

        return self._name_to_id(self._mujoco.mjtObj.mjOBJ_JOINT, name)

    def get_site_id(self, name: str) -> int:
        """Return a MuJoCo site id by name."""

        return self._name_to_id(self._mujoco.mjtObj.mjOBJ_SITE, name)

    def get_camera_id(self, name: str) -> int:
        """Return a MuJoCo camera id by name."""

        return self._name_to_id(self._mujoco.mjtObj.mjOBJ_CAMERA, name)

    def get_end_effector_position(self) -> tuple[float, float, float]:
        """Return the world position of the end-effector site."""

        self._require_loaded()
        site_id = self.get_site_id("end_effector_site")
        pos = self.data.site_xpos[site_id]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def get_object_positions(self) -> dict[str, tuple[float, float, float]]:
        """Return current world positions for all scene objects."""

        self._require_loaded()
        positions: dict[str, tuple[float, float, float]] = {}
        for obj in self.object_specs:
            body_id = self.get_body_id(obj.object_id)
            pos = self.data.xpos[body_id]
            positions[obj.object_id] = (float(pos[0]), float(pos[1]), float(pos[2]))
        return positions

    def get_ground_truth_detections(self) -> list[DetectedObject]:
        """Represent known object positions as fallback detections."""

        detections: list[DetectedObject] = []
        positions = self.get_object_positions()
        for obj in self.object_specs:
            world = positions[obj.object_id]
            pixel = self.world_to_pixel(world)
            detections.append(
                DetectedObject(
                    object_id=obj.object_id,
                    label=obj.label,
                    pixel_center=pixel,
                    world_position=world,
                    confidence=1.0,
                )
            )
        return detections

    def get_ground_truth_bins(self) -> list[DetectedBin]:
        """Represent generated bin positions as fallback detected bins."""

        bins: list[DetectedBin] = []
        for bin_spec in self.bin_specs:
            bins.append(
                DetectedBin(
                    bin_id=bin_spec.bin_id,
                    label=bin_spec.label,
                    pixel_center=self.world_to_pixel(bin_spec.position),
                    world_position=bin_spec.position,
                    orientation_rpy=bin_spec.orientation_rpy,
                    size_xyz=bin_spec.size_xyz,
                    confidence=1.0,
                )
            )
        return bins

    def set_arm_joint_angles(self, angles: JointAngles) -> None:
        """Set the simplified arm joints directly and update kinematics."""

        self._require_loaded()
        values = (angles.base_yaw, angles.shoulder, angles.elbow)
        for joint_name, value in zip(self.arm_joint_names, values, strict=True):
            joint_id = self.get_joint_id(joint_name)
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            qvel_addr = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[qpos_addr] = value
            self.data.qvel[qvel_addr] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        self._update_attached_object()

    def set_object_position(self, object_id: str, position: tuple[float, float, float]) -> None:
        """Move a free-joint object body to the requested world position."""

        self._require_loaded(allow_uninitialized=True)
        joint_name = f"{object_id}_freejoint"
        try:
            joint_id = self.get_joint_id(joint_name)
        except KeyError:
            LOGGER.debug("Ignoring object position update for unknown freejoint %s", joint_name)
            return
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qpos_addr : qpos_addr + 3] = np.array(position, dtype=float)
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self._mujoco.mj_forward(self.model, self.data)

    def attach_nearest_object(
        self,
        pick_position: tuple[float, float, float],
        *,
        preferred_object_id: str,
        max_distance: float = 0.08,
    ) -> str | None:
        """Attach the requested or nearest object if it is close enough."""

        positions = self.get_object_positions()
        if preferred_object_id in positions:
            distance = float(np.linalg.norm(np.array(positions[preferred_object_id]) - np.array(pick_position)))
            if distance <= max_distance:
                self._attached_object_id = preferred_object_id
                return preferred_object_id

        nearest_id: str | None = None
        nearest_distance = float("inf")
        for object_id, position in positions.items():
            distance = float(np.linalg.norm(np.array(position) - np.array(pick_position)))
            if distance < nearest_distance:
                nearest_id = object_id
                nearest_distance = distance
        if nearest_id is not None and nearest_distance <= max_distance:
            self._attached_object_id = nearest_id
            return nearest_id
        return None

    def release_attached_object(self, place_position: tuple[float, float, float]) -> None:
        """Release the logically attached suction object at the place target."""

        if self._attached_object_id is not None:
            self.set_object_position(self._attached_object_id, place_position)
        self._attached_object_id = None

    def world_to_pixel(self, world: tuple[float, float, float]) -> tuple[int, int]:
        """Project an approximate top-down world point into image pixels."""

        workspace = self.config.workspace
        x_norm = (world[0] - workspace.x_min) / (workspace.x_max - workspace.x_min)
        y_norm = (workspace.y_max - world[1]) / (workspace.y_max - workspace.y_min)
        px = int(np.clip(round(x_norm * (self.config.width - 1)), 0, self.config.width - 1))
        py = int(np.clip(round(y_norm * (self.config.height - 1)), 0, self.config.height - 1))
        return (px, py)

    def _update_attached_object(self) -> None:
        if self._attached_object_id is None:
            return
        ee = self.get_end_effector_position()
        object_pos = (
            ee[0],
            ee[1],
            max(self.config.workspace.z_min, ee[2] - self.config.object_half_height),
        )
        self.set_object_position(self._attached_object_id, object_pos)

    def _name_to_id(self, obj_type: object, name: str) -> int:
        self._require_loaded(allow_uninitialized=True)
        obj_id = int(self._mujoco.mj_name2id(self.model, obj_type, name))
        if obj_id < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return obj_id

    def _require_loaded(self, *, allow_uninitialized: bool = False) -> None:
        if self._mujoco is None:
            if allow_uninitialized:
                import mujoco

                self._mujoco = mujoco
            else:
                raise RuntimeError("MuJoCo environment is not loaded")
        if self.model is None or self.data is None:
            raise RuntimeError("MuJoCo model/data are not available")


def object_spec_by_id(objects: list[SceneObject], object_id: str) -> SceneObject | None:
    """Find an object spec by id."""

    return next((obj for obj in objects if obj.object_id == object_id), None)


def bin_spec_by_id(bins: list[SceneBin], bin_id: str) -> SceneBin | None:
    """Find a bin spec by id."""

    return next((item for item in bins if item.bin_id == bin_id), None)
