"""Deterministic MuJoCo XML scene builder."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import numpy as np

from robot_sorting.robot.safety import is_inside_base_exclusion_zone, is_inside_workspace
from robot_sorting.schemas import ObjectLabel, SimulationConfig, TargetBin


@dataclass(frozen=True)
class SceneObject:
    """Ground-truth object specification used to build the MuJoCo scene."""

    object_id: str
    label: ObjectLabel
    position: tuple[float, float, float]
    color: tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneBin:
    """Seed-generated target bin specification used to build the MuJoCo scene."""

    bin_id: str
    label: TargetBin
    position: tuple[float, float, float]
    size_xyz: tuple[float, float, float]
    color_rgb: tuple[int, int, int]
    orientation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)


class SceneBuilder:
    """Build a deterministic educational sorting scene as MuJoCo XML."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.objects = self._generate_objects()
        self.bins = self._generate_bins(self.objects)

    def build_xml(self) -> str:
        """Return a complete MuJoCo XML model."""

        object_xml = "\n".join(self._object_body_xml(obj) for obj in self.objects)
        conveyor_xml = self._conveyor_xml() if self.config.conveyor_enabled else ""
        normal_bin = self._bin_by_label("normal_bin")
        defect_bin = self._bin_by_label("defect_bin")
        normal_bin_pos = self._format_xyz(normal_bin.position)
        defect_bin_pos = self._format_xyz(defect_bin.position)
        bin_half_size = self._format_xyz(
            (
                self.config.bin_size_xyz[0] / 2.0,
                self.config.bin_size_xyz[1] / 2.0,
                self.config.bin_size_xyz[2] / 2.0,
            )
        )
        base_size = f"{self.config.base_radius:.6f} {self.config.base_height / 2:.6f}"
        return f"""
<mujoco model="robot_sorting">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 0"/>

  <default>
    <geom friction="1 0.005 0.0001" density="600"/>
    <joint damping="2" armature="0.01"/>
  </default>

  <asset>
    <material name="mat_table" rgba="0.72 0.72 0.68 1"/>
    <material name="mat_robot" rgba="0.70 0.76 0.82 1"/>
    <material name="mat_robot_dark" rgba="0.45 0.50 0.55 1"/>
    <material name="mat_base" rgba="0.28 0.30 0.34 1"/>
    <material name="mat_joint" rgba="0.92 0.95 0.98 1"/>
    <material name="mat_gripper" rgba="0.08 0.08 0.08 1"/>
    <material name="mat_normal" rgba="0.05 0.20 0.95 1"/>
    <material name="mat_defect" rgba="0.95 0.05 0.03 1"/>
    <material name="mat_normal_bin" rgba="0.02 0.18 0.95 1"/>
    <material name="mat_defect_bin" rgba="0.95 0.03 0.03 1"/>
    <material name="mat_conveyor" rgba="0.10 0.11 0.12 1"/>
    <material name="mat_conveyor_guide" rgba="0.58 0.62 0.66 1"/>
    <material name="mat_entry_marker" rgba="0.10 0.70 0.95 0.45"/>
    <material name="mat_inspection_marker" rgba="0.95 0.85 0.05 0.45"/>
    <material name="mat_pick_marker" rgba="0.10 0.95 0.35 0.45"/>
  </asset>

  <worldbody>
    <light name="key_light" pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" pos="0 0 -0.001" size="1.5 1.5 0.01" rgba="0.82 0.84 0.82 1"/>

    <body name="table" pos="0.16 0 {self.config.table_height / 2:.6f}">
      <geom name="table_geom" type="box" size="0.72 0.48 {self.config.table_height / 2:.6f}" material="mat_table"/>
    </body>

    {conveyor_xml}

    <body name="normal_bin" pos="{normal_bin_pos}">
      <geom name="normal_bin_geom" type="box" size="{bin_half_size}" material="mat_normal_bin"/>
    </body>

    <body name="defect_bin" pos="{defect_bin_pos}">
      <geom name="defect_bin_geom" type="box" size="{bin_half_size}" material="mat_defect_bin"/>
    </body>

    <body name="robot_base" pos="0 0 {self.config.base_height / 2:.6f}">
      <geom name="robot_base_geom" type="cylinder" size="{base_size}" material="mat_base"/>
      <geom name="robot_base_column_geom" type="cylinder" pos="0 0 0.030000"
            size="0.060000 0.045000" material="mat_robot_dark"/>
    </body>

    <body name="robot_shoulder" pos="0 0 {self.config.base_height:.6f}">
      <joint name="base_yaw_joint" type="hinge" axis="0 0 1" limited="true"
             range="{self.config.joint_limits.base_min:.6f} {self.config.joint_limits.base_max:.6f}"/>
      <geom name="shoulder_axis_geom" type="capsule" fromto="0 -0.065 0 0 0.065 0"
            size="0.044" material="mat_joint"/>
      <geom name="shoulder_cover_geom" type="box" pos="-0.010 0 0.010"
            size="0.040 0.038 0.044" material="mat_robot"/>
      <body name="robot_upper_link" pos="0 0 0">
        <joint name="shoulder_joint" type="hinge" axis="0 1 0" limited="true"
               range="{self.config.joint_limits.shoulder_min:.6f} {self.config.joint_limits.shoulder_max:.6f}"/>
        <geom name="upper_link_left_rail_geom" type="capsule"
              fromto="0.035 -0.026 0.018 {self.config.link_1 - 0.035:.6f} -0.026 0.018"
              size="0.018" material="mat_robot"/>
        <geom name="upper_link_right_rail_geom" type="capsule"
              fromto="0.035 0.026 0.018 {self.config.link_1 - 0.035:.6f} 0.026 0.018"
              size="0.018" material="mat_robot"/>
        <geom name="upper_link_spine_geom" type="capsule"
              fromto="0.030 0 -0.008 {self.config.link_1 - 0.030:.6f} 0 -0.008"
              size="0.014" material="mat_robot_dark"/>
        <body name="robot_elbow" pos="{self.config.link_1:.6f} 0 0">
          <joint name="elbow_joint" type="hinge" axis="0 1 0" limited="true"
                 range="{self.config.joint_limits.elbow_min:.6f} {self.config.joint_limits.elbow_max:.6f}"/>
          <geom name="elbow_axis_geom" type="capsule" fromto="0 -0.058 0 0 0.058 0"
                size="0.038" material="mat_joint"/>
          <geom name="elbow_cover_geom" type="sphere" size="0.044" material="mat_robot"/>
          <body name="robot_forearm_link" pos="0 0 0">
            <geom name="forearm_left_rail_geom" type="capsule"
                  fromto="0.030 -0.022 -0.010 {self.config.link_2 - 0.040:.6f} -0.022 -0.010"
                  size="0.016" material="mat_robot"/>
            <geom name="forearm_right_rail_geom" type="capsule"
                  fromto="0.030 0.022 -0.010 {self.config.link_2 - 0.040:.6f} 0.022 -0.010"
                  size="0.016" material="mat_robot"/>
            <geom name="forearm_spine_geom" type="capsule"
                  fromto="0.020 0 0.012 {self.config.link_2 - 0.030:.6f} 0 0.012"
                  size="0.012" material="mat_robot_dark"/>
            <body name="end_effector" pos="{self.config.link_2:.6f} 0 0">
              <body name="robot_wrist" pos="0 0 0">
                <geom name="wrist_roll_geom" type="capsule" fromto="-0.025 0 0 0.025 0 0"
                      size="0.026" material="mat_joint"/>
                <geom name="wrist_drop_geom" type="capsule" fromto="0 0 0.050 0 0 0.008"
                      size="0.014" material="mat_robot_dark"/>
                <geom name="gripper_palm_geom" type="box" pos="0 0 0.006"
                      size="0.018 0.040 0.012" material="mat_gripper"/>
                <body name="left_gripper_finger" pos="0 -0.030 -0.014">
                  <geom name="left_gripper_finger_geom" type="box" pos="0 0 0"
                        size="0.010 0.006 0.032" material="mat_gripper"/>
                </body>
                <body name="right_gripper_finger" pos="0 0.030 -0.014">
                  <geom name="right_gripper_finger_geom" type="box" pos="0 0 0"
                        size="0.010 0.006 0.032" material="mat_gripper"/>
                </body>
                <site name="end_effector_site" pos="0 0 0" size="0.018" rgba="0 1 0 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    {object_xml}

    <camera name="{escape(self.config.renderer_camera)}" mode="fixed" pos="0.12 0 1.05"
            xyaxes="1 0 0 0 1 0" fovy="45"/>
    <camera name="inspection_camera" mode="fixed"
            pos="{self.config.inspection_zone_center[0]:.6f} {self.config.inspection_zone_center[1]:.6f} 0.82"
            xyaxes="1 0 0 0 1 0" fovy="38"/>
  </worldbody>

  <actuator>
    <motor name="base_yaw_motor" joint="base_yaw_joint" gear="50"/>
    <motor name="shoulder_motor" joint="shoulder_joint" gear="50"/>
    <motor name="elbow_motor" joint="elbow_joint" gear="50"/>
  </actuator>
</mujoco>
""".strip()

    def _generate_objects(self) -> list[SceneObject]:
        if self.config.object_spawn_positions is not None:
            return self._generate_configured_objects()
        if self.config.conveyor_enabled:
            return self._generate_conveyor_objects()

        rng = np.random.default_rng(self.config.seed)
        objects: list[SceneObject] = []
        attempts = 0
        pick_z = self.config.table_height + self.config.object_half_height
        while len(objects) < self.config.objects and attempts < self.config.objects * 200 + 50:
            attempts += 1
            x = float(rng.uniform(0.17, 0.40))
            y = float(rng.uniform(-0.23, 0.23))
            position = (x, y, pick_z)
            if not is_inside_workspace(position, self.config.workspace):
                continue
            if is_inside_base_exclusion_zone(
                position,
                self.config.base_radius + 0.05,
                0.0,
                self.config.base_height + 0.04,
            ):
                continue
            if any(np.linalg.norm(np.array(position[:2]) - np.array(obj.position[:2])) < 0.075 for obj in objects):
                continue

            label = self._label_for_index(len(objects))
            color = (0.05, 0.20, 0.95, 1.0) if label == "normal" else (0.95, 0.05, 0.03, 1.0)
            objects.append(
                SceneObject(
                    object_id=f"object_{len(objects)}",
                    label=label,
                    position=position,
                    color=color,
                )
            )

        if len(objects) != self.config.objects:
            raise ValueError("Could not place deterministic objects inside the safe workspace")
        return objects

    def _generate_conveyor_objects(self) -> list[SceneObject]:
        objects: list[SceneObject] = []
        entry_x, entry_y, entry_z = self.config.conveyor_entry_position
        spacing = max(self.config.object_radius * 3.0, 0.075)
        for index in range(self.config.objects):
            row = index // 8
            column = index % 8
            x = max(self.config.workspace.x_min + 0.04, entry_x - column * spacing)
            y = entry_y + row * spacing
            position = (x, y, entry_z)
            if not is_inside_workspace(position, self.config.workspace):
                position = (
                    min(max(x, self.config.workspace.x_min + 0.04), self.config.workspace.x_max - 0.04),
                    min(max(y, self.config.workspace.y_min + 0.04), self.config.workspace.y_max - 0.04),
                    entry_z,
                )
            label = self._label_for_index(index)
            color = (0.05, 0.20, 0.95, 1.0) if label == "normal" else (0.95, 0.05, 0.03, 1.0)
            objects.append(
                SceneObject(
                    object_id=f"object_{index}",
                    label=label,
                    position=position,
                    color=color,
                )
            )
        return objects

    def _generate_configured_objects(self) -> list[SceneObject]:
        positions = self.config.object_spawn_positions or ()
        if len(positions) != self.config.objects:
            raise ValueError("Configured scenario object count must match SimulationConfig.objects")
        objects: list[SceneObject] = []
        for index, position in enumerate(positions):
            if not is_inside_workspace(position, self.config.workspace):
                raise ValueError(f"Configured object_{index} is outside the workspace")
            if is_inside_base_exclusion_zone(
                position,
                self.config.base_radius + 0.05,
                0.0,
                self.config.base_height + 0.04,
            ):
                raise ValueError(f"Configured object_{index} is inside the base exclusion zone")
            label = self._label_for_index(index)
            color = (0.05, 0.20, 0.95, 1.0) if label == "normal" else (0.95, 0.05, 0.03, 1.0)
            objects.append(
                SceneObject(
                    object_id=f"object_{index}",
                    label=label,
                    position=position,
                    color=color,
                )
            )
        return objects

    def _generate_bins(self, objects: list[SceneObject]) -> list[SceneBin]:
        rng = np.random.default_rng(self.config.seed + 17_000)
        bins: list[SceneBin] = []
        labels: tuple[TargetBin, ...] = ("normal_bin", "defect_bin")
        colors = {"normal_bin": (5, 45, 242), "defect_bin": (242, 8, 8)}
        half_x = self.config.bin_size_xyz[0] / 2.0
        half_y = self.config.bin_size_xyz[1] / 2.0
        z = self.config.table_height + self.config.bin_size_xyz[2] / 2.0
        for label in labels:
            configured = self._configured_bin_position(label)
            if configured is not None:
                self._validate_bin_position(configured, label, objects, bins)
                bins.append(
                    SceneBin(
                        bin_id=label,
                        label=label,
                        position=configured,
                        size_xyz=self.config.bin_size_xyz,
                        color_rgb=colors[label],
                    )
                )
                continue
            for _ in range(500):
                x = float(rng.uniform(self.config.workspace.x_min + half_x, self.config.workspace.x_max - half_x))
                y = float(rng.uniform(self.config.workspace.y_min + half_y, self.config.workspace.y_max - half_y))
                position = (x, y, z)
                if not self._is_valid_bin_position(position, objects, bins):
                    continue
                bins.append(
                    SceneBin(
                        bin_id=label,
                        label=label,
                        position=position,
                        size_xyz=self.config.bin_size_xyz,
                        color_rgb=colors[label],
                    )
                )
                break
            else:
                raise ValueError(f"Could not place deterministic {label} inside the safe workspace")
        return bins

    def _object_body_xml(self, obj: SceneObject) -> str:
        material = "mat_normal" if obj.label == "normal" else "mat_defect"
        x, y, z = obj.position
        return f"""
    <body name="{escape(obj.object_id)}" pos="{x:.6f} {y:.6f} {z:.6f}">
      <freejoint name="{escape(obj.object_id)}_freejoint"/>
      <geom name="{escape(obj.object_id)}_geom" type="box"
            size="{self.config.object_radius:.6f} {self.config.object_radius:.6f} {self.config.object_half_height:.6f}"
            material="{material}"/>
    </body>
""".rstrip()

    def _conveyor_xml(self) -> str:
        entry = self.config.conveyor_entry_position
        inspection = self.config.inspection_zone_center
        pick = self.config.pick_zone_center
        center_x = (entry[0] + inspection[0]) / 2.0
        center_y = (entry[1] + inspection[1]) / 2.0
        length = max(abs(inspection[0] - entry[0]) + 0.34, 0.45)
        width = 0.13
        belt_z = self.config.table_height + 0.004
        guide_z = self.config.table_height + 0.026
        return f"""
    <body name="conveyor_belt" pos="{center_x:.6f} {center_y:.6f} {belt_z:.6f}">
      <geom name="conveyor_belt_geom" type="box" size="{length / 2.0:.6f} {width / 2.0:.6f} 0.004000"
            material="mat_conveyor"/>
      <geom name="conveyor_left_guide_geom" type="box" pos="0 {width / 2.0 + 0.012:.6f} {guide_z - belt_z:.6f}"
            size="{length / 2.0:.6f} 0.006000 0.018000" material="mat_conveyor_guide"/>
      <geom name="conveyor_right_guide_geom" type="box" pos="0 {-width / 2.0 - 0.012:.6f} {guide_z - belt_z:.6f}"
            size="{length / 2.0:.6f} 0.006000 0.018000" material="mat_conveyor_guide"/>
    </body>
    <body name="conveyor_entry_marker" pos="{entry[0]:.6f} {entry[1]:.6f} {belt_z + 0.004:.6f}">
      <geom name="conveyor_entry_marker_geom" type="box" size="0.018000 0.055000 0.002000"
            material="mat_entry_marker"/>
    </body>
    <body name="inspection_zone_marker" pos="{inspection[0]:.6f} {inspection[1]:.6f} {belt_z + 0.006:.6f}">
      <geom name="inspection_zone_marker_geom" type="box" size="0.026000 0.060000 0.002000"
            material="mat_inspection_marker"/>
    </body>
    <body name="pick_zone_marker" pos="{pick[0]:.6f} {pick[1]:.6f} {belt_z + 0.008:.6f}">
      <geom name="pick_zone_marker_geom" type="box" size="0.020000 0.052000 0.002000"
            material="mat_pick_marker"/>
    </body>
""".rstrip()

    def _bin_by_label(self, label: TargetBin) -> SceneBin:
        return next(item for item in self.bins if item.label == label)

    def _is_reachable(self, position: tuple[float, float, float]) -> bool:
        radial = float(np.hypot(position[0], position[1]))
        reach = self.config.link_1 + self.config.link_2 - self.config.safety_margin
        return radial <= reach

    def _configured_bin_position(self, label: TargetBin) -> tuple[float, float, float] | None:
        if label == "normal_bin":
            return self.config.normal_bin_spawn_position
        return self.config.defect_bin_spawn_position

    def _validate_bin_position(
        self,
        position: tuple[float, float, float],
        label: TargetBin,
        objects: list[SceneObject],
        existing_bins: list[SceneBin],
    ) -> None:
        if not self._is_valid_bin_position(position, objects, existing_bins):
            raise ValueError(f"Configured {label} is not a safe bin position")

    def _is_valid_bin_position(
        self,
        position: tuple[float, float, float],
        objects: list[SceneObject],
        existing_bins: list[SceneBin],
    ) -> bool:
        half_x = self.config.bin_size_xyz[0] / 2.0
        half_y = self.config.bin_size_xyz[1] / 2.0
        return (
            is_inside_workspace(position, self.config.workspace)
            and self._is_reachable(position)
            and not is_inside_base_exclusion_zone(
                position,
                self.config.base_radius + self.config.safety_margin + max(half_x, half_y),
                0.0,
                self.config.base_height + self.config.safety_margin,
            )
            and not any(self._overlaps_object(position, obj.position) for obj in objects)
            and not any(self._overlaps_bin(position, existing.position) for existing in existing_bins)
        )

    def _label_for_index(self, index: int) -> ObjectLabel:
        labels = self.config.object_label_sequence
        if labels:
            return labels[index % len(labels)]
        return "normal" if index % 2 == 0 else "defect"

    def _overlaps_object(
        self,
        bin_position: tuple[float, float, float],
        object_position: tuple[float, float, float],
    ) -> bool:
        bin_half_x = self.config.bin_size_xyz[0] / 2.0
        bin_half_y = self.config.bin_size_xyz[1] / 2.0
        margin = self.config.object_radius + self.config.bin_placement.object_spacing_margin_meters
        return (
            abs(bin_position[0] - object_position[0]) <= bin_half_x + margin
            and abs(bin_position[1] - object_position[1]) <= bin_half_y + margin
        )

    def _overlaps_bin(
        self,
        candidate: tuple[float, float, float],
        existing: tuple[float, float, float],
    ) -> bool:
        min_x_gap = self.config.bin_size_xyz[0] + self.config.bin_placement.object_spacing_margin_meters
        min_y_gap = self.config.bin_size_xyz[1] + self.config.bin_placement.object_spacing_margin_meters
        return abs(candidate[0] - existing[0]) <= min_x_gap and abs(candidate[1] - existing[1]) <= min_y_gap

    @staticmethod
    def _format_xyz(position: tuple[float, float, float]) -> str:
        return f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}"
