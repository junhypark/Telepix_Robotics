"""Deterministic MuJoCo XML scene builder."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import numpy as np

from robot_sorting.robot.safety import is_inside_base_exclusion_zone, is_inside_workspace
from robot_sorting.schemas import ObjectLabel, SimulationConfig


@dataclass(frozen=True)
class SceneObject:
    """Ground-truth object specification used to build the MuJoCo scene."""

    object_id: str
    label: ObjectLabel
    position: tuple[float, float, float]
    color: tuple[float, float, float, float]


class SceneBuilder:
    """Build a deterministic educational sorting scene as MuJoCo XML."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.objects = self._generate_objects()

    def build_xml(self) -> str:
        """Return a complete MuJoCo XML model."""

        object_xml = "\n".join(self._object_body_xml(obj) for obj in self.objects)
        object_top = self.config.table_height + self.config.object_half_height
        normal_bin_pos = self._format_xyz((*self.config.normal_bin_position[:2], object_top))
        defect_bin_pos = self._format_xyz((*self.config.defect_bin_position[:2], object_top))
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
    <material name="mat_normal_bin" rgba="0.10 0.65 0.25 0.35"/>
    <material name="mat_defect_bin" rgba="0.95 0.82 0.05 0.35"/>
  </asset>

  <worldbody>
    <light name="key_light" pos="0 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" pos="0 0 -0.001" size="1.5 1.5 0.01" rgba="0.82 0.84 0.82 1"/>

    <body name="table" pos="0.16 0 {self.config.table_height / 2:.6f}">
      <geom name="table_geom" type="box" size="0.72 0.48 {self.config.table_height / 2:.6f}" material="mat_table"/>
    </body>

    <body name="normal_bin" pos="{normal_bin_pos}">
      <geom name="normal_bin_geom" type="box" size="0.075 0.065 0.018" material="mat_normal_bin"/>
    </body>

    <body name="defect_bin" pos="{defect_bin_pos}">
      <geom name="defect_bin_geom" type="box" size="0.075 0.065 0.018" material="mat_defect_bin"/>
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
  </worldbody>

  <actuator>
    <motor name="base_yaw_motor" joint="base_yaw_joint" gear="50"/>
    <motor name="shoulder_motor" joint="shoulder_joint" gear="50"/>
    <motor name="elbow_motor" joint="elbow_joint" gear="50"/>
  </actuator>
</mujoco>
""".strip()

    def _generate_objects(self) -> list[SceneObject]:
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

            label: ObjectLabel = "normal" if len(objects) % 2 == 0 else "defect"
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

    @staticmethod
    def _format_xyz(position: tuple[float, float, float]) -> str:
        return f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}"
