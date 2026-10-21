"""Which bodies the room call lets people move, and which geoms it draws, on a scene shaped like the
MuJoCo exporter's: walls and fixtures hang on a 50 t `room` body pinned by position servos (the
shake rig), doors and drawers carry motors, and visuals, walls and colliders sit in separate groups."""

from __future__ import annotations

import mujoco
import numpy as np

from litereality_agent.room_ops.roomcall.model_info import (
    body_name,
    movable_body_ids,
    servo_held_joints,
    visual_geom_ids,
)
from litereality_agent.room_ops.roomcall.sim import RoomSim

EXPORT_SHAPED = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <body name="room" gravcomp="1">
      <inertial pos="0 0 0" mass="50000" diaginertia="50000 50000 50000"/>
      <joint name="room_x" type="slide" axis="1 0 0" limited="false" damping="200"/>
      <joint name="room_y" type="slide" axis="0 1 0" limited="false" damping="200"/>
      <joint name="room_z" type="slide" axis="0 0 1" limited="false" damping="200"/>
      <joint name="room_yaw" type="hinge" axis="0 0 1" limited="false" damping="400"/>
      <geom name="floor" type="box" size="3 3 0.05" pos="0 0 -0.05" group="2"/>
      <geom name="wall_n" type="box" size="3 0.05 1.25" pos="0 3 1.25" group="1"/>
      <geom name="ceiling" type="box" size="3 3 0.05" pos="0 0 2.55"/>
      <body name="Cabinet0" pos="1 1 0.4">
        <geom type="box" size="0.3 0.3 0.4" group="2" contype="0" conaffinity="0"/>
        <geom type="box" size="0.3 0.3 0.4" group="3"/>
        <body name="Cabinet0_door" pos="0 -0.31 0">
          <joint name="Cabinet0_hinge" type="hinge" axis="0 0 1" range="0 1.6" damping="0.5"/>
          <geom type="box" size="0.3 0.01 0.4" group="2" contype="0" conaffinity="0"/>
          <geom type="box" size="0.3 0.01 0.4" group="3"/>
        </body>
      </body>
    </body>
    <body name="Chair0" pos="-1 0 0.3">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.2" group="3" mass="5"/>
    </body>
  </worldbody>
  <actuator>
    <position name="drive_room_x" joint="room_x" kp="5e7" kv="3.16e6"/>
    <position name="drive_room_y" joint="room_y" kp="5e7" kv="3.16e6"/>
    <position name="drive_room_z" joint="room_z" kp="5e7" kv="3.16e6"/>
    <position name="drive_room_yaw" joint="room_yaw" kp="5e7" kv="3.16e6"/>
    <motor name="act_Cabinet0_hinge" joint="Cabinet0_hinge" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(EXPORT_SHAPED)


def _id(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def test_the_servo_held_room_and_its_fixtures_are_static():
    model = _model()
    assert {body_name(model, b) for b in movable_body_ids(model)} == {"Cabinet0_door", "Chair0"}
    # The door's motor is not a servo: only the four room drives pin their joints.
    held = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in servo_held_joints(model)}
    assert held == {"room_x", "room_y", "room_z", "room_yaw"}


def test_an_export_from_before_the_yaw_servo_still_has_a_static_building():
    # Older exports gave room_yaw no drive_room_yaw servo; the rig is still the building.
    old = EXPORT_SHAPED.replace(
        '<position name="drive_room_yaw" joint="room_yaw" kp="5e7" kv="3.16e6"/>', "")
    assert old != EXPORT_SHAPED
    model = mujoco.MjModel.from_xml_string(old)
    held = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in servo_held_joints(model)}
    assert "room_yaw" not in held
    assert {body_name(model, b) for b in movable_body_ids(model)} == {"Cabinet0_door", "Chair0"}
    sim = RoomSim(model, settle_seconds=0.2)
    room = _id(model, "room")
    assert not sim.grab("p1", room, [0, 2.95, 1.25])  # a point on the north wall
    assert not sim.grab("p1", _id(model, "Cabinet0"), [1, 1, 0.4])
    # The names only count on the rig itself: a joint called room_x elsewhere is just a joint.
    elsewhere = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="slider"><joint name="room_x" type="slide"/>'
        '<geom size="0.1"/></body></worldbody></mujoco>')
    assert movable_body_ids(elsewhere) == {1}


def test_walls_and_ceiling_are_drawn_and_colliders_only_as_a_fallback():
    model = _model()
    drawn = visual_geom_ids(model)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in drawn}
    assert {"floor", "wall_n", "ceiling"} <= names
    groups = {body_name(model, int(model.geom_bodyid[g])): set() for g in drawn}
    for g in drawn:
        groups[body_name(model, int(model.geom_bodyid[g]))].add(int(model.geom_group[g]))
    assert groups["Cabinet0"] == {2} and groups["Cabinet0_door"] == {2}
    assert groups["Chair0"] == {3}  # a body with only colliders is still seen


def test_nobody_can_grab_the_building():
    sim = RoomSim(_model(), settle_seconds=0.2)
    assert not sim.grab("p1", _id(sim.model, "room"), [0, 0, 1])
    assert not sim.grab("p1", _id(sim.model, "Cabinet0"), [1, 1, 0.4])
    assert sim.grab("p1", _id(sim.model, "Cabinet0_door"), [1.2, 0.69, 0.4])
    sim.move("p1", [1.2, 0.2, 0.4])
    room = _id(sim.model, "room")
    start = sim.data.xpos[room].copy()
    sim.step_realtime(1.0)
    assert np.linalg.norm(sim.data.xpos[room] - start) < 1e-3
