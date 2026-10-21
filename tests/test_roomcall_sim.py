"""The room call's shared physics: the demo room is stable, grabs move free bodies, doors and
drawers, nothing a peer sends can break the simulation, and NaN never reaches a snapshot."""

from __future__ import annotations

import json
import math
import threading

import mujoco
import numpy as np
import pytest

from litereality_agent.room_ops.roomcall.demo import write_demo_scene
from litereality_agent.room_ops.roomcall.model_info import movable_body_ids
from litereality_agent.room_ops.roomcall.sim import RoomSim, load_model


@pytest.fixture(scope="module")
def model(tmp_path_factory) -> mujoco.MjModel:
    return load_model(write_demo_scene(tmp_path_factory.mktemp("demo")))


@pytest.fixture()
def sim(model) -> RoomSim:
    return RoomSim(model, settle_seconds=0.5)


def _body(model, name: str) -> int:
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert b >= 0, name
    return b


def _world(sim: RoomSim, body: int, local) -> np.ndarray:
    d = sim.data
    return d.xpos[body] + d.xmat[body].reshape(3, 3) @ np.asarray(local, dtype=float)


def _qpos(sim: RoomSim, body: int) -> float:
    m = sim.model
    return float(sim.data.qpos[m.jnt_qposadr[m.body_jntadr[body]]])


def _anchor(sim: RoomSim, owner: str) -> np.ndarray:
    grab = next(g for g in sim.snapshot()["grabs"] if g[0] == owner)
    return np.array(grab[2:5])


def test_demo_scene_has_what_the_room_call_exercises(model):
    walls = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
    assert sum(1 for n in walls if n and n.startswith("wall")) >= 4
    kinds = [int(t) for t in model.jnt_type]
    assert kinds.count(int(mujoco.mjtJoint.mjJNT_FREE)) >= 2
    assert int(mujoco.mjtJoint.mjJNT_HINGE) in kinds
    assert int(mujoco.mjtJoint.mjJNT_SLIDE) in kinds
    assert _body(model, "table") not in movable_body_ids(model)


def test_demo_scene_loads_and_stays_finite_for_ten_seconds(model):
    sim = RoomSim(model)
    start = {b[0]: np.array(b[1:4]) for b in sim.snapshot()["bodies"]}
    assert sim.step_realtime(10.0, max_steps=10_000) == round(10.0 / sim.timestep)
    assert np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel))
    assert sim.unstable_resets == 0 and sim.respawns == 0
    for b in sim.snapshot()["bodies"]:
        assert np.linalg.norm(np.array(b[1:4]) - start[b[0]]) < 0.01, b
    assert float(np.abs(sim.data.qvel).max()) < 0.05


def test_grab_and_move_pulls_a_free_object_to_its_target_and_it_floats(sim, model):
    ball = _body(model, "ball")
    top = sim.data.xpos[ball] + [0.0, 0.0, 0.11]
    assert sim.grab("ann", ball, top)
    target = top + [0.3, 0.2, 0.6]
    sim.move("ann", target)
    sim.step_realtime(1.5, max_steps=1000)
    assert np.linalg.norm(_anchor(sim, "ann") - target) < 0.02
    z_held = float(sim.data.xpos[ball][2])
    assert z_held > 0.5
    sim.step_realtime(1.0, max_steps=1000)          # still held: it hovers, it does not sag
    assert abs(float(sim.data.xpos[ball][2]) - z_held) < 0.01
    assert np.linalg.norm(_anchor(sim, "ann") - target) < 0.02


def test_grabbing_the_room_door_swings_it_on_its_hinge(sim, model):
    door = _body(model, "room_door")
    assert _qpos(sim, door) == pytest.approx(0.0, abs=1e-3)
    handle = _world(sim, door, [-0.06, 0.80, 1.0])
    assert sim.grab("bob", door, handle)
    sim.move("bob", handle + [-0.5, 0.0, 0.0])      # pull the handle into the room
    sim.step_realtime(1.5, max_steps=1000)
    assert math.degrees(_qpos(sim, door)) > 20.0
    # A hinge only rotates: the door's frame stays on its hinge line.
    assert np.allclose(sim.data.xpos[door], [2.45, 0.3, 0.0], atol=1e-6)


def test_grabbing_the_cabinet_door_and_drawer(sim, model):
    door, drawer = _body(model, "cabinet_door"), _body(model, "cabinet_drawer")
    knob = _world(sim, door, [0.70, -0.04, 0.605])
    handle = _world(sim, drawer, [0.0, -0.255, 0.24])
    assert sim.grab("cat", door, knob)
    assert sim.grab("dan", drawer, handle)
    sim.move("cat", knob + [0.0, -0.35, 0.0])
    sim.move("dan", handle + [0.0, -0.25, 0.0])
    sim.step_realtime(1.5, max_steps=1000)
    assert math.degrees(_qpos(sim, door)) < -15.0  # opens outwards (negative about +z)
    assert _qpos(sim, drawer) > 0.15                # slides out along its rail
    assert len(sim.snapshot()["grabs"]) == 2


def test_release_lets_a_held_object_fall(sim, model):
    mug = _body(model, "mug")
    rest_z = float(sim.data.xpos[mug][2])
    point = sim.data.xpos[mug].copy()
    assert sim.grab("eve", mug, point)
    sim.move("eve", point + [0.0, 0.0, 0.5])
    sim.step_realtime(1.0, max_steps=1000)
    assert float(sim.data.xpos[mug][2]) > rest_z + 0.4
    sim.release("eve")
    assert sim.snapshot()["grabs"] == []
    sim.step_realtime(1.0, max_steps=1000)
    assert float(sim.data.xpos[mug][2]) == pytest.approx(rest_z, abs=0.01)  # back on the table


@pytest.mark.parametrize("body", [0, -1, 10_000, True, None, "4", 4.5, "table"])
def test_grab_rejects_static_or_bad_bodies(sim, model, body):
    if body == "table":
        body = _body(model, "table")
    assert not sim.grab("fay", body, [0.0, 0.0, 1.0])
    assert sim.snapshot()["grabs"] == []


@pytest.mark.parametrize("point", [[0, 0, float("nan")], [float("inf"), 0, 0], [0, 0], "abc",
                                   None, [[1, 2, 3]] * 2])
def test_grab_rejects_bad_points(sim, model, point):
    assert not sim.grab("gus", _body(model, "ball"), point)
    assert sim.snapshot()["grabs"] == []


def test_move_is_ignored_without_a_grab_or_with_a_bad_target_and_grabs_replace(sim, model):
    ball, mug = _body(model, "ball"), _body(model, "mug")
    sim.move("hal", [0.0, 0.0, 1.0])                 # holds nothing: ignored
    assert sim.snapshot()["grabs"] == []
    assert sim.grab("hal", ball, sim.data.xpos[ball])
    before = sim.snapshot()["grabs"][0][5:]
    for bad in ([float("nan")] * 3, [1.0, 2.0], None, "x"):
        sim.move("hal", bad)
    assert sim.snapshot()["grabs"][0][5:] == before
    sim.move("hal", [1e9, -1e9, 1e9])               # absurd target: clipped to the room
    assert all(abs(v) < 10 for v in sim.snapshot()["grabs"][0][5:])
    assert sim.grab("hal", mug, sim.data.xpos[mug])  # a second grab replaces the first
    grabs = sim.snapshot()["grabs"]
    assert [g[:2] for g in grabs] == [["hal", mug]]
    sim.release("nobody")                            # releasing nothing is harmless
    sim.step_realtime(0.5, max_steps=1000)
    assert np.all(np.isfinite(sim.data.qpos))


def test_snapshot_lists_only_movable_bodies_and_is_json(sim, model):
    assert sim.grab("ivy", _body(model, "ball"), sim.data.xpos[_body(model, "ball")])
    snap = sim.snapshot()
    assert [b[0] for b in snap["bodies"]] == sim.movable == sorted(movable_body_ids(model))
    assert _body(model, "table") not in sim.movable and 0 not in sim.movable
    assert all(len(b) == 8 for b in snap["bodies"]) and all(len(g) == 8 for g in snap["grabs"])
    assert json.loads(json.dumps(snap)) == snap
    assert isinstance(snap["time"], float) and snap["grabs"][0][0] == "ivy"
    for b in snap["bodies"]:
        assert np.linalg.norm(b[4:8]) == pytest.approx(1.0, abs=1e-4)
        assert all(round(v, 5) == v for v in b[1:])


def test_reset_restores_positions_and_drops_grabs(sim, model):
    initial = sim.snapshot()
    chair = _body(model, "chair_1")
    seat = _world(sim, chair, [0.0, 0.0, 0.45])
    assert sim.grab("jo", chair, seat)
    sim.move("jo", seat + [0.8, 0.3, 0.4])
    sim.step_realtime(1.0, max_steps=1000)
    assert sim.snapshot()["bodies"] != initial["bodies"]
    sim.reset()
    assert sim.snapshot() == initial
    sim.step_realtime(0.2, max_steps=1000)           # and it carries on from there normally
    assert np.all(np.isfinite(sim.data.qpos)) and sim.unstable_resets == 0


def test_step_realtime_clamps_and_carries_fractions(model):
    sim = RoomSim(model, settle_seconds=0.0)
    dt = sim.timestep
    assert sim.step_realtime(100.0) == 400           # a stall is clamped, not caught up
    assert sim.data.time == pytest.approx(400 * dt)
    assert sim.step_realtime(1e9, max_steps=7) == 7
    for bad in (0.0, -1.0, float("nan"), float("inf"), None, "x"):
        assert sim.step_realtime(bad) == 0
    assert sim.step_realtime(1.5 * dt) == 1          # half a step is owed...
    assert sim.step_realtime(1.5 * dt) == 2          # ...and paid on the next call


@pytest.mark.parametrize("field,value", [("qvel", float("nan")), ("qpos", float("nan")),
                                         ("qvel", 1e6)])
def test_unstable_state_is_reset_instead_of_broadcast(sim, model, field, value, tmp_path,
                                                      monkeypatch):
    monkeypatch.chdir(tmp_path)                      # MuJoCo logs its warning to ./MUJOCO_LOG.TXT
    initial = sim.snapshot()
    assert sim.grab("kim", _body(model, "ball"), sim.data.xpos[_body(model, "ball")])
    getattr(sim.data, field)[0] = value              # what a blow-up looks like
    sim.step_realtime(0.05, max_steps=100)
    assert sim.unstable_resets == 1
    snap = sim.snapshot()
    assert snap == initial                           # back to the settled room, grabs dropped
    assert all(math.isfinite(v) for b in snap["bodies"] for v in b[1:])
    sim.step_realtime(0.1, max_steps=100)
    assert sim.unstable_resets == 1


def test_every_reset_mujoco_makes_on_its_own_is_caught(sim, model, tmp_path, monkeypatch):
    # On a bad qacc MuJoCo resets the data itself and carries on from the authored pose. Its
    # warning counter is zeroed by that reset and then incremented, so after one recovery a
    # second blow-up used to read as "no new warning" and the room ran on from qpos0.
    monkeypatch.chdir(tmp_path)                      # MuJoCo logs its warning to ./MUJOCO_LOG.TXT
    initial = sim.snapshot()
    ball = _body(model, "ball")
    for k in range(3):
        assert sim.grab("kim", ball, sim.data.xpos[ball])
        sim.data.xfrc_applied[ball, 2] = 1e30        # one step's qacc is absurd
        sim.step_realtime(0.02, max_steps=100)
        assert sim.unstable_resets == k + 1
        assert sim.snapshot() == initial             # settled room, grabs dropped


ESCAPE_SCENE = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="box" size="1.5 1.5 0.05" pos="0 0 -0.05"/>
    <geom name="wall" type="box" size="0.05 1.5 1" pos="-1.5 0 1"/>
    <body name="mug" pos="0 0 0.05">
      <freejoint/><geom type="box" size="0.04 0.04 0.05" mass="0.3"/>
    </body>
    <body name="door" pos="1 1 0.5">
      <joint name="hinge" type="hinge" axis="0 0 1" range="-90 90" damping="2"/>
      <geom type="box" size="0.3 0.02 0.4" pos="0.3 0 0" mass="3"/>
    </body>
  </worldbody>
</mujoco>"""


def test_an_object_carried_out_of_the_room_comes_back_alone():
    # An exported room's floor is a finite box with doorways in its walls: what is dropped outside
    # falls forever. It must come back by itself, not take the room down with it when its speed
    # finally reads as a blow-up.
    sim = RoomSim(mujoco.MjModel.from_xml_string(ESCAPE_SCENE), settle_seconds=0.3)
    mug, door = _body(sim.model, "mug"), _body(sim.model, "door")
    settled = next(row for row in sim.snapshot()["bodies"] if row[0] == mug)
    hinge = int(sim.model.jnt_qposadr[sim.model.body_jntadr[door]])
    sim.data.qpos[hinge] = 0.8                       # someone left the door open
    assert sim.grab("ann", mug, sim.data.xpos[mug])
    sim.move("ann", [3.0, 0.0, 0.5])                 # out past the floor's edge (x = 1.5)
    sim.step_realtime(1.5, max_steps=1000)
    assert sim.data.xpos[mug][0] > 2.5 and sim.respawns == 0  # held at arm's length: not caught
    sim.release("ann")
    sim.step_realtime(2.5, max_steps=2000)           # it falls into nothing
    assert sim.respawns == 1 and sim.unstable_resets == 0
    row = next(row for row in sim.snapshot()["bodies"] if row[0] == mug)
    assert np.allclose(row[1:], settled[1:], atol=1e-3)
    assert _qpos(sim, door) == pytest.approx(0.8, abs=1e-3)  # the rest of the room is untouched
    assert sim.data.time > 3.9                       # and it was not reset

    # Something that gets out while held is put back and let go of.
    assert sim.grab("ann", mug, sim.data.xpos[mug])
    sim.data.qpos[int(sim.model.jnt_qposadr[sim.model.body_jntadr[mug]]) + 2] = -20.0
    sim.step_realtime(0.01, max_steps=10)
    assert sim.respawns == 2 and sim.snapshot()["grabs"] == []
    assert float(sim.data.xpos[mug][2]) > 0.0


def test_nothing_heavier_than_a_hand_can_carry_is_grabbable():
    # A backstop for building-sized bodies the static test missed: pulling one shifts every wall.
    xml = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body name="vault" pos="0 0 1"><freejoint/><geom type="box" size="1 1 1" mass="20000"/>
          <body name="handle" pos="0 0 1.1"><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
        </body>
        <body name="crate" pos="3 0 0.2"><freejoint/><geom type="box" size="0.2 0.2 0.2" mass="10"/></body>
      </worldbody>
    </mujoco>"""
    sim = RoomSim(mujoco.MjModel.from_xml_string(xml), settle_seconds=0.1)
    vault, handle, crate = (_body(sim.model, n) for n in ("vault", "handle", "crate"))
    assert {vault, handle, crate} <= set(sim.movable)  # still drawn and streamed as moving
    assert not sim.grab("bo", vault, sim.data.xpos[vault])
    assert not sim.grab("bo", handle, sim.data.xpos[handle])  # it would move the whole vault
    assert sim.grab("bo", crate, sim.data.xpos[crate])


def test_settle_survives_a_bad_scene_without_touching_the_shared_model():
    # Two boxes authored 5 cm inside each other and inside the floor.
    xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <geom name="wall_a" type="box" pos="0 1 0.5" size="1 0.05 0.5"/>
        <body name="a" pos="0 0 0.05"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>
        <body name="b" pos="0.05 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    gravity = model.opt.gravity.copy()
    sim = RoomSim(model)
    assert np.array_equal(model.opt.gravity, gravity)
    assert np.all(np.isfinite(sim.data.qpos))
    assert sim.step_realtime(1.0, max_steps=1000) == 500
    assert all(math.isfinite(v) for b in sim.snapshot()["bodies"] for v in b[1:])

    static = RoomSim(mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>"))
    assert static.movable == [] and static.snapshot()["bodies"] == []
    assert static.step_realtime(0.1) == 50


def test_a_thin_rod_held_at_its_tip_stays_stable():
    # The angular damping must be per axis: a scalar sized for a rod's long axes would make its
    # thin axis blow up in a few steps.
    xml = """
    <mujoco>
      <option timestep="0.002" integrator="implicitfast"/>
      <worldbody>
        <geom type="plane" size="2 2 0.1"/>
        <body name="pencil" pos="0 0 0.0045" euler="0 90 0">
          <freejoint/><geom type="capsule" size="0.004 0.08" mass="0.01"/>
        </body>
      </worldbody>
    </mujoco>"""
    sim = RoomSim(mujoco.MjModel.from_xml_string(xml), settle_seconds=0.2)
    tip = _world(sim, 1, [0.0, 0.0, 0.084])
    assert sim.grab("pen", 1, tip)
    sim.move("pen", tip + [0.2, 0.1, 0.5])
    sim.step_realtime(1.5, max_steps=1000)
    assert sim.unstable_resets == 0 and np.all(np.isfinite(sim.data.qvel))
    assert np.linalg.norm(_anchor(sim, "pen") - (tip + [0.2, 0.1, 0.5])) < 0.02
    assert float(np.abs(sim.data.qvel).max()) < 0.5


def test_public_methods_are_safe_to_call_from_another_thread(sim, model):
    ball = _body(model, "ball")
    stop = threading.Event()

    def hands():
        k = 0
        while not stop.is_set():
            sim.grab("lee", ball, sim.snapshot()["bodies"][0][1:4])
            sim.move("lee", [0.0, 0.0, 1.0 + 0.1 * (k % 5)])
            if k % 7 == 0:
                sim.release("lee")
            k += 1

    worker = threading.Thread(target=hands)
    worker.start()
    try:
        for _ in range(40):
            sim.step_realtime(0.01)
    finally:
        stop.set()
        worker.join()
    assert np.all(np.isfinite(sim.data.qpos))
