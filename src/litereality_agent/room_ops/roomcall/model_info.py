"""What the room call needs to know about a compiled MuJoCo model, shared by the physics and the
geometry halves so the two can never disagree about which bodies move or which geoms are drawn."""

from __future__ import annotations

import mujoco

# The exporter's convention (`export/mujoco_scene.py`): visual-only geoms are group 2, colliders
# group 3, walls group 1 and the ceiling group 0 — the last two collide AND are seen. A hand-written
# scene usually leaves everything in group 0.
VISUAL_GROUP = 2
COLLISION_GROUP = 3

# The exporter's shake rig: the joints its `room` body rides (`export/mujoco_scene.py`).
RIG_BODY = "room"
RIG_JOINTS = frozenset({"room_x", "room_y", "room_z", "room_yaw"})


def body_name(model: mujoco.MjModel, body: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body{body}"


def servo_held_joints(model: mujoco.MjModel) -> set[int]:
    """Joints a position servo holds at its setpoint.

    An exported room hangs its walls, floor and every fixture on one 50 t `room` body that rides
    slide and hinge joints, each pinned by a stiff position servo: that is the shake rig, and at
    ctrl=0 it is the building standing still. A joint held that way is not something a visitor can
    move — counting it would make the whole building grabbable, and a hand pulling at 50 t would
    shift every wall. A `motor` (what the exporter gives doors and drawers) is not a servo and
    leaves its joint free.
    """
    held = set()
    for a in range(model.nu):
        if (int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
                and int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
                and float(model.actuator_biasprm[a, 1]) < 0):
            held.add(int(model.actuator_trnid[a, 0]))
    return held


def rig_joints(model: mujoco.MjModel) -> set[int]:
    """The shake rig's joints, known by name rather than by their servo.

    Exports made before the exporter gave `room_yaw` its `drive_room_yaw` servo left that one joint
    bare, so by `servo_held_joints` alone the building would count as movable and a visitor pulling
    on a wall could twist all 50 t of it about its vertical. Those joints are the building on every
    export, servo or not, so a joint with one of the rig's names on the body named `room` is held.
    """
    rig = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIG_BODY)
    if rig < 0:
        return set()
    start, count = int(model.body_jntadr[rig]), int(model.body_jntnum[rig])
    return {j for j in range(start, start + count)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) in RIG_JOINTS}


def movable_body_ids(model: mujoco.MjModel) -> set[int]:
    """Bodies that can move: those with a free (not servo-held, not shake-rig) joint of their own, a
    mocap body, or anything below one. The world body (0) and everything welded or pinned to it are
    static and are drawn once, baked in place."""
    held = servo_held_joints(model) | rig_joints(model)
    moving: set[int] = set()
    for b in range(1, model.nbody):
        start, count = int(model.body_jntadr[b]), int(model.body_jntnum[b])
        own = count > 0 and any(j not in held for j in range(start, start + count))
        if own or int(model.body_mocapid[b]) >= 0 or int(model.body_parentid[b]) in moving:
            moving.add(b)
    return moving


def visual_geom_ids(model: mujoco.MjModel) -> list[int]:
    """The geoms to draw: everything but the collider group. A body with nothing else (a hand-written
    scene that put its only geoms in group 3, or a body whose visual mesh was dropped) falls back to
    its colliders, so nothing that collides is ever invisible."""
    drawn = []
    for b in range(model.nbody):
        start, count = int(model.body_geomadr[b]), int(model.body_geomnum[b])
        if start < 0 or count == 0:
            continue
        geoms = range(start, start + count)
        seen = [g for g in geoms if int(model.geom_group[g]) != COLLISION_GROUP]
        drawn.extend(seen or geoms)
    return drawn
