"""Cut a room splat into the static room and one splat per movable body.

A splat belongs to the movable body whose geometry it lies on: for every geom of every movable
body (`model_info.movable_body_ids`), the distance from the splat to the geom's own bounding box
(MuJoCo's ``geom_aabb``, in the geom's frame) is measured, and the nearest body within `MARGIN`
takes it. Per geom rather than per body, so the chairs tucked under a table stay chairs: the space
under the table top is no geom of the table. Nothing below a body's lowest point (plus `FLOOR`)
is taken, so the floor under an object stays with the room.

Each body's splats are stored in that body's own frame at the scene's initial pose (``qpos0``,
the scanned position), so a browser that hangs them under the body's node sees them move with it.
The room keeps everything else, in the room's frame.

RoomPlan boxes chairs loosely (20-30 cm off is common next to a table), so before cutting, every
free-moving body is SNAPPED to the splat (`snap`): its box is slid up to `SNAP_SHIFT` and turned
up to `SNAP_YAW` to where it holds the most splats that belong to nothing static (not floor, wall
or table). The correction is returned (`Split.corrections`) so the physics can be moved to match
(`corrected_scene`): then the real chair is cut where it really is, and the body it hangs on is
there too.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from litereality_agent.models.splat.spz import Splats, quat_multiply
from litereality_agent.room_ops.roomcall.model_info import body_name, movable_body_ids

MARGIN = 0.02          # metres around a geom's box that still count as on it
FLOOR = 0.015          # metres above a body's lowest point before anything counts as the body
CHUNK = 200_000
SNAP_SHIFT = 0.40      # metres a box may slide when snapped to the splat
SNAP_STEP = 0.02
SNAP_YAWS = (-10.0, -5.0, 0.0, 5.0, 10.0)   # degrees
SNAP_GAIN = 1.30       # a snap must hold this many times the splats the scanned box holds ...
SNAP_MIN_GAIN = 150    # ... and this many more
STATIC_MARGIN = 0.05       # splats this close to static furniture never pull a box


@dataclass
class Split:
    room: Splats
    bodies: dict[str, Splats] = field(default_factory=dict)       # name -> splats, body frame
    counts: dict[str, int] = field(default_factory=dict)
    # name -> {"dx", "dy", "dyaw_deg", "before", "after"}: snapped free bodies (see `snap`)
    corrections: dict[str, dict] = field(default_factory=dict)


def _geom_boxes(model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]):
    """Per movable geom: (body, world rotation, world centre of its aabb, half extents)."""
    out = []
    for b in bodies:
        start, count = int(model.body_geomadr[b]), int(model.body_geomnum[b])
        for g in range(start, start + count):
            if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            aabb = np.asarray(model.geom_aabb[g], dtype=np.float64)
            R = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
            centre = np.asarray(data.geom_xpos[g]) + R @ aabb[:3]
            out.append((b, R, centre, aabb[3:]))
    return out


def _free_joint(model: mujoco.MjModel, body: int) -> int | None:
    start, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
    for j in range(start, start + count):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return j
    return None


def _inside_any(p: np.ndarray, boxes, margin: float) -> np.ndarray:
    out = np.zeros(len(p), bool)
    for _b, R, c, half in boxes:
        ext = np.abs(R) @ half + margin
        near = np.all(np.abs(p - c) <= ext, axis=1)
        if near.any():
            local = (p[near] - c) @ R
            out[np.flatnonzero(near)[np.all(np.abs(local) <= half + margin, axis=1)]] = True
    return out


def snap(pos: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData, body: int,
         static_boxes, *, shift: float = SNAP_SHIFT, step: float = SNAP_STEP,
         yaws=SNAP_YAWS) -> tuple[float, float, float, int, int]:
    """Where body's box holds the most splats: (dx, dy, dyaw_deg, splats before, after). The box
    is the body's geoms' bounds in its own frame; it turns about the body's origin and slides in
    the room's XY. Splats inside anything static (walls, the table) or at floor level never
    count, so a chair is never snapped into the table it stands at."""
    start, count = int(model.body_geomadr[body]), int(model.body_geomnum[body])
    Rb = np.asarray(data.xmat[body]).reshape(3, 3)
    xb = np.asarray(data.xpos[body], dtype=np.float64)
    corners = []
    for g in range(start, start + count):
        aabb = np.asarray(model.geom_aabb[g], dtype=np.float64)
        Rg = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        cg = np.asarray(data.geom_xpos[g]) + Rg @ aabb[:3]
        for sgn in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
            corners.append(Rb.T @ (cg + Rg @ (aabb[3:] * sgn) - xb))
    corners = np.asarray(corners)
    lo, hi = corners.min(0), corners.max(0)
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    half_xy = np.maximum(half[:2] - 0.02, 0.05)
    bottom, top = xb[2] + lo[2], xb[2] + hi[2]
    reach = float(np.hypot(*half[:2]) + shift + 0.05)
    near = (np.hypot(pos[:, 0] - xb[0], pos[:, 1] - xb[1]) < reach) \
        & (pos[:, 2] > bottom + 0.04) & (pos[:, 2] < top + 0.02)
    cand = pos[near]
    if len(cand):
        cand = cand[~_inside_any(cand, static_boxes, STATIC_MARGIN)]
    if len(cand) < 50:
        return 0.0, 0.0, 0.0, 0, 0
    yaw0 = math.atan2(Rb[1, 0], Rb[0, 0])
    n = int(round(shift / step))
    best = (0.0, 0.0, 0.0, -1)
    base = 0
    cell = 0.01
    for dyaw in yaws:
        a = yaw0 + math.radians(dyaw)
        ca, sa = math.cos(a), math.sin(a)
        rel = cand[:, :2] - xb[:2]
        u = ca * rel[:, 0] + sa * rel[:, 1] - centre[0]        # into the turned box's frame
        v = -sa * rel[:, 0] + ca * rel[:, 1] - centre[1]
        lim = shift + float(max(half_xy)) + cell
        bins = int(math.ceil(2 * lim / cell))
        H, _, _ = np.histogram2d(u, v, bins=bins, range=[[-lim, lim], [-lim, lim]])
        integral = np.pad(H.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        hx, hy = int(round(half_xy[0] / cell)), int(round(half_xy[1] / cell))
        mid = bins // 2
        for i in range(-n, n + 1):
            for k in range(-n, n + 1):
                cx = mid + int(round(i * step / cell))
                cy = mid + int(round(k * step / cell))
                x0, x1 = max(cx - hx, 0), min(cx + hx, bins)
                y0, y1 = max(cy - hy, 0), min(cy + hy, bins)
                c = int(integral[x1, y1] - integral[x0, y1] - integral[x1, y0] + integral[x0, y0])
                if dyaw == 0.0 and i == 0 and k == 0:
                    base = c
                if c > best[3]:
                    # the slide is in the turned box's frame: back into the room's
                    su, sv = i * step, k * step
                    best = (ca * su - sa * sv, sa * su + ca * sv, dyaw, c)
    dx, dy, dyaw, after = best
    if after < base * SNAP_GAIN or after < base + SNAP_MIN_GAIN:
        return 0.0, 0.0, 0.0, base, base
    return dx, dy, dyaw, base, after


def corrected_scene(scene_xml: Path, corrections: dict[str, dict],
                    out_name: str = "scene_splat.xml") -> Path | None:
    """`scene_xml` with every snapped body moved to where the splat says it is: written beside it
    (so its relative mesh and texture paths still resolve) as `out_name`; None if nothing moved.
    Only top-level bodies are moved (a free body's pos is then its room position)."""
    if not corrections:
        return None
    import xml.etree.ElementTree as ET

    tree = ET.parse(scene_xml)
    world = tree.getroot().find("worldbody")
    moved = 0
    for body in (world.findall("body") if world is not None else []):
        fix = corrections.get(body.get("name", ""))
        if not fix:
            continue
        pos = np.array([float(v) for v in re.split(r"\s+", body.get("pos", "0 0 0").strip())])
        pos[:2] += (fix["dx"], fix["dy"])
        q = np.array([float(v) for v in re.split(r"\s+", body.get("quat", "1 0 0 0").strip())])
        h = math.radians(fix["dyaw_deg"]) / 2
        w1, z1 = math.cos(h), math.sin(h)          # about world Z, applied after the old turn
        w0, x0, y0, z0 = q
        q = np.array([w1 * w0 - z1 * z0, w1 * x0 - z1 * y0, w1 * y0 + z1 * x0, w1 * z0 + z1 * w0])
        body.set("pos", " ".join(f"{v:.5g}" for v in pos))
        body.set("quat", " ".join(f"{v:.6g}" for v in q / np.linalg.norm(q)))
        moved += 1
    if not moved:
        return None
    out = Path(scene_xml).with_name(out_name)
    tree.write(out, encoding="unicode")
    return out


def _quat_xyzw(mj_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = mj_wxyz
    return np.array([x, y, z, w], dtype=np.float64)


def split(splats: Splats, model: mujoco.MjModel, *, margin: float = MARGIN,
          floor: float = FLOOR, snap_bodies: bool = True) -> Split:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    movable = sorted(movable_body_ids(model))
    corrections: dict[str, dict] = {}
    if snap_bodies and len(splats):
        static = [b for b in range(1, model.nbody) if b not in movable]
        static_boxes = _geom_boxes(model, data, static)
        pos64 = splats.positions.astype(np.float64)
        for b in movable:
            j = _free_joint(model, b)
            if j is None or int(model.body_parentid[b]) != 0:
                continue
            dx, dy, dyaw, before, after = snap(pos64, model, data, b, static_boxes)
            if dx == dy == dyaw == 0.0:
                continue
            adr = int(model.jnt_qposadr[j])
            data.qpos[adr] += dx
            data.qpos[adr + 1] += dy
            h = math.radians(dyaw) / 2
            w0, x0, y0, z0 = data.qpos[adr + 3:adr + 7]
            w1, z1 = math.cos(h), math.sin(h)
            data.qpos[adr + 3:adr + 7] = (w1 * w0 - z1 * z0, w1 * x0 - z1 * y0,
                                          w1 * y0 + z1 * x0, w1 * z0 + z1 * w0)
            corrections[body_name(model, b)] = {"dx": round(dx, 3), "dy": round(dy, 3),
                                                "dyaw_deg": dyaw, "before": before,
                                                "after": after}
        if corrections:
            mujoco.mj_kinematics(model, data)
    boxes = _geom_boxes(model, data, movable)
    if not boxes or not len(splats):
        return Split(room=splats, corrections=corrections)

    bottom: dict[int, float] = {}
    for b, R, c, half in boxes:
        low = c[2] - np.abs(R[2]) @ half
        bottom[b] = min(bottom.get(b, np.inf), low)

    pos = splats.positions.astype(np.float64)
    owner = np.full(len(pos), -1, dtype=np.int64)
    for lo in range(0, len(pos), CHUNK):
        p = pos[lo:lo + CHUNK]
        best = np.full(len(p), np.inf)
        best_body = np.full(len(p), -1, dtype=np.int64)
        for b, R, c, half in boxes:
            ext = np.abs(R) @ half + margin                        # world aabb of the box
            near = np.all(np.abs(p - c) <= ext, axis=1)
            if not near.any():
                continue
            local = (p[near] - c) @ R                               # into the geom's frame
            outside = np.maximum(np.abs(local) - half, 0.0)
            dist = np.linalg.norm(outside, axis=1)
            ok = (dist <= margin) & (p[near, 2] > bottom[b] + floor)
            idx = np.flatnonzero(near)[ok]
            closer = dist[ok] < best[idx]
            best[idx[closer]] = dist[ok][closer]
            best_body[idx[closer]] = b
        owner[lo:lo + CHUNK] = best_body

    result = Split(room=splats.subset(owner < 0), corrections=corrections)
    for b in movable:
        mine = owner == b
        if not mine.any():
            continue
        R = np.asarray(data.xmat[b], dtype=np.float64).reshape(3, 3)
        x = np.asarray(data.xpos[b], dtype=np.float64)
        q_inv = _quat_xyzw(np.asarray(data.xquat[b])) * np.array([-1, -1, -1, 1.0])
        part = splats.subset(mine)
        local = Splats(((part.positions - x) @ R).astype(np.float32), part.scales.copy(),
                       quat_multiply(np.broadcast_to(q_inv, part.rotations.shape),
                                     part.rotations),
                       part.colors.copy(), part.opacities.copy(),
                       None if part.sh is None else part.sh.copy())
        name = body_name(model, b)
        result.bodies[name] = local
        result.counts[name] = int(mine.sum())
    return result
