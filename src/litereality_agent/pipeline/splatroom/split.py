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

Nothing was ever photographed under an object, so cutting it out leaves a hole in the floor that
shows the moment someone moves it. `fill_floor` patches each free body's footprint with flat
floor splats coloured from the ring of floor around it (hidden under the object until it moves).
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
FILL_STEP = 0.015      # metres between the floor splats that patch a footprint
FILL_GROW = 0.06       # metres the patched footprint reaches past the object's box
FILL_RING = 0.35       # metres of floor around it the patch takes its colours from
STATIC_KEEP = 0.03     # metres around static furniture whose splats never go to a body
STATIC_MARGIN = 0.05       # splats this close to static furniture never pull a box


@dataclass
class Split:
    room: Splats
    bodies: dict[str, Splats] = field(default_factory=dict)       # name -> splats, body frame
    counts: dict[str, int] = field(default_factory=dict)
    # name -> {"dx", "dy", "dyaw_deg", "before", "after"}: snapped free bodies (see `snap`)
    corrections: dict[str, dict] = field(default_factory=dict)


COLLISION_GROUP = 3


def _body_geoms(model: mujoco.MjModel, body: int) -> list[int]:
    """The geoms that say where a body is: its colliders (the template's seat, back and legs).
    A generated visual mesh's box is loose (tea_room's chairs reach 20 cm under the floor), so
    it is used only by a body that has nothing else."""
    start, count = int(model.body_geomadr[body]), int(model.body_geomnum[body])
    geoms = [g for g in range(start, start + count)
             if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)]
    hard = [g for g in geoms if int(model.geom_group[g]) == COLLISION_GROUP]
    return hard or geoms


def _geom_boxes(model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]):
    """Per geom of `bodies` (`_body_geoms`): (body, world rotation, world centre, half extents)."""
    out = []
    for b in bodies:
        for g in _body_geoms(model, b):
            aabb = np.asarray(model.geom_aabb[g], dtype=np.float64)
            R = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
            centre = np.asarray(data.geom_xpos[g]) + R @ aabb[:3]
            out.append((b, R, centre, aabb[3:]))
    return out


def _union_box(model: mujoco.MjModel, data: mujoco.MjData, body: int):
    """One box around all of a body's geoms (`_body_geoms`), in the body's own frame, as a
    `_geom_boxes` entry: the real object fills its scanned box, not only the template's parts."""
    Rb = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
    xb = np.asarray(data.xpos[body], dtype=np.float64)
    corners = []
    for g in _body_geoms(model, body):
        aabb = np.asarray(model.geom_aabb[g], dtype=np.float64)
        Rg = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        cg = np.asarray(data.geom_xpos[g]) + Rg @ aabb[:3]
        for sgn in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
            corners.append(Rb.T @ (cg + Rg @ (aabb[3:] * sgn) - xb))
    corners = np.asarray(corners)
    lo, hi = corners.min(0), corners.max(0)
    return body, Rb, xb + Rb @ ((lo + hi) / 2), (hi - lo) / 2


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
    Rb = np.asarray(data.xmat[body]).reshape(3, 3)
    xb = np.asarray(data.xpos[body], dtype=np.float64)
    corners = []
    for g in _body_geoms(model, body):
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


def _footprint(model: mujoco.MjModel, data: mujoco.MjData, body: int):
    """A body's footprint on the floor: (centre xy, yaw rotation 2x2, half extents xy, bottom z),
    from all its geoms' boxes, in its own heading."""
    Rb = np.asarray(data.xmat[body]).reshape(3, 3)
    yaw = math.atan2(Rb[1, 0], Rb[0, 0])
    R2 = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    pts, low = [], np.inf
    for g in _body_geoms(model, body):
        aabb = np.asarray(model.geom_aabb[g], dtype=np.float64)
        Rg = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        cg = np.asarray(data.geom_xpos[g]) + Rg @ aabb[:3]
        for sgn in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
            corner = cg + Rg @ (aabb[3:] * sgn)
            pts.append(corner[:2])
            low = min(low, corner[2])
    local = np.asarray(pts) @ R2                           # into the heading
    lo, hi = local.min(0), local.max(0)
    return R2 @ ((lo + hi) / 2), R2, (hi - lo) / 2, low


def fill_floor(room: Splats, footprints, *, step: float = FILL_STEP, grow: float = FILL_GROW,
               ring: float = FILL_RING) -> tuple[Splats, int]:
    """`room` plus flat floor splats wherever a footprint's floor was never seen, each coloured
    by the floor splats nearest it in the ring around the footprint. Returns (room, added)."""
    from scipy.spatial import cKDTree

    if not len(room) or not footprints:
        return room, 0
    pos = room.positions.astype(np.float64)
    patches = []
    for centre, R2, half, low in footprints:
        rel = (pos[:, :2] - centre) @ R2                      # into the footprint's heading
        inside = np.all(np.abs(rel) <= half + grow, axis=1)
        band = np.all(np.abs(rel) <= half + grow + ring, axis=1) & ~inside
        # The floor as the LiDAR saw it (RoomPlan's can sit a few cm off): the median height of
        # what lies around the footprint within 10 cm of the object's lowest point.
        guess = band & (np.abs(pos[:, 2] - low) < 0.10)
        if guess.sum() < 30:
            continue
        floor_z = float(np.median(pos[guess, 2]))
        near_floor = np.abs(pos[:, 2] - floor_z) < 0.025
        around = band & near_floor
        if around.sum() < 30:
            continue
        nx, ny = (int(math.ceil((h + grow) / step)) for h in half)
        gu, gv = np.meshgrid(np.arange(-nx, nx + 1) * step, np.arange(-ny, ny + 1) * step)
        grid = np.stack([gu.ravel(), gv.ravel()], 1) @ R2.T + centre
        seen = near_floor & inside
        if seen.any():
            covered = cKDTree(pos[seen, :2]).query(grid, k=1, distance_upper_bound=step)[0]
            grid = grid[~np.isfinite(covered)]
        if not len(grid):
            continue
        tree = cKDTree(pos[around, :2])
        k = min(8, int(around.sum()))
        dist, idx = tree.query(grid, k=k)
        w = 1.0 / np.maximum(dist, 0.01) ** 2
        colour = (room.colors[around][idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
        n = len(grid)
        patches.append(Splats(
            np.column_stack([grid, np.full(n, floor_z)]).astype(np.float32),
            np.tile([step * 0.8, step * 0.8, step * 0.1], (n, 1)).astype(np.float32),
            np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
            colour.astype(np.float32), np.full(n, 0.95, np.float32)))
    added = sum(len(p) for p in patches)
    return (Splats.concat([room, *patches]) if added else room), added


VOTE_MIN_SPLATS = 300
VOTE_MIN_SHIFT, VOTE_MAX_SHIFT = 0.03, 0.5


def _corrections_from_votes(model, data, owner, pos, named: dict[str, int]) -> dict[str, dict]:
    """Where the photos say each free body really is: its box slid (no turn) so its centre sits
    on the median of the splats the masks gave it. Small or implausible shifts are left alone."""
    out = {}
    for name, b in named.items():
        if _free_joint(model, b) is None or int(model.body_parentid[b]) != 0:
            continue
        mine = owner == b
        if mine.sum() < VOTE_MIN_SPLATS:
            continue
        _b, _R, centre, _half = _union_box(model, data, b)
        dx, dy = np.median(pos[mine, :2], axis=0) - centre[:2]
        if VOTE_MIN_SHIFT <= math.hypot(dx, dy) <= VOTE_MAX_SHIFT:
            out[name] = {"dx": round(float(dx), 3), "dy": round(float(dy), 3), "dyaw_deg": 0.0,
                         "before": None, "after": int(mine.sum()), "from": "photo masks"}
    return out


PENETRATION_SLACK = 0.005     # metres a corrected body may sink into static geometry beyond before


def _static_penetration(model: mujoco.MjModel, data: mujoco.MjData, body: int,
                        movable: set[int]) -> float:
    """The deepest contact between `body` and anything static, in metres (0: none)."""
    deepest = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if body not in (b1, b2):
            continue
        other = b2 if b1 == body else b1
        if other in movable:
            continue
        deepest = max(deepest, -float(c.dist))
    return deepest


def _physical(model: mujoco.MjModel, corrections: dict[str, dict]) -> dict[str, dict]:
    """`corrections`, each scaled back (to 75, 50, 25 %, then dropped) until its body sinks no
    deeper into walls and static furniture than at the scanned pose: a body the photos put a few
    centimetres into the wall would be wedged there and could never be moved."""
    movable = movable_body_ids(model)
    out = {}
    for name, fix in corrections.items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        before = _static_penetration(model, data, b, movable)
        for scale in (1.0, 0.75, 0.5, 0.25):
            trial = {**fix, "dx": round(fix["dx"] * scale, 3), "dy": round(fix["dy"] * scale, 3),
                     "dyaw_deg": fix["dyaw_deg"] * scale}
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            _apply(model, data, {name: trial})
            mujoco.mj_forward(model, data)
            if _static_penetration(model, data, b, movable) <= before + PENETRATION_SLACK:
                if scale < 1.0:
                    trial["scaled"] = scale
                out[name] = trial
                break
    return out


def _apply(model: mujoco.MjModel, data: mujoco.MjData, corrections: dict[str, dict]) -> None:
    """Move every corrected free body by its correction (as `split` snapping does)."""
    for name, fix in corrections.items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        j = _free_joint(model, b) if b >= 0 else None
        if j is None:
            continue
        adr = int(model.jnt_qposadr[j])
        data.qpos[adr] += fix["dx"]
        data.qpos[adr + 1] += fix["dy"]
        h = math.radians(fix["dyaw_deg"]) / 2
        w0, x0, y0, z0 = data.qpos[adr + 3:adr + 7]
        w1, z1 = math.cos(h), math.sin(h)
        data.qpos[adr + 3:adr + 7] = (w1 * w0 - z1 * z0, w1 * x0 - z1 * y0,
                                      w1 * y0 + z1 * x0, w1 * z0 + z1 * w0)
    if corrections:
        mujoco.mj_kinematics(model, data)


def _quat_xyzw(mj_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = mj_wxyz
    return np.array([x, y, z, w], dtype=np.float64)


def split(splats: Splats, model: mujoco.MjModel, *, margin: float = MARGIN,
          floor: float = FLOOR, snap_bodies: bool = True, fill: bool = True,
          masks=None, corrections: dict[str, dict] | None = None,
          move_bodies: bool = True) -> Split:
    """Cut `splats` (see the module docstring). `masks` (`masks.ObjectMasks`) makes the photos'
    object outlines decide wherever a photo saw a splat. `corrections` (from an earlier split of
    the same room) are applied instead of snapping again, so every snapshot of a room is cut with
    the bodies where its first one put them. With `move_bodies` False nothing is moved (no snap,
    no correction): for a room whose physics is already live and stays as it is."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    movable = sorted(movable_body_ids(model))
    if corrections is not None:
        _apply(model, data, corrections)
    else:
        corrections = {}
    snap_bodies = snap_bodies and move_bodies
    from_votes = masks is not None and len(masks) > 0 and not corrections and snap_bodies
    if snap_bodies and not corrections and not from_votes and len(splats):
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
            kept = _physical(model, corrections)
            # Undo every snap, then apply only what is physically possible.
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            corrections = kept
            _apply(model, data, corrections)
    boxes = _geom_boxes(model, data, movable)
    if not boxes or not len(splats):
        return Split(room=splats, corrections=corrections)
    boxes += [_union_box(model, data, b) for b in movable if _body_geoms(model, b)]
    static_boxes = _geom_boxes(model, data, [b for b in range(1, model.nbody)
                                             if b not in movable])

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
        # Nothing inside static furniture moves with a body (the table top over a tucked chair).
        taken = best_body >= 0
        if taken.any() and static_boxes:
            locked = _inside_any(p[taken], static_boxes, STATIC_KEEP)
            best_body[np.flatnonzero(taken)[locked]] = -1
        owner[lo:lo + CHUNK] = best_body

    if masks is not None and len(masks):
        from litereality_agent.pipeline.splatroom.masks import vote

        with_masks = {n for per in masks.masks.values() for n in per}
        named = {body_name(model, b): b for b in movable if body_name(model, b) in with_masks}
        if named:
            voted = vote(pos, masks, model, data, named)
            low = np.array([bottom.get(b, -np.inf) for b in range(model.nbody)])
            say = voted >= 0
            say &= pos[:, 2] > low[np.maximum(voted, 0)] + floor
            if static_boxes and say.any():
                say[np.flatnonzero(say)[_inside_any(pos[say], static_boxes, STATIC_KEEP)]] = False
            owner = np.where(say, voted, owner)
            # Seen by the photos and outside every mask: not one of the masked bodies after all.
            masked_ids = np.array(sorted(named.values()))
            drop = (voted == -1) & np.isin(owner, masked_ids)
            owner[drop] = -1
            if from_votes:
                corrections.update(_physical(model, _corrections_from_votes(
                    model, data, owner, pos, named)))
                _apply(model, data, {n: corrections[n] for n in corrections})
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
    if fill:
        prints = [_footprint(model, data, b) for b in movable
                  if body_name(model, b) in result.bodies and _free_joint(model, b) is not None
                  and int(model.body_parentid[b]) == 0]
        result.room, added = fill_floor(result.room, prints)
        if added:
            result.counts["floor_fill"] = added
    return result
