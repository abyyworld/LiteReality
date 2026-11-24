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
floor splats coloured from the ring of floor around it (hidden under the object until it moves),
and `fill_walls` the strip of wall behind it the cameras never saw (else a chair carried off
leaves a chair-shaped window onto the empty background).

A body takes everything in its boxes: a plate's (door leaf, seat) thin side is grown to
`PLATE_HALF` (`_plate`), since a trained splat spreads a thin surface several centimetres off
it, and a body matched to an object found in the splat (`instances`) is cut by its boxes too, its
cluster deciding only the splats the cluster holds. The cluster alone is the object's opaque
core; the legs and the seat's rim and underside left to the room were the chair that stayed put
when the body was carried off. `left_behind` measures what the room keeps at each body's rest
pose and gates the move test on it.

The cut is not the collision box. A fitted body carries its object's extent (`_extent_boxes`,
written by `instances.fit_scene` before its collision geoms are slid apart from a neighbour), and
the cut takes that. A free body's legs, base and casters splay past it: `_grow_legs` takes what
in the body's leg zone (`_leg_zones`, `LEG_REACH` past its box, below its seat) is connected to
its splats, never the carpet's pile, the next chair or anything static. A splat within
`STATIC_KEEP` of static furniture stays the room's unless the body is one no object was found for
(a door). `fill_doorways` puts a dim space behind a door, which was never photographed open.
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
PLATE_HALF = 0.08      # metres: a plate's (door leaf, seat, back) cut reaches this far off its plane
PLATE_RATIO = 0.25     # a geom is a plate when its thinnest side is under this of its next
LEFT_MARGIN = 0.05     # metres past a body's cut box the left-behind check still looks
LEG_REACH = 0.20       # metres past a free body's object box its legs, base and casters may stand
LEG_ZONE = 0.6         # ... from the floor up to this share of its height (a chair: past the seat)
LEG_VOXEL = 0.03       # metres: the voxels a body's legs are grown through
LEG_OPACITY = 0.1      # a voxel with a splat at least this opaque carries the growth
LEG_FUZZ = 0.08        # metres above the floor (carpet pile) below which it only grows downwards
RIM_REACH = 0.10       # metres past its box, every way but down, the rest of a free body may reach
LEFT_MAX = 0.08        # share of the splat (by opacity) at a body's rest pose the room may keep
LEGS_MAX = 0.25        # ... and of what stands under it (thin legs: little opacity, but seen) ...
LEGS_CHECK = 0.4       # ... up to this share of its height (under a chair's seat)
STATIC_MARGIN = 0.05       # splats this close to static furniture never pull a box


@dataclass
class Split:
    room: Splats
    bodies: dict[str, Splats] = field(default_factory=dict)       # name -> splats, body frame
    counts: dict[str, int] = field(default_factory=dict)
    # name -> {"dx", "dy", "dyaw_deg", "before", "after"}: snapped free bodies (see `snap`)
    corrections: dict[str, dict] = field(default_factory=dict)
    # `instances.move_test` of this cut: where each real object found in the splat went
    move_test: dict = field(default_factory=dict)
    # name -> whether the body's splat holds enough of a real object to be offered for grabbing
    grabbable: dict[str, bool] = field(default_factory=dict)


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


SLAB_NORMAL_Z = 0.7    # a box whose thinnest side faces this close to up is a floor or ceiling slab


def _shell_geoms(model: mujoco.MjModel, data: mujoco.MjData) -> list[int]:
    """The room shell's upright parts (walls, window frames, doors): the worldbody's colliding
    geoms whose thinnest side does not face up. The exporter puts the whole shell on the
    worldbody, not on a body of its own. Floor and ceiling slabs are left out, or a chair's legs,
    which stand on the floor slab, would be locked to the room; the floor is already kept by the
    cut's `FLOOR` rule. Planes and visual-only geoms (the photo meshes) say nothing about where
    the walls are."""
    start, count = int(model.body_geomadr[0]), int(model.body_geomnum[0])
    out = []
    for g in range(start, start + count):
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        if not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
            continue
        half = np.asarray(model.geom_aabb[g][3:], dtype=np.float64)
        R = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        if abs(R[2, int(np.argmin(half))]) >= SLAB_NORMAL_Z:
            continue
        out.append(g)
    return out


def _static_boxes(model: mujoco.MjModel, data: mujoco.MjData, movable) -> list:
    """`_geom_boxes` of everything static: the bodies that never move and the room shell's walls
    (`_shell_geoms`, as body 0), so a chair against a wall never takes a patch of the wall."""
    movable = set(movable)
    out = _geom_boxes(model, data, [b for b in range(1, model.nbody) if b not in movable])
    for g in _shell_geoms(model, data):
        aabb = np.asarray(model.geom_aabb[g], dtype=np.float64)
        R = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        out.append((0, R, np.asarray(data.geom_xpos[g]) + R @ aabb[:3], aabb[3:]))
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


def _extent_boxes(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> list:
    """A fitted body's extent sites (``<name>_extent``, `instances._mark_extents`) as
    `_geom_boxes` entries: the real object, wherever its collision geoms were moved or narrowed
    to keep the bodies apart."""
    from litereality_agent.pipeline.splatroom.instances import EXTENT

    out = []
    for i in range(model.nsite):
        if int(model.site_bodyid[i]) != body or int(model.site_type[i]) != int(
                mujoco.mjtGeom.mjGEOM_BOX):
            continue
        if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "").endswith(EXTENT):
            continue
        R = np.asarray(data.site_xmat[i], dtype=np.float64).reshape(3, 3)
        out.append((body, R, np.asarray(data.site_xpos[i], dtype=np.float64).copy(),
                    np.asarray(model.site_size[i], dtype=np.float64).copy()))
    return out


def _shape_boxes(model: mujoco.MjModel, data: mujoco.MjData, bodies) -> list:
    """The boxes that say where each body's object is: its extent sites when it has them
    (`_extent_boxes`), else its geoms' boxes (`_geom_boxes`)."""
    out = []
    for b in bodies:
        out += _extent_boxes(model, data, b) or _geom_boxes(model, data, [b])
    return out


def _plate(half: np.ndarray, reach: float = PLATE_HALF) -> np.ndarray:
    """A geom box's half extents with a plate's thin side grown to `reach`. A trained splat puts
    a thin surface's splats several centimetres either side of it (the atrium's door: 4 cm leaf,
    its splats from 10 cm in front to 6 cm behind its box's mid-plane), so a cut that keeps to the
    geom leaves the room a full copy of the door's face that stays when the leaf swings."""
    half = np.asarray(half, dtype=np.float64).copy()
    order = np.argsort(half)
    if half[order[0]] < PLATE_RATIO * half[order[1]]:
        half[order[0]] = max(half[order[0]], reach)
    return half


def _cut_boxes(model: mujoco.MjModel, data: mujoco.MjData, bodies) -> list:
    """The boxes a body's splats are cut by: a fitted body's extent (`_extent_boxes`, the object
    as found in the splat, never the collision boxes moved or narrowed to keep bodies apart);
    else each geom's (`_geom_boxes`) and one around them all (`_union_box`). Plates are grown to
    `PLATE_HALF` (`_plate`). The union box takes what a real object holds between and around the
    template's parts: a chair's legs stand where its seat's corners are, not where the template
    puts them."""
    out = []
    for b in bodies:
        ext = _extent_boxes(model, data, b)
        if ext:
            out += [(b, R, c, _plate(h)) for b, R, c, h in ext]
            continue
        out += [(b, R, c, _plate(h)) for b, R, c, h in _geom_boxes(model, data, [b])]
        if _body_geoms(model, b):
            out.append((lambda b_, R, c, h: (b_, R, c, _plate(h)))(*_union_box(model, data, b)))
    return out


def _free_top(model: mujoco.MjModel) -> list[int]:
    """The free top-level movable bodies with geoms: what stands on the floor and is carried."""
    return [b for b in sorted(movable_body_ids(model)) if _free_joint(model, b) is not None
            and int(model.body_parentid[b]) == 0 and _body_geoms(model, b)]


def _leg_zones(model: mujoco.MjModel, data: mujoco.MjData, bodies, floor: float) -> list:
    """Per body: the box its legs, base and casters may stand in, as a `_geom_boxes` entry in
    the body's frame: its object's box (`_shape_boxes`) grown `LEG_REACH` in plan, from `floor`
    up to `LEG_ZONE` of its height. A fitted box is the object's opaque core; a chair's thin
    legs splay, and an office chair's star base reaches, 5-20 cm past it."""
    out = []
    for b in bodies:
        boxes = _shape_boxes(model, data, [b])
        if not boxes:
            continue
        Rb = np.asarray(data.xmat[b], dtype=np.float64).reshape(3, 3)
        xb = np.asarray(data.xpos[b], dtype=np.float64)
        pts = np.array([Rb.T @ (c + R @ (h * sgn) - xb) for _b, R, c, h in boxes
                        for sgn in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T
                        .reshape(-1, 3)])
        lo, hi = pts.min(0), pts.max(0)
        lo[:2] -= LEG_REACH
        hi[:2] += LEG_REACH
        bottom = xb[2] + lo[2]
        lo[2] = min(lo[2], floor - xb[2])
        hi[2] = lo[2] + max(bottom - floor, 0.0) + LEG_ZONE * (hi[2] - (bottom - xb[2]))
        out.append((b, Rb, xb + Rb @ ((lo + hi) / 2), (hi - lo) / 2))
    return out


def _rim_zones(model: mujoco.MjModel, data: mujoco.MjData, bodies) -> list:
    """Per body: its object's box (`_shape_boxes`) grown `RIM_REACH` every way but down, in its
    own frame: the seat's rim and the back's edge a trained splat smears past the opaque core."""
    out = []
    for b in bodies:
        boxes = _shape_boxes(model, data, [b])
        if not boxes:
            continue
        Rb = np.asarray(data.xmat[b], dtype=np.float64).reshape(3, 3)
        xb = np.asarray(data.xpos[b], dtype=np.float64)
        pts = np.array([Rb.T @ (c + R @ (h * sgn) - xb) for _b, R, c, h in boxes
                        for sgn in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T
                        .reshape(-1, 3)])
        lo, hi = pts.min(0), pts.max(0)
        lo[:2] -= RIM_REACH
        hi += RIM_REACH
        out.append((b, Rb, xb + Rb @ ((lo + hi) / 2), (hi - lo) / 2))
    return out


def _grow_legs(pos: np.ndarray, opacities: np.ndarray, seeds: np.ndarray, cand: np.ndarray,
               zones: list, floor: float) -> np.ndarray:
    """Which body each candidate splat belongs to by connection, -1 for none: every body's seed
    splats (`seeds`, the body id or -1) grown voxel by voxel (`LEG_VOXEL`, 26-connected) through
    the candidates (`cand`) in its own leg zone (`zones`, `_leg_zones`), a voxel carrying the
    growth only if it holds a candidate at least `LEG_OPACITY` opaque. Below `LEG_FUZZ` over
    `floor` it only grows straight down (a leg to its caster), never along the carpet's pile.
    Bodies grow together; a voxel two reach in the same step goes to the one whose zone centre
    is nearer in plan. A splat in a body's voxels goes with it, however faint."""
    from scipy import ndimage

    out = np.full(len(pos), -1, dtype=np.int64)
    if not zones or not cand.any():
        return out
    ext = np.array([np.abs(R) @ h for _b, R, c, h in zones])
    centres = np.array([c for _b, _R, c, _h in zones])
    lo = (centres - ext).min(0) - LEG_VOXEL
    hi = (centres + ext).max(0) + LEG_VOXEL
    shape = tuple(np.ceil((hi - lo) / LEG_VOXEL).astype(int) + 1)
    if int(np.prod(shape)) > 50_000_000:
        return out

    def vox(p):
        return np.floor((p - lo) / LEG_VOXEL).astype(np.int64)

    inside = np.all((pos >= lo) & (pos < hi), axis=1)
    ci = np.flatnonzero(cand & inside)
    si = np.flatnonzero((seeds >= 0) & inside)
    gc, gs = vox(pos[ci]), vox(pos[si])
    occ = np.zeros(shape, bool)
    strong = opacities[ci] >= LEG_OPACITY
    occ[tuple(gc[strong].T)] = True
    bodies = list(dict.fromkeys(int(b) for b, _R, _c, _h in zones))   # a body may have several
    index = {b: k + 1 for k, b in enumerate(bodies)}
    centres = np.array([next(c for b_, _R, c, _h in zones if b_ == b) for b in bodies])
    label = np.zeros(shape, np.int32)
    seed_label = np.array([index.get(int(b), 0) for b in seeds[si]], np.int32)
    keep = seed_label > 0
    label[tuple(gs[keep].T)] = seed_label[keep]
    # voxel centres: which zones hold them, and how far each zone's centre is in plan
    idx = np.indices(shape).reshape(3, -1).T
    centre = lo + (idx + 0.5) * LEG_VOXEL
    allowed = np.zeros((len(bodies), *shape), bool)
    for b, R, c, h in zones:
        local = (centre - c) @ R
        allowed[index[int(b)] - 1] |= np.all(np.abs(local) <= h, axis=1).reshape(shape)
    low = (centre[:, 2] < floor + LEG_FUZZ).reshape(shape)
    full = np.ones((3, 3, 3), bool)
    steps = int(np.ceil(2 * LEG_REACH / LEG_VOXEL)) + shape[2] + 4
    for _ in range(steps):
        best = np.zeros(shape, np.int32)
        dist = np.full(shape, np.inf)
        for k in range(len(bodies)):
            mine = label == k + 1
            if not mine.any():
                continue
            reach = ndimage.binary_dilation(mine, structure=full)
            below = np.zeros(shape, bool)
            below[:, :, :-1] = mine[:, :, 1:]
            new = np.where(low, below, reach) & occ & (label == 0) & allowed[k]
            if not new.any():
                continue
            w = np.flatnonzero(new.ravel())
            d = np.hypot(*(centre[w, :2] - centres[k, :2]).T)
            better = d < dist.ravel()[w]
            dist.ravel()[w[better]] = d[better]
            best.ravel()[w[better]] = k + 1
        if not best.any():
            break
        label = np.where(best > 0, best, label)
    got = label[tuple(gc.T)]
    # a candidate counts only in a zone of the body whose voxel holds it
    ok = got > 0
    if ok.any():
        k = got[ok] - 1
        ok[np.flatnonzero(ok)] = allowed[k, gc[ok, 0], gc[ok, 1], gc[ok, 2]]
    out[ci[ok]] = np.array(bodies, np.int64)[got[ok] - 1]
    return out


def left_behind(positions: np.ndarray, opacities: np.ndarray, owner: np.ndarray,
                model: mujoco.MjModel, data: mujoco.MjData, *, floor: float | None = None,
                margin: float = LEFT_MARGIN, legs: bool = True) -> dict:
    """What of each movable body the room keeps at the body's rest pose (the ghost a grab
    leaves): per body, the opacity of the room's splats and of the body's own in its cut boxes
    (`_cut_boxes`) grown by `margin`, above the floor (`floor` + `instances.CUT_FLOOR`, else
    the box's bottom + `FLOOR`) and `STATIC_KEEP` clear of every static box (a door's frame is
    the wall's). With `legs`, a free body's leg zone too (`_leg_zones`, `LEG_REACH` past its
    box): what there is connected to what its boxes hold (`_grow_legs`, so the next chair's
    legs and the carpet do not count), which is where a chair's thin legs and base stand.
    ``left`` is the room's share; a body passes at <= `LEFT_MAX`. `positions` are world,
    `owner` -1 for the room, else the body id. Independent of how the cut was made (clusters,
    boxes, masks): it looks at the splats, not at the instances the cut was told about."""
    from litereality_agent.pipeline.splatroom.instances import CUT_FLOOR

    pos = np.asarray(positions, dtype=np.float64)
    ops = np.asarray(opacities, dtype=np.float64)
    movable = sorted(movable_body_ids(model))
    static_boxes = _static_boxes(model, data, movable)

    def clear(mask: np.ndarray) -> np.ndarray:
        if static_boxes and mask.any():
            mask[np.flatnonzero(mask)[_inside_any(pos[mask], static_boxes, STATIC_KEEP)]] = False
        return mask

    regions, cuts = {}, {}
    for b in movable:
        boxes = _cut_boxes(model, data, [b])
        if not boxes:
            continue
        low = min(c[2] - np.abs(R[2]) @ h for _b, R, c, h in boxes)
        cuts[b] = (floor + CUT_FLOOR) if floor is not None else (low + FLOOR)
        regions[b] = clear(_inside_any(pos, boxes, margin) & (pos[:, 2] > cuts[b]))
    free = [b for b in _free_top(model) if b in regions]
    leg_regions = {}
    if legs and free:
        base = floor if floor is not None else min(cuts[b] - FLOOR for b in free)
        zones = _leg_zones(model, data, free, base)
        rims = _rim_zones(model, data, free)
        seeds = np.full(len(pos), -1, dtype=np.int64)
        for b in free:
            seeds[regions[b] & (seeds < 0)] = b
        cand = clear(_inside_any(pos, zones + rims, 0.0) & (seeds < 0)
                     & (pos[:, 2] > min(cuts[b] for b in free)))
        grown = _grow_legs(pos, ops, seeds, cand, zones + rims, base)
        for b, R, c, half in zones:
            regions[b] |= grown == b
            # under the seat: the zone's lower `LEGS_CHECK` of the object's height
            top = (_corners(model, data, b) @ R)[:, 2]
            low = (c @ R)[2] - half[2]
            cap = top.min() + LEGS_CHECK * (top.max() - top.min())
            under = (b, R, R @ np.r_[(c @ R)[:2], (low + cap) / 2], np.r_[half[:2], (cap - low) / 2])
            leg_regions[b] = regions[b] & _inside_any(pos, [under], 0.0)
    rows = {}
    for b, region in regions.items():
        room = float(ops[region & (owner < 0)].sum())
        mine = float(ops[region & (owner == b)].sum())
        row = {}
        if b in leg_regions:
            lr = float(ops[leg_regions[b] & (owner < 0)].sum())
            lm = float(ops[leg_regions[b] & (owner == b)].sum())
            share = lr / (lr + lm) if lr + lm > 0 else 0.0
            row = {"legs_room": round(lr, 1), "legs_body": round(lm, 1),
                   "legs_left": round(share, 4), "legs_pass": bool(share <= LEGS_MAX)}
        left = room / (room + mine) if room + mine > 0 else 0.0
        rows[body_name(model, b)] = {"room": round(room, 1), "body": round(mine, 1),
                                     "left": round(left, 4), **row,
                                     "pass": bool(left <= LEFT_MAX
                                                  and row.get("legs_pass", True))}
    return rows


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
    Only top-level bodies are moved (a free body's pos is then its room position). When nothing
    moves, an `out_name` left by an earlier run is removed: its bodies sit where this cut's do
    not, and whatever picks it up first (the evaluation does) would pair them."""
    out = Path(scene_xml).with_name(out_name)

    def unchanged() -> None:
        if out != Path(scene_xml):
            out.unlink(missing_ok=True)

    if not corrections:
        unchanged()
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
        unchanged()
        return None
    tree.write(out, encoding="unicode")
    return out


def _footprint(model: mujoco.MjModel, data: mujoco.MjData, body: int):
    """A body's footprint on the floor: (centre xy, yaw rotation 2x2, half extents xy, bottom z),
    from its object's boxes (`_corners`), in its own heading."""
    Rb = np.asarray(data.xmat[body]).reshape(3, 3)
    yaw = math.atan2(Rb[1, 0], Rb[0, 0])
    R2 = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    corners = _corners(model, data, body)
    local = corners[:, :2] @ R2                            # into the heading
    lo, hi = local.min(0), local.max(0)
    return R2 @ ((lo + hi) / 2), R2, (hi - lo) / 2, float(corners[:, 2].min())


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


WALL_REACH = 1.0       # metres from a free body to a wall whose face behind it is patched
WALL_GROW = 0.35       # metres the wall patch reaches along the wall past the body's span
WALL_TOP = 0.15        # metres the wall patch reaches above the body's top
WALL_SEEN = 0.08       # metres in front of a wall's face that a splat still shows that part of it
WALL_BEHIND = 0.30     # ... and behind it (a recess, a wall RoomPlan put in front of the real one)
WALL_STEP = 0.02       # metres between the splats that patch a wall
WALL_RING = 0.40       # metres of seen wall around the patch it takes its colours from
WALL_SOLID = 0.3       # ... only its splats at least this opaque
WALL_NEAREST = 200     # ... the median of this many nearest (a wall is one colour, not its edge's)


def _corners(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> np.ndarray:
    """World corners of a body's object boxes (`_shape_boxes`: its extent, else its geoms)."""
    out = []
    for _b, R, c, half in _shape_boxes(model, data, [body]):
        for sgn in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
            out.append(c + R @ (half * sgn))
    return np.asarray(out).reshape(-1, 3)


def fill_walls(room: Splats, model: mujoco.MjModel, data: mujoco.MjData, bodies, floor: float, *,
               step: float = WALL_STEP, reach: float = WALL_REACH, grow: float = WALL_GROW,
               ring: float = WALL_RING) -> tuple[Splats, int]:
    """`room` plus flat wall splats wherever the wall behind a free body was never seen. The
    cameras never saw the strip of wall (and under a desk, the gap) a chair stood in front of, so
    a trained room has nothing there and a chair carried off leaves a chair-shaped window onto the
    empty scene background: the "white shell" of the doubled chair. Per body in `bodies` and per
    upright wall of the room shell (`_shell_geoms`) whose room-facing side is within `reach` of
    it: the face behind the body's span (+ `grow`), from `floor` to the body's top + `WALL_TOP`,
    gets a splat every `step` wherever no splat lies within `WALL_SEEN` of the face, at the
    depth the seen wall around it has and coloured by the nearest seen wall splats (within
    `ring`). Returns (room, added)."""
    from scipy.spatial import cKDTree

    from litereality_agent.models.splat.spz import matrix_to_quat

    if not len(room) or not bodies:
        return room, 0
    pos = room.positions.astype(np.float64)
    patches = []
    for b in bodies:
        corners = _corners(model, data, b)
        if not len(corners):
            continue
        centre, top = corners.mean(0), float(corners[:, 2].max()) + WALL_TOP
        for g in _shell_geoms(model, data):
            half = np.asarray(model.geom_aabb[g][3:], dtype=np.float64)
            R = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
            c = np.asarray(data.geom_xpos[g]) + R @ np.asarray(model.geom_aabb[g][:3])
            t = int(np.argmin(half))
            v = max((a for a in range(3) if a != t), key=lambda a: abs(R[2, a]))
            u = 3 - t - v
            side = 1.0 if (centre - c) @ R[:, t] >= 0 else -1.0
            lc = (corners - c) @ R
            if np.min(side * lc[:, t]) - half[t] > reach:
                continue
            lo_u = max(lc[:, u].min() - grow, -half[u])
            hi_u = min(lc[:, u].max() + grow, half[u])
            if hi_u - lo_u < step:
                continue
            local = (pos - c) @ R
            depth = side * local[:, t] - half[t]
            z = pos[:, 2]
            band = ((depth > -0.05) & (depth < WALL_SEEN) & (local[:, u] > lo_u - ring)
                    & (local[:, u] < hi_u + ring) & (z > floor) & (z < top + ring))
            if band.sum() < 30:
                continue
            face = float(np.median(depth[band]))
            uz = np.column_stack([local[band, u], z[band]])
            gu, gz = np.meshgrid(np.arange(lo_u, hi_u + 1e-9, step),
                                 np.arange(floor + step / 2, top + 1e-9, step))
            grid = np.column_stack([gu.ravel(), gz.ravel()])
            # Seen at the face or behind it: a patch there would hide what the cameras saw. Only
            # opaque splats say a cell was seen; a little haze there still shows the background.
            seen = ((depth > -WALL_BEHIND) & (depth < WALL_SEEN) & (local[:, u] > lo_u - step)
                    & (local[:, u] < hi_u + step) & (z > floor) & (z < top + step)
                    & (room.opacities >= WALL_SOLID))
            if seen.any():
                near, _ = cKDTree(np.column_stack([local[seen, u], z[seen]])).query(
                    grid, k=1, distance_upper_bound=1.5 * step)
                grid = grid[~np.isfinite(near)]
            if not len(grid):
                continue
            # The wall's own colour: the median of the nearest opaque wall splats, so a little
            # haze or a stray splat at the edge of what was seen does not paint the patch.
            solid = room.opacities[band] >= WALL_SOLID
            src = solid if solid.sum() >= 30 else np.ones(len(uz), bool)
            k = min(WALL_NEAREST, int(src.sum()))
            _dist, idx = cKDTree(uz[src]).query(grid, k=k)
            idx = idx.reshape(len(grid), k)
            colour = np.median(room.colors[band][src][idx], axis=1)
            n = len(grid)
            lp = np.zeros((n, 3))
            lp[:, u] = grid[:, 0]
            lp[:, t] = side * (half[t] + face)
            lp[:, v] = (grid[:, 1] - c[2] - R[2, u] * lp[:, u] - R[2, t] * lp[:, t]) / R[2, v]
            frame = np.column_stack([R[:, u], R[:, v], np.cross(R[:, u], R[:, v])])
            patches.append(Splats(
                (lp @ R.T + c).astype(np.float32),
                np.tile([step * 0.8, step * 0.8, step * 0.1], (n, 1)).astype(np.float32),
                np.tile(matrix_to_quat(frame), (n, 1)).astype(np.float32),
                colour.astype(np.float32), np.full(n, 0.95, np.float32)))
    added = sum(len(p) for p in patches)
    return (Splats.concat([room, *patches]) if added else room), added


DOOR_MIN_HEIGHT = 1.4  # metres: a hinged upright plate this tall is a door with a room behind it
DOOR_BACK = 0.30       # metres the space behind an opened door reaches past the leaf's swing
DOOR_STEP = 0.04       # metres between the splats that make that space
DOOR_LOOK = 1.2        # metres either side of a leaf in which the room's splats say which is its side
DOOR_DIM = {"back": 0.55, "side": 0.7, "ceiling": 0.6, "floor": 0.75}   # shading of the space


def _hinged(model: mujoco.MjModel, body: int) -> bool:
    start, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
    return any(int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
               for j in range(start, start + count))


def fill_doorways(room: Splats, model: mujoco.MjModel, data: mujoco.MjData, bodies,
                  floor: float, *, step: float = DOOR_STEP) -> tuple[Splats, int]:
    """`room` plus a dim space behind every door in `bodies`: what an opened door shows was
    never photographed (the door was shut, or the cameras stayed in the room), so a trained room
    has nothing there and the doorway opens onto the bare scene background. Per hinged body
    whose box (`_union_box`) is an upright plate at least `DOOR_MIN_HEIGHT` tall: on the side of
    the leaf the room's splats are not on (`DOOR_LOOK`), a box-shaped space as wide as the
    opening plus the leaf's swing, `DOOR_BACK` deeper than the leaf is wide, from `floor` to above
    the opening: its floor in the colour of the room's floor at the door, its walls and ceiling in
    the colour of the wall around the opening, each shaded darker (`DOOR_DIM`), a splat every
    `step` wherever nothing was seen. Returns (room, added)."""
    from scipy.spatial import cKDTree

    from litereality_agent.models.splat.spz import matrix_to_quat

    if not len(room) or not bodies:
        return room, 0
    pos = room.positions.astype(np.float64)
    patches = []
    for b in bodies:
        if not _body_geoms(model, b):
            continue
        _b, R, c, half = _union_box(model, data, b)
        t = int(np.argmin(half))
        if abs(R[2, t]) > 0.3 or half[t] >= PLATE_RATIO * np.sort(half)[1]:
            continue
        v = max((a for a in range(3) if a != t), key=lambda a: abs(R[2, a]))
        u = 3 - t - v
        if 2 * half[v] < DOOR_MIN_HEIGHT:
            continue
        nt = R[:, t].copy()
        nt[2] = 0.0
        nt /= np.linalg.norm(nt)
        nu = np.cross([0.0, 0.0, 1.0], nt)
        rel = pos - c
        with np.errstate(all="ignore"):         # a trained splat can hold a few non-finite rows
            along, across = rel @ nu, rel @ nt
        near = (np.abs(along) < half[u] + 0.5) & (np.abs(across) < DOOR_LOOK) \
            & (np.abs(across) > half[t] + 0.05) & (pos[:, 2] > floor - 0.1)
        if near.sum() < 50:
            continue
        side = 1.0 if (across[near] > 0).sum() >= (across[near] < 0).sum() else -1.0
        back = -side * nt                                    # away from the room
        top = c[2] + half[v]
        floor_near = near & (side * across > 0) & (np.abs(pos[:, 2] - floor) < 0.03)
        wall_near = (np.abs(across) < half[t] + 0.10) & (np.abs(along) > half[u] + 0.03) \
            & (np.abs(along) < half[u] + 0.6) & (pos[:, 2] > floor + 0.3) & (pos[:, 2] < top)
        if floor_near.sum() < 20 or wall_near.sum() < 20:
            continue
        floor_rgb = np.median(room.colors[floor_near], axis=0)
        wall_rgb = np.median(room.colors[wall_near], axis=0)
        depth = 2 * half[u] + DOOR_BACK
        wide = half[u] + depth / 2
        start = half[t] + 0.03
        high = top - floor + 0.3

        def face(origin, a, b, la, lb, rgb):
            ga, gb = np.meshgrid(np.arange(step / 2, la, step), np.arange(step / 2, lb, step))
            pts = origin + ga.reshape(-1, 1) * a + gb.reshape(-1, 1) * b
            return pts, np.column_stack([a, b, np.cross(a, b)]), rgb

        base = np.array([c[0], c[1], floor])
        corner = base + back * start - nu * wide            # floor, near, left
        up = np.array([0.0, 0.0, 1.0])
        faces = [
            face(corner, nu, back, 2 * wide, depth, floor_rgb * DOOR_DIM["floor"]),
            face(corner + up * high, nu, back, 2 * wide, depth, wall_rgb * DOOR_DIM["ceiling"]),
            face(corner + back * depth, nu, up, 2 * wide, high, wall_rgb * DOOR_DIM["back"]),
            face(corner, back, up, depth, high, wall_rgb * DOOR_DIM["side"]),
            face(corner + nu * 2 * wide, back, up, depth, high, wall_rgb * DOOR_DIM["side"]),
        ]
        box_lo = np.min([f[0].min(0) for f in faces], axis=0) - 2 * step
        box_hi = np.max([f[0].max(0) for f in faces], axis=0) + 2 * step
        inside = np.all((pos >= box_lo) & (pos <= box_hi), axis=1)
        tree = cKDTree(pos[inside]) if inside.any() else None
        for pts, frame, rgb in faces:
            if tree is not None:
                d, _ = tree.query(pts, k=1, distance_upper_bound=1.5 * step)
                pts = pts[~np.isfinite(d)]
            n = len(pts)
            if not n:
                continue
            patches.append(Splats(
                pts.astype(np.float32),
                np.tile([step * 0.8, step * 0.8, step * 0.1], (n, 1)).astype(np.float32),
                np.tile(matrix_to_quat(frame), (n, 1)).astype(np.float32),
                np.tile(np.clip(rgb, 0.0, 1.0), (n, 1)).astype(np.float32),
                np.full(n, 0.95, np.float32)))
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


def _penetration(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> float:
    """The deepest contact between `body` and anything outside its own tree (walls, static
    furniture and the other movable bodies), in metres (0: none)."""
    root = int(model.body_rootid[body])
    deepest = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if body not in (b1, b2):
            continue
        other = b2 if b1 == body else b1
        if other != 0 and int(model.body_rootid[other]) == root:
            continue
        deepest = max(deepest, -float(c.dist))
    return deepest


def _plan_polygon(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """A geom's box (``geom_aabb`` in its own frame) seen from above: the convex hull of its
    corners' xy, counter-clockwise."""
    from scipy.spatial import ConvexHull

    aabb = np.asarray(model.geom_aabb[geom], dtype=np.float64)
    R = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
    c = np.asarray(data.geom_xpos[geom]) + R @ aabb[:3]
    pts = np.array([c + R @ (aabb[3:] * sgn) for sgn in
                    np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)])[:, :2]
    return pts[ConvexHull(pts).vertices]


def _plan_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """How far two convex plan polygons overlap (separating axes), metres; negative is the gap
    between them along the axis that separates them best."""
    best = np.inf
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        for n in normals:
            pa, pb = a @ n, b @ n
            best = min(best, min(pa.max(), pb.max()) - max(pa.min(), pb.min()))
    return float(best)


def plan_overlap(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> float:
    """The most `body`'s geoms (`_body_geoms`) overlap those of any other movable body seen from
    above, metres (negative: the smallest gap). Two fitted chairs whose seats overlap in plan at
    different heights do not touch at rest, but lifting the lower one straight up drives it into
    the other: grabbing one chair carries its neighbour."""
    others = [b for b in movable_body_ids(model)
              if b != body and int(model.body_rootid[b]) != int(model.body_rootid[body])]
    mine = [_plan_polygon(model, data, g) for g in _body_geoms(model, body)]
    worst = -np.inf
    for o in others:
        for g in _body_geoms(model, o):
            theirs = _plan_polygon(model, data, g)
            worst = max(worst, max(_plan_overlap(m, theirs) for m in mine))
    return float(worst)


def _physical(model: mujoco.MjModel, corrections: dict[str, dict]) -> dict[str, dict]:
    """`corrections`, each scaled back (to 75, 50, 25 %, then dropped) until its body sinks no
    deeper into anything (walls, static furniture, the other bodies) than at its pose before the
    move: a body the photos put a few centimetres into the wall, or into a chair that was itself
    moved, would be wedged there and could never be moved. Moves are accepted one at a time, each
    tested with those already accepted applied."""
    accepted: dict[str, dict] = {}

    def state(moves: dict[str, dict]) -> mujoco.MjData:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        _apply(model, data, moves)
        mujoco.mj_forward(model, data)
        return data

    for name, fix in corrections.items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        before = _penetration(model, state(accepted), b)
        for scale in (1.0, 0.75, 0.5, 0.25):
            trial = {**fix, "dx": round(fix["dx"] * scale, 3), "dy": round(fix["dy"] * scale, 3),
                     "dyaw_deg": fix["dyaw_deg"] * scale}
            if _penetration(model, state({**accepted, name: trial}), b) \
                    <= before + PENETRATION_SLACK:
                if scale < 1.0:
                    trial["scaled"] = scale
                accepted[name] = trial
                break
    return accepted


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
          move_bodies: bool = True, instances: bool = False) -> Split:
    """Cut `splats` (see the module docstring). `masks` (`masks.ObjectMasks`) makes the photos'
    object outlines decide wherever a photo saw a splat. `corrections` (from an earlier split of
    the same room) are applied instead of snapping again, so every snapshot of a room is cut with
    the bodies where its first one put them; an empty dict means the first one moved nothing, and
    nothing moves. With masks, the photos place the bodies they outline (`_corrections_from_votes`)
    and the free bodies they do not are snapped. With `move_bodies` False nothing is moved (no
    snap, no correction): for a room whose physics is already live and stays as it is.

    With `instances` (`instances.find`, `instances.match`) every free body matched to an object
    found in the splat takes that object's splats by cluster membership (its voxels grown by
    `instances.GROW`), not its box; only the bodies no object matched (a door, a body of a
    category with no object in sight) are cut by their boxes. Nothing is snapped: the bodies are
    where `instances.fit_scene` put them. Every cut is checked by `instances.move_test`
    (`Split.move_test`, `Split.grabbable`)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    movable = sorted(movable_body_ids(model))
    fixed = corrections is not None
    corrections = dict(corrections) if fixed else {}        # never the caller's dict
    if fixed:
        _apply(model, data, corrections)
    snap_bodies = snap_bodies and move_bodies and not fixed and not instances
    named: dict[str, int] = {}
    if masks is not None and len(masks):
        with_masks = {n for per in masks.masks.values() for n in per}
        named = {body_name(model, b): b for b in movable if body_name(model, b) in with_masks}
    from_votes = snap_bodies and bool(named)
    if snap_bodies and len(splats):
        static_boxes = _static_boxes(model, data, movable)
        pos64 = splats.positions.astype(np.float64)
        for b in movable:
            j = _free_joint(model, b)
            if j is None or int(model.body_parentid[b]) != 0:
                continue
            if from_votes and body_name(model, b) in named:
                continue                                    # the photos place it
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
    from litereality_agent.pipeline.splatroom import instances as objects

    cats = objects.categories(model, data)
    found = objects.find(splats.positions, splats.opacities, model, data, cats=cats) \
        if len(splats) else None
    paired: dict[int, int] = {}
    if instances and found is not None:
        paired = objects.match(found, model, data, cats).pairs        # body -> instance
    # Every body is cut by its boxes; a paired body's cluster (below) then has the last word on
    # the splats it holds. The cluster alone is only the object's opaque core (a chair's seat and
    # back): its thin legs and the rim and underside of its seat hold too few opaque splats a
    # voxel to join it, and left to the room they are a chair-shaped ghost that stays put when
    # the body is carried off.
    boxes = _cut_boxes(model, data, movable)
    if not boxes or not len(splats):
        return Split(room=splats, corrections=corrections)
    static_boxes = _static_boxes(model, data, movable)

    bottom: dict[int, float] = {}
    for b, R, c, half in boxes:
        low = c[2] - np.abs(R[2]) @ half
        bottom[b] = min(bottom.get(b, np.inf), low)
    for b in paired:
        bottom[b] = found.floor + objects.CUT_FLOOR - floor      # never the floor under it

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
            # Signed: outside, the distance to the box; inside, minus the depth in it, so where
            # two bodies' boxes overlap a splat goes to the one it is deeper in (nearer its core).
            dist = np.where(np.any(outside > 0, axis=1), np.linalg.norm(outside, axis=1),
                            -np.min(half - np.abs(local), axis=1))
            ok = (dist <= margin) & (p[near, 2] > bottom[b] + floor)
            idx = np.flatnonzero(near)[ok]
            closer = dist[ok] < best[idx]
            best[idx[closer]] = dist[ok][closer]
            best_body[idx[closer]] = b
        # Nothing inside static furniture moves with a body (the table top over a tucked chair),
        # nor anything within `STATIC_KEEP` of it (the trained desk edge spreads 3 cm past its
        # box, into the box of the chair under it). Only a body no object was found for (a door:
        # its box is all that says where it is) keeps what is inside its own box that close to
        # something static, its edge standing that close to the wall it closes.
        taken = best_body >= 0
        if taken.any() and static_boxes:
            idx = np.flatnonzero(taken)
            own = (best[idx] <= 0) & ~np.isin(best_body[idx], list(paired))
            locked = np.where(own, _inside_any(p[idx], static_boxes, 0.0),
                              _inside_any(p[idx], static_boxes, STATIC_KEEP))
            best_body[idx[locked]] = -1
        owner[lo:lo + CHUNK] = best_body

    if paired:
        # Cluster membership: the object's voxels grown a little, faint splats too, never the
        # floor under it or anything static (the desk top over a tucked chair).
        to_body = np.full(max(found.instances, default=0) + 1, -1, dtype=np.int64)
        for b, i in paired.items():
            to_body[i] = b
        claim = to_body[found.labels(pos)]
        claim[pos[:, 2] <= found.floor + objects.CUT_FLOOR] = -1
        if static_boxes and (claim >= 0).any():
            took = np.flatnonzero(claim >= 0)
            claim[took[_inside_any(pos[took], static_boxes, STATIC_KEEP)]] = -1
        owner = np.where(claim >= 0, claim, owner)

    # The legs, base and casters a free body's box misses (they splay past the object's opaque
    # core), and the seat's rim: what is connected to the body's splats in its leg zone and just
    # around its box (`_grow_legs`), never the floor, anything static or another body's.
    free = [b for b in _free_top(model) if b in bottom]
    if free:
        base = found.floor if found is not None else min(bottom[b] for b in free)
        zones = _leg_zones(model, data, free, base) + _rim_zones(model, data, free)
        low = np.array([bottom.get(b, np.inf) for b in range(model.nbody)])
        cand = (owner < 0) & (pos[:, 2] > base + objects.CUT_FLOOR) & _inside_any(pos, zones, 0.0)
        cand &= pos[:, 2] > min(low[b] for b in free) + floor
        if static_boxes and cand.any():
            cand[np.flatnonzero(cand)[_inside_any(pos[cand], static_boxes, STATIC_KEEP)]] = False
        grown = _grow_legs(pos, splats.opacities, np.where(np.isin(owner, free), owner, -1),
                           cand, zones, base)
        owner = np.where(grown >= 0, grown, owner)

    if named:
        from litereality_agent.pipeline.splatroom.masks import vote

        voted = vote(pos, masks, model, data, named)
        low = np.array([bottom.get(b, -np.inf) for b in range(model.nbody)])
        say = voted >= 0
        say &= pos[:, 2] > low[np.maximum(voted, 0)] + floor
        if static_boxes and say.any():
            say[np.flatnonzero(say)[_inside_any(pos[say], static_boxes, STATIC_KEEP)]] = False
        # Only a body's own mask moves a splat to it. Seen and in no mask (-1) never takes a
        # splat from the body whose box or cluster holds it: a photo that missed part of a chair
        # (or whose mask was given to the chair beside it) says nothing about where it belongs.
        owner = np.where(say, voted, owner)
        if from_votes:
            moved = _physical(model, _corrections_from_votes(model, data, owner, pos, named))
            corrections.update(moved)
            _apply(model, data, moved)
    result = Split(room=splats.subset(owner < 0), corrections=corrections)
    if found is not None:
        names = {b: body_name(model, b) for b in movable}
        result.move_test = objects.move_test(found, pos, splats.opacities, owner, names)
        result.move_test["cut"] = "instances" if paired else "boxes"
        result.move_test["left_behind"] = left_behind(pos, splats.opacities, owner, model, data,
                                                      floor=found.floor)
        result.move_test["pass"] = bool(result.move_test["pass"]) and all(
            r["pass"] for r in result.move_test["left_behind"].values())
        gated = {b for c, cat in cats.items() for b in cat.bodies
                 if any(r["category"] == c for r in result.move_test["objects"])}
        best = result.move_test["best_share"]
        for b in movable:
            if b in gated and (owner == b).any():
                result.grabbable[names[b]] = best.get(names[b], 0.0) >= objects.GRAB_MIN
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
        free = [b for b in movable if body_name(model, b) in result.bodies
                and _free_joint(model, b) is not None and int(model.body_parentid[b]) == 0]
        low = found.floor if found is not None else min((p[3] for p in prints), default=0.0)
        result.room, added = fill_walls(result.room, model, data, free, low)
        if added:
            result.counts["wall_fill"] = added
        doors = [b for b in movable if body_name(model, b) in result.bodies and _hinged(model, b)]
        result.room, added = fill_doorways(result.room, model, data, doors, low)
        if added:
            result.counts["door_fill"] = added
    return result
