"""build.py — a fresh scan's RoomPlan boxes as a MuJoCo room people can walk into, in seconds.

This is the room call's first version of a scanned room ("stage quick"): built on the laptop the
moment a scan lands, with no Blender, no model calls and no GPU, so the people on the call can be
standing in their room while the full pipeline is still reconstructing it. Every later stage
(TRELLIS objects, then the full export) is published as a new version of the same link and swapped
in live — which is why so much here is about agreeing with the pipeline rather than about boxes:

* **Same names.** Bodies are the scan's handles (``Chair0``, ``Table0``, ``Wall10_Door_0``), free
  joints ``<handle>_free`` as the exporter writes them: a swap carries state by name.
* **Same layout.** Counter runs are merged with ``merge_boxes`` (``Sink0`` + ``Storage2`` ->
  ``Sink_Storage0``) and the deterministic layout repair runs exactly as the ingest stage runs it
  with ``LR_LAYOUT_AGENT=0`` — no model, no disk, on the same SHELL — so a box the pipeline pulls
  out of a wall is pulled out the same way here and the swap does not make furniture jump. Within
  a budget: a room too big to repair in seconds keeps its scanned boxes, and says so.
* **Same physics tables.** Mass is the box volume times ``OCCUPANCY_DENSITY`` (the object compiler's
  table, not the exporter's older ``DENSITY``), friction and restitution come from
  ``MATERIAL_FRICTION`` / ``MATERIAL_RESTITUTION`` through a category -> material guess, a door
  hinge has ``JOINT_DEFAULTS``. On tea_room the chairs and the dishwasher weigh exactly what
  `results/tea_room/physics_table.md` says.
* **Same free/static rule.** Chairs, stools and small tables are free; floor-standing scenery
  (tables, storage, appliances, sofas, beds) is static, as `mujoco_scene.SCENERY` makes it; anything
  hanging clear of the floor with nothing under it is fixed to the structure.

What a box CANNOT be is a table: a solid block is a table with nowhere to put your knees, and the
chairs tucked under it would start inside it and be fired across the room on the first step. So
furniture is built from primitive TEMPLATES by category — a table is a top on four legs, a chair a
seat, a back and four legs, a sofa a seat between arms and a back — sized to the scan's box and
facing the way RoomPlan says the object faces (its local -Y). Anything unrecognised stays a box.

Walls are boxes 10 cm thick grown OUTWARD from the scanned plane (the scan's 0.1 mm sheet is the
room-side face and must not move, or every unit against it is pushed into the room), with
door/window cutouts, glass in the windows and a leaf on a hinge in each door, open if RoomPlan saw
it open. Finally every movable body is checked against the compiled model itself: whatever still
starts inside something is pushed clear, a door swung back until it is free, and in the last
resort pinned — so the room loads with no initial overlaps and nothing is thrown on the first step.

Walls, floor and ceiling then wear the capture's own PHOTOS (`textures`): one textured mesh per wall
with its openings cut out, a centimetre in front of the scanned plane, the floor and the ceiling as
textured polygons just off their slabs — drawn only (group 2, no contacts), so the boxes stay the
colliders and physics does not change at all. The PNGs go in ``textures/`` beside scene.xml. Without frames and depth, or with textures
turned off (``textures=False``, ``LR_QUICK_TEXTURES=0``), the room keeps its flat colours.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from litereality_agent.models.object_generation.sim import properties
from litereality_agent.pipeline.quickroom import colours as colour_estimate
from litereality_agent.pipeline.quickroom import textures as surface_photos
from litereality_agent.pipeline.quickroom.colours import RGB, Palette, Swatch, category_rgb
from litereality_agent.pipeline.quickroom.roomplan import (
    Opening,
    RoomObject,
    RoomPlan,
    find_room_usdz,
    parse_room,
)

SCENE_NAME = "scene.xml"
REPORT_NAME = "quick.json"
TEXTURE_DIR = "textures"   # beside scene.xml: the surface photos, PNG (MuJoCo reads no JPEG)

# ── structure ────────────────────────────────────────────────────────────────────────────────────
WALL_THICKNESS = 0.10      # m, all of it outside the scanned plane
WALL_SETBACK = 0.002       # m: as `mujoco_scene._wall_segments` — off the plane so nothing z-fights
FLOOR_THICKNESS = 0.10
CEILING_THICKNESS = 0.05
STRUCTURE_MARGIN = 0.30    # m of floor and ceiling beyond the walls' extent
# The nominal wall thickness the ingest layout pass gives every RoomPlan wall
# (`layout.adapter.shell_from_scene_data(thickness=0.16)`). The repair's wall-clash test is
# measured against it, so the same number here is what makes the same boxes come back repaired
# the same way.
LAYOUT_WALL_THICKNESS = 0.16
# The repair's budget. It is a beam search in which every trial move re-checks the whole room, so
# its cost grows with the objects, the violations, the walls and the floor's outline all at once:
# tea_room (11 objects, 3 violations, an L-shaped floor) takes 1-2 s, a 36-object office 7-11 s,
# 150 objects ten minutes — while the room call waits to open. A room past the size gate is not
# tried at all; one inside it gets a hard deadline. Either way it keeps its scanned boxes (the
# separation pass still starts them clear) and quick.json says the repair was skipped; the full
# pipeline repairs them, and its version swaps in when it lands.
LAYOUT_MAX_OBJECTS = 25
LAYOUT_MAX_VIOLATIONS = 8
LAYOUT_SECONDS = 3.0

# ── openings ─────────────────────────────────────────────────────────────────────────────────────
LEAF_THICKNESS = 0.04
LEAF_CLEARANCE = 0.01      # m between a leaf and its jambs, floor and lintel
# A door's mass the way the object compiler derives it: occupancy density over the door UNIT's box,
# which is the opening times the depth of its frame (tea_room's authored door: 0.213 m^3 over a
# 0.77 x 1.98 m opening = 0.14 m deep -> 19 kg). The leaf itself is only 4 cm of that depth.
DOOR_UNIT_DEPTH = 0.14
DOOR_OPEN = math.radians(80.0)   # how far an open door starts: short of the wall it swings to
DOOR_RANGE = (0.0, math.radians(100.0))
DOOR_STEP = math.radians(5.0)
FRAME_WIDTH = 0.04         # window frame bars
GLASS_THICKNESS = 0.01
GLASS_RGBA = (0.78, 0.87, 0.95, 0.22)

# ── photos ───────────────────────────────────────────────────────────────────────────────────────
# The exporter's visual group (`model_info.VISUAL_GROUP`): drawn by the room call, never collided.
PHOTO_GROUP = 2
# m a wall's photo stands IN FRONT of the scanned plane (so `PHOTO_INSET + WALL_SETBACK` in front of
# its box). Not for the box's big face — 2 mm is plenty there — but for the faces where two of a
# wall's boxes meet (on every opening's edge, floor to ceiling): they are seen edge-on, and an
# edge-on face's interpolated depth is poor enough that, 2 mm behind the photo, it shows through as
# a hairline down the whole wall (measured in Chromium's software GL; gone at 1 cm). The floor and
# ceiling are single slabs with no such faces: `textures.FLOOR_LIFT` is enough for them.
PHOTO_INSET = 0.01
PHOTO_MIN_AREA = 1e-4      # m^2: a wall smaller than this is not worth a mesh

# ── furniture ────────────────────────────────────────────────────────────────────────────────────
FREE_KINDS = {"chair", "stool"}
SMALL_TABLE_AREA = 0.36    # m^2 of footprint: a side or coffee table moves, a dining table stays
HUNG_GAP = 0.30            # m: clear of the floor with nothing under it -> fixed (the exporter's rule)
FLOOR_GAP = 0.001          # free bodies start this far above whatever they stand on
MIN_EXTENT = 0.02          # m: no box side thinner than this
LEG_RGB: dict[str, RGB] = {"chair": (0.16, 0.16, 0.17), "stool": (0.30, 0.24, 0.18),
                           "table": (0.55, 0.55, 0.57)}
CEILING_RGB: RGB = (0.95, 0.95, 0.94)
FRAME_RGB: RGB = (0.92, 0.92, 0.90)

# The material a category is most likely made of, for the friction/restitution tables. A guess, and
# labelled as one in the report: a chair's FEET touch the floor, so wood/plastic rather than fabric.
MATERIAL_GUESS = {
    "chair": "wood", "stool": "wood", "table": "wood", "desk": "wood",
    "sofa": "fabric", "bed": "fabric",
    "sink": "laminate", "storage": "laminate", "cabinet": "laminate",
    "dishwasher": "metal", "refrigerator": "metal", "oven": "metal", "stove": "metal",
    "washer": "metal", "television": "plastic", "fireplace": "stone", "stairs": "wood",
    "bathtub": "ceramic", "toilet": "ceramic", "door": "wood", "window": "glass",
}

# ── the separation pass ──────────────────────────────────────────────────────────────────────────
OVERLAP_TOL = 0.0005       # m: deeper than this at t=0 is an overlap to remove
REPORT_TOL = 0.005         # m: the room call's (and the exporter's) "initial overlap" threshold
PUSH_MARGIN = 0.002
SEPARATION_ROUNDS = 60


# ── specs ────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Part:
    """One box of a template, in the body's frame (origin at the scan box's centre, +X along the
    object's width, front towards -Y, +Z up). ``paint`` names which of the body's colours it gets."""

    name: str
    centre: tuple[float, float, float]
    half: tuple[float, float, float]
    paint: str = "body"

    @property
    def volume(self) -> float:
        return 8.0 * self.half[0] * self.half[1] * self.half[2]


@dataclass
class Hinge:
    axis_sign: float        # +1: positive angle swings the leaf into the room about +Z
    angle: float            # radians, where it starts (0 = closed)
    damping: float
    frictionloss: float


@dataclass
class Body:
    name: str
    category: str
    kind: str                       # template name
    pos: np.ndarray                 # world position of the body frame
    yaw: float                      # radians
    parts: list[Part]
    joint: str                      # "free" | "static" | "hinge"
    why: str
    mass: float
    friction: float
    restitution: float
    material: str
    paints: dict[str, RGB] = field(default_factory=dict)
    hinge: Hinge | None = None
    merged_from: tuple[str, ...] = ()
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Scene:
    label: str
    plan: RoomPlan
    statics: list[dict]             # world geoms: {name, pos, half, yaw, rgba}
    bodies: list[Body]
    palette: Palette
    report: dict[str, Any] = field(default_factory=dict)
    # The surface photos (`textures.estimate`), or None when they were not made. `compose` draws
    # those in `photos.surfaces`; `write_textures` writes their PNGs.
    photos: surface_photos.SurfaceSet | None = None


# ── lookups ──────────────────────────────────────────────────────────────────────────────────────


def _tokens(category: str) -> str:
    return (category or "").lower()


def template_kind(obj: RoomObject, floor_z: float) -> str:
    """Which primitive template stands in for this object."""
    c = _tokens(obj.category)
    standing = obj.bottom - floor_z < 0.08
    if "chair" in c:
        return "chair" if obj.size[2] > 0.6 else "stool"
    if "stool" in c:
        return "stool"
    if "table" in c or "desk" in c:
        return "table"
    if "sofa" in c or "couch" in c:
        return "sofa"
    if "bed" in c:
        return "bed"
    if "stair" in c:
        return "stairs"
    if "bathtub" in c:
        return "bathtub"
    if "toilet" in c:
        return "toilet"
    if any(t in c for t in ("storage", "cabinet", "sink", "dishwasher", "oven", "stove", "washer")):
        if standing and 0.7 <= obj.size[2] <= 1.05:
            return "counter"
    return "box"


def density(category: str) -> float:
    """kg per m^3 of the object's box: the object compiler's own lookup over its own table, so a
    category resolves to the same density here as when TRELLIS's mesh is compiled later."""
    return float(properties._lookup(properties.OCCUPANCY_DENSITY, category,
                                    properties.OCCUPANCY_DENSITY["default"]))


def material_for(category: str) -> str:
    c = _tokens(category)
    for token, material in MATERIAL_GUESS.items():
        if token in c:
            return material
    return ""


def friction_for(material: str) -> float:
    return float(properties._lookup(properties.MATERIAL_FRICTION, material,
                                    properties.DEFAULT_FRICTION))


def restitution_for(material: str) -> float:
    return float(properties._lookup(properties.MATERIAL_RESTITUTION, material,
                                    properties.DEFAULT_RESTITUTION))


# ── templates ────────────────────────────────────────────────────────────────────────────────────


def template(kind: str, size: tuple[float, float, float]) -> list[Part]:
    """The boxes that stand in for an object of ``size`` (width, depth, height).

    No box comes out thinner than ``MIN_EXTENT`` whatever the scan measured: MuJoCo refuses to
    compile a size that is not positive, and a 2 cm "table" or a 12 cm deep sofa would otherwise
    leave a part with none — one bad box would stop the whole room from loading.
    """
    return [replace(p, half=tuple(max(MIN_EXTENT / 2.0, h) for h in p.half))
            for p in _template_parts(kind, size)]


def _template_parts(kind: str, size: tuple[float, float, float]) -> list[Part]:
    hx, hy, hz = (max(MIN_EXTENT, float(v)) / 2.0 for v in size)
    H = 2.0 * hz
    if kind == "table":
        t = min(float(np.clip(0.06 * H, 0.02, 0.04)), H / 2)
        lg = 0.025 if min(hx, hy) > 0.2 else 0.015
        inset_x, inset_y = max(0.0, hx - lg - 0.02), max(0.0, hy - lg - 0.02)
        parts = [Part("top", (0.0, 0.0, hz - t / 2), (hx, hy, t / 2), "top")]
        parts += [Part(f"leg{i}", (sx * inset_x, sy * inset_y, -t / 2), (lg, lg, hz - t / 2), "legs")
                  for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1)))]
        return parts
    if kind in ("chair", "stool"):
        st, lg = 0.05, 0.015
        seat_top = H if kind == "stool" else min(0.47, 0.55 * H)
        seat_top = max(seat_top, st + 0.05)
        bt = 0.0 if kind == "stool" else min(0.05, 0.2 * hy)
        seat_bottom = -hz + seat_top - st
        parts = [Part("seat", (0.0, -bt / 2, seat_bottom + st / 2), (hx, hy - bt / 2, st / 2))]
        if bt:
            parts.append(Part("back", (0.0, hy - bt / 2, (seat_bottom + hz) / 2),
                              (hx, bt / 2, (hz - seat_bottom) / 2)))
        leg_half = (seat_bottom + hz) / 2
        parts += [Part(f"leg{i}", (sx * (hx - lg), sy * (hy - lg), -hz + leg_half),
                       (lg, lg, leg_half), "legs")
                  for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1)))]
        return parts
    if kind == "sofa":
        seat_top = min(0.45, 0.55 * H)
        bt = min(float(np.clip(0.36 * hy, 0.12, 0.25)), hy)
        aw = float(np.clip(0.12 * hx, 0.08, 0.18)) if hx > 0.45 else 0.0
        arm_top = min(H, seat_top + 0.2)
        parts = [Part("base", (0.0, -bt / 2, -hz + seat_top / 2), (hx - aw, hy - bt / 2, seat_top / 2)),
                 Part("back", (0.0, hy - bt / 2, 0.0), (hx, bt / 2, hz))]
        if aw:
            parts += [Part(f"arm{i}", (s * (hx - aw / 2), -bt / 2, -hz + arm_top / 2),
                           (aw / 2, hy - bt / 2, arm_top / 2)) for i, s in enumerate((-1, 1))]
        return parts
    if kind == "bed":
        base = min(0.55, H)
        if H - base < 0.1:
            return [Part("base", (0.0, 0.0, 0.0), (hx, hy, hz))]
        hb = min(0.08, 0.2 * hy)
        return [Part("base", (0.0, -hb / 2, -hz + base / 2), (hx, hy - hb / 2, base / 2)),
                Part("headboard", (0.0, hy - hb / 2, 0.0), (hx, hb / 2, hz))]
    if kind == "counter":
        wt = 0.03
        return [Part("carcass", (0.0, 0.0, -wt / 2), (hx, hy, hz - wt / 2)),
                Part("worktop", (0.0, 0.0, hz - wt / 2), (hx, hy, wt / 2), "top")]
    if kind == "stairs":
        n = max(2, int(round(H / 0.18)))
        rise, run = H / n, 2 * hy / n
        return [Part(f"step{i}", (0.0, -hy + (i + 0.5) * run, -hz + (i + 1) * rise / 2),
                     (hx, run / 2, (i + 1) * rise / 2)) for i in range(n)]
    if kind == "bathtub":
        tw = min(0.06, 0.2 * min(hx, hy))
        bottom = min(0.08, 0.3 * H)
        return [Part("bottom", (0.0, 0.0, -hz + bottom / 2), (hx, hy, bottom / 2)),
                Part("side0", (0.0, -hy + tw / 2, 0.0), (hx, tw / 2, hz)),
                Part("side1", (0.0, hy - tw / 2, 0.0), (hx, tw / 2, hz)),
                Part("end0", (-hx + tw / 2, 0.0, 0.0), (tw / 2, hy - tw, hz)),
                Part("end1", (hx - tw / 2, 0.0, 0.0), (tw / 2, hy - tw, hz))]
    if kind == "toilet":
        tank = min(0.2, 0.35 * hy)
        bowl_top = min(0.42, 0.6 * H)
        return [Part("bowl", (0.0, -tank / 2, -hz + bowl_top / 2), (0.7 * hx, hy - tank / 2, bowl_top / 2)),
                Part("tank", (0.0, hy - tank / 2, 0.0), (hx, tank / 2, hz))]
    return [Part("box", (0.0, 0.0, 0.0), (hx, hy, hz))]


def sample_regions(kind: str, parts: list[Part]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Where in the object's frame the capture's pixels say something about each paint.

    Only the parts that are SEEN as surfaces: a table's top, a chair's seat and back, a counter's
    carcass and worktop. Legs are thin enough that the pixels in their boxes are mostly the floor
    behind them, so legs keep a category colour. Each region reaches a few centimetres PAST its
    part: a surface pixel sits on the part's face, and a RoomPlan box is only good to a few
    centimetres — inset instead, and a fridge whose front is 2 cm proud of its box keeps none of
    its white front and all of the dark gap beside it.
    """
    pad = colour_estimate.REGION_PAD
    regions = []
    for part in parts:
        if part.paint == "legs":
            continue
        tag = "top" if part.paint == "top" else "body"
        centre = np.asarray(part.centre, dtype=float)
        half = np.asarray(part.half, dtype=float) + pad
        if kind == "counter" and part.name == "carcass":
            # Up to 3 cm under the worktop and no higher, so the worktop's colour stays its own.
            top = part.centre[2] + part.half[2] - 0.03
            bottom = part.centre[2] - part.half[2] - pad
            centre[2], half[2] = (top + bottom) / 2, (top - bottom) / 2
        if kind == "counter" and part.name == "worktop":
            top = part.centre[2] + part.half[2] + pad
            bottom = part.centre[2] - part.half[2] - 0.03
            centre[2], half[2] = (top + bottom) / 2, (top - bottom) / 2
        regions.append((tag, centre, np.maximum(half, 0.005)))
    return regions


# ── layout: merge and repair, as ingest does it ─────────────────────────────────────────────────


def merge_counter_runs(plan: RoomPlan) -> tuple[dict[str, RoomObject], dict[str, list[str]]]:
    """Fuse genuinely overlapping counter-run boxes with ``merge_boxes``' own rules and names.

    ``auto_groups`` reads pkl-style entries — ``position`` (x, up, z), ``bbox`` (w, h, d) and a
    yaw ``rotation`` — and only ever compares two boxes in the frame of one of their yaws. Handing
    it the SHELL's (x, height, y) with the SHELL's yaw is that same computation in a consistent
    frame, so no ARKit round trip (and none of its sign conventions) is needed.
    """
    from litereality_agent.pipeline.scene_init.ingest import merge_boxes

    objects = dict(plan.objects)
    thresh, vgap, guard, _agent, enabled = merge_boxes.policy_settings()
    if not enabled:
        return objects, {}
    entries = [{"object_type": o.handle, "position": [o.center[0], o.center[2], o.center[1]],
                "bbox": [o.size[0], o.size[2], o.size[1]], "rotation": o.yaw, "file": None}
               for o in objects.values()]
    walls = ([((w.start[0], w.start[1]), (w.end[0], w.end[1])) for w in plan.walls.values()]
             if guard else [])
    groups = merge_boxes.auto_groups(entries, thresh, vgap, walls)
    by_name = {e["object_type"]: e for e in entries}
    merged: dict[str, list[str]] = {}
    for members in groups:
        name = merge_boxes.name_for(members, set(objects) | set(merged))
        union = merge_boxes._union_obb([by_name[m] for m in members])
        p, b = union["position"], union["bbox"]
        first = objects[members[0]]
        objects[name] = RoomObject(
            handle=name, category="".join(c for c in name if not c.isdigit()).strip("_").lower(),
            roomplan_category="+".join(objects[m].roomplan_category for m in members),
            center=(round(float(p[0]), 4), round(float(p[2]), 4), round(float(p[1]), 4)),
            size=(round(float(b[0]), 4), round(float(b[2]), 4), round(float(b[1]), 4)),
            yaw=first.yaw, matrix_arkit=None, merged_from=tuple(members))
        for m in members:
            del objects[m]
        merged[name] = list(members)
    return dict(sorted(objects.items())), merged


class _OverBudget(BaseException):
    """Raised into the repair when its time is up. A BaseException, so the repair's own
    ``except Exception`` around every trial move cannot swallow it."""


@contextlib.contextmanager
def _deadline(seconds: float):
    """Interrupt the block with `_OverBudget` after ``seconds``. The repair is a pure-Python search
    with no deadline of its own, so this takes a timer signal — which only the main thread of a
    Unix process receives; anywhere else the block runs unbounded and the size gate is the only
    budget. A timer someone else had running is restarted with what was left of it."""
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def ring(_signum, _frame):
        raise _OverBudget

    previous = signal.signal(signal.SIGALRM, ring)
    pending, interval = signal.setitimer(signal.ITIMER_REAL, max(seconds, 1e-3))
    started = time.monotonic()
    try:
        yield
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        finally:
            signal.signal(signal.SIGALRM, previous)
            if pending:
                signal.setitimer(signal.ITIMER_REAL,
                                 max(pending - (time.monotonic() - started), 1e-3), interval)


def repair_layout(plan: RoomPlan, objects: dict[str, RoomObject]
                  ) -> tuple[dict[str, RoomObject], dict[str, Any]]:
    """The ingest stage's deterministic layout repair (`layout.converge.solve` with the agent
    off), in memory, within the quick room's budget (`LAYOUT_SECONDS`). Never raises: a repair
    that fails or runs out of time leaves the boxes as scanned."""
    if os.environ.get("LR_LAYOUT", "1").lower() in ("0", "false", "no", ""):
        return objects, {"skipped": "$LR_LAYOUT=0"}
    before: list[str] = []
    try:
        import copy

        from litereality_agent.pipeline.scene_init.layout.adjust import check
        from litereality_agent.pipeline.scene_init.layout.converge import solve

        shell = plan.shell(wall_thickness=LAYOUT_WALL_THICKNESS, objects=objects)
        before = [str(v) for v in check(shell) if v.severity == "error"]
        if not before:
            return objects, {"before": [], "after": [], "moved": [], "resized": []}
        if len(objects) > LAYOUT_MAX_OBJECTS or len(before) > LAYOUT_MAX_VIOLATIONS:
            return objects, {"skipped": f"budget: {len(objects)} objects with {len(before)} "
                                        f"violations, more than the quick room repairs "
                                        f"({LAYOUT_MAX_OBJECTS} / {LAYOUT_MAX_VIOLATIONS})",
                             "before": before, "after": before}
        with _deadline(LAYOUT_SECONDS):
            repaired, _moves, _log, progress = solve(shell, copy.deepcopy(shell),
                                                     use_agent=False)
        after = [str(v) for v in check(repaired) if v.severity == "error"]
        out, moved, resized = dict(objects), [], []
        for handle, rec in (repaired.get("objects") or {}).items():
            obj = objects.get(handle)
            if obj is None:
                continue
            centre = tuple(round(float(v), 4) for v in rec["center"])
            size = tuple(round(float(v), 4) for v in rec["size"])
            if math.dist(centre, obj.center) > 1e-4:
                moved.append(handle)
            if any(abs(a - b) > 1e-4 for a, b in zip(size, obj.size)):
                resized.append(handle)
            if handle in moved or handle in resized:
                out[handle] = replace(obj.moved(centre, size), yaw=float(rec.get("yaw", obj.yaw)))
        return out, {"before": before, "after": after, "moved": moved, "resized": resized,
                     "fidelity_errors": progress.get("fidelity_errors", [])}
    except _OverBudget:
        return objects, {"skipped": f"budget: the repair ran past {LAYOUT_SECONDS:g} s",
                         "before": before, "after": before}
    except Exception as exc:  # noqa: BLE001 — the quick room must open even if the repair cannot
        return objects, {"error": f"{type(exc).__name__}: {exc}"}


# ── furnishing ───────────────────────────────────────────────────────────────────────────────────


def _supported(obj: RoomObject, others: dict[str, RoomObject]) -> bool:
    """`mujoco_scene._supported_from_below` on axis-aligned bounds: something under it holds it."""
    from litereality_agent.room_ops.export.mujoco_scene import _supported_from_below

    def record(o: RoomObject) -> dict:
        corners = o.footprint()
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        return {"category": o.category, "center": list(o.center),
                "bbox_min": [lo[0], lo[1], o.bottom], "bbox_max": [hi[0], hi[1], o.top]}

    records = {h: record(o) for h, o in others.items()}
    return _supported_from_below(records[obj.handle], records)


def _paints(kind: str, swatch: Swatch) -> dict[str, RGB]:
    """A template's colours. A top that was never seen on its own (a counter seen only from the
    front) takes the body's colour rather than a guess; legs always keep a category colour."""
    return {"body": swatch.rgb, "top": swatch.top or swatch.rgb,
            "legs": LEG_RGB.get(kind, LEG_RGB["chair"])}


def furnish(plan: RoomPlan, objects: dict[str, RoomObject], palette: Palette) -> list[Body]:
    bodies = []
    for obj in objects.values():
        kind = template_kind(obj, plan.floor_z)
        parts = template(kind, obj.size)
        gap = obj.bottom - plan.floor_z
        if gap > HUNG_GAP and not _supported(obj, objects):
            joint, why = "static", "hung: clear of the floor with nothing under it"
        elif kind in FREE_KINDS:
            joint, why = "free", "chairs and stools are the furniture a room really moves"
        elif kind == "table" and obj.size[0] * obj.size[1] <= SMALL_TABLE_AREA:
            joint, why = "free", f"small table (footprint <= {SMALL_TABLE_AREA} m^2)"
        else:
            joint, why = "static", "floor-standing scenery"
        material = material_for(obj.category)
        mass = max(obj.volume * density(obj.category), properties.MIN_LINK_MASS)
        centre = np.asarray(obj.center, dtype=float)
        if joint == "free" and gap < 0.08:
            # Stand it ON the floor, a millimetre clear, whatever the scan's box height said.
            centre[2] = plan.floor_z + obj.size[2] / 2.0 + FLOOR_GAP
        bodies.append(Body(
            name=obj.handle, category=obj.category, kind=kind, pos=centre,
            yaw=math.radians(obj.yaw), parts=parts, joint=joint, why=why, mass=mass,
            friction=friction_for(material), restitution=restitution_for(material),
            material=material or "default",
            paints=_paints(kind, palette.swatch(obj.handle, obj.category)),
            merged_from=obj.merged_from,
            size=tuple(float(v) for v in obj.size)))
    return bodies


def _room_side(plan: RoomPlan, wall_name: str) -> np.ndarray:
    """The unit normal of a wall pointing INTO the room (`textures.inward_normal`: the photos and
    the boxes must agree on which side of a wall is the room)."""
    return surface_photos.inward_normal(plan, wall_name)


def door_leaf(plan: RoomPlan, opening: Opening, palette: Palette) -> Body:
    """A leaf on a vertical hinge at one jamb, swinging into the room.

    RoomPlan records neither the hinge side nor the swing. The hinge goes on the jamb nearer the
    end of its wall — a door is hung so that it opens back against the adjacent wall rather than
    into the middle of the room — and it swings inwards, which is where the scan saw the room.

    The leaf fills the hole `structure` actually cuts, which is the opening clipped to its wall
    and to floor and ceiling: RoomPlan does draw doors running past the end of their wall and
    sills a few centimetres under the floor, and a leaf hinged beyond the wall or reaching through
    the floor could only be pinned shut.
    """
    wall = plan.walls[opening.wall]
    along = wall.along
    inward = _room_side(plan, opening.wall)
    t0, t1 = (min(wall.length, max(0.0, t)) for t in opening.span)
    z0, z1 = max(0.0, opening.sill), min(plan.height, opening.sill + opening.height)
    near_end = (wall.length - t1) < t0
    hinge_t = t1 - LEAF_CLEARANCE if near_end else t0 + LEAF_CLEARANCE
    leaf_dir = -along if near_end else along                 # from the hinge across the opening
    width = max(0.1, (t1 - t0) - 2 * LEAF_CLEARANCE)
    height = max(0.3, (z1 - z0) - 2 * LEAF_CLEARANCE)
    # Leaf frame: origin on the hinge axis, +X across the opening. The leaf's thickness goes on
    # the OUTWARD side of the scanned plane, so closed it sits inside the wall's cutout with its
    # room-side face on the plane, and swinging it never sweeps the jamb it hangs on.
    x_axis = np.array([leaf_dir[0], leaf_dir[1], 0.0])
    yaw = math.atan2(x_axis[1], x_axis[0])
    y_axis = np.array([-math.sin(yaw), math.cos(yaw)])          # +Z x +X
    outward_is_y = float(np.dot(y_axis, -inward)) > 0
    # A positive angle about +Z turns +X towards +Y; the joint axis is flipped when +Y is out of
    # the room, so that opening is always a positive angle (and 0 is closed).
    sign = -1.0 if outward_is_y else 1.0
    offset_y = (LEAF_THICKNESS / 2 + WALL_SETBACK) * (1.0 if outward_is_y else -1.0)
    hinge = np.asarray(wall.start) + along * hinge_t
    pos = np.array([hinge[0], hinge[1], plan.floor_z + z0 + LEAF_CLEARANCE])
    parts = [Part("leaf", (width / 2, offset_y, height / 2), (width / 2, LEAF_THICKNESS / 2, height / 2))]
    mass = opening.width * opening.height * DOOR_UNIT_DEPTH * density("door")
    material = material_for("door")
    defaults = properties.JOINT_DEFAULTS["revolute"]
    angle = DOOR_OPEN if opening.is_open else 0.0
    rgb = palette.doors.get(opening.handle, colour_estimate.DOOR_RGB)
    return Body(name=opening.handle, category="door", kind="door_leaf", pos=pos, yaw=yaw,
                parts=parts, joint="hinge", why="door leaf on its hinge", mass=mass,
                friction=friction_for(material), restitution=restitution_for(material),
                material=material, paints={"body": rgb},
                hinge=Hinge(axis_sign=sign, angle=angle, damping=float(defaults["damping"]),
                            frictionloss=float(defaults["friction"])),
                size=(opening.width, LEAF_THICKNESS, opening.height))


def _wall_pieces(length: float, height: float, cuts: list[tuple[float, float, float, float]]
                 ) -> list[tuple[float, float, float, float]]:
    """The solid rectangles ``(t0, t1, z0, z1)`` of a wall face once its openings are cut out.

    Column decomposition: split the wall at every opening edge, then each column is solid except
    where an opening spans it. Exact for any number of openings, overlapping ones included, and
    every piece is a box.
    """
    ts = sorted({0.0, length, *[min(length, max(0.0, v)) for c in cuts for v in c[:2]]})
    pieces: list[tuple[float, float, float, float]] = []
    for a, b in zip(ts, ts[1:]):
        if b - a < 1e-4:
            continue
        mid = (a + b) / 2
        z = 0.0
        for z0, z1 in sorted((c[2], c[3]) for c in cuts if c[0] < mid < c[1]):
            if z0 > z + 1e-4:
                pieces.append((a, b, z, z0))
            z = max(z, z1)
        if z < height - 1e-4:
            pieces.append((a, b, z, height))
    merged: list[tuple[float, float, float, float]] = []
    for p in pieces:
        if merged and abs(merged[-1][1] - p[0]) < 1e-6 and merged[-1][2:] == p[2:]:
            merged[-1] = (merged[-1][0], p[1], p[2], p[3])
        else:
            merged.append(p)
    return merged


def structure(plan: RoomPlan, palette: Palette,
              objects: dict[str, RoomObject] | None = None) -> list[dict]:
    """Walls (with cutouts and window glass), floor and ceiling, as static world geoms. The floor
    reaches under every one of ``objects`` too: a chair seen through a doorway stands outside the
    walls, and with no floor under it it would fall for ever (and be respawned, over and over)."""
    geoms: list[dict] = []
    fz, height = plan.floor_z, plan.height
    wall_rgba = (*palette.wall, 1.0)
    T = WALL_THICKNESS
    for wall in plan.walls.values():
        length = wall.length
        if length < 1e-3:
            continue
        along = wall.along
        out = -_room_side(plan, wall.name)
        start = np.asarray(wall.start)
        openings = [op for op in plan.openings.values() if op.wall == wall.name]
        cuts = surface_photos.opening_cuts(plan, wall.name)
        # Close the OUTSIDE of a convex corner: two walls grown outward from their planes leave a
        # square notch there. At a reflex corner the same extension would poke into the room.
        grow = {}
        for end, t, direction in (("start", 0.0, -1.0), ("end", length, 1.0)):
            probe = start + along * (t + direction * T / 2) + out * (WALL_SETBACK + T / 2)
            convex = not bool(plan.on_floor(probe[None, :])[0])
            grow[end] = T + WALL_SETBACK if convex else 0.0
        for i, (t0, t1, z0, z1) in enumerate(_wall_pieces(length, height, cuts)):
            if t0 <= 1e-6:
                t0 -= grow["start"]
            if t1 >= length - 1e-6:
                t1 += grow["end"]
            mid = start + along * ((t0 + t1) / 2) + out * (WALL_SETBACK + T / 2)
            geoms.append({"name": f"{wall.name}_{i}", "pos": (mid[0], mid[1], fz + (z0 + z1) / 2),
                          "half": ((t1 - t0) / 2, T / 2, (z1 - z0) / 2),
                          "yaw": math.atan2(along[1], along[0]), "rgba": wall_rgba})
        for op in openings:
            if op.kind != "window":
                continue
            t0, t1 = (min(length, max(0.0, t)) for t in op.span)
            z0, z1 = fz + max(0.0, op.sill), fz + min(height, op.sill + op.height)
            if t1 - t0 < MIN_EXTENT or z1 - z0 < MIN_EXTENT:
                continue                # nothing of it is left inside the wall to glaze
            centre_t = (t0 + t1) / 2
            mid = start + along * centre_t + out * (WALL_SETBACK + T / 2)
            yaw = math.atan2(along[1], along[0])
            w2, h2 = (t1 - t0) / 2, (z1 - z0) / 2
            zc = (z0 + z1) / 2
            geoms.append({"name": f"{op.handle}_glass", "pos": (mid[0], mid[1], zc),
                          "half": (w2, GLASS_THICKNESS / 2, h2), "yaw": yaw, "rgba": GLASS_RGBA})
            f = FRAME_WIDTH / 2
            frame_rgba = (*FRAME_RGB, 1.0)
            for name, dt, dz, hx, hz in (("frame_bottom", 0.0, -h2 + f, w2, f),
                                         ("frame_top", 0.0, h2 - f, w2, f),
                                         ("frame_left", -w2 + f, 0.0, f, h2 - 2 * f),
                                         ("frame_right", w2 - f, 0.0, f, h2 - 2 * f)):
                if min(hx, hz) < MIN_EXTENT / 2:
                    continue            # a window too small for this bar keeps just its glass
                p = mid + along * dt
                geoms.append({"name": f"{op.handle}_{name}", "pos": (p[0], p[1], zc + dz),
                              "half": (hx, T / 2, hz), "yaw": yaw, "rgba": frame_rgba})
    pts = [p for w in plan.walls.values() for p in (w.start, w.end)]
    pts += [tuple(p) for poly in plan.outlines() for p in poly]
    pts += [tuple(c) for o in (objects or {}).values() for c in o.footprint()]
    lo = np.min(pts, axis=0) - STRUCTURE_MARGIN
    hi = np.max(pts, axis=0) + STRUCTURE_MARGIN
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    geoms.append({"name": "floor", "pos": (mid[0], mid[1], fz - FLOOR_THICKNESS / 2),
                  "half": (half[0], half[1], FLOOR_THICKNESS / 2), "yaw": 0.0,
                  "rgba": (*palette.floor, 1.0)})
    geoms.append({"name": "ceiling", "pos": (mid[0], mid[1], plan.ceiling_z + CEILING_THICKNESS / 2),
                  "half": (half[0], half[1], CEILING_THICKNESS / 2), "yaw": 0.0,
                  "rgba": (*CEILING_RGB, 1.0)})
    return geoms


# ── photos on the walls, floor and ceiling ───────────────────────────────────────────────────────


def triangulate(polygon: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping of a simple polygon into counter-clockwise triangles (indices into it). A
    floor outline is a few dozen corners at most, and L- and U-shaped rooms are not convex, so no
    fan will do. A polygon with no ear left (self-intersecting) gives [] — the floor stays flat."""
    pts = np.asarray(polygon, dtype=float)
    n = len(pts)
    if n < 3:
        return []
    signed = 0.5 * float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                                - np.roll(pts[:, 0], -1) * pts[:, 1]))
    ring = list(range(n)) if signed > 0 else list(range(n))[::-1]

    def cross(o: int, a: int, b: int) -> float:
        return float((pts[a, 0] - pts[o, 0]) * (pts[b, 1] - pts[o, 1])
                     - (pts[a, 1] - pts[o, 1]) * (pts[b, 0] - pts[o, 0]))

    tris: list[tuple[int, int, int]] = []
    while len(ring) > 3:
        m = len(ring)
        for k in range(m):
            a, b, c = ring[k - 1], ring[k], ring[(k + 1) % m]
            if cross(a, b, c) <= 1e-12:
                continue                                    # reflex (or flat): not an ear
            if any(cross(a, b, j) >= 0 and cross(b, c, j) >= 0 and cross(c, a, j) >= 0
                   for j in ring if j not in (a, b, c)):
                continue                                    # another corner inside it
            tris.append((a, b, c))
            ring.pop(k)
            break
        else:
            flat = [k for k in range(m)
                    if abs(cross(ring[k - 1], ring[k], ring[(k + 1) % m])) <= 1e-12]
            if not flat:
                return []
            ring.pop(flat[0])                              # a corner on a straight edge
    if len(ring) == 3 and cross(*ring) > 1e-12:
        tris.append((ring[0], ring[1], ring[2]))
    return tris


def _wall_grid(length: float, height: float, cuts: list[tuple[float, float, float, float]]
               ) -> tuple[np.ndarray, np.ndarray]:
    """A wall face minus its openings as ONE welded quad grid: ``(corners (n, 2) as (t, z),
    quads (m, 4) counter-clockwise in (t, z))``.

    The solid area is exactly `_wall_pieces`', but split on every opening edge in BOTH directions,
    so neighbouring quads always share whole edges. Pieces drawn as separate quads meet in
    T-junctions (a full-height column beside one cut in two by a window), and a T-junction
    rasterises with a hairline crack: the box behind shows through as a thin bright line from
    floor to ceiling."""
    def breaks(values: list[float], top: float) -> list[float]:
        out: list[float] = []
        for v in sorted({0.0, top, *[min(top, max(0.0, v)) for v in values]}):
            if not out or v - out[-1] >= 1e-3:          # a 0.1 mm door sill is no row of its own
                out.append(v)
        out[-1] = top
        return out

    ts = breaks([v for c in cuts for v in c[:2]], length)
    zs = breaks([v for c in cuts for v in c[2:]], height)
    index: dict[tuple[int, int], int] = {}
    quads = []
    for i in range(len(ts) - 1):
        tm = (ts[i] + ts[i + 1]) / 2
        for j in range(len(zs) - 1):
            zm = (zs[j] + zs[j + 1]) / 2
            if any(c[0] < tm < c[1] and c[2] < zm < c[3] for c in cuts):
                continue                                    # inside an opening
            corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            quads.append([index.setdefault(k, len(index)) for k in corners])
    pts = np.zeros((len(index), 2))
    for (i, j), k in index.items():
        pts[k] = (ts[i], zs[j])
    return pts, np.asarray(quads, dtype=np.int64).reshape(-1, 4)


def photo_geoms(plan: RoomPlan, photos: surface_photos.SurfaceSet | None) -> list[dict]:
    """The drawn-only meshes the photos go on: ``{name, surface, verts (n, 3), faces (m, 3),
    uv (n, 2)}`` in world coordinates, every face wound to face into the room.

    A wall's photo is one welded grid over its solid area (`_wall_grid`: the area its boxes
    cover, so a door or a window is a hole in the photo exactly where it is a hole in the wall),
    `PHOTO_INSET` in front of the scanned plane. The floor and the ceiling are their outlines
    triangulated, just off their slabs. Each vertex's uv is `Surface.uv` of its position, so the image lands where it
    was photographed whatever the shape."""
    if photos is None or not photos.surfaces:
        return []
    out: list[dict] = []
    fz, height = plan.floor_z, plan.height
    up = np.array([0.0, 0.0, 1.0])
    for wall in plan.walls.values():
        surface = photos.surfaces.get(wall.name)
        if surface is None or surface.kind != "wall":
            continue
        length = wall.length
        along = np.r_[wall.along, 0.0]
        inward = np.r_[_room_side(plan, wall.name), 0.0]
        start = np.array([wall.start[0], wall.start[1], fz])
        tz, quads = _wall_grid(length, height, surface_photos.opening_cuts(plan, wall.name))
        if not len(quads) or length * height < PHOTO_MIN_AREA:
            continue
        uv = surface.uv(start + tz[:, :1] * along + tz[:, 1:] * up)
        # Set in front of the plane, two photos no longer meet at a corner that juts into the room
        # (an L's inner corner): reach past such an end by the inset, so they close again. The
        # same probe `structure` uses to tell the two kinds of corner apart.
        t = tz[:, 0].copy()
        for end, at, direction in (("start", 0.0, -1.0), ("end", length, 1.0)):
            probe = (np.asarray(wall.start) + wall.along * (at + direction * WALL_THICKNESS / 2)
                     - inward[:2] * (WALL_SETBACK + WALL_THICKNESS / 2))
            if bool(plan.on_floor(probe[None, :])[0]):
                t[np.isclose(tz[:, 0], at)] += direction * PHOTO_INSET
        verts = start + t[:, None] * along + tz[:, 1:] * up + inward * PHOTO_INSET
        faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        # (t, z) counter-clockwise faces along x up; the room must be on that side
        if float(np.dot(np.cross(along, up), inward)) < 0:
            faces = faces[:, ::-1]
        out.append({"name": f"{wall.name}_photo", "surface": surface.name, "verts": verts,
                    "faces": faces, "uv": np.clip(uv, 0.0, 1.0)})
    lift = surface_photos.FLOOR_LIFT
    for kind, z, down in (("floor", fz + lift, False), ("ceiling", plan.ceiling_z - lift, True)):
        surface = photos.surfaces.get(kind)
        if surface is None:
            continue
        for k, poly in enumerate(plan.outlines()):
            tris = triangulate(poly)
            if not tris:
                continue
            verts = np.column_stack([poly, np.full(len(poly), z)])
            faces = np.asarray(tris, dtype=np.int64)
            out.append({"name": f"{kind}_photo_{k}", "surface": surface.name, "verts": verts,
                        "faces": faces[:, ::-1] if down else faces, "uv": surface.uv(verts)})
    return out


def write_textures(scene: Scene, out_dir: Path | str) -> list[Path]:
    """Write the scene's surface photos as ``textures/<surface>.png`` under ``out_dir`` (what
    `compose` points its textures at) and return the paths; none when there are no photos."""
    if scene.photos is None or not scene.photos.surfaces:
        return []
    folder = Path(out_dir) / TEXTURE_DIR
    return [surface_photos.save(s, folder) for s in scene.photos.surfaces.values()
            if s.image is not None]


# ── MJCF ─────────────────────────────────────────────────────────────────────────────────────────


def _f(values) -> str:
    return " ".join(f"{float(v):.5g}" if abs(float(v)) < 1e4 else f"{float(v):.1f}"
                    for v in values)


def _quat(yaw: float) -> str:
    return _f((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


class _Materials:
    """One material per distinct colour: `geometry` merges static geometry per material, so the
    fewer there are the fewer draw calls a phone has to make."""

    def __init__(self, asset: ET.Element) -> None:
        self.asset, self.names = asset, {}

    def __call__(self, rgba) -> str:
        key = tuple(round(float(v), 3) for v in rgba)
        if key not in self.names:
            name = "c" + "".join(f"{int(round(v * 255)):02x}" for v in key)
            ET.SubElement(self.asset, "material", name=name, rgba=_f(key),
                          specular="0.2", shininess="0.3")
            self.names[key] = name
        return self.names[key]


def _fixed(values, digits: int) -> str:
    return " ".join(f"{float(v):.{digits}f}".rstrip("0").rstrip(".") or "0" for v in values)


def compose(scene: Scene, *, poses: dict[str, np.ndarray] | None = None,
            angles: dict[str, float] | None = None, textured: bool = True) -> str:
    """The MJCF text. ``poses``/``angles`` override where free bodies and door leaves start.

    With ``textured`` (and photos in the scene) the walls, floor and ceiling also get their photo
    meshes, whose textures are FILES (``textures/<surface>.png``, `write_textures`): such a text
    compiles only from a scene.xml in the folder they were written to. ``textured=False`` is the
    same room without them — identical physics, compilable from the string alone."""
    poses, angles = poses or {}, angles or {}
    root = ET.Element("mujoco", model=f"{scene.label}_quick")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.36", diffuse="0.4 0.4 0.4",
                  specular="0.05 0.05 0.05")
    default = ET.SubElement(root, "default")
    # The exporter's contact defaults (`mujoco_scene.export`), so a swap to the full room changes
    # what the objects are, not how contacts feel.
    ET.SubElement(default, "geom", condim="4", friction="0.55 0.03 0.005",
                  solref="0.01 1", solimp="0.9 0.95 0.001")
    asset = ET.SubElement(root, "asset")
    material = _Materials(asset)
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="sun", directional="true", pos="0 0 5",
                  dir="0.3 0.2 -1", diffuse="0.6 0.6 0.6", castshadow="false")
    for g in scene.statics:
        ET.SubElement(world, "geom", name=g["name"], type="box", pos=_f(g["pos"]),
                      size=_f(g["half"]), quat=_quat(g["yaw"]), material=material(g["rgba"]))
    if textured:
        _compose_photos(scene, asset, world)
    for b in scene.bodies:
        pos = poses.get(b.name, b.pos)
        yaw = b.yaw
        if b.hinge is not None:
            angle = angles.get(b.name, b.hinge.angle)
            yaw += b.hinge.axis_sign * angle
        body = ET.SubElement(world, "body", name=b.name, pos=_f(pos), quat=_quat(yaw))
        if b.joint == "free":
            ET.SubElement(body, "freejoint", name=f"{b.name}_free")
        elif b.joint == "hinge" and b.hinge is not None:
            ET.SubElement(body, "joint", name=f"{b.name}_hinge", type="hinge",
                          axis=_f((0.0, 0.0, b.hinge.axis_sign)), range=_f(DOOR_RANGE),
                          ref=f"{angles.get(b.name, b.hinge.angle):.6g}",
                          damping=f"{b.hinge.damping:.4g}",
                          frictionloss=f"{b.hinge.frictionloss:.4g}", armature="0.002")
        # Mass spread over the template's boxes by volume: MuJoCo then puts the centre of mass and
        # the inertia where the parts are (a chair's low, under its seat), and the total is exact.
        dens = b.mass / max(sum(p.volume for p in b.parts), 1e-9)
        for p in b.parts:
            ET.SubElement(body, "geom", name=f"{b.name}_{p.name}", type="box", pos=_f(p.centre),
                          size=_f(p.half), density=f"{dens:.6g}",
                          friction=f"{b.friction:.3g} 0.03 0.005",
                          material=material((*b.paints.get(p.paint, b.paints["body"]), 1.0)))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _compose_photos(scene: Scene, asset: ET.Element, world: ET.Element) -> None:
    """One texture and material per photographed surface; one mesh and drawn-only geom per wall,
    and per floor (and ceiling) outline.

    The material is plain white so the photo shows as photographed, and matte (painted plaster,
    not gloss). ``inertia="shell"``: a flat mesh has no volume, which MuJoCo otherwise refuses even
    on a geom that has no mass to give."""
    geoms = photo_geoms(scene.plan, scene.photos)
    if not geoms:
        return
    for name in dict.fromkeys(g["surface"] for g in geoms):
        surface = scene.photos.surfaces[name]
        ET.SubElement(asset, "texture", name=f"photo_{name}", type="2d",
                      file=f"{TEXTURE_DIR}/{surface_photos.texture_file(surface)}")
        ET.SubElement(asset, "material", name=f"photo_{name}", texture=f"photo_{name}",
                      rgba="1 1 1 1", specular="0.1", shininess="0.1")
    for g in geoms:
        ET.SubElement(asset, "mesh", name=g["name"], vertex=_fixed(g["verts"].reshape(-1), 4),
                      face=" ".join(str(int(i)) for i in g["faces"].reshape(-1)),
                      texcoord=_fixed(g["uv"].reshape(-1), 5), inertia="shell")
        ET.SubElement(world, "geom", name=g["name"], type="mesh", mesh=g["name"],
                      material=f"photo_{g['surface']}", contype="0", conaffinity="0",
                      group=str(PHOTO_GROUP))


# ── the separation pass ──────────────────────────────────────────────────────────────────────────


def _overlaps(model, data, tol: float) -> list[tuple[int, int, float, np.ndarray, int, int]]:
    """Contacts deeper than ``tol``: (body1, body2, depth, normal 1->2, geom1, geom2)."""
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < -tol:
            g1, g2 = int(c.geom1), int(c.geom2)
            out.append((int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2]), -float(c.dist),
                        np.array(c.frame[:3], dtype=float), g1, g2))
    return out


def separate(scene: Scene) -> dict[str, Any]:
    """Make the room start at rest: no movable body inside anything.

    The scan's boxes overlap — a chair pushed in under a table reaches its top, a box crosses the
    wall it stands against — and MuJoCo resolves an overlap at t=0 by firing the bodies apart. The
    compiled model itself is the judge here, so the test is exactly the one the room call's own
    settle pass and the exporter's "initial interpenetrations" gate apply:

    * a free body is pushed HORIZONTALLY out along the contact normal, half each when two free
      bodies meet. It is lifted only out of something that can hold it up: a static body, or a
      free one it stands ON. Two equal chairs side by side meet seat to seat, where the shallowest
      way out is vertical — lift one and it starts on top of the other and topples off seconds
      into the call — so free bodies beside each other move apart along their centre line;
    * an open door that sweeps through something fixed is swung back towards closed until clear;
    * whatever is still stuck after that is pinned — a static body cannot be thrown — and listed.
    """
    import mujoco

    origin = {b.name: np.asarray(b.pos, dtype=float).copy() for b in scene.bodies}
    pinned: list[str] = []
    for _attempt in range(3):
        model = mujoco.MjModel.from_xml_string(compose(scene, textured=False))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        by_id = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b.name): b
                 for b in scene.bodies}
        free = {i for i, b in by_id.items() if b.joint == "free"}
        doors = {i for i, b in by_id.items() if b.joint == "hinge"}
        qadr = {i: int(model.jnt_qposadr[model.body_jntadr[i]]) for i in free | doors}
        # Each body's bottom and top relative to its origin (parts are only ever yawed).
        span = {i: (min(p.centre[2] - p.half[2] for p in b.parts),
                    max(p.centre[2] + p.half[2] for p in b.parts)) for i, b in by_id.items()}

        def holds_up(me: int, other: int) -> bool:
            if other in doors:
                return False
            if other not in free:
                return True
            # ON it: my bottom above its middle (a stool on a side table), not beside it
            return float(data.xpos[me][2]) + span[me][0] >= \
                float(data.xpos[other][2]) + (span[other][0] + span[other][1]) / 2

        stuck: set[int] = set()
        for _round in range(SEPARATION_ROUNDS):
            pushes: dict[int, dict[int, np.ndarray]] = {}
            swing: set[int] = set()
            for b1, b2, depth, normal, _g1, _g2 in _overlaps(model, data, OVERLAP_TOL):
                for me, other, n in ((b1, b2, -normal), (b2, b1, normal)):
                    if me in doors and other not in free:
                        swing.add(me)
                    if me not in free:
                        continue
                    if n[2] > 0.7 and holds_up(me, other):  # standing in what holds it up: lift
                        step = np.array([0.0, 0.0, depth + PUSH_MARGIN])
                    else:
                        h = np.array([n[0], n[1], 0.0])
                        if np.linalg.norm(h) < 0.3:         # a near-vertical normal: go sideways
                            h = data.xpos[me] - data.xpos[other]
                            h[2] = 0.0
                        if np.linalg.norm(h) < 1e-9:
                            h = np.array([1.0, 0.0, 0.0])
                        share = 0.5 if other in free else 1.0
                        step = h / np.linalg.norm(h) * (depth + PUSH_MARGIN) * share
                    # One push per neighbour (its deepest contact), summed over neighbours.
                    have = pushes.setdefault(me, {}).get(other)
                    if have is None or np.linalg.norm(step) > np.linalg.norm(have):
                        pushes[me][other] = step
            if not pushes and not swing:
                break
            for me, steps in pushes.items():
                data.qpos[qadr[me]:qadr[me] + 3] += np.sum(list(steps.values()), axis=0)
            for me in swing:
                adr = qadr[me]
                if data.qpos[adr] > 1e-9:
                    data.qpos[adr] = max(0.0, float(data.qpos[adr]) - DOOR_STEP)
                else:
                    stuck.add(me)                           # closed and still blocked
            mujoco.mj_forward(model, data)
        for i in free:
            by_id[i].pos = data.qpos[qadr[i]:qadr[i] + 3].copy()
        for i in doors:
            by_id[i].hinge.angle = float(data.qpos[qadr[i]])
        left = stuck | {b for pair in _overlaps(model, data, OVERLAP_TOL) for b in pair[:2]
                        if b in free or b in doors}
        if not left:
            break
        for i in sorted(left):
            body = by_id[i]
            if body.hinge is not None:                  # keep it where it was swung to
                body.yaw += body.hinge.axis_sign * body.hinge.angle
                body.hinge = None
            body.joint, body.why = "static", "pinned: could not be separated from its neighbours"
            pinned.append(body.name)
    moved = {b.name: round(float(np.linalg.norm(np.asarray(b.pos) - origin[b.name])), 4)
             for b in scene.bodies
             if float(np.linalg.norm(np.asarray(b.pos) - origin[b.name])) > 1e-4}
    return {"separated_m": moved, "pinned": pinned,
            "door_angles_deg": {b.name: round(math.degrees(b.hinge.angle), 1)
                                for b in scene.bodies if b.hinge is not None}}


def initial_overlaps(xml, tol: float = REPORT_TOL) -> list[dict]:
    """What the room call's settle pass would have to fight: contacts deeper than ``tol`` at t=0.
    ``xml`` is the MJCF text or an already compiled ``MjModel``."""
    import mujoco

    model = xml if isinstance(xml, mujoco.MjModel) else mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def name(b: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "world"

    worst: dict[tuple[str, str], float] = {}
    for b1, b2, depth, _n, _g1, _g2 in _overlaps(model, data, tol):
        key = tuple(sorted((name(b1), name(b2))))
        worst[key] = max(worst.get(key, 0.0), depth)
    return [{"bodies": f"{a} <-> {b}", "mm": round(-d * 1000, 1)}
            for (a, b), d in sorted(worst.items(), key=lambda kv: -kv[1])]


# ── entry points ─────────────────────────────────────────────────────────────────────────────────


def textures_enabled(textures: bool = True) -> bool:
    """``textures``, unless ``$LR_QUICK_TEXTURES`` turns them off (0/false/no/off)."""
    env = os.environ.get("LR_QUICK_TEXTURES", "").strip().lower()
    return bool(textures) and env not in ("0", "false", "no", "off")


def build_scene(scan_dir: Path | str, *, label: str | None = None, colours: bool = True,
                timings: dict[str, float] | None = None, textures: bool = True) -> Scene:
    """Everything up to (not including) writing files: parse, lay out, furnish, separate — and,
    with ``textures`` (and ``colours``: both read the capture's frames), photograph the walls,
    floor and ceiling.

    The photos take a second or two and depend only on the room and its laid-out boxes, so they
    are made on a worker thread while the colours, the furniture and the separation pass are
    worked out here; the layout repair (which has a hard deadline of its own) runs before, alone.
    """
    timings = {} if timings is None else timings
    scan_dir = Path(scan_dir)
    usdz = find_room_usdz(scan_dir) if scan_dir.is_dir() else scan_dir
    if usdz is None or not usdz.is_file():
        raise FileNotFoundError(f"no RoomPlan room.usdz in {scan_dir} "
                                "(looked for room.usdz and roomplan/room.usdz)")
    if scan_dir.is_file():
        scan_dir = scan_dir.parent
    label = label or (scan_dir.parent.name if scan_dir.name == "roomplan" else scan_dir.name)

    t = time.perf_counter()
    plan = parse_room(usdz)
    timings["parse"] = time.perf_counter() - t
    if not plan.walls:
        raise ValueError(f"{usdz} has no walls; RoomPlan's parametric export is needed")

    t = time.perf_counter()
    objects, merged = merge_counter_runs(plan)
    objects, layout = repair_layout(plan, objects)
    layout["merged"] = merged
    timings["layout"] = time.perf_counter() - t

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="quick-photos")
    try:
        photo_job = None
        if colours and textures_enabled(textures):
            photo_job = pool.submit(surface_photos.estimate, scan_dir,
                                    replace(plan, objects=objects), objects=objects)
        scene = _furnished(scan_dir, label, plan, objects, layout, colours, timings)
        if photo_job is None:
            scene.photos = surface_photos.SurfaceSet(
                note="textures off" if colours else "colours off: the capture's frames are not read")
        else:
            t = time.perf_counter()
            scene.photos = photo_job.result()
            timings["textures"] = scene.photos.seconds
            timings["textures_wait"] = time.perf_counter() - t
    finally:
        pool.shutdown(wait=False)
    return scene


def _furnished(scan_dir: Path, label: str, plan: RoomPlan, objects: dict[str, RoomObject],
               layout: dict[str, Any], colours: bool, timings: dict[str, float]) -> Scene:
    """`build_scene` from the laid-out boxes on: colours, templates, structure, separation."""
    t = time.perf_counter()
    kinds = {h: template_kind(o, plan.floor_z) for h, o in objects.items()}
    if colours:
        regions = {h: sample_regions(kinds[h], template(kinds[h], o.size))
                   for h, o in objects.items()}
        palette = colour_estimate.estimate(scan_dir, replace(plan, objects=objects),
                                           regions=regions)
    else:
        palette = Palette(objects={h: Swatch(category_rgb(o.category)) for h, o in objects.items()},
                          note="colours off")
    timings["colours"] = time.perf_counter() - t

    t = time.perf_counter()
    bodies = furnish(plan, objects, palette)
    bodies += [door_leaf(plan, op, palette) for op in plan.openings.values() if op.kind == "door"]
    scene = Scene(label=label, plan=plan, statics=structure(plan, palette, objects),
                  bodies=bodies, palette=palette)
    separation = separate(scene)
    timings["furnish_and_separate"] = time.perf_counter() - t
    scene.report = {"layout": layout, **separation}
    return scene


def write_quick_scene(scan_dir: Path | str, out_dir: Path | str, *, label: str | None = None,
                      colours: bool = True, textures: bool = True) -> Path:
    """Build the quick room for one scan into ``out_dir``; returns the ``scene.xml`` path.

    Writes ``scene.xml``, the surface photos it draws on its walls, floor and ceiling as
    ``textures/*.png`` (none with ``textures=False`` or without frames: then it is
    self-contained, boxes and materials only), and ``quick.json``, the report of what was decided
    and why — every body's template, joint, mass, friction and material guess, the layout repair's
    moves, what had to be separated or pinned, the colours and photos and where they came from.

    The textures are written first and the scene last (atomically), so a room call watching the
    folder never loads a scene whose textures are not there yet; the textured scene is compiled
    once before it replaces the old one, and if it does not compile the room goes out with flat
    walls rather than not at all.
    """
    import mujoco

    started = time.perf_counter()
    timings: dict[str, float] = {}
    scene = build_scene(scan_dir, label=label, colours=colours, timings=timings,
                        textures=textures)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    written = write_textures(scene, out_dir)
    path = out_dir / SCENE_NAME
    tmp = path.with_suffix(".xml.tmp")
    xml = compose(scene)
    tmp.write_text(xml, encoding="utf-8")
    model = None
    if written:
        try:
            model = mujoco.MjModel.from_xml_path(str(tmp))
        except Exception as exc:  # noqa: BLE001 — flat walls beat no room
            scene.photos.note = f"photos dropped, the textured scene did not compile: {exc}"
            scene.photos.surfaces = {}
            written = []
            xml = compose(scene, textured=False)
            tmp.write_text(xml, encoding="utf-8")
    if model is None:
        model = mujoco.MjModel.from_xml_string(xml)
    os.replace(tmp, path)        # a watcher never reads half a scene
    _prune_textures(out_dir, keep={p.name for p in written})
    timings["write"] = time.perf_counter() - t
    overlaps = initial_overlaps(model)
    timings["total"] = time.perf_counter() - started
    report = summary(scene, overlaps, timings)
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _prune_textures(out_dir: Path, keep: set[str]) -> None:
    """Remove photos an earlier build of this folder left that the new scene does not use (a wall
    a re-scan no longer has; all of them when textures are off)."""
    folder = out_dir / TEXTURE_DIR
    if not folder.is_dir():
        return
    for old in folder.iterdir():
        if old.is_file() and old.name.endswith((".png", ".png.tmp")) and old.name not in keep:
            with contextlib.suppress(OSError):
                old.unlink()
    with contextlib.suppress(OSError):
        folder.rmdir()           # only if now empty


def summary(scene: Scene, overlaps: list[dict], timings: dict[str, float]) -> dict[str, Any]:
    plan = scene.plan
    return {
        "stage": "quick",
        "label": scene.label,
        "source": str(plan.source),
        "floor_z": plan.floor_z,
        "ceiling_z": plan.ceiling_z,
        "walls": len(plan.walls),
        "openings": {h: op.kind for h, op in plan.openings.items()},
        "skipped": plan.skipped,
        "warnings": plan.warnings,
        "bodies": {
            b.name: {"category": b.category, "template": b.kind, "joint": b.joint, "why": b.why,
                     "mass_kg": round(b.mass, 3), "friction": b.friction,
                     "restitution": b.restitution, "material_guess": b.material,
                     "size": [round(v, 4) for v in b.size],
                     **({"merged_from": list(b.merged_from)} if b.merged_from else {})}
            for b in scene.bodies},
        "initial_overlaps": overlaps,
        **scene.report,
        "colours": scene.palette.summary(),
        "textures": (scene.photos.summary() if scene.photos is not None
                     else {"source": "none", "note": "textures off"}),
        "seconds": {k: round(v, 3) for k, v in timings.items()},
        "note": "estimated physics: mass = box volume x OCCUPANCY_DENSITY[category]; friction and "
                "restitution from a category -> material guess; MuJoCo ignores restitution",
    }
