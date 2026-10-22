"""physics.py — can people on the call actually DO things in this room, and does it hold together?

The room call's promise is interaction: grab a chair, open the door, pull a drawer, with one
shared MuJoCo simulation behind everyone. A scene can compile, load and look right and still break
that promise in ways nobody notices until a user study — a door whose hinge is on the wrong side
of a jamb it cannot pass, a chair tucked so far under the table it cannot be lifted, a 128 kg door
leaf, a stack of boxes that creeps across the floor while nobody touches it. So this audit drives
the scene through the SAME code the server runs (`RoomSim`: its settle, its grab spring, its force
clamp, its blow-up and escape guards, stepped at the server's tick) and measures:

* **Interactability**, per movable body: can it be grabbed (`RoomSim.grab` says yes), does a
  standard pull move it, and does it come back to rest afterwards.
    - a free body is grabbed at its centre of mass and the hand moved 0.5 m straight up for 2 s;
      it passes at >= 0.3 m of lift. Blocked from above (a chair tucked under a table) it is
      dragged 0.5 m sideways instead (+x, -x, +y, -y world, each from the settled room) and passes
      at >= 0.3 m — a person would pull it out before lifting it;
    - a hinge is grabbed at its handle edge (90 % of the leaf's reach from the axis, at the centre
      of mass's height) and the hand swept along the leaf's own arc for 1.5 s, then held 0.5 s; it
      passes once it turned >= min(60 % of the travel open to it, 60 deg). A slide is pulled along
      its axis the same way and passes at >= 60 % of its travel. "Travel open to it" is the
      larger of the two ways it can go from where it settled — the full range for a closed door,
      and still a fair test of a door RoomPlan saw standing open;
    - afterwards it is let go and stepped (3 s free, 2 s jointed): stable = no unstable reset, no
      escape respawn, every joint speed of the body < 0.05 at the end.
* **Masses** against a table of plausible real-world ranges per category (`MASS_RANGES`, each
  with where it came from), flagging what is outside — the tea_room door leaf at 128.7 kg is the
  bug this exists to catch. Leaves, fronts and drawers scale with their size (a wardrobe door is
  not a cupboard door), so their ranges are per square metre of face or of box; the category and
  the area come from the body's extent in its OWN frame, as RoomPlan sized it — a world-axis box
  around a table in a room scanned at 30 deg is up to twice its real footprint.
* **Overlaps** deeper than 5 mm involving anything movable, as authored and after the settle
  (what the room call's settle could not separate), and how far the settle moved each body from
  its scanned pose.
* **Stability**: the settled room stepped for 10 s with nobody touching it — the largest drift of
  any body, and the server's own reset and respawn counters.
* **Cost**: milliseconds per physics step at the server's tick (with a state snapshot at its frame
  rate) and the real-time factor — warmed up, then repeated, reported as the median with its
  min/max, because one short sample on a laptop or a shared machine moves by +-25 % between runs
  (more than the differences between the rooms being compared). And how many rooms one server
  process keeps in real time: MEASURED by running k rooms through the server's own loop (asyncio,
  one `to_thread` hop per tick, a snapshot and its JSON every 1/30 s) and finding the largest k
  that every room keeps at real time. The per-step Python around `mj_step` holds the GIL, so rooms
  do not scale with cores; a flat-out sweep over 1..N threads (N = the CPUs this process may use,
  not the host's) shows by how much.

Everything is deterministic but the timings: the same scene gives the same pass/fail and drift
numbers on any machine, so they can go in a paper; the cost numbers say which CPU they came from
and how much they spread.
"""

from __future__ import annotations

import asyncio
import math
import os
import platform
import re
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from litereality_agent.room_ops.roomcall.model_info import (
    body_name,
    movable_body_ids,
    rig_joints,
    servo_held_joints,
)
from litereality_agent.room_ops.roomcall.server import (
    MAX_ROOMS,
    PHYSICS_INTERVAL,
    STATE_INTERVAL,
    dumps,
    max_steps_per_tick,
)
from litereality_agent.room_ops.roomcall.sim import RoomSim, load_model

OWNER = "evaluate"
# The standard pull (see the module docstring).
PULL_DISTANCE = 0.5          # m the hand moves away from the grabbed point
PULL_SECONDS = 2.0           # how long the hand pulls
SWEEP_SECONDS = 1.5          # of which a jointed body's hand sweeps along its path, then holds
LIFT_PASS = 0.3              # m of lift (or drag) that counts as moved
JOINT_PASS_FRACTION = 0.6    # of the travel open to a hinge or slide
HINGE_PASS_CAP = math.radians(60.0)
SWEEP_OVERSHOOT = math.radians(15.0)   # the hand aims this far past the pass mark (hinges)
SLIDE_OVERSHOOT = 0.25                 # ... and this fraction past it (slides)
HANDLE_FRACTION = 0.9        # of a leaf's reach from its axis where the hand takes hold
UNLIMITED_SLIDE = 0.3        # m: the pass mark of a slide without a range
RELEASE_FREE_SECONDS = 3.0   # time to come to rest after being let go
RELEASE_JOINT_SECONDS = 2.0
REST_SPEED = 0.05            # m/s or rad/s: below this a body is at rest
DRAG_DIRECTIONS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}
# Overlaps and drift.
OVERLAP_TOL = 0.005          # m: the exporter's and the room call's "initial interpenetration"
STABILITY_SECONDS = 10.0
STABILITY_SAMPLE = 0.1       # s between drift samples
DRIFT_MOVING = 0.01          # m: a body that drifts further than this unattended is "moving"
# Cost.
COST_SECONDS = 3.0           # simulated seconds stepped per timed repeat of one room
COST_WARMUP_SECONDS = 0.5    # simulated seconds stepped first, untimed (caches, allocator, branch
                             # predictors: the first second of a fresh MjData is not the steady state)
COST_REPEATS = 5             # timed repeats; the median is reported with the min and max
CONCURRENCY_SECONDS = 1.0    # wall seconds each room steps flat out per sweep point
CONCURRENCY_REPEATS = 3
REALTIME_HEADROOM = 0.8      # a room needs this share of a core's budget to keep real time safely
CAPACITY_WARMUP = 0.5        # wall s the server-loop load test runs before it starts measuring
CAPACITY_WINDOW = 2.0        # wall s over which each room's real-time factor is measured
CAPACITY_OK = 0.98           # a room keeps real time when its factor over the window is >= this

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_JOINT_KIND = {_FREE: "free", _BALL: "ball", _SLIDE: "slide", _HINGE: "hinge"}
_PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)


# ── plausible masses ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MassRange:
    """What one real object of a category weighs and where that came from. Ranges are
    deliberately generous — they flag the estimate that is wrong by a factor, not the one that is
    a few kilograms off — and every range is at least as wide as the numbers its ``basis`` quotes
    (a test holds them to that). The sources are retail spec sheets and buyer's guides found on
    2026-10-21 or a density x dimensions derivation; verify the ones a paper leans on.

    ``per`` says what ``lo``/``hi`` are in: ``kg`` for the whole object; ``face`` for kg per m^2
    of a leaf's face (its two largest extents: a door, a cabinet front, a sash), which is how
    board and glass are sold and why one fixed range cannot fit a cupboard door and a wardrobe
    door at once; ``shell`` for kg per m^2 of an open-top box's five sides (a drawer)."""

    lo: float
    hi: float
    basis: str
    source: str = ""
    per: str = "kg"

    def bounds(self, size: tuple[float, float, float] | None) -> tuple[float, float] | None:
        """(lo, hi) in kg for a body of extent ``size`` (m, any order); None when the range is
        per area and no size is known."""
        if self.per == "kg":
            return self.lo, self.hi
        area = scaling_area(self.per, size)
        if area is None:
            return None
        return self.lo * area, self.hi * area


def scaling_area(per: str, size: tuple[float, float, float] | None) -> float | None:
    """The m^2 a per-area range scales with (see `MassRange`); None without a usable size."""
    if size is None:
        return None
    a, b, c = sorted((abs(float(v)) for v in size), reverse=True)
    if not a > 0:
        return None
    if per == "face":
        return a * b
    if per == "shell":
        # Five sides of an open box: its whole surface less the largest face (the open top).
        return 2.0 * (a * b + a * c + b * c) - a * b
    raise ValueError(f"unknown mass range unit {per!r}")


MASS_RANGES: dict[str, MassRange] = {
    "chair": MassRange(2.0, 15.0, "dining chairs 3.5-5.5 kg in retail listings; task/office "
                       "chairs 7-14 kg", "https://de.sihoooffice.com/blogs/bloggen/"
                       "how-heavy-is-an-office-chair"),
    "stool": MassRange(1.0, 10.0, "a seat on legs, lighter than a chair (derived)"),
    "sofa": MassRange(25.0, 120.0, "3-seater 45-113 kg, 2-seaters lighter (furniture guides)",
                      "https://houlte.com/blogs/news/how-much-does-a-sofa-weigh-comprehensive-"
                      "guide-for-2026"),
    "bed": MassRange(25.0, 150.0, "frame + mattress, single to king (derived from retail "
                     "listings)"),
    "table": MassRange(10.0, 60.0, "dining tables, retail listings (derived)"),
    "small_table": MassRange(2.0, 25.0, "side/coffee tables under 0.36 m^2 of top (derived)"),
    "desk": MassRange(15.0, 70.0, "office desks, retail listings (derived)"),
    "storage": MassRange(8.0, 120.0, "base cabinets to wardrobes, empty (derived from retail "
                         "listings)"),
    "dishwasher": MassRange(30.0, 60.0, "60 cm built-in: Bosch SGI5315EU 48.0 kg net",
                            "https://www.bosch-home.co.uk/customer-service/SGI5315EU/07"),
    "refrigerator": MassRange(8.0, 200.0, "mini fridge 10-43 kg, full size 45-180 kg",
                              "https://prudentreviews.com/refrigerator-weight/"),
    "oven": MassRange(20.0, 90.0, "built-in ovens to free-standing ranges (derived)"),
    "washer": MassRange(50.0, 100.0, "front-loading washing machines (derived)"),
    "television": MassRange(3.0, 40.0, "flat panels 32-75 in (derived)"),
    "toilet": MassRange(15.0, 50.0, "vitreous china WC (derived)"),
    "bathtub": MassRange(25.0, 200.0, "acrylic to cast iron (derived)"),
    "sink": MassRange(5.0, 40.0, "free-standing sink/basin (derived)"),
    "door_leaf": MassRange(5.0, 25.0, "a 900x2100 mm leaf (1.89 m^2): hollow core 12-18 kg, "
                           "solid core 25-40 kg, i.e. 6.3-21 kg/m^2; hollow 30-50 lb "
                           "(13.6-22.7 kg) per a retailer guide",
                           "https://lowes.com/n/buying-guide/hollow-core-vs-solid-core-doors",
                           per="face"),
    "cabinet_door": MassRange(5.0, 25.0, "19 mm MDF at 700-800 kg/m^3 is 13.3-15.2 kg/m^2, "
                              "16 mm chipboard ~10 kg/m^2; glazed and solid-wood fronts "
                              "heavier", "https://en.wikipedia.org/wiki/Medium-density_fibreboard",
                              per="face"),
    "appliance_door": MassRange(5.0, 45.0, "fridge, dishwasher and oven doors: a steel skin "
                                "over insulation or 2-3 glass panes (derived)", per="face"),
    "drawer": MassRange(2.5, 20.0, "an empty MDF/chipboard box: 12-19 mm board at 650-800 "
                        "kg/m^3 is 8-15 kg/m^2 of front, back and sides; a 3 mm hardboard "
                        "bottom ~2.5 kg/m^2 (derived)", per="shell"),
    "window_sash": MassRange(8.0, 45.0, "glass 2500 kg/m^3: a 4 mm pane is 10 kg/m^2, double "
                             "glazing 20 kg/m^2, triple 30 kg/m^2, plus frame (derived)",
                             per="face"),
    "mug": MassRange(0.2, 0.7, "ceramic mug 250-400 g, stoneware up to ~685 g",
                     "https://parcelpath.com/shipping-terms-glossary/glossary-weights-and-"
                     "measurements/how-much-does-a-mug-weigh/"),
    "ball": MassRange(0.05, 1.0, "size 5 football 410-450 g (IFAB Laws of the Game, Law 2); "
                      "basketball ~0.6 kg"),
    "box": MassRange(0.1, 20.0, "a box of unknown contents (derived)"),
    "laptop": MassRange(1.0, 3.5, "laptops (derived)"),
    "book": MassRange(0.1, 2.5, "books (derived)"),
    "lamp": MassRange(0.5, 12.0, "table to floor lamps (derived)"),
    "plant": MassRange(0.3, 25.0, "potted plants (derived)"),
    "bottle": MassRange(0.1, 2.0, "bottles, full (derived)"),
}

_APPLIANCE = {"refrigerator", "fridge", "freezer", "dishwasher", "oven", "stove", "washer",
              "dryer", "microwave"}
_FURNITURE = {"storage", "cabinet", "cupboard", "wardrobe", "closet", "shelf", "sink", "dresser"}
# token -> category in priority order: the first one the name has wins, so "Sink_Storage0" is
# storage (a sink unit is a counter) and "ChairCluster0" a chair.
_OBJECT_TOKENS = (
    ("chair", "chair"), ("armchair", "chair"), ("stool", "stool"), ("sofa", "sofa"),
    ("couch", "sofa"), ("bed", "bed"), ("desk", "desk"), ("table", "table"),
    ("storage", "storage"), ("cabinet", "storage"), ("cupboard", "storage"),
    ("wardrobe", "storage"), ("shelf", "storage"), ("bookshelf", "storage"),
    ("dresser", "storage"), ("dishwasher", "dishwasher"), ("refrigerator", "refrigerator"),
    ("fridge", "refrigerator"), ("oven", "oven"), ("stove", "oven"), ("washer", "washer"),
    ("dryer", "washer"), ("television", "television"), ("tv", "television"),
    ("monitor", "television"), ("toilet", "toilet"), ("bathtub", "bathtub"), ("bath", "bathtub"),
    ("sink", "sink"), ("mug", "mug"), ("cup", "mug"), ("ball", "ball"), ("box", "box"),
    ("cardboard", "box"), ("laptop", "laptop"), ("book", "book"), ("lamp", "lamp"),
    ("plant", "plant"), ("bottle", "bottle"),
)
SMALL_TABLE_AREA = 0.36      # m^2: as the quick room's free/static rule draws the line (<=, on
                             # RoomPlan's own width x depth)
DOOR_LEAF_HEIGHT = 1.5       # m: a hinged leaf at least this tall in no furniture is a room door
UPRIGHT_COS = 0.99           # a body whose own z is within ~8 deg of world z is measured in its frame


def name_tokens(name: str) -> list[str]:
    """``Sink_Storage0_Door_Right`` -> [sink, storage, door, right]; ``ChairCluster0`` -> [chair,
    cluster]. Exporters name parts after their object, so the words are the category."""
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", name or "")
    return [w.lower() for w in words]


def classify(name: str, joint: str, size: tuple[float, float, float]) -> str:
    """The `MASS_RANGES` category of a body from its name, joint and size (x, y, z: its extent in
    its own frame, z up — see `body_extent`), or "unknown". A leaf is told apart by what it hangs
    in: a ``door`` of a storage unit is a cabinet door whatever its size; one in no furniture and
    at least `DOOR_LEAF_HEIGHT` tall is a room's door leaf. A table is small by the quick room's
    own rule (width x depth <= `SMALL_TABLE_AREA`), so the audit judges a side table the quick
    room set free as the side table it is."""
    tokens = name_tokens(name)
    words = set(tokens)
    height = float(size[2])
    if "drawer" in words:
        return "drawer"
    if "sash" in words or ("window" in words and joint in ("hinge", "slide")):
        return "window_sash"
    if words & {"door", "leaf", "flap"}:
        if words & _APPLIANCE:
            return "appliance_door"
        if words & _FURNITURE or height < DOOR_LEAF_HEIGHT:
            return "cabinet_door"
        return "door_leaf"
    for key, category in _OBJECT_TOKENS:
        if key in words:
            if category == "table" and float(size[0]) * float(size[1]) <= SMALL_TABLE_AREA:
                return "small_table"
            return category
    return "unknown"


def mass_bounds(category: str, size: tuple[float, float, float] | None) -> tuple[float, float] | None:
    """The plausible (lo, hi) kg of a body of ``category`` and extent ``size``, or None."""
    rng = MASS_RANGES.get(category)
    return rng.bounds(size) if rng is not None else None


def mass_verdict(category: str, mass: float,
                 size: tuple[float, float, float] | None = None) -> str:
    """ok | light | heavy against `MASS_RANGES` (a per-area range scaled by ``size``), or unknown
    for a category without a range, or a per-area one without a size."""
    bounds = mass_bounds(category, size)
    if bounds is None:
        return "unknown"
    if mass < bounds[0]:
        return "light"
    if mass > bounds[1]:
        return "heavy"
    return "ok"


def _mass_fields(category: str, mass: float, size: tuple[float, float, float]) -> dict:
    """The report's mass columns for one body: its verdict, its range in kg and what that came
    from (for a per-area range, the area it was scaled by)."""
    rng = MASS_RANGES.get(category)
    bounds = mass_bounds(category, size)
    out: dict[str, Any] = {
        "mass_kg": round(mass, 3),
        "mass_range": [round(bounds[0], 2), round(bounds[1], 2)] if bounds else None,
        "mass_verdict": mass_verdict(category, mass, size),
    }
    if rng is not None and rng.per != "kg":
        area = scaling_area(rng.per, size)
        out["mass_range_per_m2"] = {"per": rng.per, "kg_per_m2": [rng.lo, rng.hi],
                                    "area_m2": round(area, 4) if area else None}
    return out


# ── model helpers ────────────────────────────────────────────────────────────────────────────────


def _own_joints(model: mujoco.MjModel, b: int, held: set[int]) -> list[int]:
    start, count = int(model.body_jntadr[b]), int(model.body_jntnum[b])
    return [j for j in range(start, start + count) if j not in held] if count > 0 else []


def _welded_geoms(model: mujoco.MjModel, b: int) -> list[int]:
    """The body's own geoms plus those of every descendant welded to it (no joint of its own):
    what moves as one piece when the body is grabbed."""
    children: dict[int, list[int]] = {}
    for c in range(1, model.nbody):
        children.setdefault(int(model.body_parentid[c]), []).append(c)
    out, stack = [], [b]
    while stack:
        body = stack.pop()
        start, count = int(model.body_geomadr[body]), int(model.body_geomnum[body])
        if start >= 0:
            out.extend(range(start, start + count))
        stack.extend(c for c in children.get(body, []) if int(model.body_jntnum[c]) == 0)
    return out


_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)


def _box_geoms(model: mujoco.MjModel, b: int) -> list[int]:
    """What collides on the body (its welded geoms; all of them if none collide), planes
    excluded: the geoms whose extent is the body's size."""
    geoms = [g for g in _welded_geoms(model, b) if int(model.geom_type[g]) != _PLANE]
    colliding = [g for g in geoms if int(model.geom_contype[g]) or int(model.geom_conaffinity[g])]
    return colliding or geoms


def _corners(model: mujoco.MjModel, data: mujoco.MjData, geoms: list[int]) -> np.ndarray:
    """World corners (8 per geom) of the geoms' own bounding boxes."""
    out = []
    for g in geoms:
        centre, half = np.asarray(model.geom_aabb[g, :3]), np.asarray(model.geom_aabb[g, 3:])
        local = centre + half * _SIGNS
        out.append(local @ np.asarray(data.geom_xmat[g]).reshape(3, 3).T + data.geom_xpos[g])
    return np.concatenate(out) if out else np.zeros((0, 3))


def body_box(model: mujoco.MjModel, data: mujoco.MjData, b: int) -> tuple[np.ndarray, np.ndarray]:
    """World AABB (lo, hi) of what collides on the body (see `_box_geoms`)."""
    corners = _corners(model, data, _box_geoms(model, b))
    if not len(corners):
        p = np.asarray(data.xpos[b], dtype=float)
        return p.copy(), p.copy()
    return corners.min(axis=0), corners.max(axis=0)


def body_extent(model: mujoco.MjModel, data: mujoco.MjData, b: int) -> tuple[float, float, float]:
    """The body's (x, y, z) extent in its OWN frame when that frame is upright (z within ~8 deg of
    world z): RoomPlan's width, depth and height for a quick-room body, however the room is turned.
    A world-axis box would not do — a 0.5 x 0.6 m side table yawed 30 deg has a 0.56 m^2 world
    box, not 0.30, and would be judged a dining table. A body that is not upright (a part
    authored in a Y-up frame) falls back to its world box."""
    geoms = _box_geoms(model, b)
    corners = _corners(model, data, geoms)
    if not len(corners):
        return 0.0, 0.0, 0.0
    rot = np.asarray(data.xmat[b], dtype=float).reshape(3, 3)
    if abs(float(rot[2, 2])) >= UPRIGHT_COS:
        local = (corners - np.asarray(data.xpos[b], dtype=float)) @ rot
        ext = local.max(axis=0) - local.min(axis=0)
    else:
        ext = corners.max(axis=0) - corners.min(axis=0)
    return tuple(round(float(v), 4) for v in ext)


@dataclass
class _Target:
    """One body the audit tests: what moves when it is grabbed and how."""

    body: int
    name: str
    kind: str                 # free | hinge | slide | ball | mocap
    joint: int                # its first own joint (-1 for mocap)


def testable_bodies(model: mujoco.MjModel) -> list[_Target]:
    """Movable bodies that move on their OWN: a joint that is not the building's (servo-held or
    shake rig), or mocap. A body welded under one of those moves with it and is tested through
    it, not twice."""
    held = servo_held_joints(model) | rig_joints(model)
    out = []
    for b in sorted(movable_body_ids(model)):
        joints = _own_joints(model, b, held)
        if joints:
            kind = _JOINT_KIND.get(int(model.jnt_type[joints[0]]), "other")
            out.append(_Target(b, body_name(model, b), kind, joints[0]))
        elif int(model.body_mocapid[b]) >= 0:
            out.append(_Target(b, body_name(model, b), "mocap", -1))
    return out


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: ``v`` turned by ``angle`` about the unit ``axis`` (right-handed, as MuJoCo
    turns a hinge's child for a positive angle)."""
    k = axis / max(float(np.linalg.norm(axis)), 1e-12)
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(k, v) * s + k * float(k @ v) * (1.0 - c)


def _body_speed(model: mujoco.MjModel, data: mujoco.MjData, b: int, held: set[int]) -> float:
    speed = 0.0
    for j in _own_joints(model, b, held):
        adr = int(model.jnt_dofadr[j])
        n = {_FREE: 6, _BALL: 3}.get(int(model.jnt_type[j]), 1)
        speed = max(speed, float(np.max(np.abs(data.qvel[adr:adr + n]))))
    return speed


def _overlaps(model: mujoco.MjModel, data: mujoco.MjData, movable: set[int],
              tol: float = OVERLAP_TOL) -> list[dict]:
    """Contacts deeper than ``tol`` that involve something movable, worst per body pair. Two
    static parts touching (a wall box into the floor slab) is how structures are built, not a
    problem anyone can feel."""
    worst: dict[tuple[str, str], float] = {}
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist >= -tol:
            continue
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if b1 not in movable and b2 not in movable:
            continue
        key = tuple(sorted((body_name(model, b1) if b1 else "world",
                            body_name(model, b2) if b2 else "world")))
        worst[key] = max(worst.get(key, 0.0), -float(c.dist))
    return [{"bodies": f"{a} <-> {b}", "mm": round(d * 1000.0, 1)}
            for (a, b), d in sorted(worst.items(), key=lambda kv: -kv[1])]


# ── the pull tests ───────────────────────────────────────────────────────────────────────────────


class _Clock:
    """Steps a room at the server's tick and keeps the books: simulated seconds, steps and the
    wall time they took, so the pulls double as a cost measurement under interaction."""

    def __init__(self, sim: RoomSim) -> None:
        self.sim = sim
        self.steps = 0
        self.wall = 0.0

    def run(self, seconds: float, target=None) -> None:
        """Step ``seconds`` of simulated time; ``target(t)`` (t from 0) moves the hand first."""
        ticks = max(1, int(round(seconds / PHYSICS_INTERVAL)))
        for k in range(ticks):
            if target is not None:
                self.sim.move(OWNER, target(k * PHYSICS_INTERVAL))
            t0 = time.perf_counter()
            self.steps += self.sim.step_realtime(PHYSICS_INTERVAL)
            self.wall += time.perf_counter() - t0


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _release(sim: RoomSim, clock: _Clock, target: _Target, seconds: float,
             held: set[int], counters: tuple[int, int]) -> dict:
    sim.release(OWNER)
    clock.run(seconds)
    speed = _body_speed(sim.model, sim.data, target.body, held)
    resets = sim.unstable_resets - counters[0]
    respawns = sim.respawns - counters[1]
    finite = bool(np.all(np.isfinite(sim.data.qpos)))
    return {"stable": bool(finite and not resets and not respawns and speed < REST_SPEED),
            "end_speed": round(speed, 4), "resets": resets, "respawns": respawns}


def _pull_free(sim: RoomSim, clock: _Clock, target: _Target, held: set[int]) -> dict:
    """Lift 0.5 m; if that does not reach `LIFT_PASS`, drag sideways (each direction from the
    settled room) and keep the best."""
    d, b = sim.data, target.body
    attempts: dict[str, float] = {}
    best: dict[str, Any] | None = None
    for mode in ("lift", *DRAG_DIRECTIONS):
        sim.reset()
        counters = (sim.unstable_resets, sim.respawns)
        com0 = np.asarray(d.subtree_com[b], dtype=float).copy()
        if not sim.grab(OWNER, b, com0):
            return {"grab": False, "kind": "free", "pass": False, "reason": "not grabbable"}
        if mode == "lift":
            goal = com0 + np.array([0.0, 0.0, PULL_DISTANCE])
        else:
            dx, dy = DRAG_DIRECTIONS[mode]
            goal = com0 + PULL_DISTANCE * np.array([dx, dy, 0.0])
        reached_at = None

        def hand(_t: float, goal=goal) -> np.ndarray:
            return goal

        ticks = int(round(PULL_SECONDS / PHYSICS_INTERVAL))
        for k in range(ticks):
            clock.run(PHYSICS_INTERVAL, hand)
            moved = _moved(d.subtree_com[b], com0, mode)
            if reached_at is None and moved >= LIFT_PASS:
                reached_at = (k + 1) * PHYSICS_INTERVAL
        moved = _moved(d.subtree_com[b], com0, mode)
        attempts[mode] = round(moved, 4)
        passed = moved >= LIFT_PASS
        result = {"grab": True, "kind": "free", "mode": mode, "moved_m": round(moved, 4),
                  "need_m": LIFT_PASS, "pass": passed,
                  "t_reach_s": round(reached_at, 3) if reached_at is not None else None}
        result.update(_release(sim, clock, target, RELEASE_FREE_SECONDS, held, counters))
        if best is None or (passed and not best["pass"]) or (
                passed == best["pass"] and moved > best["moved_m"]):
            best = result
        if passed:
            break
    assert best is not None
    best["attempts"] = attempts
    sim.reset()
    return best


def _moved(com: np.ndarray, com0: np.ndarray, mode: str) -> float:
    delta = np.asarray(com, dtype=float) - com0
    if mode == "lift":
        return float(delta[2])
    return float(np.linalg.norm(delta[:2]))


def _travel(model: mujoco.MjModel, j: int, q0: float,
            unlimited: float) -> tuple[float, float, float]:
    """(sign, travel open to it, its full range) of a hinge or slide from ``q0``: the way with
    more room to go."""
    if int(model.jnt_limited[j]):
        lo, hi = (float(v) for v in model.jnt_range[j])
        up, down = hi - q0, q0 - lo
        sign = 1.0 if up >= down else -1.0
        return sign, max(up, down, 0.0), hi - lo
    return 1.0, unlimited, math.inf


def _handle_point(sim: RoomSim, b: int, axis: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Where a hand takes a hinged leaf: on the line from the axis through the centre of mass,
    `HANDLE_FRACTION` of the way out to the leaf's far edge, at the centre of mass's height."""
    m, d = sim.model, sim.data
    k = axis / max(float(np.linalg.norm(axis)), 1e-12)
    com = np.asarray(d.subtree_com[b], dtype=float)
    rel = com - pivot
    along = float(rel @ k)
    radial = rel - along * k
    r = float(np.linalg.norm(radial))
    corners = _corners(m, d, _box_geoms(m, b))
    if not len(corners):
        return com
    if r < 1e-6:
        # Centre of mass on the axis (a turntable): take the farthest corner's direction.
        rad = (corners - pivot) - np.outer((corners - pivot) @ k, k)
        far = rad[np.argmax(np.linalg.norm(rad, axis=1))]
        r = float(np.linalg.norm(far))
        if r < 1e-6:
            return com
        radial = far
    unit = radial / r
    reach = float(np.max((corners - pivot) @ unit))
    return pivot + along * k + unit * max(r, HANDLE_FRACTION * reach)


def _pull_joint(sim: RoomSim, clock: _Clock, target: _Target, held: set[int]) -> dict:
    """Sweep the hand along a hinge's arc or a slide's axis to past the pass mark, then hold."""
    m, d, b, j = sim.model, sim.data, target.body, target.joint
    sim.reset()
    counters = (sim.unstable_resets, sim.respawns)
    adr = int(m.jnt_qposadr[j])
    q0 = float(d.qpos[adr])
    hinge = target.kind == "hinge"
    unlimited = math.inf if hinge else UNLIMITED_SLIDE / JOINT_PASS_FRACTION
    sign, room, full = _travel(m, j, q0, unlimited)
    need = JOINT_PASS_FRACTION * room
    if hinge:
        need = min(need, HINGE_PASS_CAP)
        goal = min(room, need + SWEEP_OVERSHOOT)
    else:
        goal = min(room, need * (1.0 + SLIDE_OVERSHOOT))
    axis = np.asarray(d.xaxis[j], dtype=float).copy()
    pivot = np.asarray(d.xanchor[j], dtype=float).copy()
    anchor = (_handle_point(sim, b, axis, pivot) if hinge
              else np.asarray(d.subtree_com[b], dtype=float).copy())
    unit = "deg" if hinge else "m"
    scale = math.degrees(1.0) if hinge else 1.0
    # An unlimited hinge (a swivel seat, a turntable) has no travel to report: None, as for its
    # range — `inf` would make report.json invalid JSON for any strict parser.
    base = {"kind": target.kind, "unit": unit, "need": round(need * scale, 3),
            "travel": round(room * scale, 3) if math.isfinite(room) else None,
            "range": round(full * scale, 3) if math.isfinite(full) else None}
    if need <= 1e-6:
        grabbed = bool(sim.grab(OWNER, b, anchor))
        sim.reset()
        return {**base, "grab": grabbed, "pass": False,
                "reason": "no travel from where it settles", "stable": True}
    if not sim.grab(OWNER, b, anchor):
        return {**base, "grab": False, "pass": False, "reason": "not grabbable"}

    def hand(t: float) -> np.ndarray:
        s = sign * goal * _smooth(t / SWEEP_SECONDS)
        if hinge:
            return pivot + _rotate(anchor - pivot, axis, s)
        return anchor + axis / max(float(np.linalg.norm(axis)), 1e-12) * s

    reached_at = None
    ticks = int(round(PULL_SECONDS / PHYSICS_INTERVAL))
    for k in range(ticks):
        t = k * PHYSICS_INTERVAL
        clock.run(PHYSICS_INTERVAL, lambda _t, t=t: hand(t))
        moved = abs(float(d.qpos[adr]) - q0)
        if reached_at is None and moved >= need - 1e-6:
            reached_at = (k + 1) * PHYSICS_INTERVAL
    q_held = float(d.qpos[adr])
    moved = abs(q_held - q0)
    result = {**base, "grab": True, "moved": round(moved * scale, 3),
              "pass": bool(moved >= need - 1e-6),
              "t_reach_s": round(reached_at, 3) if reached_at is not None else None}
    result.update(_release(sim, clock, target, RELEASE_JOINT_SECONDS, held, counters))
    # A door or drawer should stay where it was left (damping + friction loss), not drift back.
    result["drift_after_release"] = round(abs(float(d.qpos[adr]) - q_held) * scale, 3)
    sim.reset()
    return result


def pull_test(sim: RoomSim, target: _Target, clock: _Clock | None = None) -> dict:
    """The standard pull for one body (see the module docstring)."""
    clock = clock or _Clock(sim)
    held = servo_held_joints(sim.model) | rig_joints(sim.model)
    if target.kind == "free":
        return _pull_free(sim, clock, target, held)
    if target.kind in ("hinge", "slide"):
        return _pull_joint(sim, clock, target, held)
    sim.reset()
    grabbed = sim.grab(OWNER, target.body, sim.data.subtree_com[target.body].copy())
    sim.reset()
    return {"kind": target.kind, "grab": bool(grabbed), "pass": None,
            "reason": f"no standard pull for a {target.kind} body"}


# ── stability and cost ───────────────────────────────────────────────────────────────────────────


def stability(sim: RoomSim, targets: list[_Target], seconds: float = STABILITY_SECONDS) -> dict:
    """The settled room left alone for ``seconds``: how far anything creeps, and whether the
    server's guards had to step in. A room that cannot sit still cannot be shared."""
    m, d = sim.model, sim.data
    sim.reset()
    counters = (sim.unstable_resets, sim.respawns)
    ids = [t.body for t in targets]
    pos0 = d.xpos[ids].copy() if ids else np.zeros((0, 3))
    quat0 = d.xquat[ids].copy() if ids else np.zeros((0, 4))
    q0 = {t.name: float(d.qpos[m.jnt_qposadr[t.joint]]) for t in targets
          if t.kind in ("hinge", "slide")}
    drift = np.zeros(len(ids))
    turn = np.zeros(len(ids))
    joint_drift = dict.fromkeys(q0, 0.0)
    clock = _Clock(sim)
    samples = max(1, int(round(seconds / STABILITY_SAMPLE)))
    for _ in range(samples):
        clock.run(STABILITY_SAMPLE)
        if ids:
            drift = np.maximum(drift, np.linalg.norm(d.xpos[ids] - pos0, axis=1))
            dots = np.clip(np.abs(np.sum(d.xquat[ids] * quat0, axis=1)), 0.0, 1.0)
            turn = np.maximum(turn, np.degrees(2.0 * np.arccos(dots)))
        for t in targets:
            if t.name in q0:
                joint_drift[t.name] = max(joint_drift[t.name],
                                          abs(float(d.qpos[m.jnt_qposadr[t.joint]]) - q0[t.name]))
    per_body = {t.name: {"drift_mm": round(float(drift[i]) * 1000.0, 2),
                         "turn_deg": round(float(turn[i]), 3)} for i, t in enumerate(targets)}
    for t in targets:
        if t.name in joint_drift:
            scale = math.degrees(1.0) if t.kind == "hinge" else 1000.0
            per_body[t.name]["joint_drift"] = round(joint_drift[t.name] * scale, 3)
            per_body[t.name]["joint_unit"] = "deg" if t.kind == "hinge" else "mm"
    worst = int(np.argmax(drift)) if len(ids) else -1
    out = {
        "seconds": seconds,
        "max_drift_mm": round(float(drift.max()) * 1000.0, 2) if len(ids) else 0.0,
        "max_drift_body": targets[worst].name if worst >= 0 else None,
        "max_turn_deg": round(float(turn.max()), 3) if len(ids) else 0.0,
        "moving": sorted(t.name for i, t in enumerate(targets) if drift[i] > DRIFT_MOVING),
        "resets": sim.unstable_resets - counters[0],
        "respawns": sim.respawns - counters[1],
        "per_body": per_body,
    }
    sim.reset()
    return out


def settle_shift(model: mujoco.MjModel, sim: RoomSim, targets: list[_Target]) -> dict:
    """How far the room call's settle moved each body from its authored (scanned) pose: what a
    person sees the room do in its first second, and how far the simulated room sits from the
    scan."""
    authored = mujoco.MjData(model)
    mujoco.mj_forward(model, authored)
    sim.reset()
    rows = {}
    for t in targets:
        shift = float(np.linalg.norm(sim.data.xpos[t.body] - authored.xpos[t.body]))
        rows[t.name] = round(shift * 1000.0, 2)
    worst = max(rows.items(), key=lambda kv: kv[1]) if rows else (None, 0.0)
    return {"max_mm": worst[1], "max_body": worst[0], "per_body_mm": rows}


def _spread(values: list[float], digits: int = 4) -> dict:
    """Median, min and max of repeated timings: what a paper can quote as "x (min-max)"."""
    return {"median": round(float(statistics.median(values)), digits),
            "min": round(float(min(values)), digits), "max": round(float(max(values)), digits),
            "n": len(values)}


def _time_steps(sim: RoomSim, seconds: float) -> tuple[int, float, float, int]:
    """(steps, step wall s, snapshot wall s, snapshots) for ``seconds`` of server ticks."""
    ticks = max(1, int(round(seconds / PHYSICS_INTERVAL)))
    per_frame = max(1, int(round(STATE_INTERVAL / PHYSICS_INTERVAL)))
    steps, step_wall, snap_wall, snaps = 0, 0.0, 0.0, 0
    for k in range(ticks):
        t0 = time.perf_counter()
        steps += sim.step_realtime(PHYSICS_INTERVAL)
        t1 = time.perf_counter()
        step_wall += t1 - t0
        if k % per_frame == 0:
            sim.snapshot()
            snap_wall += time.perf_counter() - t1
            snaps += 1
    return steps, step_wall, snap_wall, snaps


def step_cost(sim: RoomSim, seconds: float = COST_SECONDS, *, repeats: int = COST_REPEATS,
              warmup: float = COST_WARMUP_SECONDS) -> dict:
    """Wall time of one room stepped like the server steps it: `PHYSICS_INTERVAL` ticks through
    `RoomSim.step_realtime`, a snapshot every `STATE_INTERVAL`, nobody holding anything.

    ``warmup`` simulated seconds are stepped first and not timed, then ``repeats`` runs of
    ``seconds`` each, every one from the settled room. ``ms_per_step`` and ``realtime_factor``
    are the medians; ``ms_per_step_spread`` / ``realtime_factor_spread`` hold median, min and
    max, so a reader can see how far one machine's numbers move between runs."""
    repeats = max(1, int(repeats))
    sim.reset()
    if warmup > 0:
        _time_steps(sim, warmup)
    per_step, rtf, per_snap = [], [], []
    steps = 0
    for _ in range(repeats):
        sim.reset()
        steps, step_wall, snap_wall, snaps = _time_steps(sim, seconds)
        wall = step_wall + snap_wall
        per_step.append(1000.0 * step_wall / max(steps, 1))
        per_snap.append(1000.0 * snap_wall / max(snaps, 1))
        if wall > 0:
            rtf.append(steps * sim.timestep / wall)
    sim.reset()
    step_spread = _spread(per_step)
    rtf_spread = _spread(rtf, 2) if rtf else None
    return {"simulated_s": round(steps * sim.timestep, 3), "steps": steps, "repeats": repeats,
            "warmup_s": warmup,
            "ms_per_step": step_spread["median"], "ms_per_step_spread": step_spread,
            "ms_per_snapshot": _spread(per_snap)["median"],
            "realtime_factor": rtf_spread["median"] if rtf_spread else None,
            "realtime_factor_spread": rtf_spread}


def usable_cpus() -> int:
    """CPUs this process may actually run on: its affinity mask, capped by a cgroup CPU quota
    (a container's ``cpu.max``) — `os.cpu_count` reports the host's, which a quota-limited
    container cannot use."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    quota = _cgroup_cpus()
    if quota is not None:
        n = min(n, max(1, math.ceil(quota - 1e-9)))
    return max(1, n)


def _cgroup_cpus() -> float | None:
    """The cgroup CPU quota in CPUs (v2 ``cpu.max``, v1 ``cfs_quota_us``), or None if unlimited."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        return None if quota == "max" else float(quota) / float(period)
    except (OSError, ValueError):
        pass
    try:
        quota = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = float(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        return None if quota <= 0 or period <= 0 else quota / period
    except (OSError, ValueError):
        return None


def _flat_out(model: mujoco.MjModel, settled: np.ndarray, threads: int, seconds: float) -> float:
    """Summed simulated seconds per wall second of ``threads`` rooms stepping flat out."""
    sims = [RoomSim(model, settled=settled) for _ in range(threads)]
    simulated = [0.0] * threads
    start = threading.Barrier(threads + 1)
    stop = threading.Event()

    def work(i: int) -> None:
        sim = sims[i]
        start.wait()
        steps = 0
        while not stop.is_set():
            steps += sim.step_realtime(PHYSICS_INTERVAL)
        simulated[i] = steps * sim.timestep

    workers = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(threads)]
    for w in workers:
        w.start()
    start.wait()
    t0 = time.perf_counter()
    time.sleep(seconds)
    stop.set()
    for w in workers:
        w.join()
    wall = time.perf_counter() - t0
    return sum(simulated) / wall if wall > 0 else 0.0


def _thread_counts(limit: int) -> list[int]:
    """1, 2, 4, ... up to ``limit``, and ``limit`` itself."""
    counts, n = [], 1
    while n < limit:
        counts.append(n)
        n *= 2
    return [*counts, limit]


def concurrency(model: mujoco.MjModel, settled: np.ndarray, *, threads: int | None = None,
                seconds: float = CONCURRENCY_SECONDS, repeats: int = CONCURRENCY_REPEATS) -> dict:
    """The CPU's flat-out capacity for this scene: one room per thread, every room stepping as
    fast as it can for ``seconds``, swept over 1, 2, 4 ... `usable_cpus` threads (or just
    ``threads``), each point repeated ``repeats`` times. The summed real-time factor at the best
    point is an upper bound on what any server could do with this CPU; how little it grows with
    threads is the GIL's share. `REALTIME_HEADROOM` of it is ``rooms_flat_out``. The rooms one
    real server keeps in real time are `server_capacity`'s measurement, not this."""
    counts = [max(1, int(threads))] if threads else _thread_counts(usable_cpus())
    sweep: dict[str, dict] = {}
    best_n, best = counts[0], -1.0
    for n in counts:
        runs = [_flat_out(model, settled, n, seconds) for _ in range(max(1, int(repeats)))]
        sweep[str(n)] = _spread(runs, 2)
        if sweep[str(n)]["median"] > best:
            best_n, best = n, sweep[str(n)]["median"]
    return {"threads": best_n, "sweep": sweep, "seconds": seconds,
            "aggregate_realtime_factor": round(best, 2),
            "rooms_flat_out": int(best * REALTIME_HEADROOM),
            "usable_cpus": usable_cpus(), "server_max_rooms": MAX_ROOMS}


def _server_tick(sim: RoomSim, dt: float, want_state: bool, max_steps: int) -> tuple[int, Any]:
    """The worker-thread half of the server's tick (`server._tick` with no commands queued)."""
    steps = sim.step_realtime(dt, max_steps=max_steps)
    return steps, (sim.snapshot() if want_state else None)


async def _serve(model: mujoco.MjModel, settled: np.ndarray, rooms: int, warmup: float,
                 window: float) -> list[float]:
    """``rooms`` rooms driven the way `RoomCallServer._physics` drives each one — its own task on
    one event loop, ``sleep(PHYSICS_INTERVAL)``, the tick on a worker thread through
    `asyncio.to_thread`, a snapshot and its ``dumps`` every `STATE_INTERVAL` — for ``warmup`` +
    ``window`` wall seconds. Returns each room's real-time factor over the window, computed as
    the server computes it (simulated seconds over the wall time between ticks)."""
    sims = [RoomSim(model, settled=settled) for _ in range(rooms)]
    max_steps = max_steps_per_tick(sims[0].timestep)
    loop = asyncio.get_running_loop()
    start = loop.time()
    measure_from, stop_at = start + warmup, start + warmup + window
    factors = [0.0] * rooms

    async def room(i: int) -> None:
        sim = sims[i]
        last = loop.time()
        next_state = last
        simulated = elapsed = 0.0
        while True:
            await asyncio.sleep(PHYSICS_INTERVAL)
            now = loop.time()
            if now >= stop_at:
                break
            want_state = now >= next_state
            steps, snapshot = await asyncio.to_thread(_server_tick, sim, now - last, want_state,
                                                      max_steps)
            if last >= measure_from:
                simulated += steps * sim.timestep
                elapsed += now - last
            last = now
            if snapshot is not None:
                next_state = max(next_state + STATE_INTERVAL, now)
                dumps({"t": "state", "v": 1, "time": snapshot.get("time", 0.0),
                       "bodies": snapshot.get("bodies", []), "grabs": snapshot.get("grabs", []),
                       "avatars": {}})
        factors[i] = simulated / elapsed if elapsed > 0 else 0.0

    await asyncio.gather(*(room(i) for i in range(rooms)))
    return factors


def server_capacity(model: mujoco.MjModel, settled: np.ndarray, *, limit: int = MAX_ROOMS,
                    warmup: float = CAPACITY_WARMUP, window: float = CAPACITY_WINDOW,
                    ok: float = CAPACITY_OK, attempts: int = 2) -> dict:
    """The most rooms of this scene ONE room-call server process keeps in real time on this
    machine, measured with the server's own loop (`_serve`): k rooms are run together and pass
    when every one of them holds a real-time factor >= ``ok`` over ``window``. k doubles from 1
    until a run fails or reaches ``limit`` (the server's `MAX_ROOMS`: it refuses more), then the
    boundary is bisected. A k that fails is run again (up to ``attempts`` in all) before it
    counts as failed: a burst of load from another process stalls a 2 s window, a k beyond the
    CPU fails every time. Each k tried is kept with its slowest room's factor (the best attempt),
    so the curve is in the report, not just its end; at ``limit`` the answer is "at least
    ``limit``"."""
    probes: dict[int, float] = {}
    load_before = _load()

    def holds(k: int) -> bool:
        best = 0.0
        for _ in range(max(1, int(attempts))):
            best = max(best, min(asyncio.run(_serve(model, settled, k, warmup, window))))
            if best >= ok:
                break
        probes[k] = round(best, 3)
        return best >= ok

    rooms = 0
    if holds(1):
        rooms, fail, k = 1, None, 2
        while fail is None and rooms < limit:
            k = min(k, limit)
            if holds(k):
                rooms, k = k, k * 2
            else:
                fail = k
        while fail is not None and fail - rooms > 1:
            mid = (rooms + fail) // 2
            if holds(mid):
                rooms = mid
            else:
                fail = mid
    # A stall of more than CATCH_UP_TICKS intervals is lost time, and another process holding
    # the CPUs causes exactly those: on a machine that was already busy the answer is about the
    # other processes, and the report has to say so.
    cpus = usable_cpus()
    return {"rooms": rooms, "at_limit": rooms >= limit, "limit": limit, "ok": ok,
            "window_s": window, "warmup_s": warmup,
            "slowest_room_rtf": {str(k): v for k, v in sorted(probes.items())},
            "load_before": load_before, "usable_cpus": cpus,
            "machine_busy": bool(load_before is not None and load_before > cpus)}


def _load() -> float | None:
    """The 1-minute load average, or None where the OS has none."""
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return None


def machine() -> dict:
    """Which CPU the cost numbers came from, and how busy the machine was while they were taken
    (a shared machine's timings are not an idle one's)."""
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        load = [round(v, 2) for v in os.getloadavg()]
    except (AttributeError, OSError):
        load = None
    return {"cpu": cpu, "cores": os.cpu_count(), "usable_cpus": usable_cpus(),
            "load_average": load, "python": platform.python_version(),
            "mujoco": mujoco.__version__, "system": platform.system()}


# ── the audit ────────────────────────────────────────────────────────────────────────────────────


def audit_physics(scene: Path | str | mujoco.MjModel, *, settle_seconds: float = 1.0,
                  stability_seconds: float = STABILITY_SECONDS,
                  cost_seconds: float = COST_SECONDS, cost_repeats: int = COST_REPEATS,
                  measure_concurrency: bool = True,
                  concurrency_seconds: float = CONCURRENCY_SECONDS,
                  threads: int | None = None, capacity_window: float = CAPACITY_WINDOW) -> dict:
    """Everything above for one scene, as a JSON-ready dict (see `markdown`).
    ``measure_concurrency`` runs the flat-out thread sweep and the server-loop load test (tens
    of seconds); without it the cost is the single-room numbers only."""
    started = time.perf_counter()
    model = scene if isinstance(scene, mujoco.MjModel) else load_model(Path(scene))
    t0 = time.perf_counter()
    sim = RoomSim(model, settle_seconds=settle_seconds)
    settle_wall = time.perf_counter() - t0
    targets = testable_bodies(model)
    movable = movable_body_ids(model)

    authored = mujoco.MjData(model)
    mujoco.mj_forward(model, authored)
    overlaps = {"tol_mm": OVERLAP_TOL * 1000.0,
                "authored": _overlaps(model, authored, movable),
                "settled": _overlaps(model, sim.data, movable)}

    rows: list[dict] = []
    clock = _Clock(sim)
    for t in targets:
        # Sized as authored (the scanned pose): the settle may tip a body, its size is the same.
        size = body_extent(model, authored, t.body)
        mass = float(model.body_subtreemass[t.body])
        category = classify(t.name, t.kind, size)
        rows.append({"name": t.name, "id": t.body, "joint": t.kind, "category": category,
                     **_mass_fields(category, mass, size), "size_m": list(size),
                     "movable": True, "pull": pull_test(sim, t, clock)})
    statics = _static_objects(model, authored, movable)

    interact = _interactability(rows)
    stable = stability(sim, targets, stability_seconds)
    shift = settle_shift(model, sim, targets)
    cost = step_cost(sim, cost_seconds, repeats=cost_repeats)
    if clock.steps:
        cost["ms_per_step_grabbed"] = round(1000.0 * clock.wall / clock.steps, 4)
    cost["settle_wall_s"] = round(settle_wall, 3)
    if measure_concurrency:
        settled = sim.settled_state()
        cost["concurrency"] = concurrency(model, settled, threads=threads,
                                          seconds=concurrency_seconds)
        cost["capacity"] = server_capacity(model, settled, window=capacity_window)
        cost["rooms_realtime"] = cost["capacity"]["rooms"]
    masses = _mass_summary(rows, statics)
    return {
        "scene": str(scene) if not isinstance(scene, mujoco.MjModel) else None,
        "model": {"bodies": int(model.nbody), "geoms": int(model.ngeom), "dofs": int(model.nv),
                  "timestep": float(model.opt.timestep),
                  "integrator": mujoco.mjtIntegrator(int(model.opt.integrator)).name},
        "interactability": interact,
        "bodies": rows,
        "static_objects": statics,
        "masses": masses,
        "overlaps": overlaps,
        "settle_shift": shift,
        "stability": stable,
        "cost": cost,
        "machine": machine(),
        "mass_ranges": {k: asdict(v) for k, v in MASS_RANGES.items()},
        "definitions": DEFINITIONS,
        "audit_seconds": round(time.perf_counter() - started, 2),
    }


def _static_objects(model: mujoco.MjModel, data: mujoco.MjData, movable: set[int]) -> list[dict]:
    """Fixed bodies that are recognisably objects (a known category): their mass does nothing in
    the simulation, but it is the estimate a later stage that frees them would inherit."""
    out = []
    for b in range(1, model.nbody):
        if b in movable:
            continue
        name = body_name(model, b)
        size = body_extent(model, data, b)
        category = classify(name, "static", size)
        mass = float(model.body_subtreemass[b])
        if category == "unknown" or mass <= 0:
            continue
        out.append({"name": name, "category": category, **_mass_fields(category, mass, size),
                    "size_m": list(size)})
    return out


def _interactability(rows: list[dict]) -> dict:
    tested = [r for r in rows if r["pull"].get("pass") is not None]
    grab = [r for r in rows if r["pull"].get("grab")]
    moved = [r for r in tested if r["pull"].get("pass")]
    stable = [r for r in tested if r["pull"].get("stable")]
    ok = [r for r in tested if r["pull"].get("grab") and r["pull"].get("pass")
          and r["pull"].get("stable")]
    n = len(tested)
    by_kind: dict[str, dict[str, int]] = {}
    for r in tested:
        k = by_kind.setdefault(r["joint"], {"tested": 0, "interactable": 0})
        k["tested"] += 1
        k["interactable"] += int(r in ok)
    return {"movable": len(rows), "tested": n, "grabbable": len(grab), "moved": len(moved),
            "stable": len(stable), "interactable": len(ok),
            "interactable_rate": round(len(ok) / n, 3) if n else None,
            "failed": sorted(r["name"] for r in tested if r not in ok),
            "by_joint": by_kind}


def _mass_summary(rows: list[dict], statics: list[dict]) -> dict:
    def outliers(items: list[dict]) -> list[dict]:
        return [{"name": r["name"], "category": r["category"], "mass_kg": r["mass_kg"],
                 "range": r["mass_range"], "verdict": r["mass_verdict"]}
                for r in items if r["mass_verdict"] in ("light", "heavy")]

    known = [r for r in rows if r["mass_verdict"] != "unknown"]
    known_static = [r for r in statics if r["mass_verdict"] != "unknown"]
    return {"checked": len(known), "outliers": outliers(rows),
            "unknown_category": sorted(r["name"] for r in rows if r["mass_verdict"] == "unknown"),
            "static_checked": len(known_static), "static_outliers": outliers(statics)}


DEFINITIONS = {
    "interactable": "grabbable (RoomSim.grab accepts it) AND the standard pull moves it past the "
                    "pass mark AND it is at rest (all its joint speeds < 0.05) with no unstable "
                    "reset or escape respawn after release",
    "pull_free": "grab at the centre of mass, hand 0.5 m straight up for 2 s, pass at >= 0.3 m "
                 "of lift; if blocked, hand 0.5 m sideways (+x,-x,+y,-y world) for 2 s, pass at "
                 ">= 0.3 m",
    "pull_hinge": "grab at 90 % of the leaf's reach from the axis, hand swept along the leaf's "
                  "arc for 1.5 s then held 0.5 s, pass at >= min(60 % of the travel open to it, "
                  "60 deg)",
    "pull_slide": "grab at the centre of mass, hand swept along the axis for 1.5 s then held "
                  "0.5 s, pass at >= 60 % of the travel open to it",
    "mass_verdict": "ok when inside the category's MASS_RANGES range, light/heavy outside it; "
                    "door leaves, cabinet/appliance doors and window sashes are ranged in kg per "
                    "m^2 of face (two largest extents), drawers per m^2 of their five sides",
    "size_m": "the body's extent in its own frame (RoomPlan's width, depth, height for a quick "
              "room body) at the authored pose; its world box if the body is not upright",
    "overlaps": "contacts deeper than 5 mm involving a movable body, at the authored pose and "
                "after the room call's 1 s settle",
    "settle_shift": "distance each movable body's frame moved during the settle (authored -> "
                    "settled)",
    "stability": "max displacement (and rotation) of every movable body over 10 s of the "
                 "settled room with no input, sampled every 0.1 s; resets/respawns are "
                 "RoomSim's own counters",
    "ms_per_step": "wall ms per mj_step through RoomSim.step_realtime at the server's 120 Hz "
                   "tick, a snapshot every 1/30 s, nobody holding anything; median of 5 runs of "
                   "3 simulated s after a 0.5 s untimed warm-up (spread: min and max)",
    "realtime_factor": "simulated seconds per wall second for one room on one thread (median "
                       "of the same 5 runs)",
    "rooms_realtime": "the most rooms one room-call server process keeps in real time: k rooms "
                      "run through the server's own loop (asyncio, a worker-thread hop per "
                      "120 Hz tick, snapshot + JSON every 1/30 s) for 0.5 s + 2 s, passing when "
                      "every room's real-time factor over the 2 s is >= 0.98; k doubled from 1 "
                      "then bisected, capped at the server's MAX_ROOMS (16)",
    "rooms_flat_out": "floor(0.8 x the best summed real-time factor of 1, 2, 4 ... N rooms "
                      "stepping flat out on one thread each, N = CPUs this process may use; "
                      "median of 3 runs per point): an upper bound for any server on this CPU",
    "timings": "machine-dependent: they say which CPU and load average they came from",
}


# ── markdown ─────────────────────────────────────────────────────────────────────────────────────


def _fmt(value, digits: int = 1, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def markdown(report: dict, title: str = "Physics audit") -> str:
    """The audit as markdown: a summary, then one row per movable body."""
    i, c, s = report["interactability"], report["cost"], report["stability"]
    conc = c.get("concurrency") or {}
    lines = [f"### {title}", ""]
    lines.append(f"- interactable: **{i['interactable']}/{i['tested']}** "
                 f"(grabbable {i['grabbable']}, moved {i['moved']}, stable {i['stable']})"
                 + (f" — failed: {', '.join(i['failed'])}" if i["failed"] else ""))
    m = report["masses"]
    lines.append(f"- masses outside the plausible range: {len(m['outliers'])}/{m['checked']} "
                 "movable" + (" — " + ", ".join(f"{o['name']} {o['mass_kg']:.1f} kg "
                                                f"({o['verdict']})" for o in m["outliers"])
                              if m["outliers"] else "")
                 + f"; static {len(m['static_outliers'])}/{m['static_checked']}")
    o = report["overlaps"]
    lines.append(f"- overlaps > {o['tol_mm']:.0f} mm: {len(o['authored'])} authored, "
                 f"{len(o['settled'])} after settle")
    lines.append(f"- settle moved bodies by up to {report['settle_shift']['max_mm']:.1f} mm "
                 f"({report['settle_shift']['max_body'] or '—'})")
    lines.append(f"- left alone {s['seconds']:.0f} s: max drift {s['max_drift_mm']:.2f} mm "
                 f"({s['max_drift_body'] or '—'}), {s['resets']} resets, {s['respawns']} respawns")
    step = c.get("ms_per_step_spread") or {}
    rtf = c.get("realtime_factor_spread") or {}
    lines.append(f"- cost: {c['ms_per_step']:.3f} ms/step"
                 + (f" ({step['min']:.3f}-{step['max']:.3f} over {step['n']} runs)" if step else "")
                 + f", real-time factor {_fmt(c['realtime_factor'], 0)}x on one thread"
                 + (f" ({rtf['min']:.0f}-{rtf['max']:.0f})" if rtf else ""))
    cap = c.get("capacity") or {}
    if cap:
        curve = ", ".join(f"{k}: {v:.2f}" for k, v in cap["slowest_room_rtf"].items())
        lines.append(f"- rooms one server keeps in real time: **{cap['rooms']}"
                     f"{'+' if cap['at_limit'] else ''}** (slowest room's real-time factor by "
                     f"rooms run: {curve})"
                     + (f" — **measured on a busy machine** (load {cap['load_before']:.1f} on "
                        f"{cap['usable_cpus']} CPUs): it says more about the other processes "
                        "than about this room" if cap.get("machine_busy") else ""))
    if conc:
        sweep = ", ".join(f"{n} threads {v['median']:.0f}x" for n, v in conc["sweep"].items())
        lines.append(f"- flat out: {sweep} (best {conc['aggregate_realtime_factor']:.0f}x, "
                     f"{conc['usable_cpus']} usable CPUs)")
    lines += ["", "| body | joint | category | mass kg | range kg | grab | pull | t (s) | "
              "stable |", "|---|---|---|---|---|---|---|---|---|"]
    for r in report["bodies"]:
        p = r["pull"]
        rng = r["mass_range"]
        if p.get("kind") == "free":
            pull = f"{p.get('mode', '—')} {_fmt(p.get('moved_m'), 2)} m"
        elif p.get("kind") in ("hinge", "slide"):
            unit = p.get("unit", "")
            digits = 1 if unit == "deg" else 3
            pull = f"{_fmt(p.get('moved'), digits)}/{_fmt(p.get('need'), digits)} {unit}"
        else:
            pull = p.get("reason", "—")
        verdict = "" if r["mass_verdict"] in ("ok", "unknown") else f" ({r['mass_verdict']})"
        mark = {True: "yes", False: "**no**", None: "—"}
        lines.append(f"| {r['name']} | {r['joint']} | {r['category']} | {r['mass_kg']:.2f}"
                     f"{verdict} | {f'{rng[0]:g}-{rng[1]:g}' if rng else '—'} | "
                     f"{mark[p.get('grab')]} | {pull} {mark[p.get('pass')]} | "
                     f"{_fmt(p.get('t_reach_s'), 2)} | {mark[p.get('stable')]} |")
    return "\n".join(lines) + "\n"


__all__ = ["DEFINITIONS", "MASS_RANGES", "MassRange", "audit_physics", "body_extent", "classify",
           "concurrency", "markdown", "mass_bounds", "mass_verdict", "name_tokens", "pull_test",
           "scaling_area", "server_capacity", "stability", "step_cost", "testable_bodies",
           "usable_cpus"]
