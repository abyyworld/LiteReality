"""articulated.py — the objects stage's openable furniture: an Infinigen-Sim asset in place of a
template, for every scanned object Infinigen has a factory for.

TRELLIS changes how a cabinet LOOKS; a cabinet is also something you open. For each object whose
category maps to an Infinigen factory (`models.articulated.spec_for_object`: cabinets, wall
cabinets, a drawer chest when asked for, dishwashers, fridges, ovens) and for every door and
window opening, this stage builds an articulated asset sized to the scanned box and puts it in
the room as the object's own body:

* **Same handle.** The object's body keeps its name, pose and joint (a cabinet stays fixed to the
  floor), so a live swap carries everything it carried before; its template boxes go and the
  asset's base link takes their place — Infinigen's hulls as colliders, its meshes as the look —
  and each door, drawer, rack or sash becomes a child body ``<handle>_<part>`` on its own hinge
  or slide. A room door is the exception that proves the rule: its LEAF keeps the door's handle
  and the hinge its joint name (``Wall10_Door_0_hinge``), exactly as the quick room's leaf, so the
  angle someone left it at survives the swap; the frame is ``<handle>_frame``.
* **Same physics estimate.** The object's mass is the quick room's (box volume x
  ``OCCUPANCY_DENSITY``) shared over its links by material volume; a door leaf takes the whole of
  the door's, as the quick room's leaf does. Joints get ``JOINT_DEFAULTS`` (`infinigen.build`).
* **Kept from the quick room.** A counter's worktop (Infinigen makes no counters: the cabinet goes
  under the top the quick room already has, so the sink unit keeps its counter), and a
  floor-standing dishwasher or oven stands on a plinth so its door clears the floor.
* **Placed by the scan.** Into the RoomPlan box with the box's yaw, its front on the box's local
  -Y — the side RoomPlan says the object faces, unless that side is against a wall and the other
  is not (RoomPlan has the front backwards for a few objects in twenty, and a cabinet facing
  the wall opens into it): then it is turned round (`faces_a_wall`). An opening's asset stands
  in the wall's cut-out,
  its front flush with the wall's room side, the door hinged on the jamb the quick room chose and
  opening into the room, open as far as RoomPlan saw it open and no further than it is free to.
* **Room to swing.** A cabinet RoomPlan put flush against a side wall cannot open the door hinged
  on that side: the door's own thickness sweeps into the wall within a few degrees (kitchens fit a
  filler strip for this). Every side-hinged door is swung to 90° in the compiled room; a side
  whose door meets the building or fixed furniture is moved in by as much as it went in
  (`clear_swings`), so the door opens — at most `MAX_INSET` per side.
* **Coloured by the photos.** The asset is tinted with the colour quickroom measured for the
  object in the capture (`infinigen.tint_texture`).

* **A doorway without a curb.** Infinigen's door frame runs round the bottom too; that bar goes
  under the floor without a collider and the leaf hangs the quick room's 1 cm above the floor
  (`sink_sill`).

Each finished asset is checked in the room before it is handed over — it must compile, and
nothing of it may start inside anything: a chair or stool it overlaps (its handles stand proud
of the scanned box, where the quick room left a tucked chair) is moved out of its way along the
floor, the move becoming part of its edit — and arrives as a scene edit (`build.Arrival`), so the
versions stream exactly as TRELLIS's meshes do. An object whose asset fails keeps its template
and is given back to TRELLIS (`build.generate_room`); Infinigen's results are cached
(`infinigen.AssetCache`), so a re-run costs nothing.
"""

from __future__ import annotations

import copy
import json
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from litereality_agent.models.articulated import infinigen as ig
from litereality_agent.pipeline.quickroom import build as quick
from litereality_agent.pipeline.quickroom import find_room_usdz, parse_room
from litereality_agent.pipeline.quickroom import textures as surface_photos
from litereality_agent.pipeline.realroom.build import Arrival, RealRoom, asset_dirs, compose

SOURCE = "infinigen"
# m in front of the wall's scanned plane that an opening's front face stands (into the room).
OPENING_PROUD = 0.005
# Initial overlaps: deeper than this between a moving part and anything is fixed (a door is swung
# back towards closed), or the asset is refused — the room call's settle would fire it.
OVERLAP_TOL = 0.001
DOOR_STEP = math.radians(5.0)
# A movable body (a chair, a stool) the asset starts in is moved out of its way, as the quick room
# separates its own furniture: an asset's handles stand up to `ig.OUTLIER_FRONT` proud of the
# scanned box, and the quick room left tucked chairs 2 mm from that box. Moved further than
# this, or still stuck after `PUSH_ROUNDS`, and the asset is refused instead (TRELLIS has it).
MAX_PUSH = 0.30
PUSH_ROUNDS = 60
PUSH_MARGIN = quick.PUSH_MARGIN
CHECK_NAME = ".articulated_check.xml"
SWING = math.radians(90.0)     # a side-hinged door must open at least this far ...
# ... and it is tried at each of these shares of that: a door's outer corner sweeps furthest past
# its cabinet's side half way open, not at the end.
SWING_STEPS = (0.25, 0.5, 0.75, 1.0)
SWING_CLEAR = 0.005            # m: ... with this much to spare once its side is moved in
MAX_INSET = 0.06               # m a side may be moved in for its door to swing
SWING_ROUNDS = 3
# A box side this close to a wall stands against it; the other side must be this much further off
# for the box to count as backwards.
AGAINST_WALL = 0.15
# A door opening starting this close to the floor reaches it: its frame's bottom bar goes under
# the floor and the leaf hangs the quick room's clearance above it (`sink_sill`).
SILL_AT_FLOOR = 0.02
LEAF_CLEARANCE = quick.LEAF_CLEARANCE


@dataclass
class Target:
    """One object or opening to build: what (``spec``), where (``placement``, in the frame of
    ``body``), with which physics (``overrides``), and how it goes into the scene."""

    handle: str
    kind: str                           # "object" | "door" | "window"
    spec: ig.Spec
    placement: ig.Placement
    overrides: ig.Overrides
    category: str = ""
    keep: tuple[str, ...] = ()          # the template's geoms that stay (a counter's worktop)
    extra: list[dict] = field(default_factory=list)      # geoms added (a plinth)
    body: dict | None = None            # a new root body {name, pos, quat} (openings)
    remove_geoms: tuple[str, ...] = ()  # world geoms it replaces (a window's glass and bars)
    remove_bodies: tuple[str, ...] = ()  # bodies it replaces (the quick room's door leaf)
    prefix: str | None = None
    primary: str | None = None
    # free bodies (by name) moved out of its way, world-frame metres (`check_in_room`)
    moves: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    on_floor: bool = False              # a door whose opening reaches the floor (`sink_sill`)
    kept_mass: float = 0.0              # kg of the template kept (a worktop), not in the estimate

    def summary(self) -> dict:
        out = {"kind": self.kind, "factory": self.spec.factory, "key": self.spec.key,
               "size": list(self.spec.size), "category": self.category}
        if self.kept_mass:
            out["kept_mass_kg"] = round(self.kept_mass, 3)
        return out


def _vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([float(v) for v in text.split()]) if text else np.array(default, float)


def _hex(value: str | None) -> tuple[float, float, float] | None:
    try:
        text = str(value).lstrip("#")
        return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except (TypeError, ValueError):
        return None


def _photo_rgb(quick_report: dict, handle: str, group: str = "objects"
               ) -> tuple[float, float, float] | None:
    """The capture's colour for ``handle`` — only when it came from the photos."""
    entry = ((quick_report.get("colours") or {}).get(group) or {}).get(handle)
    if not isinstance(entry, dict) or entry.get("source", "frames") != "frames":
        return None
    return _hex(entry.get("rgb"))


def _quat_yaw(yaw: float) -> str:
    return " ".join(f"{v:.6g}" for v in (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


# ── deciding ─────────────────────────────────────────────────────────────────────────────────────


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def faces_a_wall(box, walls: Sequence[tuple[Sequence[float], Sequence[float]]]) -> bool:
    """Whether ``box`` (a `crops.Box`) has its front (local -Y) against a wall and its back
    (local +Y) clear of one — RoomPlan's front is then backwards."""
    if not walls:
        return False
    rot = box.rotation[:2, :2]
    centre = np.asarray(box.centre[:2], dtype=float)
    front = centre + rot @ np.array([0.0, -float(box.half[1])])
    back = centre + rot @ np.array([0.0, float(box.half[1])])

    def nearest(p: np.ndarray) -> float:
        return min(_segment_distance(p, np.asarray(a, float)[:2], np.asarray(b, float)[:2])
                   for a, b in walls)

    f, k = nearest(front), nearest(back)
    return f < AGAINST_WALL and k > f + AGAINST_WALL


def _walls(room: RealRoom) -> list[tuple]:
    usdz = find_room_usdz(room.scan_dir)
    if usdz is None:
        return []
    try:
        return [(w.start, w.end) for w in parse_room(usdz).walls.values()]
    except Exception:  # noqa: BLE001 — without walls nothing is turned
        return []


def plan_objects(room: RealRoom, *, drawers: Sequence[str] = (), seed: int = ig.DEFAULT_SEED
                 ) -> dict[str, Target]:
    """A `Target` for every quick-room body Infinigen has a factory for."""
    bodies = {b.get("name"): b for b in room.base.iter("body") if b.get("name")}
    walls = _walls(room)
    info = room.quick.get("bodies") or {}
    out: dict[str, Target] = {}
    for handle, box in room.boxes.items():
        rec = info.get(handle) or {}
        if box.kind == "door_leaf" or box.joint == "hinge" or handle not in bodies:
            continue
        size = 2.0 * box.half
        gap = box.bottom - room.floor_z
        hung = str(rec.get("why", "")).startswith("hung") or gap > ig.HUNG_GAP
        body = bodies[handle]
        worktop = body.find(f"geom[@name='{handle}_worktop']")
        top = 2.0 * float(_vec(worktop.get("size"))[2]) if worktop is not None else 0.0
        rgb = _photo_rgb(room.quick, handle)
        spec = ig.spec_for_object(box.category, size, bottom_gap=gap, hung=hung,
                                  drawers=handle in set(drawers), worktop=top, rgb=rgb, seed=seed)
        if spec is None:
            continue
        mass = float(rec.get("mass_kg") or 0.0)
        kept = 0.0
        if worktop is not None and worktop.get("density"):
            kept = float(worktop.get("density")) * float(np.prod(2.0 * _vec(worktop.get("size"))))
            mass -= kept
        if mass <= 0:
            mass = float(np.prod(size)) * quick.density(box.category)
        friction = _template_friction(body)
        # The asset's box: the scanned box less the worktop (kept) and a plinth (added).
        lo = box.local_centre - box.half
        hi = box.local_centre + box.half
        hi = hi - np.array([0.0, 0.0, top])
        extra = []
        plinth = ig.PLINTH.get(spec.factory, 0.0) if gap < ig.STANDING_GAP else 0.0
        if plinth:
            extra.append(_plinth(handle, lo, hi, plinth, friction))
            lo = lo + np.array([0.0, 0.0, plinth])
        out[handle] = Target(
            handle=handle, kind="object", spec=spec, category=box.category,
            placement=ig.Placement(tuple(float(v) for v in (lo + hi) / 2),
                                   tuple(float(v) for v in hi - lo),
                                   turned=faces_a_wall(box, walls)),
            overrides=ig.Overrides(mass=max(mass, 0.5), friction=friction, tint=rgb),
            keep=(f"{handle}_worktop",) if worktop is not None else (), extra=extra,
            kept_mass=kept)
    return out


def _template_friction(body: ET.Element) -> float:
    for geom in body.findall("geom"):
        text = geom.get("friction")
        if text:
            return float(text.split()[0])
    return quick.friction_for("")


def _plinth(handle: str, lo: np.ndarray, hi: np.ndarray, height: float, friction: float) -> dict:
    """The toe-kick under a fitted appliance: full width, set back from the front."""
    setback = ig.PLINTH_SETBACK
    centre = ((lo[0] + hi[0]) / 2, (lo[1] + setback + hi[1]) / 2, lo[2] + height / 2)
    half = ((hi[0] - lo[0]) / 2 - 0.005, (hi[1] - lo[1] - setback) / 2, height / 2)
    return {"name": f"{handle}_plinth", "type": "box", "pos": centre, "size": half,
            "rgba": (*ig.PLINTH_RGB, 1.0), "friction": friction}


def plan_openings(room: RealRoom, *, seed: int = ig.DEFAULT_SEED) -> dict[str, Target]:
    """A `Target` for every door and window of the scan's RoomPlan."""
    usdz = find_room_usdz(room.scan_dir)
    if usdz is None:
        return {}
    plan = parse_room(usdz)
    world = room.base.find("worldbody")
    bodies = {b.get("name"): b for b in world.findall("body")} if world is not None else {}
    geoms = {g.get("name") for g in world.findall("geom")} if world is not None else set()
    info = room.quick.get("bodies") or {}
    out: dict[str, Target] = {}
    for handle, op in plan.openings.items():
        if op.kind not in ("door", "window") or op.wall not in plan.walls:
            continue
        wall = plan.walls[op.wall]
        length = wall.length
        t0, t1 = (min(length, max(0.0, t)) for t in op.span)
        z0, z1 = max(0.0, op.sill), min(plan.height, op.sill + op.height)
        if t1 - t0 < 0.2 or z1 - z0 < 0.2:
            continue
        inward = np.asarray(surface_photos.inward_normal(plan, op.wall), dtype=float)
        x_axis = np.array([-inward[1], inward[0]])           # local -Y is into the room
        yaw = math.atan2(x_axis[1], x_axis[0])
        centre2 = np.asarray(wall.start) + np.asarray(wall.along) * ((t0 + t1) / 2)
        pos = (float(centre2[0]), float(centre2[1]), plan.floor_z + (z0 + z1) / 2)
        width, height = t1 - t0, z1 - z0
        placement = ig.Placement((0.0, 0.0, 0.0), (width, quick.WALL_THICKNESS, height),
                                 keep_depth=True, front_y=-OPENING_PROUD)
        if op.kind == "door":
            leaf = bodies.get(handle)
            hinge = "left"
            angle = 0.0
            if leaf is not None:
                rel = _vec(leaf.get("pos"))[:2] - centre2
                hinge = "right" if float(rel @ x_axis) > 0 else "left"
                joint = leaf.find("joint")
                if joint is not None:
                    angle = float(joint.get("ref") or 0.0)
            mass = float((info.get(handle) or {}).get("mass_kg") or
                         width * height * quick.DOOR_UNIT_DEPTH * quick.density("door"))
            spec = ig.door_spec(width, height, hinge=hinge,
                                wall_thickness=quick.WALL_THICKNESS, seed=seed)
            out[handle] = Target(
                handle=handle, kind="door", spec=spec, category="door", placement=placement,
                overrides=ig.Overrides(mass=mass, friction=quick.friction_for("wood"),
                                       tint=_photo_rgb(room.quick, handle, "doors"),
                                       moving_mass=True, open_angle=angle),
                body={"name": f"{handle}_frame", "pos": pos, "quat": _quat_yaw(yaw)},
                remove_bodies=(handle,) if leaf is not None else (),
                prefix=f"{handle}_frame", primary=handle, on_floor=z0 < SILL_AT_FLOOR)
        else:
            mass = width * height * quick.WALL_THICKNESS * quick.density("window")
            spec = ig.window_spec(width, height, seed=seed)
            out[handle] = Target(
                handle=handle, kind="window", spec=spec, category="window",
                placement=placement,
                overrides=ig.Overrides(mass=mass, friction=quick.friction_for("glass"),
                                       tint=quick.FRAME_RGB, moving_mass=True),
                body={"name": handle, "pos": pos, "quat": _quat_yaw(yaw)},
                remove_geoms=tuple(g for g in geoms if g == f"{handle}_glass"
                                   or g.startswith(f"{handle}_frame_")))
    return out


# ── into the room ────────────────────────────────────────────────────────────────────────────────


def sink_sill(target: Target, share: float) -> Target:
    """``target`` with its box extended down under the floor so that the asset's lowest
    ``share`` of height (a door frame's bottom bar, `infinigen.drop_sill`) is below it and the
    leaf's bottom `LEAF_CLEARANCE` above it; the top stays where the scan put it."""
    if not target.on_floor or share <= 0 or share >= 0.5:
        return target
    cx, cy, cz = target.placement.centre
    w, d, h = target.placement.size
    top, bottom = cz + h / 2, cz - h / 2
    # bottom' + share * (top - bottom') = bottom + LEAF_CLEARANCE, solved for bottom'
    new_bottom = (bottom + LEAF_CLEARANCE - share * top) / (1.0 - share)
    placement = replace(target.placement, centre=(cx, cy, (top + new_bottom) / 2),
                        size=(w, d, top - new_bottom))
    return replace(target, placement=placement)


def make_edit(target: Target, built: ig.Built) -> Callable[[ET.Element], None]:
    """The scene edit that puts ``built`` into a scene root in place of ``target``'s template."""
    assets = [copy.deepcopy(el) for el in built.assets]
    base_geoms = [copy.deepcopy(el) for el in built.base_geoms]
    children = [copy.deepcopy(el) for el in built.children]
    extra = list(target.extra)
    moves = {k: np.asarray(v, dtype=float) for k, v in target.moves.items()}

    def edit(root: ET.Element) -> None:
        asset = root.find("asset")
        if asset is None:
            asset = ET.Element("asset")
            root.insert(0, asset)
        for el in assets:
            asset.append(copy.deepcopy(el))
        world = root.find("worldbody")
        if target.body is None:
            body = next((b for b in root.iter("body") if b.get("name") == target.handle), None)
            if body is None:
                raise KeyError(f"{target.handle} is not a body of the scene")
            materials = {}
            for geom in list(body.findall("geom")):
                if geom.get("name") not in target.keep:
                    body.remove(geom)
        else:
            for geom in list(world.findall("geom")):
                if geom.get("name") in target.remove_geoms:
                    world.remove(geom)
            for old in list(world.findall("body")):
                if old.get("name") in target.remove_bodies:
                    world.remove(old)
            body = ET.SubElement(world, "body", name=target.body["name"],
                                 pos=" ".join(f"{v:.6g}" for v in target.body["pos"]),
                                 quat=target.body["quat"])
            materials = {}
        for el in base_geoms:
            body.append(copy.deepcopy(el))
        for g in extra:
            rgba = " ".join(f"{v:.4g}" for v in g["rgba"])
            name = materials.get(rgba)
            if name is None:
                name = f"ig_{target.handle}_{g['name']}_mat"
                ET.SubElement(asset, "material", name=name, rgba=rgba)
                materials[rgba] = name
            # density 0: it is the fitting's (a welded plinth), not part of the object's mass
            ET.SubElement(body, "geom", name=g["name"], type=g["type"],
                          pos=" ".join(f"{v:.6g}" for v in g["pos"]),
                          size=" ".join(f"{v:.6g}" for v in g["size"]), material=name,
                          friction=f"{g['friction']:.3g} 0.03 0.005", density="0")
        for el in children:
            body.append(copy.deepcopy(el))
        if built.excludes:
            contact = root.find("contact")
            if contact is None:
                contact = ET.SubElement(root, "contact")
            for a, b in built.excludes:
                ET.SubElement(contact, "exclude", body1=a, body2=b)
        # Moves ADD to wherever the body is: two cabinets may each push the same stool.
        for other in world.findall("body") if moves else ():
            delta = moves.get(other.get("name"))
            if delta is not None:
                other.set("pos", " ".join(f"{v:.6g}" for v in _vec(other.get("pos")) + delta))

    return edit


def _set_open(built: ig.Built, body_name: str, angle: float) -> bool:
    """Start ``body_name`` (a child body with a hinge) at ``angle`` radians open."""
    for top in built.children:
        for body in top.iter("body"):
            if body.get("name") != body_name:
                continue
            joint = body.find("joint")
            if joint is None or joint.get("type") != "hinge":
                return False
            axis = _vec(joint.get("axis"), (0, 0, 1))
            q = [math.cos(angle / 2), *(axis / (np.linalg.norm(axis) or 1.0) * math.sin(angle / 2))]
            if abs(angle) > 1e-9:
                body.set("quat", " ".join(f"{v:.10g}" for v in q))
            elif "quat" in body.attrib:
                del body.attrib["quat"]
            joint.set("ref", f"{angle:.10g}")
            return True
    return False


def _compile_with(room: RealRoom, target: Target, built: ig.Built):
    import mujoco

    path = room.out_dir / CHECK_NAME
    try:
        root, _missing = compose(room.base, {}, [make_edit(target, built)])
        path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
        return mujoco.MjModel.from_xml_path(str(path))
    finally:
        path.unlink(missing_ok=True)


def swing_blocks(room: RealRoom, target: Target, built: ig.Built) -> dict[int, float]:
    """For each side of the asset (-1: its -X side, +1: +X), how deep a door hinged on that side
    goes into the building or fixed furniture as it swings `SWING` open. Empty: they all swing."""
    import mujoco

    model = _compile_with(room, target, built)
    data = mujoco.MjData(model)
    own = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in built.names.values()}
    centre_x = float(target.placement.centre[0])
    blocks: dict[int, float] = {}
    for j in range(model.njnt):
        b = int(model.jnt_bodyid[j])
        if b not in own or int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        if abs(float(model.jnt_axis[j][2])) < 0.9:
            continue                         # a flap: it swings down, not into a side wall
        lo, hi = (float(v) for v in model.jnt_range[j])
        open_to = math.copysign(min(SWING, max(abs(lo), abs(hi))), hi if abs(hi) >= abs(lo)
                                else lo)
        depth = 0.0
        for share in SWING_STEPS:
            mujoco.mj_resetData(model, data)
            data.qpos[int(model.jnt_qposadr[j])] = share * open_to
            mujoco.mj_forward(model, data)
            for i in range(data.ncon):
                c = data.contact[i]
                b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                if b not in (b1, b2):
                    continue
                other = b2 if b1 == b else b1
                if other in own or int(model.body_weldid[other]) != 0:
                    continue                  # its own cabinet, or something that moves
                depth = max(depth, -float(c.dist))
        if depth <= 0:
            continue
        # which side it hangs on: the hinge's x in the object's frame
        hinge_x = float(model.body_pos[b][0]) if int(model.body_parentid[b]) in own else centre_x
        side = 1 if hinge_x > centre_x else -1
        blocks[side] = max(blocks.get(side, 0.0), depth)
    return blocks


def clear_swings(room: RealRoom, target: Target, build_fn: Callable[[Target], ig.Built]
                 ) -> tuple[Target, ig.Built, dict]:
    """Build ``target``; while a side-hinged door cannot swing open, move that side in by the
    depth it met (plus `SWING_CLEAR`) and build again. Returns the target as placed, its
    `Built` and the insets (m per side)."""
    built = build_fn(target)
    inset = {-1: 0.0, 1: 0.0}
    if target.kind != "object":
        return target, built, {}
    for _ in range(SWING_ROUNDS):
        blocks = swing_blocks(room, target, built)
        grow = {s: min(MAX_INSET - inset[s], d + SWING_CLEAR) for s, d in blocks.items()
                if inset[s] < MAX_INSET - 1e-4}
        if not grow:
            break
        for s, d in grow.items():
            inset[s] += d
        cx, cy, cz = target.placement.centre
        w, dd, h = target.placement.size
        # the -X side moved in pushes the centre to +X, and the other way round
        moved = replace(target.placement,
                        centre=(cx + (grow.get(-1, 0.0) - grow.get(1, 0.0)) / 2, cy, cz),
                        size=(w - sum(grow.values()), dd, h))
        target = replace(target, placement=moved)
        built = build_fn(target)
    done = {("-X" if s < 0 else "+X"): round(v, 4) for s, v in inset.items() if v > 0}
    return target, built, done


def _contacts(model, data, tol: float) -> list[tuple[int, int, float, np.ndarray]]:
    """Contacts deeper than ``tol``: (body1, body2, depth, normal from 1 to 2)."""
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < -tol:
            out.append((int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2]),
                        -float(c.dist), np.array(c.frame[:3], dtype=float)))
    return out


def _free_bodies(model) -> dict[int, int]:
    """The room's movable bodies (a free joint, straight under the world): body -> qpos address."""
    import mujoco

    out = {}
    for b in range(1, model.nbody):
        if int(model.body_parentid[b]) != 0 or int(model.body_jntnum[b]) < 1:
            continue
        j = int(model.body_jntadr[b])
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            out[b] = int(model.jnt_qposadr[j])
    return out


def _push_free(model, data, own: set[int], free: dict[int, int]
               ) -> tuple[dict[int, np.ndarray], list[tuple[int, int, float]]]:
    """Move the free bodies the asset (``own``) starts inside out of it, along the floor, as
    `quickroom.build.separate` does: along the contact normal, half each when two free bodies
    meet; lifted only out of something below it (a cabinet's top). Moves the bodies in
    ``data.qpos``; returns each one's displacement and the overlaps no free body can resolve
    (a moving part of the asset inside the building)."""
    import mujoco

    start = {b: data.qpos[adr:adr + 3].copy() for b, adr in free.items()}
    involved = set(own)
    blocked: list[tuple[int, int, float]] = []
    for _round in range(PUSH_ROUNDS):
        pushes: dict[int, dict[int, np.ndarray]] = {}
        blocked = []
        for b1, b2, depth, normal in _contacts(model, data, OVERLAP_TOL):
            if b1 not in involved and b2 not in involved:
                continue                    # the base room's own business
            if b1 not in free and b2 not in free:
                blocked.append((b1, b2, depth))
                continue
            for me, other, n in ((b1, b2, -normal), (b2, b1, normal)):
                if me not in free:
                    continue
                if n[2] > 0.7 and other not in free:          # standing in what holds it up
                    step = np.array([0.0, 0.0, depth + PUSH_MARGIN])
                else:
                    h = np.array([n[0], n[1], 0.0])
                    if np.linalg.norm(h) < 0.3:               # near-vertical: go sideways
                        h = data.xpos[me] - data.xpos[other]
                        h[2] = 0.0
                    if np.linalg.norm(h) < 1e-9:
                        h = np.array([1.0, 0.0, 0.0])
                    share = 0.5 if other in free else 1.0
                    step = h / np.linalg.norm(h) * (depth + PUSH_MARGIN) * share
                have = pushes.setdefault(me, {}).get(other)
                if have is None or np.linalg.norm(step) > np.linalg.norm(have):
                    pushes[me][other] = step
        if blocked or not pushes:
            break
        for me, steps in pushes.items():
            adr = free[me]
            data.qpos[adr:adr + 3] += np.sum(list(steps.values()), axis=0)
            involved.add(me)                # what it is pushed into must make room too
        mujoco.mj_forward(model, data)
    else:
        left = [(b1, b2, d) for b1, b2, d, _n in _contacts(model, data, OVERLAP_TOL)
                if b1 in involved or b2 in involved]
        blocked = blocked or left
    moved = {b: data.qpos[adr:adr + 3] - start[b] for b, adr in free.items()
             if float(np.linalg.norm(data.qpos[adr:adr + 3] - start[b])) > 1e-6}
    return moved, blocked


def check_in_room(room: RealRoom, target: Target, built: ig.Built) -> tuple[Target, dict]:
    """Compile the base room with this asset alone in it and make sure nothing of it starts
    inside anything: a movable body it overlaps (a tucked chair) is moved out of its way and
    the move recorded in the target (`Target.moves`, applied by its edit); an open door that
    meets the building is swung back towards closed until it is free; anything else still
    overlapping raises. Returns the target as checked and what it found."""
    import mujoco

    path = room.out_dir / CHECK_NAME
    angle = target.overrides.open_angle if target.primary else 0.0
    moves: dict[str, np.ndarray] = {k: np.asarray(v, float) for k, v in target.moves.items()}

    def compiled():
        current = replace(target, moves={k: tuple(float(x) for x in v)
                                         for k, v in moves.items()})
        root, _missing = compose(room.base, {}, [make_edit(current, built)])
        path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return current, model, data

    def name(model, b: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "world"

    try:
        for _ in range(40):
            current, model, data = compiled()
            own = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                   for n in built.names.values()}
            moved, blocked = _push_free(model, data, own, _free_bodies(model))
            for b, delta in moved.items():
                moves[name(model, b)] = moves.get(name(model, b), np.zeros(3)) + delta
            if not blocked:
                break
            primary = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target.primary) \
                if target.primary else -1
            if angle > 1e-6 and any(primary in (a, b) for a, b, _ in blocked):
                angle = max(0.0, angle - DOOR_STEP)
                _set_open(built, target.primary, angle)
                continue
            worst = max(blocked, key=lambda h: h[2])
            raise ValueError(f"{len(blocked)} contact(s) at the start, worst "
                             f"{name(model, worst[0])} <-> {name(model, worst[1])} "
                             f"{worst[2] * 1000:.1f} mm deep")
        else:
            raise ValueError("could not free the door from what it starts in")
        far = {k: float(np.linalg.norm(v)) for k, v in moves.items()
               if float(np.linalg.norm(v)) > MAX_PUSH}
        if far:
            k = max(far, key=far.get)
            raise ValueError(f"{k} would have to move {far[k]:.2f} m out of its way "
                             f"(at most {MAX_PUSH:g} m)")
        if moved:
            current, model, data = compiled()          # the room as it will be written
        overlaps = quick.initial_overlaps(model)
        check = {"compiles": True, "start_angle_deg": round(math.degrees(angle), 1),
                 "initial_overlaps": overlaps, "nbody": int(model.nbody)}
        if current.moves:
            check["moved_out_of_its_way_m"] = {k: [round(float(x), 4) for x in v]
                                               for k, v in sorted(current.moves.items())}
        return current, check
    finally:
        path.unlink(missing_ok=True)


class ArticulatedSource:
    """Infinigen's openable furniture as a `build.Source` for `build.generate_room`: `plan`
    claims the objects it has a factory for, `stream` builds them (cached first) and hands each
    over as a scene edit once it is checked in the room."""

    name = SOURCE

    def __init__(self, runner: ig.Runner, cache: ig.AssetCache | None = None, *,
                 drawers: Sequence[str] = (), openings: bool = True,
                 seed: int = ig.DEFAULT_SEED, log: Callable[[str], Any] = print) -> None:
        self.runner = runner
        self.cache = cache or ig.AssetCache()
        self.drawers = tuple(drawers)
        self.openings = openings
        self.seed = int(seed)
        self.log = log
        self.targets: dict[str, Target] = {}
        self.planned = False

    def plan(self, room: RealRoom) -> list[str]:
        if not self.planned:
            self.targets = plan_objects(room, drawers=self.drawers, seed=self.seed)
            if self.openings:
                self.targets.update(plan_openings(room, seed=self.seed))
            self.planned = True
        room.taken |= {h for h, t in self.targets.items() if t.kind == "object"}
        return sorted(self.targets)

    def describe(self) -> str:
        return ", ".join(f"{h} ({t.spec.factory})" for h, t in sorted(self.targets.items()))

    def install(self, room: RealRoom, target: Target, folder: Path) -> Arrival:
        """One finished export into the room: clean, fit, override, write its files, check."""
        started = time.perf_counter()
        try:
            asset = ig.clean(ig.load_asset(folder, target.spec))
            if target.kind == "door":
                asset, share = ig.drop_sill(asset)
                target = sink_sill(target, share)
            mesh_dir, tex_dir = asset_dirs(room.base)

            def build_fn(t: Target) -> ig.Built:
                built = ig.build(asset, t.handle, t.placement, t.overrides, prefix=t.prefix,
                                 primary=t.primary)
                ig.write_files(built, room.out_dir / mesh_dir, room.out_dir / tex_dir)
                return built

            target, built, inset = clear_swings(room, target, build_fn)
            target, check = check_in_room(room, target, built)
        except Exception as exc:  # noqa: BLE001 — this object keeps its template
            return Arrival(target.handle, SOURCE, None, time.perf_counter() - started,
                           f"{type(exc).__name__}: {exc}")
        if inset:
            check["inset_for_swing_m"] = inset
        summary = {**target.summary(), **built.report, "check": check}
        return Arrival(target.handle, SOURCE, make_edit(target, built),
                       time.perf_counter() - started, summary=summary)

    def stream(self, room: RealRoom) -> Iterator[Arrival]:
        self.plan(room)
        if not self.targets:
            return
        self.log(f"  {SOURCE}: {len(self.targets)} openable object(s): {self.describe()}")
        by_key: dict[str, list[Target]] = {}
        for target in self.targets.values():
            by_key.setdefault(target.spec.key, []).append(target)
        seen: set[str] = set()
        for result in ig.generate([t.spec for t in self.targets.values()], self.runner,
                                  self.cache, log=self.log):
            seen.add(result.spec.key)
            for target in by_key.get(result.spec.key, []):
                if not result.ok:
                    yield Arrival(target.handle, SOURCE, None, result.seconds, result.error)
                    continue
                arrival = self.install(room, target, result.folder)
                if arrival.ok:
                    arrival = replace(arrival, seconds=result.seconds + arrival.seconds,
                                      summary={**arrival.summary, "cached": result.cached,
                                               "generate_seconds": round(result.seconds, 2)})
                yield arrival
        for key, targets in by_key.items():
            if key not in seen:
                for target in targets:
                    yield Arrival(target.handle, SOURCE, None, 0.0, "the runner never built it")


def articulate(room: RealRoom, source: ArticulatedSource, *, write: bool = True
               ) -> list[Arrival]:
    """Every planned object built and installed at once (no streaming); writes one version."""
    source.plan(room)
    arrivals = list(source.stream(room))
    for arrival in arrivals:
        if arrival.ok:
            room.edits[arrival.handle] = arrival.edit
            room.replaced[arrival.handle] = {"source": arrival.source, **arrival.summary}
        else:
            room.errors[arrival.handle] = f"{arrival.source}: {arrival.error}"
    if write:
        room.write()
    return arrivals


def report(room: RealRoom) -> dict:
    """objects.json's articulated part, for printing."""
    return json.loads(json.dumps(room.replaced, default=str))


__all__ = ["SOURCE", "ArticulatedSource", "Target", "articulate", "check_in_room", "make_edit",
           "plan_objects", "plan_openings", "report", "sink_sill"]
