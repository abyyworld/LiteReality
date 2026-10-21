"""sim_assets.py — the physics a generated object carries, read back and placed in a room.

Every object the reconstruct stage builds now leaves a sidecar beside its GLB: a
`<name>.physics.json` holding the links, joints, masses, inertias, frictions and CONVEX COLLIDERS
that `object_generation.sim` compiled from the mesh, plus the OBJ files those colliders live in.
It is the asset's own statement of what it is physically, gated by a solver before it was ever put
in a room.

This module is how the room-level MuJoCo export reads that statement instead of re-inventing it.
The difference is not cosmetic:

  * the PIVOT of a hinge is authored — `object.py` placed the hinge and recorded where. The room
    exporter's alternative is to guess it from the leaf's geometry, which is right for a door with
    an obvious free edge and wrong for a sash, a lid, or a drawer.
  * the MASS is split across the links by volume, so a door leaf weighs what a door leaf weighs
    rather than a share of the whole object's bounding box.
  * FRICTION comes from the material the recipe deliberately chose. Glass grips at 0.30, carpet at
    0.85, and a room where everything is 0.55 slides wrong in a way no amount of tuning fixes.
  * the COLLIDERS are already decomposed, so a 25-minute CoACD pass over the whole room turns into
    reading a few dozen OBJ files.

`room_ops` may not import the pipeline or the models layer (the architecture test enforces it), so
nothing here imports the compiler. The sidecar is a FILE FORMAT, and this reads it as one — the
dataclasses below mirror `object_generation.sim.properties` and are checked against
`schema_version`.

FRAMES. The sidecar is Z-up metres in the object's own build frame, with each link's geometry
expressed relative to that link's own frame origin (which sits on its joint). A room places the
object with a rotation and a NON-UNIFORM scale — RoomPlan measured a box and the asset was fitted
into it — so everything has to travel through one affine on the way in. `placement_transform`
recovers that affine exactly from the two glTF files, and `place` applies it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1

# glTF is Y-up, the SHELL and MuJoCo are Z-up. Kept local rather than imported from mujoco_scene so
# this module can be read (and tested) on its own.
_A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_A4 = np.eye(4)
_A4[:3, :3] = _A
_A4_INV = np.linalg.inv(_A4)

# trimesh appends `_<6 hex>` to a node name it has already seen, and it picks a DIFFERENT suffix
# each time a file is loaded. So `DoorLeaf_5a8bb9` in the object's own glb and `DoorLeaf_22e63d` in
# the room glb are the same node, and no amount of string equality will say so. Only names that are
# unique within their own file can be matched across the two — which is enough, because one matched
# node determines the whole placement.
_DEDUP_SUFFIX = re.compile(r"_[0-9a-f]{6}$")

MIN_LINK_MASS = 0.15


def base_name(node: str) -> str:
    """The node name with trimesh's de-duplication suffix removed."""
    return _DEDUP_SUFFIX.sub("", node)


@dataclass
class Collider:
    file: str
    volume: float


@dataclass
class Link:
    name: str
    mass: float
    com: list
    inertia: list
    friction: float
    restitution: float
    material: str = ""
    visual: str = ""
    colliders: list = field(default_factory=list)
    mass_source: str = "derived"
    inertia_source: str = "geometry"


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: list
    axis: list
    limit_lower: float
    limit_upper: float
    damping: float = 0.05
    friction: float = 0.02
    effort: float = 20.0
    velocity: float = 3.0
    origin_source: str = "node_transform"


@dataclass
class SimModel:
    name: str
    category: str
    root: str
    links: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    bbox: list = field(default_factory=list)
    total_mass: float = 0.0
    notes: list = field(default_factory=list)
    directory: Path = field(default=Path("."))

    def link(self, name: str) -> Link | None:
        return next((link for link in self.links if link.name == name), None)

    def frames(self) -> dict[str, np.ndarray]:
        """Each link's frame origin in the OBJECT's own Z-up frame.

        A joint records its pivot in the PARENT's frame, so the absolute origin is the chain of
        them accumulated from the root. Written iteratively rather than recursively because a
        malformed sidecar could otherwise describe a cycle and blow the stack.
        """
        origins = {self.root: np.zeros(3)}
        pending = list(self.joints)
        for _ in range(len(pending) + 1):
            if not pending:
                break
            rest = []
            for joint in pending:
                if joint.parent in origins:
                    origins[joint.child] = origins[joint.parent] + np.asarray(joint.origin, float)
                else:
                    rest.append(joint)
            if len(rest) == len(pending):
                break                      # an orphan or a cycle: leave the rest at the root
            pending = rest
        for link in self.links:
            origins.setdefault(link.name, np.zeros(3))
        return origins


def sidecar_dir(room: Path, object_name: str) -> Path | None:
    """The `sim/` directory an object's physics was compiled into, inside the Room package.

    `export_room` carries it in beside the object's own source, so a Room directory is a complete
    statement of the room INCLUDING its physics. Both object kinds are searched because a
    generative mesh (a TRELLIS chair) is as much a rigid body as a recipe-built one.
    """
    for kind in ("Procedural", "Static"):
        candidate = Path(room) / "Objects" / kind / object_name / "sim"
        if (candidate / f"{object_name}.physics.json").is_file():
            return candidate
    return None


def load(path: Path) -> SimModel:
    """One `<name>.physics.json` -> the model. Raises on a schema this code does not know."""
    path = Path(path)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    version = raw.pop("schema_version", 0)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path} is physics schema v{version}, this reads v{SCHEMA_VERSION}. Reading it "
            f"anyway would put a wrong mass or a wrong hinge into a scene and report success."
        )
    links = [Link(**{**row, "colliders": [Collider(**c) for c in row.get("colliders", [])]})
             for row in raw.get("links", [])]
    joints = [Joint(**row) for row in raw.get("joints", [])]
    return SimModel(name=raw["name"], category=raw.get("category", ""),
                    root=raw.get("root", "base_link"), links=links, joints=joints,
                    bbox=raw.get("bbox") or [], total_mass=float(raw.get("total_mass") or 0.0),
                    notes=list(raw.get("notes") or []), directory=path.parent)


# ------------------------------------------------------------------ placement


def placement_transform(object_scene, room_scene, room_nodes) -> np.ndarray | None:
    """The affine that carries the object's own glb frame onto where the room put it (Y-up).

    `build_room` fits an asset into the RoomPlan box it belongs to, which is a rotation about the
    vertical and a per-axis SCALE — the box was measured by the scan and the asset was built to its
    own proportions, so the two rarely agree. Any single node present in both files pins the whole
    thing down:

        T = T_room(node) . T_object(node)^-1

    and every other node then maps through it exactly. Returns None when no node can be matched,
    which is the caller's signal to fall back to deriving physics from the placed mesh instead of
    quietly using an unplaced one.
    """
    from collections import Counter

    def unique(nodes) -> dict[str, str]:
        counts = Counter(base_name(n) for n in nodes)
        return {base_name(n): n for n in nodes if counts[base_name(n)] == 1}

    in_object = unique(list(object_scene.graph.nodes_geometry))
    in_room = unique(list(room_nodes))
    shared = sorted(set(in_object) & set(in_room))
    if not shared:
        return None
    # Prefer the node with the most geometry: a big part pins the transform with the least relative
    # error if the two files ever disagree, and it is the one most likely to be the carcass rather
    # than a fitting that a later edit moved.
    def area(name: str) -> float:
        try:
            return float(object_scene.geometry[object_scene.graph[in_object[name]][1]].area)
        except Exception:                                    # noqa: BLE001 — degenerate geometry
            return 0.0

    best = max(shared, key=area)
    to_object = np.asarray(object_scene.graph[in_object[best]][0], dtype=float)
    to_room = np.asarray(room_scene.graph[in_room[best]][0], dtype=float)
    try:
        return to_room @ np.linalg.inv(to_object)
    except np.linalg.LinAlgError:
        return None


def placement_from_bbox(model: SimModel, placed_mesh, yaw: float) -> np.ndarray | None:
    """The same affine, recovered from GEOMETRY when no node name survived the room build.

    An object placed more than once — four chairs from one cluster, a repeated cabinet — has every
    node renamed for every instance after the first, and a cluster whose own parts already repeat
    has nothing unique inside its own file either. Name matching then reports nothing for objects
    that are perfectly well placed.

    But the placement has a known SHAPE. `build_room` fits an asset into its box as

        world = T(centre) . Rz(yaw) . diag(s) . (p - asset centre)

    with `yaw` stated in the SHELL, so rotating the placed mesh back by that yaw makes the fit
    axis-aligned and the two bounding boxes give `s` and `centre` directly. It is exact for exactly
    the transform the room applies — and ONLY for the right yaw: a wrong one still returns a fit
    whose bounds match, with the wrong scale. Callers go through `fit_placement`, which judges the
    candidates by geometry.

    Returns the Y-up matrix, so it feeds `place` identically to `placement_transform`.
    """
    if not model.bbox or len(model.bbox) != 2:
        return None
    lo = np.asarray(model.bbox[0], dtype=float)
    hi = np.asarray(model.bbox[1], dtype=float)
    asset_size = hi - lo
    if np.any(asset_size < 1e-6):
        return None

    c, s_ = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    aligned = np.asarray(placed_mesh.vertices, dtype=float) @ rotation     # == Rz(-yaw) . v
    placed_lo, placed_hi = aligned.min(axis=0), aligned.max(axis=0)
    scale = (placed_hi - placed_lo) / asset_size
    if np.any(~np.isfinite(scale)) or np.any(np.abs(scale) < 1e-9):
        return None

    linear = rotation @ np.diag(scale)
    centre = rotation @ ((placed_lo + placed_hi) / 2.0)
    transform = np.eye(4)
    transform[:3, :3] = linear
    transform[:3, 3] = centre - linear @ ((lo + hi) / 2.0)
    return _A4_INV @ transform @ _A4


# ------------------------------------------------------------------ proving a placement
#
# `placement_from_bbox` is exact for the yaw it is GIVEN, and nothing else. Hand it the wrong yaw
# and it still returns a transform, one whose bounding box matches the placed mesh perfectly,
# because that is how it was constructed. So the old proof (do the colliders' bounds land on the
# placed mesh's bounds?) could never reject it. tea_room's door is the case that hurt: openings
# live in the SHELL's `openings`, not its `objects`, so the door was handed yaw 0 while it stands
# in a wall at 31 degrees. The fit stretched the door's 15 cm depth to the 56 cm its rotated
# bounding box spans, the leaf's mass was scaled by the same 4.7x volume ratio (27 -> 129 kg),
# and its collider became an axis-aligned slab across the doorway. Comparing the actual GEOMETRY
# is what tells a right placement from a wrong one, and what picks the yaw when none is known.

_FIT_POINTS = 1500


def _nearest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point of `a` to its nearest point of `b` (chunked, numpy only)."""
    out = np.empty(len(a))
    b_sq = np.einsum("ij,ij->i", b, b)
    for start in range(0, len(a), 512):
        chunk = a[start:start + 512]
        d2 = (np.einsum("ij,ij->i", chunk, chunk)[:, None] + b_sq[None, :]
              - 2.0 * chunk @ b.T)
        out[start:start + 512] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def _thin(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) <= count:
        return points
    return points[np.linspace(0, len(points) - 1, count).astype(int)]


def surface_points(mesh, count: int = _FIT_POINTS, seed: int = 0) -> np.ndarray:
    """Vertices plus an even-ish surface sample, so a two-triangle face still has an inside."""
    import trimesh

    points = [np.asarray(mesh.vertices, dtype=float)]
    try:
        if len(mesh.faces) and float(mesh.area) > 0:
            points.append(np.asarray(trimesh.sample.sample_surface(mesh, count, seed=seed)[0],
                                     dtype=float))
    except Exception:                                        # noqa: BLE001 — vertices still count
        pass
    return _thin(np.vstack(points), count)


def model_points(model: SimModel, count: int = _FIT_POINTS) -> np.ndarray | None:
    """Points on the object's own geometry, in its own Z-up frame.

    The link VISUAL is preferred: it is the mesh the room placed, so on the right transform the
    two coincide. A sidecar written without visuals falls back to its colliders, which bulge over
    gaps (a chair's hull spans its legs) and so read a few centimetres off even when right.
    """
    import trimesh

    frames = model.frames()
    meshes = []
    for link in model.links:
        files = ([link.visual] if link.visual and (model.directory / link.visual).is_file()
                 else [c.file for c in link.colliders])
        for name in files:
            path = model.directory / name
            if not path.is_file():
                continue
            try:
                mesh = trimesh.load(str(path), process=False, force="mesh")
            except Exception:                                # noqa: BLE001 — unreadable file
                continue
            if not len(mesh.vertices):
                continue
            mesh = mesh.copy()
            mesh.vertices = np.asarray(mesh.vertices, float) + frames.get(link.name, np.zeros(3))
            meshes.append(mesh)
    if not meshes:
        return None
    return surface_points(trimesh.util.concatenate(meshes), count)


def placement_residual(points: np.ndarray, transform_yup: np.ndarray,
                       placed: np.ndarray, *, symmetric: bool = True) -> float:
    """How far the object, carried by `transform_yup`, lands from the placed geometry (metres).

    Symmetric — the object onto the placed mesh AND the placed mesh onto the object — when choosing
    between fits, because one direction alone forgives a fit that drops the object inside a box
    bigger than itself. The 90th percentile ignores the odd stray vertex without letting a whole
    misplaced part through.
    """
    transform = _A4 @ np.asarray(transform_yup, dtype=float) @ _A4_INV
    world = _transform_points(points, transform)
    there = float(np.percentile(_nearest(world, placed), 90))
    if not symmetric:
        return there
    back = float(np.percentile(_nearest(placed, world), 90))
    return max(there, back)


def _candidate_yaws(yaws, placed_mesh) -> list[float]:
    """Each stated yaw and its quarter turns, then the placed mesh's own principal yaw.

    A quarter turn is the same bounding box with width and depth exchanged, so it is a different
    and equally self-consistent fit; only the geometry can choose between them. The principal axis
    of the placed footprint covers the case where nothing stated a yaw at all.
    """
    import trimesh

    base = [float(y) for y in yaws if y is not None]
    try:
        xy = np.asarray(placed_mesh.vertices, dtype=float)[:, :2]
        transform, _extents = trimesh.bounds.oriented_bounds_2D(xy)
        base.append(-math.atan2(transform[1, 0], transform[0, 0]))
    except Exception:                                        # noqa: BLE001 — degenerate footprint
        pass
    out: list[float] = []
    for yaw in base:
        for k in range(4):
            angle = math.atan2(math.sin(yaw + k * math.pi / 2), math.cos(yaw + k * math.pi / 2))
            if all(abs(math.atan2(math.sin(angle - a), math.cos(angle - a))) > 1e-4 for a in out):
                out.append(angle)
    return out


@dataclass
class Fit:
    transform: np.ndarray            # Y-up, as `place` takes it
    residual: float                  # metres, see `placement_residual`
    yaw: float | None                # the yaw it was fitted with; None for a node-matched one


def fit_placement(model: SimModel, placed_mesh, yaws=()) -> Fit | None:
    """The best bounding-box placement over every plausible yaw, judged by the geometry.

    The STATED yaws are tried first and keep the win on a tie: an object symmetric under a half
    turn fits both ways equally, and there the room's own statement is the better guess.
    """
    points = model_points(model)
    if points is None:
        transform = placement_from_bbox(model, placed_mesh, next(
            (float(y) for y in yaws if y is not None), 0.0))
        return None if transform is None else Fit(transform, float("nan"), None)
    placed = surface_points(placed_mesh)
    best: Fit | None = None
    for yaw in _candidate_yaws(yaws, placed_mesh):
        transform = placement_from_bbox(model, placed_mesh, yaw)
        if transform is None:
            continue
        residual = placement_residual(points, transform, placed)
        if best is None or residual < best.residual - max(1e-3, 0.05 * best.residual):
            best = Fit(transform, residual, yaw)
    return best


def check_placement(model: SimModel, transform_yup: np.ndarray, placed_mesh) -> float | None:
    """Does the object, carried by a transform found some other way (a node match), land on what
    the room placed? One-directional: whatever else the room grouped under the same handle is not
    this transform's to explain. None when the sidecar has no geometry to compare."""
    points = model_points(model)
    if points is None:
        return None
    return placement_residual(points, transform_yup, surface_points(placed_mesh), symmetric=False)


@dataclass
class PlacedLink:
    name: str
    mass: float
    friction: float
    restitution: float
    inertia: np.ndarray              # [ixx, iyy, izz, ixy, ixz, iyz] about the com
    com: np.ndarray                  # world Z-up, metres
    colliders: list                  # trimesh meshes, world Z-up
    mass_source: str
    inertia_source: str


@dataclass
class Placed:
    """A SimModel expressed in the room's world frame, ready to be written as MJCF bodies."""
    model: SimModel
    transform: np.ndarray            # 4x4, world Z-up <- object Z-up
    volume_ratio: float
    links: dict                      # name -> PlacedLink
    pivots: dict                     # link name -> world Z-up pivot
    axes: dict                       # link name -> world Z-up unit axis
    total_mass: float

    def joint_for(self, link: str) -> Joint | None:
        return next((j for j in self.model.joints if j.child == link), None)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _inertia_about_com(mesh, mass: float):
    """Inertia about the centre of mass, from the placed geometry, at the sidecar's mass.

    The sidecar's own tensor describes the object at the size it was BUILT, and the room scales it
    — a door stretched 10% taller has a different tensor, not a scaled mass with the old one. It is
    cheaper and more honest to recompute from the geometry that is actually going into the scene.
    The ladder mesh -> convex hull -> bounding box is the one `object_generation.sim` uses, and it
    exists because a flat visual part (a pane, a poster) has no volume for trimesh to integrate and
    MuJoCo refuses a non-positive-definite inertia at compile time.
    """
    for candidate, source in ((mesh, "geometry"), (mesh.convex_hull, "hull")):
        try:
            if candidate.volume > 1e-9:
                scaled = candidate.copy()
                scaled.density = mass / float(candidate.volume)
                tensor = np.asarray(scaled.moment_inertia, dtype=float)
                if np.all(np.linalg.eigvalsh(tensor) > 0):
                    return tensor, np.asarray(scaled.center_mass, dtype=float), source
        except Exception:                                    # noqa: BLE001 — degenerate mesh
            continue
    extents = np.maximum(np.asarray(mesh.extents, dtype=float), 0.01)
    diagonal = mass / 12.0 * np.array([extents[1] ** 2 + extents[2] ** 2,
                                       extents[0] ** 2 + extents[2] ** 2,
                                       extents[0] ** 2 + extents[1] ** 2])
    return np.diag(diagonal), np.asarray(mesh.bounds, dtype=float).mean(axis=0), "bounding_box"


def place(model: SimModel, transform_yup: np.ndarray) -> Placed:
    """Carry a model's colliders, masses, pivots and axes into the room's world frame.

    `transform_yup` comes from `placement_transform`; it is converted here so that every number
    this returns is Z-up and nothing downstream has to think about frames again.
    """
    import trimesh

    transform = _A4 @ np.asarray(transform_yup, dtype=float) @ _A4_INV
    linear = transform[:3, :3]
    # How much bigger the room made this object. A mass the sidecar DERIVED came from occupancy
    # density over the object's own bounding box, so at a different size it is a different mass;
    # a mass the recipe AUTHORED is a fact about the real object and survives the fit unchanged.
    volume_ratio = abs(float(np.linalg.det(linear))) or 1.0
    frames = model.frames()

    links: dict[str, PlacedLink] = {}
    pivots: dict[str, np.ndarray] = {}
    axes: dict[str, np.ndarray] = {}
    for link in model.links:
        frame = frames.get(link.name, np.zeros(3))
        pivots[link.name] = _transform_points(frame.reshape(1, 3), transform)[0]

        pieces = []
        for collider in link.colliders:
            path = model.directory / collider.file
            if not path.is_file():
                continue
            try:
                piece = trimesh.load(str(path), process=False, force="mesh")
            except Exception:                                # noqa: BLE001 — unreadable collider
                continue
            # The collider was written relative to the link's own frame; put it back in the
            # object's frame before placing it, or every moving part lands on the carcass.
            piece.vertices = _transform_points(np.asarray(piece.vertices, float) + frame, transform)
            if piece.volume > 1e-9 or len(piece.vertices) >= 4:
                pieces.append(piece)
        if not pieces:
            continue

        mass = float(link.mass)
        if link.mass_source != "authored":
            mass *= volume_ratio
        mass = max(mass, MIN_LINK_MASS)
        merged = trimesh.util.concatenate(pieces)
        inertia, com, source = _inertia_about_com(merged, mass)
        links[link.name] = PlacedLink(
            name=link.name, mass=mass, friction=float(link.friction),
            restitution=float(link.restitution),
            inertia=np.array([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                              inertia[0, 1], inertia[0, 2], inertia[1, 2]], dtype=float),
            com=np.asarray(com, dtype=float), colliders=pieces,
            mass_source=link.mass_source, inertia_source=source)

    for joint in model.joints:
        # A joint AXIS is a direction in the object's frame and travels through the linear part —
        # the same map `mujoco_scene._axis_to_world` applies, and for the rotation-plus-diagonal
        # scale a room fit produces it is the correct one. It is renormalised because the scale is
        # not uniform and an unnormalised MuJoCo axis silently changes a slide joint's units.
        world = linear @ np.asarray(joint.axis, dtype=float)
        norm = float(np.linalg.norm(world))
        axes[joint.child] = world / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

    return Placed(model=model, transform=transform, volume_ratio=volume_ratio, links=links,
                  pivots=pivots, axes=axes,
                  total_mass=sum(link.mass for link in links.values()))


def assign_nodes(node_names, model: SimModel) -> dict[str, list[str]]:
    """Room-glb node names -> the link each belongs to.

    A link owns the node it was named for and everything welded under it, and the room appends a
    de-duplication suffix to any name it has seen before. Longest match wins so that `door_leaf`
    does not swallow a `door_leaf_handle` that has its own joint. Anything unclaimed belongs to the
    root, which is what "welded into the carcass" means.
    """
    names = [link.name for link in model.links if link.name != model.root]
    out: dict[str, list[str]] = {}
    for node in node_names:
        stem = base_name(node)
        owner = None
        for candidate in names:
            if stem == candidate or stem.startswith(candidate + "_"):
                if owner is None or len(candidate) > len(owner):
                    owner = candidate
        out.setdefault(owner or model.root, []).append(node)
    return out
