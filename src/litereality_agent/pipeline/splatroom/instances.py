"""The movable objects found in the room splat itself, matched to the physics bodies (no GPU).

RoomPlan regularises a row of chairs: the same size, the same yaw, even spacing. In the atrium
one of three chair boxes stands where there is no chair and the dark office chair beside it has
no body, so sliding or snapping boxes cannot fix the cut. Here the objects are found in the
splat and the bodies are put on them:

    find      opaque splats (`MIN_OPACITY`) above the floor and outside every static box (walls,
              the desk), in `VOXEL` voxels; 26-connected components. A component too large for
              one object of the room's movable categories (two chairs that touch, a chair and the
              sill behind it) is split at its thin bridges (`_parts`).
    classify  each instance's oriented footprint and height against each movable category's
              scanned size (`Category`): one that fits none (a wall's haze, a lamp on the desk,
              anything not standing on the floor) is not an object.
    match     per category, Hungarian matching over the whole room between the free bodies and
              those instances (`match`), by plan distance (gated at `MATCH_MAX`) and shape.
    fit       each matched body is put on its instance (`fit_scene`): free yaw (the footprint's
              principal axes, the template's back toward the instance's tall part), centred on the
              footprint, the template's geoms scaled per axis to the instance's footprint and
              height. An object no body matched gets a clone of the nearest body of its category;
              a body no object matched, with nothing opaque in its box, is a phantom and goes.
    cut       a matched body takes the splats in its instance's voxels grown by `GROW` voxels,
              faint ones too (`Found.labels`): cluster membership, not box + margin, so the haze
              around a chair (three quarters of its splats are faint) goes where the chair goes
              and the floor, the desk and the next chair stay where they are.

`move_test` is the gate every cut is checked against: each real object (a classified instance)
must have >= `MOVE_MIN` of its opaque splats in one body and <= `MOVE_LEAK` in any other.
"""

from __future__ import annotations

import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

VOXEL = 0.03            # metres
MIN_OPACITY = 0.3       # the splats that say where a surface is (most are faint haze)
MIN_OCC = 2             # opaque splats a voxel needs to count as occupied
BAND_LO = 0.05          # metres above the floor where objects start (the floor stays the room's)
CUT_FLOOR = 0.03        # metres above the floor below which nothing is cut out with an object
STATIC_MARGIN = 0.03    # metres around static boxes (walls, the desk) that are never an object
GROW = 1                # voxels an instance is grown by when it takes its splats (faint ones too)
MIN_SPLATS = 600        # opaque splats an instance needs
SPLIT_AREA = 1.5        # a component this many single footprints large is several objects
SPLIT_LEVELS = 16       # splats per voxel up to which a component is thresholded to split it
SIZE_RANGE = (0.5, 2.0)     # instance / template footprint extent that can still be the category
ASPECT_MAX = 2.4        # longest / shortest side of a footprint: longer is a sill, a wall's haze
HEIGHT_RANGE = (0.6, 1.6)   # instance / template height
STAND_MAX = 0.5         # metres above the floor an object's lowest splats must reach down to
MATCH_MAX = 1.2         # metres a body may be from the object it is matched to
SHAPE_WEIGHT = 0.15     # metres of distance one unit of log-size mismatch costs in the matching
SCALE_RANGE = (0.6, 1.6)    # how far a template is stretched per axis to fit its object
BACK_MIN = 0.04         # metres the tall part must sit off-centre to say which way it faces
PHANTOM_MAX = 300       # opaque non-static splats in a body's box below which it holds nothing
CLONE_MIN = 1500        # opaque splats an object with no body needs before a body is cloned
CLONE_SCORE = 0.9       # ... and how well (sum of |log size ratio|) it must fit its category
MOVE_MIN = 0.90         # move test: share of an object in its body
MOVE_LEAK = 0.05        # move test: share of an object in any other body
GRAB_MIN = 0.5          # a body offered for grabbing holds at least this much of a real object


@dataclass
class Category:
    name: str
    half: np.ndarray            # template half extents (x, y, z) in the body frame (median)
    bodies: list[int] = field(default_factory=list)


@dataclass
class Instance:
    id: int                     # label in `Found.grid`
    centre: np.ndarray          # world xy of the oriented footprint's centre
    axes: np.ndarray            # 2x2: footprint principal axes (columns), world xy
    half: np.ndarray            # footprint half extents along `axes`
    bottom: float               # metres above the floor (p1 of its opaque splats)
    top: float                  # metres above the floor (p99)
    back: np.ndarray            # world xy offset of its tall part from its footprint's centre
    splats: int                 # opaque splats (its core)
    cells: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))  # plan cells
    category: str | None = None
    score: float = math.inf     # sum of |log size ratio| against `category`

    def summary(self) -> dict:
        return {"id": self.id, "centre": [round(float(v), 3) for v in self.centre],
                "half": [round(float(v), 3) for v in self.half],
                "bottom": round(self.bottom, 3), "top": round(self.top, 3),
                "splats": self.splats, "category": self.category,
                "score": None if math.isinf(self.score) else round(self.score, 3)}


@dataclass
class Found:
    floor: float                # world z of the floor (the splat's own)
    origin: np.ndarray          # world xyz of voxel (0, 0, 0)
    grid: np.ndarray            # int32 labels, grown by `GROW` (0 = no instance)
    core: np.ndarray            # int32 labels, ungrown
    instances: dict[int, Instance]

    def _lookup(self, grid: np.ndarray, positions: np.ndarray) -> np.ndarray:
        g = np.floor((positions - self.origin) / VOXEL).astype(np.int64)
        ok = np.all((g >= 0) & (g < np.array(grid.shape)), axis=1)
        out = np.zeros(len(positions), np.int32)
        out[ok] = grid[tuple(g[ok].T)]
        return out

    def labels(self, positions: np.ndarray) -> np.ndarray:
        """Per splat: the instance whose grown voxels hold it (0 for none)."""
        return self._lookup(self.grid, positions)

    def core_labels(self, positions: np.ndarray, opacities: np.ndarray) -> np.ndarray:
        """Per splat: the instance whose own voxels hold it, opaque splats only (0 otherwise)."""
        out = self._lookup(self.core, positions)
        out[opacities <= MIN_OPACITY] = 0
        return out

    def objects(self) -> list[Instance]:
        return [i for i in self.instances.values() if i.category is not None]


# ---------------------------------------------------------------------------------- categories


def _free_top_level(model: mujoco.MjModel, body: int) -> bool:
    from litereality_agent.pipeline.splatroom.split import _free_joint

    return _free_joint(model, body) is not None and int(model.body_parentid[body]) == 0


def _category_names(scene_xml: Path | None) -> dict[str, str]:
    """Body name -> category from the quick room's report beside the scene, when there is one."""
    if scene_xml is None:
        return {}
    try:
        doc = json.loads((Path(scene_xml).parent / "quick.json").read_text())
        return {n: str(b.get("category")) for n, b in (doc.get("bodies") or {}).items()
                if isinstance(b, dict) and b.get("category")}
    except (OSError, ValueError, AttributeError):
        return {}


def category_of(name: str, known: dict[str, str] | None = None) -> str:
    """A body's category: the quick room's own word for it, else its name without the index
    (``Chair3`` -> ``chair``)."""
    if known and name in known:
        return known[name].lower()
    return re.sub(r"[_\d]+$", "", name).lower() or name.lower()


def body_frame_box(model: mujoco.MjModel, data: mujoco.MjData, body: int):
    """A body's geoms' bounds in its own frame: (centre, half extents)."""
    from litereality_agent.pipeline.splatroom.split import _union_box

    _b, R, c, half = _union_box(model, data, body)
    return R.T @ (c - np.asarray(data.xpos[body], dtype=np.float64)), half


def categories(model: mujoco.MjModel, data: mujoco.MjData,
               scene_xml: Path | None = None) -> dict[str, Category]:
    """The movable categories: free top-level bodies with geoms, grouped by `category_of`."""
    from litereality_agent.pipeline.splatroom.split import _body_geoms
    from litereality_agent.room_ops.roomcall.model_info import body_name, movable_body_ids

    known = _category_names(scene_xml)
    out: dict[str, Category] = {}
    halves: dict[str, list] = {}
    for b in sorted(movable_body_ids(model)):
        if not _free_top_level(model, b) or not _body_geoms(model, b):
            continue
        c = category_of(body_name(model, b), known)
        out.setdefault(c, Category(c, np.zeros(3))).bodies.append(b)
        halves.setdefault(c, []).append(body_frame_box(model, data, b)[1])
    for c, cat in out.items():
        cat.half = np.median(np.asarray(halves[c]), axis=0)
    return out


def _classify(inst: Instance, cats: dict[str, Category]) -> tuple[str | None, float]:
    """The category an instance fits best (footprint extents and height within range), or None."""
    best, best_score = None, math.inf
    if inst.bottom > STAND_MAX:
        return None, math.inf
    size = np.sort(inst.half)
    if size[1] > ASPECT_MAX * max(size[0], 1e-3):
        return None, math.inf
    for name, cat in cats.items():
        t = np.sort(cat.half[:2])
        r_xy = size / np.maximum(t, 1e-3)
        r_h = inst.top / max(2 * cat.half[2], 1e-3)
        if not (np.all(r_xy >= SIZE_RANGE[0]) and np.all(r_xy <= SIZE_RANGE[1])
                and HEIGHT_RANGE[0] <= r_h <= HEIGHT_RANGE[1]):
            continue
        score = float(np.abs(np.log(r_xy)).sum() + abs(math.log(r_h)))
        if score < best_score:
            best, best_score = name, score
    return best, best_score


# ---------------------------------------------------------------------------------- finding


def floor_height(positions: np.ndarray, opacities: np.ndarray, guess: float) -> float:
    """The floor as the splat has it: the densest centimetre of opaque splats within 15 cm of
    `guess` (RoomPlan's floor can sit a couple of centimetres off)."""
    z = positions[:, 2]
    near = (np.abs(z - guess) < 0.15) & (opacities > MIN_OPACITY)
    if near.sum() < 200:
        return float(guess)
    hist, edges = np.histogram(z[near], bins=30, range=(guess - 0.15, guess + 0.15))
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2)


def _room_xy(model: mujoco.MjModel, data: mujoco.MjData):
    """The room's plan bounds from its walls (`split._shell_geoms`), or None."""
    from litereality_agent.pipeline.splatroom.split import _shell_geoms

    pts = [np.asarray(data.geom_xpos[g])[:2] for g in _shell_geoms(model, data)]
    if len(pts) < 2:
        return None
    pts = np.asarray(pts)
    lo, hi = pts.min(0), pts.max(0)
    if np.any(hi - lo < 1.0):              # one wall, or two facing ones: no plan bounds
        return None
    return lo, hi


def _instance(label: int, pts: np.ndarray, floor: float) -> Instance:
    """An instance's oriented footprint, height and which way its tall part lies."""
    cells = np.unique(np.floor(pts[:, :2] / VOXEL).astype(np.int64), axis=0) * VOXEL + VOXEL / 2
    mu = cells.mean(0)
    if len(cells) >= 3:
        w, v = np.linalg.eigh(np.cov((cells - mu).T))
        axes = v[:, ::-1]
    else:
        axes = np.eye(2)
    local = (cells - mu) @ axes
    lo, hi = np.percentile(local, 3, axis=0), np.percentile(local, 97, axis=0)
    half = np.maximum((hi - lo) / 2 + VOXEL / 2, VOXEL)
    centre = mu + axes @ ((lo + hi) / 2)
    z = pts[:, 2] - floor
    bottom, top = float(np.percentile(z, 1)), float(np.percentile(z, 99))
    tall = z > bottom + 0.7 * (top - bottom)
    low = z < bottom + 0.55 * (top - bottom)
    back = np.zeros(2)
    if tall.sum() >= 20 and low.sum() >= 20:
        hi_cells = np.unique(np.floor(pts[tall, :2] / VOXEL), axis=0) * VOXEL
        lo_cells = np.unique(np.floor(pts[low, :2] / VOXEL), axis=0) * VOXEL
        back = hi_cells.mean(0) - lo_cells.mean(0)
    return Instance(label, centre, axes, half, bottom, top, back, len(pts), cells)


def _plan_area(mask: np.ndarray) -> float:
    return float(mask.any(axis=2).sum()) * VOXEL * VOXEL


def _parts(occ: np.ndarray, mask: np.ndarray, single: float, level: int) -> list[np.ndarray]:
    """`mask` (one component) as the objects in it: while it is `SPLIT_AREA` single footprints
    large or more, the voxels it holds are thresholded at more and more splats per voxel until
    its dense cores come apart (the haze bridging two chairs, or a chair and a sill, is thin);
    each core then takes the voxels nearest it, and is split again the same way."""
    from scipy import ndimage

    if _plan_area(mask) < SPLIT_AREA * single:
        return [mask]
    total = float(occ[mask].sum())
    for t in range(level + 1, SPLIT_LEVELS + 1):
        lab, n = ndimage.label((occ >= t) & mask, structure=np.ones((3, 3, 3)))
        if n < 2:
            continue
        sizes = ndimage.sum(occ, lab, index=np.arange(1, n + 1))
        big = [k + 1 for k in np.flatnonzero(sizes >= max(MIN_SPLATS, 0.05 * total))]
        if len(big) < 2:
            continue
        seeds = np.zeros(mask.shape, np.int32)
        for k, label in enumerate(big, 1):
            seeds[lab == label] = k
        nearest = _grow_within(seeds, mask)
        out: list[np.ndarray] = []
        for k in range(1, len(big) + 1):
            out += _parts(occ, mask & (nearest == k), single, t)
        return out
    return [mask]


def _grow_within(seeds: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """`seeds` grown through `mask` a voxel at a time (26-connected): each voxel of the mask
    goes to the seed it is fewest steps from along the mask, not through empty space. Voxels no
    seed reaches go to the nearest one in a straight line."""
    from scipy import ndimage

    out = seeds.copy()
    for _ in range(max(mask.shape) * 2):
        free = mask & (out == 0)
        if not free.any():
            break
        grown = ndimage.grey_dilation(out, size=(3, 3, 3))
        take = free & (grown > 0)
        if not take.any():
            break
        out[take] = grown[take]
    if (mask & (out == 0)).any() and out.any():
        idx = ndimage.distance_transform_edt(out == 0, return_distances=False,
                                             return_indices=True)
        out = np.where(mask & (out == 0), out[tuple(idx)], out)
    return out


def _split_large(lab: np.ndarray, occ: np.ndarray, labels: list[int], single: float,
                 next_label: int) -> int:
    """Split, in place, every component in `labels` that holds several objects (`_parts`).
    Returns the next free label."""
    from scipy import ndimage

    boxes = ndimage.find_objects(lab)
    for label in labels:
        sub = boxes[label - 1]
        if sub is None:
            continue
        mask = lab[sub] == label
        parts = _parts(occ[sub], mask, single, MIN_OCC)
        view = lab[sub]
        for part in parts[1:]:
            view[part] = next_label
            next_label += 1
    return next_label


def find(positions: np.ndarray, opacities: np.ndarray, model: mujoco.MjModel,
         data: mujoco.MjData, *, cats: dict[str, Category] | None = None,
         floor: float | None = None, grow: int = GROW) -> Found:
    """The objects standing in the room splat (see the module docstring)."""
    from scipy import ndimage

    from litereality_agent.pipeline.splatroom.split import _inside_any, _static_boxes
    from litereality_agent.room_ops.roomcall.model_info import movable_body_ids

    cats = categories(model, data) if cats is None else cats
    pos = np.asarray(positions, dtype=np.float64)
    op = np.asarray(opacities, dtype=np.float32)
    free = [b for c in cats.values() for b in c.bodies]
    if floor is None:
        guess = min((float(np.asarray(data.xpos[b])[2] + body_frame_box(model, data, b)[0][2]
                           - body_frame_box(model, data, b)[1][2]) for b in free), default=None)
        if guess is None:
            low = pos[op > MIN_OPACITY, 2]
            guess = float(np.percentile(low, 2)) if len(low) else 0.0
        floor = floor_height(pos, op, guess)
    tallest = max((2 * c.half[2] for c in cats.values()), default=1.0)
    top = floor + min(max(1.5 * tallest, 0.8), 2.2)
    sel = (op > MIN_OPACITY) & (pos[:, 2] > floor + BAND_LO) & (pos[:, 2] < top)
    room = _room_xy(model, data)
    if room is not None:
        sel &= np.all((pos[:, :2] >= room[0]) & (pos[:, :2] <= room[1]), axis=1)
    idx = np.flatnonzero(sel)
    empty = Found(floor, np.zeros(3), np.zeros((1, 1, 1), np.int32),
                  np.zeros((1, 1, 1), np.int32), {})
    if len(idx):
        static = _static_boxes(model, data, movable_body_ids(model))
        if static:
            with np.errstate(all="ignore"):
                idx = idx[~_inside_any(pos[idx], static, STATIC_MARGIN)]
    if len(idx) < MIN_SPLATS:
        return empty
    origin = np.floor(pos[idx].min(0) / VOXEL) * VOXEL - (grow + 1) * VOXEL
    g = np.floor((pos[idx] - origin) / VOXEL).astype(np.int64)
    shape = tuple(g.max(0) + grow + 2)
    occ = np.zeros(shape, np.int32)
    np.add.at(occ, tuple(g.T), 1)
    lab, n = ndimage.label(occ >= MIN_OCC, structure=np.ones((3, 3, 3)))
    lab = lab.astype(np.int32)
    of = lab[tuple(g.T)]
    counts = np.bincount(of, minlength=n + 1)
    single = max((4 * float(c.half[0] * c.half[1]) for c in cats.values()), default=0.3)
    _split_large(lab, occ, [k for k in range(1, n + 1) if counts[k] >= MIN_SPLATS], single,
                 n + 1)
    of = lab[tuple(g.T)]
    instances: dict[int, Instance] = {}
    keep = np.zeros(int(lab.max()) + 1, bool)
    for label in np.unique(of):
        if label == 0:
            continue
        rows = of == label
        if rows.sum() < MIN_SPLATS:
            continue
        inst = _instance(int(label), pos[idx[rows]], floor)
        if room is not None and not np.all((inst.centre >= room[0]) & (inst.centre <= room[1])):
            continue
        inst.category, inst.score = _classify(inst, cats)
        instances[int(label)] = inst
        keep[label] = True
    core = np.where(keep[lab], lab, 0).astype(np.int32)
    grid = core
    if grow > 0 and instances:
        grown = ndimage.grey_dilation(core, size=(2 * grow + 1,) * 3)
        grid = np.where(core > 0, core, grown).astype(np.int32)
    return Found(floor, origin, grid, core, instances)


# ---------------------------------------------------------------------------------- matching


@dataclass
class Match:
    pairs: dict[int, int]                  # body id -> instance id
    lonely_bodies: list[int]               # free bodies of a category no instance matched
    lonely_objects: list[int]              # classified instances no body matched
    distance: dict[int, float] = field(default_factory=dict)   # body id -> metres


def _body_xy(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> np.ndarray:
    c, _half = body_frame_box(model, data, body)
    R = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
    return (np.asarray(data.xpos[body], dtype=np.float64) + R @ c)[:2]


def match(found: Found, model: mujoco.MjModel, data: mujoco.MjData,
          cats: dict[str, Category]) -> Match:
    """Hungarian matching, per category over the whole room, of free bodies to objects."""
    from scipy.optimize import linear_sum_assignment

    pairs: dict[int, int] = {}
    distance: dict[int, float] = {}
    lonely_bodies: list[int] = []
    lonely_objects: list[int] = []
    for name, cat in cats.items():
        objs = [i for i in found.objects() if i.category == name]
        bodies = list(cat.bodies)
        if not objs:
            lonely_bodies += bodies
            continue
        if not bodies:
            lonely_objects += [o.id for o in objs]
            continue
        big = 1e6
        cost = np.full((len(bodies), len(objs) + len(bodies)), big)
        for r, b in enumerate(bodies):
            xy = _body_xy(model, data, b)
            for c, o in enumerate(objs):
                d = float(np.hypot(*(o.centre - xy)))
                if d <= MATCH_MAX:
                    cost[r, c] = d + SHAPE_WEIGHT * o.score
            cost[r, len(objs) + r] = MATCH_MAX + SHAPE_WEIGHT * 2   # unmatched
        rows, cols = linear_sum_assignment(cost)
        taken = set()
        for r, c in zip(rows, cols):
            if c < len(objs) and cost[r, c] < big:
                pairs[bodies[r]] = objs[c].id
                distance[bodies[r]] = float(np.hypot(*(objs[c].centre
                                                       - _body_xy(model, data, bodies[r]))))
                taken.add(objs[c].id)
            else:
                lonely_bodies.append(bodies[r])
        lonely_objects += [o.id for o in objs if o.id not in taken]
    return Match(pairs, lonely_bodies, lonely_objects, distance)


# ---------------------------------------------------------------------------------- fitting


def _yaw_of(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> float:
    R = np.asarray(data.xmat[body], dtype=np.float64).reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def fit_pose(inst: Instance, template_centre: np.ndarray, template_half: np.ndarray,
             bottom_z: float, yaw0: float, *, stretch: bool = True) -> dict:
    """Where a body with this template goes to stand on `inst`: {"pos", "yaw", "scale"}.

    The yaw turns the template's back (its +y) toward the instance's tall part (a chair's
    backrest); with no clear tall part, one of the footprint's principal axes goes along the
    body's x, the one nearest the body's own yaw (`yaw0`). The footprint is measured in that
    heading. The template is stretched per axis to the footprint and the
    height (within `SCALE_RANGE`; not at all without `stretch`), and stands on `bottom_z` as it
    did."""
    back = np.asarray(inst.back, dtype=np.float64)
    if np.hypot(*back) >= BACK_MIN:
        yaw = math.atan2(-back[0], back[1])      # body +y, (-sin yaw, cos yaw), along the back
    else:
        a1 = math.atan2(inst.axes[1, 0], inst.axes[0, 0])
        yaw = min((_wrap(a1 + k * math.pi / 2) for k in range(4)),
                  key=lambda a: abs(_wrap(a - yaw0)))
    c, s = math.cos(yaw), math.sin(yaw)
    R2 = np.array([[c, -s], [s, c]])
    local = inst.cells @ R2                     # into the body's heading
    lo, hi = np.percentile(local, 3, axis=0), np.percentile(local, 97, axis=0)
    width, depth = np.maximum((hi - lo) / 2 + VOXEL / 2, VOXEL)
    centre = R2 @ ((lo + hi) / 2)
    scale = np.array([width / max(template_half[0], 1e-3), depth / max(template_half[1], 1e-3),
                      inst.top / max(2 * template_half[2], 1e-3)])
    scale = np.clip(scale, *SCALE_RANGE) if stretch else np.ones(3)
    xy = centre - R2 @ (scale[:2] * template_centre[:2])
    bottom_local = template_centre[2] - template_half[2]
    z = bottom_z - scale[2] * bottom_local
    return {"pos": [float(xy[0]), float(xy[1]), float(z)], "yaw": float(yaw),
            "scale": [float(v) for v in scale]}


def _floats(text: str | None, default: str) -> np.ndarray:
    return np.array([float(v) for v in re.split(r"\s+", (text or default).strip())])


def _fmt(values) -> str:
    return " ".join(f"{float(v):.5g}" for v in values)


_ROTATION = ("quat", "euler", "axisangle", "xyaxes", "zaxis")


def _scalable(body: ET.Element) -> bool:
    """Whether a body's geoms can be stretched per axis: only boxes, spheres, capsules and
    cylinders (no meshes) in the body's own axes, with their sizes written out, and nothing
    nested under it."""
    for child in body:
        if child.tag in ("freejoint", "joint"):
            continue
        if child.tag != "geom":
            return False
        if child.get("type", "sphere") not in ("box", "sphere", "capsule", "cylinder"):
            return False
        if child.get("size") is None or child.get("class") is not None \
                or any(child.get(k) for k in _ROTATION if k != "quat"):
            return False
        q = child.get("quat")
        if q is not None and not np.allclose(np.abs(_floats(q, "1 0 0 0")), [1, 0, 0, 0],
                                             atol=1e-6):
            return False
    return True


def _scale_body(body: ET.Element, scale: np.ndarray) -> None:
    for geom in body.findall("geom"):
        kind = geom.get("type", "sphere")
        size = _floats(geom.get("size"), "0")
        if geom.get("fromto"):
            ft = _floats(geom.get("fromto"), "0 0 0 0 0 0")
            geom.set("fromto", _fmt(np.concatenate([ft[:3] * scale, ft[3:] * scale])))
        elif geom.get("pos"):
            geom.set("pos", _fmt(_floats(geom.get("pos"), "0 0 0") * scale))
        if kind == "box" and len(size) >= 3:
            size = size[:3] * scale
        elif kind == "sphere":
            size = size[:1] * float(np.mean(scale))
        elif kind in ("capsule", "cylinder"):
            size = np.concatenate([size[:1] * float(np.mean(scale[:2])), size[1:2] * scale[2]])
        geom.set("size", _fmt(size))


def _rename(elem: ET.Element, old: str, new: str) -> None:
    """Every name in a cloned body made its own: ``Chair1_seat`` -> ``Chair3_seat``."""
    for e in elem.iter():
        v = e.get("name")
        if v is not None:
            e.set("name", new + v[len(old):] if v.startswith(old) else f"{v}_{new}")


def _scanned(scene_xml: Path, name: str) -> ET.Element:
    """A top-level body as the scanned scene has it."""
    for e in ET.parse(scene_xml).getroot().find("worldbody").findall("body"):
        if e.get("name") == name:
            return e
    raise KeyError(name)


def _next_name(existing: set[str], like: str) -> str:
    stem = re.sub(r"\d+$", "", like) or like
    k = 0
    while f"{stem}{k}" in existing:
        k += 1
    return f"{stem}{k}"


def _phantom(positions: np.ndarray, opacities: np.ndarray, model: mujoco.MjModel,
             data: mujoco.MjData, body: int, floor: float) -> int:
    """Opaque splats inside a body's box, above the floor, outside everything static."""
    from litereality_agent.pipeline.splatroom.split import _inside_any, _static_boxes, _union_box
    from litereality_agent.room_ops.roomcall.model_info import movable_body_ids

    box = _union_box(model, data, body)
    with np.errstate(all="ignore"):
        inside = _inside_any(positions, [box], 0.0) & (opacities > MIN_OPACITY) \
            & (positions[:, 2] > floor + 0.10)
        idx = np.flatnonzero(inside)
        static = _static_boxes(model, data, movable_body_ids(model))
        if len(idx) and static:
            idx = idx[~_inside_any(positions[idx], static, STATIC_MARGIN)]
    return int(len(idx))


def fit_scene(positions: np.ndarray, opacities: np.ndarray, scene_xml: Path,
              out_name: str = "scene_splat.xml", *, remove_phantoms: bool = True,
              clone: bool = True, log=print) -> tuple[Path | None, dict]:
    """The physics put on the objects found in the splat: `scene_xml` with every matched free
    body re-posed and resized, a clone for every object no body matched, and phantoms removed,
    written beside it as `out_name` (relative mesh and texture paths still resolve). Returns
    (that path, or None when no object was found, and the report)."""
    from litereality_agent.room_ops.roomcall.model_info import body_name

    scene_xml = Path(scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cats = categories(model, data, scene_xml)
    pos = np.asarray(positions, dtype=np.float64)
    op = np.asarray(opacities, dtype=np.float32)
    found = find(pos, op, model, data, cats=cats)
    m = match(found, model, data, cats)
    report: dict = {"floor": round(found.floor, 4),
                    "instances": [i.summary() for i in found.instances.values()],
                    "bodies": {}, "clones": {}, "removed": [], "kept_unmatched": []}
    if not m.pairs and not m.lonely_objects:
        log("  instances: no object found in the splat; the boxes cut it")
        return None, report
    tree = ET.parse(scene_xml)
    world = tree.getroot().find("worldbody")
    elems = {e.get("name"): e for e in world.findall("body")}

    def place(elem: ET.Element, body: int, inst: Instance) -> dict:
        centre, half = body_frame_box(model, data, body)
        bottom = float(np.asarray(data.xpos[body])[2] + centre[2] - half[2])
        stretch = _scalable(elem)
        fit = fit_pose(inst, centre, half, bottom, _yaw_of(model, data, body), stretch=stretch)
        if stretch:
            _scale_body(elem, np.asarray(fit["scale"]))
        elem.set("pos", _fmt(fit["pos"]))
        h = fit["yaw"] / 2
        elem.set("quat", _fmt([math.cos(h), 0.0, 0.0, math.sin(h)]))
        old = np.asarray(data.xpos[body], dtype=np.float64)
        return {"instance": inst.id, "from": [round(float(v), 3) for v in old],
                "to": [round(v, 3) for v in fit["pos"]],
                "yaw_deg": [round(math.degrees(_yaw_of(model, data, body)), 1),
                            round(math.degrees(fit["yaw"]), 1)],
                "scale": [round(v, 3) for v in fit["scale"]],
                "moved_m": round(float(np.hypot(*(np.asarray(fit["pos"][:2]) - old[:2]))), 3),
                "object_centre": [round(float(v), 3) for v in inst.centre],
                "object_splats": inst.splats}

    for b, i in sorted(m.pairs.items()):
        name = body_name(model, b)
        if name in elems:
            report["bodies"][name] = place(elems[name], b, found.instances[i])
    names = {body_name(model, b) for b in range(model.nbody)}
    if clone:
        for i in m.lonely_objects:
            inst = found.instances[i]
            if inst.splats < CLONE_MIN or inst.score > CLONE_SCORE:
                continue
            cat = next((c for c in cats.values() if c.name == inst.category), None)
            if cat is None or not cat.bodies:
                continue
            donor = min(cat.bodies, key=lambda b: float(np.hypot(*(_body_xy(model, data, b)
                                                                    - inst.centre))))
            donor_name = body_name(model, donor)
            if donor_name not in elems:
                continue
            new = _next_name(names, donor_name)
            names.add(new)
            # from the scanned template: the donor itself may already have been resized
            elem = copy.deepcopy(_scanned(scene_xml, donor_name))
            _rename(elem, donor_name, new)
            report["clones"][new] = {"of": donor_name, **place(elem, donor, inst)}
            kids = list(world)
            world.insert(kids.index(elems[donor_name]) + 1, elem)
            log(f"  instances: {new} cloned from {donor_name} for an object with no body "
                f"at {np.round(inst.centre, 2).tolist()}")
    for b in m.lonely_bodies:
        name = body_name(model, b)
        held = _phantom(pos, op, model, data, b, found.floor)
        if remove_phantoms and held < PHANTOM_MAX and name in elems:
            world.remove(elems[name])
            report["removed"].append({"body": name, "opaque_splats_in_box": held})
            log(f"  instances: {name} removed: nothing stands in its box ({held} splats)")
        else:
            report["kept_unmatched"].append({"body": name, "opaque_splats_in_box": held})
    out = scene_xml.with_name(out_name)
    tree.write(out, encoding="unicode")
    try:
        mujoco.MjModel.from_xml_path(str(out))
    except Exception as exc:  # noqa: BLE001 — something else named a removed body: keep them
        if not report["removed"]:
            raise
        log(f"  instances: removing phantoms broke the scene ({exc}); kept, not grabbable")
        for r in report["removed"]:
            world.append(copy.deepcopy(_scanned(scene_xml, r["body"])))
            report["kept_unmatched"].append({**r, "not_grabbable": True})
        report["removed"] = []
        tree.write(out, encoding="unicode")
        mujoco.MjModel.from_xml_path(str(out))
    fitted = {**report["bodies"], **report["clones"]}
    shrunk = _settle(out, [n for n, r in fitted.items() if r["scale"] != [1.0, 1.0, 1.0]])
    if shrunk:
        for name, f in shrunk.items():
            elem = next(e for e in world.findall("body") if e.get("name") == name)
            _scale_body(elem, np.array([f, f, 1.0]))
            fitted[name]["scale"] = [round(fitted[name]["scale"][0] * f, 3),
                                     round(fitted[name]["scale"][1] * f, 3),
                                     fitted[name]["scale"][2]]
            fitted[name]["shrunk_to_clear"] = round(f, 3)
        tree.write(out, encoding="unicode")
    for name, r in report["bodies"].items():
        log(f"  instances: {name} -> object at {r['object_centre']} ({r['moved_m']} m, yaw "
            f"{r['yaw_deg'][0]} -> {r['yaw_deg'][1]} deg, scale {r['scale']})")
    return out, report


SETTLE_STEP = 0.94      # a fitted body overlapping another is narrowed by this much at a time ...
SETTLE_MIN = 0.7        # ... down to this much of its fitted footprint
SETTLE_SLACK = 0.005    # metres of overlap left alone


def _settle(scene: Path, names: list[str]) -> dict[str, float]:
    """Fitted bodies that overlap one another or anything static (two chairs that touch, each
    fitted to its whole footprint, would start interpenetrating and fly apart): each is narrowed
    in plan, a step at a time, until it clears (`split._penetration`) or reaches `SETTLE_MIN`.
    Returns name -> the factor its footprint was narrowed by (only those narrowed)."""
    from litereality_agent.pipeline.splatroom.split import _penetration

    if not names:
        return {}
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
    ids = {n: b for n, b in ids.items() if b >= 0}
    factor = dict.fromkeys(ids, 1.0)
    for _ in range(12):
        mujoco.mj_forward(model, data)
        bad = [n for n, b in ids.items()
               if _penetration(model, data, b) > SETTLE_SLACK and factor[n] * SETTLE_STEP
               >= SETTLE_MIN - 1e-9]
        if not bad:
            break
        for n in bad:
            b = ids[n]
            factor[n] *= SETTLE_STEP
            for g in range(int(model.body_geomadr[b]),
                           int(model.body_geomadr[b]) + int(model.body_geomnum[b])):
                model.geom_pos[g][:2] *= SETTLE_STEP
                if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                    model.geom_size[g][:2] *= SETTLE_STEP
                else:
                    model.geom_size[g][0] *= SETTLE_STEP
    return {n: f for n, f in factor.items() if f < 1.0}


# ---------------------------------------------------------------------------------- the gate


def move_test(found: Found, positions: np.ndarray, opacities: np.ndarray, owner: np.ndarray,
              names: dict[int, str]) -> dict:
    """Where each real object's opaque splats went: per object (a classified instance) the share
    in each body and in the room, by count. It passes when one body holds >= `MOVE_MIN` of it
    and no other holds > `MOVE_LEAK`. Also, per body that holds splats, whether it holds at least
    `GRAB_MIN` of some object of a category the room has objects of (`grabbable`)."""
    core = found.core_labels(np.asarray(positions, dtype=np.float64), opacities)
    objects = found.objects()
    rows = []
    best_share: dict[str, float] = {}
    for inst in objects:
        mine = core == inst.id
        n = int(mine.sum())
        if not n:
            continue
        ids, counts = np.unique(owner[mine], return_counts=True)
        shares = {("room" if i < 0 else names.get(int(i), str(i))): round(float(c / n), 4)
                  for i, c in zip(ids, counts)}
        bodies = {k: v for k, v in shares.items() if k != "room"}
        top = max(bodies.items(), key=lambda kv: kv[1]) if bodies else (None, 0.0)
        leak = max((v for k, v in bodies.items() if k != top[0]), default=0.0)
        ok = top[1] >= MOVE_MIN and leak <= MOVE_LEAK
        for k, v in bodies.items():
            best_share[k] = max(best_share.get(k, 0.0), v)
        rows.append({"object": inst.id, "category": inst.category,
                     "centre": [round(float(v), 3) for v in inst.centre], "splats": n,
                     "body": top[0], "in_body": round(float(top[1]), 4),
                     "in_room": shares.get("room", 0.0), "leak": round(float(leak), 4),
                     "shares": shares, "pass": bool(ok)})
    return {"objects": rows, "pass": bool(rows) and all(r["pass"] for r in rows),
            "best_share": best_share, "move_min": MOVE_MIN, "move_leak": MOVE_LEAK}
