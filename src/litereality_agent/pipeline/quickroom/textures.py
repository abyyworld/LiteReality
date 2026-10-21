"""textures.py — the scanned room's own walls, floor and ceiling as photographs, in a second or two.

A box room in the right colours is recognisably a room; the same room with its OWN walls on it —
the whiteboard, the door RoomPlan missed, the poster, the skirting, the floor's planks and rug —
is recognisably THAT room, which is what the people on the call scanned it for. The capture holds
every pixel needed: each frame has an RGB image, a LiDAR depth map aligned with it, and the camera
pose. So every surface of the room is laid out as a grid of texels in the pipeline frame (Z up,
where `build` puts the walls) and each texel looks itself up in the photos:

* **project** it into each frame (pinhole, the pose's inverse) and keep it only if the frame
  really saw THAT point: in the image, in front of the camera, on the room side of the surface,
  and not hidden — the depth map says what the camera actually hit along that ray, so a sofa in
  front of a wall, or a table over the floor, is never painted onto them;
* **score** each sighting (head-on, close, away from the image border) and keep the THREE best per
  texel; the texel's colour is their median. One sighting is the sharpest but carries every
  reflection and every misregistered edge; the median of three shrugs both off;
* **balance exposure**: ARKit changes exposure and white balance from frame to frame, and a wall
  stitched from five frames shows five patches. Every texel seen by several frames says how two
  frames disagree, so one gain per frame and channel (a robust fixed point over all such texels)
  brings every frame to the room's consensus before the median is taken;
* **fill** what no frame saw (the top of a wall, the floor under the sofa, the strip behind a
  cabinet) with a smooth continuation of what was seen (push-pull over an image pyramid), never
  black. A surface almost nobody saw gets no photo at all and keeps its flat colour.

It is a per-frame loop of numpy over only the texels that frame can see — surfaces facing away and
tiles outside the view are culled before a single texel is projected — so a room costs ~1-3 s on
a laptop, inside the quick room's budget, and a hard deadline stops early (with the frames spread
over the capture, so an early stop still saw every side of the room). No model, no GPU, numpy and
Pillow only.

Image layout: a wall's image runs along the wall from its start (columns) and down from the
ceiling (rows), so a point ``t`` metres along it and ``z`` above the floor is at uv
``(t / L, 1 - z / H)`` — MuJoCo's mesh texcoord convention, where v counts from the image's FIRST
row (checked: an XML ``texcoord`` is kept as written and the PNG's top row is ``tex_data``'s
first), which is glTF's too. The floor's and ceiling's images are the floor plan with +X to the
right and +Y up the image. `Surface.uv` maps any world point.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from litereality_agent.pipeline.quickroom.colours import capture_frames
from litereality_agent.pipeline.quickroom.roomplan import RoomObject, RoomPlan

# ── resolution and budget ────────────────────────────────────────────────────────────────────────
PPM = 100.0                # texels per metre. A frame decoded at half size sees a wall 3 m away at
                           # ~230 px/m, so 100 keeps what a typical view resolves.
CEILING_PPM_SCALE = 0.5    # the ceiling is seen at grazing angles, blurred anyway: half the texels
MAX_TEXELS = 900_000       # all surfaces together; a bigger room gets a lower ppm, not a longer wait
MIN_PPM = 40.0
SHRINK = 2                 # JPEG draft-mode decode at 1/2 (1920x1440 -> 960x720)
MAX_FRAMES = 40            # evenly spread over the capture
BUDGET_S = 3.0             # projection stops here; what was seen so far is used
WORKERS = 3                # frames decoded and projected in parallel (numpy and PIL drop the GIL)

# ── what counts as a sighting ────────────────────────────────────────────────────────────────────
NEAR, FAR = 0.1, 8.0       # m from the camera
DEPTH_TOL = 0.08           # m: the depth map may hit this far IN FRONT of the texel and it still
                           # counts as the texel (RoomPlan's planes are good to a few cm)
OCCLUDED = 0.15            # m: the depth map hit something this far in front: a foreground object
NEAR_WINDOW = 5            # depth pixels: the occlusion test takes the nearest return in this square
MIN_DEPTH = 0.1            # m: LiDAR returns closer than this are noise
EDGE_PX = 60.0             # px at the working resolution over which a sighting fades at the border
STRICT_FACING = 0.12       # cos of the viewing angle off the surface's plane: ~7 degrees
RELAX_FACING = 0.05        # a glancing view, used only where nothing better saw the texel
FRONT_MARGIN = 0.02        # m: a camera must be at least this far on the room side of a surface
TOPK = 3                   # the best sightings kept per texel; the colour is their median
TILE = 16                  # texels per side of a culling tile

# ── furniture hides what is behind and under it ─────────────────────────────────────────────────
BEHIND_GAP = 0.35          # m: furniture this close to a wall hides the wall behind it
HUNG = 0.25                # m above the floor: higher than this, the floor under it is seen

# ── exposure balance ─────────────────────────────────────────────────────────────────────────────
GAIN_ROUNDS = 4
GAIN_RANGE = (0.6, 1.7)    # a frame more than this far off the consensus is a bad estimate, not a
                           # bad frame
GAIN_SAMPLE = 5            # every 5th shared texel is plenty for a median per frame
GAIN_LUMA = (12.0, 245.0)  # sightings this dark or clipped say nothing about exposure

# ── what is worth a photo ────────────────────────────────────────────────────────────────────────
MIN_COVERAGE = 0.08        # share of a surface seen; below it, a flat colour is more honest
MIN_EXTENT = 0.05          # m: walls shorter than this keep their box
FLOOR_LIFT = 0.0015        # m the drawn floor and ceiling sit off their colliders (no z-fighting)


@dataclass
class Surface:
    """One textured surface: where its image lies in the room, and (once estimated) the image.

    ``origin`` is the world position of the image's top-left corner, ``u_axis`` runs along its
    columns and ``v_axis`` down its rows, ``extent`` is how many metres the image spans along each.
    ``normal`` points INTO the room — the side the photo is taken from and looked at. ``drawn``
    marks the texels the room actually draws (a wall minus its openings, the floor inside its
    outline), ``hidden`` those furniture stands in front of or on."""

    name: str
    kind: str                                   # wall | floor | ceiling
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    extent: tuple[float, float]
    normal: np.ndarray
    shape: tuple[int, int]                      # (rows, cols)
    drawn: np.ndarray | None = None             # (rows, cols) bool
    hidden: np.ndarray | None = None            # (rows, cols) bool
    image: np.ndarray | None = None             # (rows, cols, 3) uint8 sRGB
    coverage: float = 0.0                       # share of the drawn, unhidden texels a frame saw
    seen: float = 0.0                           # ... and saw head-on (the strict test)

    def uv(self, points: np.ndarray) -> np.ndarray:
        """Texture coordinates of world points: (u, v) with v from the image's first row."""
        rel = np.atleast_2d(np.asarray(points, dtype=float)) - self.origin
        return np.column_stack([rel @ self.u_axis / self.extent[0],
                                rel @ self.v_axis / self.extent[1]])

    def texel_centres(self) -> np.ndarray:
        rows, cols = self.shape
        a = (np.arange(cols) + 0.5) / cols * self.extent[0]
        b = (np.arange(rows) + 0.5) / rows * self.extent[1]
        aa, bb = np.meshgrid(a, b)
        return (self.origin + aa[..., None] * self.u_axis + bb[..., None] * self.v_axis)


@dataclass
class SurfaceSet:
    """Every surface of the room and how its photos were made. ``surfaces`` holds only the ones
    worth drawing (an image and enough coverage); `summary` reports all of them."""

    surfaces: dict[str, Surface] = field(default_factory=dict)
    source: str = "none"                        # "frames" | "none"
    frames_used: int = 0
    frames_total: int = 0
    ppm: float = 0.0
    seconds: float = 0.0
    note: str = ""
    gains: dict[int, list[float]] = field(default_factory=dict)
    rejected: dict[str, float] = field(default_factory=dict)   # name -> coverage, too little seen

    def summary(self) -> dict:
        return {"source": self.source, "frames_used": self.frames_used,
                "frames_total": self.frames_total, "ppm": round(self.ppm, 1),
                "seconds": round(self.seconds, 3), "note": self.note,
                "surfaces": {name: {"px": [s.shape[1], s.shape[0]],
                                    "coverage": round(s.coverage, 3), "seen": round(s.seen, 3)}
                             for name, s in self.surfaces.items()},
                "left_flat": {k: round(v, 3) for k, v in self.rejected.items()},
                "gains": {str(k): v for k, v in sorted(self.gains.items())}}


# ── the room's surfaces ──────────────────────────────────────────────────────────────────────────


def inward_normal(plan: RoomPlan, wall_name: str) -> np.ndarray:
    """The unit XY normal of a wall pointing INTO the room: a point just off the wall's middle on
    that side is inside the floor polygon. The room centre is not good enough — in an L-shaped
    room it lies on the outer side of the walls of the L's inner corner."""
    wall = plan.walls[wall_name]
    along = wall.along
    normal = np.array([along[1], -along[0]])
    mid = (np.asarray(wall.start) + np.asarray(wall.end)) / 2.0
    inside = plan.on_floor(np.array([mid + normal * 0.05, mid - normal * 0.05]))
    if inside[0] != inside[1]:
        return normal if inside[0] else -normal
    centre = np.mean([w.start for w in plan.walls.values()], axis=0)
    return normal if float(np.dot(normal, centre - mid)) > 0 else -normal


def opening_cuts(plan: RoomPlan, wall_name: str) -> list[tuple[float, float, float, float]]:
    """``(t0, t1, z0, z1)`` of every opening in a wall: along it from its start, up from the floor,
    clipped to floor and ceiling — the holes `build.structure` cuts (it clips t to the wall)."""
    height = plan.height
    return [(op.span[0], op.span[1], min(height, max(0.0, op.sill)),
             min(height, op.sill + op.height))
            for op in plan.openings.values() if op.wall == wall_name]


def _floor_box(plan: RoomPlan) -> tuple[np.ndarray, np.ndarray] | None:
    polys = plan.outlines()
    if not polys:
        return None
    pts = np.concatenate(polys)
    return pts.min(axis=0), pts.max(axis=0)


def texels_per_metre(plan: RoomPlan, ppm: float = PPM, *, ceiling: bool = True) -> float:
    """``ppm``, lowered (never below `MIN_PPM`) until the whole room fits in `MAX_TEXELS`: the
    cost of a frame is the texels it sees, so a big room gets coarser photos, not a longer wait."""
    area = sum(w.length for w in plan.walls.values() if w.length >= MIN_EXTENT) * plan.height
    box = _floor_box(plan)
    if box is not None:
        span = box[1] - box[0]
        area += float(span[0] * span[1]) * (1.0 + (CEILING_PPM_SCALE ** 2 if ceiling else 0.0))
    return max(MIN_PPM, min(float(ppm), math.sqrt(MAX_TEXELS / max(area, 1e-6))))


def layout_surfaces(plan: RoomPlan, objects: dict[str, RoomObject] | None = None, *,
                    ppm: float = PPM, ceiling: bool = True) -> list[Surface]:
    """Every wall, the floor and (optionally) the ceiling as texel grids, with what each draws and
    what furniture hides, at `texels_per_metre`."""
    objects = plan.objects if objects is None else objects
    fz, cz, height = plan.floor_z, plan.ceiling_z, plan.height
    walls = [w for w in plan.walls.values() if w.length >= MIN_EXTENT]
    floor_box = _floor_box(plan)
    ppm = texels_per_metre(plan, ppm, ceiling=ceiling)
    up = np.array([0.0, 0.0, 1.0])
    out: list[Surface] = []
    if height <= MIN_EXTENT:
        return out
    for wall in walls:
        L = wall.length
        along = np.r_[wall.along, 0.0]
        normal = np.r_[inward_normal(plan, wall.name), 0.0]
        cols, rows = max(4, math.ceil(L * ppm)), max(4, math.ceil(height * ppm))
        s = Surface(name=wall.name, kind="wall", origin=np.array([*wall.start, cz]),
                    u_axis=along, v_axis=-up, extent=(L, height), normal=normal,
                    shape=(rows, cols))
        t = (np.arange(cols) + 0.5) / cols * L
        z = height - (np.arange(rows) + 0.5) / rows * height      # above the floor; row 0 = top
        tt, zz = np.meshgrid(t, z)
        drawn = np.ones((rows, cols), dtype=bool)
        for t0, t1, z0, z1 in opening_cuts(plan, wall.name):
            drawn &= ~((tt > t0) & (tt < t1) & (zz > z0) & (zz < z1))
        hidden = np.zeros((rows, cols), dtype=bool)
        for o in objects.values():
            rel = o.footprint(0.03) - np.asarray(wall.start)
            ts, offs = rel @ wall.along, rel @ normal[:2]
            if offs.min() < BEHIND_GAP and offs.max() > -0.05:
                hidden |= ((tt > ts.min()) & (tt < ts.max())
                           & (zz < o.top - fz + 0.03) & (zz > o.bottom - fz - 0.03))
        s.drawn, s.hidden = drawn, hidden
        out.append(s)
    if floor_box is None:
        return out
    lo, hi = floor_box
    span = hi - lo
    for kind, z, scale, normal in (("floor", fz, 1.0, up), ("ceiling", cz, CEILING_PPM_SCALE, -up)):
        if kind == "ceiling" and not ceiling:
            continue
        cols = max(4, math.ceil(span[0] * ppm * scale))
        rows = max(4, math.ceil(span[1] * ppm * scale))
        s = Surface(name=kind, kind=kind, origin=np.array([lo[0], hi[1], z]),
                    u_axis=np.array([1.0, 0.0, 0.0]), v_axis=np.array([0.0, -1.0, 0.0]),
                    extent=(float(span[0]), float(span[1])), normal=normal, shape=(rows, cols))
        s.drawn = _raster(s, plan.outlines(), even_odd=True)
        s.hidden = _raster(s, [o.footprint(0.02) for o in objects.values()
                               if o.bottom - fz <= HUNG] if kind == "floor" else [])
        out.append(s)                          # (a wall cabinet hides none of the floor)
    return out


def _raster(surface: Surface, polygons: list[np.ndarray], *, even_odd: bool = False
            ) -> np.ndarray:
    """Which texels of a floor-plan surface lie inside ``polygons`` (XY): drawn into a mask
    image, a few milliseconds where testing every texel against every edge took a quarter of a
    second. ``even_odd`` composes them as `RoomPlan.on_floor` does (a slab inside another is a
    hole); otherwise their union."""
    from PIL import Image, ImageDraw

    rows, cols = surface.shape
    lo_x, hi_y = surface.origin[0], surface.origin[1]
    sx, sy = cols / surface.extent[0], rows / surface.extent[1]
    mask = np.zeros((rows, cols), dtype=bool)
    for poly in polygons:
        poly = np.asarray(poly, dtype=float)
        if len(poly) < 3:
            continue
        canvas = Image.new("1", (cols, rows), 0)
        ImageDraw.Draw(canvas).polygon(
            [((x - lo_x) * sx - 0.5, (hi_y - y) * sy - 0.5) for x, y in poly], fill=1)
        inside = np.asarray(canvas, dtype=bool)
        mask = (mask ^ inside) if even_odd else (mask | inside)
    return mask


# ── the texels, flattened for projection ─────────────────────────────────────────────────────────


@dataclass
class _Texels:
    """Every texel that can be photographed (drawn and not hidden), all surfaces in one array,
    sorted by culling tile so a tile is one contiguous slice."""

    points: np.ndarray          # (n, 3) float32 world
    surf: np.ndarray            # (n,) int16 surface index
    pixel: np.ndarray           # (n,) int64 flat pixel index in its surface's image
    tile_start: np.ndarray      # (m,) int64
    tile_end: np.ndarray
    tile_centre: np.ndarray     # (m, 3) float32
    tile_radius: np.ndarray     # (m,) float32
    tile_surf: np.ndarray       # (m,) int64
    surf_origin: np.ndarray     # (s, 3)
    surf_normal: np.ndarray     # (s, 3) float32


def _flatten(surfaces: list[Surface]) -> _Texels:
    points, surf, pixel = [], [], []
    starts, ends, centres, radii, owners = [], [], [], [], []
    base = 0
    for k, s in enumerate(surfaces):
        rows, cols = s.shape
        active = s.drawn & ~s.hidden
        r, c = np.nonzero(active)
        if not len(r):
            continue
        tile = (r // TILE) * (math.ceil(cols / TILE)) + (c // TILE)
        order = np.argsort(tile, kind="stable")
        r, c, tile = r[order], c[order], tile[order]
        centres_all = s.texel_centres().astype(np.float32)
        p = centres_all[r, c]
        points.append(p)
        surf.append(np.full(len(r), k, dtype=np.int16))
        pixel.append((r * cols + c).astype(np.int64))
        cut = np.flatnonzero(np.diff(tile)) + 1
        t_start = np.r_[0, cut]
        t_end = np.r_[cut, len(tile)]
        lo = np.minimum.reduceat(p, t_start, axis=0)
        hi = np.maximum.reduceat(p, t_start, axis=0)
        centres.append((lo + hi) / 2)
        radii.append(np.linalg.norm(hi - lo, axis=1) / 2 + 0.01)
        starts.append(t_start + base)
        ends.append(t_end + base)
        owners.append(np.full(len(t_start), k, dtype=np.int64))
        base += len(r)
    empty3 = np.zeros((0, 3), np.float32)
    return _Texels(
        points=np.concatenate(points) if points else empty3,
        surf=np.concatenate(surf) if surf else np.zeros(0, np.int16),
        pixel=np.concatenate(pixel) if pixel else np.zeros(0, np.int64),
        tile_start=np.concatenate(starts) if starts else np.zeros(0, np.int64),
        tile_end=np.concatenate(ends) if ends else np.zeros(0, np.int64),
        tile_centre=(np.concatenate(centres) if centres else empty3).astype(np.float32),
        tile_radius=(np.concatenate(radii) if radii else np.zeros(0)).astype(np.float32),
        tile_surf=np.concatenate(owners) if owners else np.zeros(0, np.int64),
        surf_origin=np.array([s.origin for s in surfaces], dtype=np.float64).reshape(-1, 3),
        surf_normal=np.array([s.normal for s in surfaces], dtype=np.float32).reshape(-1, 3))


# ── frames ───────────────────────────────────────────────────────────────────────────────────────


@dataclass
class _View:
    index: int                  # the capture's frame number
    rgb: np.ndarray             # (h, w, 3) uint8 at the working resolution
    depth: np.ndarray           # (dh, dw) float32 metres, 0 = no return
    nearest: np.ndarray         # the same, each pixel the nearest return around it
    rot: np.ndarray             # world -> camera, (3, 3) float32
    trans: np.ndarray           # (3,) float32
    pos: np.ndarray             # camera centre, world (3,) float32
    fx: float
    fy: float
    cx: float
    cy: float


def _load(frame) -> _View:
    """Decode one frame at the working resolution: the JPEG straight to half size in the DCT
    domain, depth in metres with low-confidence returns zeroed."""
    import json

    from PIL import Image

    from litereality_agent.pipeline.scene_init.layout.shell import ARKIT_TO_Z_UP

    meta = json.loads(frame.meta.read_text())
    K = np.asarray(meta["intrinsics"], dtype=float).reshape(3, 3)
    pose = ARKIT_TO_Z_UP @ np.asarray(meta["cameraPoseARFrame"], dtype=float).reshape(4, 4)
    with Image.open(frame.image) as im:
        full_w, full_h = im.size
        im.draft("RGB", (max(1, full_w // SHRINK), max(1, full_h // SHRINK)))
        rgb = np.asarray(im.convert("RGB"))
    with Image.open(frame.depth) as im:
        depth = np.asarray(im, dtype=np.float32) / 1000.0
    if frame.conf is not None:
        with Image.open(frame.conf) as im:
            conf = np.asarray(im)
        if conf.shape == depth.shape:
            depth = np.where(conf >= 1, depth, 0.0).astype(np.float32)
    depth = np.where(depth > MIN_DEPTH, depth, 0.0).astype(np.float32)
    h, w = rgb.shape[:2]
    # The nearest return within a few depth pixels. The depth map is 7x coarser than the photo and
    # soft at object edges, so the photo's pixels just beside a chair leg or a plant still find
    # the floor or the wall in it — and paint the leg onto them. Testing occlusion against the
    # NEAREST return around the pixel grows every foreground object by `NEAR_WINDOW` // 2 depth
    # pixels instead, and the edge stays the object's.
    k = NEAR_WINDOW // 2
    far = np.pad(np.where(depth > 0, depth, np.inf), k, mode="edge")
    dh, dw = depth.shape
    nearest = np.min([far[i:i + dh, j:j + dw] for i in range(2 * k + 1) for j in range(2 * k + 1)],
                     axis=0)
    nearest = np.where(np.isfinite(nearest), nearest, 0.0).astype(np.float32)
    sx, sy = w / full_w, h / full_h
    w2c = np.linalg.inv(pose)
    if not (np.all(np.isfinite(w2c)) and np.all(np.isfinite(K))):
        raise ValueError(f"frame {frame.index}: pose or intrinsics are not finite")
    return _View(index=frame.index, rgb=rgb, depth=depth, nearest=nearest,
                 rot=w2c[:3, :3].astype(np.float32), trans=w2c[:3, 3].astype(np.float32),
                 pos=pose[:3, 3].astype(np.float32),
                 fx=float(K[0, 0] * sx), fy=float(K[1, 1] * sy),
                 cx=float(K[0, 2] * sx), cy=float(K[1, 2] * sy))


@dataclass
class _Hits:
    """One frame's sightings: the texels it saw (indices into `_Texels`), whether each passed the
    strict test, how good a view it was, and the colour."""

    idx: np.ndarray
    strict: np.ndarray
    score: np.ndarray
    colour: np.ndarray          # (n, 3) uint8


def _visible_tiles(view: _View, tex: _Texels) -> np.ndarray:
    """Tiles that can be in view: their surface faces the camera, and their bounding sphere is in
    front of it and inside the image (or straddles the camera plane: the texel test decides)."""
    side = ((view.pos.astype(np.float64) - tex.surf_origin) * tex.surf_normal).sum(axis=1)
    tiles = np.flatnonzero(side[tex.tile_surf] > FRONT_MARGIN)
    if not len(tiles):
        return tiles
    cam = tex.tile_centre[tiles] @ view.rot.T + view.trans
    d = -cam[:, 2]
    r = tex.tile_radius[tiles]
    dz = np.maximum(d, NEAR)
    u = view.fx * cam[:, 0] / dz + view.cx
    v = -view.fy * cam[:, 1] / dz + view.cy
    m = r * max(view.fx, view.fy) / dz
    h, w = view.rgb.shape[:2]
    in_image = (u > -m) & (u < w + m) & (v > -m) & (v < h + m)
    keep = (d + r > NEAR) & (d - r < FAR) & (in_image | (d < r))
    return tiles[keep]


def _project(view: _View, tex: _Texels) -> _Hits | None:
    tiles = _visible_tiles(view, tex)
    if not len(tiles):
        return None
    starts, lens = tex.tile_start[tiles], tex.tile_end[tiles] - tex.tile_start[tiles]
    total = int(lens.sum())
    idx = np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(total)
    pts = tex.points[idx]
    cam = pts @ view.rot.T + view.trans
    d = -cam[:, 2]
    front = (d > NEAR) & (d < FAR)
    dz = np.where(front, d, 1.0)
    u = view.fx * cam[:, 0] / dz + view.cx
    v = -view.fy * cam[:, 1] / dz + view.cy
    h, w = view.rgb.shape[:2]
    ok = front & (u >= 0) & (u <= w - 2) & (v >= 0) & (v <= h - 2)
    if not ok.any():
        return None
    idx, pts, u, v, d = idx[ok], pts[ok], u[ok], v[ok], d[ok]
    to_cam = view.pos - pts
    dist = np.sqrt((to_cam * to_cam).sum(axis=1))
    facing = (to_cam * tex.surf_normal[tex.surf[idx]]).sum(axis=1) / np.maximum(dist, 1e-6)
    dh, dw = view.depth.shape
    xi = np.clip((u * (dw / w)).astype(np.int32), 0, dw - 1)
    yi = np.clip((v * (dh / h)).astype(np.int32), 0, dh - 1)
    meas = view.depth[yi, xi]
    valid = meas > 0
    near = view.nearest[yi, xi]
    occluded = (near > 0) & (d - near > OCCLUDED)
    relaxed = (facing > RELAX_FACING) & ~occluded
    if not relaxed.any():
        return None
    strict = relaxed & valid & (meas + DEPTH_TOL >= d) & (facing > STRICT_FACING)
    idx, u, v, dist, facing, strict = (a[relaxed] for a in (idx, u, v, dist, facing, strict))
    margin = np.minimum(np.minimum(u, v), np.minimum(w - 1 - u, h - 1 - v))
    score = (facing * facing * np.clip(margin / EDGE_PX, 0.0, 1.0)
             / np.maximum(dist, 0.3)).astype(np.float32)
    # bilinear sample of the photo
    x0, y0 = u.astype(np.int32), v.astype(np.int32)
    ax, ay = (u - x0)[:, None], (v - y0)[:, None]
    flat = view.rgb.reshape(-1, 3)
    at = y0 * w + x0
    c = ((flat[at] * (1 - ax) + flat[at + 1] * ax) * (1 - ay)
         + (flat[at + w] * (1 - ax) + flat[at + w + 1] * ax) * ay)
    return _Hits(idx=idx, strict=strict, score=score,
                 colour=np.clip(c + 0.5, 0, 255).astype(np.uint8))


def _spread(n: int) -> list[int]:
    """0..n-1 in an order that covers the capture evenly at every prefix (bit-reversal), so a
    deadline that stops early has still seen every side of the room."""
    if n <= 0:
        return []
    bits = max(1, (n - 1).bit_length())
    order = sorted(range(1 << bits), key=lambda i: int(f"{i:0{bits}b}"[::-1], 2))
    return [i for i in order if i < n]


# ── accumulation ─────────────────────────────────────────────────────────────────────────────────


class _Best:
    """The `TOPK` best strict sightings per texel (score, colour, frame slot), and the single best
    relaxed one. Only ever touched from one thread."""

    def __init__(self, n: int) -> None:
        self.score = np.zeros((TOPK, n), np.float32)
        self.colour = np.zeros((TOPK, n, 3), np.uint8)
        self.frame = np.full((TOPK, n), -1, np.int16)
        self.any_score = np.zeros(n, np.float32)
        self.any_colour = np.zeros((n, 3), np.uint8)
        self.any_frame = np.full(n, -1, np.int16)

    def add(self, slot: int, hits: _Hits) -> None:
        better = hits.score > self.any_score[hits.idx]
        at = hits.idx[better]
        self.any_score[at] = hits.score[better]
        self.any_colour[at] = hits.colour[better]
        self.any_frame[at] = slot
        idx = hits.idx[hits.strict]
        score, colour = hits.score[hits.strict], hits.colour[hits.strict]
        frame = np.full(len(idx), slot, np.int16)
        for j in range(TOPK):           # insertion into the sorted top-K; the displaced move down
            if not len(idx):
                break
            b = score > self.score[j, idx]
            if not b.any():
                continue
            at = idx[b]
            old = self.score[j, at], self.colour[j, at], self.frame[j, at]
            self.score[j, at], self.colour[j, at], self.frame[j, at] = score[b], colour[b], frame[b]
            score, colour, frame = score.copy(), colour.copy(), frame.copy()
            score[b], colour[b], frame[b] = old


def _gains(best: _Best, frames: int) -> np.ndarray:
    """One gain per frame and channel that brings every frame to the consensus of the frames it
    overlaps: alternate the consensus (median of the corrected sightings per texel) with each
    frame's gain (median ratio of consensus to its own sighting, in log space), normalised so the
    room as a whole keeps its exposure."""
    gains = np.ones((frames + 1, 3), np.float32)       # slot -1 (none) maps to the last row
    count = (best.score > 0).sum(axis=0)
    shared = np.flatnonzero(count >= 2)[::GAIN_SAMPLE]
    if len(shared) < 50:
        return gains[:frames]
    col = best.colour[:, shared].astype(np.float32)            # (K, n, 3)
    frm = best.frame[:, shared].astype(np.int64)               # (K, n)
    have = best.score[:, shared] > 0
    luma = col.mean(axis=2)
    usable = have & (luma > GAIN_LUMA[0]) & (luma < GAIN_LUMA[1])
    three = count[shared] >= 3
    lo, hi = math.log(GAIN_RANGE[0]), math.log(GAIN_RANGE[1])
    log_col = np.log(col + 1.0)
    for _ in range(GAIN_ROUNDS):
        corrected = log_col + np.log(gains[frm])               # (K, n, 3)
        consensus = np.where(three[:, None], _median3(corrected),
                             corrected[:2].mean(axis=0))
        delta = consensus[None] - corrected                   # what each sighting is short of
        sel = usable
        f_all, d_all = frm[sel], delta[sel]
        step = np.zeros((frames + 1, 3), np.float32)
        weight = np.zeros(frames + 1)
        order = np.argsort(f_all, kind="stable")
        f_all, d_all = f_all[order], d_all[order]
        cuts = np.flatnonzero(np.diff(f_all)) + 1
        for seg_f, seg_d in zip(np.split(f_all, cuts), np.split(d_all, cuts)):
            if len(seg_f) >= 30:
                step[seg_f[0]] = np.median(seg_d, axis=0)
                weight[seg_f[0]] = len(seg_f)
        logs = np.clip(np.log(gains) + step, lo, hi)
        if weight.sum() > 0:                                   # keep the room's overall exposure
            mean = (logs * weight[:, None]).sum(axis=0) / weight.sum()
            logs = np.clip(logs - mean, lo, hi)
        logs[frames] = 0.0
        gains = np.exp(logs).astype(np.float32)
    return gains[:frames]


def _median3(x: np.ndarray) -> np.ndarray:
    """The median along axis 0 of exactly three: a min/max network, several times quicker than
    `np.median`'s partition on half a million texels."""
    a, b, c = x[0], x[1], x[2]
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))


def push_pull(rgb: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Fill unknown texels with a smooth continuation of the known ones: average down an image
    pyramid of what is known, then come back up, keeping the known texels exactly. Never black."""
    from PIL import Image

    if known.all() or not known.any():
        return rgb
    levels = [(rgb * known[..., None], known.astype(np.float32))]
    while min(levels[-1][1].shape) > 2:
        c, k = levels[-1]
        H, W = k.shape
        H2, W2 = H // 2 * 2, W // 2 * 2
        c, k = c[:H2, :W2], k[:H2, :W2]
        levels.append((c.reshape(H2 // 2, 2, W2 // 2, 2, 3).sum((1, 3)),
                       k.reshape(H2 // 2, 2, W2 // 2, 2).sum((1, 3))))
    c, k = levels[-1]
    est = c / np.maximum(k, 1e-6)[..., None]
    if not (k > 0).all():
        est[k <= 0] = c.reshape(-1, 3).sum(0) / max(float(k.sum()), 1e-6)
    for c, k in reversed(levels[:-1]):
        H, W = k.shape
        up = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(est[..., ch]), "F")
                                  .resize((W, H), Image.Resampling.BILINEAR))
                       for ch in range(3)], axis=-1)
        a = np.clip(k, 0, 1)[..., None]
        est = a * (c / np.maximum(k, 1e-6)[..., None]) + (1 - a) * up
    return np.where(known[..., None], rgb, est)


# ── the estimate ─────────────────────────────────────────────────────────────────────────────────


def estimate(scan_dir: Path | str, plan: RoomPlan, *, objects: dict[str, RoomObject] | None = None,
             ppm: float = PPM, ceiling: bool = True, max_frames: int = MAX_FRAMES,
             budget_s: float = BUDGET_S, workers: int = WORKERS) -> SurfaceSet:
    """Photos for every wall, the floor and the ceiling of ``plan`` from the scan's frames.

    ``objects`` are the boxes that hide wall and floor (default: the plan's own). Never raises: a
    scan without frames, or one that cannot be read, gives an empty set and says why — the room
    keeps its flat colours.
    """
    started = time.perf_counter()
    result = SurfaceSet()
    try:
        _estimate(Path(scan_dir), plan, objects, result, ppm=ppm, ceiling=ceiling,
                  max_frames=max_frames, deadline=started + budget_s, workers=workers)
    except Exception as exc:  # noqa: BLE001 — photos are decoration; a bad capture must not stop the room
        result = SurfaceSet(note=f"capture unreadable ({type(exc).__name__}: {exc})")
    result.seconds = time.perf_counter() - started
    return result


def _estimate(scan_dir: Path, plan: RoomPlan, objects, result: SurfaceSet, *, ppm: float,
              ceiling: bool, max_frames: int, deadline: float, workers: int) -> None:
    frames = capture_frames(scan_dir, max_frames)
    result.frames_total = len(frames)
    if not frames:
        result.note = "no frames with both an image and a depth map"
        return
    surfaces = layout_surfaces(plan, objects, ppm=ppm, ceiling=ceiling)
    if not surfaces:
        result.note = "no surfaces to photograph"
        return
    result.ppm = texels_per_metre(plan, ppm, ceiling=ceiling)
    tex = _flatten(surfaces)
    best = _Best(len(tex.points))
    order = [frames[i] for i in _spread(len(frames))]
    used: list[int] = []
    failed: list[str] = []

    def work(frame) -> tuple[int, _Hits | None]:
        view = _load(frame)
        return view.index, _project(view, tex)

    # Frames are decoded and projected on worker threads, merged here one at a time (`_Best` has
    # a single writer). At the deadline nothing more is started, what finished is merged, and the
    # pool is left to wind down on its own: waiting for a frame already in flight would spend the
    # time the deadline exists to save.
    workers = max(1, int(workers))
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="quick-tex")

    def merge(fut: Future) -> None:
        try:
            index, hits = fut.result()
        except Exception as exc:  # noqa: BLE001 — one bad frame is skipped, not fatal
            failed.append(f"{type(exc).__name__}: {exc}")
            return
        if hits is not None:
            best.add(len(used), hits)
        used.append(index)

    pending: set[Future] = set()
    queue = list(order)
    try:
        while (queue or pending) and time.perf_counter() < deadline:
            while queue and len(pending) < workers:
                pending.add(pool.submit(work, queue.pop(0)))
            done, pending = wait(pending, timeout=max(0.0, deadline - time.perf_counter()),
                                 return_when=FIRST_COMPLETED)
            for fut in done:
                merge(fut)
        for fut in [f for f in pending if f.done()]:   # finished right on the deadline
            pending.discard(fut)
            merge(fut)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if queue or pending:
        result.note = f"time budget reached after {len(used)} of {len(frames)} frames"
    result.frames_used = len(used)
    if not used:
        result.note = result.note or (f"no frame could be read ({failed[0]})" if failed
                                      else "no frame read")
        return
    gains = _gains(best, len(used))
    result.gains = {used[s]: [round(float(g), 3) for g in gains[s]] for s in range(len(used))}
    _paint(surfaces, tex, best, gains, result)
    result.source = "frames" if result.surfaces else "none"
    if failed and not result.note:
        result.note = f"{len(failed)} frame(s) unreadable, skipped"
    if not result.surfaces and not result.note:
        result.note = "no surface was seen well enough"


def _paint(surfaces: list[Surface], tex: _Texels, best: _Best, gains: np.ndarray,
           result: SurfaceSet) -> None:
    """Each texel's colour from its sightings, exposure-balanced; each surface's image filled."""
    g = np.vstack([gains, np.ones((1, 3), np.float32)])        # slot -1 -> gain 1
    count = (best.score > 0).sum(axis=0)
    corrected = best.colour.astype(np.float32) * g[best.frame.astype(np.int64)]
    colour = corrected[0].copy()                               # one or two views: the best one
    three = count >= 3
    colour[three] = _median3(corrected[:, three])
    loose = (count == 0) & (best.any_score > 0)
    colour[loose] = best.any_colour[loose].astype(np.float32) * g[best.any_frame[loose]]
    seen_any = (count > 0) | loose
    for k, s in enumerate(surfaces):
        rows, cols = s.shape
        mine = tex.surf == k
        img = np.zeros((rows * cols, 3), np.float32)
        known = np.zeros(rows * cols, dtype=bool)
        strict = np.zeros(rows * cols, dtype=bool)
        pix = tex.pixel[mine]
        img[pix] = colour[mine]
        known[pix] = seen_any[mine]
        strict[pix] = count[mine] > 0
        # Coverage of what COULD be seen: the strip behind a wardrobe is not a failure to look.
        open_ = (s.drawn & ~s.hidden).reshape(-1)
        n_open = int(open_.sum())
        s.coverage = float((known & open_).sum() / max(n_open, 1))
        s.seen = float((strict & open_).sum() / max(n_open, 1))
        if s.coverage < MIN_COVERAGE or n_open == 0:
            result.rejected[s.name] = s.coverage
            continue
        filled = push_pull(img.reshape(rows, cols, 3), known.reshape(rows, cols))
        s.image = np.clip(filled + 0.5, 0, 255).astype(np.uint8)
        result.surfaces[s.name] = s


# ── files ────────────────────────────────────────────────────────────────────────────────────────


def texture_file(surface: Surface) -> str:
    return f"{surface.name}.png"


def save(surface: Surface, folder: Path) -> Path:
    """Write the surface's image as PNG (MuJoCo reads PNG, not JPEG), atomically: a room call
    watching the folder never loads half a texture."""
    import os

    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / texture_file(surface)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(surface.image).save(tmp, format="PNG", compress_level=3)
    os.replace(tmp, path)
    return path


__all__ = ["Surface", "SurfaceSet", "estimate", "inward_normal", "layout_surfaces",
           "opening_cuts", "push_pull", "save", "texels_per_metre", "texture_file"]
