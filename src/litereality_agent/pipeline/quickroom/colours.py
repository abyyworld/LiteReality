"""colours.py — what colour each thing in the scanned room actually is, from the capture itself.

The quick room is boxes, and a room of grey boxes is not recognisably anyone's room; the same boxes
in the right colours are — a red sofa, a pale wooden table, blue chairs. The capture already holds
the answer: every frame has an RGB image, a LiDAR depth map aligned with it, and the camera pose
that put both in the room. Unproject the depth pixels into the room, and the ones that land INSIDE
an object's box are that object's surface, seen from wherever the camera was. That is occlusion-
aware for free (a chair in front of the table lands in the chair's box, not the table's) and needs
no detector, no segmentation model and no point cloud — so it costs well under a second.

Only a SAMPLE of frames is read (evenly spread over the capture, so every side of the room is
seen), each image decoded at reduced size (JPEG draft mode) straight to the depth map's
resolution. A per-channel median over all the pixels an object collected is the colour; it shrugs
off the plant on the table and the highlight on the fridge.

Everything degrades to a category colour: a scan without depth or images, a box no frame saw
well, or a run with colours turned off still gets a red-brown door and a white fridge rather than
nothing. `Palette.source` says which it was.

Frames, as the LiteReality Scanner writes them (`frame_NNNNN.json`, `.jpg`, `depth_NNNNN.png`
in millimetres, optional `conf_NNNNN.png` 0..2): ``cameraPoseARFrame`` is camera -> ARKit world,
row-major with the translation in the last column; the camera looks down its -Z with +Y up, so
pixel (u, v) at depth d is ``((u - cx) d / fx, -(v - cy) d / fy, -d)``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from litereality_agent.pipeline.quickroom.roomplan import RoomPlan
from litereality_agent.pipeline.scene_init.layout.shell import ARKIT_TO_Z_UP

RGB = tuple[float, float, float]
# (tag, centre, half-extents) of one sampled box in an object's own frame
Region = tuple[str, np.ndarray, np.ndarray]

# Fallbacks when the capture cannot say: sRGB, what each thing usually looks like. Matched by
# substring on the category, first hit wins, so "sink_storage" reads as a sink unit.
CATEGORY_RGB: dict[str, RGB] = {
    "chair": (0.30, 0.32, 0.36), "stool": (0.45, 0.35, 0.26),
    "table": (0.76, 0.64, 0.48), "desk": (0.76, 0.64, 0.48),
    "sofa": (0.45, 0.47, 0.52), "bed": (0.86, 0.85, 0.82),
    "sink": (0.80, 0.80, 0.78), "dishwasher": (0.86, 0.87, 0.88),
    "refrigerator": (0.90, 0.90, 0.90), "oven": (0.25, 0.25, 0.27), "stove": (0.25, 0.25, 0.27),
    "washer": (0.92, 0.92, 0.92), "television": (0.07, 0.07, 0.08),
    "fireplace": (0.55, 0.45, 0.40), "stairs": (0.62, 0.52, 0.42),
    "bathtub": (0.95, 0.95, 0.95), "toilet": (0.95, 0.95, 0.95),
    "storage": (0.88, 0.87, 0.84), "cabinet": (0.88, 0.87, 0.84),
}
DEFAULT_RGB: RGB = (0.62, 0.62, 0.62)
WALL_RGB: RGB = (0.90, 0.88, 0.84)
FLOOR_RGB: RGB = (0.56, 0.54, 0.51)
DOOR_RGB: RGB = (0.50, 0.34, 0.23)

MAX_FRAMES = 24           # evenly spread over the capture; ~20 ms each
MIN_PIXELS = 40           # fewer depth pixels than this is a glimpse, not a measurement
SURFACE_BAND = 0.04       # m either side of a wall's plane that still counts as the wall
FLOOR_BAND = 0.03         # m above/below the floor
NEAR, FAR = 0.15, 6.0     # m: LiDAR depth outside this is noise
# m a sampled region reaches past the box it samples: a surface pixel lies ON the box's face, and
# a RoomPlan box is only good to a few centimetres (see `build.sample_regions`).
REGION_PAD = 0.03


@dataclass
class Swatch:
    """One object's colour, plus its top surface's when that was seen on its own (a wooden table
    top over grey legs, a worktop on white cabinets)."""

    rgb: RGB
    top: RGB | None = None
    pixels: int = 0
    source: str = "category"        # "frames" | "category"


@dataclass
class Palette:
    objects: dict[str, Swatch] = field(default_factory=dict)
    wall: RGB = WALL_RGB
    floor: RGB = FLOOR_RGB
    doors: dict[str, RGB] = field(default_factory=dict)
    source: str = "category"
    frames_used: int = 0
    seconds: float = 0.0
    # The frames each object was seen best in (most depth pixels inside its box): the starting
    # point for the object stage's per-object crops.
    views: dict[str, list[int]] = field(default_factory=dict)
    note: str = ""

    def swatch(self, handle: str, category: str) -> Swatch:
        return self.objects.get(handle) or Swatch(category_rgb(category))

    def summary(self) -> dict:
        return {"source": self.source, "frames_used": self.frames_used,
                "seconds": round(self.seconds, 3), "note": self.note,
                "wall": _hex(self.wall), "floor": _hex(self.floor),
                "doors": {k: _hex(v) for k, v in self.doors.items()},
                "objects": {k: {"rgb": _hex(s.rgb), "top": _hex(s.top) if s.top else None,
                                "pixels": s.pixels, "source": s.source}
                            for k, s in self.objects.items()},
                "views": self.views}


def category_rgb(category: str) -> RGB:
    key = (category or "").lower()
    for token, rgb in CATEGORY_RGB.items():
        if token in key:
            return rgb
    return DEFAULT_RGB


def _hex(rgb: RGB | None) -> str | None:
    if rgb is None:
        return None
    return "#" + "".join(f"{int(round(float(np.clip(c, 0, 1)) * 255)):02x}" for c in rgb)


# ── frames ───────────────────────────────────────────────────────────────────────────────────────


@dataclass
class _Frame:
    index: int
    meta: Path
    image: Path
    depth: Path
    conf: Path | None


def capture_frames(scan_dir: Path, limit: int = MAX_FRAMES) -> list[_Frame]:
    """Frames with pose, image AND depth, evenly subsampled to ``limit``. A capture often keeps
    the pose of every frame but the image or depth of only some, so all three are required."""
    found = []
    for meta in sorted(Path(scan_dir).glob("frame_*.json")):
        stem = meta.stem[len("frame_"):]
        image, depth = meta.with_suffix(".jpg"), meta.with_name(f"depth_{stem}.png")
        if image.is_file() and depth.is_file():
            conf = meta.with_name(f"conf_{stem}.png")
            try:
                index = int(stem)
            except ValueError:
                continue
            found.append(_Frame(index, meta, image, depth, conf if conf.is_file() else None))
    if len(found) > limit:
        picks = np.linspace(0, len(found) - 1, limit).round().astype(int)
        found = [found[i] for i in sorted(set(picks.tolist()))]
    return found


def _read_frame(frame: _Frame) -> tuple[np.ndarray, np.ndarray] | None:
    """(world points (n, 3) Z-up, sRGB colours (n, 3) uint8) for one frame's valid depth pixels."""
    from PIL import Image

    meta = json.loads(frame.meta.read_text())
    K = np.asarray(meta["intrinsics"], dtype=float).reshape(3, 3)
    pose = np.asarray(meta["cameraPoseARFrame"], dtype=float).reshape(4, 4)
    with Image.open(frame.depth) as im:
        depth = np.asarray(im, dtype=np.float32) / 1000.0
    h, w = depth.shape[:2]
    ok = (depth > NEAR) & (depth < FAR)
    if frame.conf is not None:
        with Image.open(frame.conf) as im:
            conf = np.asarray(im)
        if conf.shape[:2] == depth.shape[:2]:
            ok &= conf >= 1                       # ARKit confidence: 0 low, 1 medium, 2 high
    with Image.open(frame.image) as im:
        full_w, full_h = im.size
        im.draft("RGB", (w * 2, h * 2))           # DCT-domain downscale: decode at 1/4 or 1/8
        rgb = np.asarray(im.convert("RGB").resize((w, h), Image.Resampling.BOX))
    sx, sy = w / full_w, h / full_h
    fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
    ys, xs = np.nonzero(ok)
    if len(xs) == 0:
        return None
    d = depth[ys, xs]
    cam = np.stack([(xs + 0.5 - cx) / fx * d, -(ys + 0.5 - cy) / fy * d, -d,
                    np.ones_like(d)])
    world = ((ARKIT_TO_Z_UP @ pose) @ cam)[:3].T
    if not np.all(np.isfinite(world)):
        keep = np.all(np.isfinite(world), axis=1)
        world, ys, xs = world[keep], ys[keep], xs[keep]
    return world, rgb[ys, xs]


# ── the estimate ─────────────────────────────────────────────────────────────────────────────────


def estimate(scan_dir: Path | str, plan: RoomPlan, *,
             regions: dict[str, list[Region]] | None = None, max_frames: int = MAX_FRAMES,
             budget_s: float = 5.0) -> Palette:
    """Colours for every object of ``plan`` (by its handle), the walls, the floor and each door.

    ``regions`` says, per object, which parts of its box to sample and what each is called: a
    list of ``(tag, centre, half)`` boxes in the object's own frame (origin at the box centre, +X
    along its yaw). Tag ``"top"`` fills `Swatch.top`, anything else `Swatch.rgb`. Without it the
    whole box is the object. A pixel inside several regions — the table top a tucked-in chair's box
    reaches up into — belongs to the SMALLEST, which is the most specific claim on it.

    Never raises: a capture that cannot be read gives the category palette and says why.
    """
    started = time.perf_counter()
    palette = Palette(objects={h: Swatch(category_rgb(o.category)) for h, o in plan.objects.items()})
    try:
        frames = capture_frames(Path(scan_dir), max_frames)
        if not frames:
            palette.note = "no frames with both an image and a depth map"
            palette.seconds = time.perf_counter() - started
            return palette
        _measure(plan, frames, palette, started + budget_s, regions or {})
    except Exception as exc:  # noqa: BLE001 — colour is decoration; a bad capture must not stop the room
        palette = Palette(objects={h: Swatch(category_rgb(o.category))
                                   for h, o in plan.objects.items()},
                          note=f"capture unreadable ({type(exc).__name__}: {exc})")
    palette.seconds = time.perf_counter() - started
    return palette


def _measure(plan: RoomPlan, frames: list[_Frame], palette: Palette, deadline: float,
             regions: dict[str, list[Region]]) -> None:
    objects = list(plan.objects.values())
    fz, cz = plan.floor_z, plan.ceiling_z
    # Each object's box as (centre, rotation^T, half-extents) for a vectorised inside test, and
    # its sampling regions flattened into one list so ownership is decided across all objects.
    boxes, flat = [], []
    for k, o in enumerate(objects):
        a = math.radians(o.yaw)
        rot_t = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
        half = np.asarray(o.size, dtype=float) / 2.0
        boxes.append((np.asarray(o.center, dtype=float), rot_t, half))
        mine = regions.get(o.handle) or [("body", np.zeros(3), half + REGION_PAD)]
        for tag, centre, rhalf in mine:
            flat.append((k, "top" if tag == "top" else "body", np.asarray(centre, dtype=float),
                         np.asarray(rhalf, dtype=float)))
    has_floor = bool(plan.outlines())
    walls = list(plan.walls.values())
    doors = [op for op in plan.openings.values() if op.kind == "door"]

    samples: dict[tuple[int, str], list[np.ndarray]] = {}
    seen: dict[int, list[tuple[int, int]]] = {k: [] for k in range(len(objects))}
    wall_px: list[np.ndarray] = []
    floor_px: list[np.ndarray] = []
    door_px: dict[str, list[np.ndarray]] = {op.handle: [] for op in doors}

    used = 0
    for frame in frames:
        if time.perf_counter() > deadline:
            palette.note = f"time budget reached after {used} frames"
            break
        read = _read_frame(frame)
        if read is None:
            continue
        pts, rgb = read
        used += 1
        z = pts[:, 2]
        # A wall's own surface is never an object's, whatever box reaches into it: a cabinet
        # standing against a wall has its back on the plaster.
        on_wall = np.zeros(len(pts), dtype=bool)
        for wall in walls:
            rel = pts[:, :2] - np.asarray(wall.start)
            t = rel @ wall.along
            off = rel @ np.array([wall.along[1], -wall.along[0]])
            on_wall |= (np.abs(off) < 0.02) & (t > -0.02) & (t < wall.length + 0.02)
        claimed = np.zeros(len(pts), dtype=bool)      # inside some object's (grown) box
        owner = np.full(len(pts), -1, dtype=int)
        best = np.full(len(pts), np.inf)
        local_of: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for k, (c, rot_t, half) in enumerate(boxes):
            near = (np.all(np.abs((pts[:, :2] - c[:2]) @ rot_t.T) <= half[:2] + 0.05, axis=1)
                    & (z > c[2] - half[2] - 0.05) & (z < c[2] + half[2] + 0.05))
            claimed |= near
            idx = np.nonzero(near & ~on_wall & (z > fz + FLOOR_BAND))[0]
            if len(idx):
                xy = (pts[idx, :2] - c[:2]) @ rot_t.T
                local_of[k] = (idx, np.c_[xy, z[idx] - c[2]])
        for r, (k, _tag, centre, rhalf) in enumerate(flat):
            if k not in local_of:
                continue
            idx, local = local_of[k]
            inside = np.all(np.abs(local - centre) <= rhalf, axis=1)
            volume = float(np.prod(rhalf))
            hit = idx[inside]
            better = hit[volume < best[hit]]
            best[better] = volume
            owner[better] = r
        owned = np.nonzero(owner >= 0)[0]
        if len(owned):
            order = owned[np.argsort(owner[owned], kind="stable")]
            counts = np.bincount(owner[owned], minlength=len(flat))
            per_object: dict[int, int] = {}
            cursor = 0
            for r, n in enumerate(counts.tolist()):
                if n:
                    k, tag = flat[r][0], flat[r][1]
                    samples.setdefault((k, tag), []).append(rgb[order[cursor:cursor + n]])
                    per_object[k] = per_object.get(k, 0) + n
                    cursor += n
            for k, n in per_object.items():
                seen[k].append((n, frame.index))
        free = ~claimed
        # Walls: on a wall's plane, inside its length, clear of floor, ceiling and openings.
        for wall in walls:
            s = np.asarray(wall.start)
            along = wall.along
            normal = np.array([along[1], -along[0]])
            rel = pts[:, :2] - s
            t, off = rel @ along, rel @ normal
            on = (free & (np.abs(off) < SURFACE_BAND) & (t > 0.05) & (t < wall.length - 0.05)
                  & (z > fz + 0.25) & (z < cz - 0.15))
            for op in plan.openings.values():
                if op.wall == wall.name:
                    t0, t1 = op.span
                    on &= ~((t > t0 - 0.05) & (t < t1 + 0.05) & (z > fz + op.sill - 0.05)
                            & (z < fz + op.sill + op.height + 0.05))
            if on.any():
                wall_px.append(rgb[on])
            for op in doors:
                if op.wall != wall.name:
                    continue
                t0, t1 = op.span
                leaf = ((np.abs(off) < 0.05) & (t > t0 + 0.05) & (t < t1 - 0.05)
                        & (z > fz + 0.1) & (z < fz + op.height - 0.1))
                if leaf.any():
                    door_px[op.handle].append(rgb[leaf])
        on_floor = free & (np.abs(z - fz) < FLOOR_BAND)
        if on_floor.any() and has_floor:
            idx = np.nonzero(on_floor)[0]
            idx = idx[plan.on_floor(pts[idx, :2])]
            if len(idx):
                floor_px.append(rgb[idx])

    palette.frames_used = used
    if not used:
        palette.note = palette.note or "no usable depth pixels"
        return
    palette.source = "frames"
    for k, o in enumerate(objects):
        body = samples.get((k, "body"), [])
        top = samples.get((k, "top"), [])
        n_body, n_top = sum(len(r) for r in body), sum(len(r) for r in top)
        if n_body >= MIN_PIXELS or n_top >= MIN_PIXELS:
            rgb_body = _median(body) if n_body >= MIN_PIXELS else _median(top)
            palette.objects[o.handle] = Swatch(
                rgb=rgb_body, top=_median(top) if n_top >= MIN_PIXELS else None,
                pixels=n_body + n_top, source="frames")
        ranked = sorted(seen[k], reverse=True)[:3]
        if ranked:
            palette.views[o.handle] = [index for _n, index in ranked]
    if sum(len(r) for r in wall_px) >= MIN_PIXELS:
        palette.wall = _median(wall_px)
    if sum(len(r) for r in floor_px) >= MIN_PIXELS:
        palette.floor = _median(floor_px)
    for handle, rows in door_px.items():
        if sum(len(r) for r in rows) >= MIN_PIXELS:
            palette.doors[handle] = _median(rows)


def _median(rows: list[np.ndarray]) -> RGB:
    stacked = np.concatenate(rows, axis=0)
    med = np.median(stacked, axis=0) / 255.0
    return tuple(round(float(v), 4) for v in med)
