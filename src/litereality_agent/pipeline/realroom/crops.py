"""crops.py — each quick-room object as the photo TRELLIS is given: the capture's best view of
it, cut out with a real mask.

TRELLIS turns ONE image of ONE object into a mesh, and what it is given decides what comes back:
the object whole and large in the frame, upright, and cut out. A dark chair on a dark floor cut
out by "near-black means background" (the function's fallback) comes back with holes, so every
crop carries its own alpha mask, and the mask comes from geometry, not from a matting model:

* **Which frame.** Every capture frame is scored from its pose alone — project a 6x6x6 grid of
  points inside the object's box: how much of it is in the picture, how large, how central (the
  pipeline's ``object_view_quality``) — and the best ``SHORTLIST`` are then looked at in depth:
  every LiDAR pixel of the box's projection is put back in the room, and the ones inside the box
  are the object. That is occlusion-aware for free (a chair in front of the table lands in the
  chair's box) and the share of the box's projection the object fills says how unoccluded it is.
* **Whose pixel.** Boxes overlap — a chair tucked in under a table reaches up into the top — so a
  point inside another object's template PART and none of this one's belongs to the other; a
  wall's plaster and the floor belong to neither.
* **The mask.** The depth mask is 256 x 192; GrabCut (OpenCV, no learned model) snaps its edges to
  the photo at ~640 px, seeded with the eroded depth mask as certain object and everything well
  outside it as certain background, and is thrown away when it wanders off.
* **The picture.** Cropped from the full-resolution photo with a margin, turned so the room's up
  is the picture's up, and scaled so the object is 512-1024 px across: TRELLIS crops to the mask
  and conditions at 518/1024 px itself.

The views (pose, intrinsics, depth mask) stay with the crop: `fit.choose_yaw` reuses them to check
which way round a generated mesh goes.

Frames are the LiteReality Scanner's (`frame_NNNNN.json/.jpg`, `depth_NNNNN.png` in mm at 256 x
192, optional `conf_NNNNN.png`); the ARKit camera looks down its -Z with +Y up, and
``cameraPoseARFrame`` is camera -> ARKit world, row-major, put into the room's Z-up frame with
``ARKIT_TO_Z_UP`` exactly as `quickroom.colours` does.
"""

from __future__ import annotations

import io
import json
import math
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from litereality_agent.pipeline.realroom.fit import View
from litereality_agent.pipeline.scene_init.layout.shell import ARKIT_TO_Z_UP

SHORTLIST = 8          # frames per object looked at in depth, after the pose-only ranking
VIEWS = 3              # depth-masked views kept: the best is the crop, all of them check the yaw
MIN_IN_IMAGE = 0.25    # share of the box's grid that must project into a frame to count
MASK_MARGIN = 0.04     # m the box is grown by for its mask: RoomPlan boxes are good to a few cm
OWN_PAD = 0.05         # m a point may lie outside this object's own template parts and stay its own
THEIR_PAD = 0.03       # m around another object's parts that is still that object
# What stands ON another object (the plant on the table, the kettle on the counter) is that object's:
# within its footprint, from UNDER_TOP below its top to ON_TOP above it. The band reaches well under
# the top because a RoomPlan box and the LiDAR disagree by up to ~8 cm in height (tea_room's table
# top is seen 8 cm under its box's): a tucked chair's crop must lose the table top all the same.
ON_TOP = 0.40
UNDER_TOP = 0.12
# Bodies small enough that ALL of their box is theirs — a chair tucked under a table is chair
# wherever its crude template has no part (a curved back, a cushion).
WHOLE_BOX_KINDS = {"chair", "stool"}
FLOOR_BAND = 0.035     # m above a view's floor that is still floor (carpet depth is +-3 cm)
FLOOR_SEARCH = 0.12    # m either side of RoomPlan's floor where a view's own floor is looked for
FLOOR_INLIER = 0.025   # m from the first fit that counts as floor for the second
FLOOR_MIN_PX = 40      # floor pixels needed to trust that estimate
FLOOR_MAX_TILT = 10.0  # deg: steeper than this is not the floor
LOW_BAND = 0.12        # m above a view's floor where depth alone cannot tell object from floor
WALL_BAND = 0.025      # m either side of a wall's plane that is plaster
NEAR, FAR = 0.15, 6.0  # m of trustworthy LiDAR depth
MIN_MASK_PX = 40       # depth pixels: fewer is a glimpse, not a view
# A best view scoring under this is a sliver of the object (tea_room's tucked-in chair, its table
# seen only from above, 0.2-0.4; every whole view on tea_room and office_room scores 0.54+). A
# mesh made from a sliver is worse than the template: such an object borrows a twin's mesh or
# keeps its template (`build.generate_room`).
MIN_SCORE = 0.45
CROP_PAD = 0.08        # of the object's larger side, around its mask
MIN_SIDE, MAX_SIDE = 512, 1024
GRABCUT_SIDE = 360
GRABCUT_ITERATIONS = 3
GRABCUT_SEED = 20261021
SKIP_KINDS = {"door_leaf"}
SKIP_CATEGORIES = {"door", "window", "opening"}
_G = np.linspace(-0.5, 0.5, 6)
GRID = np.array([[x, y, z] for x in _G for y in _G for z in _G])


# ── what there is ────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Box:
    """One quick-room body as an oriented box: world centre, yaw (radians), half-extents, and the
    template's parts in the box frame (centre, half) — what the body is made of, for ownership."""

    handle: str
    category: str
    kind: str
    joint: str
    centre: np.ndarray
    yaw: float
    half: np.ndarray
    parts: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
    body_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    local_centre: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def rotation(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_local(self, world: np.ndarray) -> np.ndarray:
        """World points in the box frame (origin at the box centre)."""
        return (np.asarray(world, dtype=float) - self.centre) @ self.rotation

    def to_world(self, local: np.ndarray) -> np.ndarray:
        return np.asarray(local, dtype=float) @ self.rotation.T + self.centre

    @property
    def bottom(self) -> float:
        return float(self.centre[2] - self.half[2])

    def generated(self) -> bool:
        """Whether this body is one the objects stage replaces (furniture, not door leaves)."""
        return self.kind not in SKIP_KINDS and self.category not in SKIP_CATEGORIES


def _yaw_of(quat: str | None) -> float:
    if not quat:
        return 0.0
    w, _x, _y, z = (float(v) for v in quat.split())
    return 2.0 * math.atan2(z, w)


def _vec(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([float(v) for v in text.split()]) if text else np.array(default, float)


def boxes_from_scene(scene_xml: Path | str, report: dict | None = None) -> dict[str, Box]:
    """The quick room's bodies as boxes, read from its scene.xml (and quick.json for category,
    template and joint) — the room as it was BUILT, layout repair and separation included, so a
    generated mesh replaces exactly what people are standing next to."""
    root = ET.parse(scene_xml).getroot()
    bodies = (report or {}).get("bodies") or {}
    boxes: dict[str, Box] = {}
    world = root.find("worldbody")
    for body in (world.findall("body") if world is not None else []):
        name = body.get("name")
        if not name:
            continue
        parts = []
        for geom in body.findall("geom"):
            if geom.get("type", "sphere") != "box" or geom.get("group") == "2":
                continue
            half = _vec(geom.get("size"))
            if half.shape != (3,):
                continue
            parts.append((_vec(geom.get("pos")), half))
        if not parts:
            continue
        lo = np.min([c - h for c, h in parts], axis=0)
        hi = np.max([c + h for c, h in parts], axis=0)
        local_centre, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        pos, yaw = _vec(body.get("pos")), _yaw_of(body.get("quat"))
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        info = bodies.get(name) or {}
        if body.find("freejoint") is not None:
            joint = "free"
        elif any(j.get("type") == "hinge" for j in body.findall("joint")):
            joint = "hinge"
        else:
            joint = "static"
        boxes[name] = Box(
            handle=name,
            category=str(info.get("category") or "".join(ch for ch in name
                                                         if not ch.isdigit()).lower()),
            kind=str(info.get("template") or ""), joint=str(info.get("joint") or joint),
            centre=pos + rot @ local_centre, yaw=yaw, half=half,
            parts=tuple((pc - local_centre, ph) for pc, ph in parts),
            body_pos=pos, local_centre=local_centre)
    return boxes


@dataclass(frozen=True)
class Frame:
    index: int
    K: np.ndarray             # at the photo's resolution
    c2w: np.ndarray           # camera -> room (Z-up)
    image: Path
    depth: Path
    conf: Path | None
    size: tuple[int, int]     # photo (width, height)

    @property
    def w2c(self) -> np.ndarray:
        return np.linalg.inv(self.c2w)


def load_frames(scan_dir: Path | str) -> list[Frame]:
    """Every frame with a pose, a photo and a depth map. Unreadable ones are skipped."""
    from PIL import Image

    frames = []
    for meta in sorted(Path(scan_dir).glob("frame_*.json")):
        stem = meta.stem[len("frame_"):]
        image, depth = meta.with_suffix(".jpg"), meta.with_name(f"depth_{stem}.png")
        if not (image.is_file() and depth.is_file()):
            continue
        try:
            data = json.loads(meta.read_text())
            K = np.asarray(data["intrinsics"], dtype=float).reshape(3, 3)
            pose = np.asarray(data["cameraPoseARFrame"], dtype=float).reshape(4, 4)
            with Image.open(image) as im:
                size = im.size
            index = int(data.get("frame_index", int(stem)))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        conf = meta.with_name(f"conf_{stem}.png")
        frames.append(Frame(index, K, ARKIT_TO_Z_UP @ pose, image, depth,
                            conf if conf.is_file() else None, size))
    return frames


# ── choosing views ───────────────────────────────────────────────────────────────────────────────


def _project(K: np.ndarray, w2c: np.ndarray, world: np.ndarray):
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    z = -cam[:, 2]
    front = z > NEAR
    safe = np.where(front, z, 1.0)
    u = np.where(front, K[0, 0] * cam[:, 0] / safe + K[0, 2], np.nan)
    v = np.where(front, -K[1, 1] * cam[:, 1] / safe + K[1, 2], np.nan)
    return u, v, front


def quality(bbox, n_visible: float, n_max: float, width: float, height: float) -> float:
    """The pipeline's ``object_view_quality``: large, central, fully visible."""
    x0, y0, x1, y1 = bbox
    frac = max(0.0, x1 - x0) * max(0.0, y1 - y0) / float(width * height)
    size = min(1.0, (frac ** 0.5) / 0.35)
    cx, cy = (x0 + x1) / 2.0 / width * 100.0, (y0 + y1) / 2.0 / height * 100.0
    centred = max(0.0, 1.0 - float(np.hypot(cx - 50.0, cy - 50.0)) / 70.71)
    visible = min(1.0, (n_visible / max(n_max, 1)) / 0.5)
    return size * (0.85 + 0.15 * centred) * visible


def shortlist(frames: list[Frame], box: Box, top: int = SHORTLIST) -> list[tuple[float, Frame]]:
    """The frames that see most of the box, largest and most central, by pose alone."""
    grid = box.to_world(GRID * (2.0 * box.half))
    ranked = []
    for frame in frames:
        w, h = frame.size
        u, v, front = _project(frame.K, frame.w2c, grid)
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if inside.sum() < MIN_IN_IMAGE * len(grid):
            continue
        uu, vv = u[front], v[front]
        bbox = (max(0.0, uu.min()), max(0.0, vv.min()), min(w, uu.max()), min(h, vv.max()))
        ranked.append((quality(bbox, inside.sum(), len(grid), w, h), frame))
    ranked.sort(key=lambda item: -item[0])
    return ranked[:top]


@dataclass
class _Depth:
    depth: np.ndarray
    conf: np.ndarray | None
    K: np.ndarray             # at the depth map's resolution


def _read_depth(frame: Frame) -> _Depth:
    from PIL import Image

    with Image.open(frame.depth) as im:
        depth = np.asarray(im, dtype=np.float32) / 1000.0
    conf = None
    if frame.conf is not None:
        with Image.open(frame.conf) as im:
            c = np.asarray(im)
        conf = c if c.shape == depth.shape else None
    h, w = depth.shape[:2]
    sx, sy = w / frame.size[0], h / frame.size[1]
    K = frame.K.copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return _Depth(depth, conf, K)


def _inside(local: np.ndarray, centre: np.ndarray, half: np.ndarray) -> np.ndarray:
    return np.all(np.abs(local - centre) <= half, axis=1)


def object_mask(frame: Frame, depth: _Depth, box: Box, others: list[Box], floor_z: float,
                walls: list[tuple[np.ndarray, np.ndarray]]) -> dict | None:
    """The object's pixels in one frame's depth map, and how good a view that is."""
    h, w = depth.depth.shape
    corners = box.to_world(np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                                     for z in (-1, 1)], float) * box.half)
    u, v, front = _project(depth.K, frame.w2c, corners)
    if front.sum() < 8:
        return None
    gu, gv, gf = _project(depth.K, frame.w2c, box.to_world(GRID * (2.0 * box.half)))
    complete = float((gf & (gu >= 0) & (gu < w) & (gv >= 0) & (gv < h)).mean())
    x0, x1 = int(max(0, math.floor(np.nanmin(u)))), int(min(w, math.ceil(np.nanmax(u))))
    y0, y1 = int(max(0, math.floor(np.nanmin(v)))), int(min(h, math.ceil(np.nanmax(v))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    d = depth.depth[y0:y1, x0:x1]
    ok = (d > NEAR) & (d < FAR)
    if depth.conf is not None:
        ok &= depth.conf[y0:y1, x0:x1] >= 1
    ys, xs = np.nonzero(ok)
    if not len(xs):
        return None
    dd = d[ys, xs]
    K = depth.K
    cam = np.stack([((xs + x0) + 0.5 - K[0, 2]) / K[0, 0] * dd,
                    -(((ys + y0) + 0.5 - K[1, 2]) / K[1, 1] * dd), -dd], axis=1)
    world = cam @ frame.c2w[:3, :3].T + frame.c2w[:3, 3]
    local = box.to_local(world)
    keep = _inside(local, np.zeros(3), box.half + MASK_MARGIN)
    height = world[:, 2] - floor_under(world, ~keep, floor_z, others)
    keep &= height > FLOOR_BAND
    for start, end in walls:
        along = end - start
        length = float(np.linalg.norm(along))
        if length < 1e-6:
            continue
        along = along / length
        rel = world[:, :2] - start
        t, off = rel @ along, rel @ np.array([along[1], -along[0]])
        keep &= ~((np.abs(off) < WALL_BAND) & (t > -WALL_BAND) & (t < length + WALL_BAND))
    if keep.any() and others:
        idx = np.nonzero(keep)[0]
        own = np.zeros(len(idx), bool)
        for centre, half in box.parts:
            own |= _inside(local[idx], centre, half + OWN_PAD)
        theirs = np.zeros(len(idx), bool)
        for other in others:
            ol = other.to_local(world[idx])
            if other.kind in WHOLE_BOX_KINDS:
                theirs |= _inside(ol, np.zeros(3), other.half + THEIR_PAD)
                continue
            footprint = np.all(np.abs(ol[:, :2]) <= other.half[:2], axis=1)
            if not footprint.any():
                continue
            above = ol[:, 2] - other.half[2]
            theirs |= footprint & (above > -UNDER_TOP) & (above < ON_TOP)
            for centre, half in other.parts:
                theirs |= _inside(ol, centre, half + THEIR_PAD)
        keep[idx[theirs & ~own]] = False
    mask = np.zeros((h, w), bool)
    mask[ys[keep] + y0, xs[keep] + x0] = True
    # Just above the floor, depth cannot tell the object from the floor (LiDAR depth is smoothed
    # over edges: office_room's carpet rises 4-10 cm into a chair's box). Those pixels stay in the
    # mask but are only PROBABLY the object: the photo decides (`refine`).
    low = np.zeros((h, w), bool)
    sel = keep & (height < LOW_BAND)
    low[ys[sel] + y0, xs[sel] + x0] = True
    n_obj = int(mask.sum())
    if n_obj < MIN_MASK_PX:
        return None
    return {"frame": frame, "mask": mask, "low": low, "n_obj": n_obj,
            "n_box": (x1 - x0) * (y1 - y0), "complete": complete, "K": K}


def floor_under(world: np.ndarray, candidates: np.ndarray, floor_z: float,
                others: list[Box]) -> np.ndarray:
    """The floor's height under each of ``world``'s points as THIS view sees it: a plane fitted to
    the points near RoomPlan's floor that lie in no object's box (``candidates`` excludes this
    one's), refitted on its own inliers. A frame's depth and pose drift: office_room's carpet is
    seen 2-10 cm above RoomPlan's floor and tilted up to ~6 deg depending on the view, so cutting
    at RoomPlan's flat floor left a patch of carpet under every chair. Too few floor points, or a
    plane steeper than `FLOOR_MAX_TILT`, and RoomPlan's floor stands."""
    flat = np.full(len(world), floor_z)
    z = world[:, 2]
    near = candidates & (np.abs(z - floor_z) < FLOOR_SEARCH)
    for other in others:
        idx = np.nonzero(near)[0]
        if not len(idx):
            break
        near[idx[_inside(other.to_local(world[idx]), np.zeros(3), other.half + THEIR_PAD)]] = False
    pts = world[near]
    if len(pts) < FLOOR_MIN_PX:
        return flat
    coef = None
    for keep in (None, FLOOR_INLIER):
        if keep is not None:
            res = pts[:, 2] - np.c_[pts[:, :2], np.ones(len(pts))] @ coef
            pts = pts[np.abs(res) < keep]
            if len(pts) < FLOOR_MIN_PX:
                return flat
        coef, *_ = np.linalg.lstsq(np.c_[pts[:, :2], np.ones(len(pts))], pts[:, 2], rcond=None)
    if math.degrees(math.atan(math.hypot(coef[0], coef[1]))) > FLOOR_MAX_TILT:
        return flat
    return np.c_[world[:, :2], np.ones(len(world))] @ coef


def _overlapping(box: Box, boxes: dict[str, Box]) -> list[Box]:
    reach = float(np.linalg.norm(box.half)) + MASK_MARGIN
    return [o for o in boxes.values() if o.handle != box.handle
            and float(np.linalg.norm(o.centre - box.centre)) < reach + float(np.linalg.norm(o.half))]


def rank_views(frames: list[Frame], box: Box, boxes: dict[str, Box], floor_z: float,
               walls: list[tuple[np.ndarray, np.ndarray]], depth_of) -> list[dict]:
    """The object's best depth-masked views, best first."""
    rows = []
    others = _overlapping(box, boxes)
    for _score, frame in shortlist(frames, box):
        row = object_mask(frame, depth_of(frame), box, others, floor_z, walls)
        if row is not None:
            rows.append(row)
    if not rows:
        return []
    most = max(r["n_obj"] for r in rows)
    for r in rows:
        ys, xs = np.nonzero(r["mask"])
        h, w = r["mask"].shape
        bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        r["bbox"] = tuple(int(b) for b in bbox)
        unoccluded = r["n_obj"] / max(r["n_box"], 1)
        r["unoccluded"] = unoccluded
        r["score"] = float(quality(bbox, r["n_obj"], most, w, h)
                           * (0.5 + 0.5 * min(1.0, unoccluded / 0.5)) * r["complete"] ** 2)
    return sorted(rows, key=lambda r: -r["score"])[:VIEWS]


# ── cutting out ──────────────────────────────────────────────────────────────────────────────────


def upright_turns(c2w: np.ndarray) -> int:
    """Quarter turns (PIL ``rotate``, counter-clockwise) that put the room's up at the top of
    the picture: a phone held sideways or upside down still gives an upright object."""
    up = c2w[:3, :3].T @ np.array([0.0, 0.0, 1.0])     # room up, in camera coordinates
    du, dv = up[0], -up[1]                              # its direction on the image (v down)
    angle = math.degrees(math.atan2(-dv, du))           # 90: already up
    return int(round((90.0 - angle) / 90.0)) % 4


def refine(rgb: np.ndarray, coarse: np.ndarray, low: np.ndarray | None = None
           ) -> tuple[np.ndarray, str]:
    """GrabCut on the photo seeded by the coarse (depth) mask; (mask, source). ``low`` is the
    part of the mask depth is unsure of (just above the floor): there it starts as probably
    background and the colours decide. A result that lost most of the object or grew far past it
    is not trusted, and the depth mask stands."""
    import cv2

    low = np.zeros_like(coarse) if low is None else (low & coarse)
    certain = coarse & ~low
    if certain.sum() < 16:
        return coarse, "depth"
    h, w = coarse.shape
    scale = min(1.0, GRABCUT_SIDE / max(h, w))
    size = (max(8, int(w * scale)), max(8, int(h * scale))) if scale < 1 else (w, h)
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if scale < 1 else rgb.copy()

    def shrink(m: np.ndarray) -> np.ndarray:
        return cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)

    fg, unsure = shrink(certain), shrink(low)
    k = max(3, int(0.03 * max(small.shape[:2])))
    kernel = np.ones((k, k), np.uint8)
    sure = cv2.erode(fg, kernel)
    maybe = cv2.dilate(fg | unsure, kernel, iterations=2)
    labels = np.full(fg.shape, cv2.GC_BGD, np.uint8)
    labels[maybe > 0] = cv2.GC_PR_BGD
    labels[fg > 0] = cv2.GC_PR_FGD
    labels[sure > 0] = cv2.GC_FGD
    if not (labels == cv2.GC_BGD).any() or not (labels == cv2.GC_FGD).any():
        return coarse, "depth"
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    # GrabCut seeds its colour models with k-means on OpenCV's RNG; seeded here (per thread), the
    # same photo gives the same mask, so the same crop, so a cached mesh is found again.
    cv2.setRNGSeed(GRABCUT_SEED)
    try:
        cv2.grabCut(np.ascontiguousarray(small[..., ::-1]), labels, None, bgd, fgd,
                    GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return coarse, "depth"
    out = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)).astype(np.uint8)
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR) > 0
    ratio = out.sum() / max(certain.sum(), 1)
    if ratio < 0.5 or ratio > 1.8 * coarse.sum() / max(certain.sum(), 1):
        return coarse, "depth"
    return out, "depth+grabcut"


@dataclass
class ObjectCrop:
    handle: str
    category: str
    image: object | None             # PIL RGBA, or None when no frame saw the object well
    views: list[View] = field(default_factory=list)
    frame: int | None = None
    score: float = 0.0
    mask_source: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.image is not None

    @property
    def good(self) -> bool:
        """Worth generating from: a whole view of the object, not a sliver (`MIN_SCORE`)."""
        return self.ok and self.score >= MIN_SCORE

    def png(self) -> bytes:
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        return buffer.getvalue()

    def summary(self) -> dict:
        return {"frame": self.frame, "score": round(self.score, 4), "mask": self.mask_source,
                "size": list(self.image.size) if self.image is not None else None,
                "views": [v.frame for v in self.views], "note": self.note}


def cut_out(frame: Frame, row: dict, *, grabcut: bool = True) -> tuple[object, str]:
    """The RGBA crop of one view: photo at full resolution, refined mask, upright, 512-1024 px."""
    from PIL import Image

    mask_d = row["mask"]
    dh, dw = mask_d.shape
    W, H = frame.size
    sx, sy = W / dw, H / dh
    x0, y0, x1, y1 = row["bbox"]
    pad = CROP_PAD * max(x1 - x0, y1 - y0)
    box = (int(max(0, (x0 - pad) * sx)), int(max(0, (y0 - pad) * sy)),
           int(min(W, math.ceil((x1 + pad) * sx))), int(min(H, math.ceil((y1 + pad) * sy))))
    with Image.open(frame.image) as im:
        photo = im.convert("RGB").crop(box)
    rgb = np.asarray(photo)
    def full(m: np.ndarray) -> np.ndarray:
        up = Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.Resampling.BILINEAR)
        return np.asarray(up.crop(box)) > 127

    coarse = full(mask_d)
    low = full(row["low"]) if row.get("low") is not None else None
    mask, source = refine(rgb, coarse, low) if grabcut else (coarse, "depth")
    rgba = np.dstack([rgb, mask.astype(np.uint8) * 255])
    image = Image.fromarray(rgba, "RGBA")
    turns = upright_turns(frame.c2w)
    if turns:
        image = image.rotate(90 * turns, expand=True)
    alpha = np.asarray(image)[..., 3]
    ys, xs = np.nonzero(alpha > 127)
    if len(xs):
        side = max(xs.max() - xs.min(), ys.max() - ys.min())
        m = int(0.04 * side) + 2
        image = image.crop((max(0, xs.min() - m), max(0, ys.min() - m),
                            min(image.width, xs.max() + m + 1),
                            min(image.height, ys.max() + m + 1)))
    longest = max(image.size)
    if longest < MIN_SIDE or longest > MAX_SIDE:
        target = MIN_SIDE if longest < MIN_SIDE else MAX_SIDE
        f = target / longest
        image = image.resize((max(1, round(image.width * f)), max(1, round(image.height * f))),
                             Image.Resampling.LANCZOS)
    return image, source


def object_crops(scan_dir: Path | str, boxes: dict[str, Box], floor_z: float, *,
                 walls: list[tuple[np.ndarray, np.ndarray]] | None = None, grabcut: bool = True,
                 handles: list[str] | None = None, workers: int = 4) -> dict[str, ObjectCrop]:
    """A crop (and its views) for every box the objects stage generates (or ``handles``). Never
    raises for one object: one no frame saw well gets ``image=None`` and a note."""
    frames = load_frames(scan_dir)
    wanted = [h for h in (handles or [h for h, b in boxes.items() if b.generated()])
              if h in boxes]
    if not frames:
        return {h: ObjectCrop(h, boxes[h].category, None, note="no frames with photo and depth")
                for h in wanted}
    walls = [(np.asarray(s, float), np.asarray(e, float)) for s, e in (walls or [])]
    cache: dict[int, _Depth] = {}
    lock = threading.Lock()

    def depth_of(frame: Frame) -> _Depth:
        with lock:                  # shared by the workers: each depth map is read once
            if frame.index not in cache:
                cache[frame.index] = _read_depth(frame)
            return cache[frame.index]

    def one(handle: str) -> ObjectCrop:
        box = boxes[handle]
        try:
            rows = rank_views(frames, box, boxes, floor_z, walls, depth_of)
            if not rows:
                return ObjectCrop(handle, box.category, None, note="no frame sees it well")
            best = rows[0]
            image, source = cut_out(best["frame"], best, grabcut=grabcut)
            views = [View(r["frame"].index, r["K"], r["frame"].w2c, r["mask"]) for r in rows]
            return ObjectCrop(handle, box.category, image, views, best["frame"].index,
                              best["score"], source)
        except Exception as exc:  # noqa: BLE001 — one bad object keeps its template, not the room
            return ObjectCrop(handle, box.category, None, note=f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        crops = dict(zip(wanted, pool.map(one, wanted)))
    return crops


def save_crops(crops: dict[str, ObjectCrop], folder: Path) -> None:
    """The crops as PNGs plus crops.json, for looking at what TRELLIS was given."""
    folder.mkdir(parents=True, exist_ok=True)
    for handle, crop in crops.items():
        if crop.ok:
            crop.image.save(folder / f"{handle}.png")
    (folder / "crops.json").write_text(json.dumps(
        {h: c.summary() for h, c in crops.items()}, indent=1), encoding="utf-8")
