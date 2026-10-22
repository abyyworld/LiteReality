"""fidelity.py — does the room on the call look like the room that was scanned?

The capture already holds the ground truth for that question: every ``frame_NNNNN.jpg`` the
LiteReality Scanner kept comes with the camera pose and intrinsics that took it. So the room is
drawn the way the room call draws it — `geometry.build_scene_assets`' glb, in three.js 0.168 with
the client's lights, tone mapping and colour space — from exactly those cameras, and each render is
compared with its photo:

* **PSNR and SSIM** on images downsampled to a common size (320x240 by default). The quick room's
  textures are photos resampled onto walls and its furniture is boxes, so fine detail is not what
  any stage claims; at this size the numbers measure layout, colour and shading, not texture
  sharpness. SSIM is computed on luma with the standard 11-tap Gaussian (sigma 1.5).
* the same, restricted to **object pixels** and to **static surfaces** (walls, floor, ceiling:
  every pixel that is no object and no door/window), so a stage that improves the furniture can
  show it even when the walls dominate the frame. Object pixels come from the capture's own
  LiDAR (`view_regions`): each pixel's depth is unprojected and the pixel is an object's when that
  3D point lies inside a RoomPlan box and is not the floor or a wall — what the camera really saw
  of the object, occlusion included, and not the floor between a chair's legs or the wall behind
  its back, which a box silhouette would count (and which made the "objects" numbers move with
  the wall textures). The mask is the same for every stage, so it never favours one; a pixel
  without a trustworthy depth return falls back to the box silhouettes;
* an **exposure-matched PSNR** (one least-squares gain on the render per view): ARKit re-exposes
  every frame, and the photo walls were exposure-balanced, so part of any colour error is the
  camera's, not the room's;
* a **baseline** PSNR — each photo against a flat image of its own mean colour — so a reader can
  tell how much of a number is the room and how much is just the photo's own contrast.

Every view is tagged with whether the stage's build read that frame (`holdout.build_sources`):
the quick room's walls are made of these very photos, so a view the build saw is a training view.
With ``holdout=k`` the quick room is also rebuilt k times, each time without one fold of the
frames, and every view is scored against the room that never saw it (`holdout`); the report keeps
both, and the gap between them.

The pose maths is the part that must be exactly right, so it lives in one place (`camera_pose`,
`intrinsics_at`, `three_camera`): ``cameraPoseARFrame`` is camera -> ARKit world, row-major; the
room is in the pipeline's Z-up frame through ``ARKIT_TO_Z_UP`` (as `quickroom.colours` and
`quickroom.textures` read the same frames); three's camera looks down its -Z with +Y up exactly as
ARKit's does, and the page turns the room -90 deg about X as the client does. The projection
matrix is built from the intrinsics including the principal point, not from a field of view.

Rendering needs a browser. `PlaywrightRenderer` drives headless Chromium through the optional
`playwright` package with SwiftShader (software WebGL, so a render is the same on every machine);
any object with the `Renderer` protocol can stand in (tests use one that needs no browser).
"""

from __future__ import annotations

import base64
import contextlib
import functools
import http.server
import io
import json
import math
import os
import shlex
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw

from litereality_agent.pipeline.evaluate import holdout as held
from litereality_agent.pipeline.quickroom.roomplan import RoomPlan, parse_room
from litereality_agent.pipeline.scene_init.layout.shell import ARKIT_TO_Z_UP

SIZE = (320, 240)             # metric resolution (w, h); the captures are 4:3 landscape
RENDER_SCALE = 2              # render at 2x and box-filter down: antialiasing that is the same
                              # everywhere, unlike MSAA on whatever GPU a browser picked
NEAR, FAR = 0.05, 200.0       # the client's camera planes (app.js)
BACKGROUND = (0xDF, 0xE5, 0xE8)  # the room call's light-theme scene background (--scene-bg)
SSIM_SIGMA = 1.5
SSIM_TRUNCATE = 3.5           # 1.5 * 3.5 -> radius 5: the standard 11x11 window
MIN_MASK_FRACTION = 0.01      # a view whose region is smaller than this says nothing about it
MAX_PSNR = 100.0              # dB for identical images, so means stay finite
OPENING_DEPTH = 0.15          # m either side of a wall a door/window box reaches
# The LiDAR object mask (`view_regions`).
DEPTH_RANGE = (0.1, 10.0)     # m: returns outside this are noise or nothing
BOX_MARGIN = 0.03             # m around a RoomPlan box: LiDAR noise and the box fit
FLOOR_TOL = 0.03              # m above the floor (as the view's own LiDAR finds it): floor, not object
FLOOR_SEARCH = 0.06           # m either side of RoomPlan's floor height the floor is looked for
WALL_TOL = 0.05               # m from a wall's plane: wall, not object (RoomPlan's planes are good
                              # to a few cm)
OCCLUSION_MARGIN = 0.10       # m: a return this far in front of a door/window box hides it
CONTACT_VIEWS = 8
BATCH = 6                     # views per page call: keeps each CDP message a few MB
START_TIMEOUT_MS = 45_000     # the page loading three.js: an unreachable CDN fails in this, not 2 min
THREE_VERSION = "0.168.0"
CDN = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/"
RENDER_PAGE = Path(__file__).with_name("render.html")

# (x, y, z) -> (x, z, -y): the client's `world.rotation.x = -PI/2`, as a matrix.
Z_UP_TO_THREE = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])


# ── the capture's cameras ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class View:
    """One captured photo and the camera that took it, in the room's Z-up frame."""

    index: int
    image: Path
    K: np.ndarray             # (3, 3) intrinsics at the photo's full resolution
    full: tuple[int, int]     # the photo's (w, h)
    pose: np.ndarray          # (4, 4) camera -> world, Z up; the camera looks down its -Z, +Y up
    depth: Path | None = None  # depth_NNNNN.png (mm, aligned with the photo), if the capture has it
    conf: Path | None = None   # conf_NNNNN.png (ARKit confidence 0-2)


def camera_pose(pose_arkit: Sequence[float] | np.ndarray) -> np.ndarray:
    """``cameraPoseARFrame`` (16 numbers, row-major, camera -> ARKit world) in the room's Z-up
    frame — what `quickroom.colours` and `quickroom.textures` do with the same field."""
    pose = np.asarray(pose_arkit, dtype=float).reshape(4, 4)
    return ARKIT_TO_Z_UP @ pose


def capture_views(scan_dir: Path | str, limit: int | None = None) -> list[View]:
    """Every frame with a pose AND a photo (depth is not needed, but kept when there), evenly
    subsampled to ``limit``. A frame whose pose or intrinsics are not finite is skipped, not
    fatal."""
    found: list[View] = []
    for meta in sorted(Path(scan_dir).glob("frame_*.json")):
        image = meta.with_suffix(".jpg")
        if not image.is_file():
            continue
        try:
            stem = meta.stem[len("frame_"):]
            index = int(stem)
            data = json.loads(meta.read_text())
            K = np.asarray(data["intrinsics"], dtype=float).reshape(3, 3)
            pose = camera_pose(data["cameraPoseARFrame"])
            with Image.open(image) as im:
                full = im.size
        except (ValueError, KeyError, OSError):
            continue
        if not (np.all(np.isfinite(K)) and np.all(np.isfinite(pose))):
            continue
        depth, conf = meta.with_name(f"depth_{stem}.png"), meta.with_name(f"conf_{stem}.png")
        found.append(View(index, image, K, full, pose, depth if depth.is_file() else None,
                          conf if conf.is_file() else None))
    if limit and len(found) > limit:
        picks = np.linspace(0, len(found) - 1, limit).round().astype(int)
        found = [found[i] for i in sorted(set(picks.tolist()))]
    return found


def intrinsics_at(view: View, size: tuple[int, int]) -> tuple[float, float, float, float]:
    """(fx, fy, cx, cy) at another resolution. Pixel i spans [i, i+1) (centre i + 0.5), the
    convention `quickroom.colours` unprojects with, so the scaling is plain proportion."""
    sx, sy = size[0] / view.full[0], size[1] / view.full[1]
    K = view.K
    return float(K[0, 0] * sx), float(K[1, 1] * sy), float(K[0, 2] * sx), float(K[1, 2] * sy)


def project(points: np.ndarray, view: View, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection of Z-up world points: ((n, 2) pixel coordinates, (n,) depth in front of
    the camera). Depth <= 0 is behind it."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    w2c = np.linalg.inv(view.pose)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    depth = -cam[:, 2]
    fx, fy, cx, cy = intrinsics_at(view, size)
    safe = np.where(np.abs(depth) > 1e-12, depth, 1e-12)
    uv = np.column_stack([cx + fx * cam[:, 0] / safe, cy - fy * cam[:, 1] / safe])
    return uv, depth


def projection_matrix(fx: float, fy: float, cx: float, cy: float, size: tuple[int, int],
                      near: float = NEAR, far: float = FAR) -> np.ndarray:
    """OpenGL projection from intrinsics, principal point included (v counts down from the top):
    x_ndc = 2 fx / w * x / -z + (2 cx / w - 1), y_ndc = 2 fy / h * y / -z + (1 - 2 cy / h)."""
    w, h = size
    return np.array([
        [2.0 * fx / w, 0.0, 1.0 - 2.0 * cx / w, 0.0],
        [0.0, 2.0 * fy / h, 2.0 * cy / h - 1.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ])


def three_camera(view: View, size: tuple[int, int]) -> dict[str, list[float]]:
    """The camera as the render page takes it: column-major ``matrixWorld`` (camera -> three's
    Y-up world) and ``projection``."""
    world = Z_UP_TO_THREE @ view.pose
    proj = projection_matrix(*intrinsics_at(view, size), size)
    return {"matrixWorld": world.T.reshape(-1).tolist(), "projection": proj.T.reshape(-1).tolist()}


def project_three(points: np.ndarray, camera: dict[str, list[float]],
                  size: tuple[int, int]) -> np.ndarray:
    """What three.js does with `three_camera`'s matrices, in numpy: Z-up point -> the page's
    Y-up world -> view -> clip -> NDC -> pixel (v down). Used to check the two paths agree."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    world = np.asarray(camera["matrixWorld"], dtype=float).reshape(4, 4).T
    proj = np.asarray(camera["projection"], dtype=float).reshape(4, 4).T
    homo = np.c_[pts, np.ones(len(pts))] @ Z_UP_TO_THREE.T
    clip = homo @ (proj @ np.linalg.inv(world)).T
    ndc = clip[:, :3] / clip[:, 3:4]
    w, h = size
    return np.column_stack([(ndc[:, 0] + 1.0) * 0.5 * w, (1.0 - ndc[:, 1]) * 0.5 * h])


# ── which pixels are what ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Box:
    """An oriented box in the room: centre, half extents along its own axes, yaw about Z."""

    name: str
    centre: np.ndarray
    half: np.ndarray
    yaw: float                # radians


def scan_boxes(plan: RoomPlan) -> tuple[list[Box], list[Box]]:
    """(object boxes, opening boxes) from the scan's own RoomPlan export — the same for every
    stage, so a mask never favours the stage it is judging. An opening is its wall's rectangle,
    `OPENING_DEPTH` either side of the wall."""
    objects = [Box(o.handle, np.asarray(o.center, dtype=float),
                   np.asarray(o.size, dtype=float) / 2.0, math.radians(o.yaw))
               for o in plan.objects.values()]
    openings = []
    for op in plan.openings.values():
        wall = plan.walls.get(op.wall)
        if wall is None:
            continue
        along = wall.along
        mid = np.asarray(wall.start) + along * op.offset
        z = plan.floor_z + op.sill + op.height / 2.0
        openings.append(Box(op.handle, np.array([mid[0], mid[1], z]),
                            np.array([op.width / 2.0, OPENING_DEPTH, op.height / 2.0]),
                            math.atan2(along[1], along[0])))
    return objects, openings


def pixel_rays(view: View, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(camera centre (3,), world direction per pixel centre (h*w, 3), not normalised)."""
    w, h = size
    fx, fy, cx, cy = intrinsics_at(view, size)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    cam = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], axis=-1).reshape(-1, 3)
    return view.pose[:3, 3].copy(), cam @ view.pose[:3, :3].T


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_hits(boxes: Sequence[Box], view: View,
             size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """((h, w) bool, (h, w) float): which pixels' rays pass through any of ``boxes`` in front of
    the camera — exact silhouettes by slab test, so a box half behind the camera is still right —
    and the nearest box's entry distance along each ray (inf where none is hit). The rays have
    camera-z length 1 (`pixel_rays`), so that distance is directly comparable with a LiDAR depth.

    A box the camera stands INSIDE is skipped: it has no silhouette, and every ray would count as
    hitting it. That is a real case, not a corner one — people scan from doorways (tea_room's
    frame 61 is taken from inside the door's opening box)."""
    w, h = size
    origin, dirs = pixel_rays(view, size)
    hit = np.zeros(len(dirs), dtype=bool)
    entry = np.full(len(dirs), np.inf)
    for box in boxes:
        rot = _yaw_matrix(box.yaw)
        o = rot.T @ (origin - box.centre)
        if np.all(np.abs(o) < box.half):
            continue
        d = dirs @ rot
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-box.half - o) / d
            t2 = (box.half - o) / d
        lo = np.nanmax(np.where(np.isfinite(t1), np.minimum(t1, t2), -np.inf), axis=1)
        hi = np.nanmin(np.where(np.isfinite(t1), np.maximum(t1, t2), np.inf), axis=1)
        # A ray parallel to a slab is inside it for all t or for none.
        parallel = ~np.isfinite(t1)
        outside = parallel & (np.abs(o)[None, :] > box.half[None, :])
        start = np.maximum(lo, NEAR)
        this = (hi >= start) & ~outside.any(axis=1)
        hit |= this
        entry = np.where(this, np.minimum(entry, start), entry)
    return hit.reshape(h, w), entry.reshape(h, w)


def box_mask(boxes: Sequence[Box], view: View, size: tuple[int, int]) -> np.ndarray:
    """(h, w) bool: the boxes' silhouettes (`box_hits`)."""
    return box_hits(boxes, view, size)[0]


def view_depth(view: View, size: tuple[int, int]) -> np.ndarray | None:
    """The frame's LiDAR depth (camera z, metres) at ``size``, nearest-sampled from the depth map
    (aligned with the photo, so its intrinsics are the photo's scaled — as `quickroom.colours`
    reads it); NaN where there is no trustworthy return (out of `DEPTH_RANGE`, or ARKit's low
    confidence). None when the frame has no readable depth map."""
    if view.depth is None:
        return None
    try:
        with Image.open(view.depth) as im:
            depth = np.asarray(im, dtype=np.float32) / 1000.0
    except OSError:
        return None
    if depth.ndim != 2:
        return None
    dh, dw = depth.shape
    ok = (depth > DEPTH_RANGE[0]) & (depth < DEPTH_RANGE[1])
    if view.conf is not None:
        try:
            with Image.open(view.conf) as im:
                conf = np.asarray(im)
            if conf.shape[:2] == depth.shape:
                ok &= conf >= 1                 # ARKit confidence: 0 low, 1 medium, 2 high
        except OSError:
            pass
    depth = np.where(ok, depth, np.nan)
    w, h = size
    ys = np.minimum(((np.arange(h) + 0.5) * dh / h).astype(int), dh - 1)
    xs = np.minimum(((np.arange(w) + 0.5) * dw / w).astype(int), dw - 1)
    return depth[np.ix_(ys, xs)]


def _inside_any(points: np.ndarray, boxes: Sequence[Box], margin: float) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for box in boxes:
        local = (points - box.centre) @ _yaw_matrix(box.yaw)
        inside |= np.all(np.abs(local) <= box.half + margin, axis=1)
    return inside


def _on_structure(points: np.ndarray, plan: RoomPlan) -> np.ndarray:
    """Points on the floor, the ceiling or a wall's plane: what lies inside an object's box but
    is not the object (the floor under a chair, the wall behind a cabinet's margin). The floor is
    taken where this view's own LiDAR finds it near RoomPlan's floor height, so a scan whose
    floor is a couple of centimetres off does not turn the floor under a table into "object"."""
    z = points[:, 2]
    band = np.abs(z - plan.floor_z) < FLOOR_SEARCH
    floor = float(np.median(z[band])) if band.sum() >= 50 else plan.floor_z
    near = (z < max(floor, plan.floor_z) + FLOOR_TOL) | (z > plan.ceiling_z - FLOOR_TOL)
    for wall in plan.walls.values():
        along = wall.along
        normal = np.array([-along[1], along[0]])
        rel = points[:, :2] - np.asarray(wall.start, dtype=float)
        t = rel @ along
        near |= ((np.abs(rel @ normal) < WALL_TOL) & (t > -WALL_TOL)
                 & (t < wall.length + WALL_TOL))
    return near


def view_regions(view: View, plan: RoomPlan, objects: Sequence[Box], openings: Sequence[Box],
                 size: tuple[int, int]) -> dict[str, Any]:
    """Which pixels of one view are objects, doors/windows and static surfaces, from the capture
    alone (so every stage is judged on the same pixels):

    * **objects**: where the frame has depth, a pixel whose LiDAR point lies inside a RoomPlan
      object box (`BOX_MARGIN`) and is not on the floor, ceiling or a wall (`_on_structure`):
      the object as the camera saw it, occlusion included — not the floor between a chair's legs,
      not a chair hidden behind the table. Where the frame has no trustworthy return, the box
      silhouettes stand in.
    * **openings**: a door/window box's silhouette, unless the depth says something stands in
      front of it (a wall corner of an L-shaped room); never an object pixel.
    * **static**: everything else — walls, floor, ceiling.

    Also says which mask it was ("depth" or "box") and how much of the view had depth."""
    w, h = size
    silhouette = box_mask(objects, view, size)
    in_opening, entry = box_hits(openings, view, size)
    depth = view_depth(view, size)
    if depth is None:
        obj, gap, kind, valid_frac = silhouette, in_opening & ~silhouette, "box", 0.0
    else:
        valid = np.isfinite(depth)
        origin, dirs = pixel_rays(view, size)
        flat = valid.reshape(-1)
        points = origin + dirs[flat] * depth.reshape(-1)[flat, None]
        inside = _inside_any(points, objects, BOX_MARGIN) & ~_on_structure(points, plan)
        obj = np.zeros(h * w, dtype=bool)
        obj[flat] = inside
        obj = obj.reshape(h, w) | (silhouette & ~valid)
        with np.errstate(invalid="ignore"):
            hidden = valid & (depth < entry - OCCLUSION_MARGIN)
        gap = in_opening & ~hidden & ~obj
        kind, valid_frac = "depth", float(valid.mean())
    return {"objects": obj, "openings": gap, "static": ~obj & ~gap, "mask": kind,
            "depth_valid": round(valid_frac, 4)}


# ── metrics ──────────────────────────────────────────────────────────────────────────────────────


def _luma(img: np.ndarray) -> np.ndarray:
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """PSNR in dB of two float images in [0, 1] (data range 1), over ``mask`` if given. None for
    an empty mask; `MAX_PSNR` for identical pixels."""
    diff = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    if mask is not None:
        if not mask.any():
            return None
        diff = diff[mask]
    mse = float(diff.mean())
    return MAX_PSNR if mse <= 10 ** (-MAX_PSNR / 10.0) else 10.0 * math.log10(1.0 / mse)


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel SSIM of two float images in [0, 1] (luma if colour; Wang et al. 2004, Gaussian
    window sigma 1.5, K1 0.01, K2 0.03, data range 1)."""
    from scipy.ndimage import gaussian_filter

    x = _luma(a) if a.ndim == 3 else np.asarray(a, dtype=float)
    y = _luma(b) if b.ndim == 3 else np.asarray(b, dtype=float)

    def blur(z: np.ndarray) -> np.ndarray:
        return gaussian_filter(z, SSIM_SIGMA, truncate=SSIM_TRUNCATE, mode="reflect")

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cov = blur(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """Mean SSIM, over ``mask`` if given (the map is computed on the whole image first, so a
    masked region's window still sees its surroundings, as it does in the photo)."""
    smap = ssim_map(a, b)
    if mask is not None:
        if not mask.any():
            return None
        return float(smap[mask].mean())
    return float(smap.mean())


def gain_matched(render: np.ndarray, real: np.ndarray) -> np.ndarray:
    """The render times the one gain that best fits the photo (least squares), clipped to [0, 1]."""
    x = render.reshape(-1)
    denom = float(x @ x)
    gain = float(x @ real.reshape(-1)) / denom if denom > 0 else 1.0
    return np.clip(render * gain, 0.0, 1.0)


def compare(real: np.ndarray, render: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """Every metric of one view. ``masks`` holds "static" and "objects" (h, w) bool arrays."""
    flat = np.broadcast_to(real.reshape(-1, 3).mean(axis=0), real.shape)
    row: dict[str, Any] = {
        "psnr": psnr(real, render), "ssim": ssim(real, render),
        "psnr_gain": psnr(real, gain_matched(render, real)),
        "psnr_baseline": psnr(real, flat),
    }
    smap = ssim_map(real, render)
    for name, mask in masks.items():
        frac = float(mask.mean())
        row[f"frac_{name}"] = round(frac, 4)
        use = frac >= MIN_MASK_FRACTION
        row[f"psnr_{name}"] = psnr(real, render, mask) if use else None
        row[f"ssim_{name}"] = float(smap[mask].mean()) if use else None
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}


# ── images ───────────────────────────────────────────────────────────────────────────────────────


def load_photo(path: Path, size: tuple[int, int]) -> np.ndarray:
    """The photo at ``size`` as float RGB in [0, 1]: decoded straight to ~2x in the DCT domain,
    then area-averaged, so it is filtered the same way as the render."""
    with Image.open(path) as im:
        im.draft("RGB", (size[0] * 2, size[1] * 2))
        img = im.convert("RGB").resize(size, Image.Resampling.BOX)
    return np.asarray(img, dtype=np.float64) / 255.0


def composite(rgba: np.ndarray, size: tuple[int, int],
              background: tuple[int, int, int] = BACKGROUND) -> tuple[np.ndarray, np.ndarray]:
    """A render (h, w, 4 uint8, straight alpha) over the call's background, area-averaged to
    ``size``: (float RGB in [0, 1], coverage in [0, 1] — how much of each pixel the room drew)."""
    arr = np.asarray(rgba, dtype=np.float64) / 255.0
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.ones(arr.shape[:2] + (1,))], axis=2)
    alpha = arr[..., 3:4]
    bg = np.asarray(background, dtype=np.float64) / 255.0
    rgb = arr[..., :3] * alpha + bg * (1.0 - alpha)
    h, w = arr.shape[:2]
    if (w, h) != tuple(size):
        fy, fx = h // size[1], w // size[0]
        if fy * size[1] == h and fx * size[0] == w:
            rgb = rgb.reshape(size[1], fy, size[0], fx, 3).mean(axis=(1, 3))
            alpha = alpha.reshape(size[1], fy, size[0], fx, 1).mean(axis=(1, 3))
        else:
            img = Image.fromarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8))
            rgb = np.asarray(img.resize(size, Image.Resampling.BOX), dtype=np.float64) / 255.0
            a = Image.fromarray(np.clip(alpha[..., 0] * 255 + 0.5, 0, 255).astype(np.uint8))
            alpha = (np.asarray(a.resize(size, Image.Resampling.BOX), dtype=np.float64)
                     / 255.0)[..., None]
    return rgb, alpha[..., 0]


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.asarray(arr) * 255.0 + 0.5, 0, 255).astype(np.uint8))


def contact_sheet(rows: list[dict], path: Path, title: str = "") -> Path:
    """Side by side per view: the photo, the render, and the photo with the evaluation's regions
    tinted (objects orange, doors/windows blue; the rest is static surface), each row labelled
    with its numbers. ``rows``: {"frame", "real", "render", "objects", "openings", "metrics"}."""
    from litereality_agent.fonts import font

    if not rows:
        raise ValueError("no views for a contact sheet")
    h, w = rows[0]["real"].shape[:2]
    label_h, pad = 22, 4
    head = (30 if title else 4) + 18
    sheet = Image.new("RGB", (3 * w + 4 * pad, head + len(rows) * (h + label_h + pad) + pad),
                      (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    small, big = font(13), font(17)
    if title:
        draw.text((pad, 6), title, fill=(20, 20, 20), font=big)
    for k, name in enumerate(("photo", "room call render",
                              "regions: objects orange, doors/windows blue")):
        draw.text((pad + k * (w + pad), head - 16), name, fill=(90, 90, 90), font=small)
    for i, row in enumerate(rows):
        y = head + pad + i * (h + label_h + pad)
        m = row["metrics"]
        text = (f"frame {row['frame']}   PSNR {m['psnr']:.1f} dB   SSIM {m['ssim']:.3f}   "
                f"static {_num(m.get('psnr_static'))} dB   "
                f"objects {_num(m.get('psnr_objects'))} dB")
        draw.text((pad, y + 3), text, fill=(30, 30, 30), font=small)
        tint = row["real"].copy()
        for key, colour in (("objects", (1.0, 0.55, 0.1)), ("openings", (0.2, 0.5, 1.0))):
            mask = row.get(key)
            if mask is not None and mask.any():
                tint[mask] = 0.55 * tint[mask] + 0.45 * np.asarray(colour)
        for k, img in enumerate((row["real"], row["render"], tint)):
            sheet.paste(_to_image(img), (pad + k * (w + pad), y + label_h))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)
    return path


def _num(value) -> str:
    return "—" if value is None else f"{value:.1f}"


# ── renderers ────────────────────────────────────────────────────────────────────────────────────


class Renderer(Protocol):
    """Draws a glb from cameras (`three_camera` dicts) at ``size`` (w, h); returns one (h, w, 4)
    uint8 RGBA array per camera, straight alpha, transparent where nothing was drawn."""

    name: str

    def render(self, glb: bytes, cameras: list[dict], size: tuple[int, int]) -> list[np.ndarray]:
        ...


class RendererUnavailable(RuntimeError):
    """Rendering was asked for, and nothing on this machine can do it; the message says how."""


PLAYWRIGHT_HELP = (
    "rendering needs headless Chromium through Playwright, which is optional: run with "
    "`uv run --with playwright litereality evaluate ... --render` after installing a browser once "
    "with `uv run --with playwright python -m playwright install chromium`. "
    "$LR_EVAL_CHROMIUM may name another Chromium binary to drive instead.")

# Software WebGL everywhere: a GPU's driver changes the pixels, and these numbers go in a paper.
CHROMIUM_ARGS = ("--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                 "--ignore-gpu-blocklist")
# A bare Chromium build (no Playwright-managed browser, often in a container) also needs these.
EXTERNAL_CHROMIUM_ARGS = ("--no-sandbox", "--in-process-gpu", "--no-zygote")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass


@contextlib.contextmanager
def serve_page(glb: bytes):
    """A throwaway HTTP server on 127.0.0.1 holding the render page and the glb, in memory; yields
    its base URL. A file:// page cannot fetch a glb or use an import map in every browser."""
    page = RENDER_PAGE.read_bytes()

    class Handler(_Quiet):
        def do_GET(self) -> None:  # noqa: N802 — the http.server API
            route = self.path.split("?", 1)[0]
            body, kind = {"/": (page, "text/html"), "/index.html": (page, "text/html"),
                          "/scene.glb": (glb, "model/gltf-binary")}.get(route, (None, None))
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="eval-render")
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


class PlaywrightRenderer:
    """Headless Chromium via Playwright (Python), rendering `render.html`.

    ``executable`` (or ``$LR_EVAL_CHROMIUM``) drives a Chromium binary Playwright did not install;
    ``three_dir`` (or ``$LR_EVAL_THREE_DIR``, a ``three@0.168.0`` package folder holding
    ``build/`` and ``examples/``) serves three.js from disk instead of the CDN, for offline runs
    and tests. Extra browser flags may come in ``$LR_EVAL_CHROMIUM_ARGS``."""

    name = "playwright-chromium"

    def __init__(self, *, executable: str | None = None, three_dir: str | None = None,
                 timeout_s: float = 120.0) -> None:
        try:
            import playwright.sync_api  # noqa: F401
        except ImportError as exc:
            raise RendererUnavailable(PLAYWRIGHT_HELP) from exc
        self.executable = executable or os.environ.get("LR_EVAL_CHROMIUM") or None
        self.three_dir = three_dir or os.environ.get("LR_EVAL_THREE_DIR") or None
        self.timeout_ms = int(timeout_s * 1000)
        self.version = ""

    def _args(self) -> list[str]:
        args = list(CHROMIUM_ARGS)
        if self.executable:
            args += EXTERNAL_CHROMIUM_ARGS
        args += shlex.split(os.environ.get("LR_EVAL_CHROMIUM_ARGS", ""))
        return args

    def _route_three(self, page) -> None:
        if not self.three_dir:
            return
        root = Path(self.three_dir)

        def handle(route) -> None:
            rel = route.request.url.split(f"three@{THREE_VERSION}/", 1)[1].split("?", 1)[0]
            path = (root / rel).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                route.fulfill(status=404, body="")
                return
            route.fulfill(status=200, content_type="text/javascript", body=path.read_bytes())

        page.route(f"{CDN}**", handle)

    def render(self, glb: bytes, cameras: list[dict], size: tuple[int, int]) -> list[np.ndarray]:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        out: list[np.ndarray] = []
        errors: list[str] = []
        with serve_page(glb) as base, sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(executable_path=self.executable, args=self._args(),
                                             headless=True)
            except PlaywrightError as exc:
                raise RendererUnavailable(f"{PLAYWRIGHT_HELP} ({exc})") from exc
            try:
                self.version = browser.version
                page = browser.new_page(viewport={"width": size[0], "height": size[1]})
                page.set_default_timeout(self.timeout_ms)
                page.on("pageerror", lambda e: errors.append(str(e)))
                self._route_three(page)
                page.goto(f"{base}/")
                try:
                    page.wait_for_function("window.renderReady === true",
                                           timeout=min(self.timeout_ms, START_TIMEOUT_MS))
                except PlaywrightError as exc:
                    where = (f"three.js from {self.three_dir}" if self.three_dir else
                             f"three.js from {CDN} (offline? point $LR_EVAL_THREE_DIR at a local "
                             f"three@{THREE_VERSION} package)")
                    raise RuntimeError(f"the render page did not start, loading {where}: "
                                       f"{errors or exc}") from exc
                for i in range(0, len(cameras), BATCH):
                    job = {"glb": "scene.glb", "width": size[0], "height": size[1],
                           "views": cameras[i:i + BATCH]}
                    urls = page.evaluate("job => window.renderViews(job)", job)
                    for url in urls:
                        png = base64.b64decode(url.split(",", 1)[1])
                        with Image.open(io.BytesIO(png)) as im:
                            out.append(np.asarray(im.convert("RGBA")))
            finally:
                browser.close()
        return out


@functools.lru_cache(maxsize=1)
def default_renderer() -> Renderer:
    return PlaywrightRenderer()


# ── the evaluation ───────────────────────────────────────────────────────────────────────────────


def scene_glb(scene_xml: Path | str, label: str = "room") -> bytes:
    """The glb the room call serves for this scene (prebuilt artefacts are not trusted here: the
    point is to judge what the scene itself turns into)."""
    from litereality_agent.room_ops.roomcall.geometry import build_scene_assets
    from litereality_agent.room_ops.roomcall.sim import load_model

    glb, _manifest = build_scene_assets(load_model(Path(scene_xml)), label=label)
    return glb


def _score(renderer: Renderer, glb: bytes, views: list[View], regions: dict[int, dict],
           photos: dict[int, np.ndarray], size: tuple[int, int],
           sources: held.Sources) -> tuple[list[dict], list[dict], float]:
    """Render ``views`` of one glb and score each against its photo: (per-view rows, contact
    sheet rows, render seconds). Each row says whether the build behind the glb read that frame
    (``seen``, and by which ``read_by`` roles)."""
    big = (size[0] * RENDER_SCALE, size[1] * RENDER_SCALE)
    t = time.perf_counter()
    renders = renderer.render(glb, [three_camera(v, big) for v in views], big)
    seconds = time.perf_counter() - t
    if len(renders) != len(views):
        raise RuntimeError(f"{renderer.name} returned {len(renders)} images for {len(views)} views")
    rows, sheet = [], []
    for view, rgba in zip(views, renders, strict=True):
        real, reg = photos[view.index], regions[view.index]
        render, coverage = composite(rgba, size)
        masks = {"static": reg["static"], "objects": reg["objects"]}
        row = {"frame": view.index, **compare(real, render, masks),
               "coverage": round(float(coverage.mean()), 4), "mask": reg["mask"],
               "seen": sources.seen(view.index), "read_by": sources.roles(view.index)}
        rows.append(row)
        sheet.append({"frame": view.index, "real": real, "render": render,
                      "objects": reg["objects"], "openings": reg["openings"], "metrics": row})
    return rows, sheet, seconds


def evaluate_fidelity(scene_xml: Path | str, scan_dir: Path | str, *,
                      renderer: Renderer | None = None, size: tuple[int, int] = SIZE,
                      max_views: int | None = None, out_dir: Path | str | None = None,
                      contact_views: int = CONTACT_VIEWS, title: str = "", holdout: int = 0,
                      holdout_split: str = held.DEFAULT_SPLIT) -> dict:
    """Render ``scene_xml`` from every captured camera of ``scan_dir`` and compare. Writes
    ``contact_sheet.jpg`` (and ``contact_sheet_holdout.jpg``) into ``out_dir`` when given.

    The result's ``summary`` is over every view as built; ``splits`` divides it into the views
    the build read and those it did not (`holdout.build_sources`). With ``holdout`` = k >= 2 and
    a quick room, ``holdout`` holds the k-fold held-out evaluation (see `holdout`): its
    ``summary`` is over the same views, each scored by a room built without it, and ``gap`` is
    how much better the as-built room looks on those views than the held-out rooms do."""
    started = time.perf_counter()
    if holdout_split not in held.SPLITS:
        raise ValueError(f"holdout split {holdout_split!r}: expected one of "
                         f"{', '.join(held.SPLITS)}")
    views = capture_views(scan_dir, max_views)
    if not views:
        raise ValueError(f"{scan_dir}: no frame_*.json with a matching frame_*.jpg to compare with")
    plan = parse_room(scan_dir)
    objects, openings = scan_boxes(plan)
    renderer = renderer or default_renderer()
    regions = {v.index: view_regions(v, plan, objects, openings, size) for v in views}
    photos = {v.index: load_photo(v.image, size) for v in views}
    sources = held.build_sources(scene_xml, scan_dir)
    t = time.perf_counter()
    glb = scene_glb(scene_xml)
    glb_seconds = time.perf_counter() - t
    per_view, sheet_rows, render_seconds = _score(renderer, glb, views, regions, photos, size,
                                                  sources)
    masks = [r["mask"] for r in per_view]
    out: dict[str, Any] = {
        "scan": str(scan_dir), "scene": str(scene_xml), "views": len(views),
        "size": list(size), "render_scale": RENDER_SCALE, "renderer": renderer.name,
        "renderer_version": getattr(renderer, "version", ""),
        "object_mask": {"depth": masks.count("depth"), "box": masks.count("box")},
        "sources": sources.summary(),
        "views_seen": _seen_counts(per_view),
        "summary": _summarise(per_view), "splits": _splits(per_view), "per_view": per_view,
        "seconds": {"glb": round(glb_seconds, 2), "render": round(render_seconds, 2)},
        "definitions": DEFINITIONS,
    }
    hold_rows: list[dict] = []
    if holdout and holdout >= 2:
        out["holdout"], hold_rows = _holdout(scene_xml, scan_dir, views, regions, photos, size,
                                             renderer, holdout, holdout_split, per_view)
    elif holdout:
        out["holdout"] = {"skipped": f"holdout={holdout}: k-fold needs k >= 2"}
    out["seconds"]["total"] = round(time.perf_counter() - started, 2)
    if out_dir is not None and contact_views > 0:
        picks = np.linspace(0, len(sheet_rows) - 1, min(contact_views, len(sheet_rows)))
        frames = [sheet_rows[i]["frame"] for i in sorted(set(picks.round().astype(int).tolist()))]
        chosen = [r for r in sheet_rows if r["frame"] in frames]
        sheet = contact_sheet(chosen, Path(out_dir) / "contact_sheet.jpg",
                              title=f"{title} · as built" if title else "as built")
        out["contact_sheet"] = sheet.name
        if hold_rows:
            hold = out["holdout"]
            chosen = [r for r in hold_rows if r["frame"] in frames]
            name = f"{hold['k']}-fold held out ({hold['split']})"
            sheet = contact_sheet(chosen, Path(out_dir) / "contact_sheet_holdout.jpg",
                                  title=f"{title} · {name}" if title else name)
            hold["contact_sheet"] = sheet.name
    return out


def _holdout(scene_xml: Path | str, scan_dir: Path | str, views: list[View],
             regions: dict[int, dict], photos: dict[int, np.ndarray], size: tuple[int, int],
             renderer: Renderer, k: int, split: str,
             as_built: list[dict]) -> tuple[dict, list[dict]]:
    """The k-fold held-out evaluation (see `holdout`): (its report, its contact sheet rows)."""
    rebuild, options = held.quick_rebuilder(scene_xml)
    if rebuild is None:
        return {"skipped": options["reason"]}, []
    started = time.perf_counter()
    # Folds over EVERY photo frame of the capture (``views`` may be a --max-views subset), so a
    # fold's build lacks a k-th of the capture whatever is scored.
    folds = held.kfold([v.index for v in capture_views(scan_dir)], k, split)
    by_index = {v.index: v for v in views}
    rows: list[dict] = []
    sheet: list[dict] = []
    info: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="lr-eval-holdout-") as tmp:
        for i, fold in enumerate(folds):
            exclude = set(fold)
            scored = [by_index[j] for j in fold if j in by_index]
            entry: dict[str, Any] = {"fold": i, "excluded": sorted(exclude), "views": len(scored)}
            info.append(entry)
            if not scored:
                continue
            scan = held.scan_without(scan_dir, exclude, Path(tmp) / f"scan_{i}")
            left = held.frames_present(scan) & exclude
            if left:  # the protocol's one promise; cannot happen unless scan_without is broken
                raise RuntimeError(f"fold {i}: frames {sorted(left)} were not removed")
            t = time.perf_counter()
            scene = rebuild(scan, Path(tmp) / f"fold_{i}")
            entry["build_s"] = round(time.perf_counter() - t, 2)
            entry.update(_fold_build(scene))
            sources = held.build_sources(scene, scan)
            entry["leaked"] = sorted(sources.read() & exclude)
            fold_rows, fold_sheet, seconds = _score(renderer, scene_glb(scene), scored, regions,
                                                    photos, size, sources)
            entry["render_s"] = round(seconds, 2)
            for row in fold_rows:
                row["fold"] = i
            rows += fold_rows
            sheet += fold_sheet
    rows.sort(key=lambda r: r["frame"])
    sheet.sort(key=lambda r: r["frame"])
    built = {r["frame"]: r for r in as_built}
    same = [built[r["frame"]] for r in rows if r["frame"] in built]
    summary, in_sample = _summarise(rows), _summarise(same)
    gap = {}
    for key, entry in summary.items():
        other = in_sample.get(key)
        if isinstance(entry, dict) and isinstance(other, dict):
            gap[key] = round(other["mean"] - entry["mean"], 4)
    budget = [f["fold"] for f in info if f.get("textures_note", "").startswith("time budget")]
    return {"k": len(folds), "split": split, "views": len(rows), "options": options,
            "rebuilt": "the quick room, rebuilt by this checkout with the scene's own options, "
                       "from the capture minus the fold's frames",
            "folds": info, "budget_hit_folds": budget,
            "summary": summary, "as_built_same_views": in_sample, "gap": gap,
            "per_view": rows, "seconds": round(time.perf_counter() - started, 2)}, sheet


def _fold_build(scene: Path) -> dict:
    """What a fold's rebuild says it used (its quick.json)."""
    quick = held._read_json(Path(scene).parent / held.QUICK_REPORT) or {}
    tex, col = quick.get("textures") or {}, quick.get("colours") or {}
    layout = quick.get("layout") or {}
    out = {"textures_frames": [tex.get("frames_used"), tex.get("frames_total")],
           "textures_note": tex.get("note") or "", "colours_frames": col.get("frames_used")}
    # The layout repair has a deadline of its own: a fold that hit it is laid out differently.
    if layout.get("error") or layout.get("skipped"):
        out["layout_note"] = str(layout.get("error") or layout.get("skipped"))
    return out


_NOT_METRICS = frozenset({"frame", "fold", "mask", "seen", "read_by"})


def _seen_counts(rows: list[dict]) -> dict:
    seen = [r.get("seen") for r in rows]
    return {"seen": seen.count(True), "unseen": seen.count(False), "unknown": seen.count(None)}


def _splits(rows: list[dict]) -> dict:
    """The as-built numbers over the views the build read and over those it did not (None for a
    split with no views)."""
    out = {}
    for name, flag in (("seen", True), ("unseen", False)):
        chosen = [r for r in rows if r.get("seen") is flag]
        out[name] = _summarise(chosen) if chosen else None
    return out


def _summarise(rows: list[dict]) -> dict:
    """Mean and median over views of every metric (views where a region was too small to judge
    are left out of that region's numbers, and the count says how many remained)."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in _NOT_METRICS and not k.startswith("frac_")]
    out: dict[str, Any] = {}
    for key in keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))
                  and not isinstance(r.get(key), bool)]
        if not values:
            out[key] = None
            continue
        out[key] = {"mean": round(float(np.mean(values)), 4),
                    "median": round(float(np.median(values)), 4), "n": len(values)}
    return out


DEFINITIONS = {
    "views": "every capture frame with a pose and a photo (frame_*.json + frame_*.jpg)",
    "render": "geometry.build_scene_assets glb drawn by three.js 0.168 in headless Chromium "
              "(SwiftShader) with the room call's lights, tone mapping and colour space, from "
              "the frame's own pose and intrinsics, at 2x and box-filtered to the metric size, "
              "over the call's background colour; bodies at their authored (scanned) pose, as "
              "the call shows them before its first state frame (the settle moves them by "
              "millimetres; see the physics audit's settle_shift)",
    "psnr": "10 log10(1 / MSE) over RGB in [0, 1], photo vs render, both at the metric size",
    "ssim": "mean SSIM on luma (Gaussian window sigma 1.5, K1 0.01, K2 0.03)",
    "psnr_gain": "PSNR after scaling the render by its least-squares gain to the photo "
                 "(removes the camera's per-frame exposure)",
    "psnr_baseline": "PSNR of the photo against a flat image of its own mean colour",
    "objects": "pixels whose LiDAR point (the frame's own depth map, medium or high confidence) "
               "lies within 3 cm of a RoomPlan object box and not within 3 cm of the floor or "
               "ceiling or 5 cm of a wall plane: the object as the camera saw it, occlusion "
               "included; where a pixel has no depth, inside a projected box silhouette "
               "(object_mask counts the views that had depth). Furniture RoomPlan did not box "
               "counts as static",
    "static": "every pixel that is no object and no door/window (a door/window box's "
              "silhouette, unless the depth puts something in front of it): walls, floor, "
              "ceiling",
    "coverage": "fraction of the render the room drew (the rest is background: outside the "
                "room, or a surface the stage lacks)",
    "seen": "whether the stage's build read that frame (quick room: the texture and colour "
            "frames its deterministic selectors pick; objects stage: also its crop views); "
            "None when the build kept no record (the full pipeline)",
    "splits": "the as-built numbers over the views the build read (seen) and those it did not",
    "holdout": "k-fold held-out views (quick room): the capture's photo frames in k folds "
               "(interleaved: every k-th frame; block: contiguous stretches of the trajectory); "
               "per fold the quick room is rebuilt with the same options from the capture with "
               "that fold's frames removed, and only that fold's views are scored against it; "
               "gap = as-built mean - held-out mean over the same views",
}


__all__ = ["DEFINITIONS", "SIZE", "Box", "PlaywrightRenderer", "Renderer",
           "RendererUnavailable", "View", "box_hits", "box_mask", "camera_pose", "capture_views",
           "compare", "composite", "contact_sheet", "evaluate_fidelity", "intrinsics_at",
           "load_photo", "project", "project_three", "projection_matrix", "psnr", "scan_boxes",
           "scene_glb", "serve_page", "ssim", "ssim_map", "three_camera", "view_depth",
           "view_regions"]
