"""LHM++ in the GPU container: a few photos of one person -> a skinned 3D Gaussian body.

Runs inside the Modal function `litereality-avatar` / `avatar` (deploy/modal/avatar/app.py), which
packs what `build` returns into the room call's avatar file (room_ops/roomcall/avatar_format.py).
This module sits in `models`, below `room_ops`, so it must not import it: it hands back plain numpy
fields and the deploy app (outside the layers) turns them into an `Avatar`.

    photos -> decode -> segment the person (BiRefNet) -> drop views with nobody in them
           -> frame each view like LHM++'s own inputs (white background, 840 x 504, 5:3)
           -> LHM++ 700M SMPLX-FREE (pose-free views in, SMPL-X shape predicted)
           -> Gaussians re-posed to the SMPL-X ZERO pose with per-Gaussian skinning weights
           -> prune, cap, keep the top 4 weights, and measure what the top-4 cut costs

The frame is SMPL-X's canonical one: metres, +Y up, the person faces +Z, +X is the person's LEFT,
zero pose (a T-pose with flat hands). Weights are over SMPL-X's 55 joints, about the zero-pose
joints of the predicted body shape, exactly what LHM++ itself animates with, so a browser that
skins the splats with them (Spark's SplatSkinning) moves them the way LHM++ would.

Everything above `LHMRunner` is numpy + Pillow and runs anywhere; tests/test_avatar_gpu.py drives
`build` with a numpy fake runner. `LHMRunner` is the only code that touches torch and LHM++
(github.com/aigc3d/LHM-plusplus: code Apache-2.0, weights CC-BY-NC-4.0, SMPL-X non-commercial);
it is written from LHM++'s source at the pinned commit and is UNVERIFIED until the first Modal run.
Python 3.10 / numpy 1.23 compatible: LHM++ pins both, and this module runs in that image.
"""

from __future__ import annotations

import io
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CANVAS_H, CANVAS_W = 840, 504          # LHM++ input, 5:3, both multiples of 14
MIN_PERSON_FRACTION = 0.015            # a view whose mask covers less than this is dropped
OPACITY_MIN = 0.02
MAX_SPLATS = 120_000                   # == avatar_format.MAX_SPLATS (a test checks it)
# joint -> rotation vector: arms down, elbows bent, knees bent, for skin_error. A pose that bends
# the joints people actually move, so the error is measured where the top-4 cut could show.
TEST_POSE = {16: (0, 0, -1.2), 17: (0, 0, 1.2), 18: (0, -0.6, 0), 19: (0, 0.6, 0),
             4: (0.5, 0, 0), 5: (0.5, 0, 0)}
MODES = ("turnaround", "photo")
MODEL_NAME = "LHMPP-700M-SMPLX-FREE"
JOINTS = 55                            # SMPL-X: 22 body + jaw and eyes + 30 hand joints
MAX_SIDE = 2048                        # decode_image's long side
MAX_DECODE_PIXELS = 40_000_000         # what decode_image will open at all (header size)
PAD_FRACTION = 0.03                    # frame_view's margin, of the person's height
# frame_view: the visible part of a person cut off by the photo's bottom edge spans at most this
# share of the canvas height. LHM++ trains on cut views that keep the full-body scale (the top
# 60-100 % of a full-body frame, white above: UpRandomCrop(0.6, 1.1) after PadRatioWithScale), and
# the room's capture guide asks for head to knees, about 0.72 of a person's height.
CUT_FILL = 0.75
# LHM++'s own canonical render camera for a synthetic frame (scripts/inference/to_gs_ply.py):
# render_size from the model's config, 5:3 portrait.
RENDER_SIZE = 420
BIREFNET_WEIGHTS = "BiRefNet-general-epoch_244.pth"   # in LHM++'s prior bundle
BIREFNET_SIZE = 1024
LEFT_WRIST, RIGHT_WRIST = 20, 21


class PhotoError(ValueError):
    """The photos cannot make a body: an image that does not decode, nobody in them, a model
    output with nothing visible. The person can fix it, so the message goes back to them. Every
    other exception, other ValueErrors included, is a fault: logged, answered "internal: <type>"."""


@dataclass
class RawBody:
    """The model's own Gaussians, posed to the SMPL-X ZERO pose, as numpy (see the module doc)."""

    xyz: np.ndarray            # (N, 3) metres, SMPL-X frame
    rot_wxyz: np.ndarray       # (N, 4) unit quaternion, w first (pytorch3d / gsplat order)
    scale: np.ndarray          # (N, 3) linear standard deviations, metres
    opacity: np.ndarray        # (N,) 0..1
    rgb: np.ndarray            # (N, 3) 0..1 display colour
    weights: np.ndarray        # (N, J) full skinning weights, J = 55
    joints: np.ndarray         # (J, 3) zero-pose joints for the predicted betas
    parents: list[int]
    names: list[str]
    betas: np.ndarray          # (10,)


# ── images ───────────────────────────────────────────────────────────────────────────────────────


def decode_image(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB uint8 (H, W, 3), EXIF-transposed, long side <= MAX_SIDE.

    EXIF first because phones store portrait photos as landscape plus an orientation tag, and a
    person lying on their side is no person to LHM++. The pixel count is checked from the header
    before anything is decoded, and JPEGs are decoded straight at a reduced scale (`draft`), so a
    12 MP phone photo costs a fraction of a full decode."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_DECODE_PIXELS:
                raise PhotoError(f"an image is {im.width}x{im.height}: too many pixels")
            im.draft("RGB", (MAX_SIDE, MAX_SIDE))
            upright = ImageOps.exif_transpose(im)
            if upright.mode in ("RGBA", "LA", "PA") or "transparency" in upright.info:
                # A cut-out PNG: its transparent pixels hold anything (often black); LHM++ was
                # trained on people over white, so that is what they become.
                rgba = upright.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                rgb = Image.alpha_composite(white, rgba).convert("RGB")
            else:
                rgb = upright.convert("RGB")
    except PhotoError:
        raise
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        # Pillow's errors for a broken or hostile file, ValueError included: the file's fault.
        raise PhotoError("an image could not be decoded") from None
    rgb.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return np.asarray(rgb, dtype=np.uint8).copy()


def person_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1) of mask > 0.5, half-open (y1 and x1 are one past the last row/column),
    or None when the mask is empty."""
    on = np.asarray(mask) > 0.5
    if on.ndim != 2 or not on.any():
        return None
    rows = np.flatnonzero(on.any(axis=1))
    cols = np.flatnonzero(on.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _crop_white(image: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """image[y0:y1, x0:x1], with white wherever the box runs past the image."""
    h, w = image.shape[:2]
    out = np.ones((y1 - y0, x1 - x0, image.shape[2]), dtype=np.float32)
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, h), max(x0, 0), min(x1, w)
    if sy0 < sy1 and sx0 < sx1:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Float RGB in [0, 1] resized with Lanczos (through 8 bits: the photo had no more)."""
    u8 = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    out = Image.fromarray(u8).resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(out, dtype=np.float32) / 255.0


def frame_view(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """One photo framed the way LHM++ was trained to see people: (840, 504, 3) float32 in [0, 1].

    The person is composited onto WHITE with the (soft) mask, cropped to `person_box` padded by
    PAD_FRACTION of its height, resized keeping its aspect to fit 840 x 504 and padded with white,
    centred horizontally. Vertically it is centred, EXCEPT when the mask touches the image's
    bottom row: a person cut off at the legs is bottom-aligned with no margin below and spans at
    most CUT_FILL of the height, white above, as in LHM++'s training crops (cut views there keep
    the full-body scale). So the model reads the cut as "the frame ends here", not as feet, and
    sees the upper body at about the size it learned, not blown up to fill the canvas."""
    rgb = np.asarray(rgb)
    mask = np.asarray(mask, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("frame_view needs an (H, W, 3) image and an (H, W) mask")
    alpha = np.clip(mask, 0.0, 1.0)[..., None]
    composite = rgb.astype(np.float32) / 255.0 * alpha + (1.0 - alpha)
    h, w = mask.shape
    box = person_box(mask) or (0, h, 0, w)
    y0, y1, x0, x1 = box
    pad = int(round(PAD_FRACTION * (y1 - y0)))
    cut = bool((mask[-1] > 0.5).any())
    y0, x0, x1 = y0 - pad, x0 - pad, x1 + pad
    y1 = h if cut else y1 + pad
    crop = _crop_white(composite, y0, y1, x0, x1)
    ch, cw = crop.shape[:2]
    fit_h = CANVAS_H * CUT_FILL if cut else CANVAS_H
    scale = min(fit_h / ch, CANVAS_W / cw)
    nh = min(CANVAS_H, max(1, int(round(ch * scale))))
    nw = min(CANVAS_W, max(1, int(round(cw * scale))))
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.float32)
    top = CANVAS_H - nh if cut else (CANVAS_H - nh) // 2
    left = (CANVAS_W - nw) // 2
    canvas[top:top + nh, left:left + nw] = _resize(crop, nh, nw)
    return canvas


def select_views(views: list[np.ndarray], masks: list[np.ndarray], mode: str) -> list[int]:
    """Indices of the views to keep, in the given order.

    A view whose mask covers less than MIN_PERSON_FRACTION of the image has nobody in it (the
    person turned out of frame, a blurred frame) and only confuses the model. Both modes keep the
    given order: the turnaround capture starts facing the camera, so frame 0 is the frontal view,
    and an uploader's first photo is the one they chose first."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if len(views) != len(masks):
        raise ValueError("one mask per view")
    keep = []
    for i, (view, mask) in enumerate(zip(views, masks)):
        mask = np.asarray(mask)
        if mask.shape != np.asarray(view).shape[:2]:
            raise ValueError("a mask does not match its view")
        if float((mask > 0.5).mean()) >= MIN_PERSON_FRACTION:
            keep.append(i)
    if not keep:
        raise PhotoError("no person found in the images")
    return keep


# ── the body ─────────────────────────────────────────────────────────────────────────────────────


def _check_raw(raw: RawBody) -> int:
    """The runner's output is what this module expects; returns the splat count."""
    n = len(raw.xyz)
    j = len(raw.names)
    shapes = {"xyz": (n, 3), "rot_wxyz": (n, 4), "scale": (n, 3), "opacity": (n,),
              "rgb": (n, 3), "weights": (n, j), "joints": (j, 3)}
    for name, shape in shapes.items():
        if np.shape(getattr(raw, name)) != shape:
            raise ValueError(f"RawBody.{name} is {np.shape(getattr(raw, name))}, need {shape}")
    if len(raw.parents) != j or n == 0:
        raise ValueError("RawBody needs splats and one parent per joint")
    return n


def body_fields(raw: RawBody, *, max_splats: int = MAX_SPLATS) -> dict:
    """The kept splats in the avatar file's units and orders, as plain numpy.

    - splats under OPACITY_MIN are dropped (LHM++ keeps every one of its 160k points, many nearly
      transparent); if more than `max_splats` are left, the most opaque are kept, in their
      original order (stable, so the result does not depend on ties);
    - scales are clipped to [1e-4, 0.5] m, rotations go from w,x,y,z to three.js's x,y,z,w;
    - colours become u8 sRGB plus alpha = opacity * 255;
    - each splat keeps its 4 heaviest joints, weights clipped at 0 and renormalised to sum to 1
      (`skin_joints` (N, 4) int64, `skin_weights` (N, 4) float64); the file quantises them;
    - floor_y is the 0.2nd percentile of y over the splats with opacity >= 0.5: the soles,
      ignoring the odd stray splat below them.

    Also returns `kept`, the kept splats' indices into `raw`, which `skin_error` needs."""
    _check_raw(raw)
    if max_splats < 1:
        raise ValueError("max_splats must be >= 1")
    opacity = np.asarray(raw.opacity, dtype=np.float64)
    xyz = np.asarray(raw.xyz, dtype=np.float64)
    ok = (opacity >= OPACITY_MIN) & np.all(np.isfinite(xyz), axis=1) & np.isfinite(opacity)
    kept = np.flatnonzero(ok)
    if kept.size == 0:
        raise PhotoError("the model produced no visible splats")
    if kept.size > max_splats:
        order = np.argsort(-opacity[kept], kind="stable")[:max_splats]
        kept = np.sort(kept[order])

    positions = xyz[kept].astype(np.float32)
    scales = np.clip(np.nan_to_num(np.asarray(raw.scale, dtype=np.float64)[kept], nan=1e-4),
                     1e-4, 0.5).astype(np.float32)
    wxyz = np.asarray(raw.rot_wxyz, dtype=np.float64)[kept]
    xyzw = wxyz[:, [1, 2, 3, 0]]
    norm = np.linalg.norm(xyzw, axis=1, keepdims=True)
    bad = ~np.isfinite(norm[:, 0]) | (norm[:, 0] < 1e-8)
    xyzw[bad] = (0.0, 0.0, 0.0, 1.0)
    norm[bad] = 1.0
    rotations = (xyzw / norm).astype(np.float32)
    rgb = np.clip(np.nan_to_num(np.asarray(raw.rgb, dtype=np.float64)[kept]), 0.0, 1.0)
    alpha = np.clip(opacity[kept], 0.0, 1.0)[:, None]
    colors = np.round(np.concatenate([rgb, alpha], axis=1) * 255.0).astype(np.uint8)

    weights = np.clip(np.nan_to_num(np.asarray(raw.weights, dtype=np.float64)[kept]), 0.0, None)
    top = np.argsort(-weights, axis=1, kind="stable")[:, :4]
    top_w = np.take_along_axis(weights, top, axis=1)
    if top.shape[1] < 4:                      # fewer than four joints: pad with weight 0
        pad = 4 - top.shape[1]
        top = np.concatenate([top, np.zeros((len(top), pad), np.int64)], axis=1)
        top_w = np.concatenate([top_w, np.zeros((len(top_w), pad))], axis=1)
    total = top_w.sum(axis=1, keepdims=True)
    empty = total[:, 0] <= 0
    top_w[empty] = (1.0, 0.0, 0.0, 0.0)
    total[empty] = 1.0
    skin_weights = top_w / total

    solid = opacity[kept] >= 0.5
    soles = positions[solid, 1] if solid.any() else positions[:, 1]
    floor_y = float(np.percentile(soles, 0.2))
    return {
        "positions": positions, "scales": scales, "rotations": rotations, "colors": colors,
        "skin_joints": top.astype(np.int64), "skin_weights": skin_weights,
        "joint_names": [str(name) for name in raw.names],
        "parents": [int(p) for p in raw.parents],
        "rest": np.asarray(raw.joints, dtype=np.float32), "floor_y": floor_y,
        "kept": kept,
    }


def _rotation(rotvec) -> np.ndarray:
    """Rotation vector (axis * angle) -> 3x3 matrix (Rodrigues)."""
    v = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)


def _skin_transforms(joints: np.ndarray, parents, pose: dict) -> np.ndarray:
    """(J, 3, 4) skin transforms S_j(x) = G_j.R (x - rest_j) + G_j.p of a pose given as local
    rotation vectors, every rest rotation the identity (forward kinematics about the rest joints,
    the same convention as the browser's poseSkeleton and SMPL-X's own LBS)."""
    rest = np.asarray(joints, dtype=np.float64)
    count = len(rest)
    world_r = np.zeros((count, 3, 3))
    world_p = np.zeros((count, 3))
    out = np.zeros((count, 3, 4))
    for j in range(count):
        local = _rotation(pose.get(j, (0.0, 0.0, 0.0)))
        parent = int(parents[j])
        if parent < 0:
            world_r[j], world_p[j] = local, rest[j]
        else:
            if not 0 <= parent < j:
                raise ValueError("parents must come before their children")
            world_r[j] = world_r[parent] @ local
            world_p[j] = world_p[parent] + world_r[parent] @ (rest[j] - rest[parent])
        out[j, :, :3] = world_r[j]
        out[j, :, 3] = world_p[j] - world_r[j] @ rest[j]
    return out


def lbs(points, weights, joints, parents, pose: dict[int, tuple]) -> np.ndarray:
    """Linear blend skinning of rest-pose `points` (N, 3) with `weights` (N, J) to `pose`
    (joint -> rotation vector, missing joints at rest). Each point's weights are renormalised
    to sum to 1 (a point with none is bound to the root), so a cut-down weight set and the full
    one are compared on equal terms."""
    pts = np.asarray(points, dtype=np.float64)
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    total = w.sum(axis=1, keepdims=True)
    empty = total[:, 0] <= 0
    w[empty, 0] = 1.0
    total[empty] = 1.0
    w = w / total
    transforms = _skin_transforms(joints, parents, pose)
    blended = (w @ transforms.reshape(len(transforms), 12)).reshape(-1, 3, 4)
    return np.einsum("nij,nj->ni", blended[:, :, :3], pts) + blended[:, :, 3]


def skin_error(raw: RawBody, fields: dict, pose=TEST_POSE, *, skin=None) -> dict:
    """What the cut-down skin costs, on the kept splats at `pose`:

    {"skin_err_mm": {"median", "p99", "max"}  - LBS with LHM++'s full weights vs LBS with the
                                                cut-down ones, in millimetres,
     "top4_coverage": {"mean", "min"}}        - the share of each splat's weight the top 4 hold.

    `skin` is the cut-down skin to measure, (joints (N, 4), weights (N, 4)) per kept splat: the
    deploy app passes the avatar file's 8-bit weights read back, what browsers actually get.
    Without it, the float top-4 weights in `fields`. Either way it is a linear-blend ESTIMATE:
    browsers skin with Spark's dual quaternions, which part from linear blending at strongly bent
    joints, so it compares bodies and builds; it is not a measurement of what people see."""
    kept = np.asarray(fields["kept"])
    points = np.asarray(raw.xyz, dtype=np.float64)[kept]
    full = np.clip(np.asarray(raw.weights, dtype=np.float64)[kept], 0.0, None)
    top_joints = np.asarray(fields["skin_joints"])
    joints4, weights4 = (top_joints, fields["skin_weights"]) if skin is None else skin
    joints4 = np.asarray(joints4, dtype=np.int64)
    weights4 = np.asarray(weights4, dtype=np.float64)
    if joints4.shape != weights4.shape or len(joints4) != len(kept):
        raise ValueError("skin must be (joints, weights), one row per kept splat")
    dense4 = np.zeros_like(full)
    np.add.at(dense4, (np.arange(len(dense4))[:, None], joints4), weights4)
    a = lbs(points, full, raw.joints, raw.parents, pose)
    b = lbs(points, dense4, raw.joints, raw.parents, pose)
    err = np.linalg.norm(a - b, axis=1) * 1000.0
    total = full.sum(axis=1)
    held = np.take_along_axis(full, top_joints, axis=1).sum(axis=1)
    coverage = np.where(total > 0, held / np.where(total > 0, total, 1.0), 1.0)
    return {
        "skin_err_mm": {"median": round(float(np.median(err)), 3),
                        "p99": round(float(np.percentile(err, 99)), 3),
                        "max": round(float(err.max()), 3)},
        "top4_coverage": {"mean": round(float(coverage.mean()), 4),
                          "min": round(float(coverage.min()), 4)},
    }


def check_canonical_frame(joints: np.ndarray) -> None:
    """Raise unless the joints are in the frame the avatar file promises (+X the person's left,
    +Y up): in the T-pose the left wrist is at +x, the right at -x, and the head above the pelvis.
    A model build that ever handed back a mirrored or camera-space body fails loudly here instead
    of putting everyone's left arm on their right."""
    j = np.asarray(joints, dtype=np.float64)
    if not (j[LEFT_WRIST, 0] > 0 > j[RIGHT_WRIST, 0] and j[15, 1] > j[0, 1]):
        raise RuntimeError("the body is not in the SMPL-X canonical frame "
                           f"(L_Wrist {j[LEFT_WRIST].round(3).tolist()}, "
                           f"R_Wrist {j[RIGHT_WRIST].round(3).tolist()})")


def build(runner, images: list[bytes], mode: str, *, skin_codec=None) -> tuple[dict, dict]:
    """Photos -> (`body_fields`, stats) with any `runner` that has `segment(rgb) -> mask` and
    `run(views) -> RawBody` (LHMRunner in the container, a numpy fake in the tests).

    `skin_codec(skin_joints, skin_weights) -> (joints, weights)` is the skin as the file will
    store it (the deploy app passes the avatar format's 8-bit round trip), so `skin_err_mm`
    measures what is shipped; None measures the float top-4 weights.

    PhotoError means the photos cannot make a body ("no person found in the images", an image
    that does not decode); anything else is a fault. Stats are plain JSON for the file's meta."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if not images:
        raise PhotoError("no images")
    started = time.monotonic()
    pictures = [decode_image(data) for data in images]
    masks = []
    for rgb in pictures:
        mask = np.asarray(runner.segment(rgb), dtype=np.float32)
        if mask.shape != rgb.shape[:2]:
            raise RuntimeError(f"segment returned a {mask.shape} mask for a {rgb.shape} image")
        masks.append(mask)
    keep = select_views(pictures, masks, mode)
    views = np.stack([frame_view(pictures[i], masks[i]) for i in keep]).astype(np.float32)
    framed = time.monotonic()
    raw = runner.run(views)
    inferred = time.monotonic()
    fields = body_fields(raw)
    shipped = None
    if skin_codec is not None:
        shipped = skin_codec(fields["skin_joints"], fields["skin_weights"])
    quality = skin_error(raw, fields, skin=shipped)
    exported = time.monotonic()
    y = fields["positions"][:, 1]
    stats = {
        "views": len(images),
        "views_used": len(keep),
        "splats": int(len(fields["positions"])),
        "pruned": int(len(raw.xyz) - len(fields["positions"])),
        "height_m": round(float(np.percentile(y, 99.8)) - fields["floor_y"], 3),
        **quality,
        "skin_err_basis": ("linear blend, LHM++'s full weights vs the "
                           + ("file's 8-bit top 4" if shipped is not None else "float top 4")),
        "timing": {"segment_s": round(framed - started, 2),
                   "infer_s": round(inferred - framed, 2),
                   "export_s": round(exported - inferred, 2)},
        "model": MODEL_NAME,
    }
    return fields, stats


# ── LHM++ (torch; only in the container) ─────────────────────────────────────────────────────────


class LHMRunner:
    """LHM++ 700M SMPLX-FREE and BiRefNet, loaded once per container. UNVERIFIED until the first
    Modal run: written against github.com/aigc3d/LHM-plusplus at the commit the deploy app pins
    (scripts/inference/to_gs_ply.py's path, minus its downloads and pip installs).

    The container must have `repo_dir` as the working directory and on sys.path, with
    `repo_dir/pretrained_models` holding LHM++'s prior bundle: LHM++ opens SMPL-X, the voxel
    skinning grid and its sample points by paths relative to the working directory."""

    def __init__(self, repo_dir: Path, model_dir: Path, device: str = "cuda") -> None:
        import torch

        # LHM++ decorates forward_latent_points with @torch.compile; compiling would add minutes
        # to every cold start for a one-shot call, and its own export script switches it off.
        torch._dynamo.config.disable = True
        repo_dir = Path(repo_dir)
        if Path.cwd().resolve() != repo_dir.resolve():
            raise RuntimeError(f"LHMRunner must run from {repo_dir} (LHM++ uses relative paths)")
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        # LHM++ logs through accelerate.logging, whose every call (even a disabled INFO one)
        # raises RuntimeError until accelerate's process state exists; each LHM++ entry point
        # makes an Accelerator first. The model's constructor logs ("Using DINOv2 as the
        # encoder"), so this comes before anything of LHM++ is imported or built. One process,
        # no launcher: PartialState is the cheap form of the same state.
        from accelerate import PartialState

        PartialState()

        from core.models import model_dict
        from core.models.rendering.skinnings.constant import SMPLX_JOINTS
        from core.utils.hf_hub import wrap_model_hub
        from engine.BiRefNet.models.birefnet import BiRefNet
        from engine.BiRefNet.utils import check_state_dict
        from torchvision import transforms

        self.torch = torch
        self.device = device
        self.joint_names = list(SMPLX_JOINTS)
        model_cls = wrap_model_hub(model_dict["human_lrm_a4o"])
        self.model = model_cls.from_pretrained(str(model_dir)).to(device).eval()
        if getattr(self.model, "shape_head", None) is None:
            # Without the shape head every body would get SMPL-X's mean shape.
            raise RuntimeError(f"{model_dir} is not an SMPLX-FREE model (it has no shape head)")

        birefnet = BiRefNet(bb_pretrained=False)
        state = torch.load(repo_dir / "pretrained_models" / BIREFNET_WEIGHTS, map_location="cpu")
        birefnet.load_state_dict(check_state_dict(state))
        self.birefnet = birefnet.to(device).eval()
        self.birefnet_input = transforms.Compose([
            transforms.Resize((BIREFNET_SIZE, BIREFNET_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """BiRefNet's soft person mask, (H, W) float in [0, 1] - the same call and preprocessing
        LHM uses for its own inputs (engine/SegmentAPI: 1024x1024, ImageNet normalisation)."""
        torch = self.torch
        h, w = rgb.shape[:2]
        x = self.birefnet_input(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.birefnet(x)[-1].sigmoid()
            pred = torch.nn.functional.interpolate(pred, size=(h, w), mode="bilinear",
                                                   align_corners=False)
        return pred[0, 0].clamp(0.0, 1.0).float().cpu().numpy()

    def _synthetic_frame(self, betas_dim: int = 10, expr_dim: int = 100) -> tuple[dict, dict]:
        """One neutral SMPL-X frame and LHM++'s synthetic camera, shaped like to_gs_ply's
        `_build_synthetic_motion_seq` (batch 1, one frame). Only the betas and the camera are
        used by infer_single_view; the pose is the canonical one."""
        torch, dev = self.torch, self.device
        height, width = int(round(RENDER_SIZE * 5.0 / 3.0)), RENDER_SIZE
        focal = float(max(width, height))
        intr = torch.eye(4)
        intr[0, 0], intr[1, 1] = focal, focal
        intr[0, 2], intr[1, 2] = width / 2.0, height / 2.0

        def zeros(*shape):
            return torch.zeros(1, 1, *shape, device=dev)

        smplx = {
            "betas": torch.zeros(1, betas_dim, device=dev),
            "root_pose": zeros(3), "body_pose": zeros(21, 3), "jaw_pose": zeros(3),
            "leye_pose": zeros(3), "reye_pose": zeros(3), "lhand_pose": zeros(15, 3),
            "rhand_pose": zeros(15, 3), "trans": zeros(3), "expr": zeros(expr_dim),
            "focal": torch.tensor([[[focal, focal]]], device=dev),
            "princpt": torch.tensor([[[width / 2.0, height / 2.0]]], device=dev),
            "img_size_wh": torch.tensor([[[float(width), float(height)]]], device=dev),
        }
        camera = {
            "render_c2ws": torch.eye(4, device=dev)[None, None],
            "render_intrs": intr.to(dev)[None, None],
            "render_bg_colors": torch.ones(1, 1, 3, device=dev),
        }
        return smplx, camera

    def run(self, views: np.ndarray) -> RawBody:
        """(V, 840, 504, 3) float32 views -> the body in the SMPL-X zero pose (see RawBody)."""
        torch, dev = self.torch, self.device
        views = np.ascontiguousarray(views, dtype=np.float32)
        if views.ndim != 4 or views.shape[1:] != (CANVAS_H, CANVAS_W, 3):
            raise ValueError(f"views must be (V, {CANVAS_H}, {CANVAS_W}, 3)")
        model, renderer = self.model, self.model.renderer
        with torch.no_grad():
            images = torch.from_numpy(views).permute(0, 3, 1, 2).float().to(dev)[None]
            smplx, camera = self._synthetic_frame()
            outputs = model.infer_single_view(
                images, None, None,
                render_c2ws=camera["render_c2ws"], render_intrs=camera["render_intrs"],
                render_bg_colors=camera["render_bg_colors"], smplx_params=smplx,
                ref_imgs_bool=torch.ones(1, views.shape[0], dtype=torch.bool, device=dev),
                return_pred_shape=True, use_pred_shape_for_render=True,
            )
            if len(outputs) != 8:
                raise RuntimeError(f"infer_single_view returned {len(outputs)} values, need 8")
            gs_attr_list, query_points, neutral_to_zero, _feats, _img, _motion, _pos, shape = (
                outputs)
            # 2. betas: the shape head's prediction over the (zero) placeholder betas.
            betas = type(model).smplx_params_with_pred_shape_betas(smplx, shape)["betas"]

            # 3. The Gaussians re-posed to the ALL-ZERO SMPL-X pose (not LHM++'s canonical
            # "cano" template, which spreads the legs and tilts the hands): one posed view of
            # animate_gs_model, as to_gs_ply does for a given pose.
            zero = {key: torch.zeros_like(value) for key, value in smplx.items()
                    if key in ("root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
                               "lhand_pose", "rhand_pose", "trans", "expr")}
            frame = {**smplx, **zero, "betas": betas,
                     "transform_mat_neutral_pose": neutral_to_zero}
            frame = renderer.get_single_view_smpl_data(frame, 0)
            frame = renderer._get_single_batch_data(frame, 0)
            attr = gs_attr_list[0]
            posed, _template = renderer.animate_gs_model(
                attr, query_points["neutral_coords"][0], frame, debug=False,
                mesh_meta=query_points["mesh_meta"])
            gs = posed[0]

            # 4. Skinning weights: what LHM++ itself blends with, i.e. the diffused voxel field
            # at the neutral-space point (+ its predicted offset), with hands and face taken from
            # the SMPL-X template (transform_to_posed_verts_from_neutral_pose's fix mask).
            skin = renderer.smplx_model
            neutral = query_points["neutral_coords"][0] + attr.offset_xyz
            weights = skin.query_voxel_skinning_weights(neutral.reshape(1, -1, 3))
            weights = weights.reshape(neutral.shape[0], -1).clone()
            fixed = (skin.is_rhand + skin.is_lhand + skin.is_face) > 0
            weights[fixed] = skin.skinning_weight.to(weights)[fixed]

            # 5. Joints of the zero pose for these betas, and SMPL-X's tree.
            joints = skin.get_zero_pose_human(shape_param=betas, device=dev, face_offset=None,
                                              joint_offset=None)[0]
            parents = [int(p) for p in skin.smplx_layer.parents.reshape(-1).tolist()]
            parents[0] = -1

            # 6. Colour: the decoder's `use_rgb` heads output sigmoid RGB in [0, 1] (what
            # gsplat rasterises directly); a degree-0 SH coefficient would need 0.5 + C0 * f_dc.
            dc = attr.shs[:, 0, :].float()
            rgb = dc if attr.use_rgb else 0.5 + 0.28209479177387814 * dc

            def cpu(tensor):
                return tensor.detach().float().cpu().numpy()

            raw = RawBody(
                xyz=cpu(gs.xyz), rot_wxyz=cpu(gs.rotation), scale=cpu(attr.scaling),
                opacity=cpu(attr.opacity).reshape(-1), rgb=np.clip(cpu(rgb), 0.0, 1.0),
                weights=cpu(weights), joints=cpu(joints), parents=parents,
                names=self.joint_names[:len(parents)], betas=cpu(betas[0]),
            )
        if len(raw.names) != JOINTS:
            raise RuntimeError(f"{len(raw.names)} joints, need SMPL-X's {JOINTS}")
        check_canonical_frame(raw.joints)
        torch.cuda.empty_cache()
        return raw
