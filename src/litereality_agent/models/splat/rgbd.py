"""The room as splats in seconds, straight from the scan's photos and LiDAR depth — no training.

Every depth pixel of every frame is a small flat disc (a surfel): placed where the LiDAR saw the
surface, turned to face along the surface normal, sized to the patch of surface that pixel covers,
and coloured from the photo. Overlapping discs from different frames are fused on a voxel grid:
each voxel keeps the sharpest view of it (nearest, most head-on) for its position and shape, and
the confidence-weighted mean colour of every view, which evens out the exposure changes between
frames. The result already looks like the room; the GPU trainer (`deploy/modal/splat`) then starts
from it and sharpens it live.

Frames are the scan's own (`frame_*.json` + `.jpg`, `depth_*.png` in millimetres, `conf_*.png`
0..2), in the room's frame: ARKit poses turned Z up by ``ARKIT_TO_Z_UP``, the frame the quick
room and every MuJoCo export use.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from litereality_agent.models.splat.spz import Splats

ARKIT_TO_Z_UP = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

# The colour image is sampled at twice the depth map's size (512 x 384 for an iPhone's 256 x 192
# LiDAR): each depth sample becomes four discs with their own colour, nearest depth.
UPSAMPLE = 2
VOXEL = 0.008              # metres: the finest detail the fused room keeps
MAX_DEPTH = 4.5            # the LiDAR's useful range; farther samples are noise
MIN_DEPTH = 0.15
MIN_CONFIDENCE = 2         # ARKit's confidence map: 0 low, 1 medium, 2 high
# A depth discontinuity (an object's silhouette against the wall behind it): samples whose 3x3
# neighbourhood spans more than this (relative + absolute, metres) float in the air between the
# two surfaces, so they are dropped.
EDGE_RELATIVE, EDGE_ABSOLUTE = 0.04, 0.02
GRAZING = 0.25             # |cos| of the view angle below which a view is too oblique to trust
THICKNESS = 0.12           # a disc's thickness along its normal, as a fraction of its radius
OPACITY = 0.92


@dataclass
class Frame:
    index: int
    K: np.ndarray            # 3x3, at the photo's resolution
    c2w: np.ndarray          # 4x4 camera to room (OpenGL camera: x right, y up, looking down -z)
    image: Path
    depth: Path
    conf: Path | None
    size: tuple[int, int]    # photo (width, height)


def load_frames(scan_dir: Path | str) -> list[Frame]:
    """Every frame with a pose, a photo and a depth map, in frame order."""
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
        if not (np.isfinite(K).all() and np.isfinite(pose).all()):
            continue
        conf = meta.with_name(f"conf_{stem}.png")
        frames.append(Frame(index, K, ARKIT_TO_Z_UP @ pose, image, depth,
                            conf if conf.is_file() else None, size))
    frames.sort(key=lambda f: f.index)
    return frames


@dataclass
class _Obs:
    """One frame's surfels, before fusion."""

    points: np.ndarray       # (n, 3) room frame
    normals: np.ndarray      # (n, 3) unit, facing the camera
    colors: np.ndarray       # (n, 3) 0..1
    radius: np.ndarray       # (n,) metres a pixel covers there
    quality: np.ndarray      # (n,) smaller is a sharper view


def _frame_surfels(frame: Frame, *, upsample: int = UPSAMPLE, max_depth: float = MAX_DEPTH,
                   min_confidence: int = MIN_CONFIDENCE) -> _Obs | None:
    from PIL import Image

    with Image.open(frame.depth) as im:
        depth = np.asarray(im, dtype=np.float32) / 1000.0
    h, w = depth.shape
    conf = None
    if frame.conf is not None:
        with Image.open(frame.conf) as im:
            c = np.asarray(im)
        conf = c if c.shape == depth.shape else None
    valid = (depth > MIN_DEPTH) & (depth < max_depth) & np.isfinite(depth)
    if conf is not None:
        valid &= conf >= min_confidence

    # Silhouette edges: the spread of each sample's 3x3 neighbourhood.
    pad = np.pad(depth, 1, mode="edge")
    stack = np.stack([pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)])
    spread = stack.max(0) - stack.min(0)
    valid &= spread < EDGE_RELATIVE * depth + EDGE_ABSOLUTE

    W, H = w * upsample, h * upsample
    with Image.open(frame.image) as im:
        rgb = np.asarray(im.convert("RGB").resize((W, H), Image.BOX), dtype=np.float32) / 255.0
    # Intrinsics at the sampled resolution (pixel centres at +0.5).
    sx, sy = W / frame.size[0], H / frame.size[1]
    fx, fy = frame.K[0, 0] * sx, frame.K[1, 1] * sy
    cx, cy = (frame.K[0, 2] + 0.5) * sx - 0.5, (frame.K[1, 2] + 0.5) * sy - 0.5

    d = np.repeat(np.repeat(depth, upsample, 0), upsample, 1)
    ok = np.repeat(np.repeat(valid, upsample, 0), upsample, 1)
    vv, uu = np.nonzero(ok)
    if len(vv) == 0:
        return None
    z = d[vv, uu]
    cam = np.stack([(uu - cx) / fx * z, -(vv - cy) / fy * z, -z], axis=1)

    # Normals from the depth map's own point grid (at depth resolution), looked up per sample.
    dsx, dsy = w / frame.size[0], h / frame.size[1]
    dfx, dfy = frame.K[0, 0] * dsx, frame.K[1, 1] * dsy
    dcx, dcy = (frame.K[0, 2] + 0.5) * dsx - 0.5, (frame.K[1, 2] + 0.5) * dsy - 0.5
    gv, gu = np.mgrid[0:h, 0:w]
    grid = np.stack([(gu - dcx) / dfx * depth, -(gv - dcy) / dfy * depth, -depth], axis=-1)
    gp = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="edge")
    du = gp[1:-1, 2:] - gp[1:-1, :-2]
    dv = gp[2:, 1:-1] - gp[:-2, 1:-1]
    n_grid = np.cross(dv, du)
    n_grid /= np.maximum(np.linalg.norm(n_grid, axis=-1, keepdims=True), 1e-9)
    n_cam = n_grid[vv // upsample, uu // upsample]
    view = -cam / np.linalg.norm(cam, axis=1, keepdims=True)          # surface -> camera
    flip = np.sum(n_cam * view, axis=1) < 0
    n_cam[flip] *= -1
    cos = np.sum(n_cam * view, axis=1)
    keep = cos > GRAZING
    if not keep.any():
        return None

    R, t = frame.c2w[:3, :3], frame.c2w[:3, 3]
    pixel = z / fx                                                   # metres per sample
    return _Obs(points=(cam[keep] @ R.T + t).astype(np.float32),
                normals=(n_cam[keep] @ R.T).astype(np.float32),
                colors=rgb[vv[keep], uu[keep]],
                radius=pixel[keep].astype(np.float32),
                quality=(pixel[keep] / cos[keep]).astype(np.float32))


def _disc_rotations(normals: np.ndarray) -> np.ndarray:
    """Quaternions (x, y, z, w) turning +Z onto each normal."""
    n = normals.astype(np.float64)
    q = np.empty((len(n), 4))
    q[:, 0], q[:, 1], q[:, 2] = -n[:, 1], n[:, 0], 0.0         # cross((0,0,1), n)
    q[:, 3] = 1.0 + n[:, 2]
    flipped = q[:, 3] < 1e-6                                    # n is -Z: half a turn about X
    q[flipped] = (1.0, 0.0, 0.0, 0.0)
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def instant_splats(frames: list[Frame], *, voxel: float = VOXEL, upsample: int = UPSAMPLE,
                   max_depth: float = MAX_DEPTH, min_confidence: int = MIN_CONFIDENCE,
                   log=None) -> Splats:
    """Fuse every frame's surfels into one splat room (see the module docstring)."""
    t0 = time.perf_counter()
    obs = [o for f in frames
           if (o := _frame_surfels(f, upsample=upsample, max_depth=max_depth,
                                   min_confidence=min_confidence)) is not None]
    if not obs:
        raise ValueError("no frame has usable depth")
    pts = np.concatenate([o.points for o in obs])
    nrm = np.concatenate([o.normals for o in obs])
    col = np.concatenate([o.colors for o in obs])
    rad = np.concatenate([o.radius for o in obs])
    qual = np.concatenate([o.quality for o in obs])

    key = np.floor(pts / voxel).astype(np.int64)
    key -= key.min(0)
    span = key.max(0) + 1
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    order = np.lexsort((qual, flat))                     # by voxel, sharpest view first
    flat_sorted = flat[order]
    starts = np.flatnonzero(np.r_[True, flat_sorted[1:] != flat_sorted[:-1]])
    best = order[starts]

    weight = 1.0 / np.maximum(qual[order], 1e-4) ** 2
    wsum = np.add.reduceat(weight, starts)
    colour = np.add.reduceat(col[order] * weight[:, None], starts) / wsum[:, None]
    views = np.diff(np.r_[starts, len(order)])

    radius = np.maximum(rad[best] * 0.75, voxel * 0.6)
    scales = np.stack([radius, radius, radius * THICKNESS], axis=1).astype(np.float32)
    splats = Splats(positions=pts[best].astype(np.float32), scales=scales,
                    rotations=_disc_rotations(nrm[best]),
                    colors=np.clip(colour, 0.0, 1.0).astype(np.float32),
                    opacities=np.full(len(best), OPACITY, np.float32))
    if log:
        log(f"instant splats: {len(splats):,} from {len(pts):,} samples in {len(obs)} frames "
            f"(median {int(np.median(views))} views each), {time.perf_counter() - t0:.1f} s")
    return splats
