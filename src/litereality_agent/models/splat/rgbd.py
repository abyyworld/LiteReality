"""The room as splats in seconds, straight from the scan's photos and LiDAR depth — no training.

Every depth pixel of every frame is a small flat disc (a surfel): placed where the LiDAR saw the
surface, turned to face along the surface normal, sized to the patch of surface that pixel covers,
and coloured from the photo. Overlapping discs from different frames are fused on a voxel grid:
each voxel keeps the sharpest view of it (nearest, most head-on) for its position and shape, and
the confidence-weighted mean colour of every view, which evens out the exposure changes between
frames. The result already looks like the room; the GPU trainer (`deploy/modal/splat`) then starts
from it and sharpens it live.

With dense depth (PromptDA's refinement of each LiDAR map, `models/splat/promptda.py`) the discs
come from that map instead: one per sample at the same colour resolution, but each with its own
measured depth and normal rather than four sharing one LiDAR sample — sharper silhouettes and
thin things the 256 x 192 sensor smears. The dense map only refines the sensor: a dense sample
that agrees with none of the confident LiDAR samples around it is dropped (`dense_mask`).

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
# Dense depth refines the LiDAR, never overrules it: where the sensor measured confidently, a
# dense sample farther than this (metres, or a fraction of the depth) from every confident LiDAR
# sample in the 3 x 3 under it is dropped.
AGREE_ABSOLUTE, AGREE_RELATIVE = 0.05, 0.05
# A dense map's normals are finite differences this many samples apart: one-sample steps on a
# millimetre-quantised map are noisy at 2 m (1 mm of a ~5 mm step), two halve that.
DENSE_NORMAL_STEP = 2
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


def read_lidar(frame: Frame) -> tuple[np.ndarray, np.ndarray | None]:
    """The frame's LiDAR depth in metres (float32) and its confidence map (None when the scan has
    none, or one of another size)."""
    from PIL import Image

    with Image.open(frame.depth) as im:
        depth = np.asarray(im, dtype=np.float32) / 1000.0
    conf = None
    if frame.conf is not None:
        with Image.open(frame.conf) as im:
            c = np.asarray(im)
        conf = c if c.shape == depth.shape else None
    return depth, conf


def off_silhouettes(depth: np.ndarray, radius: int = 1) -> np.ndarray:
    """True where a depth map's (2 radius + 1)^2 neighbourhood spans less than an edge does: the
    samples that lie on one surface rather than between an object and what is behind it."""
    h, w = depth.shape
    k = 2 * radius + 1
    pad = np.pad(depth, radius, mode="edge")
    # A square window's min / max is the column-wise one of the row-wise one: 2k passes, not k^2.
    lo_rows, hi_rows = pad[:, :w].copy(), pad[:, :w].copy()
    for dx in range(1, k):
        np.minimum(lo_rows, pad[:, dx:dx + w], out=lo_rows)
        np.maximum(hi_rows, pad[:, dx:dx + w], out=hi_rows)
    lo, hi = lo_rows[:h].copy(), hi_rows[:h].copy()
    for dy in range(1, k):
        np.minimum(lo, lo_rows[dy:dy + h], out=lo)
        np.maximum(hi, hi_rows[dy:dy + h], out=hi)
    return (hi - lo) < EDGE_RELATIVE * depth + EDGE_ABSOLUTE


def resample_nearest(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """`grid` at `shape` (rows, columns), each output pixel taking the input pixel its centre falls
    in: never blended across an edge."""
    h, w = grid.shape[:2]
    if (h, w) == tuple(shape):
        return grid
    rows = np.minimum(((np.arange(shape[0]) + 0.5) * h / shape[0]).astype(int), h - 1)
    cols = np.minimum(((np.arange(shape[1]) + 0.5) * w / shape[1]).astype(int), w - 1)
    return grid[rows[:, None], cols[None, :]]


def dense_mask(dense: np.ndarray, lidar: np.ndarray, conf: np.ndarray | None, *,
               max_depth: float = MAX_DEPTH, min_confidence: int = MIN_CONFIDENCE,
               radius: int = 1) -> np.ndarray:
    """Which samples of a dense depth map (metres, any resolution) to trust: in range, off its own
    silhouettes, and within max(AGREE_ABSOLUTE, AGREE_RELATIVE x depth) of one of the confident
    LiDAR samples in the 3 x 3 under it. On a flat patch that is the LiDAR pixel itself; on an
    object's silhouette, where the 256 x 192 sensor straddles two surfaces, the dense sample may
    pick either side (the sharp edge it adds) but never a depth the sensor saw on neither. Where
    no confident LiDAR sample is near (low confidence, out of range) the dense map stands alone:
    that is where it adds most."""
    valid = np.isfinite(dense) & (dense > MIN_DEPTH) & (dense < max_depth)
    valid &= off_silhouettes(np.where(np.isfinite(dense), dense, 0.0), radius)
    sure = np.isfinite(lidar) & (lidar > MIN_DEPTH) & (lidar < max_depth)
    if conf is not None:
        sure &= conf >= min_confidence
    h, w = lidar.shape
    H, W = dense.shape
    # Each dense pixel's LiDAR pixel: the one its centre falls in.
    rows = np.minimum(((np.arange(H) + 0.5) * h / H).astype(np.intp), h - 1)
    cols = np.minimum(((np.arange(W) + 0.5) * w / W).astype(np.intp), w - 1)
    depth = np.where(sure, lidar, 0.0).astype(np.float32)
    under = depth.take(rows, 0).take(cols, 1)
    agree = sure.take(rows, 0).take(cols, 1)
    agree &= np.abs(dense - under) <= np.maximum(AGREE_ABSOLUTE, AGREE_RELATIVE * under)
    # Most samples agree with the LiDAR pixel under them; only the rest look at its neighbours
    # (a silhouette, an unsure pixel, a disagreement), which keeps a 960 x 720 map to ~20 ms.
    yy, xx = np.nonzero(valid & ~agree)
    if len(yy):
        z = dense[yy, xx]
        # Flat indices into the zero-padded maps (a neighbour off the map is never sure).
        pad_depth = np.pad(depth, 1).ravel()
        pad_sure = np.pad(sure, 1).ravel()
        at = (rows[yy] + 1) * (w + 2) + cols[xx] + 1
        near = np.zeros(len(yy), bool)
        hit = np.zeros(len(yy), bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                k = at + dy * (w + 2) + dx
                s_n, d_n = pad_sure[k], pad_depth[k]
                near |= s_n
                hit |= s_n & (np.abs(z - d_n) <= np.maximum(AGREE_ABSOLUTE, AGREE_RELATIVE * d_n))
        agree[yy, xx] = hit | ~near
    return valid & agree


@dataclass
class _Obs:
    """One frame's surfels, before fusion."""

    points: np.ndarray       # (n, 3) room frame
    normals: np.ndarray      # (n, 3) unit, facing the camera
    colors: np.ndarray       # (n, 3) 0..1
    radius: np.ndarray       # (n,) metres a pixel covers there
    quality: np.ndarray      # (n,) smaller is a sharper view


def _frame_surfels(frame: Frame, *, upsample: int = UPSAMPLE, max_depth: float = MAX_DEPTH,
                   min_confidence: int = MIN_CONFIDENCE,
                   dense: np.ndarray | None = None) -> _Obs | None:
    depth, conf = read_lidar(frame)
    h, w = depth.shape
    W, H = w * upsample, h * upsample
    if dense is not None:
        obs = _dense_surfels(frame, dense, depth, conf, W, H, max_depth=max_depth,
                             min_confidence=min_confidence)
        if obs is not None:
            return obs
        # Nothing of the dense map survived (it disagrees with the sensor everywhere): the LiDAR.
    valid = (depth > MIN_DEPTH) & (depth < max_depth) & np.isfinite(depth)
    if conf is not None:
        valid &= conf >= min_confidence

    # Silhouette edges: the spread of each sample's 3x3 neighbourhood.
    valid &= off_silhouettes(depth)

    rgb = _photo(frame, W, H)
    fx, fy, cx, cy = _intrinsics(frame, W, H)

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
    return _surfels(frame, cam, n_cam, rgb[vv, uu], z / fx)


def _photo(frame: Frame, W: int, H: int) -> np.ndarray:
    from PIL import Image

    with Image.open(frame.image) as im:
        return np.asarray(im.convert("RGB").resize((W, H), Image.BOX), dtype=np.float32) / 255.0


def _intrinsics(frame: Frame, W: int, H: int) -> tuple[float, float, float, float]:
    """fx, fy, cx, cy at a W x H sampling of the photo (pixel centres at +0.5)."""
    sx, sy = W / frame.size[0], H / frame.size[1]
    return (frame.K[0, 0] * sx, frame.K[1, 1] * sy,
            (frame.K[0, 2] + 0.5) * sx - 0.5, (frame.K[1, 2] + 0.5) * sy - 0.5)


def _surfels(frame: Frame, cam: np.ndarray, n_cam: np.ndarray, colors: np.ndarray,
             pixel: np.ndarray) -> _Obs | None:
    """Camera-space samples, their normals, colours and metres-per-sample -> room-frame surfels,
    normals turned to face the camera and grazing views dropped."""
    view = -cam / np.linalg.norm(cam, axis=1, keepdims=True)          # surface -> camera
    flip = np.sum(n_cam * view, axis=1) < 0
    n_cam[flip] *= -1
    cos = np.sum(n_cam * view, axis=1)
    keep = cos > GRAZING
    if not keep.any():
        return None

    R, t = frame.c2w[:3, :3], frame.c2w[:3, 3]
    return _Obs(points=(cam[keep] @ R.T + t).astype(np.float32),
                normals=(n_cam[keep] @ R.T).astype(np.float32),
                colors=colors[keep],
                radius=pixel[keep].astype(np.float32),
                quality=(pixel[keep] / cos[keep]).astype(np.float32))


def _dense_surfels(frame: Frame, dense: np.ndarray, lidar: np.ndarray, conf: np.ndarray | None,
                   W: int, H: int, *, max_depth: float, min_confidence: int) -> _Obs | None:
    """One surfel per sample of the dense map taken at W x H (the colour resolution the LiDAR path
    uses, so a room costs the same memory), each with its own depth and normal."""
    d = resample_nearest(np.asarray(dense, dtype=np.float32), (H, W))
    step = DENSE_NORMAL_STEP
    # Silhouettes over the normals' own reach: a sample `step` from an edge would difference
    # across it.
    ok = dense_mask(d, lidar, conf, max_depth=max_depth, min_confidence=min_confidence,
                    radius=step)
    vv, uu = np.nonzero(ok)
    if len(vv) == 0:
        return None
    fx, fy, cx, cy = _intrinsics(frame, W, H)

    def point(v, u):
        """Camera-space points of samples (v, u), clamped to the map like an edge pad."""
        v, u = np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)
        z = d[v, u]
        return np.stack([(u - cx) / fx * z, -(v - cy) / fy * z, -z], axis=1)

    cam = point(vv, uu)
    du = point(vv, uu + step) - point(vv, uu - step)
    dv = point(vv + step, uu) - point(vv - step, uu)
    n = np.cross(dv, du)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    return _surfels(frame, cam, n, _photo(frame, W, H)[vv, uu], cam[:, 2] / -fx)


def _disc_rotations(normals: np.ndarray) -> np.ndarray:
    """Quaternions (x, y, z, w) turning +Z onto each normal."""
    n = normals.astype(np.float64)
    q = np.empty((len(n), 4))
    q[:, 0], q[:, 1], q[:, 2] = -n[:, 1], n[:, 0], 0.0         # cross((0,0,1), n)
    q[:, 3] = 1.0 + n[:, 2]
    flipped = q[:, 3] < 1e-6                                    # n is -Z: half a turn about X
    q[flipped] = (1.0, 0.0, 0.0, 0.0)
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def _by_voxel_sharpest_first(flat: np.ndarray, qual: np.ndarray) -> np.ndarray:
    """``np.lexsort((qual, flat))``, the same order, ~3x faster on millions of samples: one sort of
    a single int64 key (voxel, then the sample's stable quality rank) instead of lexsort's two
    passes. lexsort stays for a room so large the key would overflow."""
    n = len(flat)
    if n == 0 or int(flat.max()) >= (1 << 62) // max(n, 1):
        return np.lexsort((qual, flat))
    rank = np.empty(n, np.int64)
    rank[np.argsort(qual, kind="stable")] = np.arange(n)
    return np.argsort(flat * n + rank)


def instant_splats(frames: list[Frame], *, voxel: float = VOXEL, upsample: int = UPSAMPLE,
                   max_depth: float = MAX_DEPTH, min_confidence: int = MIN_CONFIDENCE,
                   dense: dict[int, np.ndarray] | None = None, log=None) -> Splats:
    """Fuse every frame's surfels into one splat room (see the module docstring). `dense` maps
    a frame's index to its dense depth (metres, any resolution, 0 = none); frames without one
    use their LiDAR."""
    t0 = time.perf_counter()
    dense = dense or {}
    obs = [o for f in frames
           if (o := _frame_surfels(f, upsample=upsample, max_depth=max_depth,
                                   min_confidence=min_confidence,
                                   dense=dense.get(f.index))) is not None]
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
    order = _by_voxel_sharpest_first(flat, qual)
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
            f"(median {int(np.median(views))} views each"
            + (f", dense depth on {sum(f.index in dense for f in frames)}" if dense else "")
            + f"), {time.perf_counter() - t0:.1f} s")
    return splats
