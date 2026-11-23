"""A scan as a training set for gsplat's own trainer (`examples/simple_trainer.py`, Apache-2.0).

The trainer reads a COLMAP model. A LiDAR scan has everything that model holds without running
COLMAP at all — ARKit's camera poses and intrinsics are already metric and registered, and the
LiDAR depth gives the 3D points — so this writes one directly, in the room's own frame:

    images/frame_00030.jpg          the photos at TRAIN_WIDTH (960 x 720 for a 1920 x 1440 scan)
    sparse/0/cameras.bin            one PINHOLE camera per photo (ARKit refocuses, so K drifts)
    sparse/0/images.bin             world-to-camera poses (OpenCV camera: x right, y down, z ahead)
                                    and each photo's own LiDAR points as its 2D observations
    sparse/0/points3D.bin           those points (confident LiDAR samples, coloured from the photo)

COLMAP's binary format: the pycolmap fork gsplat pins (rmbrualla/pycolmap@cc7ea4b) cannot read
the text one under Python 3.

Every point is seen by exactly the photo whose depth map it came from, so the trainer's depth
loss (`--depth_loss`) compares the rendered depth with the LiDAR's own measurement at that pixel,
and its "sfm" initialisation starts from the LiDAR surface rather than from noise. The photos
the trainer holds out to score the room (every `test_every`-th by name) are listed with no points
at all: their own pixels must not seed, or colour, the room they are scored against. With
``--normalize_world_space False`` the trained splats stay in the room's frame (metres, Z up),
the frame the room call draws in.

With dense depth (PromptDA's refinement of each LiDAR map, models/splat/promptda.py) a photo's
points are drawn from that map at the training resolution instead — every pixel a candidate, not
one in ~14 — keeping only samples that agree with a confident LiDAR sample under them
(`rgbd.dense_mask`), so more of them (`points_per_frame` ~20000) still mean the sensor's surface.
With ``depth_maps=True`` the trusted dense map itself is written too, for the trainer's dense
depth and normal losses (deploy/modal/splat, ``"dense_depth"``):

    depths/frame_00030.png          uint16 millimetres at half the training size, 0 = untrusted

Half size: the tar goes up to Modal before training starts, and at the full 960 x 720 the maps
were ~18 MB of office_room's (6 MB at half). The trainer resizes them back nearest-neighbour
(deploy/modal/splat/litereality_dense.load_depth); each kept pixel is a 2 x 2 block's mean where
all four were trusted.

A photo whose dense map is missing or unusable falls back to its LiDAR points and gets no depth
map (the trainer then trains it on colour and sparse depth alone).
"""

from __future__ import annotations

import gzip
import io
import struct
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from litereality_agent.models.splat.rgbd import (
    MIN_DEPTH,
    Frame,
    dense_mask,
    off_silhouettes,
    read_lidar,
    resample_nearest,
)

TRAIN_WIDTH = 960
# A run's own photo width (`litereality splatroom --train-width`), for the process; None keeps
# TRAIN_WIDTH. Only the photos change: dense depth stays at TRAIN_WIDTH (PromptDA sees at most
# 1008 px anyway, `promptda.MODEL_SIDE`) and is resampled to the photos, so its cache is shared.
train_width_override: int | None = None
POINTS_PER_FRAME = 4000
MAX_DEPTH = 5.0
MIN_CONFIDENCE = 2
MIN_FRAME_POINTS = 20         # a photo with fewer LiDAR points is left out
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
JPEG_QUALITY = 95
DEPTH_DIR = "depths"
# Point positions are written exactly, not snapped to a grid. Snapping them to 1/1024 m shrank the
# tar by ~20%, but gsplat's MCMC then went to NaN loss within 90 steps in both tea_room runs that
# tried it (the same set unsnapped trained normally): ~3 MB more upload is the cheaper side.
# zlib level for the depth maps: level 1 encodes a 960 x 720 map in a few ms and is within ~15% of
# level 6's size (the maps are smooth); level 6 would add seconds to a 300-photo scan.
DEPTH_PNG_LEVEL = 1


@dataclass
class Dataset:
    files: dict[str, bytes]          # relative path -> contents
    images: int
    points: int
    size: tuple[int, int]            # training (width, height)
    dense_images: int = 0            # photos whose points came from dense depth
    held_out: int = 0                # photos listed for the trainer to score, with no points

    def tar(self) -> bytes:
        """The files as one gzipped tar (the trainer's `tarfile.open` reads either). gzip level 1:
        the float64 COLMAP points are ~70% of it and shrink 40% (tea_room: 28 -> 17 MB, ~3 s less
        upload to Modal) for under a second of compression."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for name, data in sorted(self.files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                tf.addfile(info, io.BytesIO(data))
        return gzip.compress(buf.getvalue(), compresslevel=1, mtime=0)

    def write(self, root: Path) -> Path:
        for name, data in self.files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return root


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> COLMAP's qvec (w, x, y, z), w >= 0."""
    from litereality_agent.models.splat.spz import matrix_to_quat

    x, y, z, w = matrix_to_quat(R)
    q = np.array([w, x, y, z], dtype=np.float64)
    return -q if q[0] < 0 else q


def _lidar_points(frame: Frame, rng: np.random.Generator, count: int, width: int, height: int,
                  rgb: np.ndarray):
    """`count` confident LiDAR samples of one frame (repeating some when it has fewer): world
    points, colours, and their pixel in the training image (COLMAP pixel coordinates: the top-left
    pixel's centre is 0.5)."""
    depth, conf = read_lidar(frame)
    h, w = depth.shape
    valid = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    if conf is not None:
        valid &= conf >= MIN_CONFIDENCE
    valid &= off_silhouettes(depth)
    vv, uu = np.nonzero(valid)
    if len(vv) == 0:
        return None
    vv, uu, uses = _pick(rng, vv, uu, count)
    # Depth-map pixel centre -> photo pixel coordinates.
    px = (uu + 0.5) * frame.size[0] / w
    py = (vv + 0.5) * frame.size[1] / h
    return (*_points(frame, px, py, depth[vv, uu], width, height, rgb), uses)


def _dense_points(frame: Frame, dense: np.ndarray, rng: np.random.Generator, count: int,
                  width: int, height: int, rgb: np.ndarray):
    """`count` trusted samples of the frame's dense depth at the training size, like
    `_lidar_points`, plus the trusted map itself (metres, 0 = untrusted); None when it has fewer
    than `MIN_FRAME_POINTS` trusted samples."""
    d = resample_nearest(np.asarray(dense, dtype=np.float32), (height, width))
    lidar, conf = read_lidar(frame)
    ok = dense_mask(d, lidar, conf, max_depth=MAX_DEPTH, min_confidence=MIN_CONFIDENCE)
    vv, uu = np.nonzero(ok)
    if len(vv) < MIN_FRAME_POINTS:
        return None
    vv, uu, uses = _pick(rng, vv, uu, count)
    # Training pixel centre -> photo pixel coordinates.
    px = (uu + 0.5) * frame.size[0] / width
    py = (vv + 0.5) * frame.size[1] / height
    return (*_points(frame, px, py, d[vv, uu], width, height, rgb), uses,
            np.where(ok, d, 0.0))


def _pick(rng: np.random.Generator, vv: np.ndarray, uu: np.ndarray, count: int):
    """Up to `count` distinct samples (rows, cols), and how many of the photo's `count` depth
    observations each one stands for (1 each, unless the photo has fewer samples than that).

    Exactly `count` observations per photo: gsplat batches photos, and its collate needs every
    photo's depth points to be the same length. A photo with fewer usable samples (a close-up of
    a bare wall, or a phone video's sparse model depth: some have a few dozen) uses all of them
    and observes some again, but a repeat is the SAME 3D point listed again, never a point of its
    own: every 3D point also starts a splat, sized by the distance to its nearest neighbours.
    Copies give log(0) = -inf; copies nudged within their pixel pile up to hundreds of splats
    into one depth pixel at ~1e-5 m (atrium_phone: 3e-5 m nearest neighbours, 1e-3 m without)."""
    n = len(vv)
    if n >= count:
        pick = rng.choice(n, size=count, replace=False)
        return vv[pick], uu[pick], np.ones(count, np.int64)
    uses = 1 + np.bincount(rng.choice(n, size=count - n), minlength=n)
    return vv, uu, uses


def _points(frame: Frame, px: np.ndarray, py: np.ndarray, z: np.ndarray, width: int, height: int,
            rgb: np.ndarray):
    """Photo pixel coordinates (top-left centre 0.5) and depth -> world points, their colours and
    their pixel in the training image."""
    K = frame.K
    cam = np.stack([(px - 0.5 - K[0, 2]) / K[0, 0] * z, -(py - 0.5 - K[1, 2]) / K[1, 1] * z, -z], 1)
    world = cam @ frame.c2w[:3, :3].T + frame.c2w[:3, 3]
    sx, sy = width / frame.size[0], height / frame.size[1]
    tx, ty = px * sx, py * sy
    ix = np.clip(tx.astype(int), 0, width - 1)
    iy = np.clip(ty.astype(int), 0, height - 1)
    return world, rgb[iy, ix], np.stack([tx, ty], 1)


def _points3d(ids: np.ndarray, xyz: np.ndarray, rgb: np.ndarray, image_id: int,
              uses: np.ndarray) -> bytes:
    """One photo's points as COLMAP points3D.bin records: each point's track lists the photo once
    per observation (`uses`), at the observations' indices in its images.bin entry, where a
    point's observations are consecutive. All tracks of length 1 (the usual LiDAR case): one
    fixed-size record per point, in id order."""
    first = np.concatenate([[0], np.cumsum(uses)[:-1]])
    out = []
    for k in np.unique(uses):
        sel = np.nonzero(uses == k)[0]
        rec = np.zeros(len(sel), dtype=[("id", "<u8"), ("xyz", "<f8", 3), ("rgb", "u1", 3),
                                        ("err", "<f8"), ("len", "<u8"),
                                        ("track", [("img", "<i4"), ("idx", "<i4")], (int(k),))])
        rec["id"], rec["xyz"], rec["rgb"] = ids[sel], xyz[sel], rgb[sel]
        rec["err"], rec["len"] = 0.01, k
        rec["track"]["img"] = image_id
        rec["track"]["idx"] = first[sel, None] + np.arange(k)
        out.append(rec.tobytes())
    return b"".join(out)


def depth_png(depth: np.ndarray) -> bytes:
    """A depth map in metres (0 = none) as a uint16 millimetre PNG."""
    from PIL import Image

    mm = np.clip(np.round(np.nan_to_num(depth) * 1000.0), 0, 65535).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(mm).save(buf, format="PNG", compress_level=DEPTH_PNG_LEVEL)
    return buf.getvalue()


def _held_out(names: list[str], test_every: int) -> set[str]:
    """The photos gsplat holds out for evaluation: every `test_every`-th by name from the first
    (its COLMAP parser sorts by name; position % test_every == 0). ``test_every <= 0``: none."""
    if test_every <= 0:
        return set()
    return {n for i, n in enumerate(sorted(names)) if i % test_every == 0}


def _half(depth: np.ndarray) -> np.ndarray:
    """A trusted depth map (0 = untrusted) at half its size: each 2 x 2 block's mean where all
    four are trusted, else 0. The trainer resizes it back nearest-neighbour, and its normals'
    central differences then still see one block's step per two pixels: the right slope."""
    h, w = depth.shape[0] // 2 * 2, depth.shape[1] // 2 * 2
    d = depth[:h, :w].reshape(h // 2, 2, w // 2, 2)
    ok = (d > 0).all(axis=(1, 3))
    return np.where(ok, d.mean(axis=(1, 3)), 0.0).astype(np.float32)


def build(frames: list[Frame], *, width: int | None = None, points_per_frame: int = POINTS_PER_FRAME,
          seed: int = 0, dense: dict[int, np.ndarray] | None = None,
          depth_maps: bool = False, test_every: int = 0) -> Dataset:
    """The COLMAP-format training set for `frames` (see the module docstring). Frames with fewer
    than `MIN_FRAME_POINTS` usable LiDAR samples are left out, the rest numbered 1..n; ValueError
    when none is left.

    `dense` maps a frame's index to its dense depth (metres, any resolution, 0 = none): those
    photos draw their `points_per_frame` points from it. `depth_maps` also writes each such
    photo's trusted dense map as ``depths/<image stem>.png`` for the trainer's dense losses.

    `test_every` is the trainer's hold-out (its "test_every"): those photos are still listed, so
    the trainer can score them, but give no 3D points (the starting splats, their colours) and no
    depth map. Their own pixels must not seed the room they are scored against."""
    from PIL import Image

    if not frames:
        raise ValueError("no frames to train on")
    width = int(width or train_width_override or TRAIN_WIDTH)
    rng = np.random.default_rng(seed)
    dense = dense or {}
    # Every photo's points first: which photos are kept sets the trainer's name order, and so
    # which of them it holds out.
    kept = []
    for frame in frames:
        scale = width / frame.size[0]
        height = int(round(frame.size[1] * scale))
        with Image.open(frame.image) as im:
            small = im.convert("RGB").resize((width, height), Image.LANCZOS)
        rgb = np.asarray(small)
        obs = trusted = None
        if frame.index in dense:
            got = _dense_points(frame, dense[frame.index], rng, points_per_frame, width, height,
                                rgb)
            if got is not None:
                *obs, trusted = got
        if obs is None:
            obs = _lidar_points(frame, rng, points_per_frame, width, height, rgb)
        if obs is None or len(obs[0]) < MIN_FRAME_POINTS:
            # gsplat's depth loss looks a training photo's points up by its name: a photo
            # without any would crash it.
            continue
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=JPEG_QUALITY)
        kept.append((frame, scale, (width, height), buf.getvalue(), obs, trusted))
    if not kept:
        raise ValueError(f"none of the {len(frames)} frames has {MIN_FRAME_POINTS} usable LiDAR "
                         "points to train on")
    held = _held_out([f"frame_{k[0].index:05d}.jpg" for k in kept], test_every)

    files: dict[str, bytes] = {}
    cameras: list[bytes] = []
    images: list[bytes] = []
    points3d: list[bytes] = []
    next_point = 1
    dense_images = 0
    for frame, scale, size, jpeg, obs, trusted in kept:
        image_id = len(images) + 1
        name = f"frame_{frame.index:05d}.jpg"
        files[f"images/{name}"] = jpeg
        held_out = name in held
        if trusted is not None:
            dense_images += 1
            if depth_maps and not held_out:
                files[f"{DEPTH_DIR}/frame_{frame.index:05d}.png"] = depth_png(_half(trusted))

        # ARKit's principal point is in pixel units with the first pixel's centre at 0; COLMAP's
        # first pixel centre is 0.5.
        K = frame.K
        fx, fy = K[0, 0] * scale, K[1, 1] * scale
        cx, cy = (K[0, 2] + 0.5) * scale, (K[1, 2] + 0.5) * scale
        cameras.append(struct.pack("<iiQQdddd", image_id, 1, *size, fx, fy, cx, cy))

        w2c = np.linalg.inv(frame.c2w @ GL_TO_CV)
        q, t = _quat_wxyz(w2c[:3, :3]), w2c[:3, 3]
        head = struct.pack("<i4d3di", image_id, *q, *t, image_id) + name.encode() + b"\0"
        n = 0 if held_out else len(obs[0])
        if n:
            world, colours, pix, uses = obs
            ids = np.arange(next_point, next_point + n, dtype=np.int64)
            next_point += n
            # One observation per use: a point the photo observes again is listed again, under
            # its own id, and its track names the photo once per listing.
            point = np.repeat(np.arange(n), uses)
            p2d = np.zeros(len(point), dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            p2d["x"], p2d["y"], p2d["id"] = pix[point, 0], pix[point, 1], ids[point]
            images.append(head + struct.pack("<Q", len(point)) + p2d.tobytes())
            points3d.append(_points3d(ids, world, colours, image_id, uses))
        else:
            images.append(head + struct.pack("<Q", 0))   # listed and scored, never a seed

    total = next_point - 1
    count = struct.pack("<Q", len(images))
    files["sparse/0/cameras.bin"] = count + b"".join(cameras)
    files["sparse/0/images.bin"] = count + b"".join(images)
    files["sparse/0/points3D.bin"] = struct.pack("<Q", total) + b"".join(points3d)
    return Dataset(files=files, images=len(images), points=total, size=kept[0][2],
                   dense_images=dense_images, held_out=len(held))
