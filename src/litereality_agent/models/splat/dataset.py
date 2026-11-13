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
and its "sfm" initialisation starts from the LiDAR surface rather than from noise. With
``--normalize_world_space False`` the trained splats stay in the room's frame (metres, Z up),
the frame the room call draws in.
"""

from __future__ import annotations

import io
import struct
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from litereality_agent.models.splat.rgbd import EDGE_ABSOLUTE, EDGE_RELATIVE, MIN_DEPTH, Frame

TRAIN_WIDTH = 960
POINTS_PER_FRAME = 4000
MAX_DEPTH = 5.0
MIN_CONFIDENCE = 2
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])
JPEG_QUALITY = 95


@dataclass
class Dataset:
    files: dict[str, bytes]          # relative path -> contents
    images: int
    points: int
    size: tuple[int, int]            # training (width, height)

    def tar(self) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for name, data in sorted(self.files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                tf.addfile(info, io.BytesIO(data))
        return buf.getvalue()

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
    """Up to `count` confident LiDAR samples of one frame: world points, colours, and their
    pixel in the training image (COLMAP pixel coordinates: the top-left pixel's centre is 0.5)."""
    from PIL import Image

    with Image.open(frame.depth) as im:
        depth = np.asarray(im, dtype=np.float32) / 1000.0
    h, w = depth.shape
    valid = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    if frame.conf is not None:
        with Image.open(frame.conf) as im:
            c = np.asarray(im)
        if c.shape == depth.shape:
            valid &= c >= MIN_CONFIDENCE
    pad = np.pad(depth, 1, mode="edge")
    stack = np.stack([pad[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)])
    valid &= (stack.max(0) - stack.min(0)) < EDGE_RELATIVE * depth + EDGE_ABSOLUTE
    vv, uu = np.nonzero(valid)
    if len(vv) == 0:
        return None
    pick = rng.choice(len(vv), size=min(count, len(vv)), replace=False)
    vv, uu = vv[pick], uu[pick]
    z = depth[vv, uu]
    # Depth-map pixel centre -> photo pixel coordinates -> camera ray.
    px = (uu + 0.5) * frame.size[0] / w
    py = (vv + 0.5) * frame.size[1] / h
    K = frame.K
    cam = np.stack([(px - 0.5 - K[0, 2]) / K[0, 0] * z, -(py - 0.5 - K[1, 2]) / K[1, 1] * z, -z], 1)
    world = cam @ frame.c2w[:3, :3].T + frame.c2w[:3, 3]
    sx, sy = width / frame.size[0], height / frame.size[1]
    tx, ty = px * sx, py * sy
    ix = np.clip(tx.astype(int), 0, width - 1)
    iy = np.clip(ty.astype(int), 0, height - 1)
    return world, rgb[iy, ix], np.stack([tx, ty], 1)


def build(frames: list[Frame], *, width: int = TRAIN_WIDTH, points_per_frame: int = POINTS_PER_FRAME,
          seed: int = 0) -> Dataset:
    """The COLMAP-format training set for `frames` (see the module docstring)."""
    from PIL import Image

    if not frames:
        raise ValueError("no frames to train on")
    rng = np.random.default_rng(seed)
    files: dict[str, bytes] = {}
    cameras = [struct.pack("<Q", len(frames))]
    images = [struct.pack("<Q", len(frames))]
    pts_xyz, pts_rgb, pts_track = [], [], []
    next_point = 1
    size = (0, 0)
    for image_id, frame in enumerate(frames, start=1):
        scale = width / frame.size[0]
        height = int(round(frame.size[1] * scale))
        size = (width, height)
        with Image.open(frame.image) as im:
            small = im.convert("RGB").resize((width, height), Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=JPEG_QUALITY)
        name = f"frame_{frame.index:05d}.jpg"
        files[f"images/{name}"] = buf.getvalue()
        rgb = np.asarray(small)

        # ARKit's principal point is in pixel units with the first pixel's centre at 0; COLMAP's
        # first pixel centre is 0.5.
        K = frame.K
        fx, fy = K[0, 0] * scale, K[1, 1] * scale
        cx, cy = (K[0, 2] + 0.5) * scale, (K[1, 2] + 0.5) * scale
        cameras.append(struct.pack("<iiQQdddd", image_id, 1, width, height, fx, fy, cx, cy))

        w2c = np.linalg.inv(frame.c2w @ GL_TO_CV)
        q, t = _quat_wxyz(w2c[:3, :3]), w2c[:3, 3]
        obs = _lidar_points(frame, rng, points_per_frame, width, height, rgb)
        n = 0 if obs is None else len(obs[0])
        ids = np.arange(next_point, next_point + n, dtype=np.int64)
        next_point += n
        head = struct.pack("<i4d3di", image_id, *q, *t, image_id) + name.encode() + b"\0"
        p2d = np.zeros(n, dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
        if n:
            world, colours, pix = obs
            p2d["x"], p2d["y"], p2d["id"] = pix[:, 0], pix[:, 1], ids
            pts_xyz.append(world)
            pts_rgb.append(colours)
            track = np.zeros(n, dtype=[("img", "<i4"), ("idx", "<i4")])
            track["img"], track["idx"] = image_id, np.arange(n)
            pts_track.append((ids, track))
        images.append(head + struct.pack("<Q", n) + p2d.tobytes())

    total = next_point - 1
    rec = np.zeros(total, dtype=[("id", "<u8"), ("xyz", "<f8", 3), ("rgb", "u1", 3),
                                 ("err", "<f8"), ("len", "<u8"), ("img", "<i4"), ("idx", "<i4")])
    if total:
        rec["id"] = np.concatenate([ids for ids, _ in pts_track])
        rec["xyz"] = np.concatenate(pts_xyz)
        rec["rgb"] = np.concatenate(pts_rgb)
        rec["err"] = 0.01
        rec["len"] = 1
        tracks = np.concatenate([track for _, track in pts_track])
        rec["img"], rec["idx"] = tracks["img"], tracks["idx"]
    files["sparse/0/cameras.bin"] = b"".join(cameras)
    files["sparse/0/images.bin"] = b"".join(images)
    files["sparse/0/points3D.bin"] = struct.pack("<Q", total) + rec.tobytes()
    return Dataset(files=files, images=len(frames), points=total, size=size)
