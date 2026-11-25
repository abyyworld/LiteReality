"""The novel poses the Difix rounds render (patch_difix.py), from the room's own layout.

    uv run python scripts/difix/poses.py DATASET_TAR BUNDLE_DIR OUT_DIR [--rounds 4] [--views 96]

Targets, where the phone never stood (the dataset's training cameras are the walk):
  chairs    12 bearings around each chair, 1.1 m out (0.25 m in from the walls), at eye height
            1.3 m, looking at its seat
  door      oblique and near edge-on views of the door from the room, eye height 1.5 m
  standing  a 0.4 m grid over the floor (0.35 m in from the walls) at least 0.4 m from every
            training camera, eye height 1.55 m, four headings, looking 12 degrees down
Each target starts at its nearest training camera and moves 0.5 m (and the same share of its
turn) towards the target per round, as Difix3D's progressive update does, so the repaired
views reach the unseen spots step by step from what the photos saw. The held-out photos (every
8th by name, as gsplat's parser) are neither targets nor starting points.

Writes OUT_DIR/novel_poses.npy ([rounds, views, 4, 4] OpenCV camera-to-world in the dataset's
frame, the room's: the trainer runs with --no-normalize-world-space) and OUT_DIR/poses.png.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
import tarfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

TEST_EVERY = 8
STEP_M = 0.5
PHOTO_RIGHT = (0.0, 0.0, -1.0)     # the atrium photos' x axis, in the room (z up): the floor


def read_images_bin(data: bytes) -> list[tuple[str, np.ndarray]]:
    """COLMAP images.bin -> [(name, camera-to-world 4x4)]."""
    f = io.BytesIO(data)
    (n,) = struct.unpack("<Q", f.read(8))
    out = []
    for _ in range(n):
        _id, qw, qx, qy, qz, tx, ty, tz, _cam = struct.unpack("<idddddddi", f.read(64))
        name = b""
        while (c := f.read(1)) != b"\x00":
            name += c
        (npts,) = struct.unpack("<Q", f.read(8))
        f.read(24 * npts)
        r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = r.T
        c2w[:3, 3] = -r.T @ np.array([tx, ty, tz])
        out.append((name.decode(), c2w))
    return sorted(out)


def look_at(eye, target) -> np.ndarray:
    """OpenCV camera-to-world (x right, y down, z forward) at `eye` looking at `target`, z up,
    rolled as the scan's photos are: the phone stood upright and its landscape frames lie on
    their side, the image's x axis pointing at the floor (`PHOTO_RIGHT`)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.asarray(PHOTO_RIGHT, float) - (np.asarray(PHOTO_RIGHT, float) @ fwd) * fwd
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, down, fwd], 1)
    c2w[:3, 3] = eye
    return c2w


def inside(poly, x, y, m) -> bool:
    sgn = None
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
        ex, ey = bx - ax, by - ay
        cr = (ex * (y - ay) - ey * (x - ax)) / math.hypot(ex, ey)
        sgn = sgn if sgn is not None else (1 if cr > 0 else -1)
        if cr * sgn < m:
            return False
    return True


def score(train: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(train[:, :3, 3] - c2w[:3, 3], axis=1)
    cos = np.clip(train[:, :3, 2] @ c2w[:3, 2], -1.0, 1.0)
    return d + 0.5 * np.arccos(cos)


def toward(start: np.ndarray, goal: np.ndarray, frac: float) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = start[:3, 3] + frac * (goal[:3, 3] - start[:3, 3])
    rots = Rotation.from_matrix(np.stack([start[:3, :3], goal[:3, :3]]))
    out[:3, :3] = Slerp([0.0, 1.0], rots)([frac]).as_matrix()[0]
    return out


def targets(layout: dict, train: np.ndarray) -> dict[str, list[np.ndarray]]:
    floor, poly = layout["floor"], [tuple(p) for p in layout["polygon"]]
    out: dict[str, list[np.ndarray]] = {"chairs": [], "door": [], "standing": []}
    for cx, cy in layout["chairs"]:
        for k in range(12):
            a = 2 * math.pi * k / 12
            x, y = cx + 1.1 * math.cos(a), cy + 1.1 * math.sin(a)
            if inside(poly, x, y, 0.25):
                out["chairs"].append(look_at([x, y, floor + 1.3], [cx, cy, floor + 0.45]))
    (mx, my), (nx, ny), (ux, uy) = layout["door_mid"], layout["door_in"], layout["door_along"]
    for deg, dist in ((-65, 1.4), (-40, 1.6), (40, 1.6), (65, 1.4), (-80, 1.0), (80, 1.0)):
        t = math.radians(deg)
        x = mx + dist * (math.cos(t) * nx + math.sin(t) * ux)
        y = my + dist * (math.cos(t) * ny + math.sin(t) * uy)
        if inside(poly, x, y, 0.2):
            out["door"].append(look_at([x, y, floor + 1.5], [mx, my, floor + 1.1]))
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    walk = train[:, :2, 3]
    for x in np.arange(min(xs), max(xs), 0.4):
        for y in np.arange(min(ys), max(ys), 0.4):
            if not inside(poly, x, y, 0.35) or np.min(np.linalg.norm(walk - [x, y], axis=1)) < 0.4:
                continue
            for h in range(4):
                a = math.pi / 2 * h
                d = [math.cos(a), math.sin(a)]
                out["standing"].append(look_at([x, y, floor + 1.55],
                                               [x + d[0], y + d[1], floor + 1.55 - math.tan(
                                                   math.radians(12))]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tar")
    ap.add_argument("bundle")
    ap.add_argument("out")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--views", type=int, default=96)
    args = ap.parse_args()
    with tarfile.open(args.tar) as tf:
        images = read_images_bin(tf.extractfile("sparse/0/images.bin").read())
    names = [n for n, _ in images]
    allc = np.stack([c for _, c in images])
    held = np.arange(len(images)) % TEST_EVERY == 0
    train = allc[~held]
    layout = json.loads((Path(args.out) / "layout.json").read_text())
    groups = targets(layout, train)
    # Even shares per group, as many as fit `views`, every k-th so each group stays spread out.
    goals: list[tuple[str, np.ndarray]] = []
    share = {"door": len(groups["door"])}
    rest = args.views - share["door"]
    share["chairs"] = min(len(groups["chairs"]), rest // 2)
    share["standing"] = min(len(groups["standing"]), args.views - share["door"] - share["chairs"])
    for g, n in share.items():
        items = groups[g]
        pick = np.linspace(0, len(items) - 1, n).round().astype(int) if n else []
        goals += [(g, items[i]) for i in pick]
    starts = [train[int(np.argmin(score(train, c)))] for _, c in goals]
    poses = np.zeros((args.rounds, len(goals), 4, 4), np.float32)
    reached = []
    for v, ((_g, goal), start) in enumerate(zip(goals, starts)):
        dist = float(np.linalg.norm(goal[:3, 3] - start[:3, 3]))
        for r in range(args.rounds):
            frac = 1.0 if dist < 1e-6 else min(1.0, (r + 1) * STEP_M / dist)
            poses[r, v] = toward(start, goal, frac)
        reached.append(min(1.0, args.rounds * STEP_M / max(dist, 1e-6)))
    out = Path(args.out)
    np.save(out / "novel_poses.npy", poses)
    summary = {"photos": len(names), "train": int((~held).sum()),
               "views": {g: n for g, n in share.items()}, "total": len(goals),
               "rounds": args.rounds, "reached_share_mean": round(float(np.mean(reached)), 2),
               "fully_reached": int(sum(r >= 1.0 for r in reached)),
               "start_to_goal_m": [round(float(x), 2) for x in np.percentile(
                   [np.linalg.norm(g[:3, 3] - s[:3, 3]) for (_, g), s in zip(goals, starts)],
                   [0, 50, 100])],
               "train_right_z_mean": round(float(train[:, 2, 0].mean()), 2),
               "novel_right_z_mean": round(float(poses[-1, :, 2, 0].mean()), 2)}
    (out / "poses.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    poly = layout["polygon"] + layout["polygon"][:1]
    ax.plot(*zip(*poly), "k-", lw=1)
    ax.plot(train[:, 0, 3], train[:, 1, 3], ".", color="0.6", ms=2, label="walk (training)")
    ax.plot(allc[held, 0, 3], allc[held, 1, 3], "x", color="0.3", ms=3, label="held out")
    colors = {"chairs": "tab:blue", "door": "tab:orange", "standing": "tab:green"}
    for v, (g, goal) in enumerate(goals):
        path = poses[:, v, :3, 3]
        ax.plot(path[:, 0], path[:, 1], "-", color=colors[g], lw=0.6, alpha=0.7)
        ax.arrow(goal[0, 3], goal[1, 3], 0.15 * goal[0, 2], 0.15 * goal[1, 2], color=colors[g],
                 head_width=0.04)
    for cx, cy in layout["chairs"]:
        ax.plot(cx, cy, "s", color="tab:red", ms=8)
    ax.plot(*layout["door_mid"], "D", color="tab:orange", ms=8)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title(f"{len(goals)} novel views, {args.rounds} rounds of {STEP_M} m")
    fig.savefig(out / "poses.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    main()
