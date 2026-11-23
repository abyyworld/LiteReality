"""Where someone should first stand in a scanned room: at one of the scan's own camera spots.

A splat looks best close to where its photos were taken, and worst where the phone never was. The
room call used to start people at the free spot nearest the room's middle, facing its centre, which
in a small room can be a door 30 cm away, seen from where no photo was. This picks the capture
frame that makes the best first view instead: held roughly level (not at the floor or ceiling),
sharp (no motion blur) and looking ACROSS the room (far LiDAR depth, not a wall up close).

The result goes into the room's `splat.json` as ``"view"`` and the browser puts a newcomer there.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from litereality_agent.models.splat.rgbd import Frame

MAX_PITCH = math.radians(25)   # steeper than this is a floor or ceiling shot, not a first view
SHARP_SIZE = 320               # sharpness is measured on the photo at this width
MIN_DEPTH_SAMPLES = 50         # fewer valid LiDAR samples than this: depth unknown, frame skipped


def _sharpness(frame: Frame) -> float:
    """Variance of the Laplacian of the photo's luminance: low for a motion-blurred photo."""
    from PIL import Image

    with Image.open(frame.image) as im:
        w, h = im.size
        g = np.asarray(im.convert("L").resize((SHARP_SIZE, max(1, round(h * SHARP_SIZE / w)))),
                       dtype=np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def _median_depth(frame: Frame) -> float | None:
    """The median LiDAR depth (metres) the frame saw: how far across the room it looks."""
    from PIL import Image

    try:
        with Image.open(frame.depth) as im:
            depth = np.asarray(im, dtype=np.float32) / 1000.0
    except OSError:
        return None
    valid = depth[(depth > 0.15) & (depth < 10.0)]
    return float(np.median(valid)) if valid.size >= MIN_DEPTH_SAMPLES else None


def pose_view(c2w: np.ndarray) -> dict:
    """A camera pose (OpenGL camera in the Z-up room frame) as a standing view: where it is and
    which way it looks, yaw about +Z with 0 along +X and pitch > 0 up (the room call's own
    convention for people)."""
    fwd = -np.asarray(c2w, dtype=float)[:3, 2]
    fwd = fwd / max(1e-9, float(np.linalg.norm(fwd)))
    pos = np.asarray(c2w, dtype=float)[:3, 3]
    return {"pos": [round(float(x), 4) for x in pos],
            "yaw": round(math.atan2(fwd[1], fwd[0]), 4),
            "pitch": round(math.asin(max(-1.0, min(1.0, fwd[2]))), 4)}


def best_view(frames: Sequence[Frame]) -> dict | None:
    """The capture frame that makes the best first view (see the module docstring), as
    `pose_view` plus ``"frame"`` (its index); None when no frame qualifies. Sharpness only ranks:
    the sharpest half of the level frames compete on how far they see."""
    level = []
    for f in frames:
        view = pose_view(f.c2w)
        if abs(view["pitch"]) > MAX_PITCH:
            continue
        depth = _median_depth(f)
        if depth is None:
            continue
        level.append((f, view, depth))
    if not level:
        return None
    sharp = [_sharpness(f) for f, _v, _d in level]
    cut = float(np.median(sharp))
    pool = [(item, s) for item, s in zip(level, sharp) if s >= cut] or list(zip(level, sharp))
    (frame, view, _depth), _s = max(
        pool, key=lambda p: p[0][2] * math.cos(p[0][1]["pitch"]))
    return {**view, "frame": int(frame.index)}


__all__ = ["best_view", "pose_view"]
