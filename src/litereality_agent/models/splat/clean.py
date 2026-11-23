"""Cleaning a splat room: what the scan says cannot be there is taken out.

A trained room carries floaters (splats hanging in the air between the camera and the wall that
make a view the scan barely saw look smeared or streaked), haze, and whatever the camera glimpsed
through a window or a doorway. Four rules take them out, cheapest first, each switched on alone:

    faint        opacity under `min_opacity`: haze, not detail
    outside      beyond the room's walls, floor or ceiling grown by `margin` (RoomPlan's room, in
                 the splat's own frame). The span of each door and window is kept, and a door's
                 swing (its leaf may stand open outward); the view through them is not.
    carved       FREE-SPACE CARVING. A depth pixel says the space between the camera and the
                 surface it measured is empty, so a splat well in front of that surface (by more
                 than max(5 cm, 3 %) of the measured depth) was "seen through" by that frame. A
                 splat seen through by at least `carve_votes` frames, and by at least half of the
                 frames whose valid depth it projects onto, is a floater. The vote, not one frame,
                 decides: a pose error at a silhouette or one bad depth pixel is outvoted by the
                 views that see the splat where it is.
    unsupported  AGREEMENT (off unless `support_views` > 0). Carving's opposite question: not "did
                 a photo see through it" but "did enough photos see it". A splat is kept only if
                 it lies ON the measured surface (within max(5 cm, 4 %) of the depth at its pixel:
                 neither floating in front of it nor hidden behind it) in at least
                 `support_views` training frames. Splats only one or two photos saw (speckle on a
                 desk, a smear behind clutter) go, and so do splats on no measured surface at all
                 (fill behind a window, a big blob fitted to a bright patch). The price: a surface
                 the depth never measured (beyond the LiDAR's range in every view, glass, a black
                 screen) goes too, and the browser shows the plain room through the hole.

Carving and agreement use the frames the trainer TRAINS on only: gsplat holds out every
`test_every`-th photo by name order and reports PSNR/SSIM/LPIPS on them, and letting their depth
decide what stays would leak those photos into the room they are scored against. Both rules share
one projection pass: turning agreement on costs a few comparisons per frame, not a second pass.

Depth is the dense map when one is given for a frame (PromptDA, metres, 0 = invalid, any size) and
the scan's LiDAR otherwise (`depth_*.png`, millimetres, confident pixels only). A dense map is
trusted only where it agrees with a confident LiDAR sample under it (`rgbd.dense_mask`, the rule
the training set and the instant room use): PromptDA refines the sensor, it never overrules it,
and a depth it invents behind glass or at a soft silhouette would carve the real surface in front.
Both drop the pixels on a depth discontinuity the way `rgbd` does: a mixed pixel there floats
between the two surfaces and would carve the nearer one.

Pure numpy: ~1M splats x 60 frames in a few seconds on a laptop (splats in chunks, one frame's
depth in memory at a time). Projection is the cost, linear in frames, so a long scan carves (and
checks agreement) with at most `CARVE_FRAMES` of its training frames, spread evenly over it; the
agreement bar then shrinks in proportion (`support_need`), so a long scan is held to the same
share of its photos as a short one. A `depth_cache` keeps each frame's filtered depth between
calls (every snapshot of one build).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from litereality_agent.models.splat.rgbd import (
    EDGE_ABSOLUTE,
    EDGE_RELATIVE,
    MAX_DEPTH,
    MIN_CONFIDENCE,
    MIN_DEPTH,
    Frame,
    dense_mask,
    read_lidar,
)
from litereality_agent.models.splat.spz import Splats

MARGIN = 0.15
MIN_OPACITY = 0.02
CARVE_VOTES = 2
CARVE_ABSOLUTE, CARVE_RELATIVE = 0.05, 0.03   # "in front of the measured surface" beyond this
SUPPORT_VIEWS = 3                             # agreement's bar when it is switched on
SUPPORT_ABSOLUTE, SUPPORT_RELATIVE = 0.05, 0.04   # "on the measured surface" within this
TEST_EVERY = 8                                # the trainer's own hold-out (deploy/modal/splat)
CHUNK = 1 << 18                               # splats projected at a time
# Frames that vote at most: ~2.5 s for 1M splats on a laptop. Two votes out of ~100 views spread
# over the scan still see every surface several times; a 300-photo scan would take ~8 s a snapshot.
CARVE_FRAMES = 96


# ── the room ─────────────────────────────────────────────────────────────────────────────────────


@dataclass
class RoomOpening:
    """A door, window or open doorway: its span on the floor plan (along its wall's centreline)
    and its height in the room's frame."""

    start: tuple[float, float]
    end: tuple[float, float]
    bottom: float
    top: float
    kind: str = "window"      # door | window | opening

    @property
    def width(self) -> float:
        return float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]))


@dataclass
class Room:
    """The room a splat must stay in: its floor polygons (XY, the union of every slab) between
    the floor and the ceiling, plus its openings. With no outline only the heights are used."""

    outlines: list[np.ndarray]
    floor_z: float
    ceiling_z: float
    openings: list[RoomOpening] = field(default_factory=list)


def room_from_plan(plan) -> Room:
    """`Room` from a parsed RoomPlan (`pipeline.quickroom.roomplan.parse_room`), read by duck
    typing so this module stays below the pipeline: ``outlines()``, ``floor_z``, ``ceiling_z``,
    ``walls`` (name -> start, end, along) and ``openings`` (wall, offset to the centre, width,
    height, sill above the floor, kind). Both are in the room's frame, the frame the splats use."""
    openings = []
    for op in plan.openings.values():
        wall = plan.walls.get(op.wall)
        if wall is None:
            continue
        along = np.asarray(wall.along, dtype=float)
        a, b = op.offset - op.width / 2.0, op.offset + op.width / 2.0
        start = np.asarray(wall.start, dtype=float)
        bottom = float(plan.floor_z) + float(op.sill)
        openings.append(RoomOpening(tuple(start + along * a), tuple(start + along * b),
                                    bottom, bottom + float(op.height), str(op.kind)))
    return Room([np.asarray(o, dtype=float) for o in plan.outlines()],
                float(plan.floor_z), float(plan.ceiling_z), openings)


def _inside_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule, one pass per edge."""
    inside = np.zeros(len(xy), dtype=bool)
    x, y = xy[:, 0], xy[:, 1]
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        crosses = (y0 > y) != (y1 > y)
        if not crosses.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < xc)
    return inside


def _near_outline(xy: np.ndarray, poly: np.ndarray, reach: float) -> np.ndarray:
    """Whether each point is within `reach` of the polygon's boundary."""
    near = np.zeros(len(xy), dtype=bool)
    for p0, p1 in zip(poly, np.roll(poly, -1, axis=0)):
        d = p1 - p0
        length2 = float(d[0] * d[0] + d[1] * d[1])
        rel = xy - p0
        s = (rel[:, 0] * d[0] + rel[:, 1] * d[1]) / max(length2, 1e-12)
        s = np.clip(s, 0.0, 1.0)
        gx, gy = rel[:, 0] - s * d[0], rel[:, 1] - s * d[1]
        near |= gx * gx + gy * gy <= reach * reach
    return near


def _near_opening(pos: np.ndarray, op: RoomOpening, margin: float, door_swing: bool):
    """Within `margin` of the opening's span on its wall's plane; a door's leaf may swing as far
    out as the door is wide, so a door keeps that much."""
    start, end = np.asarray(op.start, dtype=float), np.asarray(op.end, dtype=float)
    length = float(np.linalg.norm(end - start))
    if length < 1e-6:
        return np.zeros(len(pos), dtype=bool)
    along = (end - start) / length
    normal = np.array([-along[1], along[0]])
    rx, ry = pos[:, 0] - start[0], pos[:, 1] - start[1]
    s, n = rx * along[0] + ry * along[1], rx * normal[0] + ry * normal[1]
    reach = max(margin, length) if (door_swing and op.kind == "door") else margin
    return ((s >= -margin) & (s <= length + margin) & (np.abs(n) <= reach)
            & (pos[:, 2] >= op.bottom - margin) & (pos[:, 2] <= op.top + margin))


def inside_bounds(positions: np.ndarray, bounds, margin: float = MARGIN, *,
                  door_swing: bool = True) -> np.ndarray:
    """Per splat centre, whether it is in the room grown by `margin`. `bounds` is a `Room` or an
    axis-aligned box ``(min xyz, max xyz)`` in the splat's frame."""
    pos = np.asarray(positions, dtype=np.float64)
    if isinstance(bounds, Room):
        keep = (pos[:, 2] >= bounds.floor_z - margin) & (pos[:, 2] <= bounds.ceiling_z + margin)
        outlines = [o for o in bounds.outlines if len(o) >= 3]
        if outlines:
            xy = pos[:, :2]
            flat = np.zeros(len(pos), dtype=bool)
            for poly in outlines:
                flat |= _inside_polygon(xy, poly)
                rest = ~flat
                if rest.any():
                    flat[rest] = _near_outline(xy[rest], poly, margin)
            keep &= flat
        for op in bounds.openings:
            out = ~keep
            if out.any():
                keep[out] = _near_opening(pos[out], op, margin, door_swing)
        return keep
    lo, hi = (np.asarray(b, dtype=np.float64).reshape(3) for b in bounds)
    return np.all((pos >= lo - margin) & (pos <= hi + margin), axis=1)


# ── the frames carving may use ───────────────────────────────────────────────────────────────────


def image_name(frame: Frame) -> str:
    """The photo's name in the training set (`dataset.build`), which sets the trainer's order."""
    return f"frame_{frame.index:05d}.jpg"


def training_frames(frames: list[Frame], *, test_every: int = TEST_EVERY,
                    image_names: Iterable[str] | None = None) -> list[Frame]:
    """The frames gsplat trains on: its COLMAP parser sorts the photos by name and holds out
    every `test_every`-th from the first (position % test_every == 0) for evaluation.
    `image_names` is the training set's own photo list (``images/<name>`` without the folder):
    `dataset.build` drops photos with too few depth points, which shifts the order, and a frame
    missing from it is not trained on either. ``test_every <= 0``: nothing is held out."""
    if image_names is not None:
        names = sorted({Path(n).name for n in image_names})
    else:
        names = sorted(image_name(f) for f in frames)
    trained = {n for i, n in enumerate(names) if test_every <= 0 or i % test_every != 0}
    return [f for f in frames if image_name(f) in trained]


# ── carving ──────────────────────────────────────────────────────────────────────────────────────


def _edge_free(depth: np.ndarray) -> np.ndarray:
    """False where a pixel's 3x3 neighbourhood spans a depth discontinuity (`rgbd`'s rule)."""
    h, w = depth.shape
    pad = np.pad(depth, 1, mode="edge")
    hi = pad[1:1 + h, 1:1 + w].copy()
    lo = hi.copy()
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            win = pad[dy:dy + h, dx:dx + w]
            np.maximum(hi, win, out=hi)
            np.minimum(lo, win, out=lo)
    return (hi - lo) < EDGE_RELATIVE * depth + EDGE_ABSOLUTE


def measured_depth(frame: Frame, dense: dict | None = None,
                   max_depth: float = MAX_DEPTH) -> np.ndarray | None:
    """The frame's depth in metres with 0 where it says nothing: the dense map when `dense` has
    one for this frame, where a confident LiDAR sample does not contradict it (`dense_mask`),
    else the LiDAR's confident pixels. None when it cannot be read."""
    conf = None
    try:
        lidar, lidar_conf = read_lidar(frame)
    except (OSError, ValueError):
        lidar = lidar_conf = None
    if dense is not None and frame.index in dense and dense[frame.index] is not None:
        depth = np.asarray(dense[frame.index], dtype=np.float32)
        if depth.ndim != 2:
            return None
        depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
        if lidar is not None and lidar.ndim == 2:
            depth = np.where(dense_mask(depth, lidar, lidar_conf, max_depth=max_depth),
                             depth, 0.0).astype(np.float32)
        # (No LiDAR to check against: the dense map stands alone, as `dense_mask` lets it where
        # no confident sample is near.)
    else:
        if lidar is None:
            return None
        depth, conf = lidar, lidar_conf
    if depth.ndim != 2:
        return None
    depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    valid = (depth > MIN_DEPTH) & (depth < max_depth)
    if conf is not None:
        valid &= conf >= MIN_CONFIDENCE
    valid &= _edge_free(depth)
    return np.where(valid, depth, 0.0).astype(np.float32)


def _projection(frame: Frame, shape: tuple[int, int]) -> np.ndarray:
    """3x4 world -> (u z, v z, z) for the depth map's pixel grid, z the camera depth: the floor of
    u, v is the pixel. ARKit's intrinsics put the photo's first pixel centre at 0; a depth pixel
    covers photo pixels (u_d + 0.5) * s - 0.5 +- s/2, as in `rgbd`."""
    h, w = shape
    sx, sy = w / frame.size[0], h / frame.size[1]
    K = frame.K
    Kd = np.array([[K[0, 0] * sx, 0.0, (K[0, 2] + 0.5) * sx],
                   [0.0, K[1, 1] * sy, (K[1, 2] + 0.5) * sy],
                   [0.0, 0.0, 1.0]])
    R, t = frame.c2w[:3, :3], frame.c2w[:3, 3]
    w2c = np.diag([1.0, -1.0, -1.0]) @ R.T           # OpenGL camera -> OpenCV: y down, z ahead
    return Kd @ np.hstack([w2c, (-w2c @ t)[:, None]])


def spread(frames: list[Frame], most: int = CARVE_FRAMES) -> list[Frame]:
    """At most `most` of `frames`, evenly spaced in their order (all of them when fewer)."""
    if most <= 0 or len(frames) <= most:
        return list(frames)
    pick = np.unique(np.round(np.linspace(0, len(frames) - 1, most)).astype(int))
    return [frames[i] for i in pick]


def _cached_depth(frame: Frame, dense: dict | None, cache: dict | None) -> np.ndarray | None:
    """`measured_depth`, kept in `cache` (float16: ~4 mm at 4.5 m, well under the carve
    margin) under the frame and whether a dense map was used."""
    use_dense = dense is not None and dense.get(frame.index) is not None
    key = (frame.index, id(dense[frame.index]) if use_dense else None)
    if cache is not None and key in cache:
        got = cache[key]
        return None if got is None else got.astype(np.float32)
    depth = measured_depth(frame, dense)
    if cache is not None:
        cache[key] = None if depth is None else depth.astype(np.float16)
    return depth


def depth_counts(positions: np.ndarray, frames: list[Frame], *, dense: dict | None = None,
                 chunk: int = CHUNK, depth_cache: dict | None = None,
                 support: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """One projection pass over `frames`, per splat centre: in how many it was seen THROUGH (in
    front of the measured surface), in how many it projected onto valid depth at all, and (with
    `support`) in how many it lay ON the measured surface; and how many frames had depth."""
    pos = np.ascontiguousarray(positions, dtype=np.float32)
    n = len(pos)
    through = np.zeros(n, dtype=np.uint16)
    seen = np.zeros(n, dtype=np.uint16)
    on = np.zeros(n, dtype=np.uint16) if support else None
    used = 0
    for frame in frames:
        depth = _cached_depth(frame, dense, depth_cache)
        if depth is None or not (depth > 0).any():
            continue
        used += 1
        h, w = depth.shape
        P = _projection(frame, depth.shape).astype(np.float32)
        A, b = P[:, :3].T.copy(), P[:, 3]
        flat = depth.ravel()
        for lo in range(0, n, chunk):
            # macOS Accelerate's matmul raises spurious FP warnings on finite input.
            with np.errstate(all="ignore"):
                hom = pos[lo:lo + chunk] @ A
            hom += b
            z = hom[:, 2]
            idx = np.flatnonzero(z > MIN_DEPTH)
            if not len(idx):
                continue
            z = z[idx]
            u = np.floor(hom[idx, 0] / z)
            v = np.floor(hom[idx, 1] / z)
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            idx, z = idx[ok], z[ok]
            d = flat[v[ok].astype(np.int64) * w + u[ok].astype(np.int64)]
            valid = d > 0
            idx, z, d = idx[valid], z[valid], d[valid]
            hit = lo + idx                                     # unique within a frame
            seen[hit] += 1
            front = z < d - np.maximum(CARVE_ABSOLUTE, CARVE_RELATIVE * d)
            through[hit[front]] += 1
            if on is not None:
                close = np.abs(z - d) <= np.maximum(SUPPORT_ABSOLUTE, SUPPORT_RELATIVE * d)
                on[hit[close]] += 1
    return through, seen, on, used


def count_votes(positions: np.ndarray, frames: list[Frame], *, dense: dict | None = None,
                chunk: int = CHUNK,
                depth_cache: dict | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Per splat centre: in how many `frames` it was seen through, and in how many it projected
    onto valid depth at all; and how many frames had depth to vote with."""
    through, seen, _on, used = depth_counts(positions, frames, dense=dense, chunk=chunk,
                                            depth_cache=depth_cache, support=False)
    return through, seen, used


def support_need(support_views: int, voting: int, training: int, used: int) -> int:
    """How many frames must see a splat on its surface for it to stay: `support_views` of the
    training frames, scaled to the share of them that vote (`spread` keeps at most
    `CARVE_FRAMES`), and never more than the frames that had depth (a scan with two photos
    cannot ask for three)."""
    if support_views <= 0 or used <= 0:
        return 0
    share = voting / max(training, 1)
    need = int(np.ceil(support_views * min(1.0, share) - 1e-9))
    return max(1, min(need, used))


def clean(splats: Splats, frames: list[Frame], *, dense: dict | None = None,
          bounds=None, margin: float = MARGIN, min_opacity: float = MIN_OPACITY,
          carve_votes: int = CARVE_VOTES, support_views: int = 0,
          test_every: int = TEST_EVERY, image_names: Iterable[str] | None = None,
          protect: np.ndarray | None = None, door_swing: bool = True,
          carve_frames: int = CARVE_FRAMES, depth_cache: dict | None = None,
          log=None) -> tuple[Splats, dict]:
    """`splats` without the faint, the outside, the carved and (with `support_views`) the
    unsupported (see the module docstring). Every rule is switched off alone:
    ``min_opacity <= 0``, ``bounds=None``, ``carve_votes <= 0``, ``support_views <= 0`` (the
    default: agreement is asked for, never implied).

    `frames` are the scan's (`rgbd.load_frames`); only the ones the trainer trains on carve or
    vote for agreement (`training_frames` with `test_every` and the training set's
    `image_names`). `dense` maps a frame index to its dense depth (metres, 0 invalid); frames
    without one use their LiDAR. `bounds` is a `Room` (`room_from_plan`) or ``(min xyz, max
    xyz)``. `protect` (bool per splat) exempts splats from both depth rules. At most
    `carve_frames` training frames vote (`spread`; <= 0: all); `depth_cache` (a dict the caller
    keeps, keyed by the dense maps' identity) saves re-reading their depth next call. No frame
    with depth: agreement is skipped, not used to drop the whole room.
    Returns the kept splats and the counts each rule dropped, with the seconds it took."""
    t0 = time.perf_counter()
    n = len(splats)
    keep = np.ones(n, dtype=bool)
    stats: dict = {"input": n, "faint": 0, "outside": 0, "carved": 0, "unsupported": 0,
                   "carve_frames": 0}
    carving = carve_votes > 0
    agreeing = support_views > 0
    if n:
        if min_opacity > 0:
            keep &= np.asarray(splats.opacities, dtype=np.float32).reshape(-1) >= min_opacity
            stats["faint"] = int(n - keep.sum())

        if bounds is not None:
            idx = np.flatnonzero(keep)
            inside = inside_bounds(splats.positions[idx], bounds, margin, door_swing=door_swing)
            keep[idx[~inside]] = False
            stats["outside"] = int((~inside).sum())

        if (carving or agreeing) and frames:
            use = training_frames(frames, test_every=test_every, image_names=image_names)
            voters = spread(use, carve_frames)
            idx = np.flatnonzero(keep)
            if protect is not None:
                idx = idx[~np.asarray(protect, dtype=bool)[idx]]
            through, seen, on, used = depth_counts(splats.positions[idx], voters, dense=dense,
                                                   depth_cache=depth_cache, support=agreeing)
            stats["carve_frames"] = used
            stats["held_out_frames"] = len(frames) - len(use)
            gone = np.zeros(len(idx), dtype=bool)
            if carving:
                gone = (through >= carve_votes) & (2 * through.astype(np.int32) >= seen)
                stats["carved"] = int(gone.sum())
            if agreeing:
                need = support_need(support_views, len(voters), len(use), used)
                stats["support_views"] = need
                if need:
                    # Carved splats are counted once, as carved.
                    unsupported = (on < need) & ~gone
                    stats["unsupported"] = int(unsupported.sum())
                    gone |= unsupported
            keep[idx[gone]] = False
    stats["kept"] = int(keep.sum())
    stats["seconds"] = round(time.perf_counter() - t0, 3)
    if log:
        log(f"clean: {stats['kept']:,} of {n:,} splats kept ({stats['faint']:,} faint, "
            f"{stats['outside']:,} outside the room, {stats['carved']:,} carved, "
            f"{stats['unsupported']:,} seen on a surface by too few photos; "
            f"{stats['carve_frames']} frames), {stats['seconds']:.1f} s")
    return (splats if stats["kept"] == n else splats.subset(keep)), stats
