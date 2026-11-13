"""Which splats are which object, from the photos: SAM 2 masks voted onto the splats.

A RoomPlan box says roughly where an object is; the photos say exactly which pixels it covers.
For every movable body this projects its box (after `split.snap`) into the photos that see it
best, asks SAM 2 (`deploy/modal/segment`) for the object's outline inside each projected box,
then lets every splat near the body VOTE: projected into each of those photos, where the LiDAR
says the splat is visible there (not hidden behind something nearer), it counts whether it falls
inside the object's mask. A splat seen inside the mask in at least `MIN_VOTES` photos and in at
least `MIN_RATIO` of the photos that saw it belongs to the object (FlashSplat-style labelling, by
projection instead of rasterisation). A splat no photo saw keeps what the boxes said.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from litereality_agent.models.splat.rgbd import Frame

WIDTH = 960                    # the photos SAM 2 sees (and the masks come back at)
VIEWS_PER_OBJECT = 24
MIN_AREA = 0.005               # of the photo, for a projected box to be worth a mask
MIN_INSIDE = 0.4               # of the projected box that must lie inside the photo
MIN_VOTES = 2
MIN_RATIO = 0.5
REACH = 0.30                   # metres around a body's box where splats vote
# The projected box grows by this much of its size on each side before it prompts SAM 2 (and
# bounds the votes): RoomPlan boxes can sit 20-30 cm off, and the object must be inside the box.
GROW = 0.30
# A splat is visible in a photo if no farther than its LiDAR depth + a + b * depth. Generous: the
# depth map is 256 x 192, so an object's edge pixel often holds the wall behind it.
VISIBLE_TOL = (0.08, 0.05)


@dataclass
class View:
    frame: Frame
    K: np.ndarray              # at WIDTH
    size: tuple[int, int]
    boxes: dict[str, list[float]] = field(default_factory=dict)       # body name -> x0 y0 x1 y1
    negatives: dict[str, list[list[float]]] = field(default_factory=dict)   # name -> [[x, y]]


@dataclass
class ObjectMasks:
    views: dict[str, View]                                             # frame id -> view
    masks: dict[str, dict[str, np.ndarray]]                            # frame id -> name -> bool
    seconds: float = 0.0

    def __len__(self) -> int:
        return sum(len(m) for m in self.masks.values())


def _project(K: np.ndarray, c2w: np.ndarray, pts: np.ndarray):
    w2c = np.linalg.inv(c2w)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = -cam[:, 2]
    safe = np.where(z > 1e-6, z, 1.0)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = -K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return u, v, z


def _box_corners(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> np.ndarray:
    from litereality_agent.pipeline.splatroom.split import _union_box

    _b, R, c, half = _union_box(model, data, body)
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    return c + (signs * half) @ R.T


FURNITURE_MAX_VOLUME = 3.0     # m^3: a static body this small is furniture, not the room shell


def _furniture_tops(model: mujoco.MjModel, data: mujoco.MjData, movable: set[int]) -> np.ndarray:
    """The top centre of every static piece of furniture (a table, a cabinet): where a tucked
    chair's prompt box most likely also catches something that is not the chair."""
    from litereality_agent.pipeline.splatroom.split import _body_geoms, _union_box

    tops = []
    for b in range(1, model.nbody):
        if b in movable or not _body_geoms(model, b):
            continue
        _b, R, c, half = _union_box(model, data, b)
        if 8 * float(np.prod(half)) > FURNITURE_MAX_VOLUME:
            continue
        tops.append(c + R @ np.array([0.0, 0.0, half[2]]))
    return np.asarray(tops).reshape(-1, 3)


def plan_views(frames: list[Frame], model: mujoco.MjModel, data: mujoco.MjData,
               bodies: dict[str, int], *, per_object: int = VIEWS_PER_OBJECT) -> dict[str, View]:
    """For every body, the photos that see its box largest and most whole, with the box each
    should prompt SAM 2 with, and as negative points the tops of static furniture inside that
    box. Photos are shared: one view holds every body it was chosen for."""
    from litereality_agent.room_ops.roomcall.model_info import movable_body_ids

    tops = _furniture_tops(model, data, movable_body_ids(model))
    views: dict[str, View] = {}
    for name, body in bodies.items():
        corners = _box_corners(model, data, body)
        ranked = []
        for f in frames:
            scale = WIDTH / f.size[0]
            W, H = WIDTH, int(round(f.size[1] * scale))
            K = f.K.copy()
            K[:2] *= scale
            u, v, z = _project(K, f.c2w, corners)
            if np.any(z < 0.2):
                continue
            x0, x1, y0, y1 = u.min(), u.max(), v.min(), v.max()
            area = (x1 - x0) * (y1 - y0)
            gx, gy = GROW * (x1 - x0), GROW * (y1 - y0)
            cx0, cx1, cy0, cy1 = max(x0, 0), min(x1, W), max(y0, 0), min(y1, H)
            inside = max(cx1 - cx0, 0) * max(cy1 - cy0, 0)
            if area <= 0 or inside / area < MIN_INSIDE or inside / (W * H) < MIN_AREA:
                continue
            prompt = [max(x0 - gx, 0), max(y0 - gy, 0), min(x1 + gx, W), min(y1 + gy, H)]
            ranked.append((inside / (W * H) * (inside / area), f, K, (W, H),
                           [float(v) for v in prompt]))
        ranked.sort(key=lambda r: -r[0])
        own = corners.mean(0)
        for _score, f, K, size, box in ranked[:per_object]:
            key = f"{f.index:05d}"
            view = views.setdefault(key, View(f, K, size))
            view.boxes[name] = box
            if len(tops):
                u, v, z = _project(K, f.c2w, tops)
                _, _, z_own = _project(K, f.c2w, own[None])
                inside = (z > 0.2) & (u > box[0]) & (u < box[2]) & (v > box[1]) & (v < box[3])
                # only furniture in front of the object's centre: what would cover it
                inside &= z < z_own[0] + 0.3
                if inside.any():
                    view.negatives[name] = [[float(a), float(b)] for a, b in
                                            zip(u[inside][:3], v[inside][:3])]
    return views


STATIC_SHARE_MAX = 0.5         # a candidate mask mostly on static furniture is not the object


def _static_share(mask: np.ndarray, view: View, static_boxes) -> float:
    """How much of a mask lies on static furniture: its pixels lifted to 3D with the LiDAR depth,
    the share inside a static body's box (+3 cm)."""
    from litereality_agent.pipeline.splatroom.split import _inside_any

    depth = _depth(view.frame)
    dh, dw = depth.shape
    W, H = view.size
    vv, uu = np.nonzero(mask[::4, ::4])
    if not len(vv):
        return 1.0
    u, v = uu * 4 + 2.0, vv * 4 + 2.0
    d = depth[np.clip((v * dh / H).astype(int), 0, dh - 1), np.clip((u * dw / W).astype(int), 0, dw - 1)]
    ok = d > 0.1
    if ok.sum() < 10:
        return 0.0
    K = view.K
    cam = np.stack([(u[ok] - K[0, 2]) / K[0, 0] * d[ok], -(v[ok] - K[1, 2]) / K[1, 1] * d[ok],
                    -d[ok]], 1)
    world = cam @ view.frame.c2w[:3, :3].T + view.frame.c2w[:3, 3]
    return float(_inside_any(world, static_boxes, 0.03).mean())


def _choose(pngs: list[bytes], scores: list[float], view: View, static_boxes) -> np.ndarray | None:
    """SAM's candidate masks for one box -> the best that is not mostly static furniture."""
    from PIL import Image

    best, best_key = None, None
    for png, score in zip(pngs, scores):
        with Image.open(io.BytesIO(png)) as im:
            m = np.asarray(im.convert("L")) > 127
        if m.shape != (view.size[1], view.size[0]) or not m.any():
            continue
        share = _static_share(m, view, static_boxes) if static_boxes else 0.0
        key = (share <= STATIC_SHARE_MAX, -share if share > STATIC_SHARE_MAX else score)
        if best_key is None or key > best_key:
            best, best_key = m, key
    return best if best_key is not None and best_key[0] else None


def request(views: dict[str, View], call: Callable[[dict], dict], *,
            static_boxes=None) -> ObjectMasks:
    """Ask the segment function (`call`, e.g. a Modal Function's ``remote``) for every view's
    masks. With `static_boxes` (`split._static_boxes`: static bodies and walls) SAM's three candidates
    are fetched and the best one that is not mostly static furniture is kept (`_choose`); a box
    whose every candidate is furniture gets no mask. Raises RuntimeError with the function's own
    message when it refuses."""
    from PIL import Image

    frames = []
    for key, view in views.items():
        with Image.open(view.frame.image) as im:
            small = im.convert("RGB").resize(view.size, Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=92)
        frames.append({"id": key, "jpeg": buf.getvalue(), "boxes": view.boxes,
                       "negatives": view.negatives})
    reply = call({"frames": frames, "multimask": bool(static_boxes)})
    if not isinstance(reply, dict) or "masks" not in reply:
        raise RuntimeError(f"segment: {reply.get('error') if isinstance(reply, dict) else reply}")
    masks: dict[str, dict[str, np.ndarray]] = {}
    for key, per in reply["masks"].items():
        if key not in views:
            continue
        masks[key] = {}
        for name, got in per.items():
            if isinstance(got, dict):
                m = _choose(got.get("pngs") or [], got.get("scores") or [], views[key],
                            static_boxes)
                if m is not None:
                    masks[key][name] = m
                continue
            with Image.open(io.BytesIO(got)) as im:
                m = np.asarray(im.convert("L")) > 127
            if m.shape == (views[key].size[1], views[key].size[0]):
                masks[key][name] = m
    return ObjectMasks(views, masks, float(reply.get("seconds") or 0.0))


def _depth(frame: Frame):
    from PIL import Image

    with Image.open(frame.depth) as im:
        return np.asarray(im, dtype=np.float32) / 1000.0


def vote(positions: np.ndarray, masks: ObjectMasks, model: mujoco.MjModel, data: mujoco.MjData,
         bodies: dict[str, int], *, reach: float = REACH) -> np.ndarray:
    """Per splat: the body id its votes give it, -1 for none, -2 where no photo saw it (the
    caller keeps the boxes' answer there)."""
    pos = positions.astype(np.float64)
    owner = np.full(len(pos), -2, dtype=np.int64)
    near = np.zeros(len(pos), bool)
    for name, body in bodies.items():
        corners = _box_corners(model, data, body)
        lo, hi = corners.min(0) - reach, corners.max(0) + reach
        near |= np.all((pos >= lo) & (pos <= hi), axis=1)
    idx = np.flatnonzero(near)
    if not len(idx):
        return owner
    p = pos[idx]
    names = list(bodies)
    inside = np.zeros((len(names), len(idx)), np.int32)
    seen = np.zeros((len(names), len(idx)), np.int32)
    seen_any = np.zeros(len(idx), bool)
    for key, per in masks.masks.items():
        view = masks.views[key]
        W, H = view.size
        u, v, z = _project(view.K, view.frame.c2w, p)
        ok = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        depth = _depth(view.frame)
        dh, dw = depth.shape
        du = np.clip((u * dw / W).astype(int), 0, dw - 1)
        dv = np.clip((v * dh / H).astype(int), 0, dh - 1)
        d = depth[dv, du]
        ok &= (d <= 0) | (z <= d + VISIBLE_TOL[0] + VISIBLE_TOL[1] * d)
        ui, vi = np.clip(u.astype(int), 0, W - 1), np.clip(v.astype(int), 0, H - 1)
        for k, name in enumerate(names):
            m = per.get(name)
            box = view.boxes.get(name)
            if m is None or box is None:
                continue
            in_box = ok & (u >= box[0]) & (u <= box[2]) & (v >= box[1]) & (v <= box[3])
            seen[k] += in_box
            inside[k] += in_box & m[vi, ui]
            seen_any |= in_box
    ratio = inside / np.maximum(seen, 1)
    good = (inside >= MIN_VOTES) & (ratio >= MIN_RATIO)
    score = np.where(good, ratio, -1.0)
    best = np.argmax(score, axis=0)
    has = score[best, np.arange(len(idx))] > 0
    ids = np.array([bodies[n] for n in names])
    result = np.where(has, ids[best], np.where(seen_any, -1, -2))
    owner[idx] = result
    return owner
