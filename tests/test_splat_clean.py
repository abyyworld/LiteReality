"""The splat room's cleanup (`models.splat.clean`) on a synthetic room with known answers.

A 4 x 4 x 2.5 m box room: splats on its floor, ceiling and walls (the surface), floaters in the
free space between the cameras and the walls, splats beyond the walls, and a ring of cameras in
the middle whose LiDAR (or dense) depth is rendered exactly from the box.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from litereality_agent.models.splat import clean as C
from litereality_agent.models.splat.rgbd import Frame
from litereality_agent.models.splat.spz import Splats

LO, HI = np.array([-2.0, -2.0, 0.0]), np.array([2.0, 2.0, 2.5])
SIZE = (1920, 1440)
K = np.array([[1400.0, 0.0, 959.5], [0.0, 1400.0, 719.5], [0.0, 0.0, 1.0]])


def _c2w(eye, yaw: float, pitch: float) -> np.ndarray:
    """OpenGL camera (looking down -z, y up) at `eye`, heading `yaw`, tilted by `pitch`."""
    f = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, :3] = np.stack([r, u, -f], axis=1)
    m[:3, 3] = eye
    return m


def _render(c2w: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Camera depth (metres) of the box room's inside at each pixel centre of a `shape` map."""
    h, w = shape
    s = w / SIZE[0]
    vv, uu = np.mgrid[0:h, 0:w]
    px, py = (uu + 0.5) / s - 0.5, (vv + 0.5) / s - 0.5
    ray = np.stack([(px - K[0, 2]) / K[0, 0], -(py - K[1, 2]) / K[1, 1], -np.ones_like(px)], -1)
    o = c2w[:3, 3]
    with np.errstate(all="ignore"):
        d = ray @ c2w[:3, :3].T              # camera z = -1: the ray parameter is camera depth
        t = np.maximum((LO - o) / d, (HI - o) / d)
    return np.nanmin(np.where(np.isfinite(t), t, np.inf), axis=-1).astype(np.float32)


def _frames(tmp: Path | None, count: int = 64) -> list[Frame]:
    """Cameras at head height, as a scan walks: every other one on a small ring looking out at
    the walls, the rest near the walls looking back across the room, tilted up or down; with
    `tmp`, each gets a LiDAR depth map (256 x 192, mm) and a confident confidence map there."""
    frames = []
    for i in range(count):
        yaw = 2 * np.pi * i / count
        if i % 2 == 0:
            eye = np.array([0.5 * np.cos(yaw), 0.5 * np.sin(yaw), 1.4])
            c2w = _c2w(eye, yaw, np.radians(-20 if i % 4 else -5))
        else:
            eye = np.array([1.7 * np.cos(yaw), 1.7 * np.sin(yaw), 1.4])
            c2w = _c2w(eye, yaw + np.pi, np.radians(15 if i % 4 == 1 else -30))
        depth = conf = None
        if tmp is not None:
            depth, conf = tmp / f"depth_{i:05d}.png", tmp / f"conf_{i:05d}.png"
            mm = np.round(_render(c2w, (192, 256)) * 1000).astype(np.uint16)
            Image.fromarray(mm).save(depth)
            Image.fromarray(np.full((192, 256), 2, np.uint8)).save(conf)
        frames.append(Frame(i, K.copy(), c2w, Path(f"frame_{i:05d}.jpg"),
                            depth or Path("missing.png"), conf, SIZE))
    return frames


def _splats(pos: np.ndarray, opacity: np.ndarray | None = None) -> Splats:
    n = len(pos)
    rot = np.zeros((n, 4), np.float32)
    rot[:, 3] = 1.0
    return Splats(pos.astype(np.float32), np.full((n, 3), 0.01, np.float32), rot,
                  np.full((n, 3), 0.5, np.float32),
                  np.full(n, 0.9, np.float32) if opacity is None else opacity.astype(np.float32))


def _surface(step: float = 0.1) -> np.ndarray:
    a = np.arange(-2.0 + step / 2, 2.0, step)
    z = np.arange(step / 2, 2.5, step)
    gx, gy = np.meshgrid(a, a)
    floor = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    ceiling = floor + [0.0, 0.0, 2.5]
    ga, gz = np.meshgrid(a, z)
    ga, gz = ga.ravel(), gz.ravel()
    walls = [np.stack([np.full_like(ga, x), ga, gz], 1) for x in (-2.0, 2.0)]
    walls += [np.stack([ga, np.full_like(ga, y), gz], 1) for y in (-2.0, 2.0)]
    return np.concatenate([floor, ceiling, *walls])


def _floaters(rng, frames, count: int) -> np.ndarray:
    eyes = np.array([f.c2w[:3, 3] for f in frames])
    out = []
    while len(out) < count:
        p = rng.uniform([-1.7, -1.7, 0.3], [1.7, 1.7, 2.2])
        if np.linalg.norm(eyes - p, axis=1).min() > 0.35:
            out.append(p)
    return np.array(out)


def _room() -> C.Room:
    square = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    window = C.RoomOpening((-0.5, 2.0), (0.5, 2.0), 1.0, 2.0, "window")
    door = C.RoomOpening((2.0, -0.5), (2.0, 0.3), 0.0, 2.1, "door")
    return C.Room([square], 0.0, 2.5, [window, door])


def _scene(rng, frames):
    surface = _surface()
    floaters = _floaters(rng, frames, 300)
    far_out = np.concatenate([
        np.stack([rng.uniform(2.4, 3.5, 100), rng.uniform(1.0, 1.9, 100),
                  rng.uniform(0.2, 2.3, 100)], 1),                       # beyond a wall
        np.stack([rng.uniform(-1.5, 1.5, 100), rng.uniform(-1.5, 1.5, 100),
                  rng.uniform(-0.9, -0.3, 100)], 1),                     # under the floor
    ])
    near_out = np.array([[2.1, 1.5, 1.0],          # just behind a wall: within the margin
                         [0.0, 2.12, 1.5],         # in the window's plane
                         [2.6, -0.1, 1.0]])        # where an outward-open door leaf swings
    pos = np.concatenate([surface, floaters, far_out, near_out])
    labels = np.concatenate([np.full(len(surface), 0), np.full(len(floaters), 1),
                             np.full(len(far_out), 2), np.full(len(near_out), 3)])
    return pos, labels


def _check(kept_pos, pos, labels):
    kept = {tuple(p) for p in np.round(kept_pos, 4)}
    alive = np.array([tuple(p) in kept for p in np.round(pos.astype(np.float32), 4)])
    assert alive[labels == 0].all(), f"{(~alive[labels == 0]).sum()} surface splats were dropped"
    assert alive[labels == 1].mean() < 0.05, f"{alive[labels == 1].mean():.0%} floaters survived"
    assert not alive[labels == 2].any()
    assert alive[labels == 3].all()


def test_cleans_the_room_with_lidar(tmp_path):
    rng = np.random.default_rng(0)
    frames = _frames(tmp_path)
    pos, labels = _scene(rng, frames)
    opacity = np.full(len(pos), 0.9)
    opacity[-10:-3] = 0.01                               # faint (some of the far-outside ones)
    out, stats = C.clean(_splats(pos, opacity), frames, bounds=_room())
    _check(out.positions, pos, labels)
    assert stats["faint"] == 7
    assert stats["outside"] == 200 - 7
    assert stats["carved"] >= 0.95 * 300
    assert stats["input"] == len(pos) and stats["kept"] == len(out)
    assert stats["carve_frames"] == 56 and stats["held_out_frames"] == 8


def test_cleans_the_room_with_dense_depth():
    rng = np.random.default_rng(1)
    frames = _frames(None)                    # no LiDAR on disk: every frame must use `dense`
    dense = {f.index: _render(f.c2w, (480, 640)) for f in frames}
    pos, labels = _scene(rng, frames)
    out, stats = C.clean(_splats(pos), frames, dense=dense, bounds=_room())
    _check(out.positions, pos, labels)
    assert stats["carve_frames"] == 56


def test_a_dense_map_the_lidar_contradicts_never_carves(tmp_path):
    """PromptDA reads a monitor screen (or glass) as farther than the confident LiDAR: a splat on
    the real surface would then be in front of the measured depth in every view of it. Carving
    trusts the dense map only where the LiDAR agrees, as the training set does."""
    frames = _frames(tmp_path, count=16)
    wall = np.array([[2.0 - 1e-3, y, z] for y in np.linspace(-1.0, 1.0, 9)
                     for z in np.linspace(0.6, 2.0, 8)])
    # The dense maps put the +x wall 60 cm too far: past the LiDAR's 5 cm / 5 % agreement.
    dense = {}
    for f in frames:
        d = _render(f.c2w, (480, 640))
        h, w = d.shape
        vv, uu = np.mgrid[0:h, 0:w]
        s = w / SIZE[0]
        ray = np.stack([((uu + 0.5) / s - 0.5 - K[0, 2]) / K[0, 0],
                        -((vv + 0.5) / s - 0.5 - K[1, 2]) / K[1, 1], -np.ones((h, w))], -1)
        hit = f.c2w[:3, 3] + d[..., None] * (ray @ f.c2w[:3, :3].T)
        dense[f.index] = np.where(hit[..., 0] > 1.99, d + 0.6, d).astype(np.float32)
    out, stats = C.clean(_splats(wall), frames, dense=dense, test_every=0)
    assert stats["carved"] == 0 and len(out) == len(wall)
    # Without the LiDAR to check against, the same maps would carve that wall.
    blind = [Frame(f.index, f.K, f.c2w, f.image, Path("missing.png"), None, f.size)
             for f in frames]
    _out, stats = C.clean(_splats(wall), blind, dense=dense, test_every=0)
    assert stats["carved"] > 0.5 * len(wall)


def test_a_long_scan_carves_with_a_spread_of_its_frames():
    frames = _frames(None, count=300)
    voters = C.spread(frames, 96)
    assert len(voters) == 96 and voters[0] is frames[0] and voters[-1] is frames[-1]
    assert C.spread(frames[:50], 96) == frames[:50]
    dense = {f.index: _render(f.c2w, (120, 160)) for f in frames}
    out, stats = C.clean(_splats(_surface()), frames, dense=dense)
    assert stats["carve_frames"] == C.CARVE_FRAMES and stats["carved"] == 0


def test_the_depth_cache_reads_each_frame_once(tmp_path, monkeypatch):
    frames = _frames(tmp_path, count=16)
    reads = []
    real = C.measured_depth
    monkeypatch.setattr(C, "measured_depth", lambda f, d=None: reads.append(f.index) or real(f, d))
    cache: dict = {}
    pos, _labels = _scene(np.random.default_rng(3), frames)
    first, s1 = C.clean(_splats(pos), frames, depth_cache=cache)
    second, s2 = C.clean(_splats(pos), frames, depth_cache=cache)
    assert len(reads) == 14 and s1["carved"] == s2["carved"] and len(first) == len(second)


def test_axis_aligned_bounds_and_opacity():
    pos = np.array([[0.0, 0.0, 1.0], [2.1, 0.0, 1.0], [2.3, 0.0, 1.0], [0.0, 0.0, -0.2]])
    out, stats = C.clean(_splats(pos, np.array([0.9, 0.9, 0.9, 0.9])), [],
                         bounds=(LO, HI), margin=0.15)
    np.testing.assert_allclose(out.positions, pos[:2].astype(np.float32))
    assert stats["outside"] == 2 and stats["carved"] == 0
    out, stats = C.clean(_splats(pos, np.array([0.01, 0.9, 0.9, 0.9])), [])
    assert len(out) == 3 and stats["faint"] == 1


def test_held_out_frames_never_carve():
    """A splat only the held-out photos see through stays: their depth must not shape the room
    they are scored against."""
    frames = [Frame(i, K.copy(), _c2w([0.0, 0.0, 1.4], 0.0, 0.0), Path(f"frame_{i:05d}.jpg"),
                    Path("missing.png"), None, SIZE) for i in range(16)]
    held = {f.index for j, f in enumerate(sorted(frames, key=C.image_name)) if j % 8 == 0}
    dense = {f.index: np.full((192, 256), 3.0 if f.index in held else 0.0, np.float32)
             for f in frames}
    floater = _splats(np.array([[1.5, 0.0, 1.4]]))
    out, stats = C.clean(floater, frames, dense=dense)
    assert len(out) == 1 and stats["carve_frames"] == 0
    out, stats = C.clean(floater, frames, dense=dense, test_every=0)
    assert len(out) == 0 and stats["carve_frames"] == 2


def test_training_frames_follow_the_training_set_order():
    frames = [Frame(i, K, np.eye(4), Path("x.jpg"), Path("d.png"), None, SIZE)
              for i in range(10)]
    assert [f.index for f in C.training_frames(frames)] == [1, 2, 3, 4, 5, 6, 7, 9]
    # The training set dropped frame 3 (too few depth points): frame 9 is now 8th, held out.
    names = [f"images/frame_{i:05d}.jpg" for i in range(10) if i != 3]
    assert [f.index for f in C.training_frames(frames, image_names=names)] == [1, 2, 4, 5, 6,
                                                                                7, 8]
    assert len(C.training_frames(frames, test_every=0)) == 10


def test_protected_splats_are_not_carved():
    frames = [Frame(i, K.copy(), _c2w([0.0, 0.0, 1.4], 0.0, 0.0), Path(f"frame_{i:05d}.jpg"),
                    Path("missing.png"), None, SIZE) for i in range(4)]
    dense = {f.index: np.full((192, 256), 3.0, np.float32) for f in frames}
    floaters = _splats(np.array([[1.5, 0.0, 1.4], [1.5, 0.05, 1.4]]))
    out, stats = C.clean(floaters, frames, dense=dense, test_every=0,
                         protect=np.array([True, False]))
    assert len(out) == 1 and stats["carved"] == 1
    np.testing.assert_allclose(out.positions[0], [1.5, 0.0, 1.4], atol=1e-6)


def test_room_from_a_roomplan():
    """`room_from_plan` reads `quickroom.roomplan.RoomPlan` by its attributes."""
    wall = SimpleNamespace(start=(0.0, 0.0), end=(4.0, 0.0), along=np.array([1.0, 0.0]))
    plan = SimpleNamespace(
        outlines=lambda: [np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])],
        floor_z=-1.4, ceiling_z=1.1, walls={"Wall0": wall},
        openings={"Wall0_Door_0": SimpleNamespace(wall="Wall0", offset=1.0, width=0.8,
                                                  height=2.0, sill=0.0, kind="door"),
                  "Wall9_Window_0": SimpleNamespace(wall="Wall9", offset=1.0, width=1.0,
                                                    height=1.0, sill=1.0, kind="window")})
    room = C.room_from_plan(plan)
    assert room.floor_z == -1.4 and room.ceiling_z == 1.1 and len(room.openings) == 1
    door = room.openings[0]
    np.testing.assert_allclose([*door.start, *door.end], [0.6, 0.0, 1.4, 0.0])
    assert (door.bottom, door.top, door.kind) == (-1.4, pytest.approx(0.6), "door")
    keep = C.inside_bounds(np.array([[1.0, -0.6, 0.0], [3.0, -0.6, 0.0], [2.0, 1.5, 1.5]]), room)
    assert keep.tolist() == [True, False, False]


def test_a_million_splats_in_seconds():
    rng = np.random.default_rng(2)
    frames = _frames(None, count=60)
    dense = {f.index: _render(f.c2w, (720, 960)) for f in frames}
    n = 1_000_000
    surface = _surface(0.02)
    pos = np.concatenate([surface[rng.integers(0, len(surface), int(n * 0.7))],
                          rng.uniform(LO - 0.5, HI + 0.5, (n - int(n * 0.7), 3))])
    t = time.perf_counter()
    out, stats = C.clean(_splats(pos), frames, dense=dense, bounds=_room())
    elapsed = time.perf_counter() - t
    assert elapsed < 10.0, f"{elapsed:.1f} s for 1M splats x 60 frames"
    assert stats["carve_frames"] == 60 - 8
    assert stats["carved"] > 0 and len(out) >= int(n * 0.7)


def test_a_million_splats_of_a_300_photo_scan_in_seconds():
    """A long scan costs what ~100 frames cost (`CARVE_FRAMES`), not 300 frames' projection."""
    rng = np.random.default_rng(4)
    frames = _frames(None, count=300)
    voters = C.spread(C.training_frames(frames))             # only theirs is ever read
    dense = {f.index: _render(f.c2w, (720, 960)) for f in voters}
    n = 1_000_000
    surface = _surface(0.02)
    pos = np.concatenate([surface[rng.integers(0, len(surface), int(n * 0.7))],
                          rng.uniform(LO - 0.5, HI + 0.5, (n - int(n * 0.7), 3))])
    t = time.perf_counter()
    out, stats = C.clean(_splats(pos), frames, dense=dense, bounds=_room())
    elapsed = time.perf_counter() - t
    assert elapsed < 10.0, f"{elapsed:.1f} s for 1M splats x 300 frames"
    assert stats["carve_frames"] == C.CARVE_FRAMES and stats["held_out_frames"] == 300 - 262
    assert stats["carved"] > 0 and len(out) >= int(n * 0.7)


# ── agreement: kept only where enough training photos saw a splat on its surface ─────────────────


def _facing(yaws: list[float], tmp_eyes=None) -> list[Frame]:
    """Cameras at head height near the middle, one per yaw, level: a constant dense map of 2 m
    then puts their surface on the plane 2 m ahead (x = +-2 for yaw 0 / pi)."""
    return [Frame(i, K.copy(), _c2w([0.0, 0.1 * (i % 3 - 1), 1.4], yaw, 0.0),
                  Path(f"frame_{i:05d}.jpg"), Path("missing.png"), None, SIZE)
            for i, yaw in enumerate(yaws)]


def test_agreement_alone_keeps_the_surface_and_drops_what_no_photo_saw_on_it(tmp_path):
    """On its own (the other rules off), agreement keeps every splat on the box's surfaces and
    drops every floater and everything beyond the walls: no photo saw any of them on a surface."""
    rng = np.random.default_rng(0)
    frames = _frames(tmp_path)
    pos, labels = _scene(rng, frames)
    out, stats = C.clean(_splats(pos), frames, min_opacity=0.0, carve_votes=0, support_views=3)
    kept = {tuple(p) for p in np.round(out.positions, 4)}
    alive = np.array([tuple(p) in kept for p in np.round(pos.astype(np.float32), 4)])
    assert alive[labels == 0].all()
    assert not alive[labels == 1].any() and not alive[labels == 2].any()
    assert not alive[(labels == 3) & (pos[:, 0] > 2.5)].any()      # a door leaf's swing: air
    assert (stats["faint"], stats["outside"], stats["carved"]) == (0, 0, 0)
    assert stats["unsupported"] == len(pos) - len(out) and stats["support_views"] == 3
    assert stats["carve_frames"] == 56 and stats["held_out_frames"] == 8


def test_a_splat_too_few_photos_saw_on_its_surface_goes():
    frames = _facing([0.0, 0.0, 0.0, np.pi, np.pi])       # three look at x = +2, two at x = -2
    dense = {f.index: np.full((192, 256), 2.0, np.float32) for f in frames}
    pos = np.array([[2.0, 0.0, 1.4],       # on the +x surface in three photos
                    [2.04, 0.1, 1.4],      # 4 cm behind it: within 5 cm, still on it
                    [-2.0, 0.0, 1.4],      # on the -x surface in two photos only
                    [2.3, 0.0, 1.4],       # hidden behind the surface
                    [1.5, 0.0, 1.4]])      # floating in front of it
    out, stats = C.clean(_splats(pos), frames, dense=dense, test_every=0, carve_votes=0,
                         support_views=3)
    np.testing.assert_allclose(out.positions, pos[:2].astype(np.float32), atol=1e-6)
    assert stats["unsupported"] == 3 and stats["carved"] == 0
    out, stats = C.clean(_splats(pos), frames, dense=dense, test_every=0, carve_votes=0,
                         support_views=2)
    np.testing.assert_allclose(out.positions, pos[:3].astype(np.float32), atol=1e-6)
    # With carving too, the floater is counted once, as carved.
    out, stats = C.clean(_splats(pos), frames, dense=dense, test_every=0, carve_votes=2,
                         support_views=3)
    assert len(out) == 2 and stats["carved"] == 1 and stats["unsupported"] == 2
    # Off (the default): nothing is dropped for want of agreement.
    out, stats = C.clean(_splats(pos), frames, dense=dense, test_every=0, carve_votes=0)
    assert len(out) == len(pos) and stats["unsupported"] == 0 and "support_views" not in stats


def test_held_out_frames_never_vote_for_agreement():
    """The trainer's held-out photo (the first by name) seeing a splat on its surface does not
    count: two training photos are not three."""
    frames = _facing([0.0, 0.0, 0.0] + [np.pi] * 5)
    sees = {0, 1, 2}
    dense = {f.index: np.full((192, 256), 2.0 if f.index in sees or f.index > 2 else 0.0,
                              np.float32) for f in frames}
    splat = _splats(np.array([[2.0, 0.0, 1.4]]))
    out, stats = C.clean(splat, frames, dense=dense, carve_votes=0, support_views=3)
    assert len(out) == 0 and stats["held_out_frames"] == 1 and stats["carve_frames"] == 7
    out, stats = C.clean(splat, frames, dense=dense, test_every=0, carve_votes=0,
                         support_views=3)
    assert len(out) == 1


def test_agreement_never_empties_a_room_without_depth():
    frames = _facing([0.0] * 4)                            # no LiDAR on disk, no dense maps
    out, stats = C.clean(_splats(np.array([[2.0, 0.0, 1.4]])), frames, test_every=0,
                         support_views=3)
    assert len(out) == 1 and stats["carve_frames"] == 0 and stats["support_views"] == 0


def test_protected_splats_need_no_agreement():
    frames = _facing([0.0] * 4)
    dense = {f.index: np.full((192, 256), 2.0, np.float32) for f in frames}
    pos = np.array([[1.5, 0.0, 1.4], [1.5, 0.05, 1.4]])
    out, stats = C.clean(_splats(pos), frames, dense=dense, test_every=0, carve_votes=0,
                         support_views=3, protect=np.array([True, False]))
    assert len(out) == 1 and stats["unsupported"] == 1
    np.testing.assert_allclose(out.positions[0], pos[0], atol=1e-6)


def test_the_agreement_bar_follows_the_share_of_frames_that_vote():
    assert C.support_need(3, 56, 56, 56) == 3
    assert C.support_need(3, 96, 262, 96) == 2         # 3 of 262 photos is ~1.1 of 96 voters
    assert C.support_need(3, 96, 96 * 3, 96) == 1
    assert C.support_need(3, 2, 2, 2) == 2             # two photos cannot ask for three
    assert C.support_need(3, 10, 10, 0) == 0           # no depth: no agreement to ask for
    assert C.support_need(0, 56, 56, 56) == 0


def test_a_million_splats_with_agreement_in_seconds():
    """Agreement rides on carving's projection: 1M splats x 60 frames with every rule on."""
    rng = np.random.default_rng(5)
    frames = _frames(None, count=60)
    dense = {f.index: _render(f.c2w, (720, 960)) for f in frames}
    n = 1_000_000
    surface = _surface(0.02)
    pos = np.concatenate([surface[rng.integers(0, len(surface), int(n * 0.7))],
                          rng.uniform(LO - 0.5, HI + 0.5, (n - int(n * 0.7), 3))])
    t = time.perf_counter()
    out, stats = C.clean(_splats(pos), frames, dense=dense, bounds=_room(), support_views=3)
    elapsed = time.perf_counter() - t
    assert elapsed < 10.0, f"{elapsed:.1f} s for 1M splats x 60 frames"
    assert stats["carve_frames"] == 60 - 8 and stats["unsupported"] > 0
    assert len(out) >= int(n * 0.69)
