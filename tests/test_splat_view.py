"""A newcomer's first view in a scanned room: the scan's best camera spot (models/splat/views.py),
carried in the bundle's splat.json and served to browsers (room_ops/roomcall/splats.py)."""

import json
import math

import numpy as np
from PIL import Image

from litereality_agent.models.splat import views
from litereality_agent.models.splat.rgbd import Frame
from litereality_agent.room_ops.roomcall.splats import SplatSource


def _look(yaw: float, pitch: float, pos=(0.0, 0.0, 1.5)) -> np.ndarray:
    """An OpenGL camera (looking down its -z) in the Z-up room, looking along yaw/pitch."""
    fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw),
                    math.sin(pitch)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -fwd, pos
    return c2w


def _frame(tmp_path, index: int, c2w, *, depth_m: float, sharp: bool) -> Frame:
    rng = np.random.default_rng(index)
    pixels = (rng.random((96, 128)) > 0.5).astype(np.uint8) * 255      # fine detail: sharp
    if not sharp:
        pixels = np.full((96, 128), 128, np.uint8)                        # featureless: "blurred"
    image = tmp_path / f"frame_{index:05d}.jpg"
    Image.fromarray(pixels).convert("RGB").save(image)
    depth = tmp_path / f"depth_{index:05d}.png"
    Image.fromarray(np.full((48, 64), int(depth_m * 1000), np.uint16)).save(depth)
    return Frame(index, np.eye(3), c2w, image, depth, None, (128, 96))


def test_the_pose_becomes_a_standing_view_in_the_rooms_convention():
    view = views.pose_view(_look(math.radians(90), math.radians(-10), pos=(1.0, 2.0, 1.4)))
    assert view["pos"] == [1.0, 2.0, 1.4]
    assert math.isclose(view["yaw"], math.radians(90), abs_tol=1e-3)
    assert math.isclose(view["pitch"], math.radians(-10), abs_tol=1e-3)


def test_the_best_view_is_level_sharp_and_looks_across_the_room(tmp_path):
    frames = [
        _frame(tmp_path, 0, _look(0.0, math.radians(-60)), depth_m=4.0, sharp=True),  # the floor
        _frame(tmp_path, 1, _look(1.0, 0.0), depth_m=0.6, sharp=True),               # a wall up close
        _frame(tmp_path, 2, _look(2.0, math.radians(-5)), depth_m=3.5, sharp=False),  # blurred
        _frame(tmp_path, 3, _look(3.0, math.radians(5)), depth_m=3.0, sharp=True),    # this one
    ]
    best = views.best_view(frames)
    assert best is not None and best["frame"] == 3
    assert math.isclose(best["yaw"], 3.0, abs_tol=1e-3)
    assert views.best_view(frames[:1]) is None                  # only a floor shot: no view


def test_the_server_passes_a_well_formed_view_and_drops_a_bad_one(tmp_path):
    folder = tmp_path / "room"
    folder.mkdir()
    (folder / "room-0123456789abcdef.spz").write_bytes(b"x")
    bundle = {"stage": "trained", "room": "room-0123456789abcdef.spz", "bodies": {},
              "created": 1.0, "view": {"pos": [1, 2, 1.5], "yaw": 0.5, "pitch": -0.1, "frame": 7}}
    (folder / "splat.json").write_text(json.dumps(bundle))
    source = SplatSource.fixed(folder)
    assert source.public(None, "room")["view"] == {"pos": [1.0, 2.0, 1.5], "yaw": 0.5,
                                                   "pitch": -0.1}
    bundle["view"] = {"pos": [1, 2], "yaw": "left", "pitch": None}
    (folder / "splat.json").write_text(json.dumps(bundle))
    import os
    os.utime(folder / "splat.json", (2, 2))                      # the cache re-reads on mtime
    assert "view" not in SplatSource.fixed(folder).public(None, "room")


def test_the_bundle_carries_the_view_it_is_given(tmp_path):
    from litereality_agent.models.splat.spz import Splats
    from litereality_agent.pipeline.splatroom import read_bundle, write_bundle
    from litereality_agent.pipeline.splatroom.split import Split

    n = 4
    room = Splats(np.zeros((n, 3), np.float32), np.full((n, 3), -3, np.float32),
                  np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1)),
                  np.full((n, 3), 0.5, np.float32), np.full(n, 0.9, np.float32))
    parts = Split(room=room, bodies={}, counts={})
    view = {"pos": [0.1, 0.2, 1.5], "yaw": 1.0, "pitch": -0.1, "frame": 3}
    write_bundle(tmp_path, parts, stage="instant", view=view)
    assert read_bundle(tmp_path)["view"] == view
    write_bundle(tmp_path, parts, stage="trained")
    assert "view" not in read_bundle(tmp_path)
