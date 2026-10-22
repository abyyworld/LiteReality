"""The evaluation harness: the numbers a paper reports must come from maths that is checked.

Default tests are offline and need no browser:

* the pose maths — the render page's matrices put a point on the same pixel as the pinhole model,
  and that pinhole model inverts the quick room's own unprojection of a frame's depth (the path
  the photo walls were validated on), so all three agree;
* the region masks, the metrics (PSNR, SSIM, the exposure match) and the contact sheet;
* the physics audit on small hand-written scenes where the right answer is known: a lift, a
  tucked chair that has to be dragged, a jammed drawer, a 130 kg door, an authored overlap, a box
  creeping down a ramp;
* the reports, the comparison table and the CLI.

Rendering in headless Chromium is opt-in (``-m render``; needs ``uv run --with playwright`` and a
Chromium, see `fidelity.PLAYWRIGHT_HELP`).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

# The quick room's own synthetic RoomPlan export (a 5 x 4 m room with a table, chairs, a door).
from test_quickroom import FLOOR_Y, write_roomplan_usdz

from litereality_agent import cli
from litereality_agent.pipeline.evaluate import fidelity, holdout, physics, report
from litereality_agent.pipeline.quickroom import colours, parse_room
from litereality_agent.pipeline.scene_init.layout.shell import ARKIT_TO_Z_UP
from litereality_agent.room_ops.roomcall.demo import write_demo_scene
from litereality_agent.room_ops.roomcall.sim import RoomSim

FAST = {"stability_seconds": 1.0, "cost_seconds": 0.5, "cost_repeats": 1, "measure_concurrency": False}
K_FULL = np.array([[480.0, 0.0, 322.0], [0.0, 480.0, 238.0], [0.0, 0.0, 1.0]])
FULL = (640, 480)


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────


def _look(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera -> world (Z-up) looking from ``eye`` at ``target``: -Z forward, +Y up."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    back = eye - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, np.cross(back, right), back, eye
    return pose


def _arkit(pose_zup: np.ndarray) -> list[float]:
    """The Z-up pose as the scanner writes it: camera -> ARKit world, row-major."""
    return (np.linalg.inv(ARKIT_TO_Z_UP) @ pose_zup).reshape(-1).tolist()


def _write_frame(folder: Path, index: int, pose_zup: np.ndarray, *, image=True,
                 depth_m: float | None = None, rgb=(120, 130, 140)) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{index:05d}"
    (folder / f"frame_{stem}.json").write_text(json.dumps(
        {"cameraPoseARFrame": _arkit(pose_zup), "intrinsics": K_FULL.reshape(-1).tolist()}))
    if image:
        Image.new("RGB", FULL, rgb).save(folder / f"frame_{stem}.jpg", quality=95)
    if depth_m is not None:
        depth = np.full((12, 16), int(depth_m * 1000), dtype=np.uint16)
        Image.fromarray(depth).save(folder / f"depth_{stem}.png")


def _view(pose_zup: np.ndarray, image: Path = Path("unused.jpg")) -> fidelity.View:
    return fidelity.View(0, image, K_FULL.copy(), FULL, pose_zup)


# ── pose maths ───────────────────────────────────────────────────────────────────────────────────


def test_the_render_pages_matrices_and_the_pinhole_model_put_points_on_the_same_pixel():
    rng = np.random.default_rng(3)
    pose = _look((1.0, -2.0, 1.4), (2.0, 1.0, 0.8))
    view = _view(pose)
    pts = pose[:3, 3] + rng.uniform(-1, 1, (50, 3)) + (pose[:3, :3] @ [0, 0, -3.0])
    for size in ((320, 240), (640, 480), (100, 75)):
        uv, depth = fidelity.project(pts, view, size)
        three = fidelity.project_three(pts, fidelity.three_camera(view, size), size)
        assert np.all(depth > 0)
        np.testing.assert_allclose(three, uv, atol=1e-6)


def test_a_point_straight_ahead_lands_on_the_principal_point_and_up_is_up():
    """In the camera's own frame: forward is -Z, up is +Y, and image v counts down."""
    pose = _look((0.0, 0.0, 1.5), (0.0, 5.0, 1.5))
    view = _view(pose)
    ahead = pose[:3, 3] + pose[:3, :3] @ [0.0, 0.0, -2.0]
    above = pose[:3, 3] + pose[:3, :3] @ [0.0, 0.2, -2.0]
    right = pose[:3, 3] + pose[:3, :3] @ [0.1, 0.0, -2.0]
    uv, _ = fidelity.project(np.array([ahead, above, right]), view, FULL)
    np.testing.assert_allclose(uv[0], K_FULL[:2, 2], atol=1e-9)
    assert uv[1][1] == pytest.approx(K_FULL[1, 2] - 480.0 * 0.1)     # higher in the room: up
    assert uv[2][0] == pytest.approx(K_FULL[0, 2] + 480.0 * 0.05)    # +X: right
    # The camera looks along world +Y here, so Z-up "up" must be image up.
    assert np.allclose(pose[:3, 1], [0, 0, 1])


def test_projection_inverts_the_quick_rooms_own_unprojection_of_a_frame(tmp_path):
    """`quickroom.colours` turns a frame's depth pixels into room points (the path the photo walls
    are built on); projecting those points back must land on the same pixel centres, through
    both the pinhole model and the render page's matrices — the ARKit -> Z-up step included."""
    pose = _look((1.0, 0.5, 0.4), (3.0, 2.0, -0.2))
    _write_frame(tmp_path, 7, pose, depth_m=2.0)
    frame = colours.capture_frames(tmp_path)[0]
    world, _rgb = colours._read_frame(frame)
    (view,) = fidelity.capture_views(tmp_path)
    np.testing.assert_allclose(view.pose, pose, atol=1e-9)
    size = (16, 12)                                   # the depth map's resolution
    ys, xs = np.mgrid[0:12, 0:16]
    expected = np.column_stack([xs.ravel() + 0.5, ys.ravel() + 0.5])
    uv, depth = fidelity.project(world, view, size)
    np.testing.assert_allclose(uv, expected, atol=1e-6)
    assert np.all(depth > 0)
    three = fidelity.project_three(world, fidelity.three_camera(view, size), size)
    np.testing.assert_allclose(three, expected, atol=1e-6)


def test_capture_views_needs_a_photo_and_spreads_a_limit_over_the_capture(tmp_path):
    for i in range(10):
        _write_frame(tmp_path, i, _look((0, 0, 1), (0, 1, 1)), image=i != 3)
    (tmp_path / "frame_00099.json").write_text("{not json")
    Image.new("RGB", FULL).save(tmp_path / "frame_00099.jpg")
    views = fidelity.capture_views(tmp_path)
    assert [v.index for v in views] == [0, 1, 2, 4, 5, 6, 7, 8, 9]
    assert views[0].full == FULL
    picked = fidelity.capture_views(tmp_path, limit=3)
    assert [v.index for v in picked] == [0, 5, 9]


def test_intrinsics_scale_with_the_resolution():
    view = _view(np.eye(4))
    fx, fy, cx, cy = fidelity.intrinsics_at(view, (320, 240))
    assert (fx, fy, cx, cy) == (240.0, 240.0, 161.0, 119.0)


# ── regions ──────────────────────────────────────────────────────────────────────────────────────


def test_box_masks_are_silhouettes_and_skip_a_box_the_camera_stands_in():
    pose = _look((0.0, 0.0, 1.0), (0.0, 3.0, 1.0))
    view = _view(pose)
    size = (64, 48)
    ahead = fidelity.Box("Chair0", np.array([0.0, 3.0, 1.0]), np.array([0.3, 0.3, 0.3]), 0.3)
    mask = fidelity.box_mask([ahead], view, size)
    assert mask[24, 32] and not mask[0, 0] and not mask[47, 63]
    # The silhouette matches the projected corners' extent (to a pixel; the box is yawed).
    c, s = math.cos(0.3), math.sin(0.3)
    corners = np.array([[x, y, z] for x in (-.3, .3) for y in (-.3, .3) for z in (-.3, .3)])
    corners = corners @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + ahead.centre
    uv, _ = fidelity.project(corners, view, size)
    cols = np.flatnonzero(mask.any(axis=0))
    assert abs(cols.min() - uv[:, 0].min()) < 1.0 and abs(cols.max() + 1 - uv[:, 0].max()) < 1.0
    behind = fidelity.Box("B", np.array([0.0, -3.0, 1.0]), np.array([0.3, 0.3, 0.3]), 0.0)
    assert not fidelity.box_mask([behind], view, size).any()
    around = fidelity.Box("Door", np.array([0.0, 0.0, 1.0]), np.array([0.4, 0.15, 1.0]), 0.0)
    assert not fidelity.box_mask([around], view, size).any()


def test_scan_boxes_reads_objects_and_cuts_openings_in_their_walls(tmp_path):
    usdz = write_roomplan_usdz(tmp_path / "scan" / "room.usdz")
    plan = parse_room(usdz)
    objects, openings = fidelity.scan_boxes(plan)
    assert {b.name for b in objects} == set(plan.objects)
    door = next(b for b in openings if "Door" in b.name)
    # Wall0 runs (0,0)->(5,0); the door is centred 1 m along it, 2.05 m tall from the floor.
    np.testing.assert_allclose(door.centre[:2], [1.0, 0.0], atol=1e-3)
    assert door.centre[2] == pytest.approx(FLOOR_Y + 2.05 / 2, abs=1e-3)
    np.testing.assert_allclose(door.half, [0.45, fidelity.OPENING_DEPTH, 2.05 / 2], atol=1e-3)


def _scene_depth(view: fidelity.View, size, boxes, floor_z: float) -> np.ndarray:
    """What a LiDAR would return for a room of solid ``boxes`` on a floor at ``floor_z``: per
    pixel, the camera-z distance to the nearest box face or the floor."""
    _hit, entry = fidelity.box_hits(boxes, view, size)
    origin, dirs = fidelity.pixel_rays(view, size)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = (floor_z - origin[2]) / dirs[:, 2]
    t_floor = np.where(t_floor > 0, t_floor, np.inf).reshape(entry.shape)
    return np.minimum(entry, t_floor)


def _write_depth(path: Path, depth: np.ndarray) -> None:
    mm = np.where(np.isfinite(depth), np.clip(depth * 1000.0, 0, 65000), 0).astype(np.uint16)
    Image.fromarray(mm).save(path)


def test_lidar_object_pixels_are_what_the_camera_saw_of_the_object(tmp_path):
    """A table seen from the side, as a LiDAR sees it: its pixels are objects. Where the depth
    runs on past the box to the floor (between the legs) they are not, however the silhouette
    covers them; and a table hidden behind something RoomPlan never boxed is not counted."""
    usdz = write_roomplan_usdz(tmp_path / "scan" / "room.usdz")
    plan = parse_room(usdz)
    objects, openings = fidelity.scan_boxes(plan)
    table = next(b for b in objects if b.name.startswith("Table"))
    eye = table.centre + np.array([0.0, -1.8, 0.4])
    view = _view(_look(eye, table.centre))
    size = (64, 48)
    depth = _scene_depth(view, size, [table], plan.floor_z)
    silhouette = fidelity.box_mask([table], view, size)
    assert silhouette.sum() > 200
    # Between the legs: below the top's underside, the depth goes through to the floor.
    origin, dirs = fidelity.pixel_rays(view, size)
    with np.errstate(divide="ignore", invalid="ignore"):
        to_floor = ((plan.floor_z - origin[2]) / dirs[:, 2]).reshape(size[1], size[0])
    under = silhouette & (np.arange(size[1])[:, None] > np.flatnonzero(silhouette.any(1)).mean())
    depth[under] = to_floor[under]
    depth_path = tmp_path / "depth.png"
    _write_depth(depth_path, depth)
    seen = fidelity.View(0, Path("unused.jpg"), K_FULL.copy(), FULL, view.pose, depth_path)
    reg = fidelity.view_regions(seen, plan, [table], openings, size)
    assert reg["mask"] == "depth" and reg["depth_valid"] > 0.5      # the rest looks over the floor
    top = silhouette & ~under
    assert reg["objects"][top].mean() > 0.95            # the table as seen
    assert reg["objects"][under].mean() < 0.05           # the floor between its legs
    assert not (reg["objects"] & reg["static"]).any()
    # Something unboxed 1 m in front of the camera hides the table entirely.
    _write_depth(depth_path, np.where(silhouette, 1.0, depth))
    hidden = fidelity.view_regions(seen, plan, [table], openings, size)
    assert not hidden["objects"][silhouette].any()
    # Without a depth map the silhouette is all there is.
    blind = fidelity.view_regions(view, plan, [table], openings, size)
    assert blind["mask"] == "box" and (blind["objects"] == silhouette).all()


# ── which views a build saw, and the hold-out ────────────────────────────────────────────────────


def test_kfold_splits_every_frame_into_exactly_one_fold():
    frames = [0, 2, 3, 4, 6, 7, 8, 9, 13, 15]
    inter = holdout.kfold(frames, 3, "interleaved")
    assert inter == [[0, 4, 8, 15], [2, 6, 9], [3, 7, 13]]
    block = holdout.kfold(frames, 3, "block")
    assert block == [[0, 2, 3, 4], [6, 7, 8], [9, 13, 15]]
    for folds in (inter, block):
        assert sorted(i for f in folds for i in f) == frames
    assert len(holdout.kfold([1, 2], 5)) == 2
    with pytest.raises(ValueError, match="split"):
        holdout.kfold(frames, 3, "random")


def test_scan_without_leaves_out_every_file_of_the_held_out_frames(tmp_path):
    scan = tmp_path / "scan"
    write_roomplan_usdz(scan / "roomplan" / "room.usdz")
    for i in range(4):
        _write_frame(scan, i, _look((0, 0, 1), (0, 1, 1)), depth_m=2.0)
        (scan / f"conf_{i:05d}.png").write_bytes(b"x")
    copy = holdout.scan_without(scan, {1, 3}, tmp_path / "copy")
    assert holdout.frames_present(copy) == {0, 2}
    assert (copy / "roomplan" / "room.usdz").is_file()
    assert not any(p.name.endswith("00001.jpg") or "00003" in p.name for p in copy.iterdir())
    assert holdout.frames_present(scan) == {0, 1, 2, 3}            # the original is untouched
    assert [v.index for v in fidelity.capture_views(copy)] == [0, 2]


@pytest.fixture()
def depth_scan(tmp_path) -> Path:
    """The synthetic room with six frames that have depth, so the quick room reads them."""
    scan = tmp_path / "scans" / "depth_room"
    write_roomplan_usdz(scan / "room.usdz")
    for i in range(6):
        pose = _look((1.0 + 0.5 * i, 0.6, FLOOR_Y + 1.4), (2.5, 2.5, FLOOR_Y + 0.5))
        _write_frame(scan, i, pose, depth_m=2.0, rgb=(90 + 20 * i, 120, 150))
    return scan


class FlatRenderer:
    """One grey image per camera asked for (any number of them): plumbing, not pixels."""

    name = "flat"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def render(self, glb: bytes, cameras: list[dict], size: tuple[int, int]) -> list[np.ndarray]:
        assert glb[:4] == b"glTF"
        self.calls.append(len(cameras))
        return [np.full((size[1], size[0], 4), 128, dtype=np.uint8) for _ in cameras]


def test_build_sources_names_the_frames_the_quick_room_read(depth_scan, tmp_path):
    from litereality_agent.pipeline.quickroom import write_quick_scene

    scene = write_quick_scene(depth_scan, tmp_path / "quick")
    sources = holdout.build_sources(scene, depth_scan)
    assert sources.recorded
    assert sources.frames["textures"] == list(range(6)) == sources.frames["colours"]
    assert sources.seen(3) is True and sources.roles(3) == ["textures", "colours"]
    flat = write_quick_scene(depth_scan, tmp_path / "flat", textures=False)
    assert "textures" not in holdout.build_sources(flat, depth_scan).frames
    rebuild, options = holdout.quick_rebuilder(flat)
    assert rebuild is not None and options["textures"] is False and options["colours"] is True
    unrecorded = holdout.build_sources(write_demo_scene(tmp_path / "demo"), depth_scan)
    assert not unrecorded.recorded and unrecorded.seen(0) is None


def test_the_objects_stage_counts_its_crop_views_as_seen(tmp_path, depth_scan):
    folder = tmp_path / "objects"
    write_demo_scene(folder)
    (folder / "objects.json").write_text(json.dumps(
        {"crops": {"Chair0": {"frame": 4, "views": [4, 5]}}}))
    sources = holdout.build_sources(folder / "scene.xml", depth_scan)
    assert sources.frames == {"crops": [4, 5]} and sources.seen(1) is False
    rebuild, info = holdout.quick_rebuilder(folder / "scene.xml")
    assert rebuild is None and "not a quick room" in info["reason"]


def test_holdout_scores_every_view_against_a_room_built_without_it(depth_scan, tmp_path):
    from litereality_agent.pipeline.quickroom import write_quick_scene

    scene = write_quick_scene(depth_scan, tmp_path / "quick")
    renderer = FlatRenderer()
    out = fidelity.evaluate_fidelity(scene, depth_scan, renderer=renderer, size=(32, 24),
                                     out_dir=tmp_path / "eval", contact_views=2, holdout=3)
    assert out["views_seen"] == {"seen": 6, "unseen": 0, "unknown": 0}
    assert all(r["seen"] and r["read_by"] == ["textures", "colours"] for r in out["per_view"])
    assert out["splits"]["unseen"] is None and out["splits"]["seen"]["psnr"]["n"] == 6
    hold = out["holdout"]
    assert hold["k"] == 3 and hold["views"] == 6 and hold["split"] == "interleaved"
    assert [f["excluded"] for f in hold["folds"]] == [[0, 3], [1, 4], [2, 5]]
    for fold in hold["folds"]:
        assert fold["leaked"] == [] and fold["textures_frames"] == [4, 4]
    # Every view scored once, by the fold that left it out, which never saw it.
    assert sorted(r["frame"] for r in hold["per_view"]) == list(range(6))
    assert all(r["seen"] is False and r["fold"] == r["frame"] % 3 for r in hold["per_view"])
    assert renderer.calls == [6, 2, 2, 2]
    assert set(hold["gap"]) >= {"psnr", "ssim", "psnr_static"}
    assert (tmp_path / "eval" / "contact_sheet_holdout.jpg").is_file()


def test_holdout_is_skipped_with_its_reason_for_a_stage_it_cannot_rebuild(synthetic_scan,
                                                                          tmp_path):
    scene = write_demo_scene(tmp_path / "demo")
    out = fidelity.evaluate_fidelity(scene, synthetic_scan, renderer=FlatRenderer(),
                                     size=(32, 24), holdout=4)
    assert "not a quick room" in out["holdout"]["skipped"]
    assert out["views_seen"]["unknown"] == 3                   # no record: nothing called unseen


# ── metrics ──────────────────────────────────────────────────────────────────────────────────────


def test_psnr_and_ssim_on_known_images():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.2, 0.8, (48, 64, 3))
    assert fidelity.psnr(img, img) == fidelity.MAX_PSNR
    assert fidelity.ssim(img, img) == pytest.approx(1.0)
    shifted = np.clip(img + 0.1, 0, 1)
    assert fidelity.psnr(img, shifted) == pytest.approx(20.0, abs=0.05)   # MSE 0.01
    noisy = np.clip(img + rng.normal(0, 0.1, img.shape), 0, 1)
    assert 0.0 < fidelity.ssim(img, noisy) < 0.9
    mask = np.zeros((48, 64), dtype=bool)
    mask[:, :32] = True
    half = img.copy()
    half[:, 32:] = 0.0
    assert fidelity.psnr(img, half, mask) == fidelity.MAX_PSNR
    assert fidelity.psnr(img, half, ~mask) < 15
    assert fidelity.psnr(img, half, np.zeros_like(mask)) is None


def test_exposure_matching_forgives_a_global_gain_and_nothing_else():
    rng = np.random.default_rng(1)
    real = rng.uniform(0.1, 0.6, (48, 64, 3))
    darker = real * 0.6
    row = fidelity.compare(real, darker, {"static": np.ones((48, 64), bool),
                                          "objects": np.zeros((48, 64), bool)})
    assert row["psnr_gain"] > 60 > row["psnr"]
    assert row["psnr_objects"] is None and row["frac_objects"] == 0.0
    assert row["psnr_static"] == pytest.approx(row["psnr"], abs=1e-3)
    assert row["psnr_baseline"] < 30


def test_composite_puts_the_render_over_the_calls_background_and_reports_coverage():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:2, :, :] = (255, 0, 0, 255)
    rgb, cover = fidelity.composite(rgba, (2, 2))
    np.testing.assert_allclose(rgb[0, 0], [1, 0, 0])
    np.testing.assert_allclose(rgb[1, 0], np.asarray(fidelity.BACKGROUND) / 255.0)
    np.testing.assert_allclose(cover, [[1, 1], [0, 0]])


# ── fidelity, end to end with a stand-in renderer ────────────────────────────────────────────────


class PhotoRenderer:
    """Returns each view's own photo: a perfect room, so every metric must say so."""

    name = "photos"

    def __init__(self, views: list[fidelity.View]) -> None:
        self.views = views
        self.calls: list[tuple[int, tuple[int, int]]] = []

    def render(self, glb: bytes, cameras: list[dict], size: tuple[int, int]) -> list[np.ndarray]:
        assert glb[:4] == b"glTF"
        self.calls.append((len(cameras), size))
        out = []
        for view in self.views:
            with Image.open(view.image) as im:
                out.append(np.asarray(im.convert("RGBA").resize(size, Image.Resampling.BOX)))
        return out


@pytest.fixture()
def synthetic_scan(tmp_path) -> Path:
    """The quick room tests' RoomPlan export plus three frames looking at its table."""
    scan = tmp_path / "scans" / "test_room"
    write_roomplan_usdz(scan / "room.usdz")
    floor = FLOOR_Y
    for i, eye in enumerate(((2.5, 0.4, floor + 1.4), (1.0, 1.0, floor + 1.5),
                             (4.0, 0.8, floor + 1.3))):
        pose = _look(eye, (2.5, 2.0, floor + 0.5))
        rgb = (90 + 40 * i, 120, 150)
        _write_frame(scan, i, pose, rgb=rgb)
    return scan


def test_evaluate_fidelity_scores_a_perfect_render_perfectly(synthetic_scan, tmp_path):
    scene = write_demo_scene(tmp_path / "demo")
    views = fidelity.capture_views(synthetic_scan)
    renderer = PhotoRenderer(views)
    out = fidelity.evaluate_fidelity(scene, synthetic_scan, renderer=renderer, size=(80, 60),
                                     out_dir=tmp_path / "eval", contact_views=2, title="t")
    assert out["views"] == 3 and renderer.calls == [(3, (160, 120))]
    s = out["summary"]
    assert s["psnr"]["mean"] > 40 and s["ssim"]["mean"] > 0.99
    assert s["coverage"]["mean"] == pytest.approx(1.0)
    # Every view looks at the table: object pixels exist, and so do walls and floor.
    assert all(r["frac_objects"] > 0.05 and r["frac_static"] > 0.05 for r in out["per_view"])
    assert s["psnr_objects"]["n"] == 3
    assert (tmp_path / "eval" / out["contact_sheet"]).is_file()
    with Image.open(tmp_path / "eval" / "contact_sheet.jpg") as im:
        assert im.width == 3 * 80 + 4 * 4


def test_evaluate_fidelity_needs_frames(tmp_path):
    scan = tmp_path / "scan"
    write_roomplan_usdz(scan / "room.usdz")
    with pytest.raises(ValueError, match="no frame"):
        fidelity.evaluate_fidelity(write_demo_scene(tmp_path / "d"), scan,
                                   renderer=PhotoRenderer([]))


def test_rendering_without_playwright_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(fidelity.RendererUnavailable, match="--with playwright"):
        fidelity.PlaywrightRenderer()


# ── physics ──────────────────────────────────────────────────────────────────────────────────────

ROOM = """
<mujoco model="audit_room">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="4 4 0.1"/>
    <geom name="wall" type="box" pos="0 -1.0 1" size="2 0.05 1"/>
    <!-- 2 cm into the wall at t=0 -->
    <body name="Chair0" pos="0 -0.77 0.301">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.3" mass="5"/>
    </body>
    <!-- a leaf twice the height of a person and 130 kg: the tea_room bug -->
    <body name="Door0_Door_leaf" pos="1.5 1.5 0">
      <joint name="door_hinge" type="hinge" axis="0 0 1" range="0 100" damping="2"/>
      <geom type="box" pos="0.4 0 1.01" size="0.4 0.02 1" mass="130"/>
    </body>
    <!-- a drawer whose runner is blocked 3 cm out -->
    <geom name="stop" type="box" pos="-1.0 1.0 0.5" size="0.02 0.3 0.2"/>
    <body name="Cabinet0_drawer" pos="-1.25 1.0 0.5">
      <joint name="drawer_slide" type="slide" axis="1 0 0" range="0 0.4" damping="5"/>
      <geom type="box" size="0.2 0.2 0.05" mass="3"/>
    </body>
    <body name="Sofa0" pos="2.5 -2 0.4">
      <geom type="box" size="0.9 0.4 0.4" mass="3"/>
    </body>
  </worldbody>
</mujoco>
"""

TUCKED = """
<mujoco model="tucked">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="4 4 0.1"/>
    <body name="Table0" pos="0 0 0">
      <geom name="top" type="box" pos="0 0 0.64" size="0.5 0.5 0.02" mass="20"/>
    </body>
    <body name="Chair0" pos="0 0 0.301">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.3" mass="5"/>
    </body>
  </worldbody>
</mujoco>
"""

RAMP = """
<mujoco model="ramp">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ramp" type="plane" size="20 20 0.1" euler="10 0 0" friction="0.15 0.005 0.0001"/>
    <body name="box0" pos="0 0 0.102" euler="10 0 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1" friction="0.15 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def _audit(xml: str, tmp_path: Path, **options) -> dict:
    path = tmp_path / "scene.xml"
    path.write_text(xml)
    return physics.audit_physics(path, **{**FAST, **options})


def _row(audit: dict, name: str) -> dict:
    return next(r for r in audit["bodies"] if r["name"] == name)


def test_the_audit_on_a_room_with_known_faults(tmp_path):
    audit = _audit(ROOM, tmp_path)
    chair, door, drawer = (_row(audit, n) for n in ("Chair0", "Door0_Door_leaf",
                                                     "Cabinet0_drawer"))
    assert chair["category"] == "chair" and chair["pull"]["mode"] == "lift"
    assert chair["pull"]["pass"] and chair["pull"]["stable"] and chair["pull"]["moved_m"] > 0.45
    # The heavy door is flagged, and still swings: the grab spring scales with the mass.
    assert door["category"] == "door_leaf" and door["mass_verdict"] == "heavy"
    assert door["pull"]["pass"] and door["pull"]["need"] == pytest.approx(60.0)
    assert door["pull"]["unit"] == "deg" and door["pull"]["moved"] >= 60.0
    # The drawer is grabbed but cannot travel: 60 % of its 0.4 m range is 0.24 m.
    assert drawer["category"] == "drawer" and drawer["pull"]["grab"]
    assert not drawer["pull"]["pass"] and drawer["pull"]["moved"] < 0.05
    i = audit["interactability"]
    assert (i["tested"], i["interactable"], i["failed"]) == (3, 2, ["Cabinet0_drawer"])
    assert [o["name"] for o in audit["masses"]["outliers"]] == ["Door0_Door_leaf"]
    assert audit["masses"]["static_outliers"][0]["name"] == "Sofa0"   # a 3 kg sofa
    authored = audit["overlaps"]["authored"]
    assert any("Chair0" in o["bodies"] and o["mm"] > 15 for o in authored)
    assert not any("wall" in o["bodies"] and "floor" in o["bodies"] for o in authored)
    assert audit["settle_shift"]["max_body"] == "Chair0"     # pushed out of the wall
    assert audit["stability"]["resets"] == 0 and audit["stability"]["max_drift_mm"] < 1.0
    cost = audit["cost"]
    assert cost["ms_per_step"] > 0 and cost["realtime_factor"] > 1 and cost["steps"] == 250
    json.dumps(audit)                                         # the report is plain JSON
    md = physics.markdown(audit)
    assert "| Cabinet0_drawer | slide | drawer |" in md and "**no**" in md


def test_a_tucked_chair_that_cannot_be_lifted_is_dragged_out(tmp_path):
    audit = _audit(TUCKED, tmp_path)
    chair = _row(audit, "Chair0")["pull"]
    assert chair["attempts"]["lift"] < physics.LIFT_PASS       # the table top is in the way
    assert chair["pass"] and chair["mode"] != "lift" and chair["moved_m"] >= physics.LIFT_PASS
    assert audit["static_objects"][0]["category"] == "table"


def test_a_body_creeping_down_a_ramp_is_reported_as_moving(tmp_path):
    audit = _audit(RAMP, tmp_path, stability_seconds=2.0)
    s = audit["stability"]
    assert s["max_drift_body"] == "box0" and s["max_drift_mm"] > 50
    assert s["moving"] == ["box0"]


def _tucked_model(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text(TUCKED)
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, RoomSim(model).settled_state()


def test_concurrency_is_measured_with_one_room_per_thread(tmp_path):
    model, settled = _tucked_model(tmp_path)
    out = physics.concurrency(model, settled, threads=2, seconds=0.2, repeats=2)
    assert out["threads"] == 2 and out["aggregate_realtime_factor"] > 0
    assert list(out["sweep"]) == ["2"] and out["sweep"]["2"]["n"] == 2
    assert out["rooms_flat_out"] == int(out["aggregate_realtime_factor"] * 0.8)


def test_concurrency_sweeps_the_cpus_this_process_may_use(tmp_path, monkeypatch):
    """The sweep goes to the usable CPUs (affinity, cgroup quota), not os.cpu_count()."""
    model, settled = _tucked_model(tmp_path)
    monkeypatch.setattr(physics, "usable_cpus", lambda: 3)
    monkeypatch.setattr(physics, "_flat_out", lambda m, s, n, seconds: {1: 9.0, 2: 12.0,
                                                                          3: 11.0}[n])
    out = physics.concurrency(model, settled, seconds=0.01, repeats=1)
    assert list(out["sweep"]) == ["1", "2", "3"]
    assert out["threads"] == 2 and out["aggregate_realtime_factor"] == 12.0   # the best point


def test_usable_cpus_honours_a_cgroup_quota(monkeypatch):
    monkeypatch.setattr(physics, "_cgroup_cpus", lambda: 1.5)
    assert physics.usable_cpus() == min(2, len(os.sched_getaffinity(0)))
    monkeypatch.setattr(physics, "_cgroup_cpus", lambda: None)
    assert physics.usable_cpus() == len(os.sched_getaffinity(0))


def test_server_capacity_finds_the_largest_room_count_that_keeps_real_time(tmp_path,
                                                                          monkeypatch):
    """Doubling then bisection over k, on a stand-in loop where 5 rooms is the CPU's limit."""
    model, settled = _tucked_model(tmp_path)
    tried = []

    async def fake_serve(m, s, rooms, warmup, window):
        tried.append(rooms)
        return [1.0 if rooms <= 5 else 0.7] * rooms

    monkeypatch.setattr(physics, "_serve", fake_serve)
    out = physics.server_capacity(model, settled, limit=16, attempts=1)
    assert out["rooms"] == 5 and not out["at_limit"]
    assert tried == [1, 2, 4, 8, 6, 5]
    assert out["slowest_room_rtf"]["8"] == 0.7
    tried.clear()
    assert physics.server_capacity(model, settled, limit=4, attempts=1)["at_limit"]
    # Measured while other processes held the CPUs: flagged, never passed off as the room's.
    monkeypatch.setattr(physics, "_load", lambda: 99.0)
    busy = physics.server_capacity(model, settled, limit=2, attempts=1)
    assert busy["machine_busy"] and busy["load_before"] == 99.0


def test_the_server_loop_load_test_keeps_one_light_room_in_real_time(tmp_path):
    """The real loop (asyncio, a worker-thread hop per tick, snapshots): one light room keeps up,
    and its factor is the server's own measure (simulated / wall between ticks), close to 1."""
    model, settled = _tucked_model(tmp_path)
    best = max(min(asyncio.run(physics._serve(model, settled, 1, 0.1, 0.4))) for _ in range(3))
    assert 0.9 < best < 1.1


def test_step_cost_is_warmed_up_and_repeated(tmp_path):
    model, _settled = _tucked_model(tmp_path)
    cost = physics.step_cost(RoomSim(model), 0.2, repeats=3, warmup=0.1)
    assert cost["repeats"] == 3 and cost["steps"] == 100
    spread = cost["ms_per_step_spread"]
    assert spread["n"] == 3 and spread["min"] <= cost["ms_per_step"] <= spread["max"]
    assert cost["realtime_factor_spread"]["median"] == cost["realtime_factor"]


def test_the_demo_room_is_fully_interactable(tmp_path):
    audit = physics.audit_physics(write_demo_scene(tmp_path), **FAST)
    assert audit["interactability"]["interactable"] == audit["interactability"]["tested"] == 8
    assert not audit["masses"]["outliers"] and not audit["overlaps"]["authored"]
    kinds = audit["interactability"]["by_joint"]
    assert kinds["hinge"] == {"tested": 2, "interactable": 2}
    assert kinds["slide"] == {"tested": 1, "interactable": 1}


@pytest.mark.parametrize(("name", "joint", "size", "category"), [
    ("Door0_Door_leaf", "hinge", (0.8, 0.05, 2.0), "door_leaf"),
    ("Wall10_Door_0", "hinge", (0.77, 0.04, 1.98), "door_leaf"),
    ("Sink_Storage0_Door_Right", "hinge", (0.4, 0.02, 0.7), "cabinet_door"),
    ("Storage0_door_r", "hinge", (0.5, 0.02, 1.9), "cabinet_door"),
    ("Sink_Storage0_Drawer_Right", "slide", (0.4, 0.5, 0.2), "drawer"),
    ("Window0_Sash_Sliding_Right", "slide", (0.6, 0.05, 1.0), "window_sash"),
    ("Refrigerator0_Door", "hinge", (0.6, 0.05, 1.7), "appliance_door"),
    ("ChairCluster0", "free", (0.5, 0.5, 0.9), "chair"),
    ("Table1", "static", (0.5, 0.5, 0.5), "small_table"),
    ("Table0", "static", (1.2, 0.8, 0.75), "table"),
    ("Sink_Storage0", "static", (1.1, 0.6, 0.9), "storage"),
    ("Dishwasher0", "free", (0.6, 0.6, 0.9), "dishwasher"),
    ("cardboard_box", "free", (0.4, 0.3, 0.3), "box"),
    ("Thing0", "free", (0.1, 0.1, 0.1), "unknown"),
])
def test_bodies_are_classified_by_name_joint_and_size(name, joint, size, category):
    assert physics.classify(name, joint, size) == category


def test_every_mass_range_is_ordered_and_says_where_it_came_from():
    for category, rng in physics.MASS_RANGES.items():
        assert 0 < rng.lo < rng.hi, category
        assert rng.basis, category
        assert rng.per in ("kg", "face", "shell"), category
    # The tea_room door: 128.7 kg on a 0.9 x 2.05 m leaf.
    assert physics.mass_verdict("door_leaf", 128.7, (0.9, 0.04, 2.05)) == "heavy"
    assert physics.mass_verdict("door_leaf", 128.7) == "unknown"     # per-area needs a size
    assert physics.mass_verdict("mug", 0.1) == "light"
    assert physics.mass_verdict("chair", 5.1) == "ok"
    assert physics.mass_verdict("unknown", 5.0) == "unknown"


# Every number a range's basis quotes, as (category, kg, size of the object it describes): a
# mass the cited source calls normal must never be flagged, or the table contradicts itself.
DOOR = (0.9, 0.04, 2.1)
QUOTED_MASSES = [
    ("chair", 3.5, None), ("chair", 5.5, None), ("chair", 7.0, None), ("chair", 14.0, None),
    ("sofa", 45.0, None), ("sofa", 113.0, None),
    ("dishwasher", 48.0, None),
    ("refrigerator", 10.0, None), ("refrigerator", 43.0, None), ("refrigerator", 180.0, None),
    ("door_leaf", 12.0, DOOR), ("door_leaf", 18.0, DOOR), ("door_leaf", 25.0, DOOR),
    ("door_leaf", 40.0, DOOR), ("door_leaf", 13.6, DOOR), ("door_leaf", 22.7, DOOR),
    ("cabinet_door", 14.0 * 0.4 * 0.7, (0.4, 0.019, 0.7)),     # 19 mm MDF front
    ("cabinet_door", 10.0 * 0.5 * 0.7, (0.5, 0.016, 0.7)),     # 16 mm chipboard
    ("cabinet_door", 13.0, (0.5, 0.02, 1.9)),                  # a wardrobe door at 14 kg/m^2
    ("window_sash", 10.0 * 0.6 * 1.0 * 1.2, (0.6, 0.05, 1.0)),   # 4 mm pane + frame
    ("window_sash", 30.0 * 0.6 * 1.0 * 1.2, (0.6, 0.05, 1.0)),   # triple glazing + frame
    ("mug", 0.25, None), ("mug", 0.4, None), ("mug", 0.685, None),
    ("ball", 0.41, None), ("ball", 0.45, None), ("ball", 0.6, None),
]


@pytest.mark.parametrize(("category", "kg", "size"), QUOTED_MASSES)
def test_a_mass_inside_each_quoted_basis_range_is_ok(category, kg, size):
    assert physics.mass_verdict(category, kg, size) == "ok", physics.mass_bounds(category, size)


def test_leaf_and_drawer_ranges_scale_with_their_size():
    small, tall = (0.4, 0.02, 0.6), (0.5, 0.02, 1.9)
    assert physics.mass_verdict("cabinet_door", 13.0, tall) == "ok"
    assert physics.mass_verdict("cabinet_door", 13.0, small) == "heavy"
    assert physics.scaling_area("face", tall) == pytest.approx(0.95)
    # An open-top 0.4 x 0.5 x 0.2 m drawer: two 0.08 + two 0.10 sides and a 0.20 bottom.
    assert physics.scaling_area("shell", (0.4, 0.5, 0.2)) == pytest.approx(0.56)
    assert physics.mass_verdict("drawer", 4.0, (0.4, 0.5, 0.2)) == "ok"


SIDE_TABLE = """
<mujoco model="side_table">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="4 4 0.1"/>
    <!-- a 0.5 x 0.6 m side table (0.30 m^2), yawed 30 deg as tea_room's walls are -->
    <body name="Table1" pos="0 0 0.25" euler="0 0 30">
      <freejoint/>
      <geom type="box" size="0.25 0.3 0.25" mass="6"/>
    </body>
    <body name="Storage0_door_r" pos="1.5 0 0" euler="0 0 30">
      <joint name="door" type="hinge" axis="0 0 1" range="0 100" damping="3"/>
      <geom type="box" pos="0.25 0 0.95" size="0.25 0.01 0.94" mass="13"/>
    </body>
    <body name="Turntable0" pos="-1.5 0 0.05">
      <joint name="spin" type="hinge" axis="0 0 1" limited="false" damping="1"/>
      <geom type="cylinder" size="0.3 0.02" mass="2"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_a_turned_side_table_is_sized_in_its_own_frame(tmp_path):
    """Its world box is 0.56 m^2 (a dining table, 10-60 kg: 6 kg would be "light"); in its own
    frame it is the 0.30 m^2 side table the quick room set free."""
    audit = _audit(SIDE_TABLE, tmp_path)
    table = _row(audit, "Table1")
    np.testing.assert_allclose(table["size_m"], [0.5, 0.6, 0.5], atol=1e-3)
    assert table["category"] == "small_table" and table["mass_verdict"] == "ok"
    door = _row(audit, "Storage0_door_r")
    assert door["category"] == "cabinet_door" and door["mass_verdict"] == "ok"
    assert door["mass_range_per_m2"]["area_m2"] == pytest.approx(0.5 * 1.88, abs=0.01)


def test_an_unlimited_hinge_keeps_the_report_strict_json(tmp_path):
    audit = _audit(SIDE_TABLE, tmp_path)
    spin = _row(audit, "Turntable0")["pull"]
    assert spin["travel"] is None and spin["range"] is None and spin["pass"]
    json.loads(json.dumps(audit, allow_nan=False))       # no Infinity / NaN anywhere


def test_report_json_is_written_strict(tmp_path):
    report.write_json(tmp_path / "r.json", {"a": float("inf"), "b": [float("nan"), 1.5],
                                            "c": np.float32(2.0)})
    data = json.loads((tmp_path / "r.json").read_text(),
                      parse_constant=lambda c: pytest.fail(f"non-standard JSON {c}"))
    assert data == {"a": None, "b": [None, 1.5], "c": 2.0}


# ── reports, the comparison table and the CLI ────────────────────────────────────────────────────


@pytest.fixture()
def run_folder(tmp_path, synthetic_scan) -> Path:
    """run/<scan> with a quick and a full stage (both the demo scene: only the plumbing is under
    test here) and what each build records about itself."""
    run = tmp_path / "run" / "test_room"
    for folder in ("quick", "mujoco_seed"):
        write_demo_scene(run / folder)
    (run / "quick" / "quick.json").write_text(json.dumps(
        {"source": str(synthetic_scan / "room.usdz"),
         "seconds": {"parse": 0.02, "textures": 1.4, "total": 2.5}}))
    (run / "mujoco_seed" / "export_report.json").write_text(json.dumps(
        {"loads_clean": False, "colliders": 151, "initial_overlaps": [1, 2], "free": 5,
         "articulated": 8}))
    (run / ".litereality").mkdir()
    (run / ".litereality" / "pipeline.json").write_text(json.dumps(
        {"stages": {"ingest": {"duration_seconds": 600.0}, "simulate": {"duration_seconds": 60}}}))
    return run


def test_a_run_folder_gives_every_stage_in_order_with_its_timings(run_folder, synthetic_scan):
    scenes = report.find_scenes(run_folder)
    assert [name for name, _ in scenes] == ["quick", "full"]
    assert report.infer_scan(scenes[0][1]) == synthetic_scan
    assert report.infer_scan(scenes[1][1]) == synthetic_scan     # through its sibling quick/
    quick = report.stage_timings(scenes[0][1], "quick")
    assert quick["build_s"] == 2.5 and quick["quick_parts_s"]["textures"] == 1.4
    full = report.stage_timings(scenes[1][1], "full")
    assert full["build_s"] == 660.0 and full["export"]["initial_overlaps"] == 2
    assert report.stage_of(run_folder / "quick" / "scene.xml") == "quick"
    assert report.stage_of(write_demo_scene(run_folder.parent / "elsewhere")) == "demo"
    with pytest.raises(ValueError, match="no scene"):
        report.find_scenes(run_folder.parent)


def test_evaluate_writes_a_report_per_stage_and_the_comparison(run_folder, synthetic_scan,
                                                               tmp_path):
    out = tmp_path / "eval"
    views = fidelity.capture_views(synthetic_scan)
    result = report.evaluate(run_folder, out, render=True, renderer=PhotoRenderer(views),
                             size=(80, 60), physics_options=FAST, log=None)
    rows = result["comparison"]
    assert [r["name"] for r in rows] == ["quick", "full"]
    assert rows[0]["build_s"] == 2.5 and rows[0]["interactable"] == 8
    assert rows[0]["psnr"] > 40 and rows[0]["views"] == 3
    for name in ("quick", "full"):
        data = json.loads((out / name / "report.json").read_text())
        assert data["physics"]["interactability"]["tested"] == 8
        assert data["fidelity"]["views"] == 3
        assert (out / name / "contact_sheet.jpg").is_file()
        assert "### Visual fidelity" in (out / name / "report.md").read_text()
    md = (out / "comparison.md").read_text()
    assert md.startswith("### Physics and cost") and "| test_room | quick | 2.5 | 8/8 |" in md
    # Fidelity: one row per stage and split, the split named (no build record: unknown).
    assert "| test_room | quick | as built (build's frames not recorded) | 3 |" in md
    assert [r["split"] for r in result["fidelity"]] == ["as built (build's frames not recorded)"] * 2
    tex = (out / "comparison.tex").read_text()
    assert r"\toprule" in tex and r"test\_room" in tex and "—" not in tex
    assert tex.count(r"\begin{table}") == 2 and "machine-dependent" in tex
    json.loads((out / "comparison.json").read_text())


def test_without_a_scan_the_physics_is_still_reported(tmp_path):
    scene = write_demo_scene(tmp_path / "somewhere")
    rep = report.evaluate_scene(scene, tmp_path / "eval", render=True,
                                renderer=PhotoRenderer([]), physics_options=FAST)
    assert rep["fidelity"] is None and "--scan" in rep["fidelity_skipped"]
    assert rep["physics"]["interactability"]["tested"] == 8


def _tiny_scene(folder: Path) -> Path:
    """A one-chair scene: for plumbing tests, where the demo room's eight pulls only cost time."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scene.xml").write_text(TUCKED)
    return folder / "scene.xml"


def test_a_paired_difference_has_a_reproducible_bootstrap_interval():
    a = [10.0, 12.0, 14.0, 11.0, 13.0]
    same_step = report.paired_difference(a, [v + 0.5 for v in a])
    assert same_step == {"mean": 0.5, "ci95": [0.5, 0.5], "better": 5, "n": 5}
    mixed = report.paired_difference(a, [10.5, 11.5, 14.5, 10.5, 13.5])
    assert mixed["ci95"][0] < 0 < mixed["ci95"][1] and mixed["better"] == 3
    assert mixed == report.paired_difference(a, [10.5, 11.5, 14.5, 10.5, 13.5])   # seeded


def test_stages_of_a_room_are_compared_view_by_view_on_their_shared_split():
    def rep(name, values, held_out=None):
        rows = [{"frame": f, "psnr": v, "ssim": 0.5} for f, v in enumerate(values)]
        fid = {"per_view": rows}
        if held_out is not None:
            fid["holdout"] = {"k": 4, "split": "interleaved",
                              "per_view": [{"frame": f, "psnr": v, "ssim": 0.5}
                                           for f, v in enumerate(held_out)]}
        return {"name": name, "room": "r", "stage": "quick", "fidelity": fid}

    both = report.paired_differences([rep("quick_flat", [10, 11, 12], [9, 10, 11]),
                                      rep("quick", [11, 12, 13], [10, 11, 13])])
    assert len(both) == 1 and both[0]["split"] == "held-out (4-fold interleaved)"
    assert both[0]["psnr"]["mean"] == pytest.approx(4 / 3, abs=1e-4)
    assert both[0]["psnr_objects"] is None
    # One stage without a hold-out: both are compared as built.
    mixed = report.paired_differences([rep("quick", [11, 12, 13], [10, 11, 13]),
                                       rep("objects", [12, 12, 14])])
    assert mixed[0]["split"] == "as built" and mixed[0]["psnr"]["mean"] == pytest.approx(2 / 3,
                                                                                      abs=1e-4)
    md = report.comparison_markdown([], [], both)
    assert "### Paired differences" in md and "| r | quick - quick_flat |" in md


def test_a_fidelity_failure_keeps_the_physics_report_and_the_table(tmp_path):
    """A scan with a RoomPlan export but no photos: the physics already run is still written."""
    scan = tmp_path / "scan"
    write_roomplan_usdz(scan / "room.usdz")
    scene = _tiny_scene(tmp_path / "s")
    result = report.evaluate(scene, tmp_path / "eval", scan=scan, render=True,
                             renderer=FlatRenderer(), physics_options=FAST, log=None)
    rep = result["reports"][0]
    assert rep["fidelity"] is None and rep["fidelity_skipped"].startswith("fidelity failed")
    assert "no frame" in rep["fidelity_skipped"]
    saved = json.loads((tmp_path / "eval" / "s" / "report.json").read_text())
    assert saved["physics"]["interactability"]["tested"] == 1
    assert (tmp_path / "eval" / "comparison.md").is_file() and result["fidelity"] == []


def test_a_scan_given_for_a_scene_built_from_another_is_flagged(tmp_path):
    scans = _two_rooms(tmp_path)
    rep = report.evaluate_scene(tmp_path / "run" / "room_a" / "quick" / "scene.xml",
                                tmp_path / "eval", scan=scans["room_b"], physics_options=FAST)
    assert rep["scan"] == str(scans["room_b"].resolve()) and rep["room"] == "room_b"
    assert "room_b" in rep["scan_mismatch"] and "room_a" in rep["scan_mismatch"]


def _two_rooms(tmp_path) -> dict[str, Path]:
    """Two captures, each with a quick stage in run/<scan>/ that records its own capture, and a
    run-folder scene.json as every full-pipeline run has (what made the old CLI reuse the first
    room's scan for both)."""
    scans = {}
    for k, name in enumerate(("room_a", "room_b")):
        scan = tmp_path / "scans" / name
        write_roomplan_usdz(scan / "room.usdz")
        _write_frame(scan, 0, _look((2.5, 0.4 + k, FLOOR_Y + 1.4), (2.5, 2.0, FLOOR_Y + 0.5)),
                     rgb=(60 + 100 * k, 120, 150))
        run = tmp_path / "run" / name
        _tiny_scene(run / "quick")
        (run / "quick" / "quick.json").write_text(json.dumps(
            {"stage": "quick", "label": name, "source": str(scan / "room.usdz"),
             "seconds": {"total": 1.0}, "colours": {"source": "category"},
             "textures": {"source": "none"}}))
        (run / "scene.json").write_text(json.dumps({"scan": name, "paths": {}}))
        scans[name] = scan
    return scans


@pytest.mark.parametrize("form", ["names", "scan folders", "run folders"])
def test_the_cli_compares_each_room_with_its_own_scan(tmp_path, monkeypatch, capsys, form):
    scans = _two_rooms(tmp_path)
    monkeypatch.setenv("LR_SCANS_DIR", str(tmp_path / "scans"))
    monkeypatch.setattr(cli_evaluate_module(), "PlaywrightRenderer", FlatRenderer)
    targets = {"names": list(scans), "scan folders": [str(s) for s in scans.values()],
               "run folders": [str(tmp_path / "run" / n) for n in scans]}[form]
    out = tmp_path / "eval"
    assert _main(monkeypatch, ["evaluate", *targets, "--render", "--fast", "--out", str(out),
                               "--output-root", str(tmp_path / "run")]) == 0
    for name, scan in scans.items():
        data = json.loads((out / name / "quick" / "report.json").read_text())
        assert data["scan"] == str(scan.resolve()) and data["room"] == name
        assert data["fidelity"]["scan"] == str(scan.resolve())
        assert "scan_mismatch" not in data
    printed = capsys.readouterr().out
    assert "| room_a | quick |" in printed and "| room_b | quick |" in printed


def cli_evaluate_module():
    from litereality_agent.pipeline import evaluate as evaluation

    return evaluation


def test_the_cli_maps_a_scan_folder_to_its_run_and_refuses_what_is_not_there(tmp_path,
                                                                              monkeypatch):
    scans = _two_rooms(tmp_path)
    (tmp_path / "run" / "room_b" / "quick" / "scene.xml").unlink()
    with pytest.raises(SystemExit, match="room_b.*no scene.xml"):
        _main(monkeypatch, ["evaluate", str(scans["room_b"]), "--fast",
                            "--output-root", str(tmp_path / "run")])
    with pytest.raises(SystemExit, match="does not exist yet"):
        _main(monkeypatch, ["evaluate", str(scans["room_a"]), "--fast",
                            "--output-root", str(tmp_path / "elsewhere")])
    assert _main(monkeypatch, ["evaluate", str(scans["room_a"]), "--fast",
                               "--output-root", str(tmp_path / "run")]) == 0
    assert (tmp_path / "run" / "room_a" / "eval" / "quick" / "report.json").is_file()


def _main(monkeypatch, argv: list[str]) -> int:
    """`cli.main` without it copying the checkout's .env into this test process's environment."""
    monkeypatch.setattr(cli, "load_settings",
                        lambda: type("S", (), {"apply_environment": lambda self: None})())
    return cli.main(argv)


def test_the_cli_evaluates_a_scene_into_its_eval_folder(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run" / "r"
    scene = write_demo_scene(run / "quick")
    assert _main(monkeypatch, ["evaluate", str(scene), "--fast"]) == 0
    printed = capsys.readouterr().out
    assert "| Room | Stage |" in printed and "not rendered" in printed
    assert (run / "eval" / "quick" / "report.json").is_file()        # beside the stages
    assert (run / "eval" / "comparison.tex").is_file()


def test_the_cli_says_how_to_render_when_playwright_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    scene = write_demo_scene(tmp_path / "s")
    with pytest.raises(SystemExit, match="playwright install chromium"):
        _main(monkeypatch, ["evaluate", str(scene), "--render", "--out", str(tmp_path / "e")])
    with pytest.raises(SystemExit, match="WxH"):
        _main(monkeypatch, ["evaluate", str(scene), "--size", "big"])


# ── rendering (opt-in: -m render) ────────────────────────────────────────────────────────────────

BOX_SCENE = """
<mujoco model="one_box">
  <worldbody>
    <geom name="target" type="box" pos="0.4 3.0 1.1" size="0.3 0.25 0.2" rgba="0.9 0.1 0.1 1"/>
  </worldbody>
</mujoco>
"""


@pytest.mark.render
def test_chromium_draws_a_box_exactly_where_the_pose_maths_says(tmp_path):
    pytest.importorskip("playwright")
    path = tmp_path / "scene.xml"
    path.write_text(BOX_SCENE)
    glb = fidelity.scene_glb(path)
    view = _view(_look((0.0, 0.0, 1.0), (0.2, 3.0, 1.15)))
    size = (160, 120)
    (rgba,) = fidelity.PlaywrightRenderer().render(glb, [fidelity.three_camera(view, size)], size)
    drawn = rgba[..., 3] > 127
    box = fidelity.Box("target", np.array([0.4, 3.0, 1.1]), np.array([0.3, 0.25, 0.2]), 0.0)
    expected = fidelity.box_mask([box], view, size)
    iou = (drawn & expected).sum() / (drawn | expected).sum()
    assert iou > 0.95, iou
    inside = rgba[expected & drawn][:, :3].astype(int)
    assert np.median(inside[:, 0] - inside[:, 1]) > 60          # red, not grey


@pytest.mark.render
def test_fidelity_renders_the_synthetic_quick_room(synthetic_scan, tmp_path):
    pytest.importorskip("playwright")
    from litereality_agent.pipeline.quickroom import write_quick_scene

    scene = write_quick_scene(synthetic_scan, tmp_path / "quick")
    out = fidelity.evaluate_fidelity(scene, synthetic_scan, size=(80, 60), out_dir=tmp_path)
    assert out["views"] == 3 and out["summary"]["coverage"]["mean"] > 0.9
    assert all(r["psnr"] is not None for r in out["per_view"])


def test_render_tests_are_opt_in():
    """The render marker is registered and deselected by default (pyproject addopts)."""
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "render:" in text and "not render" in text
