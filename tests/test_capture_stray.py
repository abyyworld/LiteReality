"""Stray Scanner import: the format, the pose convention, frame selection, alignment, and a
synthetic recording end to end (a tiny mp4 made with ffmpeg, ray-cast depth, known poses)."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from litereality_agent.models.capture import align as aligner
from litereality_agent.models.capture import stray
from litereality_agent.models.splat import rgbd

try:
    FFMPEG = stray.ffmpeg_binary()
    stray.ffmpeg_binary("ffprobe")
except RuntimeError:
    FFMPEG = None
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")

# The synthetic room, in the TARGET scan's ARKit world (Y up, metres): its walls are the inside of
# ROOM, the furniture solid boxes (asymmetric, so the heading is not ambiguous).
ROOM = (np.array([-2.5, -1.4, -2.0]), np.array([2.5, 1.2, 2.0]))
FURNITURE = [(np.array([1.0, -1.4, -1.9]), np.array([2.4, -0.6, -1.0])),
             (np.array([-2.45, -1.4, 0.5]), np.array([-1.9, 0.6, 1.9])),
             (np.array([-0.4, -1.4, 1.2]), np.array([0.5, -0.9, 1.95]))]
# Photo 256 x 192 (K below), LiDAR 64 x 48: the same 4:3 field of view, like 1920x1440 / 256x192.
W, H, DW, DH = 256, 192, 64, 48
K = np.array([[200.0, 0.0, 128.0], [0.0, 200.0, 96.0], [0.0, 0.0, 1.0]])
TRUE_YAW, TRUE_T = 37.0, np.array([1.3, -0.4, -2.1])     # recording world -> target world


def _rot_y(deg):
    return aligner.yaw_transform(np.deg2rad(deg))[:3, :3]


def _rot_x(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _raycast(c2w_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth (m, along the camera's -z) and hit points of the synthetic room at DW x DH, from an
    OpenGL camera -> target-world pose."""
    sx, sy = DW / W, DH / H
    fx, fy = K[0, 0] * sx, K[1, 1] * sy
    cx, cy = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
    v, u = np.mgrid[0:DH, 0:DW]
    d_cam = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u, float)], -1).reshape(-1, 3)
    d = d_cam @ c2w_gl[:3, :3].T
    o = c2w_gl[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        lo, hi = ROOM
        t_exit = np.min(np.where(d > 0, (hi - o) / d, (lo - o) / d), axis=1)
        best = t_exit
        for blo, bhi in FURNITURE:
            t1, t2 = (blo - o) / d, (bhi - o) / d
            near = np.max(np.minimum(t1, t2), axis=1)
            far = np.min(np.maximum(t1, t2), axis=1)
            hit = (near <= far) & (near > 0)
            best = np.where(hit & (near < best), near, best)
    points = o + best[:, None] * d
    return best.reshape(DH, DW), points.reshape(DH, DW, 3)


def _box_surface(lo, hi, spacing, rng):
    pts = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        n = int(np.prod((hi[others] - lo[others]) / spacing)) + 1
        for side in (lo[axis], hi[axis]):
            p = np.empty((n, 3))
            p[:, axis] = side
            for a in others:
                p[:, a] = rng.uniform(lo[a], hi[a], n)
            pts.append(p)
    return np.concatenate(pts)


def _room_cloud(seed=0, spacing=0.03):
    rng = np.random.default_rng(seed)
    return np.concatenate([_box_surface(*ROOM, spacing, rng)]
                          + [_box_surface(lo, hi, spacing, rng) for lo, hi in FURNITURE])


def _texture(seed=0):
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(seed)
    noise = (rng.random((900, 900)) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(noise).filter(ImageFilter.GaussianBlur(1.2)))


def _frame_image(k: int, tex: np.ndarray, blur: bool) -> np.ndarray:
    from PIL import Image, ImageFilter

    y0, x0 = 2 * k, 3 * k
    g = tex[y0:y0 + H, x0:x0 + W].astype(np.float32)
    g = (g - g.mean()) * 2.5 + 128
    img = np.stack([g, g * 0.9 + 10, g * 0.8 + 20], -1).clip(0, 255).astype(np.uint8)
    img[:32, :32] = 2 * k             # the frame's number, readable after decoding
    im = Image.fromarray(img)
    if blur:
        im = im.filter(ImageFilter.GaussianBlur(3))
        a = np.asarray(im).copy()
        a[:32, :32] = 2 * k
        return a
    return np.asarray(im)


def _encode(frames_dir: Path, out: Path, bframes: int = 0) -> None:
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", "60", "-i",
                    str(frames_dir / "f_%05d.png"), "-c:v", "libx264", "-crf", "8", "-bf",
                    str(bframes), "-pix_fmt", "yuv420p", str(out)], check=True)


def _poses(n: int, still: int) -> list[np.ndarray]:
    """OpenGL camera -> TARGET-world poses: `still` frames standing, then a turn and a walk (a
    full circle in 100 frames: 3.8 rad/s, 2.3 m/s, fast but no tracking jump)."""
    poses = []
    for k in range(n):
        s = max(0, k - still) / 100.0
        yaw = 360.0 * s
        c2w = np.eye(4)
        c2w[:3, :3] = _rot_y(yaw) @ _rot_x(-12.0)
        c2w[:3, 3] = [0.6 * np.cos(2 * np.pi * s) - 0.6, 0.1, 0.4 * np.sin(2 * np.pi * s)]
        poses.append(c2w)
    return poses


def _stray_odometry(c2w_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What Stray's OdometryEncoder writes for an ARKit camera.transform: its translation and
    q_WA * q_AC with q_AC half a turn about X, as (x, y, z, w). Built with scipy, as
    StrayVisualizer reads it, not with the module under test."""
    from scipy.spatial.transform import Rotation

    q_wa = Rotation.from_matrix(c2w_gl[:3, :3])
    q_ac = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])
    return c2w_gl[:3, 3].copy(), (q_wa * q_ac).as_quat()


def make_recording(root: Path, n: int = 120, still: int = 20, header: str = "new",
                   bframes: int = 0, lose_first_video_frame: bool = False) -> dict:
    """A Stray Scanner recording of the synthetic room in its OWN world (the target world moved by
    the inverse of TRUE_YAW / TRUE_T), every frame not a multiple of 3 blurred. With
    `lose_first_video_frame` rgb.mp4 lacks row 0's frame (the -1/60 frame an edit list trimmed)
    and starts at PTS 0 with row 1's."""
    from PIL import Image

    root.mkdir(parents=True)
    (root / "depth").mkdir()
    (root / "confidence").mkdir()
    pngs = root.parent / f"{root.name}_png"
    pngs.mkdir()
    T = aligner.yaw_transform(np.deg2rad(TRUE_YAW), TRUE_T)
    T_inv = np.linalg.inv(T)
    tex = _texture()
    rows, hits = [], {}
    for k, c2w_t in enumerate(_poses(n, still)):
        depth, points = _raycast(c2w_t)
        hits[k] = points
        Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(root / "depth" / f"{k:06d}.png")
        Image.fromarray(np.full((DH, DW), 2, np.uint8)).save(root / "confidence" / f"{k:06d}.png")
        if not (lose_first_video_frame and k == 0):
            Image.fromarray(_frame_image(k, tex, blur=k % 3 != 0)).save(
                pngs / f"f_{k - int(lose_first_video_frame):05d}.png")
        xyz, q = _stray_odometry(T_inv @ c2w_t)
        rows.append((1000.0 + k / 60.0, f"{k:06d}", *xyz, *q))
    _encode(pngs, root / "rgb.mp4", bframes)
    with open(root / "odometry.csv", "w") as fh:
        if header == "new":
            fh.write("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, "
                     "distortion_center_x, distortion_center_y\n")
            for r in rows:
                fh.write(", ".join(str(v) for v in r)
                         + f", {K[0, 0]}, {K[1, 1]}, {K[0, 2]}, {K[1, 2]}, , \n")
        else:     # app versions before per-frame intrinsics
            fh.write("timestamp, frame, x, y, z, qx, qy, qz, qw\n")
            for r in rows:
                fh.write(", ".join(str(v) for v in r) + "\n")
    (root / "camera_matrix.csv").write_text("\n".join(", ".join(str(v) for v in row) for row in K))
    (root / "imu.csv").write_text("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n")
    return {"T": T, "hits": hits, "poses_target": _poses(n, still)}


def make_target_scan(root: Path) -> None:
    """A LiteReality Scanner scan of the same room: its point cloud and a (stand-in) RoomPlan."""
    root.mkdir(parents=True)
    (root / "roomplan").mkdir()
    stray.write_pcd(root / "pointcloud.pcd", _room_cloud(seed=5))
    (root / "room.usdz").write_bytes(b"usdz stand-in")
    (root / "roomplan" / "room.usdz").write_bytes(b"usdz stand-in")
    stray.write_json(root / "scene_alignment.json", {"coordinateSystem": "ARKit world", "unit":
                                                     "meters"})
    stray.write_json(root / "scan_label.json", {"displayName": "Synthetic", "roomType": "office"})


# ---------------------------------------------------------------------------------------------
# Format and pose convention
# ---------------------------------------------------------------------------------------------

def test_quaternion_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[3] >= 0 else -q
        np.testing.assert_allclose(stray.matrix_to_quat(stray.quat_to_matrix(q)), q, atol=1e-9)


def test_pose_convention_matches_the_reference_reader():
    """StrayVisualizer reads a row as T_WC with an OpenCV camera (x right, y down, z ahead): a
    world point projects to the same pixel through that and through our OpenGL pose + rgbd's
    back-projection."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(7)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        xyz = rng.normal(size=3)
        T_wc = np.eye(4)                                     # StrayVisualizer's reading
        T_wc[:3, :3] = Rotation.from_quat(q).as_matrix()
        T_wc[:3, 3] = xyz
        p_cam_cv = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(1, 4)])
        P = T_wc[:3, :3] @ p_cam_cv + xyz                    # the world point
        u = K[0, 0] * p_cam_cv[0] / p_cam_cv[2] + K[0, 2]
        v = K[1, 1] * p_cam_cv[1] / p_cam_cv[2] + K[1, 2]
        z = p_cam_cv[2]
        c2w = stray.pose_from_odometry(xyz, q)               # ours (ARKit camera.transform)
        cam_gl = np.array([(u - K[0, 2]) / K[0, 0] * z, -(v - K[1, 2]) / K[1, 1] * z, -z])
        np.testing.assert_allclose(c2w[:3, :3] @ cam_gl + c2w[:3, 3], P, atol=1e-9)
        xyz2, q2 = stray.odometry_from_pose(c2w)
        np.testing.assert_allclose(stray.pose_from_odometry(xyz2, q2), c2w, atol=1e-9)


def test_known_quaternions_give_the_known_camera(tmp_path):
    """Hand-derived rows, not round trips: Stray writes q_WA * (1, 0, 0, 0), so an ARKit camera
    at rest (looking down -Z, up +Y) is q = (1, 0, 0, 0), and the same camera turned 90 degrees
    about +Y (counter-clockwise seen from above: now looking down -X) is
    (0, s, 0, s) * (1, 0, 0, 0) = (s, 0, -s, 0) with s = sqrt(1/2). Then also pitched 30 degrees
    down first (q_WA = (0, s, 0, s) * (-a, 0, 0, b), a = sin 15, b = cos 15): (sb, sa, -sb, sa),
    every component non-zero, so a conjugated or reordered quaternion cannot pass."""
    s = np.sqrt(0.5)
    a, b = np.sin(np.deg2rad(15.0)), np.cos(np.deg2rad(15.0))
    path = tmp_path / "odometry.csv"
    path.write_text("timestamp, frame, x, y, z, qx, qy, qz, qw\n"
                    "1.0, 000000, 0.5, 1.5, -2.0, 1, 0, 0, 0\n"
                    f"1.1, 000001, 0, 0, 0, {s}, 0, {-s}, 0\n"
                    f"1.2, 000002, 0, 0, 0, {s * b}, {s * a}, {-s * b}, {s * a}\n")
    rest, turned, pitched = (stray.pose_from_odometry(r["xyz"], r["q"])
                             for r in stray.read_odometry(path))
    c30, s30 = np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0))
    np.testing.assert_allclose(-pitched[:3, 2], [-c30, -s30, 0.0], atol=1e-9)   # ahead, down
    np.testing.assert_allclose(pitched[:3, 1], [-s30, c30, 0.0], atol=1e-9)     # up
    np.testing.assert_allclose(rest[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(rest[:3, 3], [0.5, 1.5, -2.0])
    forward, up, right = -turned[:3, 2], turned[:3, 1], turned[:3, 0]   # OpenGL camera axes
    np.testing.assert_allclose(forward, [-1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(up, [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(right, [0.0, 0.0, -1.0], atol=1e-9)
    # A point 2 m straight ahead lands at the image centre through the rgbd back-projection.
    cam = np.array([0.0, 0.0, -2.0])
    np.testing.assert_allclose(turned[:3, :3] @ cam + turned[:3, 3], [-2.0, 0.0, 0.0], atol=1e-9)


def test_read_odometry_new_and_legacy_headers(tmp_path):
    new = tmp_path / "new.csv"
    new.write_text("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, "
                   "distortion_center_x, distortion_center_y\n"
                   "10.5, 000000, 1, 2, 3, 0, 0, 0, 1, 1400.5, 1401, 960, 720, , \n"
                   "10.6, 000001, 1, 2, 3, 0, 0, 0, 1, 1400.5, 1401, 961, 721, 955.1, 718.2\n"
                   "10.7, 000002, 1, 2\n")                    # cut short: dropped
    rows = stray.read_odometry(new)
    assert len(rows) == 2
    assert rows[0]["frame"] == "000000" and rows[1]["timestamp"] == 10.6
    assert (rows[1]["fx"], rows[1]["cx"], rows[1]["cy"]) == (1400.5, 961.0, 721.0)
    np.testing.assert_allclose(rows[0]["q"], [0, 0, 0, 1])
    old = tmp_path / "old.csv"
    old.write_text("timestamp, frame, x, y, z, qx, qy, qz, qw\n5, 000007, 0, 0, 0, 0, 0, 0, 1\n")
    (row,) = stray.read_odometry(old)
    assert row["frame"] == "000007" and "fx" not in row
    bare = tmp_path / "bare.csv"
    bare.write_text("0.1, 0.2, 0.3, 0, 0, 0, 1\n0.1, 0.2, 0.4, 0, 0, 0, 1\n")
    assert [r["frame"] for r in stray.read_odometry(bare)] == ["000000", "000001"]


# ---------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------

def _walk(n=600, still=150, pan=None):
    """A stand, then a slow walk and turn; `pan` = (start, stop) frames of an extra whip pan
    (90 degrees in their span) on top."""
    times = np.arange(n) / 60.0
    poses = np.stack([np.eye(4)] * n)
    for k in range(still, n):
        s = (k - still) / (n - still)
        extra = 0.0
        if pan is not None:
            extra = 90.0 * np.clip((k - pan[0]) / (pan[1] - pan[0]), 0.0, 1.0)
        poses[k, :3, :3] = _rot_y(180 * s + extra)
        poses[k, :3, 3] = [2.0 * s, 0.0, 0.0]
    return times, poses


def test_select_frames_keeps_sharp_frames_and_drops_duplicates():
    times, poses = _walk()
    rng = np.random.default_rng(0)
    sharp = np.where(np.arange(len(times)) % 4 == 0, 100.0, 20.0) + rng.random(len(times))
    sel = stray.select_frames(times, poses, sharp, np.ones(len(times), bool), target=60)
    rows = sel.rows
    assert 25 <= len(rows) <= 60
    assert np.all(sharp[rows] > 99)                        # only the sharp frames
    assert np.all(np.diff(rows) > 0)
    # 150 frames standing still -> one view of that spot (maybe the walk's first sharp step).
    at_start = [r for r in rows if np.linalg.norm(poses[r, :3, 3]) < stray.MIN_MOVE
                and np.rad2deg(stray._angle(poses[r, :3, :3], np.eye(3))) < stray.MIN_TURN_DEG]
    assert len(at_start) == 1
    for a, b in zip(rows[:-1], rows[1:]):
        moved = np.linalg.norm(poses[b, :3, 3] - poses[a, :3, 3])
        turned = np.rad2deg(stray._angle(poses[a, :3, :3], poses[b, :3, :3]))
        assert moved >= stray.MIN_MOVE or turned >= stray.MIN_TURN_DEG
    # Spread over the walk: no stretch of a fifth of it goes unsampled.
    walk = rows[rows >= 150]
    assert np.max(np.diff(np.r_[150, walk, 599])) < 0.2 * 450
    assert sel.stats["sharpness"]["kept_median"] > sel.stats["sharpness"]["all_median"]


def test_select_frames_skips_unusable_and_blurry_windows():
    times, poses = _walk(still=0, pan=(300, 360))
    sharp = np.full(len(times), 50.0)
    sharp[300:360] = 5.0                                  # a fast pan: everything blurred
    usable = np.ones(len(times), bool)
    usable[100:140] = False
    rows = stray.select_frames(times, poses, sharp, usable, target=100).rows
    assert not np.any((rows >= 100) & (rows < 140))
    assert not np.any((rows >= 300) & (rows < 360))


def test_blur_is_read_from_the_sharpness_relative_to_the_frames_around():
    # The calibration's own points come back: the locally sharpest frame is 0 px, a frame at
    # 0.4237 x its neighbour's Laplacian variance 4 px (at 1920); half the width doubles it.
    assert stray.blur_px_from_ratio([1.0, 0.6932, 0.4237, 0.0828]) == pytest.approx(
        [0.0, 2.0, 4.0, 24.0])
    assert stray.blur_px_from_ratio([1e-9])[0] == pytest.approx(24.0)
    times = np.arange(10) / 30.0                          # one 0.33 s moment of one view
    scores = np.array([100.0, 69.32, 42.37, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    assert stray.relative_sharpness(times, scores)[:3] == pytest.approx([1.0, 0.6932, 0.4237])
    assert stray.blur_estimate(times, scores, scored_width=1920)[:3] == pytest.approx(
        [0.0, 2.0, 4.0])
    assert stray.blur_estimate(times, scores, scored_width=960)[2] == pytest.approx(8.0)
    # Texture cancels: a plain wall scoring a tenth of a bookshelf is still sharp at its moment.
    wall = np.r_[np.full(15, 500.0), np.full(15, 50.0)]
    rel = stray.relative_sharpness(np.arange(30) / 2.0, wall)
    assert np.all(rel == 1.0)


def test_the_blur_picker_keeps_the_least_blurred_frames_under_the_limit():
    times, poses = _walk(n=900, still=0, pan=(600, 660))
    rng = np.random.default_rng(3)
    blur = rng.uniform(0.0, 6.0, len(times))
    blur[600:660] = 9.0                                    # a fast pan: nothing under the limit
    sharp = 100.0 / (1.0 + blur)
    usable = np.ones(len(times), bool)
    usable[100:130] = False
    sel = stray.select_frames(times, poses, sharp, usable, target=40, blur_px=blur,
                              max_blur_px=2.0)
    rows = sel.rows
    assert len(rows) >= 18 and np.all(np.diff(rows) > 0)
    assert np.all(blur[rows] <= 2.0) and usable[rows].all()
    assert not np.any((rows >= 600) & (rows < 660))
    assert sel.stats["picker"] == "blur" and sel.stats["over_max_blur"] == int(
        np.sum(usable & (blur > 2.0)))
    assert sel.stats["blur_px"]["kept_max"] <= 2.0
    # Each pick is the least blurred candidate of its stretch: none within a window's width
    # either side is cleaner and left out (a neighbour's pick may be).
    picked = set(rows.tolist())
    for r in rows:
        near = np.arange(max(0, r - 3), min(len(times), r + 4))
        near = near[usable[near] & (blur[near] <= 2.0)]
        better = [k for k in near if blur[k] < blur[r] and k not in picked]
        assert len(better) <= 3
    # Without a limit every window keeps a frame, blurred or not, and fewer windows go empty.
    loose = stray.select_frames(times, poses, sharp, usable, target=40, blur_px=blur)
    assert len(loose.rows) > len(rows) and np.any((loose.rows >= 600) & (loose.rows < 660))


def test_the_blur_picker_lets_a_window_borrow_a_cleaner_neighbour():
    times, poses = _walk(n=600, still=0)
    blur = np.full(len(times), 1.8)
    blur[[300, 302]] = 0.1                    # two clean frames in one window (of 25 frames)
    sharp = np.ones(len(times))
    usable = np.ones(len(times), bool)
    common = dict(target=24, blur_px=blur, max_blur_px=2.0, min_move=0.0, min_turn_deg=0.0)
    alone = stray.select_frames(times, poses, sharp, usable, window_grow=0.0, **common).rows
    grown = stray.select_frames(times, poses, sharp, usable, window_grow=0.5, **common).rows
    assert len(alone) == len(grown) == 24
    # Alone, the window holding both takes one; grown, its neighbour takes the other.
    assert {300, 302} & set(alone.tolist()) != {300, 302}
    assert {300, 302} <= set(grown.tolist())


def test_select_frames_keeps_a_long_still_look_at_a_plain_surface():
    """The Laplacian measures texture as much as blur: three seconds looking at a white ceiling
    score a tenth of a bookshelf's and are still the only views of it, while half a second of a
    fast pan in the middle of a textured stretch is blur."""
    times, poses = _walk(n=900, still=0, pan=(600, 630))
    rng = np.random.default_rng(1)
    sharp = 50.0 + rng.random(len(times))
    sharp[200:400] = 3.0 + rng.random(200)                 # the ceiling: plain, sharp
    sharp[600:630] = 5.0                                   # a fast pan: blurred
    usable = np.ones(len(times), bool)
    rows = stray.select_frames(times, poses, sharp, usable, target=150).rows
    textured = sharp.copy()
    textured[200:400] = 50.0 + rng.random(200)
    baseline = stray.select_frames(times, poses, textured, usable, target=150).rows
    in_ceiling = lambda r: r[(r >= 200) & (r < 400)]      # noqa: E731
    # As many views of the ceiling as if it were a bookshelf, spread over all of it.
    assert len(in_ceiling(rows)) >= len(in_ceiling(baseline)) - 1 >= 5
    assert in_ceiling(rows).min() < 240 and in_ceiling(rows).max() > 360
    assert not np.any((rows >= 600) & (rows < 630))


def test_tracking_ok_drops_frames_around_a_relocalisation_jump():
    times, poses = _walk(still=0)
    poses[400:, :3, 3] += [1.5, 0.0, 0.0]                  # 1.5 m in one frame
    ok = stray.tracking_ok(times, poses)
    assert not ok[399] and not ok[400]
    assert not ok[400 - 10] and not ok[400 + 10]          # +-0.25 s
    assert ok[:380].all() and ok[420:].all()


# ---------------------------------------------------------------------------------------------
# Pairing video frames with odometry rows
# ---------------------------------------------------------------------------------------------

def _stray_counters(n_arframes: int, skipped=()) -> tuple[np.ndarray, np.ndarray]:
    """What DatasetEncoder writes for ARFrames 0..n-1 at 60 fps, minus `skipped` (the encoding
    semaphore was full): each saved frame's video PTS (counter before the increment / 60, so
    -1/60 first) and its odometry timestamp."""
    counters = [c - 1 for c in range(n_arframes) if c not in set(skipped)]
    pts = np.array(counters, float) / 60.0
    times = 500.0 + (np.array(counters) + 1) / 60.0
    return pts, times


def test_pair_video_one_frame_per_row():
    pts, times = _stray_counters(50, skipped=(10, 11, 30))
    p = stray.pair_video(pts, np.arange(len(times)), times)
    assert p.video_of_row == {r: r for r in range(len(times))} and p.note is None


@pytest.mark.parametrize("skipped", [(), (7, 20, 21, 33, 40)])
def test_pair_video_first_frame_lost(skipped):
    """The -1/60 frame trimmed by the edit list: video frame k is row k + 1, not row k."""
    pts, times = _stray_counters(60, skipped)
    p = stray.pair_video(pts[1:], np.arange(len(times)), times)
    assert p.rows_before == 1 and p.rows_after == 0 and p.note
    assert p.video_of_row == {r: r - 1 for r in range(1, len(times))}


def test_pair_video_last_frame_lost():
    pts, times = _stray_counters(60, skipped=(9, 25))
    p = stray.pair_video(pts[:-1], np.arange(len(times)), times)
    assert p.rows_before == 0 and p.rows_after == 1
    assert p.video_of_row == {r: r for r in range(len(times) - 1)}


def test_pair_video_refuses_what_it_cannot_place():
    pts, times = _stray_counters(80, skipped=(5, 15, 30, 45, 60, 70))
    rows = np.arange(len(times))
    middle = np.delete(pts, 20)                            # an append failed mid-recording
    with pytest.raises(ValueError, match="do not line up"):
        stray.pair_video(middle, rows, times)
    p = stray.pair_video(middle, rows, times, allow_mismatch=True)
    assert "allow-frame-mismatch" in p.note
    with pytest.raises(ValueError, match="no per-frame PTS"):
        stray.pair_video(None, rows, times, n_video_hint=len(times) - 1)
    with pytest.raises(ValueError, match="more than the odometry"):
        stray.pair_video(np.r_[pts, pts[-1] + 1 / 60], rows, times)
    assert stray.pair_video(None, rows, times, n_video_hint=len(times)).note is None


# ---------------------------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("yaw,t", [(37.0, (1.3, -0.4, -2.1)), (-150.0, (-0.5, 0.2, 0.9)),
                                   (91.3, (0.0, 0.0, 0.0))])
def test_align_recovers_a_known_yaw_and_translation(yaw, t):
    target = _room_cloud(seed=1, spacing=0.03)
    world = _room_cloud(seed=2, spacing=0.03)
    world = world[world[:, 0] < 1.5]                       # the recording saw part of the room
    world += np.random.default_rng(3).normal(0, 0.005, world.shape)
    T = aligner.yaw_transform(np.deg2rad(yaw), t)
    source = (world - T[:3, 3]) @ T[:3, :3]                # into the recording's own world
    a = aligner.align(source, target)
    err = (a.yaw_deg - yaw + 180) % 360 - 180
    assert abs(err) < 1.0
    assert np.linalg.norm(a.translation - np.asarray(t)) < 0.02
    assert a.residual < 0.01 and a.overlap > 0.9


def test_coarse_candidates_are_distinct_basins_plus_quarter_turns():
    target = _room_cloud(seed=1, spacing=0.05)
    source = target @ aligner.yaw_transform(np.deg2rad(40.0))[:3, :3]
    found = aligner.coarse_candidates(source, target)
    peaks = [c for c in found if not c.get("quarter_turn")]
    for i, a in enumerate(peaks):
        for b in peaks[i + 1:]:
            assert aligner._yaw_gap(a["yaw"], b["yaw"]) >= aligner.CANDIDATE_SEPARATION
    best = np.rad2deg(peaks[0]["yaw"])
    for quarter in (90, 180, 270):
        assert any(aligner._yaw_gap(c["yaw"], np.deg2rad(best + quarter)) < 1.0 for c in found)


def test_align_flags_a_symmetric_room_as_ambiguous():
    """A bare rectangular room fits as well turned half a turn: reported, not silently taken."""
    rng = np.random.default_rng(4)
    target = _box_surface(*ROOM, 0.03, rng)
    world = _box_surface(*ROOM, 0.03, rng)
    world = world[world[:, 0] < 1.0]
    T = aligner.yaw_transform(np.deg2rad(25.0), (0.4, 0.1, -0.3))
    a = aligner.align((world - T[:3, 3]) @ T[:3, :3], target)
    assert a.ambiguous and a.runner_up is not None
    assert abs((a.runner_up["yaw_deg"] - a.yaw_deg) % 360.0 - 180.0) < 5.0      # half a turn
    assert "next heading" in a.describe_margin()


@pytest.mark.parametrize("yaw,keep", [(63.0, lambda p: p[:, 0] < -0.5),
                                      (-120.0, lambda p: p[:, 2] > 0.3),
                                      (170.0, lambda p: (p[:, 0] > 0.0) & (p[:, 2] < 0.5))])
def test_align_partial_overlap_is_right_or_refused(yaw, keep):
    """The recording saw only part of the room: the result is the true heading, or it is flagged
    (ambiguous, or too little of it on the target) - never a turned fit that passes the gates."""
    target = _room_cloud(seed=1, spacing=0.03)
    world = _room_cloud(seed=2, spacing=0.03)
    world = world[keep(world)]
    world += np.random.default_rng(3).normal(0, 0.005, world.shape)
    T = aligner.yaw_transform(np.deg2rad(yaw), (0.7, -0.2, 1.1))
    a = aligner.align((world - T[:3, 3]) @ T[:3, :3], target)
    right = (abs((a.yaw_deg - yaw + 180) % 360 - 180) < 1.0
             and np.linalg.norm(a.translation - T[:3, 3]) < 0.02)
    passes = (a.residual <= aligner.MAX_RESIDUAL and a.inliers >= aligner.MIN_INLIERS
              and not a.ambiguous)
    assert right or not passes, a.summary()


# ---------------------------------------------------------------------------------------------
# Video and the whole import
# ---------------------------------------------------------------------------------------------

@needs_ffmpeg
@pytest.mark.parametrize("bframes", [0, 2])
def test_video_frames_are_counted_and_decoded_by_index(tmp_path, bframes):
    from PIL import Image

    pngs = tmp_path / "png"
    pngs.mkdir()
    tex = _texture()
    for k in range(30):
        Image.fromarray(_frame_image(k, tex, blur=k % 3 != 0)).save(pngs / f"f_{k:05d}.png")
    _encode(pngs, tmp_path / "rgb.mp4", bframes)
    assert stray.video_size(tmp_path / "rgb.mp4") == (W, H)
    scan = stray.scan_video(tmp_path / "rgb.mp4", (W, H))
    scores = scan.scores
    assert len(scores) == 30
    np.testing.assert_allclose(scan.pts, np.arange(30) / 60.0, atol=1e-3)
    sharp = np.arange(30) % 3 == 0
    assert scores[sharp].min() > 2 * scores[~sharp].max()
    got = dict(stray.decode_frames(tmp_path / "rgb.mp4", [0, 7, 29, 13], (W, H)))
    assert sorted(got) == [0, 7, 13, 29]
    for k, img in got.items():
        assert abs(float(img[8:24, 8:24, 0].mean()) - 2 * k) < 3


@needs_ffmpeg
def test_import_aligns_to_the_target_scan_and_loads_with_rgbd(tmp_path):
    from litereality_agent.pipeline.capture import import_stray

    rec = make_recording(tmp_path / "a1b2c3d4e5")
    target = tmp_path / "scans" / "room_1"
    make_target_scan(target)
    out = tmp_path / "scans" / "room_1_stray"
    report = import_stray(tmp_path / "a1b2c3d4e5", out, align_to=target, frames=30,
                          log=lambda _m: None)

    # Alignment: the known transform, within 1 degree / 2 cm.
    a = report["alignment"]
    assert abs((a["yaw_deg"] - TRUE_YAW + 180) % 360 - 180) < 1.0
    assert np.linalg.norm(np.array(a["translation_m"]) - TRUE_T) < 0.02
    assert a["residual_m"] < 0.05

    # Selection: sharp frames only, the 20 standing frames collapsed.
    frames = rgbd.load_frames(out)
    assert 10 <= len(frames) == report["frames"] <= 30
    metas = [json.loads((out / f"frame_{f.index:05d}.json").read_text()) for f in frames]
    video_frames = [m["stray"]["video_frame"] for m in metas]
    assert video_frames == [m["stray"]["row"] for m in metas]     # one frame per row
    assert all(k % 3 == 0 for k in video_frames)
    assert sum(k < 20 for k in video_frames) == 1
    assert [f.index for f in frames] == list(range(len(frames)))

    # Our json layout: row-major pose and K, as the LiteReality Scanner writes them.
    m0 = metas[0]
    assert set(m0) >= {"cameraPoseARFrame", "intrinsics", "frame_index", "time",
                       "averageVelocity", "averageAngularVelocity"}
    assert m0["intrinsics"] == [200.0, 0.0, 128.0, 0.0, 200.0, 96.0, 0.0, 0.0, 1.0]
    assert m0["cameraPoseARFrame"][12:] == [0.0, 0.0, 0.0, 1.0]

    # Each photo is the video frame its json names.
    from PIL import Image

    for f, m in zip(frames[:5], metas[:5]):
        with Image.open(f.image) as im:
            assert im.size == (W, H)
            block = np.asarray(im)[8:24, 8:24, 0].mean()
        assert abs(block - 2 * m["stray"]["row"]) < 4

    # Pose convention end to end: a depth pixel back-projected the way rgbd does lands on the
    # room surface it was ray-cast from, in the TARGET's (Z-up) frame.
    for f, m in zip(frames, metas):
        depth, _ = rgbd.read_lidar(f)
        hits = rec["hits"][m["stray"]["row"]]
        sx, sy = DW / f.size[0], DH / f.size[1]
        fx, fy = f.K[0, 0] * sx, f.K[1, 1] * sy
        cx, cy = (f.K[0, 2] + 0.5) * sx - 0.5, (f.K[1, 2] + 0.5) * sy - 0.5
        for v, u in [(10, 10), (24, 32), (40, 50)]:
            z = depth[v, u]
            cam = np.array([(u - cx) / fx * z, -(v - cy) / fy * z, -z, 1.0])
            world_zup = f.c2w @ cam
            expected = rgbd.ARKIT_TO_Z_UP @ np.r_[hits[v, u], 1.0]
            assert np.linalg.norm(world_zup[:3] - expected[:3]) < 0.03

    # The files quickroom / splatroom / the uploader look for.
    assert (out / "room.usdz").read_bytes() == b"usdz stand-in"
    assert (out / "roomplan" / "room.usdz").is_file()
    label = json.loads((out / "scan_label.json").read_text())
    assert label["scannerName"] == "Stray Scanner" and label["roomType"] == "office"
    alignment = json.loads((out / "scene_alignment.json").read_text())
    assert alignment["strayImport"]["alignedTo"] == str(target)
    assert not a["ambiguous"]
    cloud = stray.read_pcd(out / "pointcloud.pcd")
    assert len(cloud) > 1000
    lo, hi = ROOM
    inside = np.all((cloud > lo - 0.05) & (cloud < hi + 0.05), axis=1)
    assert inside.mean() > 0.98                            # in the target's world
    assert (out / "depth_00000.png").is_file() and (out / "conf_00000.png").is_file()

    # A second import into the same folder needs --overwrite.
    with pytest.raises(ValueError, match="overwrite"):
        import_stray(tmp_path / "a1b2c3d4e5", out, align_to=target, frames=30,
                     log=lambda _m: None)


@needs_ffmpeg
def test_import_pairs_photos_with_poses_when_the_first_video_frame_is_lost(tmp_path):
    """rgb.mp4 one frame short at the start: every photo must still sit with its own pose
    (video frame k is row k + 1), not one frame later than it."""
    from PIL import Image

    from litereality_agent.pipeline.capture import import_stray

    rec = make_recording(tmp_path / "rec", n=60, still=3, lose_first_video_frame=True)
    out = tmp_path / "scan"
    report = import_stray(tmp_path / "rec", out, frames=20, log=lambda _m: None)
    assert report["video"]["frames"] == 59 and report["video"]["odometry_rows"] == 60
    assert report["video"]["rows_without_frame"] == {"start": 1, "end": 0}
    assert any("first 1" in w for w in report["warnings"])
    frames = rgbd.load_frames(out)
    assert len(frames) == report["frames"] > 5
    for f in frames:
        m = json.loads((out / f"frame_{f.index:05d}.json").read_text())
        row = m["stray"]["row"]
        assert m["stray"]["video_frame"] == row - 1
        with Image.open(f.image) as im:
            assert abs(np.asarray(im)[8:24, 8:24, 0].mean() - 2 * row) < 4
        expected = rgbd.ARKIT_TO_Z_UP @ np.linalg.inv(rec["T"]) @ rec["poses_target"][row]
        np.testing.assert_allclose(f.c2w, expected, atol=1e-5)


@needs_ffmpeg
def test_import_refuses_an_ambiguous_alignment(tmp_path, monkeypatch):
    from litereality_agent.pipeline import capture

    real = aligner.align

    def ambiguous(source, target, **kw):
        a = real(source, target, **kw)
        a.ambiguous, a.margin = True, 0.02
        a.runner_up = {"yaw_deg": a.yaw_deg + 180.0, "score": a.score - 0.02}
        return a

    monkeypatch.setattr(capture.aligner, "align", ambiguous)
    make_recording(tmp_path / "rec", n=60, still=5)
    target = tmp_path / "room"
    make_target_scan(target)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="ambiguous.*accept-ambiguous"):
        capture.import_stray(tmp_path / "rec", out, align_to=target, frames=20,
                             log=lambda _m: None)
    assert not out.exists()
    report = capture.import_stray(tmp_path / "rec", out, align_to=target, frames=20,
                                  accept_ambiguous=True, log=lambda _m: None)
    assert report["alignment"]["ambiguous"]
    assert any("accept-ambiguous" in w for w in report["warnings"])


@needs_ffmpeg
def test_import_refuses_an_alignment_that_does_not_fit(tmp_path):
    from litereality_agent.pipeline.capture import import_stray

    make_recording(tmp_path / "rec", n=60, still=5)
    other = tmp_path / "other_room"
    other.mkdir()
    rng = np.random.default_rng(9)
    stray.write_pcd(other / "pointcloud.pcd", rng.uniform(-3, 3, (20000, 3)))   # not a room
    (other / "room.usdz").write_bytes(b"x")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="not trustworthy"):
        import_stray(tmp_path / "rec", out, align_to=other, frames=20, log=lambda _m: None)
    assert not out.exists()


@needs_ffmpeg
def test_import_a_zipped_legacy_recording_without_alignment(tmp_path):
    from litereality_agent.pipeline.capture import import_stray

    rec_dir = tmp_path / "f00d"
    make_recording(rec_dir, n=45, still=3, header="old")
    archive = tmp_path / "f00d.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in rec_dir.rglob("*"):
            if p.is_file():
                zf.write(p, Path("f00d") / p.relative_to(rec_dir))
    out = tmp_path / "scan"
    report = import_stray(archive, out, frames=12, log=lambda _m: None)
    frames = rgbd.load_frames(out)
    assert len(frames) == report["frames"] > 0
    assert report["alignment"] is None
    np.testing.assert_allclose(frames[0].K, K)             # from camera_matrix.csv
    assert not (out / "room.usdz").exists()


def test_cli_registers_import_stray():
    from litereality_agent import cli

    args = cli._parser().parse_args(["import-stray", "rec.zip", "--out", "scan",
                                     "--align-to", "atrium", "--frames", "500"])
    assert args.handler is cli._import_stray
    assert (args.recording, args.out, args.align_to, args.frames) == ("rec.zip", "scan",
                                                                      "atrium", 500)
    assert not args.accept_ambiguous and not args.allow_frame_mismatch
    args = cli._parser().parse_args(["import-stray", "r", "--out", "s", "--accept-ambiguous",
                                     "--allow-frame-mismatch"])
    assert args.accept_ambiguous and args.allow_frame_mismatch
