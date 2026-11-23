"""Phone-video import without a GPU: the pose / intrinsics conventions, the Sim(3) math (closed
form, gravity, yaw + translation + scale ICP), the Modal app's pure helpers, decoding and frame
selection on tiny generated videos, and a whole import against a fake `reconstruct` that returns
the exact geometry of a synthetic room in a tilted, shifted, wrongly scaled model world."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from litereality_agent.models.capture import sim3, stray
from litereality_agent.models.capture import video as vid
from litereality_agent.models.splat import rgbd

REPO = Path(__file__).resolve().parents[1]
APP_FILE = REPO / "deploy" / "modal" / "geometry" / "app.py"

try:
    FFMPEG = stray.ffmpeg_binary()
    stray.ffmpeg_binary("ffprobe")
except RuntimeError:
    FFMPEG = None
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("geometry_deploy_under_test", APP_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis].astype(float)


def _sim(scale: float, R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = scale * R, t
    return T


# ---- the synthetic room (target ARKit world: Y up, metres) -----------------------------------

ROOM = (np.array([-2.5, -1.4, -2.0]), np.array([2.5, 1.2, 2.0]))
FURNITURE = [(np.array([1.0, -1.4, -1.9]), np.array([2.4, -0.6, -1.0])),
             (np.array([-2.45, -1.4, 0.5]), np.array([-1.9, 0.6, 1.9])),
             (np.array([-0.4, -1.4, 1.2]), np.array([0.5, -0.9, 1.95]))]


def _raycast(origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Distance along each unit-free direction (t with hit = origin + t d) to the room."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lo, hi = ROOM
        best = np.min(np.where(dirs > 0, (hi - origin) / dirs, (lo - origin) / dirs), axis=1)
        for blo, bhi in FURNITURE:
            t1, t2 = (blo - origin) / dirs, (bhi - origin) / dirs
            tn = np.nanmax(np.minimum(t1, t2), axis=1)
            tf = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tn <= tf) & (tn > 0)
            best = np.where(hit & (tn < best), tn, best)
    return best


def _surface_points(step: float = 0.04) -> np.ndarray:
    """Points on the room's visible surfaces (walls, floor, ceiling, furniture faces)."""
    pts = []
    for lo, hi in [ROOM, *FURNITURE]:
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            g = np.meshgrid(*[np.arange(lo[a], hi[a] + 1e-9, step) for a in others], indexing="ij")
            for value in (lo[axis], hi[axis]):
                p = np.zeros((g[0].size, 3))
                p[:, others[0]], p[:, others[1]], p[:, axis] = g[0].ravel(), g[1].ravel(), value
                pts.append(p)
    pts = np.concatenate(pts)
    inside = np.zeros(len(pts), bool)
    for lo, hi in FURNITURE:                       # furniture faces buried in a wall / floor
        inside |= np.all((pts > lo + 1e-6) & (pts < hi - 1e-6), axis=1)
    return pts[~inside]


# ---- conventions ------------------------------------------------------------------------------

def test_cv_gl_round_trip_and_camera_axes():
    c2w = np.eye(4)
    c2w[:3, :3] = _rot("y", 30) @ _rot("x", -10)
    c2w[:3, 3] = [0.3, 1.2, -0.7]
    gl = vid.cv_to_gl(c2w)
    assert np.allclose(vid.cv_to_gl(gl), c2w)
    # The OpenCV camera looks down +z; the OpenGL one down -z: the same direction in the world.
    assert np.allclose(c2w[:3, 2], -gl[:3, 2])
    assert np.allclose(c2w[:3, 1], -gl[:3, 1])          # y down vs y up
    assert np.allclose(gl[:3, 3], c2w[:3, 3])


def test_K_mapping_round_trip_and_projection(app):
    photo = (1920, 1080)
    sent = vid.scaled_size(photo, vid.UPLOAD_LONG)
    prep = app.prep_geometry(sent, (518, 294))
    assert sent == (1036, 582) and prep["resized"] == [523, 294] and prep["crop"] == [2, 0]
    K_photo = np.array([[1450.0, 0.0, 958.3], [0.0, 1452.0, 541.2], [0.0, 0.0, 1.0]])
    K_model = vid.photo_K_to_model(K_photo, prep, photo)
    assert np.allclose(vid.model_K_to_photo(K_model, prep, photo), K_photo)
    # A point projects to model pixel (u, v); the same point projects to the photo pixel that
    # the crop + resize maps (u, v) to.
    X = np.array([0.4, -0.2, 2.5])
    u = K_model @ X
    u = u[:2] / u[2]
    p = K_photo @ X
    p = p[:2] / p[2]
    rw, rh = prep["resized"]
    expect = (u + np.array(prep["crop"]) + 0.5) * np.array([photo[0] / rw, photo[1] / rh]) - 0.5
    assert np.allclose(p, expect, atol=1e-9)


def test_model_map_to_photo_grid_covers_the_photo(app):
    prep = app.prep_geometry((1036, 582), (518, 294))     # 1036 x 582 -> 523 x 294, crop 2
    xs = np.tile(np.arange(518, dtype=np.float32), (294, 1))
    out = vid.model_map_to_photo_grid(xs, prep, (256, 144), fill=-1)
    assert out.shape == (144, 256)
    # Columns at the photo's edges fall outside the model's crop; the middle maps through.
    assert (out[:, 0] == -1).all() and (out[:, -1] == -1).all()
    mid = out[72, 128]
    rw = prep["resized"][0]
    assert abs(mid - ((128 + 0.5) * rw / 256 - 0.5 - prep["crop"][0])) <= 0.5


def test_confidence_levels_are_global_quantiles():
    rng = np.random.default_rng(0)
    conf = rng.uniform(1, 10, (4, 20, 30)).astype(np.float32)
    valid = np.ones(conf.shape, bool)
    valid[0, :5] = False
    levels, q = vid.confidence_levels(conf, valid)
    assert set(np.unique(levels)) <= {0, 1, 2}
    assert (levels[~valid] == 0).all()
    share = np.bincount(levels[valid].ravel(), minlength=3) / valid.sum()
    assert abs(share[0] - vid.CONF_LOW) < 0.02 and abs(share[2] - (1 - vid.CONF_MID)) < 0.02



def test_confidence_levels_keep_order_when_many_pixels_tie_at_the_floor():
    # MapAnything's confidence is 1 + exp(x): in float16 a third of a plain room's pixels round
    # to exactly 1.0, which made both quantiles 1.0 and every pixel level 2.
    rng = np.random.default_rng(3)
    conf = rng.uniform(1.01, 6.0, (3, 20, 30)).astype(np.float32)
    conf[:, :8] = 1.0                                   # 40% tied at the floor
    valid = np.ones(conf.shape, bool)
    levels, q = vid.confidence_levels(conf, valid)
    assert q["low"] == q["mid"] == 1.0
    assert (levels[conf == 1.0] == 0).all()
    assert (levels[conf > 1.0] == 2).all()


def test_unpack_reads_log_confidence():
    n, h, w = 2, 4, 5
    conf_log = np.linspace(-20, 3, n * h * w).reshape(n, h, w)
    buf = io.BytesIO()
    np.savez(buf, c2w=np.tile(np.eye(4), (n, 1, 1)), K=np.tile(np.eye(3), (n, 1, 1)),
             depth=np.ones((n, h, w), np.float16), conf_log=conf_log.astype(np.float16))
    prep = [{"size": [w, h], "resized": [w, h], "crop": [0, 0]}] * n
    geo = vid.unpack({"arrays": buf.getvalue(), "prep": prep})
    assert np.all(np.diff(geo.conf.ravel()) > 0)      # order kept (float16 conf would tie)

# ---- Sim(3) -----------------------------------------------------------------------------------

def test_robust_sim3_recovers_a_known_similarity_with_outliers():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(2000, 3)) * 2
    R = _rot("z", 25) @ _rot("x", -40) @ _rot("y", 110)
    T = _sim(1.37, R, [0.5, -2.0, 3.0])
    dst = sim3.apply(T, src) + rng.normal(scale=0.002, size=src.shape)
    dst[:300] += rng.normal(scale=1.0, size=(300, 3))           # 15% gross outliers
    fit = sim3.robust_sim3(src, dst)
    assert fit["scale"] == pytest.approx(1.37, rel=2e-3)
    assert np.allclose(fit["T"], T, atol=0.01)
    s, Rf, t = sim3.umeyama(src[300:], dst[300:])
    assert s == pytest.approx(1.37, rel=1e-3) and np.allclose(Rf, R, atol=1e-3)


def test_apply_sim3_to_pose_moves_points_and_cameras_together():
    T = _sim(0.8, _rot("y", 70), [1.0, 0.2, -0.5])
    c2w = np.eye(4)
    c2w[:3, :3] = _rot("x", 20)
    c2w[:3, 3] = [0.3, 1.0, 2.0]
    X_cam = np.array([0.2, -0.1, -3.0])
    X_world = c2w[:3, :3] @ X_cam + c2w[:3, 3]
    moved = sim3.apply_sim3_to_pose(T, c2w)
    assert np.allclose(moved[:3, :3] @ moved[:3, :3].T, np.eye(3))
    # The camera sees the moved point at the same direction, at 0.8 x the depth.
    assert np.allclose(moved[:3, :3] @ (0.8 * X_cam) + moved[:3, 3], sim3.apply(T, X_world))


def test_estimate_up_and_level_transform_recover_gravity():
    rng = np.random.default_rng(2)
    pts = _surface_points(0.05)
    tilt = _rot("x", 17) @ _rot("z", -9)
    tilted = pts @ tilt.T
    true_up = tilt @ np.array([0.0, 1.0, 0.0])
    guess = true_up + rng.normal(scale=0.15, size=3)       # the cameras' rough mean up
    up, stats = sim3.estimate_up(tilted, guess)
    assert np.rad2deg(np.arccos(np.clip(up @ true_up, -1, 1))) < 0.3, stats
    L = sim3.level_transform(up, origin=np.zeros(3), forward=tilt @ np.array([1.0, 0.0, 0.0]))
    assert np.allclose(L[:3, :3] @ up, [0, 1, 0], atol=1e-9)
    f = L[:3, :3] @ (tilt @ np.array([1.0, 0.0, 0.0]))
    assert f[2] < 0 and abs(f[0]) < 1e-6                      # forward turned onto -Z


def test_align_sim3_recovers_yaw_translation_and_scale():
    target = _surface_points(0.03)
    # The source: the room seen by "a model" at 0.9 x scale, turned 63 deg, shifted.
    T_true = _sim(1 / 0.9, _rot("y", 63.0), [0.7, -0.3, 1.1])     # source -> target
    rng = np.random.default_rng(3)
    src = sim3.apply(np.linalg.inv(T_true), target[rng.random(len(target)) < 0.6])
    src += rng.normal(scale=0.004, size=src.shape)
    fit = sim3.align_sim3(src, target)
    assert fit.scale == pytest.approx(1 / 0.9, rel=0.01)
    assert abs(((fit.yaw_deg - 63.0) + 180) % 360 - 180) < 0.5
    assert np.allclose(fit.translation, T_true[:3, 3], atol=0.03)
    assert fit.residual < 0.01 and fit.inliers > 0.9
    assert not fit.ambiguous


# ---- the Modal app's helpers -----------------------------------------------------------------

def test_app_intrinsics_option_matches_the_client_conversion(app):
    with pytest.raises(ValueError):
        app.parse_options({"intrinsics": [[0, 0, 1], [0, 1, 1], [0, 0, 1]]})
    assert app.parse_options(None)["intrinsics"] is None
    size = (582, 1036)
    prep = app.prep_geometry(size, (294, 518))
    K = np.array([[835.0, 0.0, 290.5], [0.0, 836.0, 517.5], [0.0, 0.0, 1.0]])
    opts = app.parse_options({"intrinsics": K.tolist()})
    model = app.image_K_to_model(np.asarray(opts["intrinsics"]), prep)
    assert np.allclose(model, vid.photo_K_to_model(K, prep, size))
    assert np.allclose(vid.model_K_to_photo(model, prep, size), K)


def test_app_options_and_pass_plan(app):
    assert app.parse_options(None)["max_views"] == app.DEFAULT_MAX_VIEWS
    with pytest.raises(ValueError):
        app.parse_options({"nope": 1})
    assert app.plan_passes(200, 320, 32) == [list(range(200))]
    plan = app.plan_passes(1000, 300, 32)
    anchors = plan[0][:32]
    assert anchors[0] == 0 and anchors[-1] == 999
    assert all(p[:32] == anchors and len(p) <= 300 for p in plan)
    covered = sorted({i for p in plan for i in p})
    assert covered == list(range(1000))
    others = [i for p in plan for i in p[32:]]
    assert len(others) == len(set(others)) == 1000 - len(set(anchors))


def test_app_chain_maps_a_pass_onto_pass_zero(app):
    rng = np.random.default_rng(4)
    T = _sim(1.08, _rot("y", 4) @ _rot("x", -2), [0.05, -0.02, 0.1])   # pass k -> pass 0
    ref, cur = {}, {}
    for a in (0, 5, 9):
        pts0 = rng.normal(size=(30, 40, 3)) + [0, 0, 3]
        conf = rng.uniform(1, 5, (30, 40)).astype(np.float32)
        mask = np.ones((30, 40), bool)
        ref[a] = {"pts3d": pts0, "conf": conf, "mask": mask}
        cur[a] = {"pts3d": sim3.apply(np.linalg.inv(T), pts0.reshape(-1, 3)).reshape(pts0.shape),
                  "conf": conf, "mask": mask}
    fit = app._chain(ref, cur, [0, 5, 9], rng)
    assert np.allclose(fit["T"], T, atol=1e-6)
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    assert np.allclose(app.apply_sim3_to_pose(fit["T"], pose), sim3.apply_sim3_to_pose(T, pose))


# ---- decode + selection on tiny videos --------------------------------------------------------

def _encode(frames: list[np.ndarray], path: Path, fps: int = 30, rotation: int | None = None):
    h, w = frames[0].shape[:2]
    raw = path.with_suffix(".raw.mp4")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                    f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12",
                    "-pix_fmt", "yuv420p", str(raw)],
                   input=b"".join(f.tobytes() for f in frames), check=True)
    if rotation is None:
        raw.replace(path)
        return
    subprocess.run([FFMPEG, "-v", "error", "-y", "-display_rotation", str(rotation), "-i",
                    str(raw), "-c", "copy", str(path)], check=True)
    raw.unlink()


def _pattern(w: int, h: int, shift: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    big = rng.integers(0, 255, (h // 8 + 2, w // 8 + 40, 3), dtype=np.uint8)
    img = np.kron(big, np.ones((8, 8, 1), np.uint8))
    return np.ascontiguousarray(img[:h, shift:shift + w])


def _blur(img: np.ndarray, k: int = 9) -> np.ndarray:
    out = img.astype(np.float32)
    for axis in (0, 1):
        out = sum(np.roll(out, d, axis=axis) for d in range(-(k // 2), k // 2 + 1)) / k
    return out.astype(np.uint8)


@needs_ffmpeg
def test_probe_rotation_decode_and_selection(tmp_path):
    w, h = 160, 96
    frames = [_pattern(w, h, 2 * i) for i in range(40)]          # a slow pan
    frames += [frames[-1]] * 30                                   # standing still
    frames += [_blur(_pattern(w, h, 80 + 6 * i, seed=1)) for i in range(30)]   # a blurry pan
    frames += [_pattern(w, h, 4 * i, seed=2) for i in range(30)]
    path = tmp_path / "clip.mp4"
    _encode(frames, path, rotation=90)

    info = vid.probe(path)
    assert info.coded == (w, h) and abs(info.rotation) == 90
    assert info.size == (h, w)                                    # shown portrait
    scan = vid.scan_video(info)
    assert len(scan.scores) == len(frames)
    assert np.allclose(np.diff(scan.times), 1 / 30, atol=2e-3)
    assert scan.thumbs.shape[1:] == (32, 20)                      # portrait thumbnails

    sel = vid.select_frames(scan.times, scan.scores, scan.thumbs, target=20)
    idx = sel.indices
    assert 8 <= len(idx) <= 30
    assert np.all(np.diff(idx) > 0)
    still = idx[(idx >= 40) & (idx < 70)]
    assert len(still) <= 2, idx                                   # one view of the still spot
    blurry = idx[(idx >= 70) & (idx < 100)]
    sharp_mean = np.mean(scan.scores[:40])
    assert np.all(scan.scores[blurry] < sharp_mean)               # sanity: they are blurry
    assert len(blurry) < len(idx[(idx < 40)]) + len(idx[idx >= 100])

    got = dict(vid.decode_frames(info, idx[:3], (h, w)))
    assert sorted(got) == sorted(int(i) for i in idx[:3])
    assert all(img.shape == (w, h, 3) for img in got.values())    # rotated to portrait


# ---- a whole import against a fake model ------------------------------------------------------

PHOTO = (160, 120)
MODEL = (112, 70)                       # a 518-style crop: 160 x 120 -> 112 x 84 -> crop 7 rows
K_PHOTO = np.array([[120.0, 0.0, 79.5], [0.0, 120.0, 59.5], [0.0, 0.0, 1.0]])
MODEL_SCALE = 0.9
T_MODEL = _sim(MODEL_SCALE, _rot("x", 21) @ _rot("z", -12) @ _rot("y", 140), [2.0, -1.0, 0.4])


def _true_poses(n: int) -> np.ndarray:
    """OpenGL camera -> target world: a walk around the middle, looking at the walls."""
    poses = []
    for k in range(n):
        yaw = 360.0 * k / n
        c2w = np.eye(4)
        c2w[:3, :3] = _rot("y", yaw) @ _rot("x", -12 + 6 * np.sin(k))
        c2w[:3, 3] = [0.6 * np.cos(k / 2), 0.1, 0.5 * np.sin(k / 3)]
        poses.append(c2w)
    return np.stack(poses)


class FakeModel:
    """`reconstruct`'s contract with exact geometry, in a tilted, shifted, 0.9-scaled world."""

    def __init__(self, app, poses, bad=()):
        self.app, self.poses, self.calls, self.bad = app, poses, 0, set(bad)

    def remote(self, frames, options):
        from PIL import Image

        self.calls += 1
        assert len(frames) == len(self.poses)
        with Image.open(io.BytesIO(frames[0])) as im:
            sent = im.size
        prep = self.app.prep_geometry(sent, MODEL)
        K_m = vid.photo_K_to_model(K_PHOTO, prep, PHOTO)
        v, u = np.mgrid[0:MODEL[1], 0:MODEL[0]]
        rays_cam = np.stack([(u - K_m[0, 2]) / K_m[0, 0], (v - K_m[1, 2]) / K_m[1, 1],
                             np.ones_like(u, float)], -1).reshape(-1, 3)
        c2ws, depths = [], []
        for gl in self.poses:
            cv = vid.cv_to_gl(gl)
            t = _raycast(cv[:3, 3], rays_cam @ cv[:3, :3].T)
            depths.append((t * MODEL_SCALE).reshape(MODEL[1], MODEL[0]))   # z-depth: ray z = 1
            moved = sim3.apply_sim3_to_pose(T_MODEL, cv)
            if len(c2ws) in self.bad:               # a lost pose: panned 25 deg, depth unchanged
                moved[:3, :3] = moved[:3, :3] @ _rot("y", 25.0)
            c2ws.append(moved)
        n = len(self.poses)
        buf = io.BytesIO()
        np.savez(buf, c2w=np.stack(c2ws), K=np.repeat(K_m[None], n, 0),
                 depth=np.stack(depths).astype(np.float16),
                 conf=np.full((n, MODEL[1], MODEL[0]), 5.0, np.float16))
        return {"frames": n, "arrays": buf.getvalue(), "model_size": list(MODEL),
                "prep": [prep] * n, "passes": [{"kind": "reconstruct", "views": n}],
                "gpu": "fake", "seconds": {"total": 0.0}, "model": "fake"}


@needs_ffmpeg
def test_import_video_end_to_end_with_a_fake_model(tmp_path, app):
    frames = [_pattern(*PHOTO, 3 * i, seed=i % 3) for i in range(48)]
    path = tmp_path / "room.mp4"
    _encode(frames, path)
    target = tmp_path / "target"
    target.mkdir()
    stray.write_pcd(target / "pointcloud.pcd", _surface_points(0.03))
    (target / "room.usdz").write_bytes(b"usdz")
    (target / "scan_label.json").write_text(json.dumps({"roomType": "office"}))

    # Which frames get picked is decided before the model is called: find out, then pose them.
    info = vid.probe(path)
    scan = vid.scan_video(info)
    picked = vid.select_frames(scan.times, scan.scores, scan.thumbs, target=16).indices
    poses = _true_poses(len(picked))
    fake = FakeModel(app, poses)
    out = tmp_path / "scan"
    report = vid.import_video(path, out, align_to=target, frames=16, function=fake,
                              log=lambda m: None)
    assert fake.calls == 1 and report["frames"] == len(picked)
    assert report["frame_check"]["dropped"] == 0 and report["frame_check"]["failed"] == 0
    assert report["alignment"]["scale_spread"] < 0.01
    assert report["intrinsics_check"]["fx_photo"] == pytest.approx(K_PHOTO[0, 0])
    first = json.loads((out / "frame_00000.json").read_text())
    assert first["videoCheck"]["status"] in ("ok", "degenerate")
    assert report["scale"] == pytest.approx(1 / MODEL_SCALE, rel=0.01)
    assert report["alignment"]["residual_m"] < 0.01
    assert (out / "room.usdz").read_bytes() == b"usdz"
    assert json.loads((out / "scan_label.json").read_text())["roomType"] == "office"

    loaded = rgbd.load_frames(out)
    assert len(loaded) == len(picked)
    to_arkit = np.linalg.inv(rgbd.ARKIT_TO_Z_UP)
    for f, true in zip(loaded, poses):
        c2w = to_arkit @ f.c2w
        assert np.linalg.norm(c2w[:3, 3] - true[:3, 3]) < 0.02
        ang = np.rad2deg(np.arccos(np.clip((np.trace(c2w[:3, :3].T @ true[:3, :3]) - 1) / 2,
                                           -1, 1)))
        assert ang < 0.5
        assert np.allclose(f.K, K_PHOTO, atol=1e-6)
        assert f.size == PHOTO
    # The written depth back-projects onto the room's surfaces (rgbd's own convention).
    from scipy.spatial import cKDTree

    tree = cKDTree(_surface_points(0.01))
    sf = stray.StrayFrame(0, "00000", 0.0, to_arkit @ loaded[3].c2w, loaded[3].K,
                          loaded[3].depth, loaded[3].conf)
    pts = stray.frame_points(sf, PHOTO, stride=1, max_depth=10.0)
    assert len(pts) > 200
    assert np.median(tree.query(pts)[0]) < 0.02
    rep = json.loads((out / vid.REPORT).read_text())
    assert rep["model_cached"] is False

    # A second import of the same picks reuses the model's reply: no second GPU call.
    report2 = vid.import_video(path, out, align_to=target, frames=16, function=fake,
                               overwrite=True, scales=(1.1,), log=lambda m: None)
    assert fake.calls == 1 and report2["model_cached"] is True

    # Scored against itself (as against a LiDAR reference in the same world): no error.
    from litereality_agent.models.capture import video_eval

    result = video_eval.evaluate(out, out, viz_dir=tmp_path / "viz", viz_count=2,
                                 reprojection_check=False, log=lambda m: None)
    assert result["nearest_viewpoint"]["pairs"] == len(picked)
    assert result["nearest_viewpoint"]["unpaired"] == 0
    assert result["nearest_viewpoint"]["position_median_m"] == 0.0
    g = result["geometry_registration"]
    assert g["frames"] == len(picked) and g["ok"] + g["degenerate"] + g["failed"] == len(picked)
    assert g["ok"] + g["degenerate"] >= len(picked) // 2
    assert g["position_median_m"] < 0.01
    assert result["depth_vs_lidar"]["abs_median_m"] < 0.01
    assert abs(result["cloud_vs_lidar"]["scale_correction"]) < 0.005
    # the import's scale undone: the fake model's metric depth was 0.9 x the truth
    raw = result["depth_raw_metric"]
    assert raw["model_scale_error"] == pytest.approx(MODEL_SCALE - 1, abs=0.01)
    assert raw["rel_median"] == pytest.approx(1 - MODEL_SCALE, abs=0.01)
    assert len(result["visualisations"]) == 2
    assert all(Path(p).is_file() for p in result["visualisations"])


def test_matched_rotation_ignores_a_quarter_turn_image_roll():
    from litereality_agent.models.capture import video_eval

    R = _rot("y", 30) @ _rot("x", -10)
    roll = _rot("z", 90)                                # portrait frame of a landscape sensor
    assert video_eval.rotation_difference_deg(R @ roll, R) < 1e-6
    assert abs(video_eval.rotation_difference_deg(R @ _rot("x", 7) @ roll, R) - 7.0) < 1e-6


@needs_ffmpeg
def test_import_video_drops_a_frame_whose_geometry_misses_the_target(tmp_path, app):
    frames = [_pattern(*PHOTO, 3 * i, seed=i % 3) for i in range(48)]
    path = tmp_path / "room.mp4"
    _encode(frames, path)
    target = tmp_path / "target"
    target.mkdir()
    stray.write_pcd(target / "pointcloud.pcd", _surface_points(0.03))
    info = vid.probe(path)
    scan = vid.scan_video(info)
    picked = vid.select_frames(scan.times, scan.scores, scan.thumbs, target=16).indices
    fake = FakeModel(app, _true_poses(len(picked)), bad={4})
    out = tmp_path / "scan"
    report = vid.import_video(path, out, align_to=target, frames=16, function=fake,
                              self_calibrate=False, log=lambda m: None)
    assert report["frame_check"]["dropped_picks"] == [4]
    assert report["frames"] == len(picked) - 1 == len(rgbd.load_frames(out))
    assert json.loads((out / "frame_00004.json").read_text())["video"]["pick"] == 5
    # kept on request, flagged in its JSON and the warnings
    report = vid.import_video(path, out, align_to=target, frames=16, function=fake,
                              overwrite=True, keep_bad_frames=True, self_calibrate=False,
                              log=lambda m: None)
    assert report["frames"] == len(picked) and any("kept" in w for w in report["warnings"])
    assert vid.is_bad(json.loads((out / "frame_00004.json").read_text())["videoCheck"])


def test_evaluate_refuses_scans_in_different_worlds(tmp_path):
    from litereality_agent.models.capture import video_eval

    scan = tmp_path / "video"
    scan.mkdir()
    (scan / "scene_alignment.json").write_text(json.dumps({"videoImport": {"alignedTo": None}}))
    with pytest.raises(ValueError, match="own world"):
        video_eval.evaluate(scan, scan, log=lambda m: None)
    from scipy.spatial import cKDTree

    ref = _surface_points(0.05)
    assert video_eval.same_world(ref + [0.0, 0.0, 0.01], cKDTree(ref)) > 0.99
    assert video_eval.same_world(ref + [5.5, 0.0, 4.5], cKDTree(ref)) < video_eval.MIN_SAME_WORLD


def test_cli_import_video_reference_needs_align_to(tmp_path):
    from litereality_agent import cli

    with pytest.raises(SystemExit, match="--align-to"):
        cli.main(["import-video", str(tmp_path / "v.mov"), "--out", str(tmp_path / "o"),
                  "--reference", str(tmp_path)])


def test_frame_check_flags_degenerate_and_failed_registrations():
    from scipy.spatial import cKDTree

    from litereality_agent.models.capture import video_eval
    from litereality_agent.models.capture.align import estimate_normals

    tgt = _surface_points(0.02)
    tree = cKDTree(tgt)
    normals = estimate_normals(tgt, tree)
    lo, hi = ROOM
    cam = np.eye(4)
    cam[:3, 3] = [0.0, 0.0, 0.5]
    # a corner of the room (three walls) seen 6 cm off: pinned down in all 6 DoF
    corner = tgt[(tgt[:, 0] < lo[0] + 1.2) & (tgt[:, 2] < lo[2] + 1.2) & (tgt[:, 1] < lo[1] + 1.0)]
    shift = np.array([0.04, -0.03, 0.03])
    c = video_eval.frame_check(corner + shift, cam, tgt, tree, normals)
    assert c["status"] == "ok" and c["position_m"] == pytest.approx(np.linalg.norm(shift), abs=0.01)
    # one blank wall slid along itself: nothing pins that direction down
    wall = tgt[(np.abs(tgt[:, 2] - hi[2]) < 1e-6) & (np.abs(tgt[:, 0]) < 1.0)
               & (tgt[:, 1] > -1.0) & (tgt[:, 1] < 1.0)]
    c = video_eval.frame_check(wall + [0.15, 0.0, 0.0], cam, tgt, tree, normals)
    assert c["status"] == "degenerate" and c["constraint"] < video_eval.DEGENERATE_RATIO
    # far from every surface: not registered, a failure (not left out)
    c = video_eval.frame_check(corner + [-1.0, -1.0, -1.0], cam, tgt, tree, normals)
    assert c["status"] == "failed"


def _synthetic_corr(f_true: float, f_scan: float, n_frames: int = 6):
    """Video frames whose poses are right but whose K says `f_scan`, with exact 2D-3D matches
    projected with `f_true`."""
    from litereality_agent.models.capture import video_eval

    rng = np.random.default_rng(5)
    W, H = 1080, 1920
    frames, corr = [], {}
    for i in range(n_frames):
        c2w = np.eye(4)
        c2w[:3, :3] = _rot("y", 40 * i) @ _rot("x", 5)
        c2w[:3, 3] = [0.2 * i, 0.0, -0.1 * i]
        pc = np.stack([rng.uniform(-1.5, 1.5, 200), rng.uniform(-2.5, 2.5, 200),
                       -rng.uniform(1.5, 4.0, 200)], 1)             # OpenGL camera, in front
        X = pc @ c2w[:3, :3].T + c2w[:3, 3]
        K_true = np.array([[f_true, 0, W / 2 - 0.5], [0, f_true, H / 2 - 0.5], [0, 0, 1.0]])
        uv, _ = video_eval.project_points(X, c2w, K_true)
        K_scan = K_true.copy()
        K_scan[0, 0] = K_scan[1, 1] = f_scan
        frames.append(stray.StrayFrame(i, f"{i:05d}", 0.0, c2w, K_scan, None, None))
        corr[i] = {"uv": uv, "xyz": X, "refs": []}
    return frames, corr


def test_reprojection_and_focal_search_see_a_wrong_focal():
    from litereality_agent.models.capture import video_eval

    frames, corr = _synthetic_corr(1550.0, 1550.0)
    rep = video_eval.reprojection(frames, corr)
    assert rep["frames"] == 6 and rep["median_px"] < 1e-6
    frames, corr = _synthetic_corr(1550.0, 1037.0)
    rep = video_eval.reprojection(frames, corr)
    assert rep["median_px"] > 50                         # the wrong K is visible in pixels
    fs = video_eval.focal_search(frames, corr)
    assert fs["focal"] == pytest.approx(1550.0, rel=0.04)
    assert fs["cost_at_best_px"] < 0.5 * fs["cost_at_scan_fx_px"]
    assert fs["pose_vs_pnp"]["frames"] == 6
    assert fs["pose_vs_pnp"]["position_median_m"] < 0.06       # grid step ~3%


def test_selfcal_essential_criterion_finds_the_focal():
    from litereality_agent.models.capture import selfcal

    f, W, H = 1500.0, 1080, 1920
    K = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])
    R = _rot("y", 12) @ _rot("x", 4)
    t = np.array([0.3, 0.05, 0.1])
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(K).T @ tx @ R @ np.linalg.inv(K)
    focals = 1037.0 * np.asarray(selfcal.FOCAL_GRID)
    costs = [selfcal.essential_cost(F, g, ((W - 1) / 2, (H - 1) / 2)) for g in focals]
    assert focals[int(np.argmin(costs))] == pytest.approx(f, rel=0.02)
    assert selfcal.essential_cost(F, f, ((W - 1) / 2, (H - 1) / 2)) < 1e-9
    # pairs need a turn and a baseline
    poses = np.stack([np.eye(4)] * 3)
    poses[1, :3, :3], poses[1, :3, 3] = _rot("y", 10), [0.2, 0, 0]
    poses[2, :3, :3] = _rot("y", 10)                 # turned in place: no baseline from 1
    assert selfcal.pairs_by_rotation(poses) == [(0, 1)]


def test_align_sim3_verdict_describes_the_transform_it_returns(monkeypatch):
    """When the re-run at the found scale prefers another heading (which then fits worse), the
    verdict must compare the kept transform with that heading, not repeat the re-run's own."""
    from dataclasses import replace

    from litereality_agent.models.capture import align as aligner

    target = _surface_points(0.03)
    T_true = _sim(1 / 0.9, _rot("y", 63.0), [0.7, -0.3, 1.1])
    rng = np.random.default_rng(3)
    src = sim3.apply(np.linalg.inv(T_true), target[rng.random(len(target)) < 0.6])
    real = aligner.align
    calls = []

    def fake_align(source, tgt, **kw):
        fit = real(source, tgt, **kw)
        calls.append(fit)
        if len(calls) == 2:                  # the verdict pass: a quarter turn off, "confident"
            c = sim3.apply(fit.transform, source).mean(0)
            turn = aligner.yaw_transform(np.pi / 2)
            turn[:3, 3] = c - turn[:3, :3] @ c
            return replace(fit, transform=turn @ fit.transform, ambiguous=False, margin=0.9,
                           runner_up=None)
        return fit

    monkeypatch.setattr(aligner, "align", fake_align)
    fit = sim3.align_sim3(src, target, scales=(1 / 0.9,))
    assert len(calls) == 2
    assert abs(((fit.yaw_deg - 63.0) + 180) % 360 - 180) < 0.5          # T kept
    assert fit.runner_up is not None and "note" in fit.runner_up
    assert abs(((fit.runner_up["yaw_deg"] - fit.yaw_deg) + 180) % 360 - 180) > 45
    assert fit.margin == pytest.approx(fit.score - fit.runner_up["score"], abs=1e-3)
    assert fit.margin != 0.9


def test_scale_spread_counts_near_equal_fits_at_the_same_heading():
    T = _sim(0.934, _rot("y", -53.4), [0, 0, 0])
    tries = [{"scale": 0.934, "yaw_deg": -53.4, "score": 0.761},
             {"scale": 0.950, "yaw_deg": -53.0, "score": 0.745},
             {"scale": 0.954, "yaw_deg": -53.2, "score": 0.742},
             {"scale": 1.200, "yaw_deg": 36.0, "score": 0.75},      # another heading
             {"scale": 0.700, "yaw_deg": -53.4, "score": 0.40},     # a poor fit
             {"start_scale": 1.3, "error": "x"}]
    assert sim3.scale_spread(tries, T, 0.761) == pytest.approx((0.954 - 0.934) / 0.934, abs=1e-6)
