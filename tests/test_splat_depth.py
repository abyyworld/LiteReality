"""Dense depth for the photoreal room: PromptDA's client and cache (models/splat/promptda), and
the dense map feeding the training set (models/splat/dataset) and the instant splats
(models/splat/rgbd) — always checked against the LiDAR it refines."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_quickroom import FLOOR_Y, _frame_looking_down  # noqa: E402
from test_splat import _read_colmap  # noqa: E402

from litereality_agent.models.splat import dataset, promptda, rgbd  # noqa: E402
from litereality_agent.models.splat.spz import quat_to_matrices  # noqa: E402

TOP_Z = FLOOR_Y + 0.75


def true_depth(width: int, height: int) -> np.ndarray:
    """The depth `_frame_looking_down` photographs, at any resolution: the table top 1.45 m below
    the camera, the floor 2.2 m (its own geometry: a 320 x 240 photo, fx = fy = 200)."""
    sx, sy = width / 320, height / 240
    us, vs = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    dx, dy = (us - 160 * sx) / (200 * sx), -(vs - 120 * sy) / (200 * sy)
    x, y = 2.5 + dx * 1.45, 2.0 + dy * 1.45
    on_top = (np.abs(x - 2.5) < 0.6) & (np.abs(y - 2.0) < 0.4)
    return np.where(on_top, 1.45, 2.2).astype(np.float32)


@pytest.fixture
def scan(tmp_path):
    folder = tmp_path / "scan"
    folder.mkdir()
    _frame_looking_down(folder, index=0)
    _frame_looking_down(folder, index=1)
    return folder


def _on_surfaces(z: np.ndarray, offset: float = 0.0, tol: float = 0.012) -> np.ndarray:
    return (np.abs(z - (FLOOR_Y - offset)) < tol) | (np.abs(z - (TOP_Z - offset)) < tol)


# --- the cache -----------------------------------------------------------------------------------


def test_dense_depth_is_cached_beside_the_scan_to_the_millimetre(scan):
    frame = rgbd.load_frames(scan)[0]
    depth = true_depth(160, 120) + np.float32(0.0004)
    depth[:3] = 0.0                                          # nothing measured there
    path = promptda.save_dense(frame, depth)
    assert path == scan / "depth_dense" / "frame_00000.png"
    with Image.open(path) as im:
        assert np.asarray(im).dtype == np.uint16             # millimetres, as the scan's own
    back = promptda.load_dense(frame, width=160)
    assert back.dtype == np.float32 and back.shape == (120, 160)
    assert np.abs(back - depth).max() <= 0.0005 and np.all(back[:3] == 0)
    assert promptda.load_dense(frame) is not None
    assert promptda.load_dense(frame, width=960) is None     # made for another training size
    assert promptda.load_dense(rgbd.load_frames(scan)[1]) is None
    path.write_bytes(b"not a png")
    assert promptda.load_dense(frame) is None


def test_a_cache_made_from_another_capture_is_a_miss(scan):
    """A scan exported again into the same folder: frame_00000's LiDAR is new, so the depth
    cached from the old one must not be read back (nor counted as cached)."""
    frame = rgbd.load_frames(scan)[0]
    promptda.save_dense(frame, true_depth(160, 120))
    assert promptda.is_cached(frame, 160) and promptda.load_dense(frame, 160) is not None
    assert not promptda.is_cached(frame, 960)                # another training size
    lidar = np.asarray(Image.open(frame.depth)).copy()
    lidar[0, 0] += 1                                         # the new capture's LiDAR
    Image.fromarray(lidar).save(frame.depth)
    assert promptda.load_dense(frame, 160) is None and not promptda.is_cached(frame, 160)
    # A map cached before keys existed (no text chunk) is a miss too: refined again, once.
    promptda.dense_path(frame).write_bytes(promptda.encode_depth(true_depth(160, 120)))
    assert promptda.load_dense(frame, 160) is None and not promptda.is_cached(frame, 160)


def test_the_photo_goes_up_with_its_captured_size(scan):
    """The JPEG sent may be shrunk to multiples of 14 (1920 x 1080 -> 1008 x 560, another aspect);
    the item's "size" tells the function the aspect its reply must have."""
    frame = rgbd.load_frames(scan)[0]
    assert promptda.request_item(frame)["size"] == list(frame.size)
    depth_app = _depth_app()
    item = {"name": "f", "image": _jpeg(1008, 560), "depth": _lidar_png(), "size": [1920, 1080]}
    rgb, _lidar, (w, h) = depth_app._decode(item)
    assert rgb.shape == (560, 1008, 3) and (w, h) == (1920, 1080)
    assert int(round(960 * h / w)) == 540                    # what the client expects back
    del item["size"]
    assert depth_app._decode(item)[2] == (1008, 560)         # without it: the JPEG's own size


def _jpeg(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 150)).save(buf, format="JPEG")
    return buf.getvalue()


def _lidar_png() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((192, 256), 2000, np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


def _depth_app():
    """deploy/modal/depth/app.py with a stand-in `modal` (only its decorators run at import)."""
    import importlib.util
    import types

    class _Image:
        def __getattr__(self, _name):
            return lambda *a, **k: self

    fake = types.SimpleNamespace(
        Image=types.SimpleNamespace(debian_slim=lambda **k: _Image()),
        App=lambda *a, **k: types.SimpleNamespace(function=lambda **k: (lambda f: f)))
    saved = sys.modules.get("modal")
    sys.modules["modal"] = fake
    try:
        path = Path(__file__).resolve().parents[1] / "deploy" / "modal" / "depth" / "app.py"
        spec = importlib.util.spec_from_file_location("litereality_depth_app_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = saved
    return module


def test_photos_go_up_at_promptdas_own_input_size():
    assert promptda.model_size(1920, 1440) == (1008, 756)    # README: long side 1008, x14
    assert promptda.model_size(320, 240) == (308, 238)       # small: only rounded down to x14
    assert all(s % 14 == 0 for s in promptda.model_size(1440, 1920))


def test_a_photo_goes_up_shrunk_only_when_that_is_fewer_bytes(scan):
    frame = rgbd.load_frames(scan)[0]
    noisy = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    Image.fromarray(noisy).save(frame.image, quality=100)    # a high-quality capture: shrink
    item = promptda.request_item(frame)
    with Image.open(io.BytesIO(item["image"])) as im:
        assert im.size == (308, 238)
    assert len(item["image"]) < frame.image.stat().st_size
    Image.fromarray(noisy).save(frame.image, quality=20)     # already small: sent as it is
    assert promptda.request_item(frame)["image"] == frame.image.read_bytes()
    assert promptda.request_item(frame, upload_side=None)["image"] == frame.image.read_bytes()
    assert item["depth"] == frame.depth.read_bytes() and item["name"] == "frame_00000"


# --- the client ----------------------------------------------------------------------------------


class FakeDepth:
    """The deployed `refine` as far as the client can tell: `remote(frames, options)`."""

    def __init__(self, reply=None, fail: Exception | None = None):
        self.reply, self.fail, self.calls = reply, fail, []

    def remote(self, frames, options):
        self.calls.append((frames, options))
        if self.fail is not None:
            raise self.fail
        if self.reply is not None:
            return self.reply(frames, options)
        out = []
        for item in frames:
            width = options["width"]
            depth = true_depth(width, round(width * 3 / 4))
            buf = io.BytesIO()
            Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(buf, format="PNG")
            out.append({"name": item["name"], "depth": buf.getvalue(), "ms": 12.5})
        return out


def test_refine_sends_batches_caches_and_then_reads_the_cache(scan, tmp_path):
    for i in range(2, 5):
        _frame_looking_down(scan, index=i)
    frames = rgbd.load_frames(scan)
    fake = FakeDepth()
    logged = []
    refiner = promptda.ModalDepthRefiner(function=fake, batch=2, log=logged.append)
    got = refiner.refine(frames, width=160)
    assert sorted(got) == [0, 1, 2, 3, 4]
    assert np.abs(got[3] - true_depth(160, 120)).max() < 1e-3
    assert sorted(len(f) for f, _ in fake.calls) == [1, 2, 2]
    assert all(opts == {"width": 160} for _, opts in fake.calls)
    item = fake.calls[0][0][0]
    assert item["name"].startswith("frame_") and item["conf"] is not None
    with Image.open(io.BytesIO(item["image"])) as im:
        assert im.size in ((308, 238), (320, 240))           # whichever JPEG is smaller
    assert (scan / "depth_dense" / "frame_00004.png").is_file()
    assert refiner.last_stats["refined"] == 5 and refiner.last_stats["gpu_ms_median"] == 12.5
    assert any("5/5" in line for line in logged)

    # Second run: everything from the cache, Modal never called (even a broken one).
    broken = FakeDepth(fail=RuntimeError("must not be called"))
    again = promptda.ModalDepthRefiner(function=broken).refine(frames, width=160)
    assert sorted(again) == [0, 1, 2, 3, 4] and broken.calls == []


@pytest.mark.parametrize("fake", [
    FakeDepth(fail=RuntimeError("app litereality-depth not found")),
    FakeDepth(reply=lambda frames, options: "not a list"),
    FakeDepth(reply=lambda frames, options: [{"name": f["name"], "depth": b"junk"}
                                             for f in frames]),
    FakeDepth(reply=lambda frames, options: [{"name": "frame_99999", "depth": b""}]),
], ids=["call-fails", "not-a-list", "bad-png", "unknown-frame"])
def test_refine_never_raises_and_returns_nothing_it_did_not_get(scan, fake):
    logged = []
    got = promptda.ModalDepthRefiner(function=fake, log=logged.append).refine(
        rgbd.load_frames(scan), width=160)
    assert got == {} and logged
    assert not (scan / "depth_dense").exists() or not any((scan / "depth_dense").iterdir())


def test_refine_without_modal_credentials_logs_and_returns_nothing(scan):
    logged = []
    got = promptda.ModalDepthRefiner(log=logged.append).refine(rgbd.load_frames(scan))
    assert got == {} and "credentials" in logged[-1]


def test_a_reply_of_the_wrong_size_is_dropped(scan):
    def small(frames, options):
        buf = io.BytesIO()
        Image.fromarray(np.full((30, 40), 2000, np.uint16)).save(buf, format="PNG")
        return [{"name": f["name"], "depth": buf.getvalue(), "ms": 1.0} for f in frames]

    got = promptda.ModalDepthRefiner(function=FakeDepth(reply=small)).refine(
        rgbd.load_frames(scan), width=160)
    assert got == {}


# --- trusting the dense map ----------------------------------------------------------------------


def test_dense_depth_refines_the_lidar_but_never_overrules_a_confident_sample():
    lidar = np.full((12, 16), 2.0, np.float32)
    conf = np.full((12, 16), 2, np.uint8)
    conf[:, 8:] = 0                                          # the right half: low confidence
    dense = np.full((48, 64), 2.0, np.float32)
    dense[:24, :] += 0.04                                    # top: off by 4 cm, inside 5 cm
    dense[24:, :] += 0.3                                     # bottom: off by 30 cm
    ok = rgbd.dense_mask(dense, lidar, conf)
    # Silhouette filter only bites at the 4 cm step between the halves; look away from it.
    assert ok[2:20, 2:30].all()                              # agrees with confident LiDAR
    assert not ok[28:46, 2:30].any()                         # disagrees: dropped
    # Unsure LiDAR with no confident sample beside it: the dense map stands alone.
    assert ok[28:46, 38:62].all()
    assert not rgbd.dense_mask(np.zeros_like(dense), lidar, conf).any()


def test_on_a_lidar_silhouette_dense_depth_may_pick_either_side_but_nothing_between():
    # A table edge: LiDAR column 7 straddles it (a smeared 1.8 m), 0..6 the top, 8.. the floor.
    lidar = np.full((12, 16), 2.2, np.float32)
    lidar[:, :7] = 1.45
    lidar[:, 7] = 1.8
    W = 64                                                   # 4 dense columns per LiDAR column
    for value, kept in ((1.45, True), (2.2, True), (1.8, True), (1.65, False), (2.0, False)):
        dense = np.where(np.arange(W) < 28, 1.45, 2.2).astype(np.float32)[None].repeat(48, 0)
        dense[:, 29:31] = value                              # inside LiDAR column 7
        ok = rgbd.dense_mask(dense, lidar, None, radius=0)
        assert ok[10:40, 29:31].all() == kept, value


# --- the training set ----------------------------------------------------------------------------


def test_dense_points_keep_every_photo_at_exactly_its_count_and_on_the_surface(scan, tmp_path):
    frames = rgbd.load_frames(scan)
    dense = {0: true_depth(320, 240)}                        # another size than training: resampled
    ds = dataset.build(frames, width=160, points_per_frame=2000, dense=dense, depth_maps=True)
    cams, images, points = _read_colmap(ds.write(tmp_path / "ds"))
    assert ds.images == 2 and ds.dense_images == 1
    assert [len(im[5]) for im in images] == [2000, 2000] and len(points) == ds.points == 4000
    assert np.all(_on_surfaces(points["xyz"][:, 2]))
    # Frame 0's points sit on the training image's pixel centres (dense), frame 1's on the
    # LiDAR's (every 2.5 training pixels).
    first = images[0][5]
    assert np.allclose(first["x"] % 1, 0.5) and np.allclose(first["y"] % 1, 0.5)
    assert len(np.unique(first["x"])) > 64                   # more columns than the LiDAR has
    assert sorted(n for n in ds.files if n.startswith("depths/")) == ["depths/frame_00000.png"]
    with Image.open(io.BytesIO(ds.files["depths/frame_00000.png"])) as im:
        mm = np.asarray(im)
    # Half the training size (the tar's biggest part otherwise); the trainer resizes it back.
    assert mm.dtype == np.uint16 and mm.shape == (60, 80)
    assert set(np.unique(mm)) <= {0, 1450, 2200} and (mm > 0).mean() > 0.85

    plain = dataset.build(frames, width=160, points_per_frame=2000, dense=dense)
    assert not any(n.startswith("depths/") for n in plain.files)


def test_held_out_photos_are_listed_but_never_seed_the_room(scan, tmp_path):
    """gsplat scores every `test_every`-th photo by name: it must still be in the model (the
    trainer evaluates it) but give no starting points, colours or depth map, or its own pixels
    would seed the room it is scored against."""
    frames = rgbd.load_frames(scan)
    dense = {f.index: true_depth(320, 240) for f in frames}
    ds = dataset.build(frames, width=160, points_per_frame=2000, dense=dense, depth_maps=True,
                       test_every=8)
    cams, images, points = _read_colmap(ds.write(tmp_path / "ds"))
    assert ds.images == 2 and ds.held_out == 1
    by_name = {im[4]: im for im in images}
    assert len(by_name["frame_00000.jpg"][5]) == 0          # first by name: held out
    assert len(by_name["frame_00001.jpg"][5]) == 2000
    assert len(points) == ds.points == 2000
    assert set(points["img"]) == {by_name["frame_00001.jpg"][0]}
    assert "images/frame_00000.jpg" in ds.files
    assert sorted(n for n in ds.files if n.startswith("depths/")) == ["depths/frame_00001.png"]
    # Without a hold-out every photo seeds it, as before.
    assert dataset.build(frames, width=160, points_per_frame=2000, dense=dense).points == 4000


def test_dense_samples_that_disagree_with_the_lidar_never_become_points(scan, tmp_path):
    frames = rgbd.load_frames(scan)[:1]
    dense = true_depth(160, 120)
    dense[60:, :] += 0.4                                     # the lower half 40 cm too deep
    ds = dataset.build(frames, width=160, points_per_frame=1500, dense={0: dense},
                       depth_maps=True)
    _cams, images, points = _read_colmap(ds.write(tmp_path / "ds"))
    assert len(images[0][5]) == 1500 and ds.dense_images == 1
    assert np.all(_on_surfaces(points["xyz"][:, 2]))         # none 40 cm under the floor
    assert images[0][5]["y"].max() < 60                      # nothing from the lower half
    with Image.open(io.BytesIO(ds.files["depths/frame_00000.png"])) as im:
        assert not np.asarray(im)[62:].any()                 # nor in the depth map


def test_a_dense_map_that_disagrees_everywhere_falls_back_to_the_lidar(scan, tmp_path):
    frames = rgbd.load_frames(scan)[:1]
    ds = dataset.build(frames, width=160, points_per_frame=300,
                       dense={0: true_depth(160, 120) + 0.5}, depth_maps=True)
    _cams, _images, points = _read_colmap(ds.write(tmp_path / "ds"))
    assert ds.dense_images == 0 and len(points) == 300
    assert np.all(_on_surfaces(points["xyz"][:, 2]))
    assert not any(n.startswith("depths/") for n in ds.files)


# --- the instant splats --------------------------------------------------------------------------


def test_instant_splats_come_from_the_dense_map_when_given(scan):
    frames = rgbd.load_frames(scan)[:1]
    # 3 cm deeper than the LiDAR everywhere: inside the agreement tolerance, so the discs must sit
    # 3 cm under the surfaces the LiDAR saw — proof they came from the dense map.
    s = rgbd.instant_splats(frames, voxel=0.01, dense={0: true_depth(640, 480) + 0.03})
    z = s.positions[:, 2]
    assert len(s) > 500 and _on_surfaces(z, offset=0.03).mean() > 0.97
    top = np.abs(z - (TOP_Z - 0.03)) < 0.012
    assert s.colors[top][:, 0].mean() > 0.6 and s.colors[top][:, 1].mean() < 0.3
    normals = quat_to_matrices(s.rotations)[:, :, 2]
    assert np.abs(normals[:, 2]).mean() > 0.98
    assert np.all(s.scales[:, 2] < s.scales[:, 0])


def test_instant_splats_ignore_a_dense_map_the_lidar_contradicts(scan):
    frames = rgbd.load_frames(scan)[:1]
    lidar_only = rgbd.instant_splats(frames, voxel=0.01)
    s = rgbd.instant_splats(frames, voxel=0.01, dense={0: true_depth(640, 480) + 0.5})
    assert len(s) == len(lidar_only)
    assert _on_surfaces(s.positions[:, 2]).mean() > 0.97
