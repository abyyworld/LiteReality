"""The splat trainer's dense-depth options (deploy/modal/splat/): the options it accepts, the flags
they become, the patch that teaches gsplat 1.5.3's trainer the losses, and the losses' maths.

No GPU, no Modal: the loss functions are written once for numpy and torch, so the numpy runs here
check the code the H100 runs. The patch is applied to the real v1.5.3 `simple_trainer.py`, fetched
by commit (or from $LR_GSPLAT_TRAINER), and the test skips when neither is reachable.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib.util
import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPLAT = Path(__file__).resolve().parents[1] / "deploy" / "modal" / "splat"
GSPLAT_COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"   # tag v1.5.3 (gh api .../git/ref/tags)
TRAINER_URL = (f"https://raw.githubusercontent.com/nerfstudio-project/gsplat/{GSPLAT_COMMIT}"
               "/examples/simple_trainer.py")
TRAINER_SHA256 = "79319e1cd7404e4d1ba0c425634235c39e6054f0643b904a01feea6179462c05"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


app = _load("litereality_splat_app", SPLAT / "app.py")
patch_trainer = _load("litereality_patch_trainer", SPLAT / "patch_trainer.py")
dense = _load("litereality_dense", SPLAT / "litereality_dense.py")


@pytest.fixture(scope="session")
def upstream_trainer(tmp_path_factory) -> str:
    local = os.environ.get("LR_GSPLAT_TRAINER")
    if local:
        text = Path(local).read_text()
    else:
        try:
            with urllib.request.urlopen(TRAINER_URL, timeout=15) as response:
                text = response.read().decode()
        except OSError as exc:
            pytest.skip(f"gsplat v1.5.3 simple_trainer.py not reachable ({exc}); "
                        "set LR_GSPLAT_TRAINER to a copy")
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert digest == TRAINER_SHA256, f"not gsplat v1.5.3's trainer (sha256 {digest})"
    return text


# ── options ──────────────────────────────────────────────────────────────────────────────────────


def test_dense_depth_is_off_by_default_and_the_command_unchanged():
    opt = app._options({})
    assert opt["dense_depth"] is False and opt["normal_lambda"] == 0.0
    assert not app._wants_dense(opt)
    cmd = app._command(Path("/d"), Path("/r"), opt)
    assert not {"--dense-depth", "--dense-depth-lambda", "--normal-lambda"} & set(cmd)
    assert "--depth-loss" in cmd    # the sparse LiDAR loss stays on


def test_dense_options_become_the_patched_trainers_flags():
    opt = app._options({"dense_depth": True, "dense_depth_lambda": 0.2, "normal_lambda": 0.05})
    cmd = app._command(Path("/d"), Path("/r"), opt)
    i = cmd.index("--dense-depth-lambda")
    assert cmd[i - 1] == "--dense-depth" and cmd[i + 1] == "0.2"
    assert cmd[cmd.index("--normal-lambda") + 1] == "0.05"
    assert "--depth-loss" in cmd
    # The normal loss alone still needs the depth maps, without the L1.
    alone = app._options({"normal_lambda": 0.1})
    assert app._wants_dense(alone)
    assert "--dense-depth" not in app._command(Path("/d"), Path("/r"), alone)


@pytest.mark.parametrize("given", [
    {"dense_depth": "yes"}, {"dense_depth": 1}, {"dense_depth_lambda": -0.1},
    {"normal_lambda": "0.1"}, {"normal_lambda": True}, {"normal_lambda": float("nan")},
    {"dense_lambda": 0.1},
])
def test_bad_dense_options_are_refused(given):
    with pytest.raises(ValueError):
        app._options(given)


def _tar(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_dense_depth_without_depth_maps_is_an_error_before_any_training(monkeypatch):
    launched = []
    monkeypatch.setattr(app.subprocess, "Popen", lambda *a, **k: launched.append(a))
    train = app.train.get_raw_f()
    events = list(train(_tar({"images/frame_00000.jpg": b"x", "sparse/0/cameras.bin": b""}),
                        {"dense_depth": True}))
    assert [e["kind"] for e in events] == ["error"]
    assert "depths/" in events[0]["error"] and not launched
    events = list(train(b"x", {"normal_lambda": "a lot"}))
    assert events[0]["kind"] == "error" and "normal_lambda" in events[0]["error"]


def test_the_image_carries_the_patch_and_its_helpers():
    assert app.PATCH_LOCAL.is_file() and app.DENSE_LOCAL.is_file()
    assert app.DENSE_LOCAL.name == "litereality_dense.py"   # the name the patch imports


# ── the patch ────────────────────────────────────────────────────────────────────────────────────


def test_the_patch_applies_to_gsplat_1_5_3_and_adds_only_the_dense_losses(upstream_trainer):
    patched = patch_trainer.patch(upstream_trainer)
    compile(patched, "simple_trainer.py", "exec")
    for field in ("dense_depth: bool = False", "dense_depth_lambda: float = 0.1",
                  "normal_lambda: float = 0.0"):
        assert field in patched
    assert "from litereality_dense import" in patched
    assert "self.trainset = DenseDepthData(self.trainset, cfg.data_dir)" in patched
    # Nothing upstream is lost: every original line is still there, in order, except the one
    # render_mode line, which now also renders depth for the dense losses.
    diff = difflib.unified_diff(upstream_trainer.splitlines(), patched.splitlines(), n=0,
                                lineterm="")
    removed = [line[1:] for line in diff if line.startswith("-") and not line.startswith("---")]
    assert removed == ['                render_mode="RGB+ED" if cfg.depth_loss else "RGB",']
    # The validation set and the eval are untouched: the held-out numbers stay gsplat's.
    assert 'self.valset = Dataset(self.parser, split="val")' in patched
    with pytest.raises(ValueError, match="already patched"):
        patch_trainer.patch(patched)


def test_the_patch_fails_loudly_on_a_trainer_it_does_not_know(upstream_trainer):
    changed = upstream_trainer.replace("    depth_lambda: float = 1e-2\n",
                                       "    depth_lambda: float = 2e-2\n")
    with pytest.raises(ValueError, match="anchor found 0 times"):
        patch_trainer.patch(changed)


def test_dense_loss_reads_the_rendered_depth_before_the_sparse_loss_overwrites_it(
        upstream_trainer):
    patched = patch_trainer.patch(upstream_trainer)
    dense_at = patched.index("pred_depth = depths[..., 0]")
    assert patched.index("colors, depths = renders[..., 0:3], renders[..., 3:4]") < dense_at
    assert dense_at < patched.index("depths = F.grid_sample(")


# ── the maths ────────────────────────────────────────────────────────────────────────────────────

K = np.array([[500.0, 0, 160.0], [0, 500.0, 120.0], [0, 0, 1]], np.float32)
H, W = 240, 320


def plane_depth(normal, d: float, K=K) -> np.ndarray:
    """z-depth of the plane n . X = d seen by K (OpenCV camera), 1 x H x W."""
    v, u = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    return (d / (rays @ np.asarray(normal, float)))[None].astype(np.float64)


def test_normals_of_a_plane_are_its_normal_facing_the_camera():
    n = np.array([0.2, -0.9, 0.4])
    n /= np.linalg.norm(n)
    depth = plane_depth(n, 1.5)          # a floor-ish plane, seen from above
    (nx, ny, nz), valid = dense.depth_normals(depth, K[None].astype(np.float64))
    assert valid.all() and nx.shape == (1, H - 2, W - 2)
    got = np.stack([nx, ny, nz], -1).reshape(-1, 3)
    facing = -n if n[2] > 0 else n       # toward the camera: z < 0
    assert np.abs(got - facing).max() < 1e-6


def test_normal_loss_is_zero_on_agreement_and_one_minus_cosine_on_a_tilt():
    Ks = K[None].astype(np.float64)
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([np.sin(0.3), 0.0, np.cos(0.3)])
    assert dense.depth_normal_loss(plane_depth(a, 2.0), plane_depth(a, 2.0), Ks) < 1e-9
    loss = dense.depth_normal_loss(plane_depth(b, 2.0), plane_depth(a, 2.0), Ks)
    assert loss == pytest.approx(1 - np.cos(0.3), abs=1e-6)


def test_the_sensors_edges_and_holes_get_no_normal_but_the_renders_tears_do():
    Ks = K[None].astype(np.float64)
    gt = plane_depth([0, 0, 1], 2.0)
    gt[:, :, W // 2:] = 3.0                   # a step: a box edge the sensor saw
    gt[:, 10:20, 10:20] = 0                   # a hole: nothing measured
    _, ok = dense.depth_normals(gt, Ks, max_jump=dense.MAX_NORMAL_JUMP)
    assert not ok[0, :, W // 2 - 2: W // 2].any()     # either side of the step
    # Normals are for interior pixels: index i is pixel i + 1. The hole and the pixels beside it.
    assert not ok[0, 9:19, 9:19].any() and not ok[0, 8, 12] and not ok[0, 12, 19]
    assert ok[0, 100, 50] and ok[0, 100, 250] and ok[0, 7, 12]
    # A floater: the render tears a wall the sensor saw whole. Penalised, not excused.
    wall = plane_depth([0, 0, 1], 2.0)
    torn = wall.copy()
    torn[:, 100:110, 100:110] = 0.8
    # Its rim (76 pixels, in and out) has normals ~90 degrees off; its face is parallel to the
    # wall, so the normal term only sees the rim (the L1 term sees the face).
    rim = dense.depth_normal_loss(torn, wall, Ks) * (H - 2) * (W - 2)
    assert 65 < rim < 80
    assert dense.depth_normal_loss(wall, wall, Ks) < 1e-9


def test_dense_l1_is_the_mean_metre_error_where_both_have_depth():
    pred = np.array([[[1.0, 2.0], [0.0, 4.0]]])
    gt = np.array([[[1.5, 2.0], [3.0, 0.0]]])
    # (1,0): nothing rendered; (1,1): nothing measured. Left: |1-1.5| and |2-2|.
    assert dense.dense_depth_l1(pred, gt) == pytest.approx(0.25)
    assert dense.dense_depth_l1(pred, np.zeros_like(gt)) == 0


def test_the_losses_run_in_torch_with_finite_gradients_through_holes():
    torch = pytest.importorskip("torch")
    gt = plane_depth([0.1, -0.8, 0.5], 1.2)
    gt[:, :40] = 0                                           # sky / unmeasured band
    pred = plane_depth([0.0, -0.8, 0.6], 1.3)
    pred[:, 100:140, 100:140] = 0                            # nothing rendered there
    Ks = K[None].astype(np.float64)
    want_l1 = dense.dense_depth_l1(pred, gt)
    want_n = dense.depth_normal_loss(pred, gt, Ks)
    p = torch.tensor(pred, dtype=torch.float32, requires_grad=True)
    g, k = torch.tensor(gt, dtype=torch.float32), torch.tensor(Ks, dtype=torch.float32)
    l1 = dense.dense_depth_l1(p, g)
    normal = dense.depth_normal_loss(p, g, k)
    (l1 + normal).backward()
    assert float(l1) == pytest.approx(want_l1, rel=1e-4)
    assert float(normal) == pytest.approx(want_n, rel=1e-3, abs=1e-5)
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


# ── the dataset ──────────────────────────────────────────────────────────────────────────────────


def _png16(path: Path, depth_mm: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), depth_mm.astype(np.uint16))


def test_depth_maps_are_found_by_image_name_and_read_in_metres(tmp_path):
    mm = np.full((6, 8), 1234, np.uint16)
    mm[0, 0] = 0
    _png16(tmp_path / "depths" / "frame_00008.jpg.png", mm)
    _png16(tmp_path / "depths" / "frame_00009.png", mm * 2)
    assert dense.depth_path(str(tmp_path), "frame_00008.jpg").endswith("frame_00008.jpg.png")
    assert dense.depth_path(str(tmp_path), "frame_00009.jpg").endswith("frame_00009.png")
    assert dense.depth_path(str(tmp_path), "frame_00010.jpg") is None
    m = dense.load_depth(dense.depth_path(str(tmp_path), "frame_00008.jpg"), (8, 6))
    assert m.dtype == np.float32 and m[0, 0] == 0 and m[1, 1] == pytest.approx(1.234)
    # Another size is resized without blending; a missing map is all "none".
    big = dense.load_depth(dense.depth_path(str(tmp_path), "frame_00009.jpg"), (16, 12))
    assert big.shape == (12, 16) and np.allclose(big[big > 0], 2.468)
    assert (big == 0).sum() == 4                     # the one empty pixel, 2 x 2, not smeared
    assert not dense.load_depth(None, (8, 6)).any()


def test_the_training_samples_carry_their_dense_depth(tmp_path):
    torch = pytest.importorskip("torch")
    _png16(tmp_path / "depths" / "frame_00001.jpg.png", np.full((6, 8), 2000))
    Kc = np.array([[5.0, 0, 4], [0, 5.0, 3], [0, 0, 1]])
    parser = SimpleNamespace(image_names=["frame_00000.jpg", "frame_00001.jpg"],
                             camera_ids=[1, 1], params_dict={1: np.empty(0)},
                             imsize_dict={1: (8, 6)}, Ks_dict={1: Kc})

    class Base:
        indices = np.array([1, 0])

        def __init__(self):
            self.parser = parser

        def __len__(self):
            return 2

        def __getitem__(self, item):
            return {"image": torch.zeros(6, 8, 3), "K": torch.tensor(Kc, dtype=torch.float32)}

    data = dense.DenseDepthData(Base(), str(tmp_path))
    assert len(data) == 2 and data.parser is parser
    assert data[0]["dense_depth"].shape == (6, 8) and float(data[0]["dense_depth"][2, 2]) == 2.0
    assert not data[1]["dense_depth"].any()          # frame_00000 has no map: colour only
