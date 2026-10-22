"""Full-body avatars on Modal, what can be checked without a GPU: the deploy file's settings and
request handling, its weights path (populated once on CPU, never written by a GPU container), and
models/avatar/lhm.py's numpy half — framing, view selection, export to the avatar file's fields,
skinning and the top-4 error — driven end to end by a numpy fake of LHM++."""

from __future__ import annotations

import ast
import base64
import importlib.util
import io
import os
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from litereality_agent.models.avatar import lhm
from litereality_agent.room_ops.roomcall import avatar_format

REPO = Path(__file__).resolve().parents[1]
APP_FILE = REPO / "deploy" / "modal" / "avatar" / "app.py"
LHM_FILE = REPO / "src" / "litereality_agent" / "models" / "avatar" / "lhm.py"


@pytest.fixture()
def app():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("avatar_deploy_under_test", APP_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── a numpy LHM++ ────────────────────────────────────────────────────────────────────────────────

# The stand-in skeleton's rest joints (contract §4.3), +X the person's left.
BODY_REST = [
    (0, .95, 0), (.09, .88, 0), (-.09, .88, 0), (0, 1.06, -.01), (.10, .50, .01), (-.10, .50, .01),
    (0, 1.19, -.01), (.11, .08, -.02), (-.11, .08, -.02), (0, 1.25, 0), (.12, .02, .10),
    (-.12, .02, .10), (0, 1.47, -.01), (.07, 1.39, 0), (-.07, 1.39, 0), (0, 1.56, .01),
    (.18, 1.42, -.01), (-.18, 1.42, -.01), (.44, 1.42, -.02), (-.44, 1.42, -.02),
    (.69, 1.42, -.01), (-.69, 1.42, -.01),
]


def skeleton() -> np.ndarray:
    """55 SMPL-X-shaped rest joints: the stand-in body, jaw and eyes on the head, fingers fanned
    out past each wrist along its own side."""
    rest = np.zeros((55, 3))
    rest[:22] = BODY_REST
    rest[22] = rest[15] + (0, -0.03, 0.05)
    rest[23] = rest[15] + (0.03, 0.06, 0.08)
    rest[24] = rest[15] + (-0.03, 0.06, 0.08)
    for j in range(25, 55):
        side = 1.0 if j < 40 else -1.0
        rest[j] = rest[avatar_format.SMPLX_PARENTS[j]] + (0.025 * side, 0.0, 0.0)
    return rest


def synthetic_body(n: int = 5000, seed: int = 0) -> lhm.RawBody:
    """~n Gaussians around the joints, each on its own joint (0.6) plus four others (0.4 between
    them, so the top-4 cut costs something), 10 % nearly transparent."""
    rng = np.random.default_rng(seed)
    rest = skeleton()
    own = rng.integers(0, 55, n)
    xyz = rest[own] + rng.normal(0.0, 0.03, (n, 3))
    weights = np.zeros((n, 55))
    weights[np.arange(n), own] = 0.6
    for i in range(n):
        others = rng.choice(np.delete(np.arange(55), own[i]), 4, replace=False)
        weights[i, others] = rng.dirichlet(np.ones(4)) * 0.4
    opacity = rng.uniform(0.05, 1.0, n)
    opacity[rng.random(n) < 0.1] = 0.01
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return lhm.RawBody(
        xyz=xyz, rot_wxyz=quats, scale=rng.uniform(0.005, 0.02, (n, 3)), opacity=opacity,
        rgb=rng.random((n, 3)), weights=weights, joints=rest,
        parents=list(avatar_format.SMPLX_PARENTS), names=list(avatar_format.SMPLX_JOINTS),
        betas=np.zeros(10),
    )


class FakeRunner:
    """`segment` finds the non-white pixels (a centred box in the test photos); `run` records the
    framed views and answers with a synthetic body."""

    def __init__(self, *, empty: bool = False, fail: Exception | None = None) -> None:
        self.empty, self.fail, self.views = empty, fail, None

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        if self.empty:
            return np.zeros(rgb.shape[:2], np.float32)
        return (np.abs(rgb.astype(int) - 255).sum(axis=2) > 60).astype(np.float32)

    def run(self, views: np.ndarray) -> lhm.RawBody:
        if self.fail is not None:
            raise self.fail
        self.views = views
        return synthetic_body()


def photo(fmt: str = "JPEG", size=(120, 200), box=(40, 30, 80, 170), blank: bool = False) -> bytes:
    """A white photo with a red person-shaped box (x0, y0, x1, y1) in it."""
    im = Image.new("RGB", size, "white")
    if not blank:
        im.paste((200, 30, 30), box)
    out = io.BytesIO()
    im.save(out, format=fmt)
    return out.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def request(**overrides) -> dict:
    req = {"images_b64": [b64(photo()), b64(photo("PNG"))], "mode": "photo",
           "id": "a0123456789abcdef"}
    req.update(overrides)
    return req


# ── the deploy file ──────────────────────────────────────────────────────────────────────────────


def _decorator_kwargs(function_name: str) -> dict[str, ast.AST]:
    tree = ast.parse(APP_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for deco in node.decorator_list:
                if isinstance(deco, ast.Call) and getattr(deco.func, "attr", "") == "function":
                    return {kw.arg: kw.value for kw in deco.keywords}
    raise AssertionError(f"no @app.function on {function_name}")


def test_the_app_is_its_own_small_gpu_app(app):
    assert app.APP_NAME == "litereality-avatar" and app.FUNCTION_NAME == "avatar"
    assert set(app.app.registered_functions) == {app.FUNCTION_NAME, app.POPULATE_FUNCTION}
    gpu = _decorator_kwargs("avatar")
    assert ast.unparse(gpu["gpu"]) == "GPU" and app.GPU == "L4"
    assert ast.unparse(gpu["max_containers"]) == "MAX_CONTAINERS" and app.MAX_CONTAINERS <= 2
    assert ast.unparse(gpu["scaledown_window"]) == "SCALEDOWN_SECONDS"
    assert ast.unparse(gpu["timeout"]) == "TIMEOUT_SECONDS" and ast.unparse(gpu["retries"]) == "0"
    cpu = _decorator_kwargs("populate")
    assert "gpu" not in cpu and ast.unparse(cpu["max_containers"]) == "1"
    assert app.MODES == lhm.MODES == ("turnaround", "photo") and app.MAX_IMAGES == 8
    # Never part of the combined `litereality` app (that one is deployed whole, and the room
    # server finds this one by its own name), so, like roomcall, not an importable package.
    assert not (APP_FILE.parent / "__init__.py").exists()
    assert "avatar" not in (APP_FILE.parent.parent / "app.py").read_text()


def test_every_source_is_pinned(app):
    for commit in (app.LHMPP_COMMIT, app.PYTORCH3D_COMMIT, app.DGR_COMMIT):
        assert re.fullmatch(r"[0-9a-f]{40}", commit)
    for wheel in (app.GSPLAT_WHEEL, app.FLASH_ATTN_WHEEL):
        assert re.search(r"#sha256=[0-9a-f]{64}$", wheel) and "cp310" in wheel
    names = set()
    for spec in app.LHMPP_REQUIREMENTS:
        name, _, version = spec.partition("==")
        assert version and re.fullmatch(r"[0-9][0-9a-z.]*", version), spec
        names.add(name.lower().replace("_", "-"))
    assert {"numpy", "huggingface-hub", "diffusers"} <= names
    dropped = {"dna", "gradio", "spaces", "rembg", "opencv-python", "torch", "gsplat"}
    assert dropped.isdisjoint(names)
    assert dict(s.split("==") for s in app.LHMPP_REQUIREMENTS)["numpy"] == "1.23.0"


def test_the_inference_module_stays_light_and_below_room_ops():
    tree = ast.parse(LHM_FILE.read_text())
    top = set()
    for node in tree.body:                         # module level only: torch is inside LHMRunner
        if isinstance(node, ast.Import):
            top.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert "torch" not in top
    text = LHM_FILE.read_text()
    assert "room_ops" not in "".join(
        line for line in text.splitlines() if line.startswith(("import", "from")))
    assert lhm.MAX_SPLATS == avatar_format.MAX_SPLATS


# ── requests ─────────────────────────────────────────────────────────────────────────────────────


def test_a_good_request_parses_and_unknown_keys_are_ignored(app):
    req = app.parse_request(request(something_newer=True))
    assert req["id"] == "a0123456789abcdef" and req["mode"] == "photo"
    assert len(req["images"]) == 2 and req["images"][0][:3] == b"\xff\xd8\xff"
    assert req["images"][1][:4] == b"\x89PNG"
    assert app.parse_request(request(mode="turnaround",
                                     images_b64=[b64(photo())] * 8))["mode"] == "turnaround"


def test_bad_requests_are_refused(app, monkeypatch):
    monkeypatch.setattr(app, "MAX_IMAGE_BYTES", 1000)
    jpeg = b64(photo())
    bad = [
        "not a dict", [], request(id=None), request(id=""), request(id="a b"),
        request(id="x" * 65), request(id=7), request(mode="video"), request(mode=None),
        request(images_b64=[]), request(images_b64=[jpeg] * 9), request(images_b64=jpeg),
        request(images_b64=["%%%"]), request(images_b64=[jpeg[:-2] + "\n="]),
        request(images_b64=[b64(b"\xff\xd8\xff" + bytes(1001))]),
        request(images_b64=[b64(b"\xff\xd8\xff" + bytes(5000))]),
        request(images_b64=[b64(b"GIF89a" + bytes(10))]), request(images_b64=[""]),
        request(images_b64=[12]),
    ]
    for req in bad:
        with pytest.raises(ValueError):
            app.parse_request(req)


# ── handle, end to end with a fake LHM++ ─────────────────────────────────────────────────────────


def test_handle_builds_a_packed_lhm_body(app):
    runner = FakeRunner()
    res = app.handle(request(), runner)
    assert res["id"] == "a0123456789abcdef" and "error" not in res and res["seconds"] >= 0
    body = avatar_format.unpack(base64.b64decode(res["avatar_b64"], validate=True))
    assert body.kind == "lhm" and 1000 < body.count <= 5000 and len(body.joint_names) == 55
    assert body.joint_names == list(avatar_format.SMPLX_JOINTS)
    meta = body.meta
    assert set(meta["skin_err_mm"]) == {"median", "p99", "max"} and meta["model"] == lhm.MODEL_NAME
    assert 0 < meta["skin_err_mm"]["median"] < 50 and 0.5 < meta["top4_coverage"]["min"] <= 1
    assert meta == res["stats"] and meta["views"] == meta["views_used"] == 2
    assert meta["splats"] == body.count and meta["pruned"] == 5000 - body.count
    assert runner.views.shape == (2, lhm.CANVAS_H, lhm.CANVAS_W, 3)
    assert body.head_mask().any() and np.allclose(body.rest, skeleton(), atol=1e-5)


def test_a_body_in_smplx_native_coordinates_packs(app):
    """SMPL-X's zero pose is not grounded: its soles sit near y = -1.3. The file keeps that frame
    and says where the floor is."""
    class Lowered(FakeRunner):
        def run(self, views):
            body = super().run(views)
            body.xyz = body.xyz - (0.0, 1.3, 0.0)
            body.joints = body.joints - (0.0, 1.3, 0.0)
            return body

    res = app.handle(request(), Lowered())
    body = avatar_format.unpack(base64.b64decode(res["avatar_b64"]))
    assert -1.45 < body.floor_y < -1.25 and np.allclose(body.rest[0], (0, 0.95 - 1.3, 0))


def test_handle_explains_instead_of_raising(app):
    no_person = app.handle(request(), FakeRunner(empty=True))
    assert no_person == {"error": "no person found in the images", "id": "a0123456789abcdef"}
    crashed = app.handle(request(), FakeRunner(fail=RuntimeError("CUDA out of memory")))
    assert crashed == {"error": "internal: RuntimeError", "id": "a0123456789abcdef"}
    refused = app.handle(request(mode="video"), FakeRunner())
    assert refused["error"].startswith("bad request: mode") and refused["id"] == "a0123456789abcdef"
    junk = app.handle(["junk"], FakeRunner())
    assert junk == {"error": "bad request: the request must be a dict", "id": None}
    garbage = app.handle(request(images_b64=[b64(b"\xff\xd8\xff" + bytes(64))]), FakeRunner())
    assert garbage["error"] == "an image could not be decoded"


def test_only_photo_and_file_errors_reach_the_person(app, capsys):
    """A ValueError from inside the model or a library is a fault like any other: logged with its
    traceback and answered "internal", never shown to the person as if their photos were bad."""
    inner = FakeRunner(fail=ValueError("pred_shape dim 16 exceeds smplx betas dim 10"))
    assert app.handle(request(), inner) == {"error": "internal: ValueError",
                                            "id": "a0123456789abcdef"}
    assert "pred_shape dim 16" in capsys.readouterr().err               # in the Modal logs

    class Tiny(FakeRunner):                                             # a body the file refuses
        def run(self, views):
            body = super().run(views)
            body.xyz, body.joints = body.xyz * 0.2, body.joints * 0.2
            return body

    refused = app.handle(request(), Tiny())
    assert refused["error"].startswith("the body is 0.") and "tall" in refused["error"]
    assert issubclass(lhm.PhotoError, ValueError)


def test_the_modal_function_loads_the_model_only_for_a_good_request(app, monkeypatch):
    loads = []

    def loader():
        loads.append(1)
        return FakeRunner()

    monkeypatch.setattr(app, "_runner", loader)
    raw = app.avatar.get_raw_f()
    assert "bad request" in raw(request(id="no spaces allowed"))["error"] and loads == []
    assert "avatar_b64" in raw(request()) and loads == [1]

    def broken():
        raise FileNotFoundError("pretrained_models/voxel_grid/cano_1_volume.npz")

    monkeypatch.setattr(app, "_runner", broken)
    assert raw(request())["error"] == "internal: FileNotFoundError"


@pytest.fixture()
def gpu(app, monkeypatch):
    """The real `_runner` with the container setup and LHMRunner faked: `loads` counts model
    builds, `retired` the containers told to stop taking inputs."""
    import modal.experimental

    state = SimpleNamespace(loads=[], retired=[], load_error=None, runner=FakeRunner())

    def load(repo, model):
        state.loads.append(model)
        if state.load_error is not None:
            raise state.load_error
        return state.runner

    monkeypatch.setattr(app, "_prepare_container", lambda: None)
    monkeypatch.setattr(app, "_model_dir", lambda root: Path("/models/m"))
    monkeypatch.setattr(app.lhm, "LHMRunner", load)
    monkeypatch.setattr(modal.experimental, "stop_fetching_inputs",
                        lambda: state.retired.append(1))
    state.call = app.avatar.get_raw_f()
    return state


def test_a_failed_load_fails_fast_and_is_retried_later(app, gpu, monkeypatch):
    gpu.load_error = FileNotFoundError("pretrained_models/human_model_files/flame/FLAME_NEUTRAL.pkl")
    replies = [gpu.call(request()) for _ in range(3)]
    assert replies == [{"error": "internal: ModelLoadError", "id": "a0123456789abcdef"}] * 3
    assert len(gpu.loads) == 1                                  # not rebuilt per request
    assert gpu.retired == [] and app._RUNNER is None            # stays, answering fast
    assert "FLAME_NEUTRAL.pkl" in app._LOAD_FAILURE[1]
    started, _ = app._LOAD_FAILURE                              # the window passes
    monkeypatch.setattr(app, "_LOAD_FAILURE",
                        (started - app.LOAD_RETRY_SECONDS - 1, app._LOAD_FAILURE[1]))
    gpu.load_error = None
    assert "avatar_b64" in gpu.call(request()) and len(gpu.loads) == 2
    assert app._LOAD_FAILURE is None and app._RUNNER is gpu.runner
    assert 60 <= app.LOAD_RETRY_SECONDS <= 600


def test_a_fault_with_the_model_loaded_retires_the_container(app, gpu):
    assert "avatar_b64" in gpu.call(request()) and app._RUNNER is gpu.runner
    # Photo problems and bad requests are not faults: the model stays loaded here.
    assert gpu.call(request(images_b64=[b64(photo(blank=True))]))["error"] == (
        "no person found in the images")
    assert "bad request" in gpu.call(request(mode="video"))["error"]
    assert gpu.retired == [] and app._RUNNER is gpu.runner and len(gpu.loads) == 1
    # A CUDA fault may leave the context unusable: drop the model, take no more inputs here.
    gpu.runner.fail = RuntimeError("CUDA error: an illegal memory access was encountered")
    reply = gpu.call(request())
    assert reply == {"error": "internal: RuntimeError", "id": "a0123456789abcdef"}
    assert gpu.retired == [1] and app._RUNNER is None


def test_retiring_outside_a_container_still_answers(app, monkeypatch):
    """`modal.experimental.stop_fetching_inputs` raises outside a Modal container; the reply
    must still come back (and the model is dropped either way)."""
    monkeypatch.setattr(app, "_runner", lambda: FakeRunner(fail=RuntimeError("boom")))
    monkeypatch.setattr(app, "_RUNNER", FakeRunner())          # "loaded"
    reply = app.avatar.get_raw_f()(request())
    assert reply["error"] == "internal: RuntimeError" and app._RUNNER is None


# ── framing and view selection ───────────────────────────────────────────────────────────────────


def _person(h=400, w=300, box=(100, 300, 120, 180), color=(30, 160, 40)):
    rgb = np.full((h, w, 3), 90, np.uint8)            # a grey room behind the person
    mask = np.zeros((h, w), np.float32)
    y0, y1, x0, x1 = box
    rgb[y0:y1, x0:x1] = color
    mask[y0:y1, x0:x1] = 1.0
    return rgb, mask


def _extent(view, axis):
    person = np.abs(view - 1.0).sum(axis=2) > 0.2
    idx = np.flatnonzero(person.any(axis=axis))
    return int(idx[0]), int(idx[-1]) + 1


def test_frame_view_is_lhm_shaped_white_and_keeps_the_aspect():
    rgb, mask = _person()
    view = lhm.frame_view(rgb, mask)
    assert view.shape == (840, 504, 3) and view.dtype == np.float32
    assert view.min() >= 0.0 and view.max() <= 1.0
    assert np.all(view[:5] == 1.0) and np.all(view[:, :5] == 1.0)   # the grey room is gone
    y0, y1 = _extent(view, 1)
    x0, x1 = _extent(view, 0)
    assert abs((y1 - y0) / (x1 - x0) - 200 / 60) < 0.05                # aspect kept
    margin = 840 - (y1 - y0)                                            # 3 % pad, centred
    assert abs(y0 - (840 - y1)) <= 2 and abs(margin / (y1 - y0) - 0.06 / 1.0) < 0.02
    assert abs(x0 - (504 - x1)) <= 2


def test_a_person_cut_at_the_legs_is_bottom_aligned():
    rgb, mask = _person(box=(150, 400, 100, 200))                       # touches the bottom row
    view = lhm.frame_view(rgb, mask)
    y0, y1 = _extent(view, 1)
    assert y1 == 840 and y0 > 0                                         # no margin below the cut
    wide_rgb, wide_mask = _person(box=(300, 400, 20, 280))              # wide: fits the width
    y0, y1 = _extent(lhm.frame_view(wide_rgb, wide_mask), 1)
    assert y1 == 840 and y0 > 400


def test_a_cut_person_keeps_about_the_full_body_scale():
    """LHM++ trains on cut views at full-body scale with white above, so a person cut off by the
    photo spans at most CUT_FILL of the canvas, not all of it (it would read as a giant)."""
    assert 0.6 <= lhm.CUT_FILL <= 0.8                                  # LHM++'s 0.6-1.0 cuts
    rgb, mask = _person(box=(50, 400, 120, 180))                        # tall, cut at the bottom
    view = lhm.frame_view(rgb, mask)
    y0, y1 = _extent(view, 1)
    assert y1 == 840 and 0.9 * lhm.CUT_FILL * 840 < y1 - y0 <= lhm.CUT_FILL * 840 + 1
    assert np.all(view[:int(840 * (1 - lhm.CUT_FILL)) - 2] == 1.0)      # white above
    whole_rgb, whole_mask = _person(box=(25, 375, 120, 180))            # the same, not cut
    y0, y1 = _extent(lhm.frame_view(whole_rgb, whole_mask), 1)
    assert y1 - y0 > 0.9 * 840                                          # an uncut one still fills


def test_frame_view_composites_a_soft_mask_on_white():
    rgb, mask = _person(color=(0, 0, 0))
    mask[100:300, 120:150] = 0.5                                       # half the person: 50 % alpha
    view = lhm.frame_view(rgb, mask)
    y0, y1 = _extent(view, 1)
    row = view[(y0 + y1) // 2]
    assert row.min() < 0.05 and np.any(np.abs(row - 0.5) < 0.06)


def test_select_views_drops_empty_views_and_keeps_order():
    rgb, mask = _person()
    empty = np.zeros_like(mask)
    speck = np.zeros_like(mask)
    speck[:5, :5] = 1                                                   # 0.02 % of the image
    views, masks = [rgb] * 4, [empty, mask, speck, mask]
    assert lhm.select_views(views, masks, "turnaround") == [1, 3]
    assert lhm.select_views(views, masks, "photo") == [1, 3]
    with pytest.raises(lhm.PhotoError, match="no person"):
        lhm.select_views(views[:3:2], [empty, speck], "photo")
    with pytest.raises(ValueError) as wrong_mode:                       # a caller's bug, not
        lhm.select_views(views, masks, "video")                         # the photos' fault
    assert not isinstance(wrong_mode.value, lhm.PhotoError)
    assert lhm.person_box(mask) == (100, 300, 120, 180) and lhm.person_box(empty) is None


def test_decode_image_turns_and_shrinks_photos(monkeypatch):
    im = Image.new("RGB", (60, 20), "white")
    im.paste((255, 0, 0), (0, 0, 10, 20))                              # red on the left edge
    exif = Image.Exif()
    exif[0x0112] = 6                                                    # "rotate 90 CW to view"
    out = io.BytesIO()
    im.save(out, format="JPEG", exif=exif)
    upright = lhm.decode_image(out.getvalue())
    assert upright.shape == (60, 20, 3) and upright[:8].mean(axis=(0, 1))[0] > 200
    assert upright[:8].mean(axis=(0, 1))[1] < 80                       # red now on top
    big = io.BytesIO()
    Image.new("RGB", (3000, 1000), "white").save(big, format="PNG")
    assert lhm.decode_image(big.getvalue()).shape == (683, 2048, 3)
    cutout = io.BytesIO()
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(cutout, format="PNG")
    assert np.all(lhm.decode_image(cutout.getvalue()) == 255)          # transparent -> white
    with pytest.raises(lhm.PhotoError, match="could not be decoded"):
        lhm.decode_image(b"\xff\xd8\xff" + bytes(100))
    monkeypatch.setattr(lhm, "MAX_DECODE_PIXELS", 100)
    with pytest.raises(lhm.PhotoError, match="too many pixels"):
        lhm.decode_image(photo(size=(20, 20), box=(5, 5, 10, 10)))


def test_build_uses_only_the_views_with_someone_in_them():
    runner = FakeRunner()
    images = [photo(), photo(blank=True), photo("PNG")]
    fields, stats = lhm.build(runner, images, "turnaround")
    assert stats["views"] == 3 and stats["views_used"] == 2 and runner.views.shape[0] == 2
    assert set(stats["timing"]) == {"segment_s", "infer_s", "export_s"}
    assert 1.5 < stats["height_m"] < 1.9 and stats["splats"] == len(fields["positions"])
    with pytest.raises(ValueError, match="no person"):
        lhm.build(runner, [photo(blank=True)], "photo")


# ── the body ─────────────────────────────────────────────────────────────────────────────────────


def test_body_fields_prunes_caps_and_reorders():
    raw = synthetic_body(n=2000)
    raw.opacity[:] = np.linspace(0.0, 1.0, 2000)
    raw.opacity[100:110] = 0.5                                          # ties
    fields = lhm.body_fields(raw)
    assert np.all(raw.opacity[fields["kept"]] >= lhm.OPACITY_MIN)
    assert len(fields["kept"]) == np.count_nonzero(raw.opacity >= lhm.OPACITY_MIN)
    capped = lhm.body_fields(raw, max_splats=300)
    kept = capped["kept"]
    assert len(kept) == 300 and np.all(np.diff(kept) > 0)               # original order
    assert raw.opacity[kept].min() >= np.sort(raw.opacity)[-300]        # the most opaque
    assert np.allclose(capped["positions"], raw.xyz[kept].astype(np.float32))
    raw.opacity[:] = 0.5                                                # all tied: stable
    assert list(lhm.body_fields(raw, max_splats=5)["kept"]) == [0, 1, 2, 3, 4]


def test_body_fields_units_and_skin():
    raw = synthetic_body(n=1000)
    raw.rot_wxyz[:] = (0.8, 0.6, 0.0, 0.0)
    raw.scale[0] = (0.0, 1.0, 0.01)
    raw.opacity[:] = 1.0
    raw.weights[1] = 0.0                                                # no weight at all
    raw.weights[2, :] = 0.0
    raw.weights[2, [3, 9, 12]] = (-0.1, 0.5, 1.5)                       # a negative one
    fields = lhm.body_fields(raw)
    assert np.allclose(fields["rotations"], (0.6, 0.0, 0.0, 0.8))      # x, y, z, w
    assert np.allclose(fields["scales"][0], (1e-4, 0.5, 0.01))
    assert fields["colors"].dtype == np.uint8 and np.all(fields["colors"][:, 3] == 255)
    w, j = fields["skin_weights"], fields["skin_joints"]
    assert w.shape == j.shape == (1000, 4) and j.dtype == np.int64
    assert np.all(w >= 0) and np.allclose(w.sum(axis=1), 1.0)
    assert np.all(np.diff(w, axis=1) <= 1e-12)                          # heaviest first
    assert list(j[2, :2]) == [12, 9] and np.allclose(w[2, :2], (0.75, 0.25))
    assert np.allclose(w[1], (1, 0, 0, 0))
    assert fields["parents"] == list(avatar_format.SMPLX_PARENTS)
    assert fields["rest"].shape == (55, 3)


def test_floor_y_is_the_low_percentile_of_the_solid_splats():
    raw = synthetic_body(n=3000)
    raw.opacity[:] = 0.9
    raw.xyz[:10, 1] = -2.0                                              # faint splats below
    raw.opacity[:10] = 0.1
    fields = lhm.body_fields(raw)
    solid = raw.xyz[10:, 1]
    assert fields["floor_y"] == pytest.approx(float(np.percentile(solid, 0.2)), abs=1e-5)
    with pytest.raises(lhm.PhotoError, match="no visible"):
        raw.opacity[:] = 0.0
        lhm.body_fields(raw)
    raw.opacity[:] = 0.9
    raw.weights = raw.weights[:, :54]                                   # the runner's bug
    with pytest.raises(ValueError) as broken:
        lhm.body_fields(raw)
    assert not isinstance(broken.value, lhm.PhotoError)


def test_lbs_moves_points_with_their_bones():
    rest = skeleton()
    parents = list(avatar_format.SMPLX_PARENTS)
    point = rest[19] + (-0.1, 0.0, 0.0)                                 # on the right forearm
    on_elbow = np.zeros((1, 55))
    on_elbow[0, 19] = 1.0
    turned = lhm.lbs(point[None], on_elbow, rest, parents, {19: (0, 0, np.pi / 2)})[0]
    # Rz(90°) about the elbow: (-0.1, 0, 0) from the elbow becomes (0, -0.1, 0).
    assert np.allclose(turned, rest[19] + (0.0, -0.1, 0.0), atol=1e-9)
    # A parent's rotation carries its children: rotate the right shoulder, the wrist follows.
    on_wrist = np.zeros((1, 55))
    on_wrist[0, 21] = 1.0
    wrist = rest[21][None]
    moved = lhm.lbs(wrist, on_wrist, rest, parents, {17: (0, 0, -np.pi / 2)})[0]
    r = lhm._rotation((0, 0, -np.pi / 2))
    assert np.allclose(moved, r @ (rest[21] - rest[17]) + rest[17], atol=1e-9)
    assert np.allclose(lhm.lbs(wrist, on_wrist, rest, parents, {}), wrist)


def test_skin_error_is_zero_when_nothing_is_cut():
    raw = synthetic_body(n=800)
    fields = lhm.body_fields(raw)
    rest_pose = lhm.skin_error(raw, fields, pose={})
    assert rest_pose["skin_err_mm"] == {"median": 0.0, "p99": 0.0, "max": 0.0}
    four = raw.weights.copy()
    for i in range(len(four)):                                          # exactly 4 joints each
        four[i, np.argsort(-four[i])[4:]] = 0.0
    raw.weights = four
    exact = lhm.skin_error(raw, lhm.body_fields(raw))
    assert exact["skin_err_mm"]["max"] < 1e-6 and exact["top4_coverage"]["min"] == 1.0
    cut = lhm.skin_error(synthetic_body(n=800), fields)
    assert cut["skin_err_mm"]["max"] > 0 and cut["top4_coverage"]["mean"] < 1.0


def _three_joint_body(n: int = 600) -> lhm.RawBody:
    """Every splat on exactly three moving joints, 0.5 / 0.3 / 0.2: the top-4 cut costs nothing,
    the file's 8-bit rounding does (0.5 * 255 is not a whole number)."""
    raw = synthetic_body(n=n)
    raw.opacity[:] = 0.9
    raw.weights[:] = 0.0
    raw.weights[:, [18, 16, 20]] = (0.5, 0.3, 0.2)                      # elbow, shoulder, wrist
    return raw


def test_skin_error_measures_the_weights_the_file_ships(app):
    raw = _three_joint_body()
    fields = lhm.body_fields(raw)
    assert lhm.skin_error(raw, fields)["skin_err_mm"]["max"] < 1e-6      # float top 4: exact
    shipped = app.file_skin(fields["skin_joints"], fields["skin_weights"])
    quantised = lhm.skin_error(raw, fields, skin=shipped)
    assert quantised["skin_err_mm"]["median"] > 0.05                    # 8-bit: not exact
    packed = app.to_avatar(fields, {}).skin                             # what the file holds
    assert np.array_equal(shipped[0], packed >> 8)
    assert np.allclose(shipped[1], (packed & 0xFF) / 255.0)

    class ThreeJoints(FakeRunner):
        def run(self, views):
            return _three_joint_body()

    meta = avatar_format.unpack(base64.b64decode(
        app.handle(request(), ThreeJoints())["avatar_b64"])).meta
    assert meta["skin_err_mm"] == quantised["skin_err_mm"]
    assert "8-bit" in meta["skin_err_basis"] and "linear blend" in meta["skin_err_basis"]


def test_the_canonical_frame_check():
    lhm.check_canonical_frame(skeleton())
    mirrored = skeleton() * (-1, 1, 1)
    with pytest.raises(RuntimeError, match="canonical frame"):
        lhm.check_canonical_frame(mirrored)
    upside_down = skeleton() * (1, -1, 1)
    with pytest.raises(RuntimeError):
        lhm.check_canonical_frame(upside_down)


# ── LHMRunner's setup, with fake torch / accelerate / LHM++ modules ──────────────────────────────


def test_the_runner_sets_up_accelerate_before_lhm_logs(monkeypatch, tmp_path):
    """LHM++ logs through accelerate.logging, which (accelerate 0.34.2) raises RuntimeError on
    every call until accelerate's state exists; the model's constructor logs. LHMRunner must make
    that state before building the model, as LHM++'s own scripts do with Accelerator()."""
    order = []

    class PartialState:                           # accelerate.state.PartialState's shared state
        _shared_state: dict = {}

        def __init__(self):
            order.append("state")
            PartialState._shared_state["ready"] = True

    def log_info(message):                        # accelerate.logging.MultiProcessAdapter.log
        if PartialState._shared_state == {}:
            raise RuntimeError("You must initialize the accelerate state by calling either "
                               "`PartialState()` or `Accelerator()` before using the logging "
                               "utility.")
        order.append(message)

    class Module:
        def to(self, device):
            return self

        def eval(self):
            return self

    class ModelHumanA4OLRM(Module):
        shape_head = object()

        def __init__(self, **config):
            log_info("Using DINOv2 as the encoder")

    def wrap_model_hub(cls):
        class HfModel(cls):
            @classmethod
            def from_pretrained(cls, path):
                order.append("from_pretrained")
                return cls()
        return HfModel

    class BiRefNet(Module):
        def __init__(self, bb_pretrained):
            pass

        def load_state_dict(self, state):
            pass

    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    module("torch", _dynamo=SimpleNamespace(config=SimpleNamespace(disable=False)),
           load=lambda path, map_location=None: {})
    module("accelerate", PartialState=PartialState)
    module("core")
    module("core.models", model_dict={"human_lrm_a4o": ModelHumanA4OLRM})
    module("core.models.rendering")
    module("core.models.rendering.skinnings")
    module("core.models.rendering.skinnings.constant", SMPLX_JOINTS=avatar_format.SMPLX_JOINTS)
    module("core.utils")
    module("core.utils.hf_hub", wrap_model_hub=wrap_model_hub)
    module("engine")
    module("engine.BiRefNet")
    module("engine.BiRefNet.models")
    module("engine.BiRefNet.models.birefnet", BiRefNet=BiRefNet)
    module("engine.BiRefNet.utils", check_state_dict=lambda state: state)
    transforms = SimpleNamespace(Compose=list, Resize=lambda size: None, ToTensor=lambda: None,
                                 Normalize=lambda mean, std: None)
    module("torchvision", transforms=transforms)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.chdir(tmp_path)

    runner = lhm.LHMRunner(tmp_path, tmp_path / "model", device="cpu")
    assert order == ["state", "from_pretrained", "Using DINOv2 as the encoder"]
    assert sys.modules["torch"]._dynamo.config.disable is True
    assert runner.joint_names == list(avatar_format.SMPLX_JOINTS)


# ── weights: populated once on CPU, read offline by GPU containers ───────────────────────────────


def _fake_hubs(monkeypatch, app, *, hf_works: bool = True):
    """huggingface_hub and modelscope stand-ins that lay out LHM++'s repos in local_dir."""
    calls = []
    bundle = tuple(app.REQUIRED_PRIOR)                          # what the repo holds, fixed

    def lay_out(repo: str, local_dir: str) -> None:
        dest = Path(local_dir)
        if repo in (app.PRIOR_ID, app.MS_PRIOR_ID):
            for name in bundle:
                (dest / name).parent.mkdir(parents=True, exist_ok=True)
                (dest / name).write_bytes(b"x")
        else:
            (dest / "config.json").write_text("{}")
            (dest / "model.safetensors").write_bytes(b"x")

    hub = ModuleType("huggingface_hub")

    class HfApi:
        def model_info(self, repo, revision=None):
            calls.append(("info", repo, revision))
            if not hf_works:
                raise OSError("Hugging Face is unreachable")
            return SimpleNamespace(sha="f" * 40)

    def snapshot_download(repo, revision=None, local_dir=None):
        calls.append(("hf", repo, revision))
        lay_out(repo, local_dir)
        return local_dir

    hub.HfApi, hub.snapshot_download = HfApi, snapshot_download
    scope = ModuleType("modelscope")

    def ms_download(repo, local_dir=None):
        calls.append(("ms", repo))
        lay_out(repo, local_dir)
        return local_dir

    scope.snapshot_download = ms_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "modelscope", scope)
    return calls


def _volume(monkeypatch, app, tmp_path):
    root = tmp_path / "volume"
    root.mkdir()
    monkeypatch.setattr(app, "WEIGHTS_ROOT", root)
    monkeypatch.setattr(app, "PRIOR_DIR", root / "prior")
    monkeypatch.setattr(app, "MODEL_DIR", root / "LHMPP-700M-SMPLX-FREE")
    monkeypatch.setattr(app, "POPULATED_MARKER", root / ".lhmpp-populated")
    events = []
    monkeypatch.setattr(app, "weights", SimpleNamespace(reload=lambda: events.append("reload"),
                                                        commit=lambda: events.append("commit")))
    return events


def test_populate_downloads_once_and_commits(app, monkeypatch, tmp_path):
    events = _volume(monkeypatch, app, tmp_path)
    calls = _fake_hubs(monkeypatch, app)
    raw = app.populate.get_raw_f()
    first = raw()
    assert first["status"] == "downloaded" and events == ["reload", "commit"]
    assert first["revisions"]["model"] == f"hf:{app.MODEL_ID}@{'f' * 40}"
    assert ("hf", app.PRIOR_ID, "f" * 40) in calls and not any(c[0] == "ms" for c in calls)
    assert app._model_dir(app.MODEL_DIR) == app.MODEL_DIR
    events.clear()
    second = raw()                                              # another container, later
    assert second["status"] == "present" and events == ["reload"]


def test_populate_falls_back_to_modelscope_and_checks_the_bundle(app, monkeypatch, tmp_path):
    events = _volume(monkeypatch, app, tmp_path)
    calls = _fake_hubs(monkeypatch, app, hf_works=False)
    out = app.populate.get_raw_f()()
    assert out["revisions"]["prior"] == f"modelscope:{app.MS_PRIOR_ID}@master"
    assert ("ms", app.MS_MODEL_ID) in calls and events[-1] == "commit"
    events.clear()
    # A bundle laid out differently from what LHM++ opens fails here, with a listing.
    monkeypatch.setattr(app, "REQUIRED_PRIOR", app.REQUIRED_PRIOR + ("gone/forever.bin",))
    assert not app._populated()
    with pytest.raises(RuntimeError, match=r"lacks \['gone/forever.bin'\]; it holds"):
        app.populate.get_raw_f()(force=True)
    assert "commit" not in events


def test_a_gpu_container_links_the_prior_and_never_writes_the_volume(app, monkeypatch, tmp_path):
    events = _volume(monkeypatch, app, tmp_path)
    _fake_hubs(monkeypatch, app)
    checkout = tmp_path / "opt" / "lhmpp"
    checkout.mkdir(parents=True)
    monkeypatch.setattr(app, "LHMPP_DIR", checkout)
    monkeypatch.chdir(tmp_path)                                 # restored after the test
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "")
    raw_populate = app.populate.get_raw_f()

    def populate():                                             # the CPU function's download
        events.append("populate")
        raw_populate()

    monkeypatch.setattr(app, "populate", SimpleNamespace(remote=populate))
    app._prepare_container()
    assert events[0] == "populate" and events[-1] == "reload"   # populated once, by the CPU fn
    link = checkout / "pretrained_models"
    assert link.is_symlink() and link.resolve() == app.PRIOR_DIR.resolve()
    assert Path.cwd().resolve() == checkout.resolve() and sys.path[0] == str(checkout)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    events.clear()
    app._prepare_container()                                    # weights there: no populate
    assert events == []


def test_the_runner_is_loaded_once_per_container(app, monkeypatch):
    built = []
    monkeypatch.setattr(app, "_RUNNER", None)
    monkeypatch.setattr(app, "_prepare_container", lambda: built.append("prepare"))
    monkeypatch.setattr(app, "_model_dir", lambda root: Path("/models/m"))
    monkeypatch.setattr(app.lhm, "LHMRunner", lambda repo, model: built.append((repo, model)) or 1)
    assert app._runner() == 1 and app._runner() == 1
    assert built == ["prepare", (app.LHMPP_DIR, Path("/models/m"))]


def test_model_dir_finds_flat_and_stepped_checkpoints(app, tmp_path):
    assert app._model_dir(tmp_path / "missing") is None
    for step in ("step_100", "step_2000", "step_x"):
        (tmp_path / step).mkdir()
        (tmp_path / step / "config.json").write_text("{}")
        (tmp_path / step / "model.safetensors").write_bytes(b"x")
    assert app._model_dir(tmp_path) == tmp_path / "step_2000"
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert app._model_dir(tmp_path) == tmp_path


def test_model_dir_wants_what_from_pretrained_opens(app, tmp_path):
    """huggingface_hub 0.23's PyTorchModelHubMixin loads exactly config.json + model.safetensors
    from a local folder: a .bin or a sharded checkpoint would pass a looser check and then fail
    every model load."""
    for files in (["pytorch_model.bin"], ["model-00001-of-00002.safetensors",
                                          "model-00002-of-00002.safetensors"],
                  ["model.safetensors"]):
        folder = tmp_path / files[0]
        folder.mkdir()
        for name in files:
            (folder / name).write_bytes(b"x")
        if files != ["model.safetensors"]:
            (folder / "config.json").write_text("{}")
        assert app._model_dir(folder) is None, files


def test_the_prior_check_covers_flame():
    """LHM++ opens FLAME_NEUTRAL.pkl (smplx.create(path, "flame")) to graft FLAME's expressions
    onto SMPL-X, and the FLAME 2019 model for the face; populate must not finish without them."""
    text = APP_FILE.read_text()
    assert '"human_model_files/flame/FLAME_NEUTRAL.pkl"' in text
    assert '"human_model_files/flame/2019/generic_model.pkl"' in text


def test_populate_refuses_a_checkpoint_from_pretrained_cannot_open(app, monkeypatch, tmp_path):
    events = _volume(monkeypatch, app, tmp_path)
    _fake_hubs(monkeypatch, app)
    real = sys.modules["huggingface_hub"].snapshot_download

    def bin_only(repo, revision=None, local_dir=None):            # an older .bin checkpoint
        real(repo, revision=revision, local_dir=local_dir)
        if repo == app.MODEL_ID:
            (Path(local_dir) / "model.safetensors").rename(Path(local_dir) / "pytorch_model.bin")
        return local_dir

    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", bin_only)
    with pytest.raises(RuntimeError, match=r"model.safetensors under .*pytorch_model.bin"):
        app.populate.get_raw_f()()
    assert "commit" not in events


def test_nothing_on_the_request_path_writes_the_volume():
    tree = ast.parse(APP_FILE.read_text())
    hot = {"avatar", "handle", "_runner", "_prepare_container", "_retire_container"}
    seen = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in hot:
            seen.add(node.name)
            text = ast.unparse(node)
            assert "commit(" not in text and "write_" not in text, node.name
            assert "WEIGHTS_ROOT /" not in text and "PRIOR_DIR /" not in text, node.name
    assert seen == hot


def test_the_try_it_command_sends_photos_like_the_browser(app, tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6                                            # stored sideways, as phones do
    exif[0x8825] = {1: "N"}                                     # a GPS block that must not leave
    path = tmp_path / "me.jpg"
    Image.new("RGB", (4000, 3000), "white").save(path, format="JPEG", exif=exif)
    sent = app._photo(path)
    assert sent.startswith(app.JPEG_MAGIC) and len(sent) <= app.MAX_IMAGE_BYTES
    with Image.open(io.BytesIO(sent)) as im:
        assert im.size == (1200, 1600) and not im.getexif()     # upright, shrunk, no EXIF
    assert app.parse_request(request(images_b64=[b64(sent)]))["images"] == [sent]
