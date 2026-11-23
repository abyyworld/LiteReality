"""The splat room's build: which physics is served, the photo masks when training fails, what a
re-cut keeps in its report, what `--instant-only` asks of Modal, and how dense depth and the
cleanup are threaded through it, both sides of each contract checked (pipeline/splatroom, cli)."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_splat import SCENE, _grid, _red_masks, _scan_above_box, _splats_at  # noqa: E402

from litereality_agent import cli  # noqa: E402
from litereality_agent.models.splat import promptda, rgbd, spz  # noqa: E402
from litereality_agent.models.splat.spz import Splats  # noqa: E402
from litereality_agent.pipeline import splatroom  # noqa: E402


@pytest.fixture
def box_scan(tmp_path):
    """Two photos straight down on a box at (1, 0)."""
    scan = tmp_path / "scan"
    scan.mkdir()
    _scan_above_box(scan, box_x=1.0, index=0, cam_dx=-0.1)
    _scan_above_box(scan, box_x=1.0, index=1, cam_dx=0.1)
    return scan


def _scene(tmp_path: Path, x: float) -> Path:
    xml = tmp_path / "objects" / "scene.xml"
    xml.parent.mkdir(exist_ok=True)
    xml.write_text(SCENE.format(x=x))
    return xml


def test_a_cut_that_moves_nothing_serves_the_scanned_physics_again(tmp_path, box_scan):
    """An earlier run left its moved physics (`scene_splat.xml`) live; this one moves nothing, so
    its bodies are cut at the scanned poses: the scanned scene is published again."""
    xml = _scene(tmp_path, 1.0)
    served: list[Path] = []
    splatroom.build(box_scan, tmp_path / "splat", xml, instant_only=True, log=lambda _: None,
                    on_scene=served.append, move_bodies=False)
    assert served == []                                   # nothing was ever moved: nothing to undo

    (xml.parent / splatroom.SPLAT_SCENE).write_text(SCENE.format(x=1.2))   # an earlier run's
    report = splatroom.build(box_scan, tmp_path / "splat", xml, instant_only=True,
                             log=lambda _: None, on_scene=served.append, move_bodies=False)
    assert report["corrections"] == {} and report["corrected_scene"] is None
    assert served == [xml]


class _FailingTrainer:
    name = "failing"

    def stream(self, _tar, _options=None):
        yield {"kind": "error", "error": "the app is not deployed"}


def test_a_trainer_that_fails_before_a_snapshot_still_cuts_by_the_photos(tmp_path, box_scan,
                                                                         monkeypatch):
    """The masks arrive after the trainer gave up: the instant room is cut by them again, the
    bodies' corrections are decided, and report.json is written once the masks thread is done."""
    xml = _scene(tmp_path, 1.0)
    monkeypatch.setattr(splatroom.splat_dataset, "build",
                        lambda frames, **_: SimpleNamespace(tar=lambda: b"tar", files={}, images=2,
                                                            points=0, dense_images=0))

    def slow_masks(request: dict) -> dict:
        time.sleep(0.3)
        return _red_masks(request)

    out = tmp_path / "splat"
    report = splatroom.build(box_scan, out, xml, trainer=_FailingTrainer(), log=lambda _: None,
                             segment=slow_masks)
    assert report["error"] == "the app is not deployed"
    assert report["masks"]["masks"] == 2
    assert "corrections" in report and "corrected_scene" in report     # decided, by the photos
    assert [b["masks"] for b in report["bundles"]] == [False, True]
    assert json.loads((out / splatroom.REPORT).read_text())["masks"] == report["masks"]


def test_a_recut_keeps_what_the_training_run_measured(tmp_path, box_scan):
    xml = _scene(tmp_path, 1.0)
    out = tmp_path / "splat"
    out.mkdir()
    floor = _splats_at(_grid((0.4, -0.5, 0.0), (1.6, 0.5, 0.0), step=0.05))
    (out / splatroom.RAW).write_bytes(spz.encode(floor))
    trained = {"gpu": "L4", "trainer_s": 412.3, "dataset_s": 3.1, "dataset_mb": 88.4,
               "trainer_started_s": 21.7, "stats": {"psnr": 24.5},
               "bundles": [{"stage": "instant", "step": None},
                           {"stage": "trained", "step": 7000, "trainer_s": 410.0}]}
    (out / splatroom.REPORT).write_text(json.dumps(trained))

    report = splatroom.build(box_scan, out, xml, trainer=splatroom.SavedTrainer(out),
                             log=lambda _: None, move_bodies=False)
    for key in ("gpu", "trainer_s", "dataset_s", "dataset_mb", "trainer_started_s", "stats"):
        assert report[key] == trained[key], key
    assert report["recut"] is True
    last = report["bundles"][-1]
    assert last["stage"] == "trained" and last["step"] == 7000 and last["replayed"]
    assert last["trainer_s"] == 410.0

    again = splatroom.build(box_scan, out, xml, trainer=splatroom.SavedTrainer(out),
                            log=lambda _: None, move_bodies=False)
    assert again["gpu"] == "L4" and again["trainer_s"] == 412.3        # and after a second one


def test_instant_only_asks_modal_for_nothing(tmp_path, box_scan, monkeypatch):
    """`--instant-only` is no GPU: no SAM 2 client is made (which would need Modal credentials)."""
    from litereality_agent.models.splat import remote

    xml = _scene(tmp_path, 1.0)
    context = SimpleNamespace(scene_dir=tmp_path, scan="box")
    monkeypatch.setattr(cli, "_quick_target", lambda *a: (box_scan, context))
    monkeypatch.setattr(cli, "load_settings",
                        lambda: SimpleNamespace(modal_configured=lambda: False))
    monkeypatch.setattr(cli, "_modal_kwargs",
                        lambda s: {"credentials": None, "profile": None, "environment": None})
    monkeypatch.setattr(cli, "_in_modal_container", lambda: False)

    def no_modal(**_):
        raise RuntimeError("Modal calls need credentials")

    monkeypatch.setattr(remote, "segment_call", no_modal)
    built: list[dict] = []
    monkeypatch.setattr(splatroom, "build",
                        lambda *a, **kw: built.append(kw) or {"total_s": 0.0})
    args = cli._parser().parse_args(["splatroom", str(box_scan), "--scene", str(xml),
                                     "--out", str(tmp_path / "splat"), "--instant-only"])
    assert cli._splatroom(args) == 0
    assert built[0]["segment"] is None and built[0]["trainer"] is None

    args = cli._parser().parse_args(["splatroom", str(box_scan), "--scene", str(xml),
                                     "--out", str(tmp_path / "splat")])
    monkeypatch.setattr(remote, "ModalSplatTrainer", lambda **_: object())
    with pytest.raises(RuntimeError, match="credentials"):   # masks still asked for otherwise
        cli._splatroom(args)


def test_the_gpu_is_warmed_up_before_the_instant_room_and_only_when_training(tmp_path, monkeypatch):
    from litereality_agent.models.splat.remote import ModalSplatTrainer

    calls = []

    class Function:
        def remote_gen(self, data, options):
            calls.append(("gen", data, dict(options)))
            yield {"kind": "done", "warm": True} if not data else {"kind": "error", "error": "x"}

    trainer = ModalSplatTrainer(function=Function())
    trainer.warm()
    import time as _time
    for _ in range(50):
        if calls:
            break
        _time.sleep(0.02)
    assert calls and calls[0][1] == b""                       # an empty dataset: boot only

    class Failing:
        def remote_gen(self, data, options):
            raise RuntimeError("no Modal here")
    ModalSplatTrainer(function=Failing()).warm()              # never raises


# ── dense depth and the cleanup ──────────────────────────────────────────────────────────────────

SPLAT_APP = Path(__file__).resolve().parents[1] / "deploy" / "modal" / "splat"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeRefiner:
    """PromptDA as the deployed client behaves: the LiDAR upsampled to the asked width (exact for
    the box scan's flat surfaces), cached beside the scan as `promptda.save_dense` does."""

    name = "fake-promptda"

    def __init__(self, fail: bool = False) -> None:
        self.fail, self.calls, self.last_stats = fail, [], {}

    def refine(self, frames, width=960):
        self.calls.append((len(frames), width))
        if self.fail:
            self.last_stats = {"frames": len(frames), "refined": 0}
            return {}
        out = {}
        for f in frames:
            if (cached := promptda.load_dense(f, width)) is not None:
                out[f.index] = cached
                continue
            lidar, _ = rgbd.read_lidar(f)
            out[f.index] = rgbd.resample_nearest(lidar, (promptda.output_height(f, width), width))
            promptda.save_dense(f, out[f.index])
        self.last_stats = {"frames": len(frames), "refined": len(out)}
        return out


class _RecordingTrainer:
    """Keeps the tar and options it is sent; answers with one snapshot of `splats`."""

    name = "recording"

    def __init__(self, splats: Splats) -> None:
        self.splats, self.sent = splats, []

    def stream(self, tar, options=None):
        self.sent.append((tar, dict(options or {})))
        yield {"kind": "started", "seconds": 0.0, "gpu": "fake"}
        yield {"kind": "snapshot", "step": 100, "seconds": 1.0, "spz": spz.encode(self.splats)}
        yield {"kind": "done", "seconds": 1.0, "stats": {"psnr": 20.0}}


def _trained_room() -> tuple[Splats, dict[str, int]]:
    """The box scan's floor and box top as splats, plus one faint splat on the floor, one floater
    in the air between camera 1 and the floor, and one far outside the room's box."""
    floor = _grid((0.2, -0.7, 0.0), (1.8, 0.7, 0.0), step=0.05)
    floor = floor[~((np.abs(floor[:, 0] - 1.0) < 0.25) & (np.abs(floor[:, 1]) < 0.25))]
    top = _grid((0.85, -0.15, 0.5), (1.15, 0.15, 0.5), step=0.05)
    extra = np.array([[0.6, 0.5, 0.0],          # faint
                      [1.1, -0.5, 1.5],         # floater: camera 1 sees the floor through it
                      [9.0, 9.0, 9.0]])         # outside
    pts = np.concatenate([floor, top, extra])
    splats = _splats_at(pts)
    splats.opacities[:] = 0.9
    splats.opacities[len(floor) + len(top)] = 0.005
    return splats, {"surface": len(floor) + len(top)}


def test_dense_depth_and_the_cleanup_run_through_every_stage(tmp_path, box_scan, monkeypatch):
    """--depth promptda --depth-loss-dense --normal-lambda --clean on a scan not yet refined: the
    LiDAR room is up first and replaced by the dense one; the dataset carries a depth map per
    photo under the name the trainer looks up; carving holds out the trainer's photos; the uncut
    snapshot and the cleaned one are both kept."""
    from litereality_agent.models.splat import clean as splat_clean

    dense_lib = _load("litereality_dense", SPLAT_APP / "litereality_dense.py")
    xml = _scene(tmp_path, 1.0)
    out = tmp_path / "splat"
    room, counts = _trained_room()
    trainer = _RecordingTrainer(room)
    refiner = _FakeRefiner()
    calls = []
    real_clean = splat_clean.clean

    def spy(splats, frames, **kw):
        calls.append(kw)
        return real_clean(splats, frames, **kw)

    monkeypatch.setattr(splatroom.splat_clean, "clean", spy)
    options = {"dense_depth": True, "normal_lambda": 0.05}
    report = splatroom.build(box_scan, out, xml, trainer=trainer, options=options,
                             log=lambda _: None, move_bodies=False, refiner=refiner,
                             clean=True, clean_options={"carve_votes": 1,
                                                        "bounds": ((-1, -2, -0.5), (3, 2, 3.5))})
    assert refiner.calls == [(2, 960)]
    stages = [(b["stage"], b.get("depth")) for b in report["bundles"]]
    assert stages == [("instant", None), ("instant", "promptda"), ("trained", None)]
    assert report["depth"]["source"] == "promptda" and report["depth"]["frames"] == 2
    assert "instant_dense_s" in report
    assert sorted(p.name for p in (box_scan / promptda.DENSE_DIR).iterdir()) == [
        "frame_00000.png", "frame_00001.png"]

    # The trainer: the options as given, a depth map (at half size) for every photo it trains on
    # under its looked-up name; the held-out photo (the first by name) is listed, with neither.
    tar, sent = trainer.sent[0]
    assert sent == {"dense_depth": True, "normal_lambda": 0.05} == report["options"]
    data = tmp_path / "data"
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(data)
    assert report["images"] == ["frame_00000.jpg", "frame_00001.jpg"]
    for name in report["images"]:
        assert (data / "images" / name).is_file()
    assert dense_lib.depth_path(str(data), "frame_00000.jpg") is None
    path = dense_lib.depth_path(str(data), "frame_00001.jpg")
    depth = np.asarray(Image.open(path)) / 1000.0
    assert depth.shape == (360, 480) and np.isclose(np.median(depth[depth > 0]), 3.0, atol=0.01)
    assert dense_lib.load_depth(path, (960, 720)).shape == (720, 960)
    assert report["dataset"]["dense_photos"] == 2 and report["dataset"]["depth_maps"] == 1
    assert report["dataset"]["held_out"] == 1
    assert report["dataset"]["points_per_photo"] == splatroom.DENSE_POINTS_MAX

    # The cleanup: every snapshot, carving with the trainer's photo list and hold-out.
    # The dense instant room is fused once the training set exists, so it knows the list too.
    assert [c["image_names"] for c in calls] == [None, report["images"], report["images"]]
    assert all(c["test_every"] == 8 for c in calls)
    trained = report["bundles"][-1]["clean"]
    assert (trained["faint"], trained["outside"], trained["carved"]) == (1, 1, 1)
    assert trained["held_out_frames"] == 1 and trained["kept"] == counts["surface"]
    assert report["clean"]["carve_votes"] == 1 and report["clean"]["room_bounds"] is True
    assert len(spz.decode((out / splatroom.RAW).read_bytes())) == len(room)
    assert len(spz.decode((out / splatroom.RAW_CLEAN).read_bytes())) == counts["surface"]
    assert json.loads((out / splatroom.REPORT).read_text())["bundles"][-1]["clean"] == trained

    # Refined once: the next run reads the cache, so the first room is already the dense one.
    again = splatroom.build(box_scan, out, xml, instant_only=True, log=lambda _: None,
                            move_bodies=False, refiner=_FakeRefiner(), clean=False)
    assert [(b["stage"], b.get("depth")) for b in again["bundles"]] == [("instant", "promptda")]
    assert "clean" not in again and "clean" not in again["bundles"][0]


def test_without_dense_depth_the_dense_losses_are_dropped_not_the_run(tmp_path, box_scan):
    """PromptDA not deployed (the client returns {}): the LiDAR room and dataset as before, and
    the dense losses taken out of the trainer's options (it would refuse them without depths/)."""
    xml = _scene(tmp_path, 1.0)
    room, _ = _trained_room()
    trainer = _RecordingTrainer(room)
    (tmp_path / "splat").mkdir()
    (tmp_path / "splat" / splatroom.RAW_CLEAN).write_bytes(b"an earlier run's")
    report = splatroom.build(box_scan, tmp_path / "splat", xml, trainer=trainer,
                             options={"steps": 500, "dense_depth": True,
                                      "dense_depth_lambda": 0.2, "normal_lambda": 0.05},
                             log=lambda _: None, move_bodies=False,
                             refiner=_FakeRefiner(fail=True))
    tar, sent = trainer.sent[0]
    assert sent == {"steps": 500}
    assert report["dense_losses_dropped"] == {"dense_depth": True, "dense_depth_lambda": 0.2,
                                              "normal_lambda": 0.05}
    assert report["depth"]["fallback"] == "lidar"
    assert [b["stage"] for b in report["bundles"]] == ["instant", "trained"]
    assert report["dataset"]["points_per_photo"] == splatroom.splat_dataset.POINTS_PER_FRAME
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        assert not [n for n in t.getnames() if n.startswith("depths")]
    assert "clean" not in report["bundles"][-1]
    assert (tmp_path / "splat" / splatroom.RAW).is_file()
    assert not (tmp_path / "splat" / splatroom.RAW_CLEAN).exists()


class _SlowRefiner(_FakeRefiner):
    """A depth GPU that never comes free (or an upload stalled on a flaky link)."""

    def __init__(self) -> None:
        super().__init__()
        self.warmed = 0
        import threading

        self.release = threading.Event()

    def warm(self):
        self.warmed += 1

    def refine(self, frames, width=960):
        self.release.wait(10)
        return super().refine(frames, width)


def test_training_never_waits_past_the_depth_deadline(tmp_path, box_scan):
    """Depth not back in time: the room trains on the LiDAR (no dense losses), says why, and the
    depth GPU was woken at the start; the late depth still fills the cache for the next run."""
    xml = _scene(tmp_path, 1.0)
    room, _ = _trained_room()
    trainer = _RecordingTrainer(room)
    refiner = _SlowRefiner()
    t = time.perf_counter()
    report = splatroom.build(box_scan, tmp_path / "splat", xml, trainer=trainer,
                             options={"dense_depth": True, "normal_lambda": 0.05},
                             log=lambda _: None, move_bodies=False, refiner=refiner,
                             depth_deadline=0.5)
    assert time.perf_counter() - t < 8
    assert refiner.warmed == 1
    assert report["depth"]["fallback"] == "timeout" and report["depth"]["deadline_s"] == 0.5
    assert [b["stage"] for b in report["bundles"]] == ["instant", "trained"]
    assert trainer.sent[0][1] == {} and "dense_losses_dropped" in report
    assert report["dataset"]["dense_photos"] == 0
    refiner.release.set()
    for _ in range(200):
        if all(promptda.is_cached(f, 960) for f in rgbd.load_frames(box_scan)):
            break
        time.sleep(0.05)
    assert all(promptda.is_cached(f, 960) for f in rgbd.load_frames(box_scan))


def test_the_trainer_never_waits_for_the_dense_instant_room(tmp_path, box_scan, monkeypatch):
    """Depth refined during the run: the dense instant room is fused while the training set is
    built and sent, so the trainer is called before that fuse ends (were it on the trainer's
    path, the fuse below would wait its whole timeout); it is still shown before the snapshot."""
    import threading

    xml = _scene(tmp_path, 1.0)
    room, _ = _trained_room()
    called = threading.Event()
    released_by_trainer = []
    real = rgbd.instant_splats

    def slow_dense(frames, *, dense=None, **kw):
        if dense:
            released_by_trainer.append(called.wait(5))
        return real(frames, dense=dense, **kw)

    class Trainer(_RecordingTrainer):
        def stream(self, tar, options=None):
            called.set()
            yield from super().stream(tar, options)

    monkeypatch.setattr(splatroom.rgbd, "instant_splats", slow_dense)
    report = splatroom.build(box_scan, tmp_path / "splat", xml, trainer=Trainer(room),
                             log=lambda _: None, move_bodies=False, refiner=_FakeRefiner())
    assert released_by_trainer == [True]
    assert [(b["stage"], b.get("depth")) for b in report["bundles"]] == [
        ("instant", None), ("instant", "promptda"), ("trained", None)]
    assert report["instant_dense_s"] <= report["bundles"][-1]["at"]


def test_the_dense_points_share_the_trainers_splat_budget():
    assert splatroom.dense_points_for(40) == 20_000            # few photos: the cap
    assert splatroom.dense_points_for(62) == 800_000 // 62
    assert splatroom.dense_points_for(300) == 4000             # never below the LiDAR's own
    assert splatroom.dense_points_for(100, cap_max=1_000_000) == 10_000


def test_the_options_splatroom_sends_are_the_trainers_own():
    """Every option name the CLI and build may send is one the deployed trainer validates."""
    app = _load("litereality_splat_app_pipeline", SPLAT_APP / "app.py")
    sent = {"steps": 1250, "test_every": 100000, "dense_depth": True, "dense_depth_lambda": 0.2,
            "normal_lambda": 0.05}
    opt = app._options(sent)
    assert app._wants_dense(opt) and splatroom.wants_dense_maps(sent)
    assert opt["test_every"] == 100000
    assert splatroom.TEST_EVERY == app.DEFAULTS["test_every"]
    assert splatroom.CAP_MAX == app.DEFAULTS["cap_max"]
    assert not splatroom.wants_dense_maps({"normal_lambda": 0.0})


def test_the_cli_threads_depth_and_clean_into_the_build(tmp_path, box_scan, monkeypatch):
    from litereality_agent.models.splat import remote

    xml = _scene(tmp_path, 1.0)
    context = SimpleNamespace(scene_dir=tmp_path, scan="box")
    monkeypatch.setattr(cli, "_quick_target", lambda *a: (box_scan, context))
    monkeypatch.setattr(cli, "load_settings",
                        lambda: SimpleNamespace(modal_configured=lambda: True))
    monkeypatch.setattr(cli, "_modal_kwargs",
                        lambda s: {"credentials": ("id", "secret"), "profile": None,
                                   "environment": None})
    monkeypatch.setattr(remote, "ModalSplatTrainer", lambda **_: "trainer")
    made = []
    monkeypatch.setattr(promptda, "ModalDepthRefiner", lambda **kw: made.append(kw) or "refiner")
    built: list[dict] = []
    monkeypatch.setattr(splatroom, "build",
                        lambda *a, **kw: built.append(kw) or {"total_s": 0.0})
    base = ["splatroom", str(box_scan), "--scene", str(xml), "--out", str(tmp_path / "splat"),
            "--no-masks"]

    assert cli._splatroom(cli._parser().parse_args(base)) == 0
    kw = built[-1]
    assert kw["refiner"] is None and kw["clean"] is False and kw["options"] == {}
    assert kw["dense_points"] is None and kw["clean_options"] == {} and made == []

    args = cli._parser().parse_args(base + [
        "--depth", "promptda", "--dense-points", "9000", "--depth-loss-dense",
        "--dense-depth-lambda", "0.2", "--normal-lambda", "0.05", "--clean", "--carve-votes", "3"])
    assert cli._splatroom(args) == 0
    kw = built[-1]
    assert kw["refiner"] == "refiner" and made[0]["credentials"] == ("id", "secret")
    assert kw["options"] == {"dense_depth": True, "dense_depth_lambda": 0.2,
                             "normal_lambda": 0.05}
    assert kw["dense_points"] == 9000 and kw["clean"] is True
    assert kw["clean_options"] == {"carve_votes": 3}

    with pytest.raises(SystemExit, match="--depth promptda"):
        cli._splatroom(cli._parser().parse_args(base + ["--depth-loss-dense"]))
    assert cli._parser().parse_args(base + ["--clean", "--no-clean"]).clean is False
