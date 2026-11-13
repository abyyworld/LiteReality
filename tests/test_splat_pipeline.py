"""The splat room's build: which physics is served, the photo masks when training fails, what a
re-cut keeps in its report, and what `--instant-only` asks of Modal (pipeline/splatroom, cli)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_splat import SCENE, _grid, _red_masks, _scan_above_box, _splats_at  # noqa: E402

from litereality_agent import cli  # noqa: E402
from litereality_agent.models.splat import spz  # noqa: E402
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
                        lambda frames: SimpleNamespace(tar=lambda: b"tar"))

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
