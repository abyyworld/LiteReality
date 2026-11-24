"""A trained room cut again on a laptop with its bodies put on the real chairs (splatroom.recut,
`litereality splatcut`), the move test of any bundle as served (splatroom.bundle_move_test), and
the build's own instances mode (splatroom.build(instances=True))."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_splat import _scan_above_box  # noqa: E402
from test_splat_instances import _chair_points, _floor, _haze, _scene, _splats  # noqa: E402

from litereality_agent import cli  # noqa: E402
from litereality_agent.models.splat import spz  # noqa: E402
from litereality_agent.pipeline import splatroom  # noqa: E402

REAL = {"Chair0": (-1.3, 0.1, 30.0), "Chair1": (0.25, 0.45, 90.0), "Chair2": (1.35, -0.15, 180.0)}


@pytest.fixture
def trained(tmp_path):
    """A splatroom folder as a training run leaves it: the raw snapshot of a room with three real
    chairs, RoomPlan's regularised row of three boxes beside it in physics/, a report."""
    src = tmp_path / "v2"
    physics = src / "physics"
    physics.mkdir(parents=True)
    scene = _scene(physics, [("Chair0", -1.2, 0, 0), ("Chair1", 0, 0, 0), ("Chair2", 1.2, 0, 0)])
    rng = np.random.default_rng(1)
    pts = [_chair_points(*v) for v in REAL.values()]
    raw = _splats([*pts, _floor()], [_haze(p, rng, len(p) // 2) for p in pts])
    (src / splatroom.RAW).write_bytes(spz.encode(raw))
    (src / splatroom.REPORT).write_text(json.dumps({
        "scene": str(scene), "gpu": "H100", "trainer_s": 588.4, "stats": {"psnr": 22.8},
        "bundles": [{"stage": "trained", "step": 10000}]}))
    old = splatroom.split(raw, mujoco.MjModel.from_xml_path(str(scene)), snap_bodies=False)
    splatroom.write_bundle(src, old, stage="trained", step=10000, view={"pos": [0, 0, 1]})
    return src, scene


def test_a_recut_into_its_own_folder_fits_the_physics_and_passes(trained, tmp_path):
    src, scene = trained
    before = sorted(p.name for p in src.iterdir())
    out = tmp_path / "v2_chairs"
    report = splatroom.recut(src, out, log=lambda _m: None)
    assert sorted(p.name for p in src.iterdir()) == before          # the source is untouched
    assert report["recut"] and report["gpu"] == "H100" and report["stats"] == {"psnr": 22.8}
    served = Path(report["corrected_scene"])
    assert served == out / "physics" / splatroom.SPLAT_SCENE and served.is_file()
    assert (out / splatroom.RAW).is_file()                          # re-cuttable, scoreable
    bundle = splatroom.read_bundle(out)
    assert bundle["step"] == 10000 and bundle["view"] == {"pos": [0, 0, 1]}
    assert bundle["grabbable"] == {"Chair0": True, "Chair1": True, "Chair2": True}
    assert bundle["move_test"]["pass"] and report["move_test"]["pass"]
    # the physics served with the bundle has exactly its bodies, where it cut them
    model = mujoco.MjModel.from_xml_path(str(served))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name, real in REAL.items():
        b = model.body(name).id
        local = spz.decode((out / bundle["bodies"][name]).read_bytes())
        with np.errstate(all="ignore"):
            world = local.positions @ np.asarray(data.xmat[b]).reshape(3, 3).T + data.xpos[b]
        chair = np.median(_chair_points(*real)[:, :2], 0)
        assert np.hypot(*(np.median(world[:, :2], 0) - chair)) < 0.03, name


def test_the_bundle_move_test_fails_the_box_cut_and_passes_the_recut(trained, tmp_path):
    src, _scene_xml = trained
    old = splatroom.bundle_move_test(src)
    assert not old["pass"] and len(old["objects"]) == 3
    splatroom.recut(src, tmp_path / "new", log=lambda _m: None)
    new = splatroom.bundle_move_test(tmp_path / "new")
    assert new["pass"]
    assert {r["body"] for r in new["objects"]} == set(REAL)
    assert all(r["in_body"] >= 0.9 and r["leak"] <= 0.05 for r in new["objects"])


def test_the_cli_recuts_and_checks(trained, tmp_path, capsys):
    src, _scene_xml = trained
    out = tmp_path / "cli_out"
    assert cli.main(["splatcut", str(src), "--check"]) == 1                 # the box cut fails
    assert cli.main(["splatcut", str(src), "--out", str(out)]) == 0
    assert cli.main(["splatcut", str(out), "--check", "--json", str(tmp_path / "t.json")]) == 0
    assert json.loads((tmp_path / "t.json").read_text())["pass"]
    said = capsys.readouterr().out
    assert "move test: pass" in said and "move test: FAIL" in said


def test_a_recut_by_boxes_still_works(trained, tmp_path):
    src, _scene_xml = trained
    report = splatroom.recut(src, tmp_path / "boxes", instances=False, log=lambda _m: None)
    assert report["cut"]["instances"] is False and report["instances"] is None
    assert report["move_test"]["cut"] == "boxes"


def test_build_in_instances_mode_fits_on_the_trained_snapshot(trained, tmp_path):
    src, scene = trained
    scan = tmp_path / "scan"
    scan.mkdir()
    _scan_above_box(scan, box_x=1.0, index=0, cam_dx=-0.1)
    _scan_above_box(scan, box_x=1.0, index=1, cam_dx=0.1)
    served: list[Path] = []
    report = splatroom.build(scan, src, scene, trainer=splatroom.SavedTrainer(src),
                             log=lambda _m: None, on_scene=served.append, instances=True)
    assert report["corrected_scene"] == str(scene.with_name(splatroom.SPLAT_SCENE))
    assert served and served[-1] == scene.with_name(splatroom.SPLAT_SCENE)
    assert set(report["instances"]["bodies"]) == set(REAL)
    assert report["bundles"][-1]["stage"] == "trained" and report["bundles"][-1]["move_test"]
    bundle = splatroom.read_bundle(src)
    assert bundle["move_test"]["pass"] and all(bundle["grabbable"].values())
