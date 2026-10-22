"""Infinigen's openable furniture as an objects-stage source: specs from scanned boxes, the runners
and the cache, the MJCF export read back, cleaned, fitted into the box with our physics and
colour, and composed into the realroom scene in place of the template.

Everything here runs offline on a small synthetic export shaped like Infinigen's (`fake_export`:
an ``object`` body, ``link_N`` children, ``<label>_N`` meshes with baked diffuse PNGs and ``_colN``
hulls, robot-grade densities and joint dynamics). One opt-in test (``-m infinigen``) builds a real
asset in the env ``$LR_INFINIGEN_PYTHON`` names.
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import io
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_quickroom import OBJECTS, _frame_looking_down, write_roomplan_usdz  # noqa: E402

from litereality_agent import cli  # noqa: E402
from litereality_agent.models.articulated import infinigen as ig  # noqa: E402
from litereality_agent.models.articulated import infinigen_worker  # noqa: E402
from litereality_agent.models.object_generation.sim import properties  # noqa: E402
from litereality_agent.pipeline.realroom import (  # noqa: E402
    FakeGenerator,
    RealRoom,
    articulated,
    generate_room,
)
from litereality_agent.pipeline.realroom.build import Arrival  # noqa: E402
from litereality_agent.room_ops.roomcall.hosted import scene_files  # noqa: E402
from litereality_agent.room_ops.roomcall.model_info import movable_body_ids  # noqa: E402
from litereality_agent.room_ops.roomcall.sim import RoomSim  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
APP_FILE = REPO / "deploy" / "modal" / "infinigen" / "app.py"
# Read before `_hermetic` clears it for every other test: the one test that builds a real asset.
REAL_INFINIGEN = os.environ.get(ig.PYTHON_ENV, "").strip()


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    for key in (ig.PYTHON_ENV, ig.WORKERS_ENV, "LR_ARTICULATED"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(ig.CACHE_ENV, str(tmp_path / "cache"))


@pytest.fixture()
def scan(tmp_path) -> Path:
    folder = tmp_path / "scans" / "test_room"
    write_roomplan_usdz(folder / "room.usdz")
    return folder


def _export(tmp_path, spec: ig.Spec) -> Path:
    return ig.fake_export(spec, tmp_path / "exports" / spec.key)


def _compile(built: ig.Built, root: Path):
    ig.write_files(built, root, root)
    model = mujoco.MjModel.from_xml_string(ig.standalone_xml(built, root, root))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _drawn_box(model, data, group: int = 2) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_group[g] != group or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] +
                            model.mesh_vertnum[mid]]
        w = data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    return lo, hi


def _body(model, name: str) -> int:
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert b >= 0, name
    return b


# ── what becomes what ────────────────────────────────────────────────────────────────────────────


def test_categories_map_to_factories_sized_to_the_box():
    dish = ig.spec_for_object("dishwasher", (0.6, 0.6, 0.9), bottom_gap=0.0, worktop=0.03)
    assert dish.factory == "dishwasher" and dish.size == (0.6, 0.6, 0.79)   # worktop and plinth
    wall = ig.spec_for_object("storage", (1.75, 0.36, 0.78), bottom_gap=1.4)
    assert wall.factory == "cabinet" and wall.opts["bottom"] == 0.0 and wall.opts["shelves"] == 2
    floor = ig.spec_for_object("storage", (1.12, 0.63, 0.86), bottom_gap=0.0,
                               rgb=(0.9, 0.9, 0.9))
    assert floor.opts["bottom"] == ig.TOE_KICK and floor.opts["frame_material"] == "white"
    assert ig.spec_for_object("storage", (0.5, 0.6, 0.5), bottom_gap=0.0,
                              rgb=(0.1, 0.08, 0.05)).opts["frame_material"] == "black_wood"
    chest = ig.spec_for_object("storage", (0.9, 0.5, 0.8), bottom_gap=0.0, drawers=True)
    assert chest.factory == "drawer" and chest.opts["cols"] == 2 and chest.opts["rows"] == 3
    assert ig.spec_for_object("refrigerator", (0.7, 0.7, 1.8), bottom_gap=0.0).factory == \
        "refrigerator"
    assert ig.spec_for_object("oven", (0.6, 0.6, 0.9), bottom_gap=0.0).size == (0.6, 0.6, 0.82)
    # no factory: tall storage, a basin on its own, seating, a storage unit floating mid-air
    assert ig.spec_for_object("storage", (0.9, 0.4, 1.9), bottom_gap=0.0) is None
    assert ig.spec_for_object("sink", (0.5, 0.45, 0.2), bottom_gap=0.7) is None
    assert ig.spec_for_object("sofa", (1.8, 0.85, 0.8), bottom_gap=0.0) is None
    assert ig.spec_for_object("storage", (0.6, 0.4, 0.6), bottom_gap=0.2) is None


def test_the_cache_key_is_factory_size_to_the_centimetre_and_seed():
    a = ig.Spec.make("cabinet", (1.1202, 0.6252, 0.8946), seed=7, bottom=0.08)
    b = ig.Spec.make("cabinet", (1.1249, 0.6301, 0.8949), seed=7, bottom=0.08)
    assert a == b and a.key == b.key and a.key.startswith("cabinet-112x63x89cm-s7-")
    assert a.key.endswith(f"-w{ig.WORKER_VERSION}")       # a changed worker builds anew
    assert ig.Spec.make("cabinet", a.size, seed=8, bottom=0.08).key != a.key
    assert ig.Spec.make("cabinet", a.size, seed=7, bottom=0.0).key != a.key
    with pytest.raises(ValueError):
        ig.Spec.make("sofa", (1, 1, 1))
    with pytest.raises(ValueError):
        ig.door_spec(0.8, 2.0, hinge="up")
    assert ig.door_spec(0.8, 2.0, hinge="right").opts["orientation"] == "right"
    assert set(a.job()) == {"key", "factory", "size", "seed", "options"}


def test_the_worker_never_imports_litereality():
    """It runs in Infinigen's own Python, where none of this repo is installed."""
    tree = ast.parse(Path(infinigen_worker.__file__).read_text())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module]
    assert not [n for n in names if n.startswith("litereality")]
    assert ig.FACTORIES == infinigen_worker.FACTORIES


# ── reading an export ────────────────────────────────────────────────────────────────────────────


def test_the_stand_in_export_is_shaped_like_infinigens(tmp_path):
    spec = ig.Spec.make("cabinet", (1.0, 0.6, 0.9), bottom=0.08)
    folder = _export(tmp_path, spec)
    rec = json.loads((folder / "result.json").read_text())
    xml = folder / rec["mjcf"]
    root = ET.parse(xml).getroot()
    assert root.find("compiler").get("meshdir") == "assets"
    assert [b.get("name") for b in root.find("worldbody")] == ["object"]
    links = [b.get("name") for b in root.iter("body")][1:]
    assert links[0] == "link_0" and all(n.startswith("link_") for n in links)
    groups = {g.get("group") for g in root.find("worldbody").iter("geom")}
    assert groups == {"0", "1"}
    assert all(t.get("file").startswith("assets/textures/") and t.get("file").endswith(
        "_DIFFUSE.png") for t in root.iter("texture"))
    assert {j.get("frictionloss") for j in root.iter("joint")} == {"1000"}
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.njnt >= 2 and model.ngeom > 10


def test_handles_are_welded_on_and_strays_dropped(tmp_path):
    asset = ig.clean(ig.load_asset(_export(tmp_path, ig.Spec.make("dishwasher", (0.6, 0.6, 0.8)))))
    assert asset.links[asset.root].parent is None and asset.links[asset.root].part == "base"
    kinds = sorted(lk.part for lk in asset.moving())
    assert kinds == ["door", "rack"]                     # the buttons are no joints any more
    door = next(lk for lk in asset.moving() if lk.part == "door")
    labels = [v.label for v in door.visuals]
    assert labels.count("buttons") == 1                  # the one outside the body is gone
    assert any("dropped" in n for n in asset.notes) and any("welded" in n for n in asset.notes)
    # the rack poked 5 cm through the top: now inside, and it slides no further than the tub
    rack = next(lk for lk in asset.moving() if lk.part == "rack")
    base_hi = ig._bounds(asset.links[asset.root].visuals)[1]
    assert ig._bounds(rack.visuals)[1][2] <= base_hi[2] + 1e-6
    depth = float(np.ptp([v.bounds()[i][0] for v in asset.links[asset.root].visuals
                          for i in (0, 1)]))
    assert rack.joint.range[1] == pytest.approx(ig.SLIDE_CLAMP * depth, rel=1e-3)
    window = ig.clean(ig.load_asset(_export(tmp_path, ig.window_spec(1.2, 1.0))))
    assert sorted(lk.part for lk in window.moving()) == ["sash", "sash"]   # the lock is welded
    assert all(v.glass for lk in window.moving() for v in lk.visuals)


def test_part_names_follow_the_handle_and_the_doors_leaf_keeps_it(tmp_path):
    cab = ig.clean(ig.load_asset(_export(tmp_path, ig.Spec.make("cabinet", (1.0, 0.6, 0.9)))))
    names = ig.part_names(cab, "Storage3")
    assert sorted(names.values()) == ["Storage3", "Storage3_door0", "Storage3_door1"]
    door = ig.clean(ig.load_asset(_export(tmp_path, ig.door_spec(0.9, 2.0))))
    names = ig.part_names(door, "Door0_frame", primary="Door0")
    assert sorted(names.values()) == ["Door0", "Door0_frame"]


# ── fitting, physics, look ───────────────────────────────────────────────────────────────────────


def test_the_asset_fills_the_box_with_its_front_to_minus_y(tmp_path):
    spec = ig.Spec.make("cabinet", (1.0, 0.6, 0.9), bottom=0.08)
    asset = ig.clean(ig.load_asset(_export(tmp_path, spec)))
    # a box 3 % off in every direction: the residual goes into the scale
    target = (1.03, 0.58, 0.93)
    built = ig.build(asset, "Cab0", ig.Placement((0.1, -0.2, 0.3), target),
                     ig.Overrides(mass=30.0))
    model, data = _compile(built, tmp_path / "scene")
    lo, hi = _drawn_box(model, data)
    main = np.array(target)
    centre = np.array([0.1, -0.2, 0.3])
    # handles stand proud of the front (-Y); everything else is inside the box
    assert np.allclose(hi, centre + main / 2, atol=2e-3)
    assert np.allclose(lo[[0, 2]], (centre - main / 2)[[0, 2]], atol=2e-3)
    assert lo[1] <= centre[1] - main[1] / 2 + 1e-3
    # the doors hang at the front: their hinges are on the -Y face, their axes vertical
    for j in range(model.njnt):
        anchor = data.xanchor[j]
        assert anchor[1] == pytest.approx(centre[1] - main[1] / 2, abs=0.03)
        assert abs(data.xaxis[j][2]) == pytest.approx(1.0)
    # opening swings the free edge out of the front (towards -Y)
    for j in range(model.njnt):
        b = int(model.jnt_bodyid[j])
        closed = data.xipos[b].copy()
        data.qpos[model.jnt_qposadr[j]] = 1.0
        mujoco.mj_forward(model, data)
        assert data.xipos[b][1] < closed[1] - 0.05
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)


def test_mass_is_our_estimate_shared_by_material_volume(tmp_path):
    asset = ig.clean(ig.load_asset(_export(tmp_path, ig.Spec.make("cabinet", (1.0, 0.6, 0.9)))))
    built = ig.build(asset, "Cab0", ig.Placement((0, 0, 0), (1.0, 0.6, 0.9)),
                     ig.Overrides(mass=34.0))
    model, _data = _compile(built, tmp_path / "scene")
    parts = built.report["parts"]
    total = sum(p["mass_kg"] for p in parts.values())
    assert total == pytest.approx(34.0, rel=1e-3)        # not Infinigen's metal densities
    vol = {n: p["material_volume_m3"] for n, p in parts.items()}
    for name, p in parts.items():
        assert p["mass_kg"] == pytest.approx(34.0 * vol[name] / sum(vol.values()), rel=1e-3)
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert float(model.body_mass[b]) == pytest.approx(p["mass_kg"], rel=1e-3)
    # a door leaf in its frame takes the door's whole estimate (the frame is the building's)
    door = ig.clean(ig.load_asset(_export(tmp_path, ig.door_spec(0.9, 2.0))))
    built = ig.build(door, "Door0", ig.Placement((0, 0, 0), (0.9, 0.1, 2.0), keep_depth=True,
                                                  front_y=-0.005),
                     ig.Overrides(mass=23.0, moving_mass=True), prefix="Door0_frame",
                     primary="Door0")
    assert built.report["parts"]["Door0"]["mass_kg"] == pytest.approx(23.0, rel=1e-3)


def test_joints_get_our_dynamics_and_a_flap_holds_itself(tmp_path):
    defaults = properties.JOINT_DEFAULTS
    asset = ig.clean(ig.load_asset(_export(tmp_path, ig.Spec.make("dishwasher", (0.6, 0.6, 0.8)))))
    built = ig.build(asset, "Dish0", ig.Placement((0, 0, 0), (0.6, 0.6, 0.8)),
                     ig.Overrides(mass=40.0))
    model, data = _compile(built, tmp_path / "scene")
    door = _body(model, "Dish0_door")
    rack = _body(model, "Dish0_rack")
    jd, jr = int(model.body_jntadr[door]), int(model.body_jntadr[rack])
    assert float(model.jnt_stiffness[jd]) == 0.0 and float(model.jnt_stiffness[jr]) == 0.0
    assert float(model.dof_damping[model.jnt_dofadr[jr]]) == defaults["prismatic"]["damping"]
    assert float(model.dof_frictionloss[model.jnt_dofadr[jr]]) == \
        defaults["prismatic"]["friction"]
    # the drop-down door: horizontal axis, friction just over its own weight's torque
    assert abs(data.xaxis[jd][2]) < 1e-6
    m = float(model.body_subtreemass[door])
    arm = data.xipos[door] - data.xanchor[jd]
    arm -= data.xaxis[jd] * (arm @ data.xaxis[jd])
    hold = float(model.dof_frictionloss[model.jnt_dofadr[jd]])
    assert hold == pytest.approx(ig.FLAP_HOLD * m * ig.GRAVITY * np.linalg.norm(arm), rel=0.15)
    # ...so left half open it stays there (MuJoCo's dry friction creeps; this one barely)
    data.qpos[model.jnt_qposadr[jd]] = 0.7
    for _ in range(2500):
        mujoco.mj_step(model, data)
    assert float(data.qpos[model.jnt_qposadr[jd]]) == pytest.approx(0.7, abs=0.005)


def test_the_texture_takes_the_photos_colour_and_keeps_its_shading(tmp_path):
    img = np.zeros((64, 64, 3), np.uint8)              # half the atlas unused (black)
    img[:, 32:] = 120
    img[10, 40] = 200                                   # some grain
    path = tmp_path / "x_DIFFUSE.png"
    Image.fromarray(img).save(path)
    out = np.asarray(Image.open(io.BytesIO(ig.tint_texture(path, (0.8, 0.2, 0.1)))), float)
    baked = out[:, 33:].reshape(-1, 3)
    assert np.allclose(np.median(baked, axis=0), (204, 51, 25.5), atol=2)
    assert out[10, 40, 0] > out[11, 40, 0]              # the grain is still there
    asset = ig.clean(ig.load_asset(_export(tmp_path, ig.window_spec(1.2, 1.0))))
    built = ig.build(asset, "Win0", ig.Placement((0, 0, 0), (1.2, 0.1, 1.0), keep_depth=True),
                     ig.Overrides(mass=5.0, tint=(0.9, 0.9, 0.9), moving_mass=True))
    glass = [el for el in built.assets if el.tag == "material" and el.get("rgba")
             and float(el.get("rgba").split()[3]) < 1]
    assert len(glass) == 2 and not any(el.get("texture") for el in glass)
    frames = [rel for rel in built.texture_files if "Frame" in rel]
    assert frames and all(rel.endswith(".e6e6e6.png") for rel in frames)   # tinted, the lock not
    assert any("LockLever" in rel and rel.endswith("_DIFFUSE.png") for rel in built.texture_files)


def test_an_open_door_starts_open_and_reads_as_open(tmp_path):
    door = ig.clean(ig.load_asset(_export(tmp_path, ig.door_spec(0.9, 2.0))))
    built = ig.build(door, "Door0", ig.Placement((0, 0, 0), (0.9, 0.1, 2.0), keep_depth=True,
                                                  front_y=-0.005),
                     ig.Overrides(mass=23.0, moving_mass=True, open_angle=math.radians(80)),
                     prefix="Door0_frame", primary="Door0")
    model, data = _compile(built, tmp_path / "scene")
    j = int(model.body_jntadr[_body(model, "Door0")])
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) == "Door0_hinge"
    assert float(data.qpos[model.jnt_qposadr[j]]) == pytest.approx(math.radians(80), abs=1e-6)
    # open, its leaf stands out into the room (-Y)
    assert data.xipos[_body(model, "Door0")][1] < -0.2


# ── runners and the cache ────────────────────────────────────────────────────────────────────────


FAKE_WORKER = '''
import json, sys
from pathlib import Path
from litereality_agent.models.articulated import infinigen as ig
jobs = json.loads(Path(sys.argv[1]).read_text())
for job in jobs["jobs"]:
    if job["factory"] == "oven":
        print("RESULT " + json.dumps({"key": job["key"], "ok": False, "error": "boom"}), flush=True)
        continue
    if job["factory"] == "window":
        sys.exit(3)                                   # dies before finishing it
    spec = ig.Spec.make(job["factory"], job["size"], seed=job["seed"], **job["options"])
    folder = ig.fake_export(spec, Path(sys.argv[2]) / job["key"])
    rec = json.loads((folder / "result.json").read_text())
    rec.pop("fake")                                   # as Infinigen's own worker writes it
    (folder / "result.json").write_text(json.dumps(rec))
    print("RESULT " + json.dumps({"key": job["key"], "ok": True, "seconds": 0.1}), flush=True)
'''


def _fake_python(tmp_path) -> Path:
    """A "python" for LocalRunner that runs a stand-in worker in THIS env (it ignores the real
    worker path it is given, as an Infinigen env would not)."""
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    exe = tmp_path / "python"
    exe.write_text(f"#!/bin/sh\nshift; shift\nexec {sys.executable} {worker} \"$@\"\n")
    exe.chmod(0o755)
    return exe


@pytest.mark.skipif(os.name == "nt", reason="a shell script stands in for the interpreter")
def test_the_local_runner_streams_results_and_reports_what_died(tmp_path):
    specs = [ig.Spec.make("cabinet", (1.0, 0.6, 0.9)), ig.Spec.make("door", (0.9, 0.1, 2.0)),
             ig.Spec.make("oven", (0.6, 0.6, 0.6)), ig.Spec.make("window", (1.2, 0.1, 1.0))]
    runner = ig.LocalRunner(_fake_python(tmp_path), workers=2)
    cache = ig.AssetCache(tmp_path / "cache")
    got = {r.spec.factory: r for r in ig.generate(specs, runner, cache)}
    assert got["cabinet"].ok and got["door"].ok
    assert got["cabinet"].folder.parent == cache.root          # moved into the cache whole
    assert "boom" in got["oven"].error
    assert not got["window"].ok and "exited" in got["window"].error
    # the second time it is all cached: nothing is run (this interpreter does not exist)
    again = list(ig.generate(specs[:2], ig.LocalRunner(tmp_path / "gone"), cache))
    assert all(r.cached and r.ok for r in again)


def test_stand_ins_never_pass_for_infinigen_in_the_cache(tmp_path):
    """A ``--fake`` run (or a test with $LR_ARTICULATED exported) must not leave boxes that a
    real run then installs as cached Infinigen furniture: the key is only what an asset is made
    of, and a 90 x 205 cm door is any room's door."""
    spec = ig.door_spec(0.9, 2.05)
    cache = ig.AssetCache(tmp_path / "cache")
    fake = list(ig.generate([spec], ig.FakeRunner(), cache))
    assert fake[0].ok and fake[0].folder.parent == cache.root / ig.FAKE_DIR
    real = list(ig.generate([spec], ig.LocalRunner(tmp_path / "gone"), cache))
    assert not real[0].cached and not real[0].ok          # it had to build, and could not
    # a stand-in already in the real place (the cache before this fix) is built over too
    poisoned = ig.fake_export(spec, tmp_path / "old")
    ig.AssetCache(cache.root).put(spec, poisoned)
    assert cache.get(spec) is None and cache.get(spec, fake=True) is not None
    real = list(ig.generate([spec], ig.LocalRunner(tmp_path / "gone"), cache))
    assert not real[0].cached
    # stand-ins are still cached for stand-ins
    runner = ig.FakeRunner()
    assert list(ig.generate([spec], runner, cache))[0].cached and runner.calls == []


# ── the Modal function ───────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def app():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("infinigen_deploy_under_test", APP_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_modal_app_checks_requests_and_pins_the_study(app):
    job = ig.Spec.make("cabinet", (1.0, 0.6, 0.9), bottom=0.08).job()
    parsed = app.parse_request({"job": job, "image_res": 256, "newer": 1})
    assert parsed["job"]["key"] == job["key"] and parsed["image_res"] == 256
    bad = [{}, {"job": {**job, "factory": "sofa"}}, {"job": {**job, "key": "../x"}},
           {"job": {**job, "size": [1, 2]}}, {"job": {**job, "size": [1, 2, 9]}},
           {"job": job, "image_res": 99999}, {"job": {**job, "options": {"x": [1]}}}]
    for request in bad:
        with pytest.raises(ValueError):
            app.parse_request(request)
    assert app.FACTORIES == infinigen_worker.FACTORIES
    assert app.INFINIGEN_COMMIT == ig.INFINIGEN_COMMIT
    assert app.APP_NAME == ig.MODAL_APP and app.FUNCTION_NAME == ig.MODAL_FUNCTION
    assert app.WORKER_LOCAL.resolve() == ig.WORKER


def test_the_modal_app_imports_inside_its_container():
    """Modal imports the deployed file again in every container, alone, as /root/app.py: a path
    worked out from the checkout must not fail there, or every asset built on Modal fails."""
    pytest.importorskip("modal")
    code = compile(APP_FILE.read_text(), "/root/app.py", "exec")
    namespace = {"__name__": "infinigen_in_container", "__file__": "/root/app.py"}
    exec(code, namespace)                                # noqa: S102 — our own deploy file
    assert namespace["FUNCTION_NAME"] == ig.MODAL_FUNCTION
    assert namespace["WORKER_LOCAL"].name == "infinigen_worker.py"


def test_the_modal_app_runs_the_worker_and_zips_its_export(app, tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    job = ig.Spec.make("cabinet", (1.0, 0.6, 0.9)).job()
    out = app.run_job({"job": job}, python=sys.executable, worker=str(worker))
    assert out["ok"] and out["key"] == job["key"]
    names = zipfile.ZipFile(io.BytesIO(base64.b64decode(out["zip_b64"]))).namelist()
    assert "result.json" in names and any(n.endswith("_DIFFUSE.png") for n in names)
    failed = app.run_job({"job": ig.Spec.make("window", (1, 0.1, 1)).job()},
                         python=sys.executable, worker=str(worker))
    assert not failed["ok"] and "no result" in failed["error"]
    assert not app.run_job({"job": {"key": "x"}})["ok"]


# ── into the room ────────────────────────────────────────────────────────────────────────────────


def _articulated_room(scan, tmp_path, runner=None, **kw):
    room = RealRoom(scan, tmp_path / "out")
    source = articulated.ArticulatedSource(runner or ig.FakeRunner(),
                                           ig.AssetCache(tmp_path / "cache"), log=lambda *_: None,
                                           **kw)
    generator = FakeGenerator()
    progress = generate_room(room, generator, passes=["final"], min_interval=float("inf"),
                             log=lambda *_: None, extras=[source])
    return room, source, generator, progress


def test_openable_objects_replace_their_templates_under_the_same_handles(scan, tmp_path):
    room, source, generator, progress = _articulated_room(scan, tmp_path)
    assert set(source.targets) == {"Dishwasher0", "Sink_Storage0", "Storage1", "Wall0_Door_0",
                                   "Wall2_Window_0"}
    # TRELLIS never generated what Infinigen took over
    assert not {h for h, _ in generator.calls} & set(source.targets)
    assert {h for h, s in progress.installed.items() if s == "infinigen"} == set(source.targets)
    root = ET.parse(room.scene_path).getroot()
    model = mujoco.MjModel.from_xml_path(str(room.scene_path))
    bodies = {b.get("name"): b for b in root.iter("body")}
    # the dishwasher: same body, its template gone but the counter's worktop kept, a plinth
    dish = bodies["Dishwasher0"]
    geoms = {g.get("name") for g in dish.findall("geom")}
    assert "Dishwasher0_worktop" in geoms and "Dishwasher0_plinth" in geoms
    assert "Dishwasher0_carcass" not in geoms
    assert {c.get("name") for c in dish.findall("body")} == {"Dishwasher0_door", "Dishwasher0_rack"}
    # the window's glass and bars are gone: its body holds the frame, the sashes slide
    world_geoms = {g.get("name") for g in root.find("worldbody").findall("geom")}
    assert not {g for g in world_geoms if g.startswith("Wall2_Window_0_")}
    window = _body(model, "Wall2_Window_0")
    sashes = [b for b in range(model.nbody) if model.body_parentid[b] == window]
    assert len(sashes) == 2 and all(
        model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_SLIDE for b in sashes)
    # the door's LEAF keeps the handle and the hinge its name, as the quick room's leaf did
    leaf = _body(model, "Wall0_Door_0")
    assert model.body_parentid[leaf] == _body(model, "Wall0_Door_0_frame")
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                             int(model.body_jntadr[leaf])) == "Wall0_Door_0_hinge"
    moving = movable_body_ids(model)
    assert leaf in moving and _body(model, "Dishwasher0_door") in moving
    assert _body(model, "Dishwasher0") not in moving          # the unit stays put
    report = json.loads((room.out_dir / "objects.json").read_text())
    assert set(report["replaced"]) == set(source.targets)
    assert report["replaced"]["Dishwasher0"]["factory"] == "dishwasher"


def test_the_room_starts_at_rest_and_its_doors_open(scan, tmp_path):
    room, _source, _gen, _progress = _articulated_room(scan, tmp_path)
    from litereality_agent.pipeline.quickroom.build import initial_overlaps

    model = mujoco.MjModel.from_xml_path(str(room.scene_path))
    assert initial_overlaps(model) == []
    sim = RoomSim(model)
    assert sim.unstable_resets == 0
    d = sim.data
    door = _body(model, "Sink_Storage0_door0")
    j = int(model.body_jntadr[door])
    adr = int(model.jnt_qposadr[j])
    q0 = float(d.qpos[adr])
    assert sim.grab("t", door, d.xipos[door].copy())
    for _ in range(90):                                 # pull straight out of the front
        g = sim._grabs["t"]
        anchor = d.xpos[door] + d.xmat[door].reshape(3, 3) @ g.local
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, d, jacp, None, anchor, door)
        v = jacp[:, model.jnt_dofadr[j]]
        sim.move("t", anchor + 0.15 * v / (np.linalg.norm(v) or 1))
        sim.step_realtime(1 / 60)
    assert abs(float(d.qpos[adr]) - q0) > math.radians(60)
    sim.release("t")
    for _ in range(300):
        sim.step_realtime(1 / 60)
    assert np.all(np.isfinite(d.qpos)) and sim.unstable_resets == 0


def test_a_live_swap_from_the_quick_room_carries_the_door(scan, tmp_path):
    room, _source, _gen, _progress = _articulated_room(scan, tmp_path)
    old = RoomSim(mujoco.MjModel.from_xml_path(str(room.base_path)))
    leaf = _body(old.model, "Wall0_Door_0")
    adr = int(old.model.jnt_qposadr[int(old.model.body_jntadr[leaf])])
    old.data.qpos[adr] = 0.5
    new = RoomSim(mujoco.MjModel.from_xml_path(str(room.scene_path)))
    report = new.adopt(old)
    assert "Wall0_Door_0" in report["carried"]
    nleaf = _body(new.model, "Wall0_Door_0")
    nadr = int(new.model.jnt_qposadr[int(new.model.body_jntadr[nleaf])])
    assert float(new.data.qpos[nadr]) == pytest.approx(0.5, abs=0.05)


def test_every_file_the_scene_names_is_published_with_it(scan, tmp_path):
    room, _source, _gen, _progress = _articulated_room(scan, tmp_path)
    files = scene_files(room.scene_path)
    paths = {str(rel) for _local, rel in files}
    assert any(p.startswith("articulated/") and p.endswith(".obj") for p in paths)
    assert any(p.startswith("articulated/") and p.endswith(".png") for p in paths)
    assert all(local.is_file() for local, _rel in files)


def test_a_failed_asset_keeps_its_template_and_the_rest_arrive(scan, tmp_path):
    room, _source, _gen, progress = _articulated_room(scan, tmp_path,
                                                      runner=ig.FakeRunner(fail=["cabinet"]))
    assert "Sink_Storage0" in room.errors and "Storage1" in room.errors
    root = ET.parse(room.scene_path).getroot()
    body = next(b for b in root.iter("body") if b.get("name") == "Sink_Storage0")
    assert {g.get("name") for g in body.findall("geom")} >= {"Sink_Storage0_carcass"}
    assert progress.installed["Dishwasher0"] == "infinigen"
    mujoco.MjModel.from_xml_path(str(room.scene_path))
    # ...given back to TRELLIS (this scan has no frames to crop, so they stay templates)
    assert not {"Sink_Storage0", "Storage1"} & room.taken
    assert "given back by infinigen" in progress.skipped["Sink_Storage0"]


def test_a_cabinet_flush_with_a_side_wall_is_moved_in_so_its_door_swings(tmp_path):
    objects = [o for o in OBJECTS if o[0] not in ("Storage0", "Sink0", "Chair1")]
    # in the corner of Wall1 (x = 5, its back) and Wall0 (y = 0, its side), facing -X
    objects.append(("Storage5", "Storage", (4.7, 0.6, 0.0), (1.2, 0.6, 0.9), -90.0))
    scan = tmp_path / "scans" / "corner"
    write_roomplan_usdz(scan / "room.usdz", objects=objects)
    room, source, _gen, _progress = _articulated_room(scan, tmp_path)
    assert "Storage5" in source.targets
    check = room.replaced["Storage5"]["check"]
    assert check.get("inset_for_swing_m")
    model = mujoco.MjModel.from_xml_path(str(room.scene_path))
    data = mujoco.MjData(model)
    for name in ("Storage5_door0", "Storage5_door1"):
        b = _body(model, name)
        j = int(model.body_jntadr[b])
        mujoco.mj_resetData(model, data)
        data.qpos[model.jnt_qposadr[j]] = math.copysign(math.pi / 2, model.jnt_range[j][1])
        mujoco.mj_forward(model, data)
        statics = [i for i in range(data.ncon)
                   if b in (model.geom_bodyid[data.contact[i].geom1],
                            model.geom_bodyid[data.contact[i].geom2])
                   and data.contact[i].dist < -1e-3
                   and model.body_weldid[model.geom_bodyid[data.contact[i].geom1]] == 0
                   and model.body_weldid[model.geom_bodyid[data.contact[i].geom2]] == 0]
        assert not statics, name


def test_a_box_whose_front_is_against_a_wall_is_turned_round(tmp_path):
    # Storage1 hangs on Wall1 (x = 5) facing into the room (yaw -90); here RoomPlan says +90
    objects = [(*o[:4], 90.0) if o[0] == "Storage1" else o for o in OBJECTS]
    scan = tmp_path / "scans" / "backwards"
    write_roomplan_usdz(scan / "room.usdz", objects=objects)
    room, source, _gen, _progress = _articulated_room(scan, tmp_path)
    assert source.targets["Storage1"].placement.turned
    assert not source.targets["Sink_Storage0"].placement.turned
    model = mujoco.MjModel.from_xml_path(str(room.scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name in ("Storage1_door0", "Storage1_door1"):
        j = int(model.body_jntadr[_body(model, name)])
        assert data.xanchor[j][0] < 5.0 - 0.3          # the hinges are on the room side
        b = int(model.jnt_bodyid[j])
        closed = data.xipos[b][0]
        data.qpos[model.jnt_qposadr[j]] = 1.0
        mujoco.mj_forward(model, data)
        assert data.xipos[b][0] < closed - 0.05        # and it opens into the room (-X)
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)


def test_cached_assets_are_installed_without_building(scan, tmp_path):
    _articulated_room(scan, tmp_path)
    runner = ig.FakeRunner()
    room, _source, _gen, progress = _articulated_room(scan, tmp_path, runner=runner)
    assert runner.calls == []
    assert all(room.replaced[h]["cached"] for h in room.replaced)


def _collider_boxes(model, data, body: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """World boxes of ``body``'s collider meshes."""
    out = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not model.geom_contype[g] or \
                model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] +
                            model.mesh_vertnum[mid]]
        w = data.geom_xpos[g] + v @ data.geom_xmat[g].reshape(3, 3).T
        out.append((w.min(0), w.max(0)))
    return out


def _pull(model, data, joint: int, force: float = 40.0, seconds: float = 1.0) -> float:
    """``force`` N on the joint's body along its opening direction; how far it went."""
    b = int(model.jnt_bodyid[joint])
    adr = int(model.jnt_qposadr[joint])
    q0 = float(data.qpos[adr])
    sign = 1.0 if abs(model.jnt_range[joint][1]) >= abs(model.jnt_range[joint][0]) else -1.0
    for _ in range(int(seconds / model.opt.timestep)):
        data.qfrc_applied[:] = 0
        p = data.xipos[b].copy()
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, p, b)
        v = sign * jacp[:, model.jnt_dofadr[joint]]
        if np.linalg.norm(v) > 1e-9:
            mujoco.mj_applyFT(model, data, force * v / np.linalg.norm(v), np.zeros(3), p, b,
                              data.qfrc_applied)
        mujoco.mj_step(model, data)
    return abs(float(data.qpos[adr]) - q0)


def test_parts_of_one_object_block_each_other_unless_they_touch_at_rest(tmp_path):
    """Only the pairs that meet at rest (a cabinet's two doors) stop colliding: a dishwasher's
    rack must not slide out through its closed door, and slides out once it is open."""
    asset = ig.clean(ig.load_asset(_export(tmp_path, ig.Spec.make("dishwasher", (0.6, 0.6, 0.8)))))
    built = ig.build(asset, "Dish0", ig.Placement((0, 0, 0), (0.6, 0.6, 0.8)),
                     ig.Overrides(mass=40.0))
    assert ("Dish0_door", "Dish0_rack") not in built.excludes
    model, data = _compile(built, tmp_path / "scene")
    rack = int(model.body_jntadr[_body(model, "Dish0_rack")])
    door = int(model.body_jntadr[_body(model, "Dish0_door")])
    span = float(model.jnt_range[rack][1])
    opened = float(model.jnt_range[door][1])
    model.jnt_range[door] = (0.0, 0.0)                      # latched shut
    assert _pull(model, data, rack) < 0.15 * span          # the closed door holds it in
    model.jnt_range[door] = (0.0, opened)
    mujoco.mj_resetData(model, data)
    data.qpos[model.jnt_qposadr[door]] = opened
    mujoco.mj_forward(model, data)
    assert _pull(model, data, rack) > 0.9 * span           # down, it rolls out over it
    # a cabinet's two doors meet in the middle: that pair alone does not collide
    asset = ig.clean(ig.load_asset(_export(tmp_path, ig.Spec.make("cabinet", (1.0, 0.6, 0.9)))))
    built = ig.build(asset, "Cab0", ig.Placement((0, 0, 0), (1.0, 0.6, 0.9)),
                     ig.Overrides(mass=34.0))
    assert built.report["contacts_off"] == ["Cab0_door0 <-> Cab0_door1"]
    assert set(built.excludes) == {("Cab0", "Cab0_door0"), ("Cab0", "Cab0_door1"),
                                   ("Cab0_door0", "Cab0_door1")}
    model, data = _compile(built, tmp_path / "scene2")
    assert all(-data.contact[i].dist < 1e-3 for i in range(data.ncon))


def test_a_chair_tucked_in_front_of_a_cabinet_is_moved_out_of_its_way(tmp_path):
    """The asset's handle stands proud of the scanned box, and the quick room leaves a tucked
    chair 2 mm from that box: the chair is moved, rather than the asset refused."""
    # Dishwasher0: centre x 0.3, 0.6 deep, facing +X: its front at x = 0.6; the chair's back at 0.61
    objects = [o for o in OBJECTS if o[0] != "Chair1"]
    objects.append(("Chair1", "Chair", (0.86, 3.2, 0.0), (0.5, 0.5, 0.85), 90.0))
    scan = tmp_path / "scans" / "tucked"
    write_roomplan_usdz(scan / "room.usdz", objects=objects)
    room, _source, _gen, progress = _articulated_room(scan, tmp_path)
    assert progress.installed["Dishwasher0"] == "infinigen", room.errors.get("Dishwasher0")
    moved = room.replaced["Dishwasher0"]["check"]["moved_out_of_its_way_m"]["Chair1"]
    assert 0.002 < moved[0] < 0.05 and abs(moved[1]) < 1e-3   # straight out of its front
    base = {b.get("name"): b for b in ET.parse(room.base_path).getroot().iter("body")}
    scene = {b.get("name"): b for b in ET.parse(room.scene_path).getroot().iter("body")}
    shift = (np.array([float(v) for v in scene["Chair1"].get("pos").split()])
             - np.array([float(v) for v in base["Chair1"].get("pos").split()]))
    assert shift == pytest.approx(moved, abs=1e-4)
    from litereality_agent.pipeline.quickroom.build import initial_overlaps

    model = mujoco.MjModel.from_xml_path(str(room.scene_path))
    assert initial_overlaps(model, 0.001) == []
    sim = RoomSim(model)
    assert sim.unstable_resets == 0


def test_a_doorway_has_no_curb_and_its_leaf_clears_the_floor(scan, tmp_path):
    """Infinigen's square frame has a bottom bar: it goes under the floor, without a collider,
    and the leaf hangs the quick room's clearance above the floor."""
    room, _source, _gen, _progress = _articulated_room(scan, tmp_path)
    assert any("bottom bar" in note for note in room.replaced["Wall0_Door_0"]["notes"])
    model = mujoco.MjModel.from_xml_path(str(room.scene_path))
    data = mujoco.MjData(model)
    data.qpos[:] = 0                                       # closed
    mujoco.mj_forward(model, data)
    floor = room.floor_z
    frame = _collider_boxes(model, data, _body(model, "Wall0_Door_0_frame"))
    assert frame and all(hi[2] > floor + 1.0 for _lo, hi in frame)     # jambs and the head
    leaf = _body(model, "Wall0_Door_0")
    bottom = min(lo[2] for lo, _hi in _collider_boxes(model, data, leaf))
    assert bottom == pytest.approx(floor + articulated.LEAF_CLEARANCE, abs=0.003)


def test_masses_in_the_scene_are_the_masses_reported(scan, tmp_path):
    """What compiles is what objects.json says: the plinth carries no mass, and a frame's mass
    (the building's, not in the estimate) is reported on its own."""
    room, _source, _gen, _progress = _articulated_room(scan, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(room.scene_path))
    root = ET.parse(room.scene_path).getroot()
    report = room.replaced["Dishwasher0"]
    top = next(g for g in root.iter("geom") if g.get("name") == "Dishwasher0_worktop")
    worktop = float(top.get("density", 1000)) * 8 * np.prod([float(v) for v in
                                                             top.get("size").split()])
    total = float(model.body_subtreemass[_body(model, "Dishwasher0")])
    assert total == pytest.approx(report["mass_kg"] + worktop, rel=1e-3)
    assert report["kept_mass_kg"] == pytest.approx(worktop, rel=1e-3)
    door = room.replaced["Wall0_Door_0"]
    frame = float(model.body_mass[_body(model, "Wall0_Door_0_frame")])
    assert frame == pytest.approx(door["static_mass_kg"], rel=1e-3)
    assert float(model.body_subtreemass[_body(model, "Wall0_Door_0")]) == \
        pytest.approx(door["mass_kg"], rel=1e-3)


class _Broken:
    """A generator whose every call fails at once (the TRELLIS app is not deployed)."""

    name = "trellis-modal"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def stream(self, images, passes):
        self.calls += list(images)
        raise RuntimeError("Lookup failed for Function 'trellis'")
        yield  # pragma: no cover


def test_a_generator_that_fails_whole_stops_only_its_own_objects(scan, tmp_path):
    """TRELLIS failing as a whole (its app not deployed) must not throw away the Infinigen
    assets still being built: the stream goes on to the last producer."""
    _frame_looking_down(scan)                              # Table0 is seen: TRELLIS is asked
    room = RealRoom(scan, tmp_path / "out")
    source = articulated.ArticulatedSource(ig.FakeRunner(delay=0.05),
                                           ig.AssetCache(tmp_path / "cache"), log=lambda *_: None)
    generator = _Broken()
    progress = generate_room(room, generator, passes=["final"], min_interval=float("inf"),
                             log=lambda *_: None, extras=[source])
    assert generator.calls == ["Table0"]
    assert {h for h, s in progress.installed.items() if s == "infinigen"} == set(source.targets)
    assert progress.versions == 1 and "Lookup failed" in progress.failed["Table0:final"]
    assert "Lookup failed" in room.errors["Table0"]


def test_an_object_its_source_fails_is_given_back_to_trellis(tmp_path):
    """Asking for more (--articulated) never leaves an object worse than --realistic alone."""
    scan = tmp_path / "scans" / "seen"
    write_roomplan_usdz(scan / "room.usdz")
    _frame_looking_down(scan)                              # the one object a frame sees: Table0

    class Claims:
        def __init__(self, whole: bool) -> None:
            self.name, self.whole = "claims", whole

        def plan(self, room):
            room.taken.add("Table0")
            return ["Table0"]

        def stream(self, room):
            if self.whole:
                raise RuntimeError("the source fell over")
            yield Arrival("Table0", self.name, None, 0.1, "no factory today")

    for whole in (False, True):
        room = RealRoom(scan, tmp_path / f"out{whole}")
        room.make_crops()
        generator = FakeGenerator()
        progress = generate_room(room, generator, passes=["final"], min_interval=float("inf"),
                                 log=lambda *_: None, extras=[Claims(whole)])
        assert generator.calls == [("Table0", "final")], whole
        assert progress.installed["Table0"] == "final" and "Table0" not in room.taken
        assert "Table0" in room.installed and "Table0" not in room.errors


# ── the CLI ──────────────────────────────────────────────────────────────────────────────────────


def test_cli_realroom_articulated_offline(scan, tmp_path, capsys):
    out = tmp_path / "objects"
    assert cli.main(["realroom", str(scan), "--out", str(out), "--fake", "--articulated",
                     "--no-textures"]) == 0
    text = capsys.readouterr().out
    assert "openable 5 with infinigen-fake" in text and "openable 5 from Infinigen" in text
    report = json.loads((out / "objects.json").read_text())
    assert set(report["replaced"]) >= {"Dishwasher0", "Wall0_Door_0"}


def test_cli_articulated_is_opt_in(scan, tmp_path, capsys, monkeypatch):
    out = tmp_path / "objects"
    assert cli.main(["realroom", str(scan), "--out", str(out), "--fake", "--no-textures"]) == 0
    assert json.loads((out / "objects.json").read_text())["replaced"] == {}
    monkeypatch.setenv("LR_ARTICULATED", "1")
    assert cli.main(["realroom", str(scan), "--out", str(out), "--fake", "--no-textures",
                     "--no-articulated"]) == 0
    assert json.loads((out / "objects.json").read_text())["replaced"] == {}
    args = cli._parser().parse_args(["realroom", str(scan), "--fake"])
    assert isinstance(cli._articulated_source(args).runner, ig.FakeRunner)
    args = cli._parser().parse_args(["roomcall", str(scan), "--realistic", "--articulated",
                                     "--drawers", "Storage1, Sink_Storage0"])
    monkeypatch.setattr(cli, "load_settings", lambda: type("S", (), {
        "modal_configured": lambda self: False})())
    with pytest.raises(SystemExit, match="needs Infinigen"):
        cli._articulated_source(args)


def test_cli_drawers_makes_a_storage_unit_a_drawer_chest(scan, tmp_path):
    args = cli._parser().parse_args(["realroom", str(scan), "--fake", "--articulated",
                                     "--drawers", "Sink_Storage0"])
    source = cli._articulated_source(args)
    room = RealRoom(scan, tmp_path / "out")
    source.plan(room)
    assert source.targets["Sink_Storage0"].spec.factory == "drawer"
    assert source.targets["Storage1"].spec.factory == "cabinet"      # hung: never drawers


# ── one real asset (opt-in) ──────────────────────────────────────────────────────────────────────


@pytest.mark.infinigen
def test_a_real_infinigen_cabinet_opens_in_mujoco(tmp_path):
    python = REAL_INFINIGEN
    if not python:
        pytest.skip(f"set ${ig.PYTHON_ENV} to an Infinigen env's python")
    spec = ig.Spec.make("cabinet", (0.5, 0.5, 0.5), bottom=0.08, shelves=1,
                        frame_material="white")
    results = list(ig.generate([spec], ig.LocalRunner(python, image_res=128),
                               ig.AssetCache(tmp_path / "cache")))
    assert results and results[0].ok, results[0].error if results else "nothing"
    asset = ig.clean(ig.load_asset(results[0].folder))
    built = ig.build(asset, "Cab0", ig.Placement((0, 0, 0), spec.size),
                     ig.Overrides(mass=6.9, tint=(0.8, 0.4, 0.2)))
    model, data = _compile(built, tmp_path / "scene")
    assert model.njnt == 1                               # one door (narrower than 0.55 m)
    lo, hi = _drawn_box(model, data)
    assert np.allclose(hi[[0, 2]], 0.25, atol=0.01) and hi[1] == pytest.approx(0.25, abs=0.01)
    door = built.report["parts"]["Cab0_door"]
    assert 0.5 < door["mass_kg"] < 3.0 and door["frictionloss"] < 1.0
    j = 0
    b = int(model.jnt_bodyid[j])
    for _ in range(1500):                                # 40 N at the free edge, 3 s
        data.qfrc_applied[:] = 0
        p = data.xipos[b]
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, p, b)
        v = jacp[:, 0] * math.copysign(1.0, model.jnt_range[j][1])
        mujoco.mj_applyFT(model, data, 40 * v / np.linalg.norm(v), np.zeros(3), p, b,
                          data.qfrc_applied)
        mujoco.mj_step(model, data)
    assert abs(float(data.qpos[0])) > math.radians(80)
