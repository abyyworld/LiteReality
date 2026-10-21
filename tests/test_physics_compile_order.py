"""tea_room's Dishwasher0 shipped with no physics: it was compiled before its GLB existed.

The procedural branch compiled every glb in `reconstruct/` the moment its agent finished, while the
TRELLIS branch — running in parallel — was still writing into the same folder; `Dishwasher0.glb`
(forced to TRELLIS) landed after the sweep and nothing ever looked again. A TRELLIS-only run never
compiled anything at all. Now each branch compiles its OWN objects once it has finished writing
them, a sweep after every branch catches the rest, and a glb still being written is refused.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

trimesh = pytest.importorskip("trimesh")

from sim_fixtures import box, write_glb  # noqa: E402

from litereality_agent.models.object_generation.sim import pipeline  # noqa: E402

# ------------------------------------------------------------------ the compile layer


def test_a_glb_still_being_written_is_refused_not_half_compiled(tmp_path):
    glb = write_glb(tmp_path / "Dishwasher0.glb", {"Body": box((0.6, 0.6, 0.85), (0, 0, 0.425))})
    whole = glb.read_bytes()
    assert pipeline.glb_problem(glb) is None
    glb.write_bytes(whole[: len(whole) // 2])
    assert "still being written" in pipeline.glb_problem(glb)
    row = pipeline.compile_object(glb)
    assert row["status"] == "error" and "incomplete GLB" in row["error"]
    assert not pipeline.sidecar_path(glb).exists()
    assert pipeline.glb_problem(tmp_path / "Absent0.glb") == "no glb"


@pytest.fixture
def fake_compile(monkeypatch):
    """compile_object without CoACD or MuJoCo: records what it saw and writes a sidecar."""
    calls: list[dict] = []
    lock = threading.Lock()

    def compile_object(glb, *, run_checks=True, decompose=True):
        glb = Path(glb)
        with lock:
            calls.append({"object": glb.stem, "exists": glb.is_file(),
                          "problem": pipeline.glb_problem(glb), "at": time.monotonic(),
                          "thread": threading.current_thread().name})
        out = pipeline.sidecar_path(glb)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return {"object": glb.stem, "status": "ok"}

    monkeypatch.setattr(pipeline, "compile_object", compile_object)
    return calls


def _glb(path: Path) -> Path:
    return write_glb(path, {"Body": box((0.4, 0.4, 0.4), (0, 0, 0.2))})


def test_compile_directory_compiles_only_the_named_objects_and_merges_the_report(
        tmp_path, fake_compile):
    _glb(tmp_path / "Table0" / "Table0.glb")
    _glb(tmp_path / "Dishwasher0.glb")
    _glb(tmp_path / "ChairCluster0.glb")

    pipeline.compile_directory(tmp_path, names=["Table0", "Missing0"])
    assert [c["object"] for c in fake_compile] == ["Table0"]
    pipeline.compile_directory(tmp_path, names=["Dishwasher0"])
    rows = json.loads((tmp_path / "sim_report.json").read_text())["rows"]
    assert sorted(r["object"] for r in rows) == ["Dishwasher0", "Table0"]   # merged, not replaced

    fake_compile.clear()
    summary = pipeline.compile_directory(tmp_path, only_stale=True)
    assert [c["object"] for c in fake_compile] == ["ChairCluster0"]
    assert summary["current"] == 2
    assert pipeline.missing_sidecars(tmp_path, None) == []

    time.sleep(0.01)
    _glb(tmp_path / "Dishwasher0.glb")                  # rebuilt: its sidecar is now stale
    assert pipeline.missing_sidecars(tmp_path, None) == ["Dishwasher0"]


# ------------------------------------------------------------------ the reconstruction phase


def _flow(monkeypatch, tmp_path):
    pytest.importorskip("cv2", reason="scene_init.flow needs the full preprocessing stack")
    sys.modules.setdefault("open3d", types.ModuleType("open3d"))
    from litereality_agent.pipeline.scene_init import flow
    from litereality_agent.pipeline.scene_init import paths as config

    monkeypatch.setenv("LITEREALITY_FINAL", str(tmp_path / "final"))
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "out"))
    config.set_scan("tea_room")
    routing = config.work_root() / "routing" / "routing_manifest.json"
    routing.parent.mkdir(parents=True, exist_ok=True)
    routing.write_text(json.dumps({"scans": {"tea_room": [
        {"name": "Dishwasher0", "route": "trellis"},          # LR_FORCE_TRELLIS=Dishwasher0
        {"name": "ChairCluster0", "route": "trellis"},
        {"name": "Table0", "route": "procedural"},
    ]}}))
    config.reconstruct_dir("tea_room").mkdir(parents=True, exist_ok=True)
    return flow, config


def _args(**overrides):
    base = dict(reconstruct=True, procedural=True, build_openings=False, chair_qc=False,
                trellis_python=None, seed=42, decimation=50000, force_reconstruct=False,
                dry_run_reconstruct=False, procedural_python=None, blender_path=None,
                concurrency=0, retries=1, budget=5.0, agent_model=None, qc_attempts=1,
                qc_include_warn=False, image_model=None)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("sequential", [False, True])
def test_every_object_is_compiled_after_its_own_glb_exists(tmp_path, monkeypatch, fake_compile,
                                                            sequential):
    flow, config = _flow(monkeypatch, tmp_path)
    recon = config.reconstruct_dir("tea_room")
    if sequential:
        monkeypatch.setenv("LR_SEQUENTIAL", "1")
    else:
        monkeypatch.delenv("LR_SEQUENTIAL", raising=False)
    written: dict[str, float] = {}

    def trellis(scan, **_kw):
        _glb(recon / "ChairCluster0.glb")
        time.sleep(0.8)                                    # the dishwasher lands LATE
        _glb(recon / "Dishwasher0.glb")
        written["Dishwasher0"] = time.monotonic()
        return {"status": "ok", "generated": 2}

    def agent(scan, *, openings=False, **_kw):
        _glb(recon / "Table0" / "Table0.glb")
        written["Table0"] = time.monotonic()
        return {"status": "ok", "generated": 1}

    monkeypatch.setattr(flow.reconstruct, "run_for_scan", trellis)
    monkeypatch.setattr(flow.reconstruct, "run_procedural_for_scan", agent)
    result: dict = {}
    flow._reconstruction_phase("tea_room", _args(), result, full=False)

    compiled = {c["object"]: c for c in fake_compile}
    assert set(compiled) == {"Dishwasher0", "ChairCluster0", "Table0"}
    assert len(fake_compile) == 3                          # each exactly once
    for name, call in compiled.items():
        assert call["exists"] and call["problem"] is None, name
    assert compiled["Dishwasher0"]["at"] >= written["Dishwasher0"]
    assert result["physics"].get("missing") is None
    assert result["procedural"]["sim"]["objects"] == 1     # its own object, nothing else


def test_a_trellis_only_run_compiles_too(tmp_path, monkeypatch, fake_compile):
    flow, config = _flow(monkeypatch, tmp_path)
    recon = config.reconstruct_dir("tea_room")

    def trellis(scan, **_kw):
        _glb(recon / "Dishwasher0.glb")
        return {"status": "ok", "generated": 1}

    monkeypatch.setenv("LR_SEQUENTIAL", "1")
    monkeypatch.setattr(flow.reconstruct, "run_for_scan", trellis)
    flow._reconstruction_phase("tea_room", _args(procedural=False), {}, full=False)
    assert [c["object"] for c in fake_compile] == ["Dishwasher0"]


def test_an_object_no_branch_claims_is_caught_by_the_sweep(tmp_path, monkeypatch, fake_compile):
    """An opening TRELLIS built because there was no routing, a glb left by an earlier run."""
    flow, config = _flow(monkeypatch, tmp_path)
    recon = config.reconstruct_dir("tea_room")
    _glb(recon / "Wall10_Door_0.glb")
    monkeypatch.setenv("LR_SEQUENTIAL", "1")
    monkeypatch.setattr(flow.reconstruct, "run_for_scan",
                        lambda scan, **_kw: {"status": "ok", "generated": 0})
    result: dict = {}
    flow._reconstruction_phase("tea_room", _args(procedural=False), result, full=False)
    assert [c["object"] for c in fake_compile] == ["Wall10_Door_0"]
    assert result["physics"]["objects"] == 1
