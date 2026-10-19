"""The `simulate` stage — the room becoming a MuJoCo scene, and saying what it had to invent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litereality_agent.pipeline import simulate
from litereality_agent.pipeline.context import RunContext
from litereality_agent.pipeline.result import StageStatus
from litereality_agent.settings import LiteRealitySettings


@pytest.fixture
def context(tmp_path: Path) -> RunContext:
    settings = LiteRealitySettings(repo_root=tmp_path, output_root=tmp_path / "run")
    return RunContext("scan", tmp_path / "capture", tmp_path / "run" / "scan",
                      tmp_path / "run", settings=settings)


def _built_room(context: RunContext) -> None:
    context.authored_room.mkdir(parents=True, exist_ok=True)
    (context.authored_room / "Room.py").write_text("SHELL = {}\n")
    context.preview_dir.mkdir(parents=True, exist_ok=True)
    (context.preview_dir / "Room.glb").write_bytes(b"glTF")
    (context.preview_dir / "room_layout.json").write_text('{"objects": []}')
    (context.preview_dir / "manifest.json").write_text('{"assets": []}')


def _exports(context: RunContext, report: dict):
    """Stand in for the exporter: write the scene and the report it would have written."""
    def fake(_context, _module, _args=(), *, log_name=None):
        out = simulate.scene_dir(context)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.xml").write_text("<mujoco/>")
        (out / "export_report.json").write_text(json.dumps(report))
        return 0, out / "log"
    return fake


def test_it_refuses_a_room_that_was_never_built(context, monkeypatch):
    """The export reads `Room.glb` and `room_layout.json`; without them it would fail deep inside
    trimesh with something that does not name the missing step."""
    context.authored_room.mkdir(parents=True, exist_ok=True)
    (context.authored_room / "Room.py").write_text("SHELL = {}\n")
    result = simulate.run(context, {})
    assert result.status is StageStatus.FAILED
    assert "publish" in result.error


def test_objects_without_compiled_physics_are_reported_not_buried(context, monkeypatch):
    """Falling back to the category table is a worse scene, not a broken one — but it IS the
    difference between physics that came from the assets and physics invented at export time."""
    _built_room(context)
    monkeypatch.setattr(simulate, "run_module",
                        _exports(context, {"loads_clean": True, "from_sidecar": ["Table0"],
                                           "no_sidecar": ["Mug0", "Cable0"],
                                           "placement_rejected": ["Chair1: colliders land 0.4 m"],
                                           "structure": 12, "free": 4, "attached": 2,
                                           "articulated": 1, "colliders": 40}))
    result = simulate.run(context, {})
    assert result.status is StageStatus.COMPLETED
    assert result.details["from_sidecar"] == 1
    joined = " ".join(result.warnings)
    assert "no compiled physics" in joined and "Mug0" in joined
    assert "rejected as mis-placed" in joined


def test_a_clean_export_warns_about_nothing(context, monkeypatch):
    _built_room(context)
    monkeypatch.setattr(simulate, "run_module",
                        _exports(context, {"loads_clean": True, "from_sidecar": ["Table0", "Chair0"],
                                           "structure": 12, "free": 4, "attached": 2,
                                           "articulated": 1, "colliders": 40}))
    result = simulate.run(context, {})
    assert result.status is StageStatus.COMPLETED
    assert result.warnings == []
    assert simulate.complete(context)


def test_the_shake_is_opt_in_because_it_is_a_measurement_and_costs_minutes(context, monkeypatch):
    _built_room(context)
    calls: list[str] = []

    def record(_context, module, args=(), *, log_name=None):
        calls.append(module)
        out = simulate.scene_dir(context)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.xml").write_text("<mujoco/>")
        (out / "export_report.json").write_text('{"loads_clean": true}')
        return 0, out / "log"

    monkeypatch.setattr(simulate, "run_module", record)
    simulate.run(context, {})
    assert calls == ["litereality_agent.room_ops.export.mujoco_scene"]

    calls.clear()
    simulate.run(context, {"shake": True})
    assert calls[-1] == "litereality_agent.room_ops.export.mujoco_shake"


def _seeded(context: RunContext) -> None:
    """A scan that has been through `seed` but never authored."""
    context.seed_room.mkdir(parents=True, exist_ok=True)
    (context.seed_room / "Room.py").write_text("SHELL = {}\n")
    preview = context.seed_room.parent / "room_preview"
    preview.mkdir(parents=True, exist_ok=True)
    (preview / "Room.glb").write_bytes(b"glTF")
    (preview / "room_layout.json").write_text('{"objects": []}')
    (preview / "manifest.json").write_text('{"assets": []}')


def test_from_seed_exports_the_unauthored_room(context, monkeypatch):
    """Authoring adds materials, wall fixtures and clutter — appearance. The physical scene is
    complete without it: the shell with its openings cut out, and every reconstructed object in its
    measured box carrying the physics its own build compiled."""
    _seeded(context)
    seen: list[list] = []

    def record(_context, module, args=(), *, log_name=None):
        seen.append([str(a) for a in args])
        out = simulate.scene_dir(context, seed=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.xml").write_text("<mujoco/>")
        (out / "export_report.json").write_text('{"loads_clean": true, "from_sidecar": ["Table0"]}')
        return 0, out / "log"

    monkeypatch.setattr(simulate, "run_module", record)
    result = simulate.run(context, {"from_seed": True})

    assert result.status is StageStatus.COMPLETED
    assert result.details["source"] == "seed"
    assert str(context.seed_room) in seen[0]
    # ...and it goes somewhere of its own, so an authored export is not overwritten by one that
    # deliberately has no authoring in it.
    assert result.details["scene"].endswith("mujoco_seed/scene.xml")


def test_from_seed_asks_for_seed_rather_than_author_when_the_room_is_missing(context):
    result = simulate.run(context, {"from_seed": True})
    assert result.status is StageStatus.FAILED
    assert "seed" in result.error and "author" not in result.error


@pytest.mark.parametrize('report', [{}, {'loads_clean': False},
    {'loads_clean': None, 'load_check_error': 'coplanar mesh'}])
def test_unloadable_unchecked_or_interpenetrating_scene_fails(context, monkeypatch, report):
    _built_room(context)
    monkeypatch.setattr(simulate, 'run_module', _exports(context, report))
    result = simulate.run(context, {})
    assert result.status is StageStatus.FAILED
    assert not simulate.complete(context)


def test_old_load_verdict_is_not_reused_for_new_export(context, monkeypatch):
    _built_room(context)
    _exports(context, {'loads_clean': True})(None, None)
    def no_report(*args, **kw):
        return 0, 'export.log'
    monkeypatch.setattr(simulate, 'run_module', no_report)
    assert simulate.run(context, {}).status is StageStatus.FAILED


def test_requested_shake_failure_fails_stage(context, monkeypatch):
    _built_room(context)
    export = _exports(context, {'loads_clean': True})
    def run(ctx, module, args, **kwargs):
        if module.endswith('mujoco_shake'):
            return 1, 'shake.log'
        return export(ctx, module, args, **kwargs)
    monkeypatch.setattr(simulate, 'run_module', run)
    assert simulate.run(context, {'shake': True}).status is StageStatus.FAILED


def test_changed_simulator_mesh_invalidates_acceptance(context, monkeypatch):
    _built_room(context)
    monkeypatch.setattr(simulate, 'run_module', _exports(context, {'loads_clean': True}))
    assert simulate.run(context, {}).ok
    assert simulate.complete(context)
    (simulate.scene_dir(context) / 'body.obj').write_text('v 0 0 0')
    assert not simulate.complete(context)


def test_changed_room_invalidates_simulation_acceptance(context, monkeypatch):
    _built_room(context)
    monkeypatch.setattr(simulate, 'run_module', _exports(context, {'loads_clean': True}))
    assert simulate.run(context, {}).ok
    (context.authored_room / 'Room.py').write_text('SHELL = {"changed": True}')
    assert not simulate.complete(context)
