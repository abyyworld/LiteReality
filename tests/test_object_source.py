"""tea_room's Wall10_Door_0 shipped an `object.py` that ended at the OBJECT DEFINITION marker.

Valid Python, so everything accepted it: the builder checked `is_file()`, the progress board did
too, the Room package copied it, the seed bake ran it in Blender — which exited 0 having defined
the inlined helpers and built nothing — and `seed` reported COMPLETED without a preview. The
first sign was `simulate --from-seed` refusing to run. Every one of those places now applies the
same test, and the bundler refuses to write such a file in the first place.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from litereality_agent import object_source
from litereality_agent.object_source import EDIT_HEADER

SKILL = (Path(__file__).resolve().parents[1] / "src" / "litereality_agent" / "models"
         / "object_generation" / "articulated-glb-agent" / ".claude" / "skills"
         / "image-to-articulated-glb" / "scripts")

HELPERS = '''
def export_glb(path):
    import bpy
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", export_extras=True)


def add_box(name, dims, loc):
    return name
'''

# What tea_room's door looked like: the helpers, the marker, and nothing after it.
TRUNCATED = f'"""Wall10_Door_0."""\nimport bpy, os, sys\n{HELPERS}\n{EDIT_HEADER}\n'
COMPLETE = TRUNCATED + 'leaf = add_box("Door_leaf", (0.8, 0.04, 2.0), (0, 0, 1))\nexport_glb(OUT_GLB)\n'


def _bundler():
    spec = importlib.util.spec_from_file_location("make_selfcontained",
                                                  SKILL / "make_selfcontained.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ the test itself


def test_a_bundle_that_stops_at_the_marker_is_incomplete():
    problems = object_source.problems_in_text(TRUNCATED)
    assert problems and "OBJECT DEFINITION section is empty" in problems[0]


def test_a_complete_bundle_and_the_static_wrapper_pass():
    from litereality_agent.room_ops.export.export_room import STATIC_OBJECT_PY

    assert object_source.problems_in_text(COMPLETE) == []
    assert object_source.problems_in_text(STATIC_OBJECT_PY.replace("__NAME__", "Sofa0")) == []


@pytest.mark.parametrize("text, words", [
    ("", "empty"),
    (COMPLETE[: len(COMPLETE) - 30], "does not parse"),
    (TRUNCATED + "leaf = add_box('Door_leaf', (1, 1, 1), (0, 0, 0))\n", "never exports"),
    ("import bpy\nbpy.ops.import_scene.gltf(filepath='x.glb')\n", "never exports"),
])
def test_what_is_wrong_is_said_in_words(text, words):
    problems = object_source.problems_in_text(text)
    assert problems and words in problems[0]


def test_a_missing_file_is_a_problem_and_edits_are_seen(tmp_path):
    path = tmp_path / "object.py"
    assert object_source.problems(path) == ["no object.py"]
    path.write_text(TRUNCATED)
    assert not object_source.is_complete(path)
    path.write_text(COMPLETE)                        # cached on mtime+size, so a fix is noticed
    assert object_source.is_complete(path)


# ------------------------------------------------------------------ the producer


def test_the_bundler_refuses_a_recipe_with_no_build_code_and_keeps_the_last_good_file(tmp_path):
    bundler = _bundler()
    good = tmp_path / "object.py"
    good.write_text(COMPLETE)
    hollow = tmp_path / "build_Wall10_Door_0.py"
    hollow.write_text('"""Door recipe."""\nimport sys\nsys.path.insert(0, SKILL)\n'
                      "from blender_lib import (add_box,\n    export_glb)\n"
                      "TEXDIR = sys.argv[-2]\nOUT_GLB = sys.argv[-1]\n")
    lib = tmp_path / "blender_lib.py"
    lib.write_text('"""lib"""\nimport bpy\n' + HELPERS)

    with pytest.raises(bundler.BundleError, match="OBJECT DEFINITION section would be empty"):
        bundler.bundle(str(hollow), str(lib), "Wall10_Door_0")
    import sys

    argv = sys.argv
    try:
        sys.argv = ["make_selfcontained.py", str(hollow), str(good), "--lib", str(lib)]
        assert bundler.main() == 1
    finally:
        sys.argv = argv
    assert good.read_text() == COMPLETE


def test_the_bundler_writes_a_file_the_pipeline_accepts(tmp_path):
    bundler = _bundler()
    recipe = tmp_path / "build_Door.py"
    recipe.write_text('"""Door recipe."""\nimport sys\n'
                      "from blender_lib import add_box, export_glb\n"
                      "OUT_GLB = sys.argv[-1]\n"
                      "add_box('Door_leaf', (0.8, 0.04, 2.0), (0, 0, 1))\nexport_glb(OUT_GLB)\n")
    lib = tmp_path / "blender_lib.py"
    lib.write_text('"""lib"""\nimport bpy\n' + HELPERS)
    out = tmp_path / "object.py"
    bundler.write_bundle(str(out), bundler.bundle(str(recipe), str(lib), "Door"))
    assert object_source.problems(out) == []
    assert not Path(f"{out}.tmp").exists()


# ------------------------------------------------------------------ the consumers


def _object_dir(root: Path, name: str, source: str) -> Path:
    d = root / name
    (d / "previews").mkdir(parents=True)
    for i in range(2):
        (d / "previews" / f"view{i}.png").write_bytes(b"png")
    (d / f"{name}.glb").write_bytes(b"glTF")
    (d / "object.md").write_text("# door")
    (d / "object.py").write_text(source)
    return d


def test_the_builder_does_not_count_a_truncated_object_py_as_built(tmp_path, monkeypatch):
    from litereality_agent.models.object_generation import generate

    monkeypatch.setattr(generate, "probe_glb", lambda _glb: {"pass": True})
    d = _object_dir(tmp_path, "Wall10_Door_0", TRUNCATED)
    job = SimpleNamespace(glb=d / "Wall10_Door_0.glb", probe={})
    reasons = generate.verify_reasons(job)
    assert any("OBJECT DEFINITION" in r for r in reasons)
    assert not generate.verify(job)
    assert "object.py IS INCOMPLETE" in generate.object_py_feedback(reasons)

    (d / "object.py").write_text(COMPLETE)
    assert generate.verify_reasons(job) == []
    assert generate.verify(job)


def test_the_room_package_falls_back_to_the_built_glb_and_says_so(tmp_path, capsys):
    from litereality_agent.room_ops.export.export_room import recover_broken_sources

    _object_dir(tmp_path, "Wall10_Door_0", TRUNCATED)
    _object_dir(tmp_path, "Storage0", COMPLETE)
    broken_no_glb = _object_dir(tmp_path, "Window0", TRUNCATED)
    (broken_no_glb / "Window0.glb").unlink()
    manifest = {"assets": [{"object": n, "source": "procedural"}
                           for n in ("Wall10_Door_0", "Storage0", "Window0")]
                + [{"object": "Sofa0", "source": "trellis"}]}

    recovered = recover_broken_sources(manifest, tmp_path)

    assert list(recovered) == ["Wall10_Door_0"]
    by_name = {a["object"]: a for a in manifest["assets"]}
    assert by_name["Wall10_Door_0"]["source"] == "static"
    assert by_name["Wall10_Door_0"]["recovered_from"] == "procedural"
    assert by_name["Storage0"]["source"] == "procedural"
    assert by_name["Window0"]["source"] == "procedural"   # nothing to fall back on: fails by name
    out = capsys.readouterr().out
    assert "Wall10_Door_0" in out and "Window0" in out


def test_the_bake_refuses_a_truncated_object_py_before_running_blender(tmp_path, capsys):
    from litereality_agent.room_ops.compile.build_from_room import run_object_py

    d = _object_dir(tmp_path, "Wall10_Door_0", TRUNCATED)
    ok = run_object_py(d, "Wall10_Door_0", True, tmp_path / "out" / "Wall10_Door_0.glb", d,
                       blender=str(tmp_path / "no-such-blender"), regenerate=True)
    assert ok is False                     # and no FileNotFoundError: Blender was never started
    assert "object.py unusable" in capsys.readouterr().out


def test_seed_warns_when_the_room_glb_was_not_built(tmp_path, monkeypatch):
    import litereality_agent.room_ops as room_ops
    from litereality_agent.pipeline.scene_init import artifacts, paths, seed

    room = tmp_path / "Room"
    room.mkdir()
    (room / "Room.py").write_text("SHELL = {}\n")
    monkeypatch.setattr(room_ops, "export_scene", lambda scan: room)
    monkeypatch.setattr(room_ops, "compile_room", lambda *a, **k: None)
    monkeypatch.setattr(artifacts, "ensure_stage_links", lambda scan: None)
    monkeypatch.setattr(artifacts, "finalize",
                        lambda scan, capture_mode="link": SimpleNamespace(manifest=tmp_path / "m"))
    monkeypatch.setattr(paths, "set_scan", lambda scan: None)

    result = seed.run(SimpleNamespace(scan="tea_room"), {})
    assert any("seed preview FAILED" in w for w in result.warnings)


def test_the_progress_board_agrees_with_the_builder(tmp_path, monkeypatch):
    pytest.importorskip("cv2", reason="scene_init.flow needs the full preprocessing stack")
    import sys
    import types

    sys.modules.setdefault("open3d", types.ModuleType("open3d"))
    from litereality_agent.pipeline.scene_init import flow

    _object_dir(tmp_path, "Wall10_Door_0", TRUNCATED)
    _object_dir(tmp_path, "Storage0", COMPLETE)
    assert not flow._asset_is_built(tmp_path, "Wall10_Door_0")
    assert flow._asset_is_built(tmp_path, "Storage0")


def test_the_manifest_records_why_an_object_changed_kind(tmp_path):
    """The recovered asset still serialises: the extra keys are plain data."""
    from litereality_agent.room_ops.export.export_room import recover_broken_sources

    _object_dir(tmp_path, "Wall10_Door_0", TRUNCATED)
    manifest = {"assets": [{"object": "Wall10_Door_0", "source": "procedural"}]}
    recover_broken_sources(manifest, tmp_path)
    again = json.loads(json.dumps(manifest))
    assert again["assets"][0]["source_problems"]


def test_the_recipe_is_kept_while_object_py_cannot_replace_it(tmp_path):
    """`drop_build_recipe` deleted the agent's recipe as soon as an object.py EXISTED — so a
    truncated bundle took the only working definition with it."""
    from litereality_agent.models.object_generation.generate import drop_build_recipe

    (tmp_path / "build_Wall10_Door_0.py").write_text("export_glb(OUT_GLB)\n")
    (tmp_path / "object.py").write_text(TRUNCATED)
    assert drop_build_recipe(tmp_path) == []
    assert (tmp_path / "build_Wall10_Door_0.py").is_file()
    (tmp_path / "object.py").write_text(COMPLETE)
    assert drop_build_recipe(tmp_path) == ["build_Wall10_Door_0.py"]
