"""The hosted room call: which files a scene upload sends, the Volume layout the container reads
back, the container's app (scene or demo), the Modal deploy file, and `roomcall --deploy`.
Nothing here talks to Modal: the Volume is a stand-in that records uploads."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from litereality_agent.room_ops import roomcall
from litereality_agent.room_ops.roomcall import hosted

REPO = Path(__file__).resolve().parents[1]

CUBE_OBJ = """v -0.1 -0.1 -0.1
v 0.1 -0.1 -0.1
v 0.1 0.1 -0.1
v -0.1 0.1 -0.1
v -0.1 -0.1 0.1
v 0.1 -0.1 0.1
v 0.1 0.1 0.1
v -0.1 0.1 0.1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""

SCENE = """<mujoco model="tea_room">
  <compiler angle="radian" meshdir="meshes" texturedir="meshes"/>
  <asset>
    <texture name="t_wood" type="2d" file="wood.png"/>
    <material name="m_wood" texture="t_wood"/>
    <mesh name="Chair0" file="Chair0.obj"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <body name="Chair0" pos="0 0 0.5">
      <freejoint/>
      <geom type="mesh" mesh="Chair0" material="m_wood" mass="5"/>
    </body>
  </worldbody>
  <include file="extra.xml"/>
</mujoco>
"""

EXTRA = """<mujoco>
  <worldbody>
    <body name="Mug0" pos="1 0 0.2"><freejoint/><geom type="box" size="0.04 0.04 0.05" mass="0.3"/></body>
  </worldbody>
</mujoco>
"""


def _png(path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (4, 4), (150, 100, 60)).save(path)


@pytest.fixture()
def export(tmp_path: Path) -> Path:
    """An exporter-shaped scene folder, with the clutter an export folder really holds."""
    folder = tmp_path / "run" / "tea_room" / "mujoco_seed"
    (folder / "meshes").mkdir(parents=True)
    (folder / "scene.xml").write_text(SCENE)
    (folder / "extra.xml").write_text(EXTRA)
    (folder / "meshes" / "Chair0.obj").write_text(CUBE_OBJ)
    _png(folder / "meshes" / "wood.png")
    (folder / "meshes" / "Old0.obj").write_text(CUBE_OBJ)  # an earlier export's leftover
    (folder / "shake.mp4").write_bytes(b"\0" * 1000)
    (folder / "shake_report.json").write_text("{}")
    return folder / "scene.xml"


class FakeVolume:
    """`modal.Volume.batch_upload` as far as `publish` uses it."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def batch_upload(self, force: bool = False):
        volume = self

        class Batch:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def put_file(self, local, remote):
                data = local.read() if isinstance(local, io.BytesIO) else Path(local).read_bytes()
                volume.files[str(remote)] = data

        assert force, "a re-upload must overwrite the previous copy of the scene"
        return Batch()

    def materialise(self, root: Path) -> Path:
        for remote, data in self.files.items():
            target = root / remote.lstrip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return root


def test_upload_sends_the_scene_and_only_what_it_names(export):
    relative = {str(rel) for _local, rel in hosted.scene_files(export)}
    assert relative == {"scene.xml", "extra.xml", "meshes/Chair0.obj", "meshes/wood.png"}


def test_a_scene_naming_a_missing_file_is_refused_before_upload(export):
    (export.parent / "meshes" / "wood.png").unlink()
    with pytest.raises(ValueError, match="wood.png"):
        hosted.scene_files(export)


def test_the_container_serves_the_uploaded_scene(export, tmp_path):
    volume = FakeVolume()
    sent = hosted.publish(export, name="tea_room", volume=volume)
    assert sent > 0
    assert volume.files["/current"] == b"tea_room"
    assert "/tea_room/shake.mp4" not in volume.files
    root = volume.materialise(tmp_path / "volume")

    assert hosted.served_scene(root) == (root / "tea_room" / "scene.xml", "tea_room")
    app = hosted.build_app(root)
    with TestClient(app) as client:
        manifest = client.get("/scene.json").json()
        assert manifest["label"] == "tea room"
        names = {body["name"] for body in manifest["bodies"]}
        assert {"Chair0", "Mug0"} <= names  # the included file made it across too
        assert client.get("/scene.glb").status_code == 200
        assert "Content-Security-Policy" in client.get("/").headers


def test_switching_back_to_the_demo(export, tmp_path):
    volume = FakeVolume()
    hosted.publish(export, name="tea_room", volume=volume)
    hosted.publish(None, name=hosted.DEMO, volume=volume)
    root = volume.materialise(tmp_path / "volume")
    assert hosted.served_scene(root) is None
    with TestClient(hosted.build_app(root, scratch=tmp_path / "demo")) as client:
        assert client.get("/scene.json").json()["label"] == "demo room"


@pytest.mark.parametrize("current", [None, "", "demo", "../etc", "a/b", "missing"])
def test_anything_but_a_real_uploaded_scene_means_the_demo(tmp_path, current):
    if current is not None:
        (tmp_path / "current").write_text(current)
    assert hosted.served_scene(tmp_path) is None


def test_scene_names_are_safe_folder_names():
    assert hosted.scene_name("tea_room") == "tea_room"
    assert hosted.scene_name("Tea room / v2") == "Tea-room-v2"
    assert hosted.scene_name("demo") == "room"  # never collides with the demo marker
    assert hosted.scene_name("../..") == "room"


def test_room_links_are_fresh_and_valid():
    a, b = hosted.room_link("https://x.modal.run/"), hosted.room_link("https://x.modal.run")
    assert a != b and a.startswith("https://x.modal.run/?room=")
    room = a.split("=", 1)[1]
    assert hosted._NAME.match(room)


def test_the_deploy_file_defines_one_web_function():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("roomcall_deploy", REPO / hosted.DEPLOY_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.app.name == hosted.APP_NAME
    assert hosted.FUNCTION_NAME in module.app.registered_functions
    assert (module.WEB / "index.html").is_file()


@pytest.mark.skipif(sys.version_info[:2] != (3, 11), reason="pins are the Python 3.11 resolution")
def test_the_image_pins_match_this_checkout():
    for pin in hosted.IMAGE_PACKAGES:
        name, version = pin.split("==")
        assert importlib.metadata.version(name) == version, f"{name}: lock moved, update the pin"


def _parse(argv):
    parser = argparse.ArgumentParser()
    roomcall.add_arguments(parser)
    return parser.parse_args(argv)


def test_deploy_flag_resolves_the_scene_and_label(export, monkeypatch):
    pytest.importorskip("modal")
    calls = []
    monkeypatch.setattr(hosted, "deploy", lambda scene, **kw: calls.append((scene, kw)) or 0)
    args = _parse([str(export.parents[1]), "--deploy", "--modal-env", "dev"])
    assert roomcall.run(args, credentials=("id", "secret")) == 0
    scene, kw = calls[0]
    assert scene == export and kw["label"] == "tea_room" and kw["environment"] == "dev"
    assert kw["credentials"] == ("id", "secret")

    assert roomcall.run(_parse(["--demo", "--deploy"])) == 0
    assert calls[1][0] is None and calls[1][1]["environment"] == "main"
