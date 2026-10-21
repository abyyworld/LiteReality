"""The hosted room call: which files a scene upload sends, the versioned Volume layout the
container reads back (and the order it is written in), the container's app (scene or demo) and
its watcher that swaps a newly published version into running rooms, the Modal deploy file, and
`roomcall --deploy` / `--publish`. Nothing here talks to Modal: the Volume is a stand-in that
records uploads."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest
from fastapi.testclient import TestClient

from litereality_agent.room_ops import roomcall
from litereality_agent.room_ops.roomcall import hosted, server

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
    """`modal.Volume` as far as `publish` uses it: `batch_upload` (committed when the batch
    exits, as Modal's is), `read_file`, `listdir` of the root and `remove_file`."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.batches: list[list[str]] = []   # the paths each committed batch wrote, in order
        self.mtimes: dict[str, int] = {}
        self.clock = 0

    def batch_upload(self, force: bool = False):
        volume = self

        class Batch:
            def __enter__(self):
                self.staged: dict[str, bytes] = {}
                return self

            def __exit__(self, *exc):
                if exc[0] is None:
                    volume.files.update(self.staged)
                    volume.batches.append(sorted(self.staged))
                    volume.clock += 1
                    for path in self.staged:
                        volume.mtimes[path.strip("/").split("/")[0]] = volume.clock
                return False

            def put_file(self, local, remote):
                data = local.read() if isinstance(local, io.BytesIO) else Path(local).read_bytes()
                self.staged[str(remote)] = data

        assert force, "a re-upload must overwrite the previous copy of the scene"
        return Batch()

    def read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    def listdir(self, path: str):
        assert path == "/"
        tops = {remote.strip("/").split("/")[0] for remote in self.files}
        return [SimpleNamespace(path=top, mtime=self.mtimes.get(top, 0)) for top in sorted(tops)]

    def remove_file(self, path: str, recursive: bool = False):
        assert recursive
        prefix = path.rstrip("/") + "/"
        for remote in [r for r in self.files if r.startswith(prefix)]:
            del self.files[remote]

    def folders(self) -> set[str]:
        return {remote.strip("/").split("/")[0] for remote in self.files if "/" in remote.strip("/")}

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


def test_the_container_serves_the_uploaded_scene(export, tmp_path, monkeypatch):
    volume = FakeVolume()
    published = hosted.publish(export, name="tea_room", volume=volume)
    folder = published.version
    assert re.fullmatch(r"tea_room-[0-9a-f]{12}", folder) and hosted._NAME.match(folder)
    assert published.sent > 0
    assert volume.files["/current"] == folder.encode()
    assert not [r for r in volume.files if "shake" in r or "Old0" in r]
    # The geometry came along, built here, so the container does not have to build it.
    assert {f"/{folder}/scene.glb", f"/{folder}/scene.json", f"/{folder}/roomcall.json"} \
        <= set(volume.files)
    meta = json.loads(volume.files[f"/{folder}/roomcall.json"])
    assert meta["version"] == folder and meta["label"] == "tea room" and meta["stage"] == "scene"
    assert meta["mujoco"] == mujoco.__version__
    root = volume.materialise(tmp_path / "volume")

    assert hosted.served_scene(root) == (root / folder / "scene.xml", folder)

    def no_building(*args, **kwargs):
        raise AssertionError("the prebuilt geometry should have been used")

    monkeypatch.setattr(server, "build_scene_assets", no_building)
    app = hosted.build_app(root)
    with TestClient(app) as client:
        manifest = client.get("/scene.json").json()
        assert manifest["label"] == "tea room" and manifest["v"] == folder
        names = {body["name"] for body in manifest["bodies"]}
        assert {"Chair0", "Mug0"} <= names  # the included file made it across too
        assert client.get("/" + manifest["glb"]).status_code == 200
        assert client.get("/scene.glb").status_code == 200
        assert "Content-Security-Policy" in client.get("/").headers
        assert client.get("/healthz").json()["version"] == folder


def test_a_version_folder_is_committed_before_current_names_it(export):
    volume = FakeVolume()
    folder = hosted.publish(export, name="tea_room", volume=volume).version
    folder_batch, current_batch = volume.batches
    assert current_batch == ["/current"]               # its own batch, after the folder's
    assert all(path.startswith(f"/{folder}/") for path in folder_batch)
    assert f"/{folder}/scene.xml" in folder_batch


def test_publishing_is_content_addressed_and_keeps_the_newest_two(export):
    volume = FakeVolume()
    first = hosted.publish(export, name="tea_room", volume=volume, prebuild=False).version
    assert hosted.publish(export, name="tea_room", volume=volume, prebuild=False).version == first
    versions = [first]
    for k in range(3):
        export.write_text(re.sub(r'mass="[0-9]+"', f'mass="{6 + k}"', export.read_text(), count=1))
        versions.append(hosted.publish(export, name="tea_room", volume=volume,
                                       prebuild=False).version)
    assert len(set(versions)) == 4                     # each edit is a new version
    assert volume.files["/current"] == versions[-1].encode()
    assert volume.folders() == {versions[-1], versions[-2]}
    # Another room's versions are never pruned by this one's.
    other = export.parent / "other.xml"
    other.write_text(export.read_text())
    office = hosted.publish(other, name="office", volume=volume, prebuild=False).version
    hosted.publish(export, name="tea_room", volume=volume, prebuild=False)
    assert office in volume.folders()


def test_the_same_files_as_another_stage_or_label_are_a_new_version(export):
    volume = FakeVolume()

    def publish(**kwargs) -> str:
        return hosted.publish(export, name="tea_room", volume=volume, prebuild=False,
                              **kwargs).version

    quick = publish(stage="quick")
    assert publish(stage="quick") == quick             # nothing new: the version it already is
    full = publish(stage="full")
    assert full != quick and volume.files["/current"] == full.encode()
    assert json.loads(volume.files[f"/{full}/roomcall.json"])["stage"] == "full"
    assert publish(stage="full", label="Tea room") not in {quick, full}


def test_overlapping_publishes_never_prune_what_current_names(export, tmp_path, monkeypatch):
    """The quick stage auto-published while the full one is published by hand: B's whole publish
    lands between A's upload and A's CURRENT. A still names its own folder last, so that folder
    must survive B's prune — and B's folder A's."""
    def variant(tag: str) -> Path:
        folder = tmp_path / tag
        for local, relative in hosted.scene_files(export):
            (folder / relative).parent.mkdir(parents=True, exist_ok=True)
            (folder / relative).write_bytes(local.read_bytes())
        (folder / "extra.xml").write_text(EXTRA.replace("0.3", tag))  # Mug0's mass: new content
        return folder / "scene.xml"

    volume = FakeVolume()
    live = hosted.publish(variant("0.31"), name="tea_room", volume=volume, prebuild=False).version
    real_read = hosted._read_current
    b: list[str] = []

    def read_current(vol):
        current = real_read(vol)
        if not b:                                      # A's first read, after its upload
            b.append("publishing")
            b[0] = hosted.publish(variant("0.33"), name="tea_room", volume=vol,
                                  prebuild=False).version
        return current

    monkeypatch.setattr(hosted, "_read_current", read_current)
    a = hosted.publish(variant("0.32"), name="tea_room", volume=volume, prebuild=False).version
    monkeypatch.setattr(hosted, "_read_current", real_read)
    assert volume.files["/current"] == a.encode()
    assert volume.folders() == {live, a, b[0]}         # CURRENT's folder is there, and B's too
    root = volume.materialise(tmp_path / "volume")
    assert asyncio.run(hosted.VolumeWatch(root).check()).id == a
    # Publishing one after the other again prunes back to the newest two.
    for tag in ("0.34", "0.35"):
        last = hosted.publish(variant(tag), name="tea_room", volume=volume, prebuild=False)
    assert len(volume.folders()) == 2 and last.version in volume.folders()


@pytest.mark.parametrize("current", ["tea_room-000000000000", "../etc", "a/b", ""])
def test_a_watcher_never_serves_the_demo_for_a_current_naming_nothing(tmp_path, current):
    """Only a CURRENT that is absent or says demo means the demo room to a running container:
    swapping every room into the demo over a dangling CURRENT would end everyone's call."""
    (tmp_path / "current").write_text(current)
    watch = hosted.VolumeWatch(tmp_path, scratch=tmp_path / "demo")
    with pytest.raises(ValueError, match="CURRENT"):
        asyncio.run(watch.check())
    assert hosted.served_scene(tmp_path) is None       # at startup it still means the demo
    for meaning_demo in (None, "demo"):
        (tmp_path / "current").unlink(missing_ok=True)
        if meaning_demo:
            (tmp_path / "current").write_text(meaning_demo)
        assert asyncio.run(watch.check()).stage == "demo"


def test_a_dangling_current_leaves_every_room_on_its_version(export, tmp_path):
    volume = FakeVolume()
    first = hosted.publish(export, name="tea_room", volume=volume).version
    root = volume.materialise(tmp_path / "volume")
    app = hosted.build_app(root, scratch=tmp_path / "demo", reload=lambda: None,
                           poll_seconds=0.05)
    hub = app.state.roomcall
    with TestClient(app) as client, contextlib.ExitStack() as stack:
        ws = stack.enter_context(client.websocket_connect("/ws?room=call&name=Ada"))
        ws.send_json({"t": "hello", "name": "Ada"})
        assert _recv(ws)["v"] == first
        (root / "current").write_text("tea_room-000000000000")    # pruned by another publish
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not hub.version_error:
            time.sleep(0.02)
        health = client.get("/healthz").json()
        assert "tea_room-000000000000" in health["version_error"] and health["version"] == first
        frames = [_recv(ws) for _ in range(10)]
        assert not [m for m in frames if m["t"] == "scene"]
        assert all(m["v"] == first for m in frames if m["t"] == "state")


def test_a_scene_naming_what_the_room_call_keeps_beside_it_is_refused(export):
    (export.parent / "scene.json").write_text("<mujoco/>")
    export.write_text(export.read_text().replace(
        '<include file="extra.xml"/>', '<include file="extra.xml"/><include file="scene.json"/>'))
    volume = FakeVolume()
    with pytest.raises(ValueError, match="keeps beside it: scene.json"):
        hosted.publish(export, name="tea_room", volume=volume)
    assert not volume.files


def test_prebuilt_geometry_is_trusted_only_when_it_fits(export, tmp_path):
    folder = tmp_path / "version"
    folder.mkdir()
    model = server.load_model(export)
    for name, data in server.prebuilt_files(export, version="v", label="tea room", stage="quick",
                                            model=model).items():
        (folder / name).write_bytes(data)
    assert server.read_prebuilt(folder, model) is not None
    other = mujoco.MjModel.from_xml_string(EXTRA)      # a model with other bodies
    assert server.read_prebuilt(folder, other) is None
    meta = json.loads((folder / "roomcall.json").read_text())
    for key, value in (("mujoco", "0.0.1"), ("geometry", "000000000000")):
        (folder / "roomcall.json").write_text(json.dumps({**meta, key: value}))
        assert server.read_prebuilt(folder, model) is None  # built by other code: rebuild
    assert server.read_prebuilt(tmp_path / "nothing", model) is None
    assert server.read_prebuilt(None, model) is None


def test_an_upload_from_before_versions_is_still_served(export, tmp_path):
    root = tmp_path / "volume"
    for local, relative in hosted.scene_files(export):
        target = root / "tea_room" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(local.read_bytes())
    (root / "current").write_text("tea_room")
    ref = hosted.served_ref(root)
    assert (ref.id, ref.label, ref.stage) == ("tea_room", "tea room", "scene")
    with TestClient(hosted.build_app(root)) as client:
        manifest = client.get("/scene.json").json()
        assert manifest["v"] == "tea_room" and manifest["label"] == "tea room"


def _recv(ws, timeout: float = 10.0) -> dict:
    import anyio

    async def get():
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()
    return json.loads(ws.portal.call(get)["text"])


def _until(ws, predicate, limit: int = 900) -> dict:
    for _ in range(limit):
        message = _recv(ws)
        if predicate(message):
            return message
    raise AssertionError("no matching frame")


def test_a_published_version_is_swapped_into_the_running_container(export, tmp_path,
                                                                   monkeypatch):
    monkeypatch.setattr(server, "PREFETCH_SECONDS", 0.2)
    volume = FakeVolume()
    first = hosted.publish(export, name="tea_room", volume=volume).version
    root = volume.materialise(tmp_path / "volume")
    reloads = []

    def reload():                                      # what Volume.reload does in a container
        reloads.append(time.monotonic())
        volume.materialise(root)
        if len(reloads) == 3:
            raise RuntimeError("there are open files preventing the operation: /scenes/x")

    app = hosted.build_app(root, reload=reload, poll_seconds=0.05)
    hub = app.state.roomcall
    with TestClient(app) as client, contextlib.ExitStack() as stack:
        ws = stack.enter_context(client.websocket_connect("/ws?room=call&name=Ada"))
        ws.send_json({"t": "hello", "name": "Ada"})
        welcome = _recv(ws)
        assert welcome["v"] == first
        chair = next(b["id"] for b in client.get("/scene.json").json()["bodies"]
                     if b["name"] == "Chair0")
        state = _until(ws, lambda m: m["t"] == "state")
        point = next(r for r in state["bodies"] if r[0] == chair)[1:4]
        ws.send_json({"t": "grab", "v": first, "body": chair, "point": point})
        _until(ws, lambda m: m["t"] == "state" and m["grabs"])
        _until(ws, lambda m: len(reloads) >= 4)
        assert hub.version_error is None               # an open-files round is just skipped

        # During the call, on the laptop: a better version of the room, with a new object first.
        export.write_text(export.read_text().replace(
            "<worldbody>", '<worldbody><body name="Lamp0" pos="-1 -1 0.3"><freejoint/>'
                           '<geom type="box" size="0.05 0.05 0.3" mass="2"/></body>', 1))
        second = hosted.publish(export, name="tea_room", volume=volume, stage="full").version
        scene = _until(ws, lambda m: m["t"] == "scene")
        assert scene == {"t": "scene", "v": second, "label": "tea room", "stage": "full"}
        swapped = _until(ws, lambda m: m["t"] == "state" and m["v"] == second)
        manifest = client.get("/scene.json").json()
        chair2 = next(b["id"] for b in manifest["bodies"] if b["name"] == "Chair0")
        assert chair2 != chair and [g[1] for g in swapped["grabs"]] == [chair2]
        assert manifest["stage"] == "full" and client.get("/healthz").json()["version"] == second


def test_a_watcher_round_that_fails_is_reported_and_the_next_one_carries_on(export, tmp_path):
    volume = FakeVolume()
    hosted.publish(export, name="tea_room", volume=volume)
    root = volume.materialise(tmp_path / "volume")
    fail = [True]

    def reload():
        if fail[0]:
            raise ConnectionError("modal.com unreachable")

    app = hosted.build_app(root, reload=reload, poll_seconds=0.05)
    hub = app.state.roomcall
    with TestClient(app) as client:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not client.get("/healthz").json()["version_error"]:
            time.sleep(0.02)
        assert "unreachable" in client.get("/healthz").json()["version_error"]
        fail[0] = False
        while time.monotonic() < deadline and client.get("/healthz").json()["version_error"]:
            time.sleep(0.02)
        assert client.get("/healthz").json()["version_error"] is None
        assert hub.last_check is not None


def test_switching_back_to_the_demo(export, tmp_path):
    volume = FakeVolume()
    hosted.publish(export, name="tea_room", volume=volume, prebuild=False)
    assert hosted.publish(None, name=hosted.DEMO, volume=volume).version == hosted.DEMO
    root = volume.materialise(tmp_path / "volume")
    assert hosted.served_scene(root) is None
    with TestClient(hosted.build_app(root, scratch=tmp_path / "demo")) as client:
        manifest = client.get("/scene.json").json()
        assert manifest["label"] == "demo room" and manifest["stage"] == "demo"
        assert manifest["v"].startswith("demo-room-")


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
    assert kw["redeploy"] is True and kw["stage"] == "full"  # mujoco_seed/ is the full room

    assert roomcall.run(_parse(["--demo", "--deploy"])) == 0
    assert calls[1][0] is None and calls[1][1]["environment"] == "main"
    assert calls[1][1]["stage"] == "demo"

    assert roomcall.run(_parse([str(export), "--publish", "--stage", "quick"])) == 0
    assert calls[2][1]["redeploy"] is False and calls[2][1]["stage"] == "quick"
    with pytest.raises(SystemExit, match="--watch"):
        roomcall.run(_parse([str(export), "--publish", "--watch"]))


def test_publish_without_redeploying(export, monkeypatch, tmp_path, capsys):
    modal = pytest.importorskip("modal")
    volume = FakeVolume()
    monkeypatch.setattr(modal.Volume, "from_name", lambda *a, **kw: volume)

    class Function:
        @staticmethod
        def from_name(*args, **kwargs):
            return SimpleNamespace(get_web_url=lambda: "https://x.modal.run")

    monkeypatch.setattr(modal, "Function", Function)

    def no_deploy(*args, **kwargs):
        raise AssertionError("--publish must not run `modal deploy`")

    monkeypatch.setattr(hosted.subprocess, "run", no_deploy)
    assert hosted.deploy(export, label="tea_room", repo_root=tmp_path, stage="quick",
                         redeploy=False) == 0
    current = volume.files["/current"].decode()
    assert current.startswith("tea_room-")
    assert json.loads(volume.files[f"/{current}/roomcall.json"])["stage"] == "quick"
    assert "room call updated  https://x.modal.run" in capsys.readouterr().out

    def not_deployed(*args, **kwargs):
        raise RuntimeError("app not found")

    Function.from_name = staticmethod(not_deployed)
    assert hosted.deploy(export, label="tea_room", repo_root=tmp_path, redeploy=False) == 1
    assert "--deploy" in capsys.readouterr().err
