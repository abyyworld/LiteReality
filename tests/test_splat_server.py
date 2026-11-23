"""The photoreal room in a call, server side: which bundle each room's version finds, the
watcher that tells rooms a bundle came or changed, and the splat child a scan runs beside its
objects (room_ops/roomcall/server, splats, scans). Offline: the scans run the stand-in CLI of
`test_scan_upload`."""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

import mujoco
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_scan_upload as upload  # noqa: E402
from test_scan_upload import (  # noqa: E402
    DEMO,
    _geometry_here,
    children,
    gone,
    join,
    make_zip,
    post,
    scan_entries,
    scan_msg,
    until,
    wait_for,
)

from litereality_agent.room_ops.roomcall import scans, server  # noqa: E402
from litereality_agent.room_ops.roomcall.splats import (  # noqa: E402
    NO_NAME,
    SplatSource,
    room_name,
    scan_name,
)

SPZ = "room-0123456789abcdef.spz"
web_dir = upload.web_dir          # the fixtures, shared
fake_cli = upload.fake_cli


def write_bundle(folder: Path, created: float, content: bytes = b"splats") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / SPZ).write_bytes(content)
    (folder / "splat.json").write_text(json.dumps({"room": SPZ, "bodies": {}, "stage": "instant",
                                                   "created": created}))


@contextlib.contextmanager
def splat_client(web_dir, tmp_path, monkeypatch, *, label: str, splats, fake_cli=None,
                 **config):
    monkeypatch.setattr(server, "OFFER_SLACK", 0.05)
    monkeypatch.setattr(server, "build_assets", _geometry_here)
    options = {"mode": "off"}
    if fake_cli is not None:
        options = {"mode": "quick", "fake": True, "scratch": tmp_path / "scratch",
                   "watch_seconds": 0.05, "keep_seconds": 0, "command": fake_cli.command,
                   "splats_root": tmp_path / "splats", **config}
    app = server.create_app(model=mujoco.MjModel.from_xml_string(DEMO), web_dir=web_dir,
                            label=label, settle_seconds=0.02, prefetch_seconds=0.1,
                            scans=scans.ScanConfig(**options), splats=splats)
    with TestClient(app) as client:
        yield client, app.state.roomcall


def test_the_first_bundle_a_room_gets_is_told_to_it(web_dir, tmp_path, monkeypatch):
    """Browsers look for a bundle when a version goes in and again only when told: a room whose
    bundle comes after that (a scan's instant room, a first publish) must hear of it."""
    root = tmp_path / "volume"
    source = SplatSource.per_room(root, poll_seconds=0.05)
    with splat_client(web_dir, tmp_path, monkeypatch, label="tea room", splats=source) as (
            client, hub), contextlib.ExitStack() as stack:
        ws, welcome = join(client, stack, "r", "Ada")
        name = room_name(welcome["v"])
        assert client.get("/splat.json", params={"v": welcome["v"]}).status_code == 404
        wait_for(lambda: ("r", name) in hub._splat_stamps)          # it has looked once
        write_bundle(root / name, created=1.0)
        told = until(ws, lambda m: m["t"] == "splat")
        assert told == {"t": "splat", "v": welcome["v"], "created": 1.0}
        write_bundle(root / name, created=2.0)                       # and every change after
        assert until(ws, lambda m: m["t"] == "splat")["created"] == 2.0


def test_a_room_called_room_has_its_files_served(web_dir, tmp_path, monkeypatch):
    root = tmp_path / "volume"
    write_bundle(root / "room", created=1.0)
    source = SplatSource.per_room(root, poll_seconds=0.05)
    with splat_client(web_dir, tmp_path, monkeypatch, label="room", splats=source) as (
            client, hub):
        assert room_name(hub.version.id) == "room"
        manifest = client.get("/splat.json").json()
        assert manifest["room"] == f"splat/room/{SPZ}"
        got = client.get("/" + manifest["room"])
        assert got.status_code == 200 and got.content == b"splats"


def test_a_version_with_no_room_name_gets_a_url_name_no_room_has(tmp_path):
    write_bundle(tmp_path / "fixed", created=1.0, content=b"fixed")
    app = server.create_app(model=mujoco.MjModel.from_xml_string(DEMO), web_dir=None,
                            version="scene", splats=SplatSource.fixed(tmp_path / "fixed"))
    with TestClient(app) as client:
        manifest = client.get("/splat.json").json()
        assert manifest["room"] == f"splat/{NO_NAME}/{SPZ}"
        assert client.get("/" + manifest["room"]).content == b"fixed"
    assert room_name(f"{NO_NAME}-0123456789ab") is None


def test_a_fixed_splat_folder_is_the_served_rooms_only(tmp_path):
    """`roomcall SCENE --splat DIR --uploads ...`: a room scanned during the call is drawn with
    its own bundle, not the served room's."""
    write_bundle(tmp_path / "fixed", created=1.0, content=b"tea room")
    write_bundle(tmp_path / "splats" / "scan-s1", created=2.0, content=b"scan")
    chain = SplatSource.chain(SplatSource.fixed(tmp_path / "fixed", name="tea_room"),
                              SplatSource.per_room(tmp_path / "splats"))
    assert chain.file("tea_room", SPZ) == b"tea room"
    assert chain.file("scan-s1", SPZ) == b"scan"
    assert chain.bundle("other") is None
    assert SplatSource.fixed(tmp_path / "fixed").file("anything", SPZ) == b"tea room"


def test_each_scan_has_its_own_bundle_which_goes_with_its_room(web_dir, tmp_path, fake_cli,
                                                                monkeypatch):
    """Two rooms' scans of one label, labelled like the deployed room too: each room is drawn
    with its own scan's splats, the deployed room keeps its own, and a room's scan bundle is
    deleted once the room is let go."""
    volume = tmp_path / "volume"
    write_bundle(volume / "My-Room", created=1.0, content=b"deployed")
    source = SplatSource.chain(SplatSource.per_room(tmp_path / "splats", poll_seconds=0.05),
                               SplatSource.per_room(volume, poll_seconds=0.05))
    with splat_client(web_dir, tmp_path, monkeypatch, label="My Room", splats=source,
                      fake_cli=fake_cli) as (client, hub):
        live = hub.version.id
        assert room_name(live) == "My-Room"
        got = {}
        with contextlib.ExitStack() as stack:
            for room in ("a", "b"):
                ws, welcome = join(client, stack, room, "Ada")
                job = post(client, room, welcome,
                           make_zip(scan_entries(tmp_path))).json()["job"]
                seen: list = []
                told = until(ws, lambda m: m["t"] == "splat", seen=seen)   # its instant room
                if not any(scan_msg("done", job)(m) for m in seen):
                    until(ws, scan_msg("done", job))
                manifest = client.get("/splat.json", params={"v": told["v"]}).json()
                assert manifest["name"] == scan_name(job)
                got[room] = client.get("/" + manifest["room"]).content
            assert got == {"a": b"splats of My Room", "b": b"splats of My Room"}
            folders = sorted(p.name for p in (tmp_path / "splats").iterdir())
            assert folders == sorted(scan_name(j.id) for j in hub.scans.jobs.values())
            deployed = client.get("/splat.json", params={"v": live}).json()
            assert deployed["name"] == "My-Room"
            assert client.get("/" + deployed["room"]).content == b"deployed"
        wait_for(lambda: not hub.room_versions)
        wait_for(lambda: not any((tmp_path / "splats").iterdir()))


def test_a_rooms_next_scan_drops_the_bundle_of_the_one_before(web_dir, tmp_path, fake_cli,
                                                             monkeypatch):
    source = SplatSource.per_room(tmp_path / "splats", poll_seconds=0.05)
    with splat_client(web_dir, tmp_path, monkeypatch, label="lobby", splats=source,
                      fake_cli=fake_cli, cooldown=0) as (client, hub), \
            contextlib.ExitStack() as stack:
        ws, welcome = join(client, stack, "r", "Ada")
        jobs = []
        for _ in range(2):
            job = post(client, "r", welcome, make_zip(scan_entries(tmp_path))).json()["job"]
            until(ws, scan_msg("done", job))
            jobs.append(job)
        assert [p.name for p in (tmp_path / "splats").iterdir()] == [scan_name(jobs[1])]


def test_the_splat_child_is_timed_out_alone_and_never_outlives_its_build(web_dir, tmp_path,
                                                                         fake_cli, monkeypatch):
    """The splat child runs beside the objects one: its timeout kills it, not the objects;
    and a build that ends early (the objects failed) takes it along."""
    monkeypatch.setenv("SCAN_FAKE_SPLAT", "hang")
    monkeypatch.setenv("SCAN_FAKE_STEP", "1.0")
    source = SplatSource.per_room(tmp_path / "splats", poll_seconds=0.05)
    with splat_client(web_dir, tmp_path, monkeypatch, label="lobby", splats=source,
                      fake_cli=fake_cli, mode="realistic", splat_timeout=0.8,
                      cooldown=0) as (client, hub), contextlib.ExitStack() as stack:
        ws, welcome = join(client, stack, "r", "Ada")
        job = post(client, "r", welcome, make_zip(scan_entries(tmp_path))).json()["job"]
        done = until(ws, lambda m: m["t"] == "scan" and m["job"] == job
                     and m["status"] in ("done", "failed"), seconds=30)
        assert done["status"] == "done" and "2 realistic objects" in done["detail"], done
        splat_pid = children(fake_cli, "splatroom")[0]["pid"]
        wait_for(lambda: gone(splat_pid))

        hub.scans.config.splat_timeout = 600.0
        monkeypatch.setenv("SCAN_FAKE_REAL", "fail")
        monkeypatch.setenv("SCAN_FAKE_VERSIONS", "1")
        monkeypatch.setenv("SCAN_FAKE_STEP", "0.3")
        job = post(client, "r", welcome, make_zip(scan_entries(tmp_path))).json()["job"]
        failed = until(ws, scan_msg("failed", job), seconds=30)
        assert "TRELLIS went away" in failed["detail"]
        splat_pid = next(c["pid"] for c in children(fake_cli, "splatroom")
                         if c["pid"] != splat_pid)
        started = time.monotonic()
        wait_for(lambda: gone(splat_pid))
        assert time.monotonic() - started < 5
        wait_for(lambda: not hub.scans.jobs[job].procs)


def test_the_manifest_carries_the_view_and_the_capture_spots(web_dir, tmp_path, monkeypatch):
    """What the browser's photo mode reads: where to stand first and where the phone was (the
    walk zone), both as the pipeline wrote them."""
    root = tmp_path / "volume"
    write_bundle(root / "room", created=1.0)
    raw = json.loads((root / "room" / "splat.json").read_text())
    raw.update(view={"pos": [0.5, 0.25, 1.4], "yaw": 1.0, "pitch": 0.0, "frame": 3},
               capture=[[0.5, 0.25, 1.4], [1.0, -0.5, 1.38]])
    (root / "room" / "splat.json").write_text(json.dumps(raw))
    source = SplatSource.per_room(root, poll_seconds=0.05)
    with splat_client(web_dir, tmp_path, monkeypatch, label="room", splats=source) as (
            client, _hub):
        manifest = client.get("/splat.json").json()
    assert manifest["view"] == {"pos": [0.5, 0.25, 1.4], "yaw": 1.0, "pitch": 0.0}
    assert manifest["capture"] == [[0.5, 0.25, 1.4], [1.0, -0.5, 1.38]]
