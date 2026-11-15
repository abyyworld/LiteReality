"""The room picker, server side: the library of ready rooms (room_ops/roomcall/library), the
{"t": "pick"} message that moves one call room into one of them or back to the default room, and
GET /rooms.json and /rooms/<name>/thumb.jpg (room_ops/roomcall/server). Offline: the library is a
folder shaped like the Volume's, and scans run the stand-in CLI of `test_scan_upload`."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import mujoco
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_scan_upload as upload  # noqa: E402
from test_scan_upload import (  # noqa: E402
    DEMO,
    _geometry_here,
    join,
    make_zip,
    post,
    recv,
    scan_entries,
    scan_msg,
    until,
    wait_for,
)
from test_splat_server import write_bundle  # noqa: E402

from litereality_agent.room_ops.roomcall import scans, server  # noqa: E402
from litereality_agent.room_ops.roomcall.library import (  # noqa: E402
    LIBRARY_DIR,
    LIBRARY_FILE,
    LibraryRoom,
    RoomLibrary,
)
from litereality_agent.room_ops.roomcall.splats import SplatSource, room_name  # noqa: E402

web_dir = upload.web_dir          # the fixtures, shared
fake_cli = upload.fake_cli

OFFICE = "office_room-0123456789ab"
LOFT = "loft-0123456789ab"
JPEG = b"\xff\xd8\xff\xe0thumbnail\xff\xd9"


def add_version(root: Path, folder: str, *, size: str = "0.12", stage: str | None = None) -> Path:
    """A version folder like `hosted.publish` writes: scene.xml (and roomcall.json)."""
    out = root / folder
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.xml").write_text(DEMO.replace('size="0.1"', f'size="{size}"'))
    if stage is not None:
        (out / "roomcall.json").write_text(json.dumps({"version": folder, "label": folder,
                                                       "stage": stage}))
    return out


def write_library(root: Path, rooms: list[dict] | str) -> Path:
    path = root / LIBRARY_DIR / LIBRARY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rooms if isinstance(rooms, str) else json.dumps({"rooms": rooms}))
    return path


def volume(tmp_path: Path) -> Path:
    """The Volume: office_room (with its photoreal bundle and a thumbnail) and loft."""
    root = tmp_path / "volume"
    add_version(root, OFFICE, stage="objects")
    add_version(root, LOFT, size="0.15")
    write_bundle(root / "splats" / "office_room", created=1.0, content=b"office")
    write_library(root, [{"name": "office_room", "label": "Office room", "version": OFFICE,
                          "thumb": True},
                         {"name": "loft", "label": "Loft", "version": LOFT}])
    (root / LIBRARY_DIR / "office_room.jpg").write_bytes(JPEG)
    return root


@contextlib.contextmanager
def picker_client(web_dir, tmp_path, monkeypatch, *, root: Path | None = None, fake_cli=None,
                  prefetch: float = 0.1, **config):
    monkeypatch.setattr(server, "OFFER_SLACK", 0.05)
    monkeypatch.setattr(server, "build_assets", _geometry_here)
    root = root or volume(tmp_path)
    options = {"mode": "off", "keep_seconds": 0}
    if fake_cli is not None:
        options = {"mode": "quick", "fake": True, "scratch": tmp_path / "scratch",
                   "watch_seconds": 0.05, "keep_seconds": 0, "command": fake_cli.command}
    app = server.create_app(model=mujoco.MjModel.from_xml_string(DEMO), web_dir=web_dir,
                            label="tea_room", settle_seconds=0.02, prefetch_seconds=prefetch,
                            scans=scans.ScanConfig(**{**options, **config}),
                            splats=SplatSource.per_room(root / "splats", poll_seconds=0.05),
                            library=RoomLibrary.folder(root, poll_seconds=0.05))
    with TestClient(app) as client:
        yield client, app.state.roomcall


def _assert_security_headers(response) -> None:
    """The headers every route carries (`server.security_headers`), this one included."""
    headers = response.headers
    assert headers["strict-transport-security"].startswith("max-age=")
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"]
    assert "img-src 'self' data: blob:" in headers["content-security-policy"]


def quiet(ws, kind: str, seconds: float = 0.4) -> None:
    """Nothing of `kind` arrives at `ws` for `seconds` (state frames keep coming meanwhile)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            message = recv(ws, max(0.05, deadline - time.monotonic()))
        except TimeoutError:
            return
        assert message["t"] != kind, message


def test_a_pick_moves_only_that_call_room_with_its_splat(web_dir, tmp_path, monkeypatch):
    with picker_client(web_dir, tmp_path, monkeypatch) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "call1", "Ada")
        b, _ = join(client, stack, "call1", "Bo")
        c, wc = join(client, stack, "call2", "Cy")
        live = wa["v"]
        assert room_name(live) == "tea_room" and wa["pick"] == "tea_room"
        assert wa["rooms"] == [
            {"name": "tea_room", "label": "tea_room", "thumb": False, "default": True},
            {"name": "office_room", "label": "Office room", "thumb": True, "default": False},
            {"name": "loft", "label": "Loft", "thumb": False, "default": False}]
        a.send_json({"t": "pick", "room": "office_room"})
        loading = until(b, lambda m: m["t"] == "pick")
        assert loading == {"t": "pick", "room": "office_room", "label": "Office room",
                           "by": "Ada", "state": "loading"}
        scene = until(b, lambda m: m["t"] == "scene")
        assert scene["v"] == OFFICE and scene["label"] == "Office room"
        assert scene["stage"] == "objects"
        assert scene["pick"] == {"room": "office_room", "by": "Ada"}
        until(a, lambda m: m["t"] == "state" and m["v"] == OFFICE)
        assert hub.room_versions["call1"].id == OFFICE
        # Its own photoreal room, by the library room's name.
        bundle = client.get("/splat.json", params={"v": OFFICE}).json()
        assert client.get("/" + bundle["room"]).content == b"office"
        # The other call room is where it was, and heard nothing.
        assert until(c, lambda m: m["t"] == "state")["v"] == live
        assert "call2" not in hub.room_versions
        shown = client.get("/rooms.json", params={"room": "call1"}).json()["room"]
        assert shown == {"name": "office_room", "label": "Office room", "peers": 2}
        assert client.get("/rooms.json", params={"room": "call2"}).json()["room"]["name"] == \
            "tea_room"
        # Whoever comes in next is in the office too.
        _d, wd = join(client, stack, "call1", "Di")
        assert wd["v"] == OFFICE and wd["pick"] == "office_room"
        assert client.get("/healthz").json()["library"] == 2


def test_someone_joining_before_the_swap_hears_whose_pick_it_is(web_dir, tmp_path,
                                                                monkeypatch):
    with picker_client(web_dir, tmp_path, monkeypatch, prefetch=3.0) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, _ = join(client, stack, "call1", "Ada")
        a.send_json({"t": "pick", "room": "loft"})
        until(a, lambda m: m["t"] == "scene" and m["v"] == LOFT)
        d, wd = join(client, stack, "call1", "Di")
        assert wd["v"] == hub.version.id and wd["pick"] == "loft"      # not swapped yet
        pending = until(d, lambda m: m["t"] == "scene")
        assert pending["v"] == LOFT and pending["pick"] == {"room": "loft", "by": "Ada"}


def test_picking_the_default_room_lets_the_pick_go(web_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PICK_COOLDOWN", 0.0)
    with picker_client(web_dir, tmp_path, monkeypatch) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "call1", "Ada")
        b, _ = join(client, stack, "call1", "Bo")
        live = wa["v"]
        a.send_json({"t": "pick", "room": "office_room"})
        until(a, lambda m: m["t"] == "state" and m["v"] == OFFICE)
        b.send_json({"t": "pick", "room": "tea_room"})
        loading = until(a, lambda m: m["t"] == "pick")
        assert loading == {"t": "pick", "room": "tea_room", "label": "tea_room", "by": "Bo",
                           "state": "loading"}
        scene = until(a, lambda m: m["t"] == "scene")
        assert scene["v"] == live and scene["pick"] == {"room": "tea_room", "by": "Bo"}
        until(a, lambda m: m["t"] == "state" and m["v"] == live)
        assert "call1" not in hub.room_versions and "call1" not in hub.picked
        assert client.get("/rooms.json", params={"room": "call1"}).json()["room"]["name"] == \
            "tea_room"
        # The default is what it shows now: picking it again does nothing.
        a.send_json({"t": "pick", "room": "tea_room"})
        quiet(a, "pick")


def test_a_picked_room_is_let_go_once_its_call_room_empties(web_dir, tmp_path, monkeypatch):
    with picker_client(web_dir, tmp_path, monkeypatch) as (client, hub):
        with contextlib.ExitStack() as stack:
            a, wa = join(client, stack, "call1", "Ada")
            live = wa["v"]
            a.send_json({"t": "pick", "room": "office_room"})
            until(a, lambda m: m["t"] == "state" and m["v"] == OFFICE)
        wait_for(lambda: "call1" not in hub.room_versions)     # keep_seconds=0
        with contextlib.ExitStack() as stack:
            _a, wa = join(client, stack, "call1", "Ada")
            assert wa["v"] == live and wa["pick"] == "tea_room"


def test_bad_and_unknown_picks_are_ignored_or_refused(web_dir, tmp_path, monkeypatch):
    with picker_client(web_dir, tmp_path, monkeypatch) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, _ = join(client, stack, "call1", "Ada")
        b, _ = join(client, stack, "call1", "Bo")
        for bad in ("../office_room", "", 7, None, ["loft"], "x" * 49, "office room"):
            a.send_json({"t": "pick", "room": bad})
        a.send_json({"t": "pick"})
        quiet(a, "pick")
        a.send_json({"t": "pick", "room": "nowhere"})
        refused = until(a, lambda m: m["t"] == "pick")
        assert set(refused) == {"t", "error"} and "library" in refused["error"]
        quiet(b, "pick")                                   # the sender's alone
        assert "call1" not in hub.room_versions


def test_one_pick_at_a_time_and_a_cooldown_between_them(web_dir, tmp_path, monkeypatch):
    slow = server.load_version

    async def slow_load(ref):
        await asyncio.sleep(0.5)
        return await slow(ref)

    monkeypatch.setattr(server, "load_version", slow_load)
    with picker_client(web_dir, tmp_path, monkeypatch) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, _ = join(client, stack, "call1", "Ada")
        b, _ = join(client, stack, "call1", "Bo")
        a.send_json({"t": "pick", "room": "office_room"})
        until(b, lambda m: m["t"] == "pick" and m.get("state") == "loading")
        b.send_json({"t": "pick", "room": "loft"})
        refused = until(b, lambda m: m["t"] == "pick")
        assert "already on its way" in refused["error"]
        until(a, lambda m: m["t"] == "state" and m["v"] == OFFICE)
        b.send_json({"t": "pick", "room": "loft"})
        refused = until(b, lambda m: m["t"] == "pick")
        assert "you can pick another in" in refused["error"]
        assert hub.room_versions["call1"].id == OFFICE
        hub._picked_at.clear()                               # the cooldown is up
        b.send_json({"t": "pick", "room": "loft"})
        until(a, lambda m: m["t"] == "state" and m["v"] == LOFT)


def test_picks_share_a_load_and_only_so_many_run_at_once(web_dir, tmp_path, monkeypatch):
    """Call rooms are free to open: picking one room from many of them at once loads it once, and
    a pick that would start a load beyond ROOM_LOADS is refused rather than run (each holds a
    model and a glb, on a small container)."""
    slow, started = server.load_version, []

    async def slow_load(ref):
        started.append(ref.id)
        await asyncio.sleep(0.6)
        return await slow(ref)

    monkeypatch.setattr(server, "load_version", slow_load)
    monkeypatch.setattr(server, "ROOM_LOADS", 1)
    with picker_client(web_dir, tmp_path, monkeypatch) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, _ = join(client, stack, "call1", "Ada")
        b, _ = join(client, stack, "call2", "Bo")
        c, _ = join(client, stack, "call3", "Cy")
        a.send_json({"t": "pick", "room": "office_room"})
        until(a, lambda m: m["t"] == "pick" and m.get("state") == "loading")
        b.send_json({"t": "pick", "room": "office_room"})     # the same room: waits on that load
        until(b, lambda m: m["t"] == "pick" and m.get("state") == "loading")
        c.send_json({"t": "pick", "room": "loft"})            # another: one load runs already
        refused = until(c, lambda m: m["t"] == "pick")
        assert refused == {"t": "pick", "error": "The server is loading another room; "
                                                 "try again in a moment."}
        until(a, lambda m: m["t"] == "state" and m["v"] == OFFICE)
        until(b, lambda m: m["t"] == "state" and m["v"] == OFFICE)
        assert started == [OFFICE] and not hub._loading
        assert hub.room_versions["call1"] is hub.room_versions["call2"]
        c.send_json({"t": "pick", "room": "loft"})            # free again (no cooldown: refused)
        until(c, lambda m: m["t"] == "state" and m["v"] == LOFT)
        assert started == [OFFICE, LOFT]


def test_a_room_that_cannot_load_leaves_the_call_room_as_it_was(web_dir, tmp_path, monkeypatch):
    root = volume(tmp_path)
    (root / LOFT / "scene.xml").write_text("<mujoco><worldbody><geom type='nonsense'/>")
    with picker_client(web_dir, tmp_path, monkeypatch, root=root) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "call1", "Ada")
        a.send_json({"t": "pick", "room": "loft"})
        failed = until(a, lambda m: m["t"] == "pick" and "error" in m)
        assert failed == {"t": "pick", "room": "loft", "error": "Couldn't load Loft"}
        assert "call1" not in hub.room_versions
        assert until(a, lambda m: m["t"] == "state")["v"] == wa["v"]
        assert client.get("/rooms.json", params={"room": "call1"}).json()["room"]["name"] == \
            "tea_room"


def test_no_pick_while_a_scan_is_being_built_here(web_dir, tmp_path, fake_cli, monkeypatch):
    monkeypatch.setenv("SCAN_FAKE_QUICK", "hang")
    with picker_client(web_dir, tmp_path, monkeypatch, fake_cli=fake_cli) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "call1", "Ada")
        job = post(client, "call1", wa, make_zip(scan_entries(tmp_path))).json()["job"]
        until(a, scan_msg("building", job))
        assert hub.scans.busy("call1") and not hub.scans.busy("call2")
        a.send_json({"t": "pick", "room": "office_room"})
        refused = until(a, lambda m: m["t"] == "pick")
        assert "being built" in refused["error"]
        assert "call1" not in hub.room_versions
        # Another call room picks as it likes meanwhile.
        c, _ = join(client, stack, "call2", "Cy")
        c.send_json({"t": "pick", "room": "office_room"})
        until(c, lambda m: m["t"] == "state" and m["v"] == OFFICE)


def test_a_pick_replaces_a_finished_scan_and_its_photoreal_room(web_dir, tmp_path, fake_cli,
                                                                  monkeypatch):
    with picker_client(web_dir, tmp_path, monkeypatch, fake_cli=fake_cli,
                       splats_root=tmp_path / "scan-splats") as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "call1", "Ada")
        job = post(client, "call1", wa, make_zip(scan_entries(tmp_path))).json()["job"]
        until(a, scan_msg("done", job))
        assert client.get("/rooms.json", params={"room": "call1"}).json()["room"]["name"] == \
            "scan"
        bundle = tmp_path / "scan-splats" / f"scan-{job}"
        wait_for(bundle.is_dir)
        a.send_json({"t": "pick", "room": "office_room"})
        until(a, lambda m: m["t"] == "state" and m["v"] == OFFICE)
        wait_for(lambda: not bundle.exists())
        assert hub.scans.latest("call1") is None            # joiners are not told of the scan
        _b, wb = join(client, stack, "call1", "Bo")
        assert wb["pick"] == "office_room" and "scan" not in wb


def test_rooms_json_and_thumbnails(web_dir, tmp_path, monkeypatch):
    with picker_client(web_dir, tmp_path, monkeypatch) as (client, _hub):
        listing = client.get("/rooms.json", params={"room": "nobody-here"})
        assert listing.status_code == 200 and listing.headers["cache-control"] == "no-store"
        _assert_security_headers(listing)
        body = listing.json()
        assert [r["name"] for r in body["rooms"]] == ["tea_room", "office_room", "loft"]
        assert body["room"] == {"name": "tea_room", "label": "tea_room", "peers": 0}
        assert client.get("/rooms.json").status_code == 200                # the lobby
        bad = client.get("/rooms.json", params={"room": "../etc"})
        assert bad.status_code == 400
        _assert_security_headers(bad)
        thumb = client.get("/rooms/office_room/thumb.jpg")
        assert thumb.status_code == 200 and thumb.content == JPEG
        assert thumb.headers["content-type"] == "image/jpeg"
        assert "max-age" in thumb.headers["cache-control"]
        _assert_security_headers(thumb)
        for missing in ("/rooms/loft/thumb.jpg", "/rooms/nowhere/thumb.jpg",
                        "/rooms/..%2f.library%2foffice_room/thumb.jpg", "/rooms/office_room/x.jpg"):
            response = client.get(missing)
            assert response.status_code == 404, missing
            _assert_security_headers(response)


def test_a_library_room_named_like_the_default_is_the_default(web_dir, tmp_path, monkeypatch):
    root = volume(tmp_path)
    add_version(root, "tea_room-0123456789ab")
    write_library(root, [{"name": "tea_room", "label": "Tea", "version": "tea_room-0123456789ab"},
                         {"name": "loft", "label": "Loft", "version": LOFT}])
    (root / LIBRARY_DIR / "tea_room.jpg").write_bytes(JPEG)
    with picker_client(web_dir, tmp_path, monkeypatch, root=root) as (client, hub), \
            contextlib.ExitStack() as stack:
        rooms = client.get("/rooms.json").json()["rooms"]
        assert rooms == [{"name": "tea_room", "label": "tea_room", "thumb": True, "default": True},
                         {"name": "loft", "label": "Loft", "thumb": False, "default": False}]
        a, _ = join(client, stack, "call1", "Ada")
        a.send_json({"t": "pick", "room": "tea_room"})          # what it shows already
        quiet(a, "pick")
        assert "call1" not in hub.room_versions


def test_without_a_library_only_the_default_room_is_listed(web_dir, tmp_path):
    app = server.create_app(model=mujoco.MjModel.from_xml_string(DEMO), web_dir=web_dir,
                            settle_seconds=0.02)
    with TestClient(app) as client, contextlib.ExitStack() as stack:
        a, welcome = join(client, stack, "r", "Ada")
        assert welcome["rooms"] == [{"name": "room", "label": "room", "thumb": False,
                                     "default": True}]
        assert welcome["pick"] == "room"
        assert client.get("/rooms/room/thumb.jpg").status_code == 404
        assert client.get("/healthz").json()["library"] == 0
        a.send_json({"t": "pick", "room": "office_room"})
        refused = until(a, lambda m: m["t"] == "pick")
        assert "error" in refused


# -- the library file -----------------------------------------------------------------------------

def test_the_library_reads_its_file_and_skips_what_it_cannot_use(tmp_path):
    root = tmp_path / "volume"
    office = add_version(root, OFFICE, stage="objects")
    (office / "scene.glb").write_bytes(b"glb")
    (office / "scene.json").write_text("{}")
    add_version(root, LOFT)
    (root / "outside").mkdir()
    (root / "outside" / "scene.xml").write_text(DEMO)
    write_library(root, [
        {"name": "office_room", "label": "  Office room  ", "version": OFFICE, "thumb": True},
        {"name": "office_room", "label": "again", "version": LOFT},        # a duplicate
        {"name": "bad name", "label": "x", "version": LOFT},
        {"name": "../up", "label": "x", "version": LOFT},
        {"name": "gone", "label": "Gone", "version": "gone-0123456789ab"},  # no folder
        {"name": "sneaky", "label": "x", "version": "../outside"},
        {"name": "dot", "label": "x", "version": ".rooms"},
        {"name": "scan", "label": "x", "version": LOFT},   # what a call room on a scan is called
        {"name": "nolabel", "version": LOFT},
        "not an entry", {"label": "no name", "version": LOFT},
    ])
    (root / LIBRARY_DIR / "office_room.jpg").write_bytes(JPEG)
    rooms = RoomLibrary.folder(root).rooms()
    assert rooms == [
        LibraryRoom(name="office_room", label="Office room", version=OFFICE,
                    scene=office / "scene.xml", stage="objects", prebuilt=office,
                    thumb=root / LIBRARY_DIR / "office_room.jpg"),
        LibraryRoom(name="nolabel", label="nolabel", version=LOFT,
                    scene=root / LOFT / "scene.xml")]
    library = RoomLibrary.folder(root)
    assert library.get("nolabel").stage == "scene" and library.get("gone") is None


@pytest.mark.parametrize("content", ["{not json", "[]", '{"rooms": 3}', '{"rooms": [1, 2]}',
                                     "\xff\xfe"])
def test_a_broken_library_file_is_no_rooms(tmp_path, content):
    root = tmp_path / "volume"
    add_version(root, LOFT)
    write_library(root, content)
    assert RoomLibrary.folder(root).rooms() == []


def test_the_library_follows_its_file(tmp_path):
    root = tmp_path / "volume"
    add_version(root, OFFICE)
    add_version(root, LOFT)
    library = RoomLibrary.folder(root, poll_seconds=0.0)
    assert library.rooms() == []                                         # no file yet
    path = write_library(root, [{"name": "loft", "label": "Loft", "version": LOFT}])
    assert [r.name for r in library.rooms()] == ["loft"]
    path.write_text(json.dumps({"rooms": [{"name": "office_room", "label": "Office",
                                           "version": OFFICE},
                                          {"name": "loft", "label": "Loft", "version": LOFT}]}))
    later = time.time() + 5                      # however coarse the file system's clock is
    os.utime(path, (later, later))
    assert [r.name for r in library.rooms()] == ["office_room", "loft"]
    path.write_text("{broken")
    os.utime(path, (later + 5, later + 5))
    assert library.rooms() == []
    # Looked at no more often than poll_seconds: a change inside the window is not seen yet.
    slow = RoomLibrary.folder(root, poll_seconds=60.0)
    assert slow.rooms() == []
    path.write_text(json.dumps({"rooms": [{"name": "loft", "label": "Loft", "version": LOFT}]}))
    os.utime(path, (later + 10, later + 10))
    assert slow.rooms() == []


def test_a_fixed_library_and_a_load_that_fails():
    room = LibraryRoom(name="loft", label="Loft", version=LOFT, scene=Path("/x/scene.xml"))
    assert RoomLibrary.fixed([room]).get("loft") is room

    def broken():
        raise OSError("the Volume went away")

    assert RoomLibrary(broken).rooms() == []
