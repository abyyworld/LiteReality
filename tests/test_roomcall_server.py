"""The room call server: its HTTP surface, the Zoom-required security headers, and the WebSocket
room protocol, end to end through the real app.

Runs in-process with Starlette's TestClient against a tiny MJCF scene and a stand-in index.html,
so it depends on neither the demo room nor the real web client. Every receive has a timeout: a
protocol bug must fail a test, not hang the suite.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import threading
import time
from urllib.parse import urlencode

import anyio
import mujoco
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from litereality_agent import cli
from litereality_agent.room_ops.roomcall import server

SCENE = """
<mujoco model="roomcall-test">
  <option timestep="0.004"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1" rgba="0.8 0.3 0.2 1"/>
    </body>
    <body name="door" pos="1 0 0.52">
      <joint name="hinge" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="1"/>
      <geom type="box" size="0.4 0.02 0.5" pos="0.4 0 0" mass="5"/>
    </body>
  </worldbody>
</mujoco>
"""

IMPORT_MAP = ('{ "imports": {\n'
              '  "three": "https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js"\n'
              '}}')
INDEX = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>test</title>
<link rel="stylesheet" href="app.css"></head>
<body>
<script type="importmap">{IMPORT_MAP}</script>
<script type="module" src="app.js"></script>
</body></html>
"""


def _sha256(text: str) -> str:
    return base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()


@pytest.fixture(scope="module")
def web_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("roomcall")
    web = root / "web"
    web.mkdir()
    (web / "index.html").write_text(INDEX)
    (web / "app.js").write_text("console.log('room call');\n")
    (web / "app.css").write_text("body { margin: 0; }\n")
    (root / "secret.txt").write_text("not for browsers\n")
    return web


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(SCENE)


@pytest.fixture(scope="module")
def app(model, web_dir):
    return server.create_app(model=model, label="test room", web_dir=web_dir,
                             settle_seconds=0.05)


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as client:
        yield client


@pytest.fixture
def hub(app) -> server.RoomHub:
    return app.state.roomcall


def _body(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


# -- websocket helpers ---------------------------------------------------------------------------

def recv(ws, timeout: float = 5.0) -> dict:
    """The next server frame, decoded; WebSocketDisconnect if the server closed instead."""
    async def get():
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()

    message = ws.portal.call(get)
    if message["type"] == "websocket.close":
        raise WebSocketDisconnect(message.get("code", 1000), message.get("reason", ""))
    return json.loads(message["text"])


def until(ws, predicate, limit: int = 300) -> tuple[dict, list[dict]]:
    """Read frames until one matches; return it and everything seen up to and including it.
    States arrive at 30 Hz, so `limit` bounds the wait to ~10 s even when nothing matches."""
    seen = []
    for _ in range(limit):
        message = recv(ws)
        seen.append(message)
        if predicate(message):
            return message, seen
    raise AssertionError(f"no matching frame in {limit}; last: {seen[-3:]}")


def join(client, stack, room: str, name: str, color: str = "#336699"):
    query = urlencode({"room": room, "name": f"q{name}"})
    ws = stack.enter_context(client.websocket_connect(f"/ws?{query}"))
    ws.send_json({"t": "hello", "name": name, "color": color})
    welcome = recv(ws)
    assert welcome["t"] == "welcome", welcome
    assert welcome["room"] == room
    return ws, welcome


def is_state(message: dict) -> bool:
    return message["t"] == "state"


def wait_for(condition, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


# -- HTTP ----------------------------------------------------------------------------------------

ROUTES = {
    "/": "text/html",
    "/app.js": "text/javascript",
    "/app.css": "text/css",
    "/scene.json": "application/json",
    "/scene.glb": "model/gltf-binary",
    "/healthz": "application/json",
}


def _assert_security_headers(response) -> None:
    headers = response.headers
    assert headers["strict-transport-security"].startswith("max-age=")
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"]
    csp = headers["content-security-policy"]
    directives = dict(part.strip().split(" ", 1) for part in csp.split(";"))
    script_src = directives["script-src"]
    assert f"'sha256-{_sha256(IMPORT_MAP)}'" in script_src
    assert "'unsafe-inline'" not in script_src and "'unsafe-eval'" not in script_src
    assert "https://cdn.jsdelivr.net" in script_src and "https://appssdk.zoom.us" in script_src
    assert "'self'" in script_src
    assert "ws:" in directives["connect-src"] and "wss:" in directives["connect-src"]
    assert "blob:" in directives["img-src"] and "data:" in directives["img-src"]
    assert "blob:" in directives["media-src"]
    assert "zoom.us" in directives["frame-ancestors"]
    assert directives["object-src"] == "'none'"


@pytest.mark.parametrize("path", sorted(ROUTES))
def test_routes_are_served_with_security_headers(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(ROUTES[path])
    _assert_security_headers(response)


def test_route_bodies(client, model, web_dir):
    assert client.get("/").content == (web_dir / "index.html").read_bytes()
    assert client.get("/app.js").text == "console.log('room call');\n"
    glb = client.get("/scene.glb")
    assert glb.content[:4] == b"glTF"
    # Revalidation instead of a stale cached room after a restart with another scene.
    again = client.get("/scene.glb", headers={"If-None-Match": glb.headers["etag"]})
    assert again.status_code == 304 and not again.content
    manifest = client.get("/scene.json").json()
    assert manifest["glb"] == "scene.glb" and manifest["label"] == "test room"
    ids = {body["id"] for body in manifest["bodies"]}
    assert {_body(model, "box"), _body(model, "door")} <= ids
    health = client.get("/healthz").json()
    assert health["ok"] is True and health["label"] == "test room"


@pytest.mark.parametrize("path", [
    "/index.html", "/web/app.js", "/app.jsx", "/server.py", "/.env", "/secret.txt",
    "/../secret.txt", "/%2e%2e/secret.txt", "/app.js/../../secret.txt", "/..%2fsecret.txt",
    "/web/..%2f..%2fsecret.txt", "/docs", "/openapi.json", "/redoc", "/ws",
])
def test_anything_else_is_404_and_still_has_headers(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert b"not for browsers" not in response.content
    _assert_security_headers(response)


def test_inline_script_hashes_follow_the_browser():
    page = INDEX.replace("\n", "\r\n")
    assert server.inline_script_hashes(page) == [_sha256(IMPORT_MAP)]
    assert server.inline_script_hashes('<script src="a.js">x</script><SCRIPT>y</SCRIPT >') \
        == [_sha256("y")]


def test_missing_client_fails_at_startup(model, tmp_path):
    with pytest.raises(RuntimeError, match="index.html"):
        server.create_app(model=model, web_dir=tmp_path)


# -- the room protocol ---------------------------------------------------------------------------

def test_two_peers_see_each_other(client):
    with contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "pair", "Ada", "#AA0000")
        assert wa["peers"] == [] and wa["name"] == "Ada" and wa["color"] == "#aa0000"
        b, wb = join(client, stack, "pair", "Bo\x07b " + "x" * 60, "not-a-colour")
        assert len(wb["name"]) <= 40 and "\x07" not in wb["name"]
        assert wb["color"].startswith("#") and len(wb["color"]) == 7
        assert wb["peers"] == [{"id": wa["id"], "name": "Ada", "color": "#aa0000",
                                "media": {"camera": False, "mic": False}}]
        joined, _ = until(a, lambda m: m["t"] == "join")
        assert joined["peer"]["id"] == wb["id"]

        a.send_json({"t": "pose", "p": [0.5, 0.2, 1.6], "yaw": 0.25, "pitch": -0.1})
        b.send_json({"t": "pose", "p": [-0.5, 0.2, 1.6], "yaw": 3.0, "pitch": 0.0})
        for ws in (a, b):
            state, _ = until(ws, lambda m: is_state(m) and len(m["avatars"]) == 2)
            assert state["avatars"][wa["id"]] == [0.5, 0.2, 1.6, 0.25, -0.1]
            assert state["avatars"][wb["id"]] == [-0.5, 0.2, 1.6, 3.0, 0.0]
            assert isinstance(state["time"], float) and state["bodies"]


def test_grab_is_broadcast_and_release_clears_it(client, model):
    box = _body(model, "box")
    with contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "grab", "Ada")
        b, _ = join(client, stack, "grab", "Bob")
        state, _ = until(a, is_state)
        x, y, z = next(row[1:4] for row in state["bodies"] if row[0] == box)
        a.send_json({"t": "grab", "body": box, "point": [x, y, z]})
        a.send_json({"t": "move", "target": [x, y, z + 0.4]})
        state, _ = until(b, lambda m: is_state(m) and m["grabs"])
        owner, body, *anchor_target = state["grabs"][0]
        assert (owner, body) == (wa["id"], box) and len(anchor_target) == 6
        # Held up: the box rises toward the target instead of sitting on the floor.
        until(b, lambda m: is_state(m) and next(r[3] for r in m["bodies"] if r[0] == box) > z + 0.1)
        a.send_json({"t": "release"})
        until(b, lambda m: is_state(m) and not m["grabs"])


def test_rtc_reaches_only_the_addressee(client):
    with contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "rtc", "Ada")
        b, wb = join(client, stack, "rtc", "Bob")
        c, _ = join(client, stack, "rtc", "Cy")
        a.send_json({"t": "rtc", "to": "nobody", "data": {"sdp": "lost"}})
        a.send_json({"t": "rtc", "to": wa["id"], "data": {"sdp": "self"}})
        a.send_json({"t": "rtc", "to": wb["id"], "data": {"type": "offer", "sdp": "v=0"}})
        # Then something everybody gets: anything misrouted would have arrived before it.
        a.send_json({"t": "media", "camera": True, "mic": False})
        got, seen_b = until(b, lambda m: m["t"] == "rtc")
        assert got == {"t": "rtc", "from": wa["id"], "data": {"type": "offer", "sdp": "v=0"}}
        media, seen_c = until(c, lambda m: m["t"] == "media")
        assert media == {"t": "media", "id": wa["id"], "camera": True, "mic": False}
        assert not [m for m in seen_c if m["t"] == "rtc"]
        _, seen_a = until(a, is_state)
        assert not [m for m in seen_a if m["t"] in {"rtc", "media"}]


def test_rooms_are_isolated(client):
    with contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "iso-a", "Ada")
        b, wb = join(client, stack, "iso-b", "Bob")
        assert wb["peers"] == []
        a.send_json({"t": "pose", "p": [0, 0, 1.6], "yaw": 0, "pitch": 0})
        b.send_json({"t": "pose", "p": [1, 0, 1.6], "yaw": 0, "pitch": 0})
        until(a, lambda m: is_state(m) and wa["id"] in m["avatars"])
        until(b, lambda m: is_state(m) and wb["id"] in m["avatars"])
        seen = [recv(a) for _ in range(10)]
        assert all(m["t"] == "state" and set(m["avatars"]) == {wa["id"]} for m in seen)


@pytest.mark.parametrize("room", ["bad%20id", "x" * 65, "..%2Fx", "caf%C3%A9", "a.b"])
def test_bad_room_id_is_refused(client, room):
    with client.websocket_connect(f"/ws?room={room}") as ws:
        error = recv(ws)
        assert error["t"] == "error" and "room id" in error["msg"]
        with pytest.raises(WebSocketDisconnect) as closed:
            recv(ws)
        assert closed.value.code == server.CLOSE_REFUSED


def test_malformed_and_oversized_messages_are_ignored(client, hub):
    with contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "junk", "Ada")
        b, wb = join(client, stack, "junk", "Bob")
        for junk in [
            "not json", "[1, 2, 3]", '{"t": 5}', '"hello"', "{}", '{"t": "teleport"}',
            '{"t": "pose", "p": [NaN, 0, 0], "yaw": 0, "pitch": 0}',
            '{"t": "pose", "p": [1e300, 0, 0], "yaw": 0, "pitch": 0}',
            '{"t": "pose", "p": [0, 0], "yaw": 0, "pitch": 0}',
            '{"t": "grab", "body": "box", "point": [0, 0, 0]}',
            '{"t": "grab", "body": true, "point": [0, 0, 0]}',
            '{"t": "grab", "body": 99999, "point": [0, 0, 0]}',
            '{"t": "grab", "body": 1, "point": [0, 0, Infinity]}',
            '{"t": "move", "target": "up"}', '{"t": "media", "camera": "yes"}',
            '{"t": "rtc", "data": {}}', '{"t": "rtc", "to": 5, "data": {}}',
            '{"t": "rtc", "to": "%s", "data": NaN}' % wb["id"],
            "[" * 30000 + "]" * 30000,
            json.dumps({"t": "rtc", "to": wb["id"], "data": "x" * (server.MAX_MESSAGE_BYTES + 1)}),
        ]:
            a.send_text(junk)
        a.send_bytes(b'{"t": "pose", "p": [9, 9, 9], "yaw": 0, "pitch": 0}')
        a.send_json({"t": "rtc", "to": wb["id"], "data": "small"})
        a.send_json({"t": "pose", "p": [0.1, 0.2, 1.6], "yaw": 0.5, "pitch": 0.0})
        state, _ = until(a, lambda m: is_state(m) and wa["id"] in m["avatars"])
        assert state["avatars"][wa["id"]] == [0.1, 0.2, 1.6, 0.5, 0.0]
        assert not state["grabs"]
        first_rtc, _ = until(b, lambda m: m["t"] == "rtc")
        assert first_rtc["data"] == "small"
        assert len(hub.rooms["junk"].peers) == 2


def test_seventeenth_peer_is_refused(client, hub):
    with contextlib.ExitStack() as stack:
        for i in range(server.MAX_PEERS):
            join(client, stack, "full", f"p{i}")
        assert len(hub.rooms["full"].peers) == server.MAX_PEERS
        with client.websocket_connect("/ws?room=full&name=late") as late:
            error = recv(late)
            assert error["t"] == "error" and "full" in error["msg"]
            with pytest.raises(WebSocketDisconnect) as closed:
                recv(late)
            assert closed.value.code == server.CLOSE_REFUSED
        assert len(hub.rooms["full"].peers) == server.MAX_PEERS


def test_reset_is_announced_to_everyone(client):
    with contextlib.ExitStack() as stack:
        a, _ = join(client, stack, "reset", "Ada")
        b, wb = join(client, stack, "reset", "Bob")
        b.send_json({"t": "reset"})
        for ws in (a, b):
            reset, _ = until(ws, lambda m: m["t"] == "reset")
            assert reset == {"t": "reset", "by": wb["id"]}


def test_leaving_drops_grabs_and_the_empty_room(client, hub, model):
    box = _body(model, "box")
    with contextlib.ExitStack() as stack:
        b, _ = join(client, stack, "leave", "Bob")
        with client.websocket_connect("/ws?room=leave") as a:
            a.send_json({"t": "hello", "name": "Ada"})
            wa = recv(a)
            a.send_json({"t": "pose", "p": [0, 0, 1.6], "yaw": 0, "pitch": 0})
            a.send_json({"t": "grab", "body": box, "point": [0, 0, 0.1]})
            until(b, lambda m: is_state(m) and m["grabs"] and wa["id"] in m["avatars"])
        until(b, lambda m: m["t"] == "leave" and m["id"] == wa["id"])
        state, _ = until(b, is_state)
        assert not state["grabs"] and wa["id"] not in state["avatars"]
        task = hub.rooms["leave"].task
        assert task is not None and not task.done()
    assert wait_for(lambda: "leave" not in hub.rooms)
    assert wait_for(task.done)
    # A fresh room comes back on the next join.
    with contextlib.ExitStack() as stack:
        _, welcome = join(client, stack, "leave", "Cy")
        assert welcome["peers"] == []


def test_a_room_that_resets_itself_tells_everyone(client, hub, model):
    # Each client would otherwise keep dragging an object the server no longer holds for it.
    box = _body(model, "box")
    with contextlib.ExitStack() as stack:
        a, _ = join(client, stack, "blowup", "Ada")
        until(a, is_state)
        sim = hub.rooms["blowup"].sim
        with sim._lock:
            sim.data.qvel[int(model.jnt_dofadr[model.body_jntadr[box]])] = 1e6  # a blow-up
        reset, _ = until(a, lambda m: m["t"] == "reset")
        assert reset == {"t": "reset", "by": "server"}
        assert sim.unstable_resets == 1


def test_new_rooms_are_refused_past_the_cap(client, hub, monkeypatch):
    assert wait_for(lambda: not hub.rooms)
    monkeypatch.setattr(server, "MAX_ROOMS", 2)
    with contextlib.ExitStack() as stack:
        join(client, stack, "cap-a", "Ada")
        join(client, stack, "cap-b", "Bob")
        with client.websocket_connect("/ws?room=cap-c&name=Cy") as late:
            error = recv(late)
            assert error["t"] == "error" and "rooms" in error["msg"]
            with pytest.raises(WebSocketDisconnect) as closed:
                recv(late)
            assert closed.value.code == server.CLOSE_REFUSED
        _, welcome = join(client, stack, "cap-a", "Dee")  # a room that exists still takes people
        assert len(welcome["peers"]) == 1
        assert set(hub.rooms) == {"cap-a", "cap-b"}


def test_a_client_that_hangs_up_while_its_room_settles_is_never_added(model, web_dir,
                                                                       monkeypatch):
    gate = threading.Event()
    settled_on: list[str] = []

    class SlowSettle(server.RoomSim):
        def __init__(self, *args, **kwargs):
            settled_on.append(threading.current_thread().name)
            gate.wait(10)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(server, "RoomSim", SlowSettle)
    monkeypatch.setattr(server, "SETTLE_WORKERS", 1)
    app = server.create_app(model=model, web_dir=web_dir, settle_seconds=0.05)
    hub = app.state.roomcall
    try:
        with TestClient(app) as client, contextlib.ExitStack() as stack:
            a = stack.enter_context(client.websocket_connect("/ws?room=slow&name=Ada"))
            a.send_json({"t": "hello", "name": "Ada"})
            assert wait_for(lambda: len(settled_on) == 1)  # Ada's room is settling...
            b = stack.enter_context(client.websocket_connect("/ws?room=slow&name=Bob"))
            b.send_json({"t": "hello", "name": "Bob"})
            c = stack.enter_context(client.websocket_connect("/ws?room=gone&name=Cy"))
            c.send_json({"t": "hello", "name": "Cy"})  # ...and Cy's waits behind it
            assert wait_for(lambda: "gone" in hub.rooms and hub.rooms["slow"].joining == 2)
            b.close()  # hang up, as a browser does (leaving the server side running)
            c.close()
            assert wait_for(lambda: hub.rooms["slow"].joining == 1 and "gone" not in hub.rooms)
            gate.set()
            welcome = recv(a)
            assert welcome["t"] == "welcome" and welcome["peers"] == []
            _, seen = until(a, is_state)
            assert not [m for m in seen if m["t"] in {"join", "leave"}]  # Bob never joined
            join(client, stack, "later", "Dee")  # its settle was queued after Cy's
            # Cy's settle never ran, and none ran on the threads that step live rooms.
            assert len(settled_on) == 2
            assert all(name.startswith("roomcall-settle") for name in settled_on)
    finally:
        gate.set()


def test_commands_coalesce_but_a_release_or_reset_is_never_dropped(monkeypatch):
    monkeypatch.setattr(server, "MAX_PENDING_COMMANDS", 4)
    room = server.Room("q", ready=None)
    for i in range(10):                     # a fast hand: one queued move, the latest target
        room.command("move", "a", [i, 0, 0])
    assert list(room.commands) == [["move", "a", [9, 0, 0]]]
    room.command("grab", "a", 3, [0, 0, 0])
    room.command("move", "a", [1, 1, 1])    # after a grab: a new move, never merged across it
    room.command("move", "b", [2, 2, 2])
    room.command("grab", "b", 4, [0, 0, 0])  # the queue is full: dropped
    room.command("move", "a", [5, 5, 5])    # merges into a's newest, which is a move
    room.command("release", "a")            # full, but a release always gets in...
    room.command("release", "a")            # ...once: a second in a row says nothing new
    room.command("release", "c")
    assert list(room.commands) == [
        ["move", "a", [9, 0, 0]], ["grab", "a", 3, [0, 0, 0]], ["move", "a", [5, 5, 5]],
        ["move", "b", [2, 2, 2]], ["release", "a"], ["release", "c"]]
    room.command("reset")                   # supersedes everything queued before it
    assert list(room.commands) == [["reset"]]
    assert list(room.take()) == [["reset"]] and not room.commands
    room.command("move", "a", [0, 0, 0])    # nothing merges into a batch already taken
    assert list(room.commands) == [["move", "a", [0, 0, 0]]]


def test_an_overloaded_room_runs_in_slow_motion_instead_of_seizing_up(model):
    # When a step costs more wall time than it simulates, catching up makes every tick longer
    # than the last until a tick is 400 steps: about one state frame a second.
    cap = server.max_steps_per_tick(model.opt.timestep)
    assert server.max_steps_per_tick(0.002) == 17 and cap == 9
    most = [0]

    class Heavy(server.RoomSim):
        def step_realtime(self, wall_dt, *, max_steps=400):
            steps = super().step_realtime(wall_dt, max_steps=max_steps)
            most[0] = max(most[0], steps)
            time.sleep(1.25 * steps * self.timestep)  # each step costs 1.25x what it simulates
            return steps

    async def run() -> tuple[list[float], server.Room, dict]:
        hub = server.RoomHub(model)
        ready = asyncio.get_running_loop().create_future()
        ready.set_result(Heavy(model, settle_seconds=0.0))
        room = hub.rooms["heavy"] = server.Room("heavy", ready)
        frames: list[float] = []
        peer = server.Peer("p", None, "p", "#000000")
        peer.push_state = lambda text: frames.append(time.monotonic())
        room.peers[peer.id] = peer
        task = asyncio.create_task(hub._physics(room))
        await asyncio.sleep(1.4)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return frames, room, hub.stats()

    frames, room, stats = asyncio.run(run())
    assert most[0] <= cap
    assert sum(1 for t in frames if t > frames[-1] - 1.0) >= 8  # frames keep coming
    assert 0.3 < room.realtime < 0.95                            # in slow motion
    assert stats["realtime"] == round(room.realtime, 3)          # and /healthz says so


def test_serve_notices_a_dropped_network_within_seconds(monkeypatch, tmp_path, web_dir):
    import uvicorn

    seen: dict = {}

    class Config:
        def __init__(self, app, **kwargs):
            seen.update(kwargs)

    class Server:
        started = True

        def __init__(self, config):
            pass

        def run(self, sockets):
            pass

    monkeypatch.setattr(uvicorn, "Config", Config)
    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(server, "WEB_DIR", web_dir)
    scene = tmp_path / "scene.xml"
    scene.write_text(SCENE)
    assert server.serve(scene, port=0, open_browser=False) == 0
    # uvicorn's default (20 s + 20 s) leaves a gone peer's frozen avatar, and its grab, for ~40 s.
    assert seen["ws_ping_interval"] + seen["ws_ping_timeout"] <= 10


def test_query_name_is_used_without_hello(client):
    with contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "query", "Ada")
        b = stack.enter_context(client.websocket_connect("/ws?room=query&name=Bea"))
        b.send_json({"t": "pose", "p": [0, 0, 1.6], "yaw": 0, "pitch": 0})
        wb = recv(b)
        assert wb["t"] == "welcome" and wb["name"] == "Bea"
        joined, _ = until(a, lambda m: m["t"] == "join")
        assert joined["peer"]["name"] == "Bea"
        # The first, non-hello message still counts.
        until(a, lambda m: is_state(m) and wb["id"] in m["avatars"])


# -- running it ----------------------------------------------------------------------------------

def test_scene_resolution_and_label(tmp_path):
    run = tmp_path / "tea_room"
    (run / "mujoco_seed").mkdir(parents=True)
    scene = run / "mujoco_seed" / "scene.xml"
    scene.write_text(SCENE)
    assert server.resolve_scene(run) == scene
    assert server.resolve_scene(scene) == scene
    assert server.scene_label(scene) == "tea_room"
    (run / "scene.xml").write_text(SCENE)
    assert server.resolve_scene(run) == run / "scene.xml"
    with pytest.raises(ValueError, match="no such scene"):
        server.resolve_scene(tmp_path / "missing")
    with pytest.raises(ValueError, match="mujoco_seed"):
        server.resolve_scene(tmp_path)


def test_cli_wiring():
    args = cli._parser().parse_args(["roomcall", "--demo", "--no-open", "--port", "9001"])
    assert args.handler is cli._roomcall
    assert (args.scene, args.demo, args.port, args.open_browser, args.host, args.label) == \
        (None, True, 9001, False, "127.0.0.1", None)
    args = cli._parser().parse_args(["roomcall", "run/tea_room", "--label", "Tea"])
    assert (args.scene, args.demo, args.port, args.open_browser, args.label) == \
        ("run/tea_room", False, 8790, True, "Tea")
