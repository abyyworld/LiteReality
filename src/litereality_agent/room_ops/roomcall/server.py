"""The room call's server: one shared MuJoCo simulation per room, streamed to every browser in it.

Why a server-side simulation rather than physics in each browser: everyone has to see the SAME
chair fall over. Peer-to-peer physics drifts apart within seconds (floating point, different frame
rates, two people grabbing one object), and reconciling it is a research project. One MjData per
room on the server, stepped in real time and broadcast as body poses at 30 Hz, is boring and right.

Shape of it:

- The scene is compiled and its geometry built ONCE at startup (`geometry.build_scene_assets`);
  the glb and manifest are served from memory. Rooms share that one MjModel — MuJoCo models are
  read-only after compilation, so each room only needs its own MjData (inside `RoomSim`).
- Rooms are created lazily on first join (`/ws?room=<id>`) and dropped, physics task and all, when
  the last person leaves. A room nobody is in costs nothing. At most MAX_ROOMS run at once, and a
  new room settles on a small pool of its own, so a burst of new room ids (anyone with the link
  can invent them) can neither take the threads live rooms step on nor outlast the clients that
  asked for them.
- Each room runs one asyncio task: every ~1/120 s it hands the elapsed wall time to
  `RoomSim.step_realtime` on a worker thread (so a heavy scene never stalls the event loop that
  carries everyone's messages) and every 1/30 s broadcasts a state frame. A tick catches up at
  most a few intervals' worth of steps, so a scene too heavy for the machine runs in slow motion
  (the slowest room's real-time factor is on /healthz) instead of seizing up.
- Commands from browsers (grab / move / release / reset) are queued and applied on that same
  worker thread right before stepping, in arrival order. They never wait on a running step, and
  a grab can never be overtaken by the move that follows it.
- Whenever the room is put back without anyone asking (the simulation blew up), everyone is told
  with a reset "by" "server", so no client keeps dragging an object nobody holds any more.
- Each peer has its own sender task: control messages queue in order, state frames are a single
  "latest" slot. A slow phone therefore skips frames instead of holding up the room or growing an
  unbounded backlog of stale poses.

Every HTTP response carries the OWASP headers the Zoom client checks before it will render a Zoom
App at all (Strict-Transport-Security, X-Content-Type-Options, Content-Security-Policy,
Referrer-Policy — developers.zoom.us/docs/zoom-apps/security/owasp/). The CSP admits the page's one
inline script, the import map, by the sha256 the server computes from the served file at startup,
so editing index.html can never silently break the policy.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import functools
import hashlib
import json
import logging
import math
import re
import secrets
import socket
import tempfile
import threading
import time
import webbrowser
from collections import deque
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

from litereality_agent.room_ops.roomcall import DEFAULT_PORT, add_arguments
from litereality_agent.room_ops.roomcall.geometry import build_scene_assets
from litereality_agent.room_ops.roomcall.model_info import movable_body_ids
from litereality_agent.room_ops.roomcall.sim import RoomSim, load_model

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
PORT_TRIES = 20

ROOM_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
COLOR = re.compile(r"#[0-9a-fA-F]{6}")
DEFAULT_ROOM = "lobby"
MAX_PEERS = 16
MAX_NAME = 40
# Application limit: anything longer is dropped unread. The transport limit (`WS_MAX_SIZE`, set on
# uvicorn) is looser on purpose — a frame between the two is ignored rather than killing the
# connection, so one oversized signaling blob costs a renegotiation, not a rejoin.
MAX_MESSAGE_BYTES = 64 * 1024
WS_MAX_SIZE = 1024 * 1024
# Positions/angles beyond this are not a room; they are someone poking at the protocol.
COORD_LIMIT = 1.0e4
# Grabs and moves beyond this many waiting for one tick are dropped (releases and resets never are;
# see `Room.command`).
MAX_PENDING_COMMANDS = 2048
# Control messages a peer may have unsent before it is considered stuck and closed.
OUTBOX_LIMIT = 512
HELLO_TIMEOUT = 10.0
# Rooms one server runs at once. Each is a whole physics simulation, and with the link on a public
# tunnel a room id costs an attacker nothing to invent: past this a new room is refused rather than
# slowing every room people are already in.
MAX_ROOMS = 16
SERVER_FULL = "this server is running all the rooms it can; join an existing room or try later"
# A new room settles its scene on this small pool of its own, never on the threads that step live
# rooms, so a burst of new rooms queues behind itself instead of freezing the lobby.
SETTLE_WORKERS = 2
# Frames a client sends while its room settles, kept for after it joins. A well-behaved client sends
# nothing before its welcome but hello; this only bounds what a misbehaving one can make us hold.
EARLY_FRAMES = 32
PHYSICS_INTERVAL = 1.0 / 120.0
STATE_INTERVAL = 1.0 / 30.0
# How many physics intervals' worth of steps one tick may catch up. A tick that comes late (a busy
# event loop, a GC pause) is caught up in full; one that asks for more than this means the steps
# themselves cost more than real time, and catching up would only make the next tick longer — the
# spiral ends at `step_realtime`'s 400-step stall clamp, one state frame and one command batch per
# second or so. Capped, an overloaded room runs in slow motion but keeps its frame rate.
CATCH_UP_TICKS = 4
# Window over which a room's real-time factor (simulated seconds per wall second) is measured.
REALTIME_WINDOW = 1.0
# uvicorn's default keepalive (20 s ping, 20 s to answer) leaves someone whose network dropped
# standing frozen in the room, perhaps still holding something, for ~40 s. This notices in ~10 s.
WS_PING_INTERVAL = 5.0
WS_PING_TIMEOUT = 5.0
# Sent with every refusal. A client should show the message and NOT auto-reconnect on this code.
CLOSE_REFUSED = 1008
CLOSE_TOO_SLOW = 1013

SCENE_CANDIDATES = ("scene.xml", "mujoco_seed/scene.xml", "mujoco/scene.xml")
# Folder names that say how a scene was exported, not which room it is — skipped for the label.
_EXPORT_DIRS = {"mujoco_seed", "mujoco"}

THREE_CDN = "https://cdn.jsdelivr.net"
ZOOM_SDK = "https://appssdk.zoom.us"
ZOOM_FRAMES = "https://zoom.us https://*.zoom.us"

# Distinct, readable on a light or dark room, and never the same for two ids in a row by accident.
PALETTE = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#9a6324", "#469990", "#800000", "#808000", "#000075")

JS = "text/javascript; charset=utf-8"
CSS = "text/css; charset=utf-8"
HTML = "text/html; charset=utf-8"


# ---------------------------------------------------------------------------------------------
# The page and its security headers
# ---------------------------------------------------------------------------------------------

_SCRIPT = re.compile(r"<script\b([^>]*)>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
_SRC_ATTR = re.compile(r"\bsrc\s*=", re.IGNORECASE)


def inline_script_hashes(page: str) -> list[str]:
    """Base64 sha256 of each inline <script> body, exactly as a browser hashes it for CSP.

    The browser hashes the element's text after HTML input preprocessing, which turns CRLF and CR
    into LF — so a file saved with Windows line endings must hash the same as one without.
    Scripts with a `src` are skipped: their body is ignored by browsers and `'self'`/the CDN
    already admits the file itself.
    """
    text = page.replace("\r\n", "\n").replace("\r", "\n")
    hashes = []
    for match in _SCRIPT.finditer(text):
        if _SRC_ATTR.search(match.group(1)):
            continue
        digest = hashlib.sha256(match.group(2).encode("utf-8")).digest()
        hashes.append(base64.b64encode(digest).decode("ascii"))
    return hashes


def content_security_policy(script_hashes: Iterable[str]) -> str:
    """The page's CSP: our own files, three.js from jsDelivr, the Zoom Apps SDK, and nothing else.

    - script-src: no 'unsafe-inline' — the import map is let in by hash.
    - style-src keeps 'unsafe-inline': styles are not code, and the client may set style
      attributes from JS; forbidding that buys nothing here and breaks UI in surprising places.
    - connect-src: ws:/wss: for the room socket (also behind a tunnel on another host), blob:/data:
      because GLTFLoader fetches its embedded textures through blob URLs.
    - img/media: blob:/data: for textures and webcam video. frame-ancestors lets Zoom (desktop and
      the zoom.us web client) embed the page, and nobody else.
    """
    hashes = " ".join(f"'sha256-{h}'" for h in script_hashes)
    directives = {
        "default-src": "'self'",
        "script-src": " ".join(filter(None, ("'self'", hashes, THREE_CDN, ZOOM_SDK))),
        "style-src": "'self' 'unsafe-inline'",
        "img-src": "'self' data: blob:",
        "media-src": "'self' data: blob: mediastream:",
        "font-src": "'self' data:",
        "connect-src": f"'self' ws: wss: blob: data: {THREE_CDN} {ZOOM_SDK}",
        "worker-src": "'self' blob:",
        "object-src": "'none'",
        "base-uri": "'self'",
        "form-action": "'self'",
        "frame-ancestors": f"'self' {ZOOM_FRAMES}",
    }
    return "; ".join(f"{name} {value}" for name, value in directives.items())


def security_headers(script_hashes: Iterable[str]) -> dict[str, str]:
    """The headers Zoom's embedded browser requires on a Zoom App's pages (it refuses to render
    an HTML response missing any of them), sent on every response so no route is the exception.
    HSTS is ignored by browsers over plain http, so it is harmless on localhost."""
    return {
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Content-Security-Policy": content_security_policy(script_hashes),
    }


class SecurityHeaders:
    """Pure ASGI middleware adding fixed headers to every HTTP response — including Starlette's
    own 404s, which a route decorator would miss. A handler's own value for a header wins."""

    def __init__(self, app, headers: dict[str, str]) -> None:
        self.app = app
        self.headers = [(k.lower().encode("latin-1"), v.encode("latin-1"))
                        for k, v in headers.items()]

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {key.lower() for key, _ in headers}
                headers.extend(h for h in self.headers if h[0] not in present)
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


@dataclass(frozen=True)
class WebAssets:
    """index.html read once (so the CSP hash and the served page can never disagree) and the
    folder app.js / app.css are read from per request."""

    root: Path
    index: bytes
    script_hashes: tuple[str, ...]

    @classmethod
    def load(cls, root: Path) -> WebAssets:
        page = root / "index.html"
        if not page.is_file():
            raise RuntimeError(f"room call web client missing: {page}")
        index = page.read_bytes()
        hashes = tuple(inline_script_hashes(index.decode("utf-8")))
        if len(hashes) != 1:
            log.warning("%s has %d inline scripts; the CSP admits each by hash, but the page is "
                        "meant to have exactly one (the import map)", page, len(hashes))
        return cls(root, index, hashes)

    def asset(self, name: str) -> bytes | None:
        """One of the two vendored files by fixed name. The name never comes from a request."""
        path = self.root / name
        return path.read_bytes() if path.is_file() else None


# ---------------------------------------------------------------------------------------------
# Message validation
# ---------------------------------------------------------------------------------------------

def dumps(message: Any) -> str:
    # allow_nan=False: a NaN that sneaks through must fail here, loudly, not reach a browser.
    return json.dumps(message, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def parse_message(text: str) -> dict | None:
    """A client frame as a dict with a string "t", or None for anything else (too long, not JSON,
    not an object). Never raises: a bad frame is dropped, the connection and the room carry on."""
    if len(text) > MAX_MESSAGE_BYTES or len(text.encode("utf-8", "surrogatepass")) > MAX_MESSAGE_BYTES:
        return None
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("t"), str):
        return None
    return data


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and abs(value) <= COORD_LIMIT else None


def vec3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    out = [finite(v) for v in value]
    return None if any(v is None for v in out) else out  # type: ignore[return-value]


def clean_name(value: Any) -> str | None:
    """A display name with control characters dropped and whitespace collapsed, or None."""
    if not isinstance(value, str):
        return None
    name = " ".join("".join(ch for ch in value if ch.isprintable()).split())[:MAX_NAME].strip()
    return name or None


def clean_color(value: Any) -> str | None:
    return value.lower() if isinstance(value, str) and COLOR.fullmatch(value) else None


# ---------------------------------------------------------------------------------------------
# Rooms and peers
# ---------------------------------------------------------------------------------------------

@dataclass(eq=False)
class Peer:
    id: str
    ws: WebSocket
    name: str
    color: str
    camera: bool = False
    mic: bool = False
    pose: list[float] | None = None  # [x, y, z, yaw, pitch], eye position in world frame
    outbox: deque[str] = field(default_factory=deque)
    latest_state: str | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    stuck: bool = False
    sender: asyncio.Task | None = None

    def info(self) -> dict:
        return {"id": self.id, "name": self.name, "color": self.color,
                "media": {"camera": self.camera, "mic": self.mic}}

    def push(self, text: str) -> None:
        if len(self.outbox) >= OUTBOX_LIMIT:
            self.stuck = True
        else:
            self.outbox.append(text)
        self.wake.set()

    def push_state(self, text: str) -> None:
        self.latest_state = text
        self.wake.set()


@dataclass(eq=False)
class Room:
    id: str
    ready: asyncio.Future  # -> RoomSim, settling on the settle pool
    peers: dict[str, Peer] = field(default_factory=dict)
    commands: deque[list] = field(default_factory=deque)
    joining: int = 0
    task: asyncio.Task | None = None
    realtime: float | None = None  # simulated seconds per wall second, last REALTIME_WINDOW
    _newest: dict[str, list] = field(default_factory=dict)  # each peer's newest queued command

    @property
    def sim(self) -> RoomSim:
        return self.ready.result()

    def broadcast(self, message: dict, *, exclude: str | None = None) -> None:
        text = dumps(message)
        for peer in self.peers.values():
            if peer.id != exclude:
                peer.push(text)

    def command(self, op: str, *args) -> None:
        """Queue a command for the next tick. Everything queued is applied before the same step,
        which is what lets the queue stay short without losing anything that matters:

        - a reset supersedes everything queued before it (it restores the room and drops every
          grab), so it replaces the queue;
        - a release is never dropped — losing one leaves a ghost holding an object after its owner
          has gone — though a peer's second release in a row says nothing new and is skipped;
        - a move replaces the peer's queued move when nothing of theirs came after it, so a fast
          hand costs one entry per tick rather than one per message;
        - only grabs and moves are dropped once MAX_PENDING_COMMANDS are waiting.
        """
        if op == "reset":
            self.commands.clear()
            self._newest.clear()
            self.commands.append([op])
            return
        owner = args[0]
        newest = self._newest.get(owner)
        if newest is not None and newest[0] == op:
            if op == "release":
                return
            if op == "move":
                newest[2] = args[1]
                return
        if op != "release" and len(self.commands) >= MAX_PENDING_COMMANDS:
            return
        entry = [op, *args]
        self.commands.append(entry)
        self._newest[owner] = entry

    def take(self) -> deque[list]:
        """Hand everything queued to a tick and start an empty queue. Swapping it here rather than
        draining it on the worker thread keeps the queue, and the entries `command` coalesces in
        place, touched by the event loop alone."""
        batch, self.commands = self.commands, deque()
        self._newest.clear()
        return batch


def max_steps_per_tick(timestep: float) -> int:
    """The most physics steps one tick may take: `CATCH_UP_TICKS` intervals' worth (17 at
    MuJoCo's default 2 ms)."""
    if not timestep > 0:
        return 1
    return max(1, math.ceil(CATCH_UP_TICKS * PHYSICS_INTERVAL / timestep - 1e-9))


def _tick(sim: RoomSim, commands: deque[list], dt: float, want_state: bool,
          max_steps: int) -> tuple[int, dict | None]:
    """Worker-thread half of one physics tick: apply queued commands in order, step, snapshot.
    Returns the steps taken and the snapshot (None unless `want_state`)."""
    while commands:
        op, *args = commands.popleft()
        try:
            getattr(sim, op)(*args)
        except Exception:  # noqa: BLE001 — one bad command must not stop the room
            log.exception("roomcall: %s%r failed", op, tuple(args))
    steps = sim.step_realtime(dt, max_steps=max_steps)
    return steps, (sim.snapshot() if want_state else None)


class RoomHub:
    """Every live room of one server, keyed by room id, all on one shared model."""

    def __init__(self, model: mujoco.MjModel, *, settle_seconds: float = 1.0) -> None:
        self.model = model
        self.settle_seconds = settle_seconds
        self.rooms: dict[str, Room] = {}
        self._settler: ThreadPoolExecutor | None = None

    # -- lifecycle ------------------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def lifespan(self, _app: FastAPI):
        try:
            yield
        finally:
            for room in list(self.rooms.values()):
                room.ready.cancel()
                if room.task:
                    room.task.cancel()
                for peer in room.peers.values():
                    if peer.sender:
                        peer.sender.cancel()
            self.rooms.clear()
            if self._settler is not None:
                self._settler.shutdown(wait=False, cancel_futures=True)
                self._settler = None

    def stats(self) -> dict:
        """Counts, and the slowest room's real-time factor (None until a room has run for
        `REALTIME_WINDOW`): below 1 means the machine cannot keep up and that room is in slow
        motion."""
        rates = [room.realtime for room in self.rooms.values() if room.realtime is not None]
        return {"rooms": len(self.rooms),
                "peers": sum(len(room.peers) for room in self.rooms.values()),
                "realtime": round(min(rates), 3) if rates else None}

    def _room(self, room_id: str) -> Room | None:
        """The room, made and set settling if new; None if that would exceed MAX_ROOMS."""
        room = self.rooms.get(room_id)
        if room is None:
            if len(self.rooms) >= MAX_ROOMS:
                return None
            if self._settler is None:
                self._settler = ThreadPoolExecutor(SETTLE_WORKERS,
                                                   thread_name_prefix="roomcall-settle")
            settling = asyncio.get_running_loop().run_in_executor(
                self._settler, functools.partial(RoomSim, self.model,
                                                 settle_seconds=self.settle_seconds))
            room = self.rooms[room_id] = Room(room_id, settling)
        return room

    def _drop_if_empty(self, room: Room) -> None:
        if room.peers or room.joining:
            return
        # Nobody is waiting for it any more: a settle still queued never runs (one already running
        # finishes on its worker and is discarded — MuJoCo cannot be interrupted mid-step).
        room.ready.cancel()
        if room.task:
            room.task.cancel()
            room.task = None
        if self.rooms.get(room.id) is room:
            del self.rooms[room.id]

    def _new_id(self, room: Room) -> str:
        while True:
            peer_id = secrets.token_hex(4)
            if peer_id not in room.peers:
                return peer_id

    # -- one connection -------------------------------------------------------------------------

    async def session(self, ws: WebSocket) -> None:
        """One browser, from handshake to goodbye. Everything after `accept` is wrapped so the
        peer always leaves cleanly — including on cancellation, which is how a dropped
        connection often arrives."""
        room_id = ws.query_params.get("room") or DEFAULT_ROOM
        await ws.accept()
        if not ROOM_ID.fullmatch(room_id):
            await self._refuse(ws, "bad room id: use 1-64 letters, digits, '_' or '-'")
            return
        existing = self.rooms.get(room_id)
        if existing is not None and len(existing.peers) >= MAX_PEERS:
            await self._refuse(ws, f"this room is full ({MAX_PEERS} people)")
            return
        if existing is None and len(self.rooms) >= MAX_ROOMS:
            await self._refuse(ws, SERVER_FULL)
            return

        name = clean_name(ws.query_params.get("name"))
        color = None
        first: dict | None = None
        try:
            first = await asyncio.wait_for(self._next(ws), HELLO_TIMEOUT)
        except asyncio.TimeoutError:
            pass
        except WebSocketDisconnect:
            return
        if first is not None and first["t"] == "hello":
            name = clean_name(first.get("name")) or name
            color = clean_color(first.get("color"))
            first = None
        early = [first] if first is not None else []

        room = self._room(room_id)
        if room is None:
            await self._refuse(ws, SERVER_FULL)
            return
        room.joining += 1
        try:
            reading = await self._until_ready(ws, room, early)
        except WebSocketDisconnect:
            # Gone while the room settled: never added, never ticked for, never broadcast to.
            room.joining -= 1
            self._drop_if_empty(room)
            return
        except Exception:  # noqa: BLE001 — a scene that cannot simulate refuses, never crashes
            log.exception("roomcall: room %s could not start", room_id)
            room.joining -= 1
            if self.rooms.get(room_id) is room and not room.peers:
                del self.rooms[room_id]
            await self._refuse(ws, "this room could not start its physics")
            return
        except BaseException:
            room.joining -= 1
            self._drop_if_empty(room)
            raise
        room.joining -= 1
        if len(room.peers) >= MAX_PEERS:
            if reading is not None:
                reading.cancel()
            self._drop_if_empty(room)
            await self._refuse(ws, f"this room is full ({MAX_PEERS} people)")
            return

        peer_id = self._new_id(room)
        peer = Peer(peer_id, ws, name or "guest", color or PALETTE[int(peer_id, 16) % len(PALETTE)])
        welcome = {"t": "welcome", "id": peer.id, "room": room.id, "name": peer.name,
                   "color": peer.color, "peers": [p.info() for p in room.peers.values()]}
        room.peers[peer.id] = peer
        try:
            peer.sender = asyncio.create_task(self._send_loop(peer))
            peer.push(dumps(welcome))
            room.broadcast({"t": "join", "peer": peer.info()}, exclude=peer.id)
            if room.task is None or room.task.done():
                room.task = asyncio.create_task(self._physics(room))
            for message in early:
                self._handle(room, peer, message)
            while True:
                if reading is not None:
                    message, reading = await reading, None
                else:
                    message = await self._next(ws)
                if message is not None:
                    self._handle(room, peer, message)
        except WebSocketDisconnect:
            pass
        finally:
            self._leave(room, peer)

    async def _until_ready(self, ws: WebSocket, room: Room,
                           early: list[dict]) -> asyncio.Task | None:
        """Wait for the room's physics to settle while still reading the client, so one that hangs
        up meanwhile is noticed now (WebSocketDisconnect) — before it is added to the room, and in
        time for `_drop_if_empty` to call off a settle nobody is waiting for. Frames it sends in the
        meantime are kept in order (up to EARLY_FRAMES) for after it joins. Raises whatever the
        settle raised.

        Returns the read still in flight, if any, for the caller to await next: cancelling it
        instead could drop a frame that had already been handed to it."""
        reader: asyncio.Task | None = None
        try:
            while not room.ready.done():
                if reader is None:
                    reader = asyncio.ensure_future(self._next(ws))
                await asyncio.wait((room.ready, reader), return_when=asyncio.FIRST_COMPLETED)
                if reader.done():
                    message = reader.result()  # raises WebSocketDisconnect if the client left
                    reader = None
                    if message is not None and len(early) < EARLY_FRAMES:
                        early.append(message)
            room.ready.result()
        except BaseException:
            if reader is not None:
                reader.cancel()  # not joining after all, so nothing it reads matters
            raise
        return reader

    @staticmethod
    async def _next(ws: WebSocket) -> dict | None:
        """The next client frame, parsed; None for one to ignore. Raises on disconnect."""
        message = await ws.receive()
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        text = message.get("text")
        return parse_message(text) if isinstance(text, str) else None  # binary: not our protocol

    @staticmethod
    async def _refuse(ws: WebSocket, reason: str) -> None:
        with contextlib.suppress(Exception):
            await ws.send_text(dumps({"t": "error", "msg": reason}))
            await ws.close(code=CLOSE_REFUSED, reason=reason[:120])

    def _leave(self, room: Room, peer: Peer) -> None:
        """Synchronous on purpose: it runs in `finally`, possibly while being cancelled."""
        if room.peers.get(peer.id) is not peer:
            return
        del room.peers[peer.id]
        if peer.sender:
            peer.sender.cancel()
        room.command("release", peer.id)
        room.broadcast({"t": "leave", "id": peer.id})
        self._drop_if_empty(room)

    async def _send_loop(self, peer: Peer) -> None:
        try:
            while True:
                await peer.wake.wait()
                peer.wake.clear()
                if peer.stuck:
                    await peer.ws.close(code=CLOSE_TOO_SLOW, reason="not keeping up")
                    return
                while peer.outbox:
                    await peer.ws.send_text(peer.outbox.popleft())
                if peer.latest_state is not None:
                    text, peer.latest_state = peer.latest_state, None
                    await peer.ws.send_text(text)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the socket went away; the receive side cleans up
            log.debug("roomcall: send to %s failed", peer.id, exc_info=True)

    # -- messages -------------------------------------------------------------------------------

    def _handle(self, room: Room, peer: Peer, message: dict) -> None:
        """Apply one client message. Anything malformed is ignored; nothing here may raise into
        the connection loop, so the whole body is guarded as a last resort."""
        try:
            self._dispatch(room, peer, message)
        except Exception:  # noqa: BLE001
            log.debug("roomcall: ignored message from %s", peer.id, exc_info=True)

    def _dispatch(self, room: Room, peer: Peer, message: dict) -> None:
        kind = message["t"]
        if kind == "pose":
            p, yaw, pitch = vec3(message.get("p")), finite(message.get("yaw")), \
                finite(message.get("pitch", 0.0))
            if p is not None and yaw is not None and pitch is not None:
                peer.pose = [round(v, 4) for v in (*p, yaw, pitch)]
        elif kind == "grab":
            body, point = message.get("body"), vec3(message.get("point"))
            if isinstance(body, int) and not isinstance(body, bool) and point is not None:
                room.command("grab", peer.id, body, point)
        elif kind == "move":
            target = vec3(message.get("target"))
            if target is not None:
                room.command("move", peer.id, target)
        elif kind == "release":
            room.command("release", peer.id)
        elif kind == "reset":
            room.command("reset")
            room.broadcast({"t": "reset", "by": peer.id})
        elif kind == "media":
            camera, mic = message.get("camera"), message.get("mic")
            if isinstance(camera, bool):
                peer.camera = camera
            if isinstance(mic, bool):
                peer.mic = mic
            room.broadcast({"t": "media", "id": peer.id, "camera": peer.camera, "mic": peer.mic},
                           exclude=peer.id)
        elif kind == "rtc":
            to = message.get("to")
            target = room.peers.get(to) if isinstance(to, str) else None
            if target is not None and target is not peer and "data" in message:
                target.push(dumps({"t": "rtc", "from": peer.id, "data": message["data"]}))
        # "hello" after joining, and unknown kinds, are ignored: names are fixed at join so every
        # client's peer list stays consistent without a rename message.

    # -- physics --------------------------------------------------------------------------------

    async def _physics(self, room: Room) -> None:
        """Real-time stepping and the 30 Hz state broadcast for one room, until it empties."""
        loop = asyncio.get_running_loop()
        sim = room.sim
        max_steps = max_steps_per_tick(sim.timestep)
        last = loop.time()
        next_state = last
        simulated = elapsed = 0.0  # for the real-time factor
        while room.peers:
            await asyncio.sleep(PHYSICS_INTERVAL)
            now = loop.time()
            want_state = now >= next_state
            resets = sim.unstable_resets
            try:
                steps, snapshot = await asyncio.to_thread(_tick, sim, room.take(), now - last,
                                                          want_state, max_steps)
            except Exception:  # noqa: BLE001
                log.exception("roomcall: room %s physics failed; resetting it", room.id)
                await self._reset_after_failure(room)
                steps, snapshot = 0, None
            else:
                if sim.unstable_resets != resets:
                    # The sim put itself back after a blow-up, and everyone's grab went with it:
                    # say so, or each client would keep dragging an object nobody holds.
                    log.warning("roomcall: room %s went unstable; reset it", room.id)
                    room.broadcast({"t": "reset", "by": "server"})
            simulated += steps * sim.timestep
            elapsed += now - last
            if elapsed >= REALTIME_WINDOW:
                room.realtime = simulated / elapsed
                simulated = elapsed = 0.0
            last = now
            if snapshot is None:
                continue
            next_state = max(next_state + STATE_INTERVAL, now)
            frame = {"t": "state", "time": snapshot.get("time", 0.0),
                     "bodies": snapshot.get("bodies", []), "grabs": snapshot.get("grabs", []),
                     "avatars": {p.id: p.pose for p in room.peers.values() if p.pose}}
            try:
                text = dumps(frame)
            except ValueError:  # non-finite pose: the sim blew up — never broadcast NaN
                log.warning("roomcall: room %s produced a non-finite state; resetting it", room.id)
                await self._reset_after_failure(room)
                continue
            for peer in room.peers.values():
                peer.push_state(text)

    async def _reset_after_failure(self, room: Room) -> None:
        room.take()  # whatever was queued is moot once the room is put back
        with contextlib.suppress(Exception):
            await asyncio.to_thread(room.sim.reset)
        room.broadcast({"t": "reset", "by": "server"})


# ---------------------------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------------------------

def create_app(scene_xml: Path | None = None, *, model: mujoco.MjModel | None = None,
               label: str = "room", web_dir: Path | None = None,
               settle_seconds: float = 1.0) -> FastAPI:
    """The room call app for one scene: geometry built here, once; rooms made on demand.

    `web_dir` exists for tests (a stand-in client) and for working on the client from a checkout;
    `settle_seconds` is passed to every new room's `RoomSim`.
    """
    if model is None:
        if scene_xml is None:
            raise ValueError("create_app needs a scene_xml or a model")
        model = load_model(Path(scene_xml))
    web = WebAssets.load(Path(web_dir) if web_dir is not None else WEB_DIR)
    glb, manifest = build_scene_assets(model, label=label)
    scene_json = json.dumps({**manifest, "glb": "scene.glb"}).encode("utf-8")
    glb_etag = '"' + hashlib.sha256(glb).hexdigest()[:24] + '"'
    hub = RoomHub(model, settle_seconds=settle_seconds)

    # No /docs, /redoc or /openapi.json: nothing to document for a browser, and less surface.
    app = FastAPI(title="LiteReality room call", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=hub.lifespan)
    app.add_middleware(SecurityHeaders, headers=security_headers(web.script_hashes))
    app.state.roomcall = hub
    app.state.web = web

    def file_response(name: str, media_type: str) -> Response:
        body = web.asset(name)
        if body is None:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return Response(body, media_type=media_type, headers={"Cache-Control": "no-cache"})

    @app.get("/")
    async def index() -> Response:
        # no-store: the CSP hash belongs to this server's copy; a restart may change both.
        return Response(web.index, media_type=HTML, headers={"Cache-Control": "no-store"})

    @app.get("/app.js")
    async def app_js() -> Response:
        return file_response("app.js", JS)

    @app.get("/app.css")
    async def app_css() -> Response:
        return file_response("app.css", CSS)

    @app.get("/scene.json")
    async def scene_manifest() -> Response:
        return Response(scene_json, media_type="application/json",
                        headers={"Cache-Control": "no-cache"})

    @app.get("/scene.glb")
    async def scene_glb(request: Request) -> Response:
        # Revalidated, not blindly cached: a different scene can come up on the same port.
        headers = {"ETag": glb_etag, "Cache-Control": "no-cache"}
        if request.headers.get("if-none-match") == glb_etag:
            return Response(status_code=304, headers=headers)
        return Response(glb, media_type="model/gltf-binary", headers=headers)

    @app.get("/healthz")
    async def healthz() -> Response:
        return JSONResponse({"ok": True, "label": label, **hub.stats()},
                            headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await hub.session(ws)

    return app


# ---------------------------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------------------------

def resolve_scene(path: Path | str) -> Path:
    """A scene.xml, or a folder holding scene.xml / mujoco_seed/scene.xml / mujoco/scene.xml."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for name in SCENE_CANDIDATES:
            if (candidate / name).is_file():
                return candidate / name
        raise ValueError(f"no scene in {candidate}: looked for {', '.join(SCENE_CANDIDATES)} — "
                         "export one with `litereality stage simulate <run> --from-seed`")
    raise ValueError(f"no such scene: {candidate}")


def scene_label(scene: Path) -> str:
    """run/tea_room/mujoco_seed/scene.xml -> "tea_room": the room, not the exporter's folder."""
    for folder in [scene.parent, *scene.parent.parents]:
        if folder.name and folder.name not in _EXPORT_DIRS:
            return folder.name
    return "room"


def bind_socket(host: str, port: int) -> socket.socket:
    """The first free port at or above `port`, like the other viewers (`room_ops.serve.bind`):
    an older server still being up is no reason to refuse to start."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    for candidate in range(port, port + PORT_TRIES):
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, candidate))
        except OSError as exc:
            sock.close()
            if exc.errno != errno.EADDRINUSE:
                raise
            continue
        return sock
    raise SystemExit(f"no free port in {port}-{port + PORT_TRIES - 1} on {host}")


def _open_when_up(server, url: str) -> None:
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    with contextlib.suppress(Exception):  # headless or no handler; the url is printed either way
        webbrowser.open(url)


def serve(scene_xml: Path | None, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          label: str | None = None, demo: bool = False, open_browser: bool = True) -> int:
    """Serve one scene (or the demo room, with `demo` or no scene) until interrupted."""
    import uvicorn

    with tempfile.TemporaryDirectory(prefix="roomcall-demo-") as scratch:
        if demo or scene_xml is None:
            from litereality_agent.room_ops.roomcall.demo import write_demo_scene

            scene = write_demo_scene(Path(scratch))
            label = label or "demo room"
        else:
            scene = resolve_scene(scene_xml)
            label = label or scene_label(scene)
        print(f"room call  building {label} from {scene} …")
        model = load_model(scene)
        app = create_app(scene, model=model, label=label)
        sock = bind_socket(host, port)
        shown = "localhost" if host in {"0.0.0.0", "::", ""} else host
        url = f"http://{shown}:{sock.getsockname()[1]}/"
        config = uvicorn.Config(app, log_level="warning", ws_max_size=WS_MAX_SIZE,
                                ws_ping_interval=WS_PING_INTERVAL,
                                ws_ping_timeout=WS_PING_TIMEOUT)
        server = uvicorn.Server(config)
        print(f"room call  {url}?room={DEFAULT_ROOM}")
        print(f"  scene  {scene}  ({len(movable_body_ids(model))} movable bodies)  ·  "
              f"up to {MAX_PEERS} people per room")
        print(f"  share  expose it over https (e.g. `cloudflared tunnel --url {url.rstrip('/')}`) "
              "and paste the link in the Zoom/Meet/Teams chat · ctrl-c to stop")
        if open_browser:
            threading.Thread(target=_open_when_up, args=(server, url), daemon=True).start()
        try:
            server.run(sockets=[sock])
        except KeyboardInterrupt:
            pass
        finally:
            sock.close()
    print("stopped")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m litereality_agent.room_ops.roomcall",
                                     description="a shared physics room for a video call")
    add_arguments(parser)
    args = parser.parse_args(argv)
    return run(args)


def run(args) -> int:
    """Run from parsed `add_arguments` arguments — shared by `litereality roomcall` and
    `python -m litereality_agent.room_ops.roomcall` so the two can never drift apart."""
    try:
        return serve(Path(args.scene) if args.scene else None, host=args.host, port=args.port,
                     label=args.label, demo=args.demo, open_browser=args.open_browser)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
