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

Versions and live swaps. A room's scene can get better while people are in it — RoomPlan boxes
seconds after the scan, the full pipeline's room later — and each stage is just a new scene.xml.
Every scene the server holds is a `SceneVersion` with a content-derived id, and that id travels
with everything that depends on it: body ids in "state" frames and "grab" messages are integers
that mean something only within one version, so both carry "v". A new version (from `--watch`
here, or from the Volume when hosted) is loaded entirely off the event loop — compiled on a worker
thread (MuJoCo releases the GIL), its geometry built in a child process unless prebuilt files came
with it, settled once on the settle pool — then OFFERED: every room is sent {"t": "scene"} so
browsers fetch it, and after `PREFETCH_SECONDS` each room's own physics task swaps between two
ticks, carrying what people did across by body name (`RoomSim.adopt`). The first state frame with
the new "v" is the switch; nobody reconnects. A version that fails to load is reported on /healthz
and the room carries on with the one it has.

Full-body people (`avatars`). A person may POST /avatar a turnaround of themselves or a few photos
(authorised by the upload key in their welcome, sent in headers so a stranger's upload is refused
unread); the hub's `AvatarService` builds a body in the
background — on the Modal GPU function, or the stand-in — and tells the room ({"t": "avatar"}),
and every browser fetches it from GET /avatars/<id>.bin. Bodies live in memory only. A tracked
sender's pose message carries "j" (its arms and head), which rides on its row of the state frames.

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
import gc
import hashlib
import json
import logging
import math
import os
import re
import secrets
import signal
import socket
import sys
import tempfile
import threading
import time
import webbrowser
import xml.etree.ElementTree as ET
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
from litereality_agent.room_ops.roomcall import avatars as bodies
from litereality_agent.room_ops.roomcall.avatars import (
    AvatarConfig,
    AvatarRequestError,
    AvatarService,
)
from litereality_agent.room_ops.roomcall.geometry import build_scene_assets
from litereality_agent.room_ops.roomcall.hosted import scene_digest, scene_files, version_name
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

# How long browsers get to fetch a new version before rooms switch to it (seconds). The switch is
# the first state frame carrying the new "v"; a client that has not finished loading by then keeps
# drawing the old room (its bodies frozen) until it has, so this only has to cover most people.
PREFETCH_SECONDS = 5.0
# How often `--watch` looks at the scene's files (seconds). A change is loaded once it has looked
# the same twice in a row, so a scene caught halfway through being written is not served.
WATCH_SECONDS = 0.5
# A geometry build in the child process that takes longer than this is abandoned (seconds).
BUILD_TIMEOUT = 15 * 60
# Niceness of the geometry child (`build_assets`). A big scene's build is ~30 s of busy cores; on a
# 2-CPU container the event loop and the rooms' physics must win every contest for a core against
# it, or people feel the load as stutter. Only the child: the compile thread shares this process's
# GIL, holds it in short stretches, and a deprioritised thread preempted while holding it stalls
# the loop instead (measured: worst 50-58 ms niced, 35-39 ms not, idle machine 35-40 ms).
BACKGROUND_NICE = 10
# Versions are content-addressed, so a version's glb never changes: browsers may keep it forever.
IMMUTABLE = "public, max-age=31536000, immutable"

# Most finished first: a folder holding both the full pipeline's scene and the quick room serves
# the full one, so `--watch` on a run folder moves from the quick room to the full one by itself.
SCENE_CANDIDATES = ("scene.xml", "mujoco_seed/scene.xml", "mujoco/scene.xml", "objects/scene.xml",
                    "quick/scene.xml")
# Folder names that say how a scene was exported, not which room it is — skipped for the label.
_EXPORT_DIRS = {"mujoco_seed", "mujoco", "objects", "quick"}
# ...and which stage of the room each one is (`scene_stage`).
_STAGE_DIRS = {"mujoco_seed": "full", "mujoco": "full", "objects": "objects", "quick": "quick"}

# The prebuilt-geometry files a published version folder carries (`prebuilt_files`).
PREBUILT_GLB = "scene.glb"
PREBUILT_MANIFEST = "scene.json"
PREBUILT_META = "roomcall.json"

THREE_CDN = "https://cdn.jsdelivr.net"
ZOOM_SDK = "https://appssdk.zoom.us"
ZOOM_FRAMES = "https://zoom.us https://*.zoom.us"

# Distinct, readable on a light or dark room, and never the same for two ids in a row by accident.
PALETTE = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#9a6324", "#469990", "#800000", "#808000", "#000075")

JS = "text/javascript; charset=utf-8"
CSS = "text/css; charset=utf-8"
HTML = "text/html; charset=utf-8"
BINARY = "application/octet-stream"

# The hologram's person-segmentation model: MediaPipe's selfie segmenter (float16, 250 KB,
# Apache-2.0, from storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/
# float16/latest/). It is vendored in web/ and served from here under this fixed name, so a browser
# never fetches it from Google at runtime. The runtime that runs it comes from jsDelivr, pinned in
# app.js like three.js.
SEGMENTER_MODEL = "selfie_segmenter.tflite"
# Full-body people (`avatars`): the sender's upper body is tracked in its own browser by
# MediaPipe's PoseLandmarker (lite, float16, 5.8 MB, Apache-2.0, from storage.googleapis.com/
# mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/), vendored and served like the
# segmenter. Its joint rotations ride on each "pose" message as "j": SENT_JOINTS rotation vectors
# in centiradians (web/body.js `packJoints`), each an integer within JOINT_LIMIT.
POSE_MODEL = "pose_landmarker_lite.task"
SENT_JOINTS = 13
JOINT_LIMIT = 400
# An avatar file never changes once built (a new body is a new id), so a browser keeps it.
AVATAR_CACHE = "private, max-age=86400, immutable"


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

    - script-src: no 'unsafe-inline' and no 'unsafe-eval'. The import map is let in by its hash.
      'wasm-unsafe-eval' lets the hologram's segmenter compile its WebAssembly runtime (MediaPipe,
      from jsDelivr). It allows WebAssembly only: JavaScript eval and new Function stay blocked.
    - style-src keeps 'unsafe-inline': styles are not code, and the client may set style
      attributes from JS; forbidding that buys nothing here and breaks UI in surprising places.
    - connect-src: ws:/wss: for the room socket (also behind a tunnel on another host), blob:/data:
      because GLTFLoader fetches its embedded textures through blob URLs. The segmenter's .wasm
      comes from jsDelivr and its model from 'self'; both are already covered.
    - img/media: blob:/data: for textures and webcam video. frame-ancestors lets Zoom (desktop and
      the zoom.us web client) embed the page, and nobody else.
    """
    hashes = " ".join(f"'sha256-{h}'" for h in script_hashes)
    directives = {
        "default-src": "'self'",
        "script-src": " ".join(filter(None, ("'self'", "'wasm-unsafe-eval'", hashes, THREE_CDN,
                                              ZOOM_SDK))),
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
    folder app.js, app.css and the segmenter model are read from per request."""

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
        """One of the vendored files by fixed name. The name never comes from a request."""
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


def clean_joints(value: Any) -> list[int] | None:
    """A pose message's "j": exactly 3 * SENT_JOINTS integers within JOINT_LIMIT, else None.
    Floats are accepted only when they are whole (JSON has one number type); bools never."""
    if not isinstance(value, list) or len(value) != 3 * SENT_JOINTS:
        return None
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if isinstance(v, float) and not (math.isfinite(v) and v.is_integer()):
            return None
        if abs(v) > JOINT_LIMIT:
            return None
        out.append(int(v))
    return out


# ---------------------------------------------------------------------------------------------
# Scene versions
# ---------------------------------------------------------------------------------------------

def scene_stage(scene: Path) -> str:
    """Which stage of a room a scene file is, from the folder the pipeline wrote it to:
    run/<scan>/quick -> "quick", run/<scan>/mujoco_seed -> "full", anything else "scene"."""
    return _STAGE_DIRS.get(Path(scene).parent.name, "scene")


def geometry_fingerprint() -> str:
    """Which glb + manifest this checkout's builder makes: a digest of its source. Prebuilt
    geometry from a laptop running other code than the server is rebuilt rather than trusted."""
    from litereality_agent.room_ops.roomcall import geometry, model_info

    digest = hashlib.sha256()
    for module in (geometry, model_info):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()[:12]


def assets_meta(*, version: str, label: str, stage: str) -> dict:
    """roomcall.json: what a prebuilt version folder was built from and with."""
    return {"version": version, "label": label, "stage": stage, "mujoco": mujoco.__version__,
            "geometry": geometry_fingerprint(), "created": round(time.time(), 3)}


def prebuilt_files(scene_xml: Path, *, version: str, label: str, stage: str,
                   model: mujoco.MjModel | None = None) -> dict[str, bytes]:
    """The geometry a published version carries, built where the scene was published (the
    laptop) so the server that serves it does not have to: {file name: bytes} for
    `PREBUILT_GLB`, `PREBUILT_MANIFEST` and `PREBUILT_META`. Compiling the scene here is also the
    check that it loads at all, before anyone else finds out."""
    model = load_model(Path(scene_xml)) if model is None else model
    glb, manifest = build_scene_assets(model, label=label)
    meta = assets_meta(version=version, label=label, stage=stage)
    return {PREBUILT_GLB: glb,
            PREBUILT_MANIFEST: json.dumps(manifest).encode("utf-8"),
            PREBUILT_META: json.dumps(meta, indent=1).encode("utf-8")}


def _prebuilt_files(folder: Path | None) -> tuple[bytes, dict] | None:
    """A version folder's prebuilt glb + manifest, if they were built by this MuJoCo (body ids
    and mesh processing are its compiler's) and this geometry code; else None."""
    if folder is None:
        return None
    folder = Path(folder)
    try:
        meta = json.loads((folder / PREBUILT_META).read_text(encoding="utf-8"))
        if not isinstance(meta, dict) or meta.get("mujoco") != mujoco.__version__ \
                or meta.get("geometry") != geometry_fingerprint():
            return None
        manifest = json.loads((folder / PREBUILT_MANIFEST).read_text(encoding="utf-8"))
        glb = (folder / PREBUILT_GLB).read_bytes()
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or glb[:4] != b"glTF":
        return None
    return glb, manifest


def _fits(manifest: dict, model: mujoco.MjModel) -> bool:
    """Whether a manifest's bodies are exactly this model's movable bodies."""
    try:
        return {int(body["id"]) for body in manifest["bodies"]} == movable_body_ids(model)
    except (KeyError, TypeError, ValueError):
        return False


def read_prebuilt(folder: Path | None, model: mujoco.MjModel) -> tuple[bytes, dict] | None:
    """`_prebuilt_files`, if they also fit `model`. Anything else is None — build it."""
    assets = _prebuilt_files(folder)
    return assets if assets is not None and _fits(assets[1], model) else None


@dataclass(frozen=True)
class SceneRef:
    """A scene to load as a version: where it is, its id, and how to present it. `prebuilt` is a
    folder that may hold its geometry already (`read_prebuilt`)."""

    scene: Path
    id: str
    label: str
    stage: str = "scene"
    prebuilt: Path | None = None


@dataclass(eq=False)
class SceneVersion:
    """One version of the room: its compiled model, the glb and manifest browsers draw, and its
    settled starting state, computed once and shared by every room on it.

    The manifest a browser fetches (`scene_json`) names the glb by its hash, so a version's glb is
    cached forever and two versions can never be mixed up in a cache.
    """

    id: str
    label: str
    stage: str
    model: mujoco.MjModel
    glb: bytes
    manifest: dict
    glb_hash: str = ""
    scene_json: bytes = b""
    settling: asyncio.Future | None = None    # the one settle, on the settle pool (`RoomHub.offer`)
    _settled: Any = None                      # its result: RoomSim.settled_state()
    _spare: RoomSim | None = None             # the sim that settled, for the first room that asks
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(cls, model: mujoco.MjModel, glb: bytes, manifest: dict, *, label: str,
               stage: str = "scene", id: str | None = None) -> SceneVersion:
        glb_hash = hashlib.sha256(glb).hexdigest()[:24]
        vid = id or version_name(label, glb_hash)
        manifest = {**manifest, "label": label}
        scene_json = json.dumps({**manifest, "v": vid, "stage": stage,
                                 "glb": f"scene.glb?h={glb_hash}"}).encode("utf-8")
        return cls(vid, label, stage, model, glb, manifest, glb_hash, scene_json)

    @classmethod
    def build(cls, model: mujoco.MjModel, *, label: str, stage: str = "scene",
              id: str | None = None, prebuilt: Path | None = None) -> SceneVersion:
        """In this thread: for startup, before anyone is connected (`load_version` is the
        off-the-event-loop way for a running server)."""
        assets = read_prebuilt(prebuilt, model)
        glb, manifest = assets if assets is not None else build_scene_assets(model, label=label)
        return cls.create(model, glb, manifest, label=label, stage=stage, id=id)

    def message(self) -> dict:
        """The "fetch this version now" announcement."""
        return {"t": "scene", "v": self.id, "label": self.label, "stage": self.stage}

    def settle(self, settle_seconds: float) -> None:
        """Settle this version once (blocking: run it on the settle pool). Settling is
        deterministic, so every room on this version starts from the one state it produces; the
        sim that did it is kept for the first room that asks."""
        with self._lock:
            if self._settled is None:
                sim = RoomSim(self.model, settle_seconds=settle_seconds)
                self._settled = sim.settled_state()
                self._spare = sim

    def new_sim(self, settle_seconds: float) -> RoomSim:
        """A fresh room on this version at its settled state (blocking: settles first if nobody
        has yet)."""
        with self._lock:
            if self._settled is None:
                sim = RoomSim(self.model, settle_seconds=settle_seconds)
                self._settled = sim.settled_state()
                return sim
            spare, self._spare = self._spare, None
            settled = self._settled
        return spare if spare is not None else RoomSim(self.model, settled=settled)


def _child_build(scene: str, out: str, label: str) -> None:
    """Child-process half of `build_assets`: compile, build, write the two files, exit."""
    with contextlib.suppress(OSError, AttributeError):
        os.nice(BACKGROUND_NICE)
    model = load_model(Path(scene))
    glb, manifest = build_scene_assets(model, label=label)
    (Path(out) / PREBUILT_GLB).write_bytes(glb)
    (Path(out) / PREBUILT_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


_CHILD = ("import sys; from litereality_agent.room_ops.roomcall.server import _child_build; "
          "_child_build(*sys.argv[1:4])")


async def build_assets(scene: Path, label: str, *, timeout: float = BUILD_TIMEOUT
                       ) -> tuple[bytes, dict]:
    """A scene's glb + manifest, built in a child process.

    Not in a thread: `build_scene_assets` is numpy and trimesh holding the GIL, and on a real room
    (~200 geoms, ~300 MB of meshes) a thread stalled the event loop that carries everyone's
    messages by ~0.8 s. In a child (at `BACKGROUND_NICE`), alongside the compile thread, the whole
    ~30 s load kept the loop's worst tick ~50 ms late, p99 ~5 ms — no worse than an idle probe in
    another process on the same machine. A plain subprocess rather than a
    multiprocessing pool: spawn would re-import the parent's `__main__` (inside a Modal container
    that is Modal's own entrypoint), and pool results come back pickled through the loop's
    process. The child writes two files and exits, handing the several GB a big build peaks at
    back to the OS.
    """
    # The folder holding the `litereality_agent` package, so the child imports this very copy.
    package_root = str(Path(__file__).resolve().parents[3])
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(filter(None, (package_root, os.environ.get("PYTHONPATH"))))}
    with tempfile.TemporaryDirectory(prefix="roomcall-build-") as out:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", _CHILD, str(scene), out, label, env=env,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE)
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.shield(proc.wait())
            raise
        if proc.returncode != 0:
            lines = [line for line in err.decode("utf-8", "replace").splitlines() if line.strip()]
            raise RuntimeError(f"building the room's geometry failed: "
                               f"{lines[-1] if lines else f'exit code {proc.returncode}'}")
        glb = await asyncio.to_thread((Path(out) / PREBUILT_GLB).read_bytes)
        text = await asyncio.to_thread((Path(out) / PREBUILT_MANIFEST).read_text, "utf-8")
    return glb, json.loads(text)


async def load_version(ref: SceneRef) -> SceneVersion:
    """Load a scene as a version without ever blocking the event loop: the compile runs on a
    worker thread (MuJoCo releases the GIL while it compiles) while the geometry comes from the
    prebuilt folder or a child process (`build_assets`), both at once. Whichever fails first
    fails the load and stops the other (a scene that does not compile kills its build)."""
    async def geometry() -> tuple[bytes, dict] | None:
        assets = await asyncio.to_thread(_prebuilt_files, ref.prebuilt)
        return assets if assets is not None else await build_assets(ref.scene, ref.label)

    compiling = asyncio.ensure_future(asyncio.to_thread(load_model, ref.scene))
    building = asyncio.ensure_future(geometry())
    try:
        done, _ = await asyncio.wait((compiling, building), return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            if task.exception() is not None:
                raise task.exception()
        model, assets = compiling.result(), building.result()
        if not _fits(assets[1], model):  # prebuilt for another model after all: build it
            assets = await build_assets(ref.scene, ref.label)
    finally:
        for task in (compiling, building):
            if not task.done():
                task.cancel()
        for task in (compiling, building):  # never leave an exception unretrieved
            if task.done() and not task.cancelled():
                task.exception()
    glb, manifest = assets
    if not _fits(manifest, model):
        raise RuntimeError("the geometry was built from another model than the one compiled "
                           "(the scene changed while loading?)")
    return await asyncio.to_thread(SceneVersion.create, model, glb, manifest, label=ref.label,
                                   stage=ref.stage, id=ref.id)


class LocalWatch:
    """`--watch`: the SCENE path re-read whenever its files change, each change a new version.

    Polls mtimes and sizes of the scene file and everything it names (`scene_files`) every
    `poll_seconds`; a change is loaded once two looks in a row agree, so a file caught mid-write
    is not served. A folder is re-resolved every time, so a run folder moves on from
    quick/scene.xml to mujoco_seed/scene.xml as soon as the full pipeline has written it. The id
    is the content digest, so touching a file without changing it is not a new version.
    """

    def __init__(self, path: Path, *, label: str, stage: str | None = None,
                 poll_seconds: float = WATCH_SECONDS) -> None:
        self.path = Path(path)
        self.label = label
        self.stage = stage
        self.poll_seconds = poll_seconds
        self._candidate: tuple | None = None
        try:
            self._seen: tuple | None = self._signature()
        except (ValueError, OSError):
            self._seen = None

    def _signature(self) -> tuple:
        scene = resolve_scene(self.path)
        try:
            files = [local for local, _ in scene_files(scene)]
        except ET.ParseError as exc:
            raise ValueError(f"{scene} is not readable XML yet: {exc}") from None
        stats = []
        for local in files:
            try:
                st = local.stat()
                stats.append((str(local), st.st_mtime_ns, st.st_size))
            except OSError:
                stats.append((str(local), None, None))
        return (str(scene), tuple(sorted(stats)))

    async def check(self) -> SceneRef | None:
        return await asyncio.to_thread(self._check)

    def _check(self) -> SceneRef | None:
        signature = self._signature()   # ValueError: no scene there right now, or files missing
        if signature == self._seen:
            self._candidate = None
            return None
        if signature != self._candidate:
            self._candidate = signature  # changed: look once more before trusting it
            return None
        self._seen, self._candidate = signature, None
        scene = Path(signature[0])
        return SceneRef(scene=scene, id=version_name(self.label, scene_digest(scene)),
                        label=self.label, stage=self.stage or scene_stage(scene))


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
    joints: list[int] | None = None  # the tracked upper body ("j"), while the sender tracks it
    # This connection's upload key (in its welcome): POST /avatar must name it.
    key: str = field(default_factory=lambda: secrets.token_hex(16))
    avatar_id: str | None = None     # its body (`AvatarService` keeps these two)
    avatar_info: dict | None = None
    outbox: deque[str] = field(default_factory=deque)
    latest_state: str | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    stuck: bool = False
    sender: asyncio.Task | None = None

    def info(self) -> dict:
        info = {"id": self.id, "name": self.name, "color": self.color,
                "media": {"camera": self.camera, "mic": self.mic}}
        if self.avatar_info is not None:
            info["avatar"] = dict(self.avatar_info)
        return info

    def row(self) -> list:
        """Its state frame entry: the pose, and the tracked joints after it when there are any."""
        return self.pose + self.joints if self.joints else self.pose

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
    ready: asyncio.Future  # -> RoomSim, settling on the settle pool (-> None once it runs)
    version: SceneVersion | None = None  # the version `sim` is on: every frame's "v"
    pending: SceneVersion | None = None  # offered and announced, not swapped in yet
    offered_at: float = 0.0              # loop time of that offer
    peers: dict[str, Peer] = field(default_factory=dict)
    commands: deque[list] = field(default_factory=deque)
    joining: int = 0
    task: asyncio.Task | None = None
    realtime: float | None = None  # simulated seconds per wall second, last REALTIME_WINDOW
    _newest: dict[str, list] = field(default_factory=dict)  # each peer's newest queued command
    _sim: RoomSim | None = None    # the room's sim once its physics runs (`ready` lets go of it)

    @property
    def sim(self) -> RoomSim:
        return self._sim if self._sim is not None else self.ready.result()

    @property
    def v(self) -> str | None:
        return self.version.id if self.version is not None else None

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


def _apply(sim: RoomSim, commands: deque[list]) -> None:
    while commands:
        op, *args = commands.popleft()
        try:
            getattr(sim, op)(*args)
        except Exception:  # noqa: BLE001 — one bad command must not stop the room
            log.exception("roomcall: %s%r failed", op, tuple(args))


def _tick(sim: RoomSim, commands: deque[list], dt: float, want_state: bool,
          max_steps: int) -> tuple[int, dict | None]:
    """Worker-thread half of one physics tick: apply queued commands in order, step, snapshot.
    Returns the steps taken and the snapshot (None unless `want_state`)."""
    _apply(sim, commands)
    steps = sim.step_realtime(dt, max_steps=max_steps)
    return steps, (sim.snapshot() if want_state else None)


def _swap_sims(old: RoomSim, commands: deque[list], version: SceneVersion,
               settle_seconds: float) -> tuple[RoomSim, dict]:
    """Worker-thread half of a swap: first what was asked of the old version (a grab sent just
    before the switch is carried across, not lost), then a fresh room on the new version, from
    its settled state, takes over the old room's (`RoomSim.adopt`)."""
    _apply(old, commands)
    new = version.new_sim(settle_seconds)
    return new, new.adopt(old)


class RoomHub:
    """Every live room of one server, keyed by room id. Rooms on one version share its model;
    `version` is the live one (what /scene.json serves and new rooms start on), `versions` every
    one still needed — live, or still being run or fetched by some room."""

    def __init__(self, version: SceneVersion | mujoco.MjModel, *, settle_seconds: float = 1.0,
                 prefetch_seconds: float | None = None, watch: Any = None,
                 avatars: AvatarConfig | None = None) -> None:
        if isinstance(version, mujoco.MjModel):  # a bare model: a version with nothing to draw
            version = SceneVersion("scene", "room", "scene", version, b"", {})
        self.version = version
        self.versions: dict[str, SceneVersion] = {version.id: version}
        self.settle_seconds = settle_seconds
        self.prefetch_seconds = PREFETCH_SECONDS if prefetch_seconds is None else prefetch_seconds
        # Anything with `poll_seconds` and `async check() -> SceneRef | None`: `LocalWatch` for
        # `--watch`, `hosted.VolumeWatch` on Modal.
        self.watch = watch
        self.rooms: dict[str, Room] = {}
        self._settler: ThreadPoolExecutor | None = None
        self.last_check: float | None = None   # unix time of the watcher's last look
        self.loading: str | None = None        # a version being loaded right now
        self._check_error: str | None = None
        self._load_error: str | None = None
        self._failed: str | None = None        # a version id that failed: not tried again
        self._watcher: asyncio.Task | None = None
        self._watcher_loop: asyncio.AbstractEventLoop | None = None
        # Full bodies: one store and one set of limits for every room of this server.
        self.avatars = AvatarService(avatars or AvatarConfig(), find_peer=self._find_peer,
                                     broadcast=self._broadcast_room,
                                     tell=lambda peer, message: peer.push(dumps(message)))

    def _find_peer(self, room_id: str, peer_id: str) -> Peer | None:
        room = self.rooms.get(room_id)
        return room.peers.get(peer_id) if room is not None else None

    def _broadcast_room(self, room_id: str, message: dict) -> None:
        room = self.rooms.get(room_id)
        if room is not None:
            room.broadcast(message)

    @property
    def model(self) -> mujoco.MjModel:
        return self.version.model

    @property
    def version_error(self) -> str | None:
        return self._load_error or self._check_error

    # -- lifecycle ------------------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def lifespan(self, _app: FastAPI):
        self.ensure_watching()
        try:
            yield
        finally:
            if self._watcher is not None:
                self._watcher.cancel()
                self._watcher = None
            for room in list(self.rooms.values()):
                room.ready.cancel()
                if room.task:
                    room.task.cancel()
                for peer in room.peers.values():
                    if peer.sender:
                        peer.sender.cancel()
            self.rooms.clear()
            self.avatars.close()
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

    def status(self) -> dict:
        """The version half of /healthz."""
        return {"version": self.version.id, "stage": self.version.stage,
                "last_check": round(self.last_check, 3) if self.last_check else None,
                "loading": self.loading, "version_error": self.version_error}

    def _settle_pool(self) -> ThreadPoolExecutor:
        if self._settler is None:
            self._settler = ThreadPoolExecutor(SETTLE_WORKERS, thread_name_prefix="roomcall-settle")
        return self._settler

    def _room(self, room_id: str) -> Room | None:
        """The room, made and set settling if new; None if that would exceed MAX_ROOMS."""
        room = self.rooms.get(room_id)
        if room is None:
            if len(self.rooms) >= MAX_ROOMS:
                return None
            version = self.version
            settling = asyncio.get_running_loop().run_in_executor(
                self._settle_pool(), functools.partial(version.new_sim, self.settle_seconds))
            room = self.rooms[room_id] = Room(room_id, settling, version=version)
        return room

    # -- versions -------------------------------------------------------------------------------

    def ensure_watching(self) -> None:
        """Start the version watcher on this event loop unless it already runs there.

        The app's lifespan calls it at startup. Modal's `asgi_app` does run the ASGI lifespan
        (modal/_runtime/asgi.py: `LifespanManager`, started before the first input, on the same
        event loop as every request), but every route calls this too, which costs a comparison:
        a host that skips the lifespan still gets its watcher, on the first request."""
        if self.watch is None:
            return
        loop = asyncio.get_running_loop()
        task = self._watcher
        if task is not None and not task.done() and self._watcher_loop is loop:
            return
        self._watcher = loop.create_task(self._watching())
        self._watcher_loop = loop

    async def _watching(self) -> None:
        """Every `watch.poll_seconds`: ask the source for a newer version, load it, offer it.
        Never raises: a failed look or load is reported on /healthz and the next look carries on."""
        reported: str | None = None
        while True:
            await asyncio.sleep(max(0.05, float(self.watch.poll_seconds)))
            try:
                ref = await self.watch.check()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                message = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
                self._check_error = message[:500]
                if message != reported:
                    log.warning("roomcall: looking for a new version: %s", message)
                    reported = message
                continue
            finally:
                self.last_check = time.time()
            self._check_error = reported = None
            if ref is None or ref.id == self.version.id or ref.id == self._failed:
                continue
            await self.load_and_offer(ref)

    async def load_and_offer(self, ref: SceneRef) -> SceneVersion | None:
        """Load `ref` off the event loop (`load_version`) and offer it to every room. One that
        fails is reported (/healthz version_error) and not tried again: rooms carry on with what
        they have until a newer version comes.

        A big scene loads for tens of seconds, and the source may name another version by the time
        it is done (the full room published while the quick one loaded): it is asked once more
        before the offer, and if it has moved on, the newer one is loaded instead of offering one
        that is already stale — rooms swap once, not twice."""
        while True:
            self.loading = ref.id
            started = time.perf_counter()
            try:
                version = await load_version(ref)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._failed = ref.id
                self._load_error = f"{ref.id}: {exc}"[:500]
                log.warning("roomcall: version %s could not be loaded, still serving %s: %s",
                            ref.id, self.version.id, exc)
                return None
            finally:
                self.loading = None
            log.info("roomcall: loaded version %s (%s) in %.1f s", version.id, version.stage,
                     time.perf_counter() - started)
            newer = await self._moved_on(ref)
            if newer is None:
                break
            log.info("roomcall: version %s was superseded by %s while it loaded; not offering it",
                     ref.id, newer.id)
            if newer.id in (self.version.id, self._failed):
                return None
            ref = newer
        self.offer(version)
        return version

    async def _moved_on(self, ref: SceneRef) -> SceneRef | None:
        """What the watch source names now, if that is no longer `ref`; None if it still is, or
        cannot say (no source, a round it skips, a look that fails: offer what was loaded)."""
        if self.watch is None:
            return None
        try:
            now = await self.watch.check()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the regular look reports it
            return None
        finally:
            self.last_check = time.time()
        return now if now is not None and now.id != ref.id else None

    def offer(self, version: SceneVersion) -> None:
        """Make `version` the live one — what /scene.json serves and new rooms start on — and
        offer it to every room: each is told {"t": "scene"} at once, so browsers start fetching
        it, while it settles (once, on the settle pool); each room's physics task swaps it in
        when both are done (`_swap_due`). Offering the version a room already runs withdraws
        anything pending there; a newer offer replaces an older one still pending, so a room may
        skip versions but never goes backwards."""
        version = self.versions.setdefault(version.id, version)
        if version is self.version and all(room.version is version and room.pending is None
                                           for room in self.rooms.values()):
            return
        self.version = version
        self._failed = None
        self._load_error = None
        loop = asyncio.get_running_loop()
        if version.settling is None:
            version.settling = loop.run_in_executor(
                self._settle_pool(), functools.partial(version.settle, self.settle_seconds))
            version.settling.add_done_callback(functools.partial(self._settled, version))
        now = loop.time()
        message = version.message()
        for room in self.rooms.values():
            room.pending = None if room.version is version else version
            room.offered_at = now
            room.broadcast(message)
        self._prune()

    def _settled(self, version: SceneVersion, future: asyncio.Future) -> None:
        """A version that cannot even settle is withdrawn (it never happens with a scene that
        compiled — `RoomSim` settles any scene — but it must not take rooms down if it does)."""
        if future.cancelled() or future.exception() is None:
            return
        log.error("roomcall: version %s could not settle", version.id, exc_info=future.exception())
        self._failed = version.id
        self._load_error = f"{version.id}: could not settle: {future.exception()}"[:500]
        for room in self.rooms.values():
            if room.pending is version:
                room.pending = None
        if self.version is version:
            fallback = next((room.version for room in self.rooms.values()
                             if room.version is not None and room.version is not version), None)
            if fallback is not None:
                self.version = fallback
        self._prune()

    def _prune(self) -> None:
        """Forget versions nothing needs any more (a model and its glb can be hundreds of MB)."""
        keep = {self.version.id}
        for room in self.rooms.values():
            keep.update(v.id for v in (room.version, room.pending) if v is not None)
        for vid in [vid for vid in self.versions if vid not in keep]:
            del self.versions[vid]

    def glb_version(self, glb_hash: str) -> SceneVersion | None:
        return next((v for v in self.versions.values() if v.glb_hash == glb_hash), None)

    def _swap_due(self, room: Room, now: float) -> bool:
        """Whether this room should swap to its pending version now: browsers have had
        `prefetch_seconds` and the version has settled."""
        pending = room.pending
        if pending is None or pending is room.version:
            room.pending = None
            return False
        if now - room.offered_at < self.prefetch_seconds:
            return False
        settling = pending.settling
        if settling is None:   # offered by hand (tests): settle it as part of the swap
            return True
        if not settling.done():
            return False
        if settling.cancelled() or settling.exception() is not None:
            room.pending = None
            return False
        return True

    async def _swap(self, room: Room) -> bool:
        """Swap a room to its pending version between two ticks (the caller is its physics
        task, so no tick runs meanwhile). The room's "v" changes the moment the old version's
        commands are taken, so a grab still aimed at old body ids is refused rather than applied
        to whatever has those ids now. On failure the room keeps its old sim and version."""
        version, old_version = room.pending, room.version
        room.pending = None
        commands = room.take()
        room.version = version
        started = time.perf_counter()
        try:
            new, report = await asyncio.to_thread(_swap_sims, room.sim, commands, version,
                                                  self.settle_seconds)
        except Exception as exc:  # noqa: BLE001
            log.exception("roomcall: room %s could not move to version %s; it stays on %s",
                          room.id, version.id, old_version.id if old_version else None)
            room.version = old_version
            if old_version is not None:
                # Another room's swap may have pruned it meanwhile (this one's "v" had already
                # moved on): served again, or this room's browsers could not fetch what it runs.
                self.versions.setdefault(old_version.id, old_version)
            self._load_error = f"{version.id}: swap failed: {exc}"[:500]
            self._prune()
            return False
        room._sim = new
        log.info("roomcall: room %s on version %s in %.0f ms — carried %d, reset %d, gone %d, "
                 "grabs kept %d, dropped %d%s", room.id, version.id,
                 1e3 * (time.perf_counter() - started), len(report["carried"]),
                 len(report["reset"]), len(report["gone"]), len(report["grabs_kept"]),
                 len(report["grabs_dropped"]), ", relaxed" if report.get("relaxed") else "")
        self._prune()
        return True

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
        self._prune()

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
                   "color": peer.color, "v": room.v,
                   "peers": [p.info() for p in room.peers.values()],
                   "key": peer.key, "avatars": self.avatars.welcome_info()}
        room.peers[peer.id] = peer
        try:
            peer.sender = asyncio.create_task(self._send_loop(peer))
            peer.push(dumps(welcome))
            if room.pending is not None:
                # Joined during a prefetch window: the room still runs the old version, and is
                # about to switch — fetch the new one too.
                peer.push(dumps(room.pending.message()))
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
        self.avatars.leave(peer)
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
        self.avatars.seen(peer)
        if kind == "pose":
            p, yaw, pitch = vec3(message.get("p")), finite(message.get("yaw")), \
                finite(message.get("pitch", 0.0))
            if p is not None and yaw is not None and pitch is not None:
                peer.pose = [round(v, 4) for v in (*p, yaw, pitch)]
            # No "j" (the sender stopped tracking) or a bad one: receivers relax the arms.
            peer.joints = clean_joints(message.get("j")) if "j" in message else None
        elif kind == "grab":
            body, point = message.get("body"), vec3(message.get("point"))
            v = message.get("v")
            if v is not None and v != room.v:
                return  # aimed at another version's body ids: they may name another object here
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
        elif kind == "avatar":
            if "claim" in message:
                self.avatars.claim(peer, room.id, message.get("claim"), message.get("key"))
        elif kind == "rtc":
            to = message.get("to")
            target = room.peers.get(to) if isinstance(to, str) else None
            if target is not None and target is not peer and "data" in message:
                target.push(dumps({"t": "rtc", "from": peer.id, "data": message["data"]}))
        # "hello" after joining, and unknown kinds, are ignored: names are fixed at join so every
        # client's peer list stays consistent without a rename message.

    # -- physics --------------------------------------------------------------------------------

    async def _physics(self, room: Room) -> None:
        """Real-time stepping and the 30 Hz state broadcast for one room, until it empties. A
        version swap happens here too, between two ticks (`_swap`)."""
        loop = asyncio.get_running_loop()
        if room.version is None:
            room.version = self.version
        sim = room.sim
        if room._sim is None:
            # From here on the sim lives in `_sim`, which a swap replaces. `ready` holding it as
            # its result would keep the room's FIRST sim — and its version's model, hundreds of
            # MB for a real room — alive for as long as the room is, however many swaps later.
            room._sim, room.ready = sim, loop.create_future()
            room.ready.set_result(None)
        max_steps = max_steps_per_tick(sim.timestep)
        last = loop.time()
        next_state = last
        simulated = elapsed = 0.0  # for the real-time factor
        while room.peers:
            await asyncio.sleep(PHYSICS_INTERVAL)
            now = loop.time()
            if room.pending is not None and self._swap_due(room, now):
                if await self._swap(room):
                    sim = room.sim
                    max_steps = max_steps_per_tick(sim.timestep)
                    simulated = elapsed = 0.0
                # The swap's own time is not owed to the new simulation; and the next tick sends
                # a frame at once, so everyone switches as early as possible.
                last = next_state = loop.time()
                continue
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
            frame = {"t": "state", "v": room.v, "time": snapshot.get("time", 0.0),
                     "bodies": snapshot.get("bodies", []), "grabs": snapshot.get("grabs", []),
                     "avatars": {p.id: p.row() for p in room.peers.values() if p.pose}}
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
               settle_seconds: float = 1.0, stage: str = "scene", version: str | None = None,
               prebuilt: Path | None = None, watch: Any = None,
               prefetch_seconds: float | None = None,
               avatars: AvatarConfig | None = None) -> FastAPI:
    """The room call app for one scene: its first version's geometry built here, once (or read
    from `prebuilt`, see `read_prebuilt`); rooms made on demand.

    `version` is the first version's id (default: from the scene's content, or the glb's for a
    bare `model`). `watch` is a source of newer versions (`LocalWatch`, `hosted.VolumeWatch`),
    polled for the app's lifetime; each is offered to every room and swapped in after
    `prefetch_seconds` (default `PREFETCH_SECONDS`). `web_dir` exists for tests (a stand-in client) and for working on the
    client from a checkout; `settle_seconds` is passed to every version's `RoomSim`.
    `avatars` says how full bodies are made (default: off — everyone a capsule or hologram).
    """
    if model is None:
        if scene_xml is None:
            raise ValueError("create_app needs a scene_xml or a model")
        model = load_model(Path(scene_xml))
    web = WebAssets.load(Path(web_dir) if web_dir is not None else WEB_DIR)
    if version is None and scene_xml is not None:
        version = version_name(label, scene_digest(Path(scene_xml)))
    first = SceneVersion.build(model, label=label, stage=stage, id=version, prebuilt=prebuilt)
    hub = RoomHub(first, settle_seconds=settle_seconds, watch=watch,
                  prefetch_seconds=PREFETCH_SECONDS if prefetch_seconds is None else prefetch_seconds,
                  avatars=avatars)

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
        hub.ensure_watching()
        # no-store: the CSP hash belongs to this server's copy; a restart may change both.
        return Response(web.index, media_type=HTML, headers={"Cache-Control": "no-store"})

    @app.get("/app.js")
    async def app_js() -> Response:
        return file_response("app.js", JS)

    @app.get("/app.css")
    async def app_css() -> Response:
        return file_response("app.css", CSS)

    @app.get("/body.js")
    async def body_js() -> Response:
        return file_response("body.js", JS)

    @app.get("/pose_worker.js")
    async def pose_worker_js() -> Response:
        return file_response("pose_worker.js", JS)

    def model_response(name: str, request: Request) -> Response:
        """A vendored model by its fixed name. Every page that uses one fetches it, so a repeat
        visit revalidates against its hash and is not sent again."""
        body = web.asset(name)
        if body is None:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        etag = f'"{hashlib.sha256(body).hexdigest()[:32]}"'
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(body, media_type=BINARY, headers=headers)

    @app.get("/" + SEGMENTER_MODEL)
    async def segmenter_model(request: Request) -> Response:
        """The hologram's segmentation model."""
        return model_response(SEGMENTER_MODEL, request)

    @app.get("/" + POSE_MODEL)
    async def pose_model(request: Request) -> Response:
        """The pose tracker's model (full-body people: the sender's arms and head)."""
        return model_response(POSE_MODEL, request)

    def avatar_error(status: int, message: str, retry: int | None = None) -> Response:
        body: dict = {"error": message}
        headers = {"Cache-Control": "no-store"}
        if retry is not None:
            body["retry"] = retry
            headers["Retry-After"] = str(retry)
        return JSONResponse(body, status_code=status, headers=headers)

    @app.post("/avatar")
    async def avatar_upload(request: Request) -> Response:
        """A person's photos (a turnaround from the camera, or uploads) -> their body, built in
        the background; everyone in their room hears "building" now and "ready" when it is.
        Who is asking comes in headers (`bodies.ROOM_HEADER`, `PEER_HEADER`, `KEY_HEADER`), so
        a stranger, or someone over a limit, is refused before the body is read at all; the
        body is then read with a cap, into one buffer, so an oversized upload is refused before
        it is in memory."""
        service = hub.avatars
        if not service.enabled:
            return avatar_error(503, "full bodies are off on this server")
        limit = bodies.MAX_UPLOAD_BYTES
        length = request.headers.get("content-length")
        if length is not None:
            try:
                declared = int(length)
            except ValueError:
                return avatar_error(400, "bad Content-Length")
            if declared > limit:
                return avatar_error(413, "the upload is too big")

        async def read() -> bytearray:
            body = bytearray()
            async for chunk in request.stream():
                body += chunk
                if len(body) > limit:
                    raise AvatarRequestError(413, "the upload is too big")
            return body

        headers = request.headers
        try:
            result = await service.upload(
                read, room=headers.get(bodies.ROOM_HEADER), peer=headers.get(bodies.PEER_HEADER),
                key=headers.get(bodies.KEY_HEADER),
                client=request.client.host if request.client else None)
        except AvatarRequestError as exc:
            return avatar_error(exc.status, exc.message, exc.retry_after)
        return JSONResponse(result, status_code=202, headers={"Cache-Control": "no-store"})

    @app.get("/avatars/{name}")
    async def avatar_file(name: str) -> Response:
        """A ready body (LRAV, `avatar_format`). Its name is its id, so it never changes."""
        data = hub.avatars.avatar_bytes(name)
        if data is None:
            return JSONResponse({"detail": "Not Found"}, status_code=404,
                                headers={"Cache-Control": "no-store"})
        return Response(data, media_type=BINARY,
                        headers={"Cache-Control": AVATAR_CACHE, "ETag": f'"{name[:-4]}"'})

    @app.get("/scene.json")
    async def scene_manifest(request: Request) -> Response:
        """The live version's manifest, or with ?v= that version's (404 once nothing needs it)."""
        hub.ensure_watching()
        vid = request.query_params.get("v")
        found = hub.version if vid is None else hub.versions.get(vid)
        if found is None:
            return JSONResponse({"detail": "no such version", "live": hub.version.id},
                                status_code=404, headers={"Cache-Control": "no-store"})
        return Response(found.scene_json, media_type="application/json",
                        headers={"Cache-Control": "no-cache"})

    @app.get("/scene.glb")
    async def scene_glb(request: Request) -> Response:
        """?h=<hash> (what every manifest names): that glb, cacheable forever. Without it the live
        version's, revalidated rather than cached: another scene can come up on the same port."""
        hub.ensure_watching()
        wanted = request.query_params.get("h")
        if wanted is not None:
            found = hub.glb_version(wanted)
            if found is None:
                return JSONResponse({"detail": "no such scene"}, status_code=404,
                                    headers={"Cache-Control": "no-store"})
            return Response(found.glb, media_type="model/gltf-binary",
                            headers={"Cache-Control": IMMUTABLE, "ETag": f'"{found.glb_hash}"'})
        live = hub.version
        etag = f'"{live.glb_hash}"'
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(live.glb, media_type="model/gltf-binary", headers=headers)

    @app.get("/healthz")
    async def healthz() -> Response:
        hub.ensure_watching()
        return JSONResponse({"ok": True, "label": hub.version.label, **hub.stats(),
                             **hub.status(), "avatars": hub.avatars.stats()},
                            headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        hub.ensure_watching()
        await hub.session(ws)

    return app


# ---------------------------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------------------------

def freeze_startup_objects() -> None:
    """Exempt everything alive now — modules, classes, the first version — from future garbage
    collections. A full collection walks every tracked object while holding the GIL: ~40 ms for the
    ~120k objects FastAPI, trimesh and MuJoCo leave behind, often enough (loading a version
    allocates a lot) to stall the event loop that carries everyone's frames. Frozen, a full
    collection only walks what was made since. Called once a process's app is built."""
    gc.collect()
    gc.freeze()


def resolve_scene(path: Path | str) -> Path:
    """A scene.xml, or a folder holding one of `SCENE_CANDIDATES` (the first found)."""
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
          label: str | None = None, demo: bool = False, open_browser: bool = True,
          watch: bool = False, stage: str | None = None, avatars: str = "standin",
          modal_env: str | None = None) -> int:
    """Serve one scene (or the demo room, with `demo` or no scene) until interrupted. With
    `watch`, every change to the scene's files is a new version, swapped into the rooms live.
    `avatars` is how full bodies are made: "standin" here, "modal" on the GPU function in the
    Modal environment `modal_env` (the stand-in when that fails), "off" not at all."""
    import uvicorn

    with tempfile.TemporaryDirectory(prefix="roomcall-demo-") as scratch:
        if demo or scene_xml is None:
            from litereality_agent.room_ops.roomcall.demo import write_demo_scene

            scene = write_demo_scene(Path(scratch))
            label = label or "demo room"
            watched = scene
            stage = stage or "demo"
        else:
            scene = resolve_scene(scene_xml)
            label = label or scene_label(scene)
            watched = Path(scene_xml).expanduser()
        # Both looked at before the first build, so a change made while it compiles is not
        # missed: the watcher sees the files move on from what was hashed, and the id it computes
        # then differs from this version's (an id hashed after the compile would already be the
        # changed files', and the change would pass for the version being served).
        watcher = LocalWatch(watched, label=label, stage=stage) if watch else None
        version = version_name(label, scene_digest(scene))
        print(f"room call  building {label} from {scene} …")
        model = load_model(scene)
        app = create_app(scene, model=model, label=label, stage=stage or scene_stage(scene),
                         version=version, watch=watcher,
                         avatars=AvatarConfig(mode=avatars, environment=modal_env or "main"))
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
        print(f"  bodies {_BODIES[avatars]}")
        if watcher is not None:
            print(f"  watch  {watched}: a change to the scene is swapped in live, nobody leaves "
                  f"(version {app.state.roomcall.version.id})")
        if open_browser:
            threading.Thread(target=_open_when_up, args=(server, url), daemon=True).start()
        freeze_startup_objects()
        # uvicorn shuts down on SIGTERM and then re-raises it to whatever handler was installed
        # before it ran. Left at the default, that kills the process on the spot — before this
        # `with` can delete the demo room's scratch folder, so every `kill` left one in /tmp.
        # Turning it into SystemExit lets the shutdown finish here like ctrl-c does.
        on_main = threading.current_thread() is threading.main_thread()
        previous = signal.signal(signal.SIGTERM, _exit_quietly) if on_main else None
        try:
            server.run(sockets=[sock])
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            sock.close()
            if on_main:
                signal.signal(signal.SIGTERM, previous)
    print("stopped")
    return 0


def _exit_quietly(_signum, _frame) -> None:
    raise SystemExit(0)


_BODIES = {
    "off": "off: everyone is a capsule, or their hologram",
    "standin": "full bodies from each person's photos, as simple stand-ins (--avatars modal for "
               "photoreal ones on your Modal GPU)",
    "modal": "photoreal full bodies with LHM++ on Modal (litereality-avatar); the stand-in "
             "whenever that fails",
}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m litereality_agent.room_ops.roomcall",
                                     description="a shared physics room for a video call")
    add_arguments(parser)
    args = parser.parse_args(argv)
    return run(args)


def run(args, *, modal_env: str | None = None) -> int:
    """Run from parsed `add_arguments` arguments — shared by `litereality roomcall` and
    `python -m litereality_agent.room_ops.roomcall` so the two can never drift apart."""
    try:
        return serve(Path(args.scene) if args.scene else None, host=args.host, port=args.port,
                     label=args.label, demo=args.demo, open_browser=args.open_browser,
                     watch=getattr(args, "watch", False), stage=getattr(args, "stage", None),
                     avatars=getattr(args, "avatars", "standin"),
                     modal_env=modal_env or getattr(args, "modal_env", None))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
