"""Full-body people for the room call: the upload, the stand-in body, the limits and the store.

A laptop webcam frames a head and shoulders, so a hologram alone is a floating bust. At join each
person may give a few frames of themselves turning around (or a photo or two); a body is built
from them — on a Modal GPU with LHM++ (`deploy/modal/avatar/app.py`), or here as a stand-in — and
served to everyone in the room as one LRAV file (`avatar_format`). Browsers draw it with Spark and
drive it live: the sender's own arms from MediaPipe, legs procedurally, the live face on top.

What lives here, and why it is shaped the way it is:

- `parse_upload` checks a POST body against every limit without decoding a single pixel: the
  JSON's shape, strict base64, image magic, Pillow's header (so a 30000x30000 "PNG" costs
  nothing), the pixel count and the sides. Uploads are untrusted bytes from anyone holding the
  link; the pixels are decoded only by a build, which the limits below ration.
- The service checks who is asking (room, peer and upload key, sent as headers) and every limit
  that needs no body BEFORE the body is read, and reads and checks at most `UPLOAD_SLOTS` bodies
  at once: a stranger's upload costs the server nothing, and nobody's can slow the rooms' 30 Hz
  state streams, which share this process.
- `make_standin` is a procedural Gaussian humanoid on a fixed SMPL-X-like skeleton, tinted from
  the photos (`person_tints`). It is what `--avatars standin` serves, what tests use, and what a
  person gets whenever the GPU path fails: nobody is ever left without a body because Modal is
  down, not deployed or over budget.
- `AvatarService` owns the store and the limits. Avatars are photo-derived personal data, so they
  live in this process's memory only — never on disk or a Volume — and are dropped `KEEP_SECONDS`
  after their owner leaves (time enough for a page reload to claim it back). GPU minutes cost
  money, so builds are limited per connection, per room (counted per room id, surviving the room
  being dropped and recreated) and per day (`MemoryBudget`, or `ModalDictBudget` when hosted).
  Connections and room ids are free to make, so the same limits also hold per client address
  (`client_key`): its uploads per hour, the bodies it holds, and its share of the day's GPU
  builds (after which it gets stand-ins). At most `GPU_MAX_CALLS` GPU calls run at once, and at
  most `MAX_BUILDING` builds are in flight on the whole server.

Nothing here imports Modal at module level, and nothing touches it until a build asks for the GPU.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import functools
import hashlib
import hmac
import importlib.util
import io
import ipaddress
import json
import logging
import math
import re
import secrets
import sys
import time
import weakref
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from litereality_agent.room_ops.roomcall.avatar_format import (
    BODY_JOINTS,
    SMPLX_JOINTS,
    SMPLX_PARENTS,
    Avatar,
    AvatarFormatError,
    pack,
    quantize_skin,
    unpack,
)

log = logging.getLogger(__name__)

AVATAR_APP, AVATAR_FUNCTION = "litereality-avatar", "avatar"
BUDGET_DICT = "litereality-avatar-budget"
MODES = ("off", "standin", "modal")
UPLOAD_MODES = ("turnaround", "photo")
AVATAR_ID = re.compile(r"a[0-9a-f]{16}")
AVATAR_FILE = re.compile(r"(a[0-9a-f]{16})\.bin")
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 2 * 1024 * 1024      # decoded, each
MAX_UPLOAD_BYTES = 22 * 1024 * 1024    # the whole POST body
MIN_SIDE, MAX_SIDE, MAX_PIXELS = 128, 4096, 12_000_000
COOLDOWN_SECONDS = 120                 # per peer (connection)
ROOM_MAX_BUILDING = 2                  # builds in flight per room
ROOM_BUILDS_PER_HOUR = 24
GPU_DAILY_CAP = 40                     # GPU builds per UTC day (persisted when hosted)
KEEP_SECONDS = 600                     # an avatar outlives its owner's connection this long (re-claim)
MAX_STORED = 40                        # avatars held in memory (<= 5 MiB each)
CALL_TIMEOUT = 420                     # cold start (1-2 min) plus the build
STANDIN_SPLATS = 24_000
STANDIN_NOTE = "a simple stand-in body"
# Who is uploading travels in headers, so it is checked before the body is read.
ROOM_HEADER, PEER_HEADER, KEY_HEADER = "x-roomcall-room", "x-roomcall-peer", "x-roomcall-key"
UPLOAD_SLOTS = 2                       # bodies read and checked at once, server-wide
BUSY_RETRY = 3                         # seconds a busy server asks an upload to wait
MAX_BUILDING = 8                       # builds in flight on the whole server
GPU_MAX_CALLS = 2                      # GPU calls at once: the GPU app's MAX_CONTAINERS
GPU_QUEUE_SECONDS = 180                # a build waits this long for a GPU, then is a stand-in
BUDGET_TIMEOUT = 10                    # one look at the GPU budget (a Modal Dict call), at most
BUILD_TIMEOUT = GPU_QUEUE_SECONDS + CALL_TIMEOUT + 60   # a whole build, at most
IP_ATTEMPTS_PER_MINUTE = 20            # bodies read per client address, accepted or not
IP_UPLOADS_PER_HOUR = 16               # uploads accepted per client address
IP_MAX_STORED = 10                     # bodies held per client address (not counting released)
IP_GPU_PER_DAY = 10                    # GPU builds per client address per UTC day, then stand-ins
IDLE_EVICT_SECONDS = 1800              # a full store may drop the body of a connection this idle

# The client's ids and keys are short tokens; anything longer is not one of ours.
_TOKEN = 64
_ROOM = re.compile(r"[A-Za-z0-9_-]{1,64}")
_JPEG, _PNG = b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"
# A real upload is one JSON object of a few strings and one list of them. More structure than
# this is refused before `json.loads` builds it: 22 MB of "[[],[]," would be millions of objects.
_MAX_CONTAINERS, _MAX_COMMAS = 4, 64


def now() -> float:
    """The service's clock (monotonic seconds); tests replace it to skip ahead."""
    return time.monotonic()


def utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class AvatarRequestError(Exception):
    """An upload refused: `status` is the HTTP status, `message` what the person is shown,
    `retry_after` (seconds) when waiting would help."""

    def __init__(self, status: int, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


# ---------------------------------------------------------------------------------------------
# The upload
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Upload:
    room: str
    peer: str
    key: str
    mode: str
    images: tuple[bytes, ...]


def _bad(message: str) -> AvatarRequestError:
    return AvatarRequestError(400, message)


def _check_image(data: bytes, index: int) -> None:
    """A JPEG or PNG with sane sides, from its header alone: no pixel is decoded here (that is
    the build's work, which the limits ration), so a file that claims 30000x30000 costs nothing.
    A PNG's chunk checksums are verified too, which reads no pixels either.

    No `warnings` filter for Pillow's DecompressionBombWarning: filters are process-wide and not
    thread-safe, and this runs in threads. The explicit MAX_PIXELS check refuses those images;
    DecompressionBombError (past twice Pillow's limit) needs no filter."""
    from PIL import Image

    what = f"image {index + 1}"
    jpeg, png = data.startswith(_JPEG), data.startswith(_PNG)
    if not (jpeg or png):
        raise _bad(f"{what} is not a JPEG or PNG")
    try:
        with Image.open(io.BytesIO(data), formats=("JPEG",) if jpeg else ("PNG",)) as image:
            width, height = image.size
            if width * height > MAX_PIXELS:
                raise _bad(f"{what} has too many pixels ({width}x{height})")
            if min(width, height) < MIN_SIDE or max(width, height) > MAX_SIDE:
                raise _bad(f"{what} is {width}x{height}: each side must be "
                           f"{MIN_SIDE}..{MAX_SIDE} pixels")
            if png:
                image.verify()
    except AvatarRequestError:
        raise
    except Image.DecompressionBombError:
        raise _bad(f"{what} has too many pixels") from None
    except Exception:  # noqa: BLE001 — truncated, corrupt, unsupported: all the same to the person
        raise _bad(f"{what} could not be read as an image") from None


def parse_upload(body: bytes | bytearray, *, room: str | None = None, peer: str | None = None,
                 key: str | None = None) -> Upload:
    """A POST /avatar body -> Upload, checking every limit. Raises AvatarRequestError (400, or
    413 for a body over MAX_UPLOAD_BYTES). Blocking (Pillow): run it in a thread.

    `room`, `peer` and `key` are who the request says it is (the server passes its headers,
    checked before the body was read); the body may repeat them, and then they must match.
    Without them, the body must carry all three."""
    if len(body) > MAX_UPLOAD_BYTES:
        raise AvatarRequestError(413, "the upload is too big")
    if (body.count(b"[") + body.count(b"{") > _MAX_CONTAINERS
            or body.count(b",") > _MAX_COMMAS):
        raise _bad("the upload is not a body upload")
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        raise _bad("the upload is not JSON") from None
    if not isinstance(data, dict):
        raise _bad("the upload must be a JSON object")
    given = (room, peer, key)
    named = (data.get("room"), data.get("peer"), data.get("key"))
    if any(v is not None for v in given):
        if any(n is not None and n != g for n, g in zip(named, given, strict=True)):
            raise _bad("the upload names another room, person or key than its request")
        room, peer, key = given
    else:
        room, peer, key = named
    if not all(isinstance(v, str) and 0 < len(v) <= _TOKEN for v in (room, peer, key)):
        raise _bad("room, peer and key are required")
    mode = data.get("mode")
    if mode not in UPLOAD_MODES:
        raise _bad(f"mode must be one of {', '.join(UPLOAD_MODES)}")
    encoded = data.get("images_b64")
    if not isinstance(encoded, list) or not 1 <= len(encoded) <= MAX_IMAGES:
        raise _bad(f"send 1 to {MAX_IMAGES} images")
    limit = 4 * math.ceil(MAX_IMAGE_BYTES / 3)
    images = []
    for index, text in enumerate(encoded):
        if not isinstance(text, str):
            raise _bad(f"image {index + 1} is not base64 text")
        if len(text) > limit:
            raise _bad(f"image {index + 1} is over {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")
        try:
            raw = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError):
            raise _bad(f"image {index + 1} is not valid base64") from None
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise _bad(f"image {index + 1} is empty or over {MAX_IMAGE_BYTES // (1024 * 1024)} MiB")
        _check_image(raw, index)
        images.append(raw)
    return Upload(room=room, peer=peer, key=key, mode=mode, images=tuple(images))


# ---------------------------------------------------------------------------------------------
# The stand-in body
# ---------------------------------------------------------------------------------------------

# SMPL-X-like rest joints (metres, +Y up, facing +Z, +X the person's left), T-pose, about 1.75 m.
STANDIN_REST = np.array([
    (0, .95, 0), (.09, .88, 0), (-.09, .88, 0), (0, 1.06, -.01), (.10, .50, .01), (-.10, .50, .01),
    (0, 1.19, -.01), (.11, .08, -.02), (-.11, .08, -.02), (0, 1.25, 0), (.12, .02, .10),
    (-.12, .02, .10), (0, 1.47, -.01), (.07, 1.39, 0), (-.07, 1.39, 0), (0, 1.56, .01),
    (.18, 1.42, -.01), (-.18, 1.42, -.01), (.44, 1.42, -.02), (-.44, 1.42, -.02),
    (.69, 1.42, -.01), (-.69, 1.42, -.01),
], dtype=np.float32)

DEFAULT_TINTS = {"skin": (205, 160, 130), "hair": (60, 45, 35), "top": (90, 110, 140),
                 "bottom": (70, 75, 90)}
FEET_RGB = (40, 40, 45)
HEAD_CENTRE = STANDIN_REST[15] + np.array([0, 0.08, 0.01], np.float32)
HEAD_RADII = np.array([0.085, 0.115, 0.10], np.float32)
# Torso skin bands: (joint, y) — a torso point is skinned linearly between the two around it.
TORSO_BANDS = ((0, 0.92), (3, 1.03), (6, 1.15), (9, 1.30), (12, 1.45))
TORSO_Y = (0.86, 1.45)
TORSO_HALF_X, TORSO_HALF_Z = 0.16, 0.10
BELT_Y = 0.97          # torso points below this are the trousers (the pelvis)
BLEND = 0.06           # metres from a segment's start over which it blends toward its parent


def _small_rgb(data: bytes) -> np.ndarray:
    """An upload as RGB, at most 128 px tall (EXIF orientation applied)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as image:
        image.draft("RGB", (256, 256))
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.height > 128:
            image = image.resize((max(1, round(image.width * 128 / image.height)), 128),
                                 Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def _filled_rows(loose: np.ndarray, strict: np.ndarray, width: int, need: int) -> int | None:
    """Where the person fills the frame from edge to edge down to its bottom (a shirt in a
    waist-up photo, or someone too close to the camera), or None.

    Those rows' ends are the person, so the row-end test leaves nothing of them. They are a run
    of rows, down to the frame's bottom (a gap of two rows allowed), where at least 60 % of the
    central width differs from the background; at least a fifth of the height; under a person
    (strict rows above it) at most 70 % of the height tall; and with person rows of its own in at
    most a quarter of its rows. That last one tells a shirt from a floor or a beach in front of
    someone standing on it: their legs and feet stand out in those rows. A floor seen only in
    front of the feet can still pass for clothing; a stand-in tint is all that is at stake."""
    h = len(loose)
    full = loose.sum(axis=1) >= 0.6 * width
    start, gap = h, 0
    for r in range(h - 1, -1, -1):
        if full[r]:
            start, gap = r, 0
        elif start < h and gap < 2:
            gap += 1
        elif start == h and r >= h - 3:
            continue                   # a row or two of noise at the very bottom
        else:
            break
    rows = np.flatnonzero(strict.sum(axis=1) >= need)
    if start >= h or h - start < 0.2 * h or rows.size == 0 or rows[0] >= start:
        return None
    if start - rows[0] > 0.7 * h or (strict[start:].sum(axis=1) >= need).mean() > 0.25:
        return None
    return start


def person_tints(images: Sequence[bytes]) -> dict[str, tuple[int, int, int]]:
    """Skin, hair, top and bottom colours of the person in the photos.

    Per image: the background is the median of the top border and the upper two thirds of the
    side borders (6 % wide; the bottom is left out because a person is cut off by the bottom of
    a frame, never the top). Person pixels differ from it by more than 40 (RGB distance) — and
    from their own row's ends (the 6 % at its left and right), so a sky or a lit wall that
    shades from top to bottom is not taken for the person — inside the central 70 % of the width.
    Rows the person fills from edge to edge (`_filled_rows`) are the person whatever their ends.
    Too few person pixels: a central box instead.
    The person is the rows from the first to the last run of person rows at least 5 % of the
    height long (so a cloud or a lamp above the head is not where they start). Bands are
    fractions of that extent — head 0-13 % (hair its top quarter), top 18-50 %, bottom 55-95 % —
    pooled over every image; a band with under 30 pixels keeps its default. A person cut off by
    the frame's bottom is assumed to fill it from the shoulders (18 %) down, so a waist-up photo
    gives a shirt and no trousers. Crude on purpose: it only has to make the stand-in look like
    the right person from across a room."""
    pools: dict[str, list[np.ndarray]] = {name: [] for name in DEFAULT_TINTS}
    bands = {"skin": (0.0, 0.13), "hair": (0.0, 0.13 / 4), "top": (0.18, 0.50),
             "bottom": (0.55, 0.95)}
    for data in images:
        try:
            rgb = _small_rgb(data)
        except Exception:  # noqa: BLE001 — an unreadable image just adds nothing
            continue
        h, w = rgb.shape[:2]
        if h < 4 or w < 4:
            continue
        bh, bw = max(1, round(0.06 * h)), max(1, round(0.06 * w))
        upper = max(bh, round(2 * h / 3))
        border = np.concatenate([rgb[:bh].reshape(-1, 3), rgb[:upper, :bw].reshape(-1, 3),
                                 rgb[:upper, -bw:].reshape(-1, 3)])
        background = np.median(border, axis=0)
        row_background = np.median(np.concatenate([rgb[:, :bw], rgb[:, -bw:]], axis=1), axis=1)
        pixels = rgb.astype(np.float32)
        loose = np.linalg.norm(pixels - background, axis=2) > 40
        c0, c1 = round(0.15 * w), round(0.85 * w)
        loose[:, :c0] = False
        loose[:, c1:] = False
        mask = loose & (np.linalg.norm(pixels - row_background[:, None, :], axis=2) > 40)
        need = max(2, round(0.01 * w))
        filled = _filled_rows(loose, mask, c1 - c0, need)
        if filled is not None:
            mask[filled:] |= loose[filled:]
        if mask.mean() < 0.04:
            mask = np.zeros((h, w), bool)
            mask[round(0.10 * h):round(0.95 * h), round(0.30 * w):round(0.70 * w)] = True
            filled = None
        rows = np.flatnonzero(mask.sum(axis=1) >= need)
        if rows.size == 0:
            continue
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > max(1, round(0.03 * h))) + 1)
        big = [run for run in runs if len(run) >= max(3, 0.05 * h)] or [max(runs, key=len)]
        top = int(big[0][0])
        extent = int(big[-1][-1] + 1 - top)
        if filled is not None:            # cut off by the frame: it starts at the shoulders
            extent = max(extent, round((filled - top) / 0.18))
        for name, (f0, f1) in bands.items():
            r0 = min(h, top + int(f0 * extent))
            r1 = min(h, max(r0 + 1, top + int(round(f1 * extent))))
            pools[name].append(rgb[r0:r1][mask[r0:r1]])
    tints = {}
    for name, default in DEFAULT_TINTS.items():
        pixels = np.concatenate(pools[name]) if pools[name] else np.zeros((0, 3), np.uint8)
        tints[name] = (tuple(int(v) for v in np.median(pixels, axis=0).round())
                       if len(pixels) >= 30 else default)
    return tints


@dataclass
class _Part:
    """Splats of one body part: positions, the side each faces (local z < 0 is the back),
    two skin joints with the second's weight, and a colour key."""

    points: np.ndarray
    back: np.ndarray
    own: np.ndarray
    parent: np.ndarray
    w_parent: np.ndarray
    color: np.ndarray       # (n,) indices into the tint list


def _unit(n: int, rng: np.random.Generator) -> np.ndarray:
    d = rng.normal(size=(n, 3))
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


def _ellipsoid_area(radii: np.ndarray) -> float:
    a, b, c = (float(v) for v in radii)
    p = 1.6075  # Knud Thomsen's approximation, within ~1 %
    return 4 * math.pi * ((a ** p * b ** p + a ** p * c ** p + b ** p * c ** p) / 3) ** (1 / p)


def _ellipsoid(n: int, rng, centre, radii) -> tuple[np.ndarray, np.ndarray]:
    local = _unit(n, rng) * radii
    return centre + local, local


def _perpendicular(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0, 0])
    v = np.cross(u, helper)
    v /= np.linalg.norm(v)
    return v, np.cross(u, v)


def _capsule_area(a, b, r0, r1) -> float:
    length = float(np.linalg.norm(b - a))
    return math.pi * (r0 + r1) * length + 2 * math.pi * (r0 * r0 + r1 * r1)


def _capsule(n: int, rng, a, b, r0, r1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A tapered capsule's surface: points, their offset from the axis, and how far along the
    axis from `a` each is (negative on the start cap)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    axis = b - a
    length = float(np.linalg.norm(axis))
    u = axis / length
    v, w = _perpendicular(u)
    side = math.pi * (r0 + r1) * length
    caps = (2 * math.pi * r0 * r0, 2 * math.pi * r1 * r1)
    total = side + caps[0] + caps[1]
    pick = rng.random(n) * total
    t = rng.random(n)
    theta = rng.random(n) * 2 * math.pi
    radius = r0 + (r1 - r0) * t
    offset = radius[:, None] * (np.cos(theta)[:, None] * v + np.sin(theta)[:, None] * w)
    base = a + t[:, None] * axis
    d = _unit(n, rng)
    on_start = pick >= side
    on_end = pick >= side + caps[0]
    on_start &= ~on_end
    # Caps: hemispheres facing away from the segment.
    d_start = np.where((d @ u)[:, None] > 0, -d, d)
    d_end = np.where((d @ u)[:, None] < 0, -d, d)
    offset = np.where(on_start[:, None], r0 * d_start, offset)
    offset = np.where(on_end[:, None], r1 * d_end, offset)
    base = np.where(on_start[:, None], a, base)
    base = np.where(on_end[:, None], b, base)
    points = base + offset
    along = (points - a) @ u
    radial = offset - (offset @ u)[:, None] * u
    return points, radial, along


def _blend(along: np.ndarray) -> np.ndarray:
    """The parent's weight near a segment's start: 0.5 at the start, 0 from BLEND on."""
    return 0.5 * np.clip(1.0 - along / BLEND, 0.0, 1.0)


def _counts(areas: list[float], total: int) -> list[int]:
    """`total` split by area, summing exactly (largest remainders get the leftovers)."""
    weights = np.asarray(areas) / sum(areas) * total
    counts = np.floor(weights).astype(int)
    for i in np.argsort(-(weights - counts))[: total - int(counts.sum())]:
        counts[i] += 1
    return [int(c) for c in counts]


# Colour keys, in `_Part.color`.
SKIN, HAIR, TOP, BOTTOM, FEET = range(5)


def _standin_parts(rng: np.random.Generator, splats: int) -> list[_Part]:
    rest = STANDIN_REST.astype(np.float64)
    specs: list[tuple[float, Callable[[int], _Part]]] = []

    y0, y1 = TORSO_Y
    hz = TORSO_HALF_Z
    side_area = 2 * math.pi * math.sqrt((0.15 ** 2 + hz ** 2) / 2) * (y1 - y0)
    cap_area = math.pi * TORSO_HALF_X * hz

    def hx_at(y: np.ndarray) -> np.ndarray:
        """The torso's half-width: TORSO_HALF_X, narrowing to 0.13 at the waist."""
        return TORSO_HALF_X - 0.03 * np.exp(-((y - 1.03) / 0.09) ** 2)

    def torso(n: int) -> _Part:
        m_side, m_top, m_bottom = _counts([side_area, cap_area, cap_area], n)
        y = rng.uniform(y0, y1, m_side)
        theta = rng.random(m_side) * 2 * math.pi
        side = np.stack([hx_at(y) * np.cos(theta), y, hz * np.sin(theta)], axis=1)
        discs = []
        for m, yc in ((m_top, y1), (m_bottom, y0)):
            r, th = np.sqrt(rng.random(m)), rng.random(m) * 2 * math.pi
            discs.append(np.stack([hx_at(np.full(m, yc)) * r * np.cos(th), np.full(m, yc),
                                   hz * r * np.sin(th)], axis=1))
        points = np.concatenate([side, *discs])
        ys = points[:, 1]
        joints = [j for j, _ in TORSO_BANDS]
        levels = np.array([y for _, y in TORSO_BANDS])
        k = np.clip(np.searchsorted(levels, ys) - 1, 0, len(levels) - 2)
        frac = np.clip((ys - levels[k]) / (levels[k + 1] - levels[k]), 0.0, 1.0)
        own = np.array(joints)[k]
        parent = np.array(joints)[k + 1]
        color = np.where(ys < BELT_Y, BOTTOM, TOP)
        return _Part(points, points[:, 2] < 0, own, parent, frac, color)

    specs.append((side_area + 2 * cap_area, torso))

    def ellipsoid(centre, radii, joint: int, color: int) -> Callable[[int], _Part]:
        def make(n: int) -> _Part:
            points, local = _ellipsoid(n, rng, centre, radii)
            colors = np.full(n, color)
            if joint == 15:   # the head: a hair cap over the top and the back
                colors = np.where((points[:, 1] > centre[1] + 0.04) & (points[:, 2] < 0.07),
                                  HAIR, colors)
            zeros = np.zeros(n, int)
            return _Part(points, local[:, 2] < 0, np.full(n, joint), zeros, np.zeros(n), colors)
        return make

    specs.append((_ellipsoid_area(HEAD_RADII), ellipsoid(HEAD_CENTRE, HEAD_RADII, 15, SKIN)))

    def capsule(a, b, r0, r1, own: int, parent: int, color: int) -> Callable[[int], _Part]:
        def make(n: int) -> _Part:
            points, radial, along = _capsule(n, rng, a, b, r0, r1)
            return _Part(points, radial[:, 2] < 0, np.full(n, own), np.full(n, parent),
                         _blend(along) if parent != own else np.zeros(n), np.full(n, color))
        return make

    neck_a, neck_b = np.array([0, 1.40, 0.0]), np.array([0, 1.57, 0.005])
    specs.append((_capsule_area(neck_a, neck_b, 0.05, 0.05),
                  capsule(neck_a, neck_b, 0.05, 0.05, 12, 12, SKIN)))
    for side in (0, 1):   # 0 left (+x), 1 right
        sh, el, wr = rest[16 + side], rest[18 + side], rest[20 + side]
        hip, knee, ankle = rest[1 + side], rest[4 + side], rest[7 + side]
        for a, b, r0, r1, own, parent, color in (
                (sh, el, 0.05, 0.042, 16 + side, 13 + side, TOP),
                (el, wr, 0.04, 0.032, 18 + side, 16 + side, TOP),
                (hip, knee, 0.075, 0.055, 1 + side, 0, BOTTOM),
                (knee, ankle, 0.05, 0.04, 4 + side, 1 + side, BOTTOM)):
            specs.append((_capsule_area(a, b, r0, r1), capsule(a, b, r0, r1, own, parent, color)))

        u = (wr - el) / np.linalg.norm(wr - el)
        hand_radii = np.array([0.048, 0.018, 0.042])

        def hand(n: int, wr=wr, u=u, side=side) -> _Part:
            centre = wr + 0.05 * u
            points, local = _ellipsoid(n, rng, centre, hand_radii)
            along = (points - wr) @ u
            return _Part(points, local[:, 2] < 0, np.full(n, 20 + side), np.full(n, 18 + side),
                         _blend(along), np.full(n, SKIN))

        specs.append((_ellipsoid_area(hand_radii), hand))
        foot_radii = np.array([0.045, 0.045, 0.10])

        def foot(n: int, ankle=ankle, side=side) -> _Part:
            centre = np.array([ankle[0], 0.045, ankle[2] + 0.06])
            points, local = _ellipsoid(n, rng, centre, foot_radii)
            front = points[:, 2] > centre[2]
            own = np.where(front, 10 + side, 7 + side)
            near = np.linalg.norm(points - ankle, axis=1)
            return _Part(points, local[:, 2] < 0, own, np.full(n, 4 + side), _blend(near),
                         np.full(n, FEET))

        specs.append((_ellipsoid_area(foot_radii), foot))

    counts = _counts([area for area, _ in specs], splats)
    return [make(n) for (_, make), n in zip(specs, counts, strict=True) if n > 0]


def make_standin(images: Sequence[bytes], *, splats: int = STANDIN_SPLATS, seed: int = 0
                 ) -> Avatar:
    """A procedural Gaussian person on the SMPL-X body skeleton, tinted from `images`.

    Deterministic for the same images, `splats` and `seed`. Splats sit on the surfaces of simple
    solids (an elliptic torso, an ellipsoid head, tapered capsules for limbs, ellipsoid hands and
    feet), the budget split by area; each part is skinned to its own joint, blending toward its
    parent over the first 6 cm, and the torso between spine bands. A stand-in, not a likeness:
    it gives the room a whole person with the right colours while the real one is unavailable."""
    if splats < 100:
        raise ValueError("a stand-in needs at least 100 splats")
    rng = np.random.default_rng(seed)
    tints = person_tints(images)
    parts = _standin_parts(rng, splats)
    points = np.concatenate([p.points for p in parts]).astype(np.float32)
    n = len(points)
    palette = np.array([tints["skin"], tints["hair"], tints["top"], tints["bottom"], FEET_RGB],
                       np.float64)
    rgb = palette[np.concatenate([p.color for p in parts])]
    rgb *= (1 + rng.normal(0, 0.04, n))[:, None]
    rgb[np.concatenate([p.back for p in parts])] *= 0.92
    colors = np.empty((n, 4), np.uint8)
    colors[:, :3] = np.clip(np.round(rgb), 0, 255)
    colors[:, 3] = round(0.92 * 255)
    own = np.concatenate([p.own for p in parts])
    parent = np.concatenate([p.parent for p in parts])
    w_parent = np.concatenate([p.w_parent for p in parts])
    skin = quantize_skin(np.stack([own, parent], axis=1),
                         np.stack([1.0 - w_parent, w_parent], axis=1))
    scales = np.repeat(rng.uniform(0.012, 0.016, n)[:, None], 3, axis=1).astype(np.float32)
    rotations = np.zeros((n, 4), np.float32)
    rotations[:, 3] = 1.0
    return Avatar(positions=points, scales=scales, rotations=rotations, colors=colors, skin=skin,
                  joint_names=list(SMPLX_JOINTS[:BODY_JOINTS]),
                  parents=list(SMPLX_PARENTS[:BODY_JOINTS]), rest=STANDIN_REST.copy(),
                  floor_y=0.0, kind="standin",
                  meta={"tints": {k: list(v) for k, v in tints.items()}, "note": STANDIN_NOTE})


# ---------------------------------------------------------------------------------------------
# Budgets and builders
# ---------------------------------------------------------------------------------------------

class MemoryBudget:
    """GPU builds per UTC day, counted in this process (a local server)."""

    def __init__(self, cap: int, today: Callable[[], str] = utc_date) -> None:
        self.cap = cap
        self.today = today
        self.day = ""
        self.used = 0

    async def take(self) -> bool:
        day = self.today()
        if day != self.day:
            self.day, self.used = day, 0
        if self.used >= self.cap:
            return False
        self.used += 1
        return True


class ModalDictBudget:
    """GPU builds per UTC day in the Modal Dict `name`, so a container restart does not reset the
    count. Fails CLOSED: if the Dict cannot be read or written, no GPU build. The hosted server is
    one container, so a lock here makes read-then-write safe. Each Dict call gets `timeout`
    seconds: the lock is held across them, so one hung call must not hold up every later build
    (and keep their rooms' build slots and the store full) — a slow budget is a spent one."""

    def __init__(self, cap: int, name: str = BUDGET_DICT, timeout: float = BUDGET_TIMEOUT
                 ) -> None:
        self.cap = cap
        self.name = name
        self.timeout = timeout
        self.used = 0
        self._lock: asyncio.Lock | None = None

    async def take(self) -> bool:
        self._lock = self._lock or asyncio.Lock()
        async with self._lock:
            try:
                import modal

                store = modal.Dict.from_name(self.name, create_if_missing=True)
                day = utc_date()
                used = int(await asyncio.wait_for(store.get.aio(day, 0), self.timeout) or 0)
                if used >= self.cap:
                    self.used = used
                    return False
                await asyncio.wait_for(store.put.aio(day, used + 1), self.timeout)
                self.used = used + 1
                return True
            except Exception:  # noqa: BLE001 — fail closed: an unknown budget is a spent one
                log.warning("roomcall: the GPU budget could not be checked; using the stand-in",
                            exc_info=True)
                return False


Builder = Callable[[Sequence[bytes], str, str], Any]   # async -> (lrav bytes, kind, note)


async def build_standin(images: Sequence[bytes], mode: str, aid: str, *,
                        splats: int = STANDIN_SPLATS, note: str = STANDIN_NOTE
                        ) -> tuple[bytes, str, str]:
    """The stand-in, packed (in a thread: ~0.1 s of numpy and Pillow)."""
    def work() -> bytes:
        return pack(make_standin(images, splats=splats))

    return await asyncio.to_thread(work), "standin", note


class _Fallback(Exception):
    """The GPU path failed in a way the person should hear about; the message says how."""


class ModalBuilder:
    """Builds on the GPU function (`deploy/modal/avatar/app.py`), the stand-in on any failure:
    the budget used up, no Modal client, the app not deployed, a timeout, an error reply, a reply
    for another request, or bytes that are not a valid avatar.

    At most `gpu_calls` calls run at once (the GPU app's own container cap): each holds its
    request (the images as base64, up to ~22 MB) in this process for up to `timeout` seconds. A
    build waits `queue_seconds` for its turn, then takes the stand-in; the budget is spent only
    once a build has its turn."""

    def __init__(self, *, environment: str | None, budget: Any, timeout: float = CALL_TIMEOUT,
                 splats: int = STANDIN_SPLATS, gpu_calls: int = GPU_MAX_CALLS,
                 queue_seconds: float = GPU_QUEUE_SECONDS) -> None:
        self.environment = environment
        self.budget = budget
        self.timeout = timeout
        self.splats = splats
        self.queue_seconds = queue_seconds
        self._turns = asyncio.Semaphore(gpu_calls)

    async def __call__(self, images: Sequence[bytes], mode: str, aid: str
                       ) -> tuple[bytes, str, str]:
        try:
            await asyncio.wait_for(self._turns.acquire(), self.queue_seconds)
        except asyncio.TimeoutError:
            return await self._standin(images, mode, aid, "the GPU builders were all busy")
        try:
            if not await self.budget.take():
                why = "daily GPU budget used up"
            else:
                try:
                    return await self._remote(images, mode, aid), "lhm", ""
                except _Fallback as exc:
                    log.warning("roomcall: avatar %s: %s; using the stand-in", aid, exc)
                    why = str(exc)
        finally:
            self._turns.release()
        return await self._standin(images, mode, aid, why)

    async def _standin(self, images, mode, aid, why: str) -> tuple[bytes, str, str]:
        return await build_standin(images, mode, aid, splats=self.splats,
                                   note=f"{STANDIN_NOTE} ({why})")

    async def _remote(self, images: Sequence[bytes], mode: str, aid: str) -> bytes:
        try:
            import modal
        except ImportError:
            raise _Fallback("the Modal client isn't installed on this server") from None
        request = {"images_b64": [base64.b64encode(image).decode("ascii") for image in images],
                   "mode": mode, "id": aid}
        try:
            fn = modal.Function.from_name(AVATAR_APP, AVATAR_FUNCTION,
                                          environment_name=self.environment)
            reply = await asyncio.wait_for(fn.remote.aio(request), self.timeout)
        except asyncio.TimeoutError:
            raise _Fallback("the GPU builder took too long") from None
        except Exception as exc:  # noqa: BLE001
            if _not_found(modal, exc):
                raise _Fallback("the GPU avatar app isn't deployed") from None
            raise _Fallback(f"the GPU builder failed: {type(exc).__name__}") from None
        if not isinstance(reply, dict):
            raise _Fallback("the GPU builder failed: not a reply")
        if "error" in reply:
            raise _Fallback(f"the GPU builder failed: {str(reply['error'])[:200]}")
        if reply.get("id") != aid:
            raise _Fallback("the GPU builder failed: a reply for another request")
        encoded = reply.get("avatar_b64")
        try:
            if not isinstance(encoded, str):
                raise ValueError
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise _Fallback("the GPU builder failed: its reply is not base64") from None
        try:
            await asyncio.to_thread(unpack, data)
        except AvatarFormatError as exc:
            raise _Fallback(f"the GPU builder failed: {exc}") from None
        return data


def _modal_installed() -> bool:
    """Whether `import modal` would work, without importing it (or touching it if it is)."""
    if "modal" in sys.modules:
        return sys.modules["modal"] is not None
    try:
        return importlib.util.find_spec("modal") is not None
    except (ImportError, ValueError):
        return False


def _not_found(modal: Any, exc: Exception) -> bool:
    cls = getattr(getattr(modal, "exception", None), "NotFoundError", None)
    if isinstance(cls, type) and isinstance(exc, cls):
        return True
    return type(exc).__name__ == "NotFoundError"


# ---------------------------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------------------------

@dataclass
class AvatarConfig:
    """How a server makes bodies. `builder` replaces the standard one (tests): an async
    (images, mode, avatar id) -> (lrav bytes, kind, note). The limits are fields so tests can
    shrink them."""

    mode: str = "off"
    environment: str | None = "main"
    budget: Any = None                 # default MemoryBudget(GPU_DAILY_CAP)
    builder: Any = None
    cooldown: float = COOLDOWN_SECONDS
    room_max_building: int = ROOM_MAX_BUILDING
    room_builds_per_hour: int = ROOM_BUILDS_PER_HOUR
    keep_seconds: float = KEEP_SECONDS
    max_stored: int = MAX_STORED
    call_timeout: float = CALL_TIMEOUT
    standin_splats: int = STANDIN_SPLATS
    upload_slots: int = UPLOAD_SLOTS
    max_building: int = MAX_BUILDING
    build_timeout: float = BUILD_TIMEOUT
    ip_attempts_per_minute: int = IP_ATTEMPTS_PER_MINUTE
    ip_uploads_per_hour: int = IP_UPLOADS_PER_HOUR
    ip_max_stored: int = IP_MAX_STORED
    ip_gpu_per_day: int = IP_GPU_PER_DAY
    idle_evict_seconds: float = IDLE_EVICT_SECONDS

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"avatars mode must be one of {', '.join(MODES)}")
        if self.budget is None:
            self.budget = MemoryBudget(GPU_DAILY_CAP)


@dataclass(eq=False)
class StoredAvatar:
    id: str
    key_hash: str
    room: str
    owner: Any                          # the peer it belongs to; None once released
    created: float
    status: str = "building"            # building | ready
    data: bytes | None = None
    kind: str = ""
    note: str = ""
    released_at: float | None = None
    client: str = ""                    # who uploaded it (`client_key`)
    gpu: bool = True                    # within that client's share of today's GPU builds

    @property
    def url(self) -> str:
        return f"/avatars/{self.id}.bin"

    def ready_info(self) -> dict:
        return {"status": "ready", "url": self.url, "kind": self.kind}


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _same(a: str, b: str) -> bool:
    """Constant-time equality of two texts (`hmac.compare_digest` refuses non-ASCII str)."""
    return hmac.compare_digest(a.encode("utf-8", "surrogatepass"), b.encode("utf-8",
                                                                           "surrogatepass"))


def client_key(host: str | None) -> str:
    """Who is asking, for the per-address limits: the client's IP address (`request.client`,
    which uvicorn sets from a trusted local proxy's X-Forwarded-For and Modal sets to the real
    client), an IPv6 one by its /64 — one home or phone network holds a whole /64, and picking
    another address in it is free."""
    host = (host or "").strip()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host[:_TOKEN]
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.IPv6Network((address, 64), strict=False))
    return str(address)


class _Window:
    """Times of events per key within the last `seconds` (a sliding window)."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.times: dict[str, deque[float]] = {}

    def count(self, key: str, t: float) -> int:
        recent = self.times.get(key)
        while recent and t - recent[0] >= self.seconds:
            recent.popleft()
        return len(recent) if recent else 0

    def wait(self, key: str, t: float) -> int:
        """Seconds until the oldest event leaves the window (call after `count`)."""
        recent = self.times.get(key)
        return max(1, math.ceil(self.seconds - (t - recent[0]))) if recent else 1

    def add(self, key: str, t: float) -> None:
        self.times.setdefault(key, deque()).append(t)

    def sweep(self, t: float) -> None:
        """Forget keys with nothing in the window (room ids and addresses are free to invent)."""
        for key, recent in list(self.times.items()):
            if not recent or t - recent[-1] >= self.seconds:
                del self.times[key]


class AvatarService:
    """The store, the limits and the builds of one server (owned by its `RoomHub`).

    Peers are the server's own objects; this needs from each only `id`, `key` (its connection's
    upload key) and two fields it maintains itself: `avatar_id` (its ready body) and
    `avatar_info` (what its people-list entry says, or None). `find_peer(room, peer_id)` finds a
    live peer; `broadcast(room, message)` sends to everyone in a room, `tell(peer, message)` to
    one connection. The server calls `seen(peer)` for each message a connection sends."""

    def __init__(self, config: AvatarConfig, *, find_peer: Callable[[str, str], Any],
                 broadcast: Callable[[str, dict], None],
                 tell: Callable[[Any, dict], None] | None = None) -> None:
        self.config = config
        self.find_peer = find_peer
        self.broadcast = broadcast
        self.tell = tell or (lambda _peer, _message: None)
        self.store: dict[str, StoredAvatar] = {}
        self._room_builds = _Window(3600)
        self._client_uploads = _Window(3600)
        self._client_attempts = _Window(60)
        self._gpu_day = ""
        self._gpu_used: dict[str, int] = {}          # GPU builds per client today
        self._last_upload: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._active: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # Never waited on (a busy server answers 503 at once), so it binds to no event loop.
        self._slots = asyncio.Semaphore(config.upload_slots)
        self._tasks: set[asyncio.Task] = set()
        if config.builder is not None:
            self.builder = config.builder
        elif config.mode == "modal":
            self.builder = ModalBuilder(environment=config.environment, budget=config.budget,
                                        timeout=config.call_timeout,
                                        splats=config.standin_splats)
            if not _modal_installed():
                log.warning("roomcall: --avatars modal, but the Modal client is not installed: "
                            "everyone gets the stand-in body")
        else:
            self.builder = functools.partial(build_standin, splats=config.standin_splats)

    # -- what clients are told ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    def welcome_info(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {"enabled": True, "mode": self.config.mode, "cooldown": int(self.config.cooldown),
                "max_images": MAX_IMAGES}

    def stats(self) -> dict:
        statuses = [entry.status for entry in self.store.values()]
        return {"mode": self.config.mode, "stored": statuses.count("ready"),
                "building": statuses.count("building"),
                "gpu_today": int(getattr(self.config.budget, "used", 0) or 0)}

    def seen(self, peer: Any) -> None:
        """`peer` sent something: it is not idle (a full store may drop an idle one's body)."""
        self._active[peer] = now()

    # -- the store ------------------------------------------------------------------------------

    def _live(self, entry: StoredAvatar) -> bool:
        owner = entry.owner
        return owner is not None and self.find_peer(entry.room, owner.id) is owner

    def sweep(self) -> None:
        """Forget avatars released longer than `keep_seconds` ago, and the counts of rooms and
        clients that have done nothing for a window (room ids are free to invent)."""
        t = now()
        for aid, entry in list(self.store.items()):
            if entry.released_at is not None and t - entry.released_at >= self.config.keep_seconds:
                self._delete(aid)
        for window in (self._room_builds, self._client_uploads, self._client_attempts):
            window.sweep(t)

    def _delete(self, aid: str) -> None:
        self.store.pop(aid, None)

    def avatar_bytes(self, name: str) -> bytes | None:
        """GET /avatars/<name>: a ready avatar's file, else None."""
        self.sweep()
        match = AVATAR_FILE.fullmatch(name)
        entry = self.store.get(match.group(1)) if match else None
        return entry.data if entry is not None and entry.status == "ready" else None

    def _evictable(self, t: float) -> list[StoredAvatar]:
        """What a full store may drop, in order: released bodies, oldest first; then the bodies
        of connections idle for `idle_evict_seconds` (a script that holds connections open and
        says nothing), the longest idle first. Builds in flight are never dropped."""
        released = sorted((e for e in self.store.values() if e.released_at is not None),
                          key=lambda e: e.released_at)
        idle = []
        for entry in self.store.values():
            if entry.status == "ready" and entry.released_at is None and self._live(entry):
                quiet = t - self._active.get(entry.owner, entry.created)
                if quiet >= self.config.idle_evict_seconds:
                    idle.append((quiet, entry))
        idle.sort(key=lambda pair: -pair[0])
        return released + [entry for _, entry in idle]

    def _has_room(self, t: float) -> bool:
        return len(self.store) - len(self._evictable(t)) < self.config.max_stored

    def _make_room(self, t: float) -> bool:
        """Room in the store for one more, evicting what `_evictable` allows."""
        for entry in self._evictable(t):
            if len(self.store) < self.config.max_stored:
                break
            self._evict(entry)
        return len(self.store) < self.config.max_stored

    def _evict(self, entry: StoredAvatar) -> None:
        live = entry.released_at is None and self._live(entry)
        self._delete(entry.id)
        owner = entry.owner
        if live and owner.avatar_id == entry.id:
            log.info("roomcall: avatar %s dropped: the store is full and its owner is idle",
                     entry.id)
            self._tell_left(owner, entry)

    # -- uploads --------------------------------------------------------------------------------

    async def upload(self, read: Callable[[], Awaitable[bytes | bytearray]], *, room: Any,
                     peer: Any, key: Any, client: str | None = None) -> dict:
        """POST /avatar. Who is asking (`room`, `peer`, `key`: the request's headers) and every
        limit that needs no body are checked first; then the body is read (`read` enforces the
        size cap) and checked in a thread, in one of `upload_slots` — a busy server answers 503
        with a short Retry-After rather than queueing bodies in memory; then the checks run
        again (the world moved on meanwhile) and the build starts. Raises AvatarRequestError."""
        if not self.enabled:
            raise AvatarRequestError(503, "full bodies are off on this server")
        self.sweep()
        who = client_key(client)
        self._admit(room, peer, key, who)
        t = now()
        if self._client_attempts.count(who, t) >= self.config.ip_attempts_per_minute:
            raise AvatarRequestError(429, "too many uploads from your network; wait a minute",
                                     self._client_attempts.wait(who, t))
        if self._slots.locked():
            raise AvatarRequestError(503, "the server is busy with other uploads; try again in "
                                          "a moment", BUSY_RETRY)
        async with self._slots:
            self._client_attempts.add(who, t)
            body = await read()
            upload = await asyncio.to_thread(parse_upload, body, room=room, peer=peer, key=key)
            del body
        owner = self._admit(room, peer, key, who)
        return self._accept(owner, upload, who)

    def _admit(self, room: Any, peer_id: Any, key: Any, who: str) -> Any:
        """The live peer an upload comes from, if every limit that needs no body lets it build
        now; else AvatarRequestError."""
        if not all(isinstance(v, str) and 0 < len(v) <= _TOKEN for v in (room, peer_id, key)):
            raise _bad("room, peer and key are required")
        peer = self.find_peer(room, peer_id) if _ROOM.fullmatch(room) else None
        if peer is None or not _same(str(getattr(peer, "key", "")), key):
            raise AvatarRequestError(403, "not a person in that room (reload the page?)")
        c, t = self.config, now()
        building = [e for e in self.store.values() if e.status == "building"]
        if any(e.owner is peer for e in building):
            raise AvatarRequestError(409, "your body is still being built")
        last = self._last_upload.get(peer)
        if last is not None and t - last < c.cooldown:
            wait = max(1, math.ceil(c.cooldown - (t - last)))
            raise AvatarRequestError(429, f"you can make a new body in {wait} s", wait)
        if sum(e.room == room for e in building) >= c.room_max_building:
            raise AvatarRequestError(429, "this room is building bodies already; try again "
                                          "in a minute", 60)
        if len(building) >= c.max_building:
            raise AvatarRequestError(429, "this server is building many bodies right now; try "
                                          "again in a minute", 60)
        if self._room_builds.count(room, t) >= c.room_builds_per_hour:
            raise AvatarRequestError(429, "this room has made all the bodies it can this hour",
                                     self._room_builds.wait(room, t))
        if self._client_uploads.count(who, t) >= c.ip_uploads_per_hour:
            raise AvatarRequestError(429, "your network has made all the bodies it can this hour",
                                     self._client_uploads.wait(who, t))
        held = sum(1 for e in self.store.values()
                   if e.client == who and e.released_at is None and e.owner is not peer)
        if held >= c.ip_max_stored:
            raise AvatarRequestError(429, "your network holds all the bodies it can; try again "
                                          "later")
        if not self._has_room(t):
            raise AvatarRequestError(503, "this server is holding all the bodies it can; "
                                          "try again later")
        return peer

    def _accept(self, peer: Any, upload: Upload, who: str) -> dict:
        t = now()
        if not self._make_room(t):
            raise AvatarRequestError(503, "this server is holding all the bodies it can; "
                                          "try again later")
        aid, akey = "a" + secrets.token_hex(8), secrets.token_hex(16)
        entry = StoredAvatar(aid, _hash(akey), upload.room, peer, created=t, client=who,
                             gpu=self._gpu_share(who))
        self.store[aid] = entry
        self._last_upload[peer] = t
        self._room_builds.add(upload.room, t)
        self._client_uploads.add(who, t)
        if peer.avatar_info is None:
            peer.avatar_info = {"status": "building"}
        self.broadcast(upload.room, {"t": "avatar", "id": peer.id, "status": "building"})
        task = asyncio.create_task(self._build(entry, upload))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return {"avatar": aid, "key": akey, "status": "building"}

    def _gpu_share(self, who: str) -> bool:
        """Whether a new upload from `who` may use the GPU: each client address gets
        `ip_gpu_per_day` of the day's GPU builds, so nobody can spend everyone's budget; past
        that it gets the stand-in. Counted in this process (a restart forgets it; the day's cap
        itself is the budget's)."""
        if self.config.mode != "modal":
            return True
        day = utc_date()
        if day != self._gpu_day:
            self._gpu_day, self._gpu_used = day, {}
        used = self._gpu_used.get(who, 0)
        if used >= self.config.ip_gpu_per_day:
            return False
        self._gpu_used[who] = used + 1
        return True

    async def _build(self, entry: StoredAvatar, upload: Upload) -> None:
        """Never raises into the loop: the builder (`build_timeout` at most), then the stand-in,
        then `failed`."""
        started = time.perf_counter()
        builder = self.builder
        if not entry.gpu:
            builder = functools.partial(
                build_standin, splats=self.config.standin_splats,
                note=f"{STANDIN_NOTE} (your network has had its share of today's GPU bodies)")
        try:
            data, kind, note = await asyncio.wait_for(
                builder(upload.images, upload.mode, entry.id), self.config.build_timeout)
            await asyncio.to_thread(unpack, data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            slow = isinstance(exc, asyncio.TimeoutError)
            log.warning("roomcall: avatar %s: the builder %s (%s: %s); using the stand-in",
                        entry.id, "took too long" if slow else "failed", type(exc).__name__, exc)
            try:
                data, kind, note = await build_standin(
                    upload.images, upload.mode, entry.id, splats=self.config.standin_splats,
                    note=f"{STANDIN_NOTE} (the body builder "
                         f"{'took too long' if slow else 'failed'})")
                await asyncio.to_thread(unpack, data)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("roomcall: avatar %s: the stand-in failed too", entry.id)
                self._failed(entry, "the body could not be built")
                return
        log.info("roomcall: avatar %s (%s) ready in %.1f s, %d bytes", entry.id, kind,
                 time.perf_counter() - started, len(data))
        if self.store.get(entry.id) is not entry:
            return   # expired or evicted while it was built
        entry.status, entry.data, entry.kind, entry.note = "ready", data, str(kind), str(note)
        if self._live(entry):
            self._attach(entry.owner, entry)

    def _attach(self, peer: Any, entry: StoredAvatar) -> None:
        """`entry` (ready) becomes `peer`'s body, replacing any it had; the room is told."""
        old = peer.avatar_id
        peer.avatar_id = entry.id
        peer.avatar_info = entry.ready_info()
        if old and old != entry.id:
            self._delete(old)
        message = {"t": "avatar", "id": peer.id, **entry.ready_info()}
        if entry.note:
            message["note"] = entry.note
        self.broadcast(entry.room, message)

    def _failed(self, entry: StoredAvatar, note: str) -> None:
        self._delete(entry.id)
        if self._live(entry):
            peer = entry.owner
            if peer.avatar_id is None:
                peer.avatar_info = None
            self.broadcast(entry.room, {"t": "avatar", "id": peer.id, "status": "failed",
                                        "note": note})

    # -- connections ----------------------------------------------------------------------------

    def claim(self, peer: Any, room: str, aid: Any, key: Any) -> None:
        """{"t": "avatar", "claim": aid, "key": akey} after a reconnect: a body this browser made
        (it kept the key), ready or still building, moves to this connection. A live connection
        it moves away from (a duplicated tab carries the key along) is told, with its room, what
        it has left. A claim that does not check out — a wrong key, or a body forgotten since
        (released over `keep_seconds` ago, or the server restarted) — gets "none" back, to the
        claimer alone while it has no body: its page stops waiting and can offer a new one."""
        self.sweep()
        entry = None
        if (isinstance(aid, str) and AVATAR_ID.fullmatch(aid) and isinstance(key, str)
                and 0 < len(key) <= _TOKEN):
            entry = self.store.get(aid)
            if entry is not None and not _same(entry.key_hash, _hash(key)):
                entry = None
        if entry is None:
            if peer.avatar_info is None:
                self.tell(peer, {"t": "avatar", "id": peer.id, "status": "none"})
            return
        previous = entry.owner
        if previous is not None and previous is not peer and self._live(entry):
            self._tell_left(previous, entry)
        t = now()
        for other in self.store.values():   # a body of its own this connection had: let it go
            if other is not entry and other.owner is peer:
                other.owner, other.released_at = None, t
        entry.owner, entry.room, entry.released_at = peer, room, None
        if entry.status == "ready":
            self._attach(peer, entry)
        else:
            if peer.avatar_id is None:
                peer.avatar_info = {"status": "building"}
            self.broadcast(room, {"t": "avatar", "id": peer.id, "status": "building"})

    def _tell_left(self, peer: Any, gone: StoredAvatar) -> None:
        """`gone` (ready or building) is no longer `peer`'s — it moved to another connection, or
        a full store dropped it. `peer`, still live, and its room hear what it has left: an
        older ready body, a build of its own in flight, or nothing."""
        if peer.avatar_id == gone.id:
            peer.avatar_id = None
        kept = self.store.get(peer.avatar_id) if peer.avatar_id else None
        building = any(e is not gone and e.owner is peer and e.status == "building"
                       for e in self.store.values())
        if kept is not None and kept.status == "ready":
            peer.avatar_info = kept.ready_info()
            self.broadcast(gone.room, {"t": "avatar", "id": peer.id, **kept.ready_info()})
        else:
            peer.avatar_id = None
            peer.avatar_info = {"status": "building"} if building else None
            self.broadcast(gone.room, {"t": "avatar", "id": peer.id, "status": "none"})
        if building:
            self.broadcast(gone.room, {"t": "avatar", "id": peer.id, "status": "building"})

    def leave(self, peer: Any) -> None:
        """A connection is gone: its avatars are released (claimable for `keep_seconds`)."""
        t = now()
        for entry in self.store.values():
            if entry.owner is peer:
                entry.owner, entry.released_at = None, t
        peer.avatar_id = None
        self.sweep()

    def close(self) -> None:
        """The server is stopping: builds in flight are abandoned."""
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
