"""Scans sent into a call: someone scans a room and the room they are all in becomes it.

During a call, someone scans a room with the LiteReality Scanner app, opens the room link on that
phone and sends the scan — the scan's folder compressed to a .zip (iOS Files does that), or a bare
room.usdz — as one POST /scan. This module takes it from there:

1. **Who and how much, before a byte is read.** Who is asking comes in headers (the room, the
   peer and the upload key from its welcome, as POST /avatar does), so a stranger, someone over a
   limit, or a busy server is refused unread. The body is then streamed to disk (never memory)
   with a hard cap.
2. **Untrusted bytes, opened carefully.** A scan zip is anyone's file. Its central directory is
   sized up from the end record before `zipfile` parses it, then every member is checked: no
   absolute paths or `..` (zip-slip), no symlinks or devices, no encryption, at most
   `MAX_MEMBERS`, at most `MAX_UNCOMPRESSED` bytes unpacked, and nothing compressed more than
   `MAX_RATIO` times (a zip bomb). Only the scanner's own kinds of file (`ALLOWED_SUFFIXES`) are
   unpacked — into a fresh folder per job, by paths computed here, each copy counted against what
   the directory declared — and the room.usdz inside gets the same checks again. What the files
   SAY is checked too, since the builds act on it: the room.usdz's prim names (they become file
   names) and how much reading it takes (`_check_layers`), and each photo's and depth map's size
   in pixels, from its header (`_check_image`: a small PNG can decode to gigabytes).
3. **The builds run as child processes of the CLI** (`python -m litereality_agent quickroom` /
   `realroom`), never imported: the room call is a room operation and knows nothing of the
   pipeline, and a build in a child can be niced, timed out, held to a share of memory
   (`QUICK_MEMORY`, `CHILDREN_MEMORY`) and killed outright, and hands its memory back when it
   ends. The quick room comes first (seconds), then — on a server that makes
   them — the realistic objects (TRELLIS on Modal), each version the child writes swapped in as it
   lands (watched like `--watch`).
4. **One room only.** Every version goes to the room it was sent from, through the same version /
   settle / offer / swap path every other version takes (`server.RoomHub.offer_room`): what people
   hold and moved carries across by name. Other rooms on the server keep theirs. The room keeps its
   scan until a newer one, or until it has been empty for `KEEP_SECONDS` (time for a reload or a
   dropped network to come back); then its builds are killed, its files deleted and it goes back to
   the server's own room.
5. **Everyone hears how it goes**: {"t": "scan", "job", "status", "detail", "progress", "by"} to
   the room — uploading, building, quick, objects, done, failed — and GET /scan/<job> says the same
   to whoever polls.

Limits mirror `avatars`: per connection (one upload in flight, then `COOLDOWN_SECONDS`), per room
(one quick build at a time — a newer upload stops the room's realistic build — and
`ROOM_UPLOADS_PER_HOUR`), per client address (attempts per minute, uploads per hour, a share of the
day's realistic builds) and server-wide (`UPLOAD_SLOTS` bodies read at once, `MAX_JOBS` quick builds,
`MAX_REALISTIC` realistic builds, `REALISTIC_PER_DAY` GPU builds a day — past which a room still
gets its quick room, with a note saying why the objects are not coming).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import stat
import struct
import sys
import tempfile
import time
import weakref
import zipfile
import zlib
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from litereality_agent.room_ops.roomcall.avatars import (
    MemoryBudget,
    _same,
    _Window,
    client_key,
)

log = logging.getLogger(__name__)

MODES = ("off", "quick", "realistic")
# Who is uploading travels in headers (the same three as POST /avatar), checked before the body.
ROOM_HEADER, PEER_HEADER, KEY_HEADER = "x-roomcall-room", "x-roomcall-peer", "x-roomcall-key"
# What a scan may be sent as. A usdz is a zip too; the type says which one it is meant to be.
CONTENT_TYPES = {"application/zip": "zip", "application/x-zip-compressed": "zip",
                 "application/x-zip": "zip", "model/vnd.usdz+zip": "usdz", "model/usd": "usdz"}

MAX_UPLOAD_BYTES = 200 * 1024 * 1024    # the whole POST body
MAX_MEMBERS = 3000                       # entries in a scan zip (tea_room: ~180)
MAX_UNCOMPRESSED = 1024 ** 3             # bytes unpacked from one scan
MAX_RATIO = 100                          # unpacked / packed, per member and overall
RATIO_FLOOR = 1024 * 1024                # members (and scans) smaller than this are not ratio-checked
MAX_USDZ_BYTES = 16 * 1024 * 1024        # the room.usdz itself (RoomPlan's: ~40-100 KB)
MAX_USDZ_UNPACKED = 16 * 1024 * 1024     # what its layers unpack to (RoomPlan stores them as is)
MAX_NAME = 512                           # a member's path, characters
MAX_DEPTH = 12                           # ...and folders
ALLOWED_SUFFIXES = (".usdz", ".usda", ".jpg", ".jpeg", ".png", ".json", ".pcd", ".txt")
# What the room.usdz may hold. The builds read it with the quick room's own small USDA reader, and
# names in it become file names (a wall's texture is `<wall>.png`, an object's mesh
# `<object>-<digest>.obj`), so its CONTENT is checked as well as its container. Every prim name must be an identifier
# (USD's own rule: `USD_NAME`) — never a path. What the reader holds in memory grows with a
# layer's size times how deep its prims nest, and with every time a layer is referenced (it is
# parsed again each time), so those are bounded too. RoomPlan's (tea_room, office_room): 20-30
# layers of 1-6 KB, 100-130 prims nested 6 brackets deep, 40-50 asset paths, 60-90 KB read.
MAX_LAYER_BYTES = 2 * 1024 * 1024        # one member of the room.usdz
MAX_USD_PRIMS = 10_000                   # prims in all its layers
MAX_USD_REFERENCES = 5_000               # asset paths (@...@) in all its layers
MAX_USD_READ = 16 * 1024 * 1024          # layer bytes read, a layer counted each time it is named
MAX_USD_NESTING = 32                     # brackets open at once in a layer
USD_NAME = re.compile(r"[^\W\d]\w{0,63}")   # a USD identifier (letters, digits, _; UTF-8 allowed)
# A scan's photos and depth maps (the scanner's: 1920x1440 photos, 256x192 depth and confidence
# maps) are checked by their headers, never decoded: a PNG of a few hundred KB can say it is
# 13000 x 13000 and take gigabytes to decode. The builds keep every depth map of a capture in
# memory at once, so those are counted together too.
MAX_PHOTO_SIDE = 4096                    # px, either side of a photo (frame_*.jpg and any other)
MAX_PHOTO_PIXELS = 4096 * 4096
MAX_MAP_SIDE = 2048                      # px, either side of a depth_*.png / conf_*.png
MAX_MAP_PIXELS = 1024 * 1024
MAX_MAPS_PIXELS = 160 * 1024 * 1024      # all of them together (tea_room's 124: 6 M)
MAX_IMAGE_BYTES = 32 * 1024 * 1024       # one image file
MAX_JSON_BYTES = 1024 * 1024             # one .json (a frame's is under 1 KB)
_MAP = re.compile(r"(depth|conf)_\d+\.png", re.IGNORECASE)
_IMAGE_SUFFIXES = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg"}
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
JPEG_HEADER_BYTES = 1024 * 1024          # how far into a JPEG its size may be (after EXIF, ICC...)
JPEG_SEGMENTS = 512                      # ...and how many segments before it (a camera's: ~10-20)
MAX_HEADER_BYTES = 128 * 1024 * 1024     # image headers read in all, checking one scan
# Files a zip made on a Mac or Windows carries beside the scan: never part of it, never unpacked.
_JUNK_DIRS = {"__macosx"}
_JUNK_FILES = {".ds_store", "thumbs.db", "desktop.ini"}
# A file the CLI would read as something other than a scan (`RunContext.resolve`: a folder with a
# scene.json is a scene package, and its manifest names where the capture is): never unpacked.
_NEVER = {"scene.json"}
_FRAME = re.compile(r"frame_\d+\.jpe?g", re.IGNORECASE)
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06")

COOLDOWN_SECONDS = 60          # per connection, from an accepted upload
ROOM_UPLOADS_PER_HOUR = 12
UPLOAD_SLOTS = 2               # bodies read at once, server-wide
BUSY_RETRY = 3                 # seconds a busy server asks an upload to wait
MAX_JOBS = 2                   # quick builds (and uploads) in flight, server-wide
MAX_REALISTIC = 2              # realistic builds running, server-wide
REALISTIC_PER_DAY = 6          # GPU realistic builds per UTC day (persisted when hosted)
IP_ATTEMPTS_PER_MINUTE = 10    # uploads started per client address, accepted or not
IP_UPLOADS_PER_HOUR = 10       # uploads accepted per client address
IP_REALISTIC_PER_DAY = 3       # each address's share of the day's realistic builds
QUICK_TIMEOUT = 120.0          # seconds a quick build may take
REALISTIC_TIMEOUT = 30 * 60.0  # ...and a realistic one
# Memory, the backstop behind the checks on what a scan holds: a build's child that grows past its
# share is killed, and when all of them together pass `CHILDREN_MEMORY` the biggest is. Hosted,
# every room shares one container (6 GB, of which a room's model takes ~0.7 GB and twice that
# during a swap). Measured on tea_room: ~0.2 GB for a quick room, ~0.4 GB for `realroom --fake`.
# Resident memory, read from /proc: an address-space rlimit would count what the libraries only
# reserve (the realistic build's imports alone reserve 2.4 GB for 0.4 GB used, more with more
# cores) and fail good builds. Where /proc is missing (macOS) nothing is watched.
QUICK_MEMORY = 1536 * 1024 * 1024
REALISTIC_MEMORY = 2560 * 1024 * 1024
CHILDREN_MEMORY = 3 * 1024 ** 3
MEMORY_SECONDS = 0.25          # how often a child's memory is read
CHUNK_TIMEOUT = 60.0           # seconds an upload may send nothing before it is dropped
UPLOAD_TIMEOUT = 30 * 60.0     # a whole upload, however slow the phone's network
# ...but one trickling in slower than this on average, once it has had a minute, is dropped: an
# upload holds one of the server's few `UPLOAD_SLOTS` while it lasts.
MIN_UPLOAD_RATE = 8 * 1024     # bytes a second
SLOW_GRACE = 60.0
KEEP_SECONDS = 60.0            # a room's scan outlives its last person leaving this long
RECORD_SECONDS = 30 * 60.0     # GET /scan/<job> answers this long after a job ends
MAX_RECORDS = 200
OBJECTS_INTERVAL = 10.0        # seconds between the realistic build's versions (`realroom --interval`)
FAKE_INTERVAL = 3.0            # ...with the stand-in generator, which is much faster
FAKE_DELAY = 1.0               # seconds per stand-in object (`realroom --fake-delay`)
WATCH_SECONDS = 0.5            # how often the realistic build's folder is looked at
CHILD_NICE = 10                # the builds yield every contested core to the rooms' physics
PROGRESS_SECONDS = 0.5         # "uploading N%" at most this often
WRITE_CHUNK = 1024 * 1024      # an upload is written to disk in pieces this big, off the loop

UPLOADING, BUILDING, QUICK, OBJECTS, DONE, FAILED = (
    "uploading", "building", "quick", "objects", "done", "failed")
STATUSES = (UPLOADING, BUILDING, QUICK, OBJECTS, DONE, FAILED)
# A job in these holds its peer's one upload, and its room's one quick build.
_BUILDING = {UPLOADING, BUILDING}
_ENDED = {DONE, FAILED}

# The child: the `litereality` CLI in a fresh interpreter, at a lower priority. The same as
# `python -m litereality_agent ...`, with the niceness set first (no preexec_fn: it is unsafe in a
# process with threads, which this one has).
RUNNER = ("import os, sys\n"
          "try:\n"
          "    os.nice(int(sys.argv[1]))\n"
          "except (OSError, AttributeError, ValueError):\n"
          "    pass\n"
          "from litereality_agent.cli import main\n"
          "raise SystemExit(main(sys.argv[2:]))\n")


def now() -> float:
    """The service's clock (monotonic seconds); tests replace it to skip ahead."""
    return time.monotonic()


class ScanRequestError(Exception):
    """An upload refused: `status` is the HTTP status, `message` what the person is shown,
    `retry_after` (seconds) when waiting would help."""

    def __init__(self, status: int, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


class ChildFailed(Exception):
    """A build's child process failed; the message is what people are told."""


def _bad(message: str) -> ScanRequestError:
    return ScanRequestError(400, message)


def _too_big(message: str) -> ScanRequestError:
    return ScanRequestError(413, message)


# ---------------------------------------------------------------------------------------------
# Opening a scan
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Limits:
    """What one scan may be (from `ScanConfig`, so tests can shrink them)."""

    members: int = MAX_MEMBERS
    unpacked: int = MAX_UNCOMPRESSED
    ratio: float = MAX_RATIO
    ratio_floor: int = RATIO_FLOOR
    usdz_bytes: int = MAX_USDZ_BYTES
    usdz_unpacked: int = MAX_USDZ_UNPACKED
    layer_bytes: int = MAX_LAYER_BYTES
    usd_prims: int = MAX_USD_PRIMS
    usd_references: int = MAX_USD_REFERENCES
    usd_read: int = MAX_USD_READ
    usd_nesting: int = MAX_USD_NESTING
    photo_side: int = MAX_PHOTO_SIDE
    photo_pixels: int = MAX_PHOTO_PIXELS
    map_side: int = MAX_MAP_SIDE
    map_pixels: int = MAX_MAP_PIXELS
    maps_pixels: int = MAX_MAPS_PIXELS
    image_bytes: int = MAX_IMAGE_BYTES
    json_bytes: int = MAX_JSON_BYTES


@dataclass(frozen=True)
class ScanPlan:
    """What an upload holds and what will be unpacked from it (`inspect_upload`).

    `members` maps each member to unpack (by its name in the archive) to its path in the scan
    folder; `usdz` is where the room.usdz goes in it. A bare usdz upload has no members."""

    kind: str                                   # "zip" | "usdz"
    archive: Path
    members: tuple[tuple[str, PurePosixPath], ...]
    usdz_member: str | None                     # the room.usdz's name in the zip
    usdz: PurePosixPath                         # ...and where it goes
    label: str | None                           # the scan folder's own name, if it has one
    frames: int                                 # capture photos (frame_*.jpg) in it
    skipped: int                                # members left out (not a scan's kind of file)
    unpacked: int                               # bytes the members unpack to


@dataclass(frozen=True)
class Scan:
    """An unpacked scan, ready for the CLI: its folder, its name, how many photos."""

    root: Path
    label: str
    frames: int


def clean_label(value: Any, limit: int = 40) -> str | None:
    """A room name for people to see: printable, whitespace collapsed, at most `limit` chars."""
    if not isinstance(value, str):
        return None
    text = "".join(ch if ch.isprintable() else " " if ch.isspace() else "" for ch in value)
    text = " ".join(text.split())[:limit].strip()
    return text or None


def _end_record(path: Path) -> tuple[int, int]:
    """(entries, central directory bytes) from a zip's end records, read from the end of the file
    before `zipfile` parses anything: `zipfile` builds one object per directory entry, and a
    200 MB upload could hold millions of tiny ones. Raises ScanRequestError for a file with no
    end record (not a zip, or cut short)."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        tail_len = min(size, 22 + 65535)
        fh.seek(size - tail_len)
        tail = fh.read(tail_len)
        at = tail.rfind(b"PK\x05\x06")
        if at < 0 or len(tail) - at < 22:
            raise _bad("that file is not a zip archive (or it was cut short)")
        (_sig, _disk, _cd_disk, _here, entries, cd_size, cd_offset,
         _comment) = struct.unpack("<IHHHHIIH", tail[at:at + 22])
        if entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
            # ZIP64: the real counts are in the zip64 end record its locator points at.
            locator = tail.rfind(b"PK\x06\x07", 0, at)
            if locator < 0 or at - locator < 20:
                raise _bad("that zip archive is damaged (a zip64 record is missing)")
            (_sig, _disk, record, _disks) = struct.unpack("<IIQI", tail[locator:locator + 20])
            if record + 56 > size:
                raise _bad("that zip archive is damaged")
            fh.seek(record)
            head = fh.read(56)
            if len(head) < 56 or head[:4] != b"PK\x06\x06":
                raise _bad("that zip archive is damaged")
            (_sig, _rsize, _made, _need, _disk, _cd_disk, _here, entries, cd_size,
             cd_offset) = struct.unpack("<IQHHIIQQQQ", head)
    return entries, cd_size


def _open_zip(path: Path, *, members: int, what: str) -> zipfile.ZipFile:
    """`path` opened as a zip once its end record says it is small enough to parse."""
    entries, cd_size = _end_record(path)
    # A directory entry is at least 46 bytes plus its name; 1 KiB each is far more than a scan's.
    if entries > members or cd_size > members * 1024 + 65536:
        raise _too_big(f"{what} holds more than {members} files")
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError, EOFError):
        raise _bad(f"{what} is not a readable zip archive") from None
    if len(zf.infolist()) > members:
        zf.close()
        raise _too_big(f"{what} holds more than {members} files")
    return zf


def _member_path(info: zipfile.ZipInfo) -> PurePosixPath | None:
    """A member's path inside the scan, or None for one never unpacked (a folder entry, a Mac's
    or Windows' own files). Anything that would land outside the folder — an absolute path, a
    drive, `..` — refuses the whole upload: no scanner writes such a zip."""
    raw = info.filename
    if "\x00" in raw or len(raw) > MAX_NAME:
        raise _bad("that zip names a file the scanner never writes")
    name = raw.replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise _bad("that zip names a file outside its own folder (an absolute path)")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise _bad("that zip names a file outside its own folder ('..')")
    if len(parts) > MAX_DEPTH:
        raise _bad("that zip's folders nest too deep for a scan")
    mode = (info.external_attr >> 16) & 0xFFFF
    kind = stat.S_IFMT(mode)
    if info.create_system == 3 and kind and kind not in (stat.S_IFREG, stat.S_IFDIR):
        raise _bad("that zip holds links or special files, which a scan never has")
    if info.flag_bits & 0x1:
        raise _bad("that zip is encrypted")
    if info.is_dir() or not parts:
        return None
    lower = [p.lower() for p in parts]
    if any(p in _JUNK_DIRS for p in lower) or lower[-1] in _JUNK_FILES or parts[-1].startswith("._"):
        return None
    return PurePosixPath(*parts)


def _check_ratio(packed: int, unpacked: int, limits: Limits, what: str) -> None:
    if unpacked >= limits.ratio_floor and unpacked > limits.ratio * max(packed, 1):
        raise _too_big(f"{what} unpacks to more than {limits.ratio:g} times its size — "
                       "a scan never does (a zip bomb?)")


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):g} MB"


def _amount(n: int) -> str:
    """Memory for people: "1.5 GB", "64 MB"."""
    return f"{n / 1024 ** 3:.1f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.0f} MB"


# Where the quick room's reader would take a prim's name — after `def` / `over` (and `class`) and
# an optional type, wherever that is in a layer, in a string or not, exactly as its own pattern
# finds one — and the name there is not an identifier (or is not closed): the reader's
# `"([^"]+)"` would hand that text on as a name, and from there into a file's path.
_BAD_PRIM = re.compile(r'\b(?:def|over|class)\s+(?:\w+\s+)?"(?!' + USD_NAME.pattern + '")')
_PRIM = re.compile(r"\b(?:def|over|class)\s")
_ASSET = re.compile(r"@([^@]*)@")
_QUOTED = re.compile(r'"[^"]*"|@[^@]*@')


def _nesting(text: str) -> int:
    """How many brackets of any kind are open at once at most, outside "strings" and @paths@."""
    import numpy as np

    bare = np.frombuffer(_QUOTED.sub("", text).encode("utf-8"), dtype=np.uint8)
    step = np.zeros(len(bare), dtype=np.int32)
    step[np.isin(bare, (0x28, 0x5B, 0x7B))] = 1           # ( [ {
    step[np.isin(bare, (0x29, 0x5D, 0x7D))] = -1          # ) ] }
    return int(np.cumsum(step).max(initial=0)) if len(step) else 0


def _check_layers(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo], limits: Limits) -> None:
    """What the room.usdz's layers SAY, as the quick room's reader would read it: every prim name
    an identifier, and no more prims, asset paths, nesting or bytes read (a layer counted again
    each time it is named: the reader parses it again each time) than `limits` allow. Any member
    is a layer if it is text — the reader reads whatever a reference names."""
    sizes = {info.filename: info.file_size for info in infos}
    prims = references = 0
    named: list[tuple[str, str]] = []
    for info in infos:
        try:
            text = zf.read(info).decode("utf-8")
        except UnicodeDecodeError:
            continue                     # not text: no layer the reader could read either
        except (zipfile.BadZipFile, OSError, EOFError, ValueError, NotImplementedError,
                RuntimeError, zlib.error) as exc:
            raise _bad(f"that room.usdz is damaged ({type(exc).__name__})") from None
        if _BAD_PRIM.search(text):
            raise _bad("that room.usdz names a part with something other than letters, digits "
                       "and _ — not a RoomPlan export")
        prims += sum(1 for _ in _PRIM.finditer(text))
        if prims > limits.usd_prims:
            raise _too_big(f"that room.usdz has more than {limits.usd_prims} parts — more than "
                           "a RoomPlan room ever has")
        for m in _ASSET.finditer(text):
            references += 1
            if references > limits.usd_references:
                raise _too_big(f"that room.usdz names more than {limits.usd_references} files")
            named.append((info.filename, m.group(1)))
        if _nesting(text) > limits.usd_nesting:
            raise _bad("that room.usdz nests its parts deeper than a RoomPlan room ever does")
    # The reader reads the root layer, then each layer a reference names, every time it is named
    # (resolved as the reader resolves it: relative to the naming layer when it starts with '.').
    root = next((n for n in sizes if n.endswith(".usda") and "/" not in n), None)
    read = sizes.get(root, 0) if root is not None else 0
    for layer, ref in named:
        target = str(PurePosixPath(PurePosixPath(layer).parent / ref)) if ref.startswith(".") \
            else ref
        read += sizes.get(target, 0)
        if read > limits.usd_read:
            raise _too_big("that room.usdz names its layers so many times that reading it would "
                           f"take more than {_mb(limits.usd_read)} — a RoomPlan room's is ~0.1 MB")


def check_usdz(path: Path, limits: Limits = Limits()) -> None:
    """A room.usdz that the quick build may open: a small zip of ASCII USD layers, with the same
    checks as a scan zip, then the layers themselves (`_check_layers`): the builds turn what they
    say into memory and into file names."""
    path = Path(path)
    if path.stat().st_size > limits.usdz_bytes:
        raise _too_big(f"that room.usdz is bigger than {_mb(limits.usdz_bytes)}")
    with open(path, "rb") as fh:
        if fh.read(4) not in _ZIP_MAGIC:
            raise _bad("that is not a RoomPlan room.usdz")
    zf = _open_zip(path, members=limits.members, what="that room.usdz")
    with zf:
        infos = zf.infolist()
        total = packed = 0
        layers = 0
        for info in infos:
            if info.flag_bits & 0x1:
                raise _bad("that room.usdz is encrypted")
            if info.file_size > limits.layer_bytes:
                raise _too_big(f"{PurePosixPath(info.filename).name} in that room.usdz is bigger "
                               f"than {_mb(limits.layer_bytes)} — a RoomPlan room's are a few KB")
            total += info.file_size
            packed += info.compress_size
            _check_ratio(info.compress_size, info.file_size, limits, "that room.usdz")
            name = info.filename.lower()
            if name.endswith((".usdc", ".usd")):
                raise _bad("that room.usdz holds binary USD layers: send the scanner's own "
                           "export (RoomPlan writes text layers)")
            if name.endswith(".usda"):
                layers += 1
        if total > limits.usdz_unpacked:
            raise _too_big(f"that room.usdz unpacks to more than {_mb(limits.usdz_unpacked)}")
        _check_ratio(packed, total, limits, "that room.usdz")
        if not layers:
            raise _bad("that is not a RoomPlan room.usdz (it holds no USD layers)")
        _check_layers(zf, infos, limits)


def _image_size(fh, budget: int = JPEG_HEADER_BYTES) -> tuple[str, int, int, int]:
    """("png" | "jpeg", width, height, bytes read) from an image's header, without decoding it:
    the size Pillow would decode it at (Pillow goes by the content, not the file's name). A
    JPEG's size is in its SOF segment, after any EXIF / ICC / XMP ones: at most `budget` bytes
    and `JPEG_SEGMENTS` segments are read to find it. Raises ValueError for anything else."""
    head = fh.read(24)
    if head.startswith(_PNG_MAGIC):
        if len(head) < 24 or head[12:16] != b"IHDR":
            raise ValueError("a PNG without its header")
        width, height = struct.unpack(">II", head[16:24])
        return "png", width, height, len(head)
    if not head.startswith(b"\xff\xd8"):
        raise ValueError("neither a PNG nor a JPEG")
    buf, pos, seen = bytearray(head), 2, len(head)

    def need(n: int) -> None:            # buf[pos:pos + n] read in
        nonlocal pos, seen
        while len(buf) - pos < n:
            if pos:
                del buf[:pos]
                pos = 0
            data = fh.read(8192)
            if not data:
                raise ValueError("a JPEG cut short")
            seen += len(data)
            if seen > budget:
                raise ValueError("a JPEG whose size is not in its header")
            buf.extend(data)

    width = height = 0
    for _segment in range(JPEG_SEGMENTS):    # segment by segment, up to the image data (SOS)
        need(2)
        if buf[pos] != 0xFF:
            raise ValueError("a damaged JPEG")
        marker = buf[pos + 1]
        if marker == 0xFF:               # a fill byte
            pos += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            pos += 2                     # markers without a length
            continue
        if marker == 0xD9:
            break                        # the end, and no image data
        need(4)
        length = struct.unpack(">H", buf[pos + 2:pos + 4])[0]
        if length < 2:
            raise ValueError("a damaged JPEG")
        if marker in _JPEG_SOF:
            need(9)
            h, w = struct.unpack(">HH", buf[pos + 5:pos + 9])
            if not (w and h):
                raise ValueError("a JPEG without its size")
            width, height = max(width, w), max(height, h)
        if marker == 0xDA:
            break
        need(2 + length)
        pos += 2 + length
    else:
        raise ValueError("a JPEG with more segments before its image than any camera writes")
    if not (width and height):
        raise ValueError("a JPEG without its size")
    return "jpeg", width, height, seen


def _check_image(zf: zipfile.ZipFile, info: zipfile.ZipInfo, name: str, limits: Limits,
                 budget: list[int]) -> int:
    """One of a scan's images, by its header: the format its name says, and no bigger than a scan's
    photos (or depth / confidence maps) are. `budget` holds the header bytes the scan's images
    may still take to read, and is used up. Returns its pixels if it is a map, else 0."""
    if info.file_size > limits.image_bytes:
        raise _too_big(f"{name} in that zip is bigger than {_mb(limits.image_bytes)} — more than "
                       "any scan's photo")
    try:
        with zf.open(info) as fh:
            kind, width, height, seen = _image_size(fh, min(JPEG_HEADER_BYTES, budget[0]))
        budget[0] -= seen
    except ValueError as exc:
        raise _bad(f"{name} in that zip is not an image a scan has ({exc})") from None
    except (zipfile.BadZipFile, OSError, EOFError, NotImplementedError, RuntimeError,
            zlib.error) as exc:
        raise _bad(f"that zip is damaged ({name}: {type(exc).__name__})") from None
    expected = _IMAGE_SUFFIXES[PurePosixPath(name).suffix.lower()]
    if kind != expected:
        raise _bad(f"{name} in that zip is a {kind.upper()}, not the {expected.upper()} its name "
                   "says")
    is_map = bool(_MAP.fullmatch(name))
    side, pixels = (limits.map_side, limits.map_pixels) if is_map else \
        (limits.photo_side, limits.photo_pixels)
    if width > side or height > side or width * height > pixels:
        what = "depth maps" if is_map else "photos"
        raise _too_big(f"{name} in that zip is {width} x {height} pixels — bigger than a scan's "
                       f"{what} are")
    return width * height if is_map else 0


def inspect_upload(path: Path, kind: str, limits: Limits = Limits(), *,
                   usdz_out: Path | None = None) -> ScanPlan:
    """Check an upload without unpacking it, and plan what will be. Blocking: run it in a thread.

    A zip: its directory is read (after `_end_record` sized it up), every member is checked
    (`_member_path`, sizes, ratio), and the scan is found in it — the folder holding room.usdz (or
    roomplan/room.usdz; else the one .usdz there is), the shallowest if the zip wraps it in
    folders. Only that folder's members of the kinds a scan has are planned for unpacking, each
    image among them sized up from its header (`_check_image`). The room.usdz is unpacked now, to
    `usdz_out`, and checked too (`check_usdz`), so a scan whose room cannot be read — or would be
    read into something harmful — is refused here rather than failing its build. Raises
    ScanRequestError."""
    path = Path(path)
    if path.stat().st_size == 0:
        raise _bad("the upload is empty")
    if kind == "usdz":
        check_usdz(path, limits)
        return ScanPlan("usdz", path, (), None, PurePosixPath("room.usdz"), None, 0, 0,
                        path.stat().st_size)
    with open(path, "rb") as fh:
        if fh.read(4) not in _ZIP_MAGIC:
            raise _bad("that file is not a zip archive")
    zf = _open_zip(path, members=limits.members, what="that zip")
    with zf:
        infos = zf.infolist()
        seen: set[str] = set()
        files: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        for info in infos:
            rel = _member_path(info)
            if rel is None:
                continue
            key = str(rel).lower()
            if key in seen:
                raise _bad(f"that zip holds {rel.name} twice")
            seen.add(key)
            files.append((info, rel))
        usdz = [(info, rel) for info, rel in files if rel.name.lower() == "room.usdz"]
        if not usdz:
            usdz = [(info, rel) for info, rel in files if rel.suffix.lower() == ".usdz"]
            if len(usdz) > 1:
                raise _bad("that zip holds several .usdz files and none is room.usdz; "
                           "compress the scan's own folder")
        if not usdz:
            raise _bad("there is no room.usdz in that zip: compress the scan's own folder "
                       "(the one holding room.usdz and the frame_… photos)")

        def root_of(rel: PurePosixPath) -> PurePosixPath:
            parent = rel.parent
            return parent.parent if parent.name.lower() == "roomplan" else parent

        roots = sorted({root_of(rel) for _info, rel in usdz}, key=lambda p: (len(p.parts), str(p)))
        if len(roots) > 1 and len(roots[0].parts) == len(roots[1].parts):
            raise _bad("that zip holds more than one scan; send one at a time")
        root = roots[0]
        usdz_info, usdz_rel = next((i, r) for i, r in usdz if root_of(r) == root)
        usdz_at = usdz_rel.relative_to(root)
        if usdz_at.name != "room.usdz":                 # "Room.usdz", "MyRoom.usdz": the CLI's name
            usdz_at = usdz_at.with_name("room.usdz")
        if usdz_at.parent.name.lower() == "roomplan":
            usdz_at = PurePosixPath("roomplan", "room.usdz")
        members, skipped, unpacked, packed, frames, maps = [], 0, 0, 0, 0, 0
        headers = [MAX_HEADER_BYTES]
        for info, rel in files:
            if info is usdz_info:
                continue
            try:
                inside = rel.relative_to(root) if root.parts else rel
            except ValueError:
                skipped += 1                            # beside the scan's folder, not in it
                continue
            if (inside.suffix.lower() not in ALLOWED_SUFFIXES or inside.name.lower() in _NEVER
                    or inside.name.lower() == "room.usdz" or str(inside).lower() == str(usdz_at).lower()):
                skipped += 1
                continue
            _check_ratio(info.compress_size, info.file_size, limits, f"{rel.name} in that zip")
            unpacked += info.file_size
            packed += info.compress_size
            if unpacked > limits.unpacked:
                raise _too_big(f"that scan unpacks to more than {limits.unpacked / 1e9:.1f} GB")
            suffix = inside.suffix.lower()
            if suffix in _IMAGE_SUFFIXES:
                maps += _check_image(zf, info, inside.name, limits, headers)
                if maps > limits.maps_pixels:
                    raise _too_big("that scan's depth maps are more than "
                                   f"{limits.maps_pixels / 1e6:.0f} million pixels in all — far "
                                   "more than a scan has")
            elif suffix == ".json" and info.file_size > limits.json_bytes:
                raise _too_big(f"{inside.name} in that zip is bigger than "
                               f"{_mb(limits.json_bytes)} — a scan's are a few KB")
            if len(inside.parts) == 1 and _FRAME.fullmatch(inside.name):
                frames += 1
            members.append((info.filename, inside))
        _check_ratio(packed, unpacked, limits, "that zip")
        if usdz_info.file_size > limits.usdz_bytes:
            raise _too_big(f"that room.usdz is bigger than {_mb(limits.usdz_bytes)}")
        if usdz_out is not None:
            _copy_member(zf, usdz_info, Path(usdz_out), limit=limits.usdz_bytes)
            check_usdz(Path(usdz_out), limits)
        label = clean_label(root.name) if root.parts else None
    return ScanPlan("zip", path, tuple(members), usdz_info.filename, usdz_at, label, frames,
                    skipped, unpacked + usdz_info.file_size)


def _copy_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path, *, limit: int) -> int:
    """Unpack one member to `target`, counting: a member that unpacks to more than its directory
    entry said (or than `limit`) is refused, whatever `zipfile` would have done about it."""
    allowed = min(info.file_size, limit)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with zf.open(info) as src, open(target, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > allowed:
                    raise _too_big(f"{PurePosixPath(info.filename).name} in that zip is bigger "
                                   "than it says")
                dst.write(chunk)
    except ScanRequestError:
        target.unlink(missing_ok=True)
        raise
    except (zipfile.BadZipFile, OSError, EOFError, ValueError, NotImplementedError,
            RuntimeError) as exc:
        target.unlink(missing_ok=True)
        raise _bad(f"that zip is damaged ({PurePosixPath(info.filename).name}: "
                   f"{type(exc).__name__})") from None
    return written


def unpack_scan(plan: ScanPlan, dest: Path, *, usdz: Path | None = None,
                limits: Limits = Limits()) -> Scan:
    """Unpack a planned upload into the fresh folder `dest` (blocking: run it in a thread). Every
    target is computed here from the plan and checked to be inside `dest`; nothing the archive
    says about paths is used as it is. `usdz` is the room.usdz `inspect_upload` already unpacked
    (moved into place); for a bare usdz upload it is the upload itself."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    root = dest.resolve()

    def inside(rel: PurePosixPath) -> Path:
        target = (dest / Path(*rel.parts)).resolve()
        if target == root or root not in target.parents:
            raise _bad("that zip names a file outside its own folder")
        return target

    room = inside(plan.usdz)
    room.parent.mkdir(parents=True, exist_ok=True)
    source = Path(usdz) if usdz is not None else plan.archive
    if plan.kind == "zip" and usdz is None:
        with _open_zip(plan.archive, members=limits.members, what="that zip") as zf:
            _copy_member(zf, zf.getinfo(plan.usdz_member), room, limit=limits.usdz_bytes)
    else:
        shutil.move(str(source), room)
    total = 0
    if plan.members:
        with _open_zip(plan.archive, members=limits.members, what="that zip") as zf:
            for name, rel in plan.members:
                total += _copy_member(zf, zf.getinfo(name), inside(rel),
                                      limit=limits.unpacked - total)
                if total > limits.unpacked:
                    raise _too_big("that scan unpacks to more than "
                                   f"{limits.unpacked / 1e9:.1f} GB")
    label = None
    named = dest / "scan_label.json"
    if named.is_file() and named.stat().st_size <= 64 * 1024:
        with contextlib.suppress(OSError, ValueError):
            info = json.loads(named.read_text(encoding="utf-8"))
            if isinstance(info, dict):
                label = clean_label(info.get("displayName")) or clean_label(info.get("sceneName"))
    label = label or plan.label or "scanned room"
    return Scan(dest, label, plan.frames)


# ---------------------------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------------------------

@dataclass
class ScanConfig:
    """How a server takes scans. `mode`: "off", "quick" (the quick room only) or "realistic"
    (then the realistic objects too). `fake` makes those with the offline stand-in (`realroom
    --fake`): no GPU, not counted against `budget`. The limits are fields so tests can shrink
    them; `command` replaces the child launcher (tests: a stand-in CLI)."""

    mode: str = "off"
    fake: bool = False
    fake_delay: float = FAKE_DELAY
    articulated: bool = False
    budget: Any = None                       # default MemoryBudget(realistic_per_day)
    realistic_per_day: int = REALISTIC_PER_DAY
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    limits: Limits = field(default_factory=Limits)
    cooldown: float = COOLDOWN_SECONDS
    room_uploads_per_hour: int = ROOM_UPLOADS_PER_HOUR
    upload_slots: int = UPLOAD_SLOTS
    max_jobs: int = MAX_JOBS
    max_realistic: int = MAX_REALISTIC
    ip_attempts_per_minute: int = IP_ATTEMPTS_PER_MINUTE
    ip_uploads_per_hour: int = IP_UPLOADS_PER_HOUR
    ip_realistic_per_day: int = IP_REALISTIC_PER_DAY
    quick_timeout: float = QUICK_TIMEOUT
    realistic_timeout: float = REALISTIC_TIMEOUT
    quick_memory: int = QUICK_MEMORY             # bytes resident; 0: not watched
    realistic_memory: int = REALISTIC_MEMORY
    children_memory: int = CHILDREN_MEMORY       # all builds' children together
    memory_seconds: float = MEMORY_SECONDS
    chunk_timeout: float = CHUNK_TIMEOUT
    upload_timeout: float = UPLOAD_TIMEOUT
    min_upload_rate: float = MIN_UPLOAD_RATE
    slow_grace: float = SLOW_GRACE
    keep_seconds: float = KEEP_SECONDS
    record_seconds: float = RECORD_SECONDS
    interval: float | None = None            # default OBJECTS_INTERVAL, FAKE_INTERVAL with `fake`
    watch_seconds: float = WATCH_SECONDS
    nice: int = CHILD_NICE
    scratch: Path | None = None              # where jobs unpack (default: a temp folder)
    command: Sequence[str] | None = None     # the child launcher before the CLI's arguments
    store: Any = None                        # rooms' scanned versions across restarts (`RoomStore`)
    # Where each scan's photoreal splat bundle is written (`<root>/<splats.scan_name(job id)>/`,
    # read by the server's `splats.SplatSource.per_room`; deleted when the room's next scan goes
    # in or the room is let go); None: no splats for scanned rooms.
    splats_root: Path | None = None
    splat_timeout: float = 1200.0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"uploads mode must be one of {', '.join(MODES)}")
        if self.budget is None:
            self.budget = MemoryBudget(self.realistic_per_day)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def launcher(self) -> list[str]:
        if self.command is not None:
            return list(self.command)
        return [sys.executable, "-c", RUNNER, str(int(self.nice))]

    @property
    def objects_interval(self) -> float:
        if self.interval is not None:
            return float(self.interval)
        return FAKE_INTERVAL if self.fake else OBJECTS_INTERVAL


@dataclass(eq=False)
class ScanJob:
    id: str
    room: str
    owner: Any                  # the peer it came from (its id is "by")
    client: str
    dir: Path
    kind: str
    created: float
    status: str = UPLOADING
    stage: str = "upload"       # what it is working on: upload | quick | objects
    progress: float = 0.0
    detail: str = ""
    ended: float | None = None
    cancelled: bool = False
    task: asyncio.Task | None = None
    # The children running now: the splat child runs beside the objects one, so each is killed
    # by its own handle, and both are when the job stops.
    procs: list = field(default_factory=list)
    rss: dict = field(default_factory=dict)   # ...and the memory each held when last looked at
                                              # (bytes, by pid)
    offered: list[str] = field(default_factory=list)   # versions it gave the room, in order
    label: str = ""
    saving: list = field(default_factory=list)          # copies to the store, still being made
    offered_at: float = 0.0     # loop time of its latest offer to the room

    @property
    def by(self) -> str:
        return str(getattr(self.owner, "id", ""))

    def message(self) -> dict:
        return {"t": "scan", "job": self.id, "status": self.status, "detail": self.detail,
                "progress": round(min(1.0, max(0.0, float(self.progress))), 3), "by": self.by}

    def record(self) -> dict:
        return {"job": self.id, "status": self.status, "stage": self.stage,
                "progress": round(min(1.0, max(0.0, float(self.progress))), 3),
                "detail": self.detail}


@dataclass(eq=False)
class _RoomScans:
    """What the service knows of one room id: its latest job and message, whether it runs a
    scanned version, and the timer that lets it go after it empties."""

    id: str
    job: ScanJob | None = None
    latest: dict | None = None
    scanned: bool = False
    splats: list[str] = field(default_factory=list)   # its scans' bundles on disk, by name
    drop: asyncio.TimerHandle | None = None


class _ObjectsProgress:
    """The realistic build's progress, from the lines `litereality realroom` prints."""

    HEADER = re.compile(r"^realroom\s.*?:\s+(\d+) objects?, pass")
    GENERATING = re.compile(r"^\s*generating (\d+) object\(s\) with \S+:(.*)$")
    ARRIVED = re.compile(r"^\s*([A-Za-z0-9_]+) ([a-z_]+) in [0-9.]+ s$")
    FAILED = re.compile(r"^\s*([A-Za-z0-9_]+) ([a-z_]+) failed: (.*)$")

    def __init__(self, passes: int) -> None:
        self.total = 0
        self.passes = max(1, passes)
        self.arrived: dict[str, set[str]] = {}
        self.failed = 0

    def feed(self, line: str) -> bool:
        """Take one line; True if it changed the progress."""
        if m := self.HEADER.match(line):
            self.total = self.total or int(m.group(1))
            return False
        if m := self.GENERATING.match(line):
            self.total = max(int(m.group(1)), len(self.arrived))
            return True
        if m := self.FAILED.match(line):
            self.failed += 1
            return True
        if m := self.ARRIVED.match(line):
            self.arrived.setdefault(m.group(1), set()).add(m.group(2))
            self.total = max(self.total, len(self.arrived))
            return True
        return False

    @property
    def fraction(self) -> float:
        steps = sum(min(len(p), self.passes) for p in self.arrived.values())
        return steps / max(1, self.total * self.passes)

    def detail(self) -> str:
        total = max(self.total, len(self.arrived))
        some = len(self.arrived)
        if not total:
            return "Making realistic objects…"
        finals = sum(1 for p in self.arrived.values() if len(p) >= self.passes)
        if self.passes > 1 and some >= total:
            return f"Refining objects: {finals} of {total} final"
        return f"Making realistic objects: {some} of {total}"


def _last_line(lines: Sequence[str], scrub: Sequence[str] = ()) -> str:
    """What a failed child said last (its SystemExit message, or an exception's last line), with
    this server's own paths taken out."""
    for line in reversed(lines):
        text = line.strip()
        if not text or text.startswith(("Traceback", "File ", "^", "~")):
            continue
        for prefix in scrub:
            text = text.replace(prefix.rstrip("/") + "/", "").replace(prefix, "")
        return text[:300]
    return ""


def _consume(future: asyncio.Future) -> None:
    """Take a finished future's outcome, so asyncio never reports it as never retrieved."""
    if not future.cancelled():
        future.exception()


def resident_bytes(pid: int) -> int | None:
    """How much memory process `pid` holds (resident, from Linux's /proc), or None where that
    cannot be read: another system, or the process gone."""
    try:
        with open(f"/proc/{int(pid)}/statm", "rb") as fh:
            fields = fh.read().split()
        return int(fields[1]) * _page_size()
    except (OSError, ValueError, IndexError):
        return None


def _page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):
        return 4096


Installer = Callable[[str, Any], Awaitable[str]]


class ScanService:
    """The scans of one server (owned by its `RoomHub`).

    From the hub it needs: `find_peer(room, peer_id)` (a live peer), `broadcast(room, message)`,
    `install(room, scene_ref) -> version id` (load a scene as a version and offer it to that room
    only) and `release(room)` (the room's scan is over: back to the server's own room). The hub
    calls `room_emptied` / `room_back` as rooms empty and fill, and `latest(room)` for a joiner."""

    def __init__(self, config: ScanConfig, *, find_peer: Callable[[str, str], Any],
                 broadcast: Callable[[str, dict], None], install: Installer,
                 release: Callable[[str], None], offer_gap: float = 0.0) -> None:
        self.config = config
        # Seconds between two versions offered to one room: a room swaps a prefetch window after
        # an offer, and a newer offer before then replaces the one pending, so versions closer
        # than that would never be seen. The newest one waiting goes in once the gap is up.
        self.offer_gap = offer_gap
        self.find_peer = find_peer
        self.broadcast = broadcast
        self.install = install
        self.release = release
        self.jobs: dict[str, ScanJob] = {}
        self.rooms: dict[str, _RoomScans] = {}
        self._room_uploads = _Window(3600)
        self._client_uploads = _Window(3600)
        self._client_attempts = _Window(60)
        self._last_upload: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._realistic_day = ""
        self._realistic_used: dict[str, int] = {}
        self._realistic_running = 0
        # Never waited on (a busy server answers 503 at once), so it binds to no event loop.
        self._slots = asyncio.Semaphore(config.upload_slots)
        self._tasks: set[asyncio.Task] = set()
        self._scratch: Path | None = None

    # -- what clients are told ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def welcome_info(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "mode": "off"}
        return {"enabled": True, "mode": self.config.mode,
                "max_bytes": int(self.config.max_upload_bytes),
                "cooldown": int(self.config.cooldown)}

    def latest(self, room_id: str) -> dict | None:
        """The room's latest scan message, for someone joining in the middle of a build (or just
        after one: how the room they walk into came about)."""
        state = self.rooms.get(room_id)
        return dict(state.latest) if state is not None and state.latest else None

    def status(self, job_id: str) -> dict | None:
        """GET /scan/<job>."""
        self.sweep()
        job = self.jobs.get(job_id) if isinstance(job_id, str) else None
        return job.record() if job is not None else None

    def stats(self) -> dict:
        active = [j for j in self.jobs.values() if j.status not in _ENDED]
        return {"mode": self.config.mode, "building": sum(j.status in _BUILDING for j in active),
                "realistic": self._realistic_running,
                "rooms": sum(1 for r in self.rooms.values() if r.scanned),
                "realistic_today": int(getattr(self.config.budget, "used", 0) or 0)}

    # -- rooms ----------------------------------------------------------------------------------

    def room_back(self, room_id: str) -> None:
        """Someone is in `room_id` again: whatever was going to let its scan go waits."""
        state = self.rooms.get(room_id)
        if state is not None and state.drop is not None:
            state.drop.cancel()
            state.drop = None

    def room_emptied(self, room_id: str) -> None:
        """The last person left `room_id`: after `keep_seconds` (at once with 0) its builds are
        stopped, its files deleted and its scanned version let go."""
        state = self.rooms.get(room_id)
        if state is None:
            return
        if state.drop is not None:
            state.drop.cancel()
        if self.config.keep_seconds <= 0:
            self._drop_room(room_id)
            return
        state.drop = asyncio.get_running_loop().call_later(
            self.config.keep_seconds, self._drop_room, room_id)

    def _drop_room(self, room_id: str) -> None:
        state = self.rooms.pop(room_id, None)
        if state is None:
            return
        if state.drop is not None:
            state.drop.cancel()
        job = state.job
        if job is not None and job.status not in _ENDED:
            self._stop(job, "Stopped: everyone left the room", tell=False)
        if state.scanned:
            log.info("roomcall: room %s is empty; its scanned room is let go", room_id)
            self.release(room_id)
            store = self.config.store
            if store is not None:
                self._spawn(asyncio.to_thread(store.forget, room_id))
        # Its photoreal rooms too, once its build has ended (its splat child writes there).
        waits = [job.task] if job is not None and job.task is not None else []
        for name in state.splats:
            folder = self._splat_folder(name)
            if folder is not None:
                self._spawn(self._remove(folder, waits))

    def _splat_folder(self, name: str) -> Path | None:
        """Where the bundle `name` (`splats.scan_name`) is written (None: no splats here)."""
        root = self.config.splats_root
        return Path(root) / name if root is not None else None

    @staticmethod
    def _own(job: ScanJob, ref):
        """`ref` as one of this job's versions: its photoreal room is the scan's own bundle
        (`splats.scan_name`), and its id is the job's alone, so the same files sent into two
        rooms are two versions, each finding its own room's bundle (a version is shared by every
        room that has its id)."""
        from dataclasses import replace

        from litereality_agent.room_ops.roomcall.hosted import version_name
        from litereality_agent.room_ops.roomcall.splats import scan_name

        digest = hashlib.sha256(f"{ref.id}/{job.id}".encode()).hexdigest()
        return replace(ref, id=version_name(ref.label, digest), splat=scan_name(job.id))

    def scanned(self, room_id: str) -> None:
        """`room_id` runs a scanned version (restored from the store, say): let it go like one."""
        self.rooms.setdefault(room_id, _RoomScans(room_id)).scanned = True

    # -- uploads --------------------------------------------------------------------------------

    def sweep(self) -> None:
        t = now()
        ended = sorted((j for j in self.jobs.values() if j.ended is not None),
                       key=lambda j: j.ended)
        for job in ended:
            if t - job.ended >= self.config.record_seconds or len(self.jobs) > MAX_RECORDS:
                self.jobs.pop(job.id, None)
        for window in (self._room_uploads, self._client_uploads, self._client_attempts):
            window.sweep(t)

    def _scratch_root(self) -> Path:
        if self._scratch is None:
            base = self.config.scratch
            if base is None:
                self._scratch = Path(tempfile.mkdtemp(prefix="roomcall-scans-"))
            else:
                self._scratch = Path(base)
                self._scratch.mkdir(parents=True, exist_ok=True)
        return self._scratch

    async def upload(self, stream: Callable[[], AsyncIterator[bytes]], *, room: Any, peer: Any,
                     key: Any, client: str | None = None, content_type: str | None = None,
                     length: str | None = None) -> dict:
        """POST /scan. Everything that needs no body is checked first — the server takes scans,
        the type, the declared length, who is asking and every limit — then one of
        `upload_slots` is taken (a busy server answers 503 at once), the body is streamed to the
        job's folder with a hard cap and checked (`inspect_upload`, in a thread), and the build
        starts in the background. Returns {"job": id, "status"}; raises ScanRequestError."""
        if not self.enabled:
            raise ScanRequestError(503, "adding rooms is off on this server")
        kind = CONTENT_TYPES.get((content_type or "").split(";")[0].strip().lower())
        if kind is None:
            raise ScanRequestError(415, "send the scan as a .zip of its folder, or its room.usdz")
        declared = None
        if length is not None:
            try:
                declared = int(length)
            except ValueError:
                raise _bad("bad Content-Length") from None
            if declared > self.config.max_upload_bytes:
                raise _too_big(f"the scan is bigger than "
                               f"{self.config.max_upload_bytes // (1024 * 1024)} MB")
            if declared <= 0:
                raise _bad("the upload is empty")
        self.sweep()
        who = client_key(client)
        owner = self._admit(room, peer, key, who)
        t = now()
        if self._client_attempts.count(who, t) >= self.config.ip_attempts_per_minute:
            raise ScanRequestError(429, "too many scans from your network; wait a minute",
                                   self._client_attempts.wait(who, t))
        if self._slots.locked():
            raise ScanRequestError(503, "the server is busy with other uploads; try again in a "
                                        "moment", BUSY_RETRY)
        scratch = self._scratch_root()
        need = self.config.max_upload_bytes + self.config.limits.unpacked
        with contextlib.suppress(OSError):
            if shutil.disk_usage(scratch).free < need:
                raise ScanRequestError(503, "the server is short of disk space for scans; try "
                                            "again later", 60)
        async with self._slots:
            self._client_attempts.add(who, t)
            job = self._new_job(owner, room, who, kind, scratch)
            try:
                body = await self._receive(job, stream, declared)
                plan = await asyncio.to_thread(inspect_upload, body, kind, self.config.limits,
                                               usdz_out=job.dir / "room.usdz")
            except ScanRequestError as exc:
                self._fail(job, exc.message)
                raise
            except asyncio.CancelledError:
                self._fail(job, "The upload stopped part-way")
                raise
            except Exception as exc:  # noqa: BLE001 — the client went away, a read failed
                log.info("roomcall: scan %s: the upload failed (%s: %s)", job.id,
                         type(exc).__name__, exc)
                self._fail(job, "The upload stopped part-way")
                raise _bad("the upload stopped part-way") from None
        if job.cancelled:            # stopped while it came in (everyone left, a newer upload)
            raise ScanRequestError(409, job.detail or "the scan was stopped")
        self._accepted(job, owner)
        job.task = self._spawn(self._build(job, plan))
        return {"job": job.id, "status": job.status}

    def _admit(self, room: Any, peer_id: Any, key: Any, who: str) -> Any:
        """The live peer an upload comes from, if every limit that needs no body lets it in."""
        if not all(isinstance(v, str) and 0 < len(v) <= 64 for v in (room, peer_id, key)):
            raise _bad("room, peer and key are required")
        peer = self.find_peer(room, peer_id) if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", room) else None
        if peer is None or not _same(str(getattr(peer, "key", "")), key):
            raise ScanRequestError(403, "not a person in that room (reload the page?)")
        c, t = self.config, now()
        active = [j for j in self.jobs.values() if j.status in _BUILDING]
        if any(j.owner is peer for j in active):
            raise ScanRequestError(409, "your scan is still on its way")
        last = self._last_upload.get(peer)
        if last is not None and t - last < c.cooldown:
            wait = max(1, math.ceil(c.cooldown - (t - last)))
            raise ScanRequestError(429, f"you can add another room in {wait} s", wait)
        if any(j.room == room for j in active):
            raise ScanRequestError(429, "a room is being built here already; wait for it", 15)
        if len(active) >= c.max_jobs:
            raise ScanRequestError(429, "the server is building other rooms right now; try "
                                        "again in a minute", 30)
        if self._room_uploads.count(room, t) >= c.room_uploads_per_hour:
            raise ScanRequestError(429, "this room has added all the rooms it can this hour",
                                   self._room_uploads.wait(room, t))
        if self._client_uploads.count(who, t) >= c.ip_uploads_per_hour:
            raise ScanRequestError(429, "your network has added all the rooms it can this hour",
                                   self._client_uploads.wait(who, t))
        return peer

    def _new_job(self, owner: Any, room: str, who: str, kind: str, scratch: Path) -> ScanJob:
        jid = "s" + secrets.token_hex(8)
        folder = scratch / jid
        folder.mkdir(parents=True)
        job = ScanJob(jid, room, owner, who, folder, kind, created=now())
        state = self.rooms.setdefault(room, _RoomScans(room))
        previous = state.job
        if previous is not None and previous.status not in _ENDED:
            # One build per room: the room's realistic build (its quick room is in) gives way.
            self._stop(previous, "Stopped: a newer scan is on its way")
        state.job = job
        self.jobs[jid] = job
        self._set(job, UPLOADING, detail="Uploading the scan…", progress=0.0)
        return job

    def _accepted(self, job: ScanJob, owner: Any) -> None:
        t = now()
        self._last_upload[owner] = t
        self._room_uploads.add(job.room, t)
        self._client_uploads.add(job.client, t)

    async def _receive(self, job: ScanJob, stream: Callable[[], AsyncIterator[bytes]],
                       declared: int | None) -> Path:
        """The body, streamed into the job's folder: at most `max_upload_bytes`, nothing for
        `chunk_timeout` ends it, and so do `upload_timeout` overall and an average below
        `min_upload_rate` after `slow_grace`. Written in pieces of `WRITE_CHUNK` on a worker
        thread; the room hears how far it has got."""
        limit = self.config.max_upload_bytes
        path = job.dir / "upload.bin"
        loop = asyncio.get_running_loop()
        started = told = loop.time()
        received = 0
        buffer = bytearray()
        chunks = stream().__aiter__()
        handle = await asyncio.to_thread(open, path, "wb")
        try:
            while True:
                elapsed = loop.time() - started
                if elapsed > self.config.upload_timeout:
                    raise ScanRequestError(408, "the upload took too long")
                if elapsed > self.config.slow_grace and \
                        received < self.config.min_upload_rate * elapsed:
                    raise ScanRequestError(408, "the upload is coming in too slowly; try again "
                                                "on a better network")
                try:
                    chunk = await asyncio.wait_for(chunks.__anext__(), self.config.chunk_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    raise ScanRequestError(408, "the upload stalled") from None
                if job.cancelled:
                    raise ScanRequestError(409, job.detail or "the scan was stopped")
                received += len(chunk)
                if received > limit:
                    raise _too_big(f"the scan is bigger than {limit // (1024 * 1024)} MB")
                buffer += chunk
                if len(buffer) >= WRITE_CHUNK:
                    data, buffer = bytes(buffer), bytearray()
                    await asyncio.to_thread(handle.write, data)
                if loop.time() - told >= PROGRESS_SECONDS:
                    told = loop.time()
                    self._uploading(job, received, declared)
            if buffer:
                await asyncio.to_thread(handle.write, bytes(buffer))
        finally:
            await asyncio.to_thread(handle.close)
        if received == 0:
            raise _bad("the upload is empty")
        self._uploading(job, received, declared or received)
        return path

    def _uploading(self, job: ScanJob, received: int, declared: int | None) -> None:
        if declared:
            share = min(1.0, received / declared)
            self._set(job, UPLOADING, detail=f"Uploading the scan… {int(share * 100)}%",
                      progress=share)
        else:
            self._set(job, UPLOADING, detail=f"Uploading the scan… {received / 1e6:.0f} MB",
                      progress=0.0)

    # -- the build ------------------------------------------------------------------------------

    def _set(self, job: ScanJob, status: str, *, detail: str | None = None,
             progress: float | None = None, stage: str | None = None, tell: bool = True) -> None:
        """Move `job` on and tell its room (and keep it as the room's latest message)."""
        if job.status in _ENDED:
            return
        job.status = status
        if detail is not None:
            job.detail = detail[:300]
        if progress is not None and math.isfinite(progress):
            job.progress = progress
        if stage is not None:
            job.stage = stage
        if status in _ENDED:
            job.ended = now()
        state = self.rooms.get(job.room)
        if state is None or state.job is not job:
            return               # superseded or let go: nobody is following it any more
        message = job.message()
        state.latest = message
        if tell:
            self.broadcast(job.room, message)

    def _fail(self, job: ScanJob, detail: str) -> None:
        self._set(job, FAILED, detail=detail)
        self._cleanup(job)

    def _stop(self, job: ScanJob, detail: str, *, tell: bool = True) -> None:
        """Stop a job now: its child is killed (in the task's cleanup) and its room told."""
        if job.status in _ENDED:
            return
        job.cancelled = True
        self._set(job, FAILED, detail=detail, tell=tell)
        if job.task is not None and not job.task.done():
            job.task.cancel()              # its own cleanup kills the child and deletes the files
        else:
            self._kill(job)
            self._cleanup(job)

    async def _build(self, job: ScanJob, plan: ScanPlan) -> None:
        """Unpack, the quick room, then (where this server makes them) the realistic objects.
        Never raises into the loop; everything it started is stopped and deleted at the end."""
        realistic_slot = False
        splats = None
        try:
            self._set(job, BUILDING, stage="quick", detail="Building the quick room…",
                      progress=0.0)
            usdz = job.dir / "room.usdz"
            scan = await asyncio.to_thread(unpack_scan, plan, job.dir / "scan",
                                           usdz=usdz if usdz.is_file() else None,
                                           limits=self.config.limits)
            job.label = scan.label
            if plan.kind == "zip":
                plan.archive.unlink(missing_ok=True)       # unpacked: the zip is not needed now
            started = time.perf_counter()
            quick_out = job.dir / "quick"
            await self._child(job, ["quickroom", str(scan.root), "--out", str(quick_out),
                                    f"--label={scan.label}"], timeout=self.config.quick_timeout,
                              memory=self.config.quick_memory)
            ref = await asyncio.to_thread(self._ref, quick_out / "scene.xml", scan.label, "quick",
                                          job)
            await self._replace_splats(job)
            vid = await self.install(job.room, ref)
            job.offered.append(vid)
            job.offered_at = asyncio.get_running_loop().time()
            self._mark_scanned(job, quick_out / "scene.xml", vid, "quick")
            log.info("roomcall: scan %s: quick room %s for room %s in %.1f s", job.id, vid,
                     job.room, time.perf_counter() - started)
            go, note = await self._realistic_allowed(job, scan)
            splats = self._splat_task(job, scan, quick_out / "scene.xml",
                                      train=go and not self.config.fake)
            if not go:
                self._set(job, QUICK, detail="Quick room ready" + (f" ({note})" if note else ""),
                          progress=1.0)
                await self._splats_done(job, splats)
                self._set(job, DONE, detail="The room is ready" + (f" — {note}" if note else ""),
                          progress=1.0)
                return
            realistic_slot = True                 # taken by `_realistic_allowed`
            self._set(job, QUICK, detail="Quick room ready — realistic objects are next",
                      progress=1.0)
            count = await self._objects(job, scan)
            await self._splats_done(job, splats)
            self._set(job, DONE, stage="objects", progress=1.0,
                      detail=f"Done: {count} realistic object{'s' if count != 1 else ''}" if count
                      else "Done — no object could be made realistic; the quick room stays")
        except asyncio.CancelledError:
            if not job.cancelled:
                job.cancelled = True
                self._set(job, FAILED, detail="Stopped")
        except (ScanRequestError, ChildFailed) as exc:
            message = exc.message if isinstance(exc, ScanRequestError) else str(exc)
            if job.offered:
                self._set(job, FAILED, detail=f"Realistic objects stopped: {message}. The quick "
                                              "room stays.")
            else:
                self._set(job, FAILED, detail=f"Failed: {message}")
        except Exception as exc:  # noqa: BLE001 — a bug or a bad scene: say so, keep the room
            log.exception("roomcall: scan %s failed", job.id)
            what = f"{type(exc).__name__}: {exc}"[:200]
            self._set(job, FAILED, detail=(f"Realistic objects stopped ({what}). The quick room "
                                           "stays.") if job.offered else f"Failed: {what}")
        finally:
            if realistic_slot:
                self._realistic_running -= 1
            # The splat child runs beside the objects: stopped too (with train, a GPU run that
            # costs money) before the folder it reads goes, whatever ended the build.
            if splats is not None:
                splats.cancel()                   # a no-op once it has ended
                with contextlib.suppress(BaseException):
                    await splats
            self._kill(job)
            self._cleanup(job)

    async def _replace_splats(self, job: ScanJob) -> None:
        """The room's earlier scans' photoreal rooms are deleted as this scan's room goes in (their
        builds were stopped when this one came); this one's is kept for when the room is let go."""
        from litereality_agent.room_ops.roomcall.splats import scan_name

        state = self.rooms.get(job.room)
        if state is None or state.job is not job:
            return
        for name in state.splats:
            folder = self._splat_folder(name)
            if folder is not None:
                await asyncio.to_thread(shutil.rmtree, folder, True)
        state.splats = [scan_name(job.id)]

    def _splat_task(self, job: ScanJob, scan: Scan, scene: Path, *, train: bool):
        """The photoreal room for this scan, in a child beside the objects (`splatroom`): the
        instant splat from the photos and LiDAR in seconds, then, when `train`, the GPU's
        snapshots. Written where the server's splat source reads the scan's versions' bundle
        (`splats.scan_name`, set by `_own`), so browsers swap each in by themselves. The physics
        in the call stays as it is (``--keep-physics``).
        None when this server makes no splats or the upload has no photos."""
        from litereality_agent.room_ops.roomcall.splats import scan_name

        out = self._splat_folder(scan_name(job.id))
        if out is None or scan.frames == 0:
            return None
        args = ["splatroom", str(scan.root), "--scene", str(scene), "--out", str(out),
                "--keep-physics", f"--label={scan.label}"]
        if not train:
            args += ["--instant-only", "--no-masks"]
        return asyncio.ensure_future(self._child(job, args, timeout=self.config.splat_timeout,
                                                 memory=self.config.realistic_memory))

    async def _splats_done(self, job: ScanJob, task) -> None:
        """Wait for the splat child; its failure costs the room its photoreal look, never the
        scan (the meshes stay)."""
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("roomcall: scan %s: the photoreal room failed: %s", job.id, exc)

    def _ref(self, scene: Path, label: str, stage: str, job: ScanJob):
        from litereality_agent.room_ops.roomcall.hosted import scene_digest, version_name
        from litereality_agent.room_ops.roomcall.server import SceneRef

        if not scene.is_file():
            raise ChildFailed("the build wrote no scene")
        return self._own(job, SceneRef(scene=scene, id=version_name(label, scene_digest(scene)),
                                       label=label, stage=stage))

    def _mark_scanned(self, job: ScanJob, scene: Path, vid: str, stage: str) -> None:
        state = self.rooms.get(job.room)
        if state is not None and state.job is job:
            state.scanned = True
        store = self.config.store
        if store is not None:
            job.saving.append(self._spawn(asyncio.to_thread(
                store.save, job.room, scene, version=vid, label=job.label, stage=stage)))

    async def _realistic_allowed(self, job: ScanJob, scan: Scan) -> tuple[bool, str]:
        """Whether this job goes on to realistic objects — then one of `max_realistic` slots is
        taken for it, which the caller gives back — and if not, why (for people)."""
        if self.config.mode != "realistic":
            return False, ""
        if scan.frames == 0:
            return False, "no photos in the upload, so no realistic objects"
        if self._realistic_running >= self.config.max_realistic:
            return False, "realistic objects skipped: the server is making other rooms' right now"
        self._realistic_running += 1         # taken before the budget's await: no overbooking
        if self.config.fake:
            return True, ""
        day = time.strftime("%Y-%m-%d", time.gmtime())
        if day != self._realistic_day:
            self._realistic_day, self._realistic_used = day, {}
        used = self._realistic_used.get(job.client, 0)
        granted = False
        if used < self.config.ip_realistic_per_day:
            try:
                granted = await self.config.budget.take()
            except asyncio.CancelledError:
                self._realistic_running -= 1
                raise
            except Exception:  # noqa: BLE001 — an unknown budget is a spent one
                log.warning("roomcall: the realistic budget could not be checked", exc_info=True)
        if not granted:
            self._realistic_running -= 1
            if used >= self.config.ip_realistic_per_day:
                return False, "realistic objects skipped: your network has had its share for today"
            return False, "realistic objects skipped: today's limit on this server is reached"
        self._realistic_used[job.client] = used + 1
        return True, ""

    async def _objects(self, job: ScanJob, scan: Scan) -> int:
        """The realistic build: `realroom` in a child, writing a version every
        `objects_interval` into its folder, each one installed in the room as it lands (watched
        the way `--watch` watches a scene). Returns how many objects the room got."""
        from litereality_agent.room_ops.roomcall.server import LocalWatch

        c = self.config
        out = job.dir / "objects"
        args = ["realroom", str(scan.root), "--out", str(out), f"--label={scan.label}",
                "--pass", "both", "--interval", f"{c.objects_interval:g}"]
        if c.fake:
            args += ["--fake", "--fake-delay", f"{c.fake_delay:g}"]
        args += ["--articulated"] if c.articulated else ["--no-articulated"]
        job.stage = "objects"
        progress = _ObjectsProgress(passes=2)
        self._set(job, OBJECTS, detail=progress.detail(), progress=0.0)

        def on_line(line: str) -> None:
            if progress.feed(line):
                self._set(job, OBJECTS, detail=progress.detail(), progress=progress.fraction)

        watch = LocalWatch(out, label=scan.label, stage="objects", poll_seconds=c.watch_seconds)
        child = asyncio.ensure_future(self._child(job, args, timeout=c.realistic_timeout,
                                                  memory=c.realistic_memory, on_line=on_line))
        loop = asyncio.get_running_loop()
        waiting = None                  # the newest whole version not offered yet
        try:
            while not child.done():
                await asyncio.wait({child}, timeout=c.watch_seconds)
                waiting = await self._look(job, watch, out) or waiting
                if waiting is not None and loop.time() - job.offered_at >= self.offer_gap:
                    await self._install_objects(job, waiting)
                    waiting = None
            error = child.exception()
            # The last version, written just before the child ended, in once the gap is up.
            with contextlib.suppress(ChildFailed, ValueError, OSError):
                if (out / "objects.json").is_file():
                    final = await asyncio.to_thread(self._ref, out / "scene.xml", scan.label,
                                                    "objects", job)
                    if final.id not in job.offered:
                        waiting = final
            if waiting is not None:
                await asyncio.sleep(max(0.0, job.offered_at + self.offer_gap - loop.time()))
                await self._install_objects(job, waiting)
            if error is not None:
                raise error
        finally:
            if not child.done():
                child.cancel()
                with contextlib.suppress(BaseException):
                    await child
        if len(job.offered) > 1:
            self._mark_scanned(job, out / "scene.xml", job.offered[-1], "objects")
        try:
            report = json.loads((out / "objects.json").read_text(encoding="utf-8"))
            return len(report.get("objects") or {}) + len(report.get("replaced") or {})
        except (OSError, ValueError, AttributeError):
            return 0

    async def _look(self, job: ScanJob, watch, out: Path):
        """One look at the realistic build's folder: a new whole version there, or None. The
        quick room the child builds there first is no version of its own (no objects.json yet)."""
        try:
            ref = await watch.check()
        except (ValueError, OSError):
            return None               # nothing there yet, or caught mid-write: look again
        if ref is not None:
            ref = self._own(job, ref)
        if ref is None or ref.id in job.offered or not (out / "objects.json").is_file():
            return None
        return ref

    async def _install_objects(self, job: ScanJob, ref) -> None:
        """One objects version into the room. One that cannot be loaded is skipped (the next one
        may well load): it costs the room a version, not the whole build."""
        try:
            vid = await self.install(job.room, ref)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("roomcall: scan %s: objects version %s could not be loaded: %s", job.id,
                        ref.id, exc)
            job.offered.append(ref.id)        # not tried again
            return
        job.offered.append(vid)
        job.offered_at = asyncio.get_running_loop().time()
        log.info("roomcall: scan %s: objects version %s for room %s", job.id, vid, job.room)

    async def _child(self, job: ScanJob, args: Sequence[str], *, timeout: float, memory: int = 0,
                     on_line: Callable[[str], None] | None = None) -> None:
        """One CLI command in a child process: its own session (so it and anything it starts can
        be killed as one), niced, in the job's folder, with `timeout` and at most `memory` bytes
        resident (`_watch_memory`). Raises ChildFailed with what it said last, or on the timeout
        or the memory (after killing it)."""
        package_root = str(Path(__file__).resolve().parents[3])
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": os.pathsep.join(filter(None, (package_root,
                                                           os.environ.get("PYTHONPATH"))))}
        proc = await asyncio.create_subprocess_exec(
            *self.config.launcher(), *args, cwd=str(job.dir), env=env,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True, limit=1 << 20)
        job.procs.append(proc)
        errors: deque[str] = deque(maxlen=60)

        async def pump(reader: asyncio.StreamReader, sink: Callable[[str], None]) -> None:
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    line = await reader.read(1 << 16)
                if not line:
                    return
                text = line.decode("utf-8", "replace").rstrip()
                try:
                    sink(text)
                except Exception:  # noqa: BLE001 — a progress line we misread is no failure
                    log.debug("roomcall: scan %s: line %r", job.id, text, exc_info=True)

        def out_line(text: str) -> None:
            log.debug("roomcall: scan %s: %s", job.id, text)
            if on_line is not None:
                on_line(text)

        # Stopping a build cancels this (a newer scan, everyone gone, the server stopping): the
        # cancelled gather then ends with an exception nothing awaits, which asyncio would print
        # as "exception was never retrieved" — so its outcome is always taken here.
        work = asyncio.gather(pump(proc.stdout, out_line), pump(proc.stderr, errors.append),
                              proc.wait())
        work.add_done_callback(_consume)
        over: list[str] = []
        watch = asyncio.ensure_future(self._watch_memory(job, proc, args[0], memory, over))
        try:
            await asyncio.wait_for(work, timeout)
        except asyncio.TimeoutError:
            self._kill_proc(proc)
            raise ChildFailed(f"{args[0]} took longer than {timeout:g} s") from None
        finally:
            watch.cancel()
            if proc.returncode is None:
                self._kill_proc(proc)
                with contextlib.suppress(Exception):
                    await asyncio.shield(proc.wait())
            with contextlib.suppress(ValueError):
                job.procs.remove(proc)
            job.rss.pop(proc.pid, None)
        if over:
            raise ChildFailed(over[0])
        if proc.returncode != 0:
            said = _last_line(list(errors), scrub=[str(job.dir), str(job.dir.resolve())])
            log.warning("roomcall: scan %s: %s exited %s: %s", job.id, args[0], proc.returncode,
                        said)
            raise ChildFailed(said or f"{args[0]} exited with code {proc.returncode}")

    async def _watch_memory(self, job: ScanJob, proc, what: str, memory: int,
                            over: list[str]) -> None:
        """Every `memory_seconds`, how much the child holds: past `memory`, or the biggest of the
        builds' children when all of them together pass `children_memory`, it is killed, and
        `over` says why. Nothing is watched where that cannot be read (`resident_bytes`)."""
        c = self.config
        if memory <= 0 and c.children_memory <= 0:
            return
        while proc.returncode is None:
            rss = resident_bytes(proc.pid)
            if rss is None:
                return
            job.rss[proc.pid] = rss
            held = [n for j in {*self.jobs.values(), job} for n in j.rss.values()]
            total = sum(held)
            if memory > 0 and rss > memory:
                over.append(f"{what} needed more than {_amount(memory)} of memory")
            elif c.children_memory > 0 and total > c.children_memory and \
                    all(rss >= n for n in held):
                over.append("the server ran short of memory for building rooms; try again "
                            "later")
            if over:
                log.warning("roomcall: scan %s: %s killed at %.0f MB (%s)", job.id, what,
                            rss / 1e6, over[0])
                self._kill_proc(proc)
                return
            await asyncio.sleep(c.memory_seconds)

    def _kill(self, job: ScanJob) -> None:
        """Kill every child of the job and everything they started."""
        for proc in list(job.procs):
            self._kill_proc(proc)

    @staticmethod
    def _kill_proc(proc) -> None:
        """Kill one child and everything it started (its process group)."""
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError, AttributeError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()

    def _cleanup(self, job: ScanJob) -> None:
        """Delete the job's folder, off the loop (a scan can be a GB of files), once whatever
        it is still being copied to the store from is done."""
        if job.dir.exists():
            self._spawn(self._remove(job.dir, list(job.saving)))

    @staticmethod
    async def _remove(folder: Path, waits: list) -> None:
        if waits:
            await asyncio.gather(*waits, return_exceptions=True)
        await asyncio.to_thread(shutil.rmtree, folder, True)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._spawned_done)
        return task

    def _spawned_done(self, task: asyncio.Task) -> None:
        """A background task ended: one that failed (a copy to the store, a deletion) says so in
        a line of its own, rather than as asyncio's "exception was never retrieved"."""
        self._tasks.discard(task)
        if not task.cancelled() and (exc := task.exception()) is not None:
            log.warning("roomcall: scans: a background task failed: %s: %s",
                        type(exc).__name__, exc)

    def close(self) -> None:
        """The server is stopping: every build is killed and every job's files deleted."""
        for job in list(self.jobs.values()):
            job.cancelled = True
            self._kill(job)
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()
        for state in self.rooms.values():
            if state.drop is not None:
                state.drop.cancel()
        if self._scratch is not None:
            if self.config.scratch is None:
                shutil.rmtree(self._scratch, ignore_errors=True)
            else:
                for job in self.jobs.values():
                    shutil.rmtree(job.dir, ignore_errors=True)
        for state in self.rooms.values():
            for name in state.splats:
                folder = self._splat_folder(name)
                if folder is not None:
                    shutil.rmtree(folder, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# Keeping a room's scan across a restart
# ---------------------------------------------------------------------------------------------

STORE_TTL = 6 * 3600.0          # a kept scan older than this is no call any more (the hosted
                                # function's timeout is 6 h)
STORE_META = "room.json"


class DirRoomStore:
    """Rooms' scanned versions in a folder that outlives the server process — hosted, on the
    Volume — so a container that restarts during a call (a redeploy, a crash) puts each room back
    on its scan when its people reconnect, instead of on the demo.

    One folder per room id holding the version's scene files (`hosted.scene_files`: only what the
    scene names) and `STORE_META` {room, version, label, stage, saved}. A save replaces the
    folder whole; `commit` (the Volume's) runs after every change, under `lock` (shared with the
    Volume's reload, which refuses to run while files are open). Kept scans are forgotten when
    their room is let go after emptying, and ignored (then removed) once older than `ttl`.
    Blocking: the service calls it in a thread."""

    def __init__(self, root: Path, *, ttl: float = STORE_TTL, commit: Callable[[], Any] | None = None,
                 lock: Any = None) -> None:
        self.root = Path(root)
        self.ttl = ttl
        self.commit = commit
        self.lock = lock if lock is not None else contextlib.nullcontext()

    def _folder(self, room_id: str) -> Path | None:
        return self.root / room_id if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", room_id or "") else None

    def _commit(self) -> None:
        if self.commit is not None:
            try:
                self.commit()
            except Exception as exc:  # noqa: BLE001 — kept locally; the next commit carries it
                log.warning("roomcall: keeping a room's scan: commit failed: %s", exc)

    def save(self, room_id: str, scene: Path, *, version: str, label: str, stage: str) -> None:
        from litereality_agent.room_ops.roomcall.hosted import scene_files

        folder = self._folder(room_id)
        if folder is None:
            return
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            staging = self.root / f".{room_id}.{secrets.token_hex(4)}"
            try:
                for local, relative in scene_files(Path(scene)):
                    target = staging / Path(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(local, target)
                meta = {"room": room_id, "version": version, "label": label, "stage": stage,
                        "saved": time.time()}
                (staging / STORE_META).write_text(json.dumps(meta), encoding="utf-8")
                old = None
                if folder.exists():
                    old = self.root / f".{room_id}.old.{secrets.token_hex(4)}"
                    folder.rename(old)
                staging.rename(folder)
                if old is not None:
                    shutil.rmtree(old, ignore_errors=True)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            self._commit()

    def load(self, room_id: str):
        """The kept scan of `room_id` as a `server.SceneRef`, or None."""
        from litereality_agent.room_ops.roomcall.server import SceneRef

        folder = self._folder(room_id)
        if folder is None or not (folder / "scene.xml").is_file():
            return None
        try:
            meta = json.loads((folder / STORE_META).read_text(encoding="utf-8"))
            saved = float(meta["saved"])
            version, label, stage = str(meta["version"]), str(meta["label"]), str(meta["stage"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if time.time() - saved > self.ttl:
            self.forget(room_id)
            return None
        return SceneRef(scene=folder / "scene.xml", id=version, label=label[:80],
                        stage=stage if stage in ("quick", "objects") else "scene")

    def forget(self, room_id: str) -> None:
        folder = self._folder(room_id)
        if folder is None or not folder.exists():
            return
        with self.lock:
            shutil.rmtree(folder, ignore_errors=True)
            self._commit()

    def sweep(self) -> list[str]:
        """Remove kept scans older than `ttl` (and leftovers of interrupted saves)."""
        removed: list[str] = []
        if not self.root.is_dir():
            return removed
        with self.lock:
            for entry in list(self.root.iterdir()):
                if entry.name.startswith("."):
                    shutil.rmtree(entry, ignore_errors=True)
                    continue
                try:
                    saved = float(json.loads((entry / STORE_META).read_text())["saved"])
                except (OSError, ValueError, KeyError, TypeError):
                    saved = 0.0
                if time.time() - saved > self.ttl:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed.append(entry.name)
            if removed:
                self._commit()
        return removed
