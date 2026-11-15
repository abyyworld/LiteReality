"""The library of ready-made rooms a call can move into (the room picker).

Every call room starts on the deployed default room. The library is a short list of other rooms
already on the hosted Volume — the office, later someone's own scan — that anyone in a call can
move everyone in their call room into, photoreal splat and all, and back.

On the Volume it is one small file, `.library/rooms.json`:

    {"rooms": [{"name": "office_room", "label": "Office room",
                "version": "office_room-0123456789ab", "thumb": true}]}

`version` names a version folder beside it (`hosted.publish` writes those: `scene.xml`, and the
prebuilt `scene.glb`/`scene.json`/`roomcall.json`), and `.library/<name>.jpg` is the room's
thumbnail when there is one. The folder is dot-named, so it is never mistaken for a version and
never pruned. `name` is the room name the photoreal bundle is kept under (`splats/<name>/`), so a
picked room finds its splat the way the default room does.

This module reads the file and nothing else: it imports nothing from the server (which imports
it), and the hosted side writes the file (`hosted.add_to_library`).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LIBRARY_DIR = ".library"
LIBRARY_FILE = "rooms.json"
NAME = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
# A version folder's name: one path part, never "..", never a dot folder such as `.rooms`.
_FOLDER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# More than a picker can show; a file listing more is cut here rather than read whole each time.
MAX_ROOMS = 64
# What the server calls a call room that shows a scan of its own (`server.room_pick`): a ready
# room with that name could not be told apart from it, so none may have it.
SCAN = "scan"
MAX_LABEL = 80
MAX_STAGE = 16
META = "roomcall.json"


@dataclass(frozen=True)
class LibraryRoom:
    """One ready room: its name (also its splat bundle's name), how to show it, and the version
    folder it is loaded from (`prebuilt` when the geometry is already built there)."""

    name: str
    label: str
    version: str
    scene: Path
    stage: str = "scene"
    prebuilt: Path | None = None
    thumb: Path | None = None


class RoomLibrary:
    """The ready rooms, read through `load` and kept until the source changes.

    `stamp` says whether it has (any value that changes when the rooms may have: the file's
    modification time, say); it is looked at no more than every `poll_seconds`, so asking for
    the rooms on every join costs a comparison. Without `stamp` the rooms are read once.
    `rooms()` never raises: a source that fails is an empty library until it is fixed."""

    def __init__(self, load: Callable[[], list[LibraryRoom]], *, poll_seconds: float = 10.0,
                 stamp: Callable[[], Any] | None = None) -> None:
        self._load = load
        self._stamp = stamp
        self.poll_seconds = poll_seconds
        self._rooms: list[LibraryRoom] | None = None
        self._seen: Any = None
        self._looked = 0.0
        self._lock = threading.Lock()   # rooms() is called from worker threads

    @classmethod
    def folder(cls, root: Path, *, poll_seconds: float = 10.0) -> RoomLibrary:
        """The library on the Volume mounted at `root` (`root/.library/rooms.json`)."""
        root = Path(root)
        index = root / LIBRARY_DIR / LIBRARY_FILE

        def stamp() -> Any:
            # The folder's own time too: a thumbnail added beside an unchanged file shows up.
            return tuple(_mtime(path) for path in (index, index.parent))

        return cls(lambda: read_library(root), poll_seconds=poll_seconds, stamp=stamp)

    @classmethod
    def fixed(cls, rooms: list[LibraryRoom]) -> RoomLibrary:
        """A library that never changes (tests, a local server)."""
        rooms = list(rooms)
        return cls(lambda: rooms)

    def rooms(self) -> list[LibraryRoom]:
        with self._lock:
            t = time.monotonic()
            if self._rooms is not None and (self._stamp is None
                                            or t - self._looked < self.poll_seconds):
                return list(self._rooms)
            self._looked = t
            seen = None
            if self._stamp is not None:
                try:
                    seen = self._stamp()
                except Exception:  # noqa: BLE001 — looked at again next round
                    seen = None
            if self._rooms is None or seen != self._seen:
                self._seen = seen
                try:
                    self._rooms = list(self._load())
                except Exception as exc:  # noqa: BLE001 — an empty library, never a crash
                    log.warning("roomcall: reading the room library: %s", exc)
                    self._rooms = []
            return list(self._rooms)

    def get(self, name: str) -> LibraryRoom | None:
        return next((room for room in self.rooms() if room.name == name), None)


def _mtime(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def read_library(root: Path) -> list[LibraryRoom]:
    """The ready rooms `root/.library/rooms.json` lists, in its order. An entry is skipped when
    its name is not a room name (`NAME`) or is `SCAN`, its version folder has no scene.xml, or
    an earlier entry has its name; a missing or broken file is no rooms."""
    root = Path(root)
    path = root / LIBRARY_DIR / LIBRARY_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as exc:
        log.warning("roomcall: the room library %s cannot be read: %s", path, exc)
        return []
    entries = data.get("rooms") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return []
    rooms: list[LibraryRoom] = []
    names: set[str] = set()
    for entry in entries[:MAX_ROOMS]:
        if not isinstance(entry, dict):
            continue
        name, version = entry.get("name"), entry.get("version")
        if (not isinstance(name, str) or not NAME.fullmatch(name) or name in names
                or name == SCAN):
            continue
        if not isinstance(version, str) or not _FOLDER.fullmatch(version):
            continue
        folder = root / version
        scene = folder / "scene.xml"
        if not scene.is_file():
            continue
        label = entry.get("label")
        label = label.strip()[:MAX_LABEL] if isinstance(label, str) and label.strip() else name
        prebuilt = folder if ((folder / "scene.glb").is_file()
                              and (folder / "scene.json").is_file()) else None
        thumb = root / LIBRARY_DIR / f"{name}.jpg"
        names.add(name)
        rooms.append(LibraryRoom(name=name, label=label, version=version, scene=scene,
                                 stage=_stage(folder), prebuilt=prebuilt,
                                 thumb=thumb if thumb.is_file() else None))
    return rooms


def _stage(folder: Path) -> str:
    """The stage the version was published as (its `roomcall.json`), else "scene"."""
    try:
        meta = json.loads((folder / META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "scene"
    stage = meta.get("stage") if isinstance(meta, dict) else None
    return stage if isinstance(stage, str) and 0 < len(stage) <= MAX_STAGE else "scene"
