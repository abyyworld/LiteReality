"""The photoreal room in the call: splat bundles (`pipeline/splatroom`) served beside the physics.

A splat bundle is a folder holding ``splat.json`` and the SPZ files it names: the static room and
one splat per movable body, in that body's frame (`pipeline/splatroom/split.py`). Bundles change
on their own clock — the instant room in seconds, then a sharper snapshot every so often while
the GPU trains — so they never become new physics versions: the server watches the folder and
tells every room ``{"t": "splat", "name", "created"}``, and browsers fetch the new files and swap
them in under the bodies already on screen.

Where a room's bundle is:
- locally (`litereality roomcall ... --splat DIR`), one fixed folder for the room served;
- hosted, ``<Volume>/splats/<room name>/``, the room name being its versions' (`tea_room` for
  ``tea_room-1a2b3c4d5e6f``), uploaded by ``litereality splatroom <scan> --publish``;
- for a scan sent into a room during the call, ``<scratch>/splats/<scan_name(job id)>/``: named
  by the scan, not its label, so two rooms' scans of a "Living Room" never share one, and a scan
  labelled like the deployed room never stands in for the deployed room's bundle.

    GET /splat.json?v=<version>   the bundle for that version's room (404 when it has none), with
                                  every file as a URL: {"name", "stage", "step", "created",
                                  "room": url, "bodies": {body name: url}}, and when the bundle
                                  has them (well formed) "view" (where a newcomer stands),
                                  "capture" (the scan's camera spots: the walk zone) and
                                  "grabbable" ({body name: false} for a body whose splat holds
                                  too little of a real object to be offered for grabbing)
    GET /splat/<name>/<file>      one SPZ file (content-named: cached forever)
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path

BUNDLE = "splat.json"
SPLATS_DIR = "splats"
FILE = re.compile(r"^(room|body-[A-Za-z0-9_.-]{1,96})-[0-9a-f]{16}\.spz$")
NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_VERSION = re.compile(r"^(?P<name>[A-Za-z0-9_-]{1,48})-[0-9a-f]{12}$")
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_CAPTURE = 400              # capture spots a bundle may carry (the pipeline writes <= 200)
POLL_SECONDS = 2.0
# The URL name (``splat/<name>/<file>``) of the bundle of a version with no room name: never a
# room name itself (`hosted.scene_name` strips '-' from both ends; `room_name` refuses it).
NO_NAME = "-"


def room_name(version_id: str) -> str | None:
    """`tea_room` for the version id `tea_room-1a2b3c4d5e6f`; None for anything else."""
    m = _VERSION.match(version_id or "")
    return m.group("name") if m and m.group("name") != NO_NAME else None


# The bundle name of a version that has none (a room's scan restored after a restart: its
# bundle was on the old container's disk). No source has a folder for it.
NO_BUNDLE = ""


def scan_name(job_id: str) -> str:
    """The bundle name of one scan sent into a room (`scans.ScanJob.id`, `s` and 16 hex): every
    version that scan gives the room shares it."""
    return f"scan-{job_id}"


class SplatSource:
    """Finds and reads bundles. `folder_for(name)` maps a room name (or None: no name could be
    read from the version) to its bundle folder, or None when that room has none."""

    def __init__(self, folder_for: Callable[[str | None], Path | None], *,
                 poll_seconds: float = POLL_SECONDS) -> None:
        self.folder_for = folder_for
        self.poll_seconds = poll_seconds
        self._cache: dict[Path, tuple[float, dict | None]] = {}

    @classmethod
    def fixed(cls, folder: Path, *, name: str | None = None) -> "SplatSource":
        """One folder (the local `--splat DIR`): for the room called `name` only, when given,
        so a room scanned during the call is not drawn as the served one; else for any."""
        folder = Path(folder)
        if name is None:
            return cls(lambda _name: folder)
        return cls(lambda asked: folder if asked == name else None)

    @classmethod
    def per_room(cls, root: Path, *, poll_seconds: float = 10.0) -> "SplatSource":
        """`<root>/<room name>/` per room: the hosted Volume's ``splats/`` folder."""
        root = Path(root)

        def folder_for(name: str | None) -> Path | None:
            return root / name if name and NAME.match(name) else None
        return cls(folder_for, poll_seconds=poll_seconds)

    @classmethod
    def chain(cls, *sources: "SplatSource | None") -> "SplatSource | None":
        """The first of `sources` that has a bundle for a room (a room scanned during the call
        has its own; the deployed room's comes from the Volume). None if no source is given."""
        given = [s for s in sources if s is not None]
        if not given:
            return None

        def folder_for(name: str | None) -> Path | None:
            fallback = None
            for source in given:
                folder = source.folder_for(name)
                if folder is not None:
                    if (folder / BUNDLE).is_file():
                        return folder
                    fallback = fallback or folder
            return fallback
        return cls(folder_for, poll_seconds=min(s.poll_seconds for s in given))

    def _folder(self, name: str | None) -> Path | None:
        return None if name == NO_BUNDLE else self.folder_for(name)

    def bundle(self, name: str | None) -> dict | None:
        """The raw ``splat.json`` of `name`'s room (re-read only when its mtime changes)."""
        folder = self._folder(name)
        if folder is None:
            return None
        path = folder / BUNDLE
        try:
            mtime = path.stat().st_mtime
        except OSError:
            self._cache.pop(path, None)
            return None
        hit = self._cache.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = None
        if not _valid(raw):
            raw = None
        self._cache[path] = (mtime, raw)
        return raw

    def public(self, name: str | None, url_name: str) -> dict | None:
        """The bundle as browsers get it: files as URLs under ``splat/<url_name>/``."""
        raw = self.bundle(name)
        if raw is None:
            return None
        base = f"splat/{url_name}/"
        out = {"name": url_name, "stage": raw.get("stage"), "step": raw.get("step"),
               "created": raw.get("created"), "room": base + raw["room"],
               "bodies": {body: base + file for body, file in raw["bodies"].items()}}
        view = _view(raw.get("view"))
        if view is not None:
            out["view"] = view
        capture = _capture(raw.get("capture"))
        if capture is not None:
            out["capture"] = capture
        grabbable = _grabbable(raw.get("grabbable"), raw["bodies"])
        if grabbable:
            out["grabbable"] = grabbable
        return out

    def file(self, name: str | None, file: str) -> bytes | None:
        """One SPZ file of `name`'s room, if the name is a bundle file name at all."""
        if not FILE.match(file or ""):
            return None
        folder = self._folder(name)
        if folder is None:
            return None
        path = folder / file
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                return None
            return path.read_bytes()
        except OSError:
            return None

    def stamp(self, name: str | None) -> float | None:
        """What changes when `name`'s bundle does (its ``created``)."""
        raw = self.bundle(name)
        return None if raw is None else raw.get("created")


def _view(raw) -> dict | None:
    """A bundle's first view (where a newcomer stands: the scan's best camera spot) if it is
    well formed: ``{"pos": [x, y, z], "yaw": rad, "pitch": rad}``, all finite."""
    if not isinstance(raw, dict):
        return None
    pos, yaw, pitch = raw.get("pos"), raw.get("yaw"), raw.get("pitch")
    nums = [*(pos if isinstance(pos, list) else []), yaw, pitch]
    if not (isinstance(pos, list) and len(pos) == 3 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in nums)):
        return None
    return {"pos": [float(v) for v in pos], "yaw": float(yaw), "pitch": float(pitch)}


def _grabbable(raw, bodies: dict) -> dict[str, bool] | None:
    """Which of the bundle's bodies may be grabbed while their splat is shown (the cut's move
    test, `pipeline/splatroom/instances`): ``{body: bool}`` for bodies the bundle has."""
    if not isinstance(raw, dict):
        return None
    return {k: v for k, v in raw.items() if k in bodies and isinstance(v, bool)} or None


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _capture(raw) -> list[list[float]] | None:
    """A bundle's capture spots (the scan's camera positions, room frame, Z up) if they are well
    formed: a non-empty list of at most `MAX_CAPTURE` ``[x, y, z]``, all finite. One bad entry
    drops the field: a bundle that malformed is not one to trust spot by spot."""
    if not isinstance(raw, list) or not 0 < len(raw) <= MAX_CAPTURE:
        return None
    if not all(isinstance(p, list) and len(p) == 3 and all(_number(v) for v in p) for p in raw):
        return None
    return [[float(v) for v in p] for p in raw]


def _valid(raw) -> bool:
    if not isinstance(raw, dict) or not isinstance(raw.get("room"), str):
        return False
    bodies = raw.get("bodies")
    if not isinstance(bodies, dict) or not FILE.match(raw["room"]):
        return False
    return all(isinstance(k, str) and len(k) <= 200 and isinstance(v, str) and FILE.match(v)
               for k, v in bodies.items())
