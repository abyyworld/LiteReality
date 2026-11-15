"""The room call as a permanent public link: the server hosted on Modal, the scene on a Volume.

Run locally, the room call needs a tunnel before anyone else on the call can reach it, and the
link dies with the laptop. Hosted, it is one https address that lasts: whoever is on the Zoom, Meet
or Teams call clicks it in the chat and is in the room — webcam allowed, because it is https.

    uv run litereality roomcall run/tea_room --deploy     # upload the scene, deploy, print the link

    uv run litereality roomcall run/tea_room --publish    # a new VERSION, live, no redeploy

What runs where:
- Every upload is a new VERSION of the room: scene.xml plus the meshes and textures it names, in a
  folder of its own on the Modal Volume `SCENES_VOLUME` named after the room and its content
  (`tea_room-1a2b3c4d5e6f`, see `version_name`), next to the room's geometry (scene.glb + scene.json)
  built here on the laptop, so the container never has to. The file `CURRENT` at the Volume's root
  names the version to serve. It is written in a second batch, after the folder's batch has
  committed, so a container can never read a CURRENT that names a half-uploaded folder. Older
  versions of the same room are pruned, keeping the newest two.
- `deploy/modal/roomcall/app.py` serves `build_app` from ONE CPU container (`max_containers=1`) that
  takes every connection concurrently — rooms live in that process's memory, so a second container
  would split one call's people into two rooms that cannot see each other. Each WebSocket is one
  Modal input that lasts as long as the function's timeout allows, and the container scales to
  zero once the last person has left and the idle window has passed.
- While it runs, the container re-reads the Volume every `POLL_SECONDS` (`VolumeWatch`): a CURRENT
  naming a new version is loaded and swapped into every room live, with what people did carried
  across (`server.RoomHub.offer`, `sim.RoomSim.adopt`). So `--publish` changes the room during a
  call in seconds, without a `modal deploy` and without anyone leaving; `--deploy` (publish AND
  deploy) is for the first time and for code changes.
- Beside the default room CURRENT names, the Volume keeps a LIBRARY of ready rooms anyone in a
  call can move their call room into (`library`, the Rooms panel): `/.library/rooms.json` lists
  each one's version folder, `/.library/<name>.jpg` its thumbnail. `--library` adds a scene there
  (`add_to_library`) without making it current; a library room's photoreal bundle is the same
  /splats/<name>/ that `litereality splatroom --label <name> --publish` uploads. Pruning never
  removes a version the library lists.

    uv run litereality roomcall run/office_room/objects/scene_splat.xml --library --thumb a.jpg

- A scan sent into a room during the call (POST /scan, `scans`) is unpacked and built in the
  container itself, by the CLI in child processes: the quick room, then realistic objects from
  TRELLIS, called from inside the container with the workspace's own (ambient) credentials. The
  upload and the builds live on the container's own disk (`SCAN_SCRATCH`), never the Volume; each
  version a room gets is also kept on the Volume under `ROOMS_DIR` (`scans.DirRoomStore`), so a
  container restart mid-call puts that room back on its scan. Realistic builds are capped per UTC
  day in a Modal Dict (`SCAN_BUDGET_DICT`), like the GPU bodies.

This module is what both halves share: the upload (run on the laptop) and the app factory (run in
the container). It imports MuJoCo, FastAPI and Modal only inside the functions that need them, so
the deploy file can read its constants without either.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import xml.etree.ElementTree as ET
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)

APP_NAME = "litereality-roomcall"
FUNCTION_NAME = "web"
SCENES_VOLUME = "litereality-roomcall-scenes"
SCENES_MOUNT = "/scenes"
WEB_IN_CONTAINER = "/roomcall-web"
CURRENT = "current"
DEMO = "demo"
DEPLOY_FILE = "deploy/modal/roomcall/app.py"

# What the container installs: the distributions the server's imports resolve to, at the versions
# this checkout's lock resolves for Python 3.11 (the image's Python). Pinned rather than open so a
# redeploy months later serves the physics these tests ran against. `tests/test_roomcall_hosted.py`
# checks that every third-party module the hosted app imports is covered by this list or by what
# these packages themselves require, so an import added to the server cannot silently break only
# the hosted copy.
#
# The second group is what the container's child builds need (`scans`: `litereality quickroom`
# and `realroom` on a scan sent into a room): the CLI's settings, and OpenCV for the object
# cut-outs. Checked by building a clean venv with exactly these packages and running both
# commands on the tea_room scan. OpenCV's wheel links the system's libGL and GLib, which the slim
# image lacks: `IMAGE_APT` adds them.
IMAGE_PACKAGES = (
    "mujoco==3.12.0",
    "numpy==2.4.6",
    "pillow==12.2.0",
    "trimesh==4.12.2",
    "scipy==1.17.1",
    "networkx==3.6.1",
    "python-fcl==0.7.0.11",
    "fastapi==0.138.2",
    "pydantic==2.13.4",
    # the child builds
    "pydantic-settings==2.14.2",
    "python-dotenv==1.2.2",
    "opencv-python==4.13.0.92",
)
IMAGE_APT = ("libgl1", "libglib2.0-0")
# Imports the hosted app and its child builds never reach in the container, with why — every
# other third-party import must be covered by `IMAGE_PACKAGES` (tests/test_roomcall_hosted.py).
NOT_IN_IMAGE = {
    "modal": "the Modal runtime provides its own client in every container",
    "uvicorn": "`server.serve` only; Modal serves the ASGI app itself",
    "open3d": "`realroom.fit.decimate`, only for a mesh over 150k faces; TRELLIS's passes ask for "
              "20k and 50k",
    "coacd": "the full pipeline's convex decomposition (`sim.properties`), not the quick room",
}
# Where the container unpacks and builds scans sent into rooms: its own disk, not the Volume.
SCAN_SCRATCH = "/tmp/roomcall-scans"
# The Volume folder keeping each room's scanned version across a container restart. A dot name:
# never a version folder (`_VERSION`), never pruned by a publish.
ROOMS_DIR = ".rooms"
SCAN_BUDGET_DICT = "litereality-roomcall-scan-budget"

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# A version folder: the room's name (cut to leave room for the suffix inside `_NAME`'s 64) and the
# first 12 hex of its content digest.
NAME_CHARS = 48
DIGEST_CHARS = 12
_VERSION = re.compile(rf"^(?P<name>[A-Za-z0-9_-]{{1,{NAME_CHARS}}})-(?P<digest>[0-9a-f]{{{DIGEST_CHARS}}})$")
# Versions of one room kept on the Volume: the one just published and the one before it (which a
# running container may still be serving until its rooms have swapped).
KEEP_VERSIONS = 2
# How often the container looks at the Volume for a new version (seconds). Each look is one
# `Volume.reload()`, cheap, and a published version is live within this plus the prefetch window.
POLL_SECONDS = 10.0
# Elements whose `file` MuJoCo resolves against the compiler's meshdir (else assetdir, else the
# model's folder), against its texturedir, or against the including file's folder.
_MESH_LIKE = {"mesh", "hfield", "skin"}


def scene_name(label: str) -> str:
    """A folder name for the Volume: the label with anything unsafe collapsed to '-'."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")[:64]
    return name if name and name != DEMO else "room"


def version_name(label: str, digest: str) -> str:
    """The folder (and wire id) of one version of a room: `tea_room-1a2b3c4d5e6f`. Content-
    addressed, so publishing the same files twice is the same version, and any change — one mesh,
    one joint range, or (`publish`) the label or stage it is shown with — is a new one."""
    return f"{scene_name(label)[:NAME_CHARS]}-{digest[:DIGEST_CHARS]}"


def files_digest(files: list[tuple[Path, PurePosixPath]], *, also: tuple[str, ...] = ()) -> str:
    """sha256 over every scene file's relative path and bytes, in path order. The path counts, so
    renaming a mesh is a change; each part is length-prefixed, so no two layouts hash alike.
    `also` is hashed first: what a version is beside its files (`publish` passes its label and
    stage, so the same files published as another stage are a new version the container picks
    up, not the one it already serves)."""
    digest = hashlib.sha256()
    for text in also:
        data = text.encode("utf-8")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    for local, relative in sorted(files, key=lambda item: str(item[1])):
        name = str(relative).encode("utf-8")
        digest.update(len(name).to_bytes(8, "little"))
        digest.update(name)
        digest.update(Path(local).stat().st_size.to_bytes(8, "little"))
        with open(local, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def scene_digest(scene_xml: Path) -> str:
    """`files_digest` of everything the scene needs (`scene_files`)."""
    return files_digest(scene_files(Path(scene_xml)))


def scene_files(scene_xml: Path) -> list[tuple[Path, PurePosixPath]]:
    """Every file the scene needs, as (local file, path relative to the scene's folder).

    Read from the XML itself rather than by copying the scene's folder, because the exporter's
    folder also holds what is not the scene — the shake video, reports, an earlier export's meshes —
    and a folder copy would push all of it to the Volume. The root file is always sent as
    `scene.xml`; everything it names keeps its path relative to the scene's folder, which is what
    MuJoCo resolves `meshdir`, `texturedir` and `include` against.
    """
    scene_xml = Path(scene_xml)
    root_dir = scene_xml.parent
    found: dict[Path, PurePosixPath] = {scene_xml.resolve(): PurePosixPath("scene.xml")}
    missing: list[str] = []

    def walk(xml_path: Path) -> None:
        tree = ET.parse(xml_path)
        compiler = tree.getroot().find("compiler")
        asset_dir = (compiler.get("assetdir") if compiler is not None else None) or ""
        mesh_dir = (compiler.get("meshdir") if compiler is not None else None) or asset_dir
        tex_dir = (compiler.get("texturedir") if compiler is not None else None) or asset_dir
        for element in tree.iter():
            file = element.get("file")
            if not file:
                continue
            if element.tag == "include":
                local = xml_path.parent / file
            elif element.tag in _MESH_LIKE:
                local = root_dir / mesh_dir / file
            elif element.tag == "texture":
                local = root_dir / tex_dir / file
            else:
                continue
            if not local.is_file():
                missing.append(str(local))
                continue
            resolved = local.resolve()
            if resolved in found:
                continue
            try:
                relative = PurePosixPath(local.relative_to(root_dir).as_posix())
            except ValueError:
                # Outside the scene's folder: there is no relative path that still resolves once
                # uploaded, so name it rather than send a scene that cannot compile.
                missing.append(f"{local} (outside {root_dir})")
                continue
            found[resolved] = relative
            if element.tag == "include":
                walk(local)

    walk(scene_xml)
    if missing:
        raise ValueError("the scene names files that are not there:\n  " + "\n  ".join(missing))
    return [(Path(local), relative) for local, relative in found.items()]


@dataclass(frozen=True)
class Published:
    """What an upload did: the version folder it sent (the one CURRENT now names, after
    `publish`), and the bytes sent."""

    version: str
    sent: int


# Files a version folder holds beside the scene's own (see `server.prebuilt_files`).
_PREBUILT = ("scene.glb", "scene.json", "roomcall.json")


def upload_version(scene_xml: Path, *, name: str, volume, label: str | None = None,
                   stage: str = "scene", prebuild: bool = True) -> Published:
    """Upload a scene into `volume` as a new version folder of the room `name`, in one batch: the
    scene's files plus, with `prebuild`, its geometry built here (`server.prebuilt_files`; it
    also proves the scene compiles). Nothing else: CURRENT is not touched, so a container keeps
    serving what it serves (`publish` makes it current; `add_to_library` lists it as a ready
    room). The batch commits as a whole when it returns, so whatever names the folder afterwards
    always finds all of it."""
    scene_xml = Path(scene_xml)
    files = scene_files(scene_xml)
    clash = sorted(str(rel) for _local, rel in files if str(rel) in _PREBUILT)
    if clash:
        raise ValueError(f"the scene names files the room call keeps beside it: {', '.join(clash)}")
    shown = _shown(name, label)
    folder = version_name(name, files_digest(files, also=(shown, stage)))
    if prebuild:
        from litereality_agent.room_ops.roomcall.server import prebuilt_files

        extra = prebuilt_files(scene_xml, version=folder, label=shown, stage=stage)
    else:
        meta = {"version": folder, "label": shown, "stage": stage}
        extra = {"roomcall.json": json.dumps(meta, indent=1).encode("utf-8")}
    sent = 0
    with volume.batch_upload(force=True) as batch:
        for local, relative in files:
            batch.put_file(local, f"/{folder}/{relative}")
            sent += local.stat().st_size
        for file_name, data in extra.items():
            batch.put_file(io.BytesIO(data), f"/{folder}/{file_name}")
            sent += len(data)
    return Published(folder, sent)


def _shown(name: str, label: str | None) -> str:
    """The label a version is shown with: the given one, else the room's name, words spaced."""
    return label.replace("_", " ") if label else name.replace("_", " ").replace("-", " ")


def publish(scene_xml: Path | None, *, name: str, volume, label: str | None = None,
            stage: str = "scene", prebuild: bool = True) -> Published:
    """Upload a scene into `volume` as a new version of the room `name` and make it the one served.

    Three steps, in this order, each finished before the next starts:
    1. one batch with the version's folder (`upload_version`);
    2. a second batch with `CURRENT` naming that folder: a batch commits as a whole when it
       returns, so a container that sees the new CURRENT always finds the whole folder;
    3. pruning older versions of `name`, keeping `KEEP_VERSIONS` (this one and the one served
       before it, which a running container may still have rooms on), whatever CURRENT names by
       then, any folder uploaded since the one served before (another publish of the room
       may be about to name it: `_prune`), and any the room library lists. A prune that fails
       is reported, never fatal: the version is already live.

    `scene_xml=None` switches the link back to the demo room without uploading anything. `volume`
    is a `modal.Volume`; anything with its `batch_upload`, `read_file`, `listdir` and
    `remove_file` works, which is how the tests run offline.
    """
    if scene_xml is None:
        with volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(DEMO.encode()), f"/{CURRENT}")
        return Published(DEMO, 0)
    published = upload_version(scene_xml, name=name, volume=volume, label=label, stage=stage,
                               prebuild=prebuild)
    folder = published.version
    previous = _read_current(volume)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(folder.encode()), f"/{CURRENT}")
    try:
        # CURRENT read again: another publish of this room (the quick stage auto-published while
        # the full one is published by hand) may have named its own version since ours.
        _prune(volume, name, keep={folder, previous, _read_current(volume)},
               since={folder, previous})
    except Exception as exc:  # noqa: BLE001 — the new version is live either way
        print(f"  (could not prune older versions of {name}: {exc})", file=sys.stderr)
    return published


def _read_current(volume) -> str | None:
    try:
        return b"".join(volume.read_file(f"/{CURRENT}")).decode("utf-8").strip() or None
    except Exception:  # noqa: BLE001 — none yet, or unreadable: nothing extra to keep
        return None


def _prune(volume, name: str, *, keep: set[str | None], since: set[str | None] | None = None
           ) -> list[str]:
    """Remove all but the newest `KEEP_VERSIONS` version folders of `name`, never one in `keep`
    and never one at least as new as the oldest folder in `since` (`publish`: its own and the
    one CURRENT named before it). A folder uploaded since then may be another publish's that has
    not written CURRENT yet; removing it would leave CURRENT naming nothing. Never one the room
    library lists either (`add_to_library`): people pick those in calls, whatever the newest
    is. Newest by the listing's mtime; the folders in `keep` count first. Returns what was
    removed. A library file that cannot be read raises: nothing is removed then, since what it
    protects is unknown."""
    prefix = scene_name(name)[:NAME_CHARS]
    keep = set(keep) | {room.get("version") for room in _read_library(volume)}
    entries = []
    for entry in volume.listdir("/"):
        folder = PurePosixPath(str(entry.path)).name
        match = _VERSION.match(folder)
        if match and match["name"] == prefix:
            entries.append((int(getattr(entry, "mtime", 0) or 0), folder))
    floor = min((mtime for mtime, folder in entries if folder in (since or ())), default=None)
    kept = {folder for mtime, folder in entries
            if folder in keep or (floor is not None and mtime >= floor)}
    removed = []
    for _mtime, folder in sorted(entries, reverse=True):
        if folder in kept:
            continue
        if len(kept) < KEEP_VERSIONS:
            kept.add(folder)
            continue
        volume.remove_file(f"/{folder}", recursive=True)
        removed.append(folder)
    return removed


# The thumbnail a library room is shown with in the room picker: a JPEG, small enough that a
# list of them loads at once on a phone.
MAX_THUMB_BYTES = 1 << 20


def _library_paths(name: str | None = None) -> tuple[str, str]:
    """The Volume paths of the library file and of `name`'s thumbnail."""
    from litereality_agent.room_ops.roomcall.library import LIBRARY_DIR, LIBRARY_FILE

    return f"/{LIBRARY_DIR}/{LIBRARY_FILE}", f"/{LIBRARY_DIR}/{name}.jpg"


def _read_library(volume) -> list[dict]:
    """The entries of the Volume's room library (`library.RoomLibrary.folder` reads the same
    file in the container); none when there is no library yet. A file that is there but cannot
    be read or parsed raises ValueError rather than reading as empty: whoever writes it next
    would otherwise drop every room it lists, and a prune would remove their versions."""
    path, _ = _library_paths()
    try:
        data = b"".join(volume.read_file(path))
    except FileNotFoundError:
        return []
    try:
        rooms = json.loads(data.decode("utf-8"))["rooms"]
        if not isinstance(rooms, list):
            raise TypeError("'rooms' is not a list")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"the room library {path} on the Volume cannot be read ({exc}); fix "
                         f"or remove it (`modal volume rm {SCENES_VOLUME} {path}`)") from None
    return [room for room in rooms if isinstance(room, dict)]


def _write_library(batch, rooms: list[dict]) -> None:
    path, _ = _library_paths()
    data = json.dumps({"rooms": rooms}, indent=1).encode("utf-8")
    batch.put_file(io.BytesIO(data), path)


def add_to_library(scene_xml: Path, *, name: str, volume, label: str | None = None,
                   stage: str = "scene", thumb: Path | None = None,
                   prebuild: bool = True) -> Published:
    """Upload a scene as a version of the room `name` and list it in the Volume's room library,
    which people in a call pick rooms from (`library`, the Rooms panel) — without making it the
    link's default room: CURRENT is never touched.

    `name` is the room's folder name (`scene_name`), at most `NAME_CHARS`, so the version it
    gets (`name-<12 hex>`) maps back to it (`splats.room_name`) and the room's photoreal bundle
    on the Volume (/splats/<name>/, `litereality splatroom --label <name> --publish`) is found
    with no further wiring. Adding a name the library already lists replaces that entry and
    keeps its thumbnail unless `thumb` (a JPEG, at most `MAX_THUMB_BYTES`) brings a new one.

    Order: the version folder in one batch (`upload_version`), then the library file and the
    thumbnail together in a second, so a container never lists a room whose folder is still
    arriving. The library file is read again just before it is written, but two adds at the
    very same moment can still lose one entry (the Volume has no compare-and-swap). Older
    versions of `name` are then pruned as `publish` does; the library's own never are.
    """
    from litereality_agent.room_ops.roomcall.library import MAX_ROOMS, NAME, SCAN

    if not NAME.match(name or "") or scene_name(name) != name or name == SCAN:
        raise ValueError(f"a library room's name is letters, digits, '_' and '-', at most "
                         f"{NAME_CHARS} (and not '{DEMO}' or '{SCAN}'): {name!r}")
    picture = None
    if thumb is not None:
        try:   # a mistyped path is a plain message too, like every other bad thumbnail
            picture = Path(thumb).read_bytes()
        except OSError as exc:
            raise ValueError(f"the thumbnail cannot be read: {thumb} "
                             f"({exc.strerror or exc})") from None
        if len(picture) > MAX_THUMB_BYTES or not picture.startswith(b"\xff\xd8\xff"):
            raise ValueError(f"the thumbnail must be a JPEG of at most "
                             f"{MAX_THUMB_BYTES // 1024} KB: {thumb}")
    # A broken library refuses before anything is uploaded; so does a full one (the container
    # reads no more than MAX_ROOMS, so another would be written but never listed).
    listed = _read_library(volume)
    if len(listed) >= MAX_ROOMS and not any(room.get("name") == name for room in listed):
        raise ValueError(f"the room library is full ({MAX_ROOMS} rooms): take one out first "
                         "(`litereality roomcall --library-remove NAME`)")
    published = upload_version(scene_xml, name=name, volume=volume, label=label, stage=stage,
                               prebuild=prebuild)
    rooms = _read_library(volume)
    before = next((room for room in rooms if room.get("name") == name), None)
    entry = {"name": name, "label": _shown(name, label)[:80], "version": published.version,
             "thumb": picture is not None or bool(before and before.get("thumb") is True)}
    if before is None:
        rooms.append(entry)
    else:
        rooms = [entry if room is before else room for room in rooms
                 if room is before or room.get("name") != name]
    _, thumb_path = _library_paths(name)
    with volume.batch_upload(force=True) as batch:
        _write_library(batch, rooms)
        if picture is not None:
            batch.put_file(io.BytesIO(picture), thumb_path)
    try:
        _prune(volume, name, keep={published.version, before and before.get("version"),
                                   _read_current(volume)},
               since={published.version})
    except Exception as exc:  # noqa: BLE001 — the room is in the library either way
        print(f"  (could not prune older versions of {name}: {exc})", file=sys.stderr)
    return published


def remove_from_library(name: str, *, volume) -> bool:
    """Take the room `name` out of the Volume's room library (and its thumbnail); False when the
    library does not list it. Its version folder stays: a call room may be on it right now, and
    with the entry gone the next publish or add of that room prunes it as any other."""
    from litereality_agent.room_ops.roomcall.library import NAME

    if not NAME.match(name or ""):
        return False
    rooms = _read_library(volume)
    kept = [room for room in rooms if room.get("name") != name]
    if len(kept) == len(rooms):
        return False
    with volume.batch_upload(force=True) as batch:
        _write_library(batch, kept)
    _, thumb_path = _library_paths(name)
    with suppress(FileNotFoundError):
        volume.remove_file(thumb_path)
    return True


def served_scene(root: Path, *, strict: bool = False) -> tuple[Path, str] | None:
    """The scene `CURRENT` names under the mounted Volume (its scene.xml and folder name), or None
    for the demo room. Works for version folders and for the un-versioned folders of uploads
    made before versions existed alike.

    A CURRENT that names nothing servable — a folder that is not there (pruned by an overlapping
    publish), an unreadable file, a name that is not one — is the demo room too, unless `strict`:
    then only a CURRENT that is absent or says demo is, and anything else raises ValueError. A
    container starting up has nothing better to serve than the demo; a running one (`VolumeWatch`)
    does — what its rooms are on — and must not swap every room into the demo over a bad CURRENT.
    """
    marker = Path(root) / CURRENT
    try:
        name = marker.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        if strict:
            raise ValueError(f"CURRENT on the Volume cannot be read: {exc}") from None
        return None
    if name == DEMO:
        return None
    scene = Path(root) / name / "scene.xml"
    if _NAME.match(name) and scene.is_file():
        return scene, name
    if strict:
        raise ValueError(f"CURRENT names {name[:80]!r}, which is not a scene on the Volume; "
                         "still serving what the rooms are on")
    return None


def served_ref(root: Path, *, strict: bool = False):
    """`served_scene` as a `server.SceneRef`: its id is the folder name; label and stage come from
    the folder's roomcall.json (an older upload without one: from the folder's name); and the
    folder is offered as prebuilt geometry, used if it fits (`server.read_prebuilt`)."""
    from litereality_agent.room_ops.roomcall import STAGES
    from litereality_agent.room_ops.roomcall.server import SceneRef

    current = served_scene(root, strict=strict)
    if current is None:
        return None
    scene, name = current
    try:
        meta = json.loads((scene.parent / "roomcall.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    meta = meta if isinstance(meta, dict) else {}
    match = _VERSION.match(name)
    fallback = (match["name"] if match else name).replace("_", " ").replace("-", " ")
    label = meta.get("label") if isinstance(meta.get("label"), str) and meta["label"] else fallback
    stage = meta.get("stage") if meta.get("stage") in STAGES else "scene"
    return SceneRef(scene=scene, id=name, label=label[:80], stage=stage, prebuilt=scene.parent)


class VolumeWatch:
    """The hosted app's source of new versions (`server.RoomHub.ensure_watching` polls it).

    Every `poll_seconds`: `reload()` the Volume on a worker thread (a mounted Volume shows only
    what was committed when the container started, until it is reloaded), then read what CURRENT
    names now. A reload refused because files are open — a version being read at that very
    moment — skips the round. CURRENT naming the demo (or absent) serves the demo room again; one
    naming a folder that is not there raises (`served_scene` strict), so the round is reported on
    /healthz and every room keeps the version it has.
    """

    def __init__(self, root: Path, *, reload=None, poll_seconds: float = POLL_SECONDS,
                 scratch: Path | None = None) -> None:
        self.root = Path(root)
        self.reload = reload
        self.poll_seconds = poll_seconds
        self._scratch = Path(scratch) if scratch else None
        self._demo = None

    async def check(self):
        if self.reload is not None:
            try:
                await asyncio.to_thread(self.reload)
            except RuntimeError as exc:
                if "open files" in str(exc):
                    log.info("roomcall: Volume busy (open files); looking again next round")
                    return None
                raise
        ref = await asyncio.to_thread(served_ref, self.root, strict=True)
        return ref if ref is not None else await asyncio.to_thread(self.demo)

    def demo(self):
        """The demo room as a version, written once on the container's own disk — never on the
        Volume, where it would race a laptop uploading a real scene."""
        if self._demo is None:
            import tempfile

            from litereality_agent.room_ops.roomcall.demo import write_demo_scene
            from litereality_agent.room_ops.roomcall.server import SceneRef

            folder = self._scratch or Path(tempfile.mkdtemp(prefix="roomcall-demo-"))
            folder.mkdir(parents=True, exist_ok=True)
            scene = write_demo_scene(folder)
            self._demo = SceneRef(scene=scene, id=version_name("demo room", scene_digest(scene)),
                                  label="demo room", stage="demo")
        return self._demo


def build_app(root: Path, *, web_dir: Path | None = None, scratch: Path | None = None,
              reload=None, poll_seconds: float = POLL_SECONDS, freeze: bool = False,
              avatars: str = "modal", uploads: str = "realistic", commit=None,
              scan_scratch: Path | None = None, keep_rooms: bool = True,
              realistic_per_day: int | None = None, scan_budget=None):
    """The room call app the container serves: the Volume's current version, else the demo room,
    watching the Volume (`VolumeWatch`; `reload` is the Volume's own `reload`) for the next one.
    `freeze` (the container: the app is the process) exempts startup objects from the garbage
    collector (`server.freeze_startup_objects`).

    Full bodies are built on the GPU function in the container's own Modal environment, at most
    `avatars.GPU_DAILY_CAP` a day across restarts (`ModalDictBudget`), the stand-in otherwise.
    Nothing here touches Modal: the budget and the function are looked up by the first build.

    Scans sent into a room (`uploads`, see `scans`) are built under `scan_scratch` (default
    `SCAN_SCRATCH`), at most `realistic_per_day` (default `scans.REALISTIC_PER_DAY`) realistic
    builds a UTC day across restarts (`scan_budget`, default a Modal Dict). With `keep_rooms`
    each room's scanned version is also kept under `ROOMS_DIR` on the Volume (`commit` is the
    Volume's own, run under one lock with `reload`)."""
    import threading

    from litereality_agent.room_ops.roomcall.avatars import (
        GPU_DAILY_CAP,
        AvatarConfig,
        ModalDictBudget,
    )
    from litereality_agent.room_ops.roomcall.scans import (
        REALISTIC_PER_DAY,
        DirRoomStore,
        ScanConfig,
    )
    from litereality_agent.room_ops.roomcall.server import create_app, freeze_startup_objects

    if avatars == "modal":
        bodies = AvatarConfig(mode="modal", environment=None,
                              budget=ModalDictBudget(GPU_DAILY_CAP))
    else:
        bodies = AvatarConfig(mode=avatars)
    # One lock for the Volume: a reload refuses to run while files are open, and the room store
    # writes files there.
    volume_lock = threading.Lock()
    if reload is not None:
        volume_reload = reload

        def reload():
            with volume_lock:
                volume_reload()
    store = None
    if keep_rooms and uploads != "off":
        store = DirRoomStore(Path(root) / ROOMS_DIR, commit=commit, lock=volume_lock)
        with suppress(OSError):
            store.sweep()
    cap = REALISTIC_PER_DAY if realistic_per_day is None else realistic_per_day
    budget = scan_budget if scan_budget is not None else ModalDictBudget(cap, name=SCAN_BUDGET_DICT)
    scratch_root = Path(scan_scratch or SCAN_SCRATCH)
    scans = ScanConfig(mode=uploads, realistic_per_day=cap, budget=budget, store=store,
                       scratch=scratch_root, splats_root=scratch_root / "splats")
    watch = VolumeWatch(root, reload=reload, poll_seconds=poll_seconds, scratch=scratch)
    ref = served_ref(root) or watch.demo()
    from litereality_agent.room_ops.roomcall.splats import SPLATS_DIR, SplatSource

    # The photoreal rooms (`litereality splatroom --publish`): read after the version watcher's
    # Volume reload, so a snapshot is live within POLL_SECONDS of its upload.
    # A room scanned during the call has its own bundle on the container's disk (`scans`).
    splats = SplatSource.chain(SplatSource.per_room(scratch_root / "splats", poll_seconds=2.0),
                               SplatSource.per_room(Path(root) / SPLATS_DIR,
                                                    poll_seconds=poll_seconds))
    from litereality_agent.room_ops.roomcall.library import RoomLibrary

    # The ready rooms people pick in a call (`add_to_library`): their list is read again once
    # the version watcher's Volume reload shows it changed.
    library = RoomLibrary.folder(Path(root), poll_seconds=poll_seconds)
    app = create_app(ref.scene, label=ref.label, stage=ref.stage, version=ref.id,
                     prebuilt=ref.prebuilt, web_dir=web_dir, watch=watch, avatars=bodies,
                     scans=scans, splats=splats, library=library)
    if freeze:
        freeze_startup_objects()
    return app


def room_link(url: str) -> str:
    """A link into a fresh room: a public address needs a room id strangers will not guess, and
    `lobby` — the default — is the first one anyone would try."""
    return f"{url.rstrip('/')}/?room={secrets.token_urlsafe(6)}"


@contextmanager
def _modal_credentials(credentials: tuple[str, str] | None, profile: str | None):
    """The Modal client in THIS process reads its token from the environment when it first
    connects; set it for the upload and the URL lookup, then put the environment back."""
    keys = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_PROFILE")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        if credentials:
            os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"] = credentials
        if profile:
            os.environ["MODAL_PROFILE"] = profile
        yield {**os.environ}
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def deploy(scene_xml: Path | None, *, label: str, repo_root: Path, environment: str = "main",
           credentials: tuple[str, str] | None = None, profile: str | None = None,
           stage: str = "scene", redeploy: bool = True) -> int:
    """Upload the scene as a new version (or select the demo), deploy the app, print the link.

    `redeploy=False` is `--publish`: the version only, no `modal deploy` — seconds instead of
    minutes, and the running container swaps it into every room without anyone leaving. Building
    the version's geometry here compiles the scene first: one that does not load locally would
    not load in the container either, and finding out there means a link that only shows an
    error.
    """
    import modal

    app_file = Path(repo_root) / DEPLOY_FILE
    if redeploy and not app_file.is_file():
        raise SystemExit(f"{DEPLOY_FILE} not found under {repo_root}: deploy from a source checkout")
    name = scene_name(label)
    with _modal_credentials(credentials, profile) as env:
        volume = modal.Volume.from_name(SCENES_VOLUME, environment_name=environment,
                                        create_if_missing=True)
        if scene_xml is not None:
            print(f"→ Building {label} ({stage}) and uploading it to the Modal Volume "
                  f"{SCENES_VOLUME}")
            published = publish(Path(scene_xml), name=name, volume=volume, label=label,
                                stage=stage)
            print(f"  version {published.version}  {published.sent / 1e6:.1f} MB")
        else:
            print("→ Serving the demo room")
            published = publish(None, name=DEMO, volume=volume)
        if redeploy:
            print(f"→ Deploying {APP_NAME} ({DEPLOY_FILE})")
            result = subprocess.run(
                [sys.executable, "-m", "modal", "deploy", "--env", environment, str(app_file)],
                env=env, check=False, text=True,
            )
            if result.returncode != 0:
                print("deploying the room call failed; see the Modal output above.",
                      file=sys.stderr)
                return result.returncode
        try:
            url = modal.Function.from_name(APP_NAME, FUNCTION_NAME,
                                           environment_name=environment).get_web_url()
        except Exception:  # noqa: BLE001 — published, but there is no app to serve it yet
            if redeploy:
                raise
            print(f"\nuploaded {published.version}, but no room call is deployed in "
                  f"{environment!r} yet: run once with --deploy", file=sys.stderr)
            return 1
    if not redeploy:
        print(f"\nroom call updated  {url}")
        print(f"  everyone in it switches to {published.version} within ~{POLL_SECONDS:.0f} s "
              "plus a few to fetch it, without leaving; a link nobody is using picks it up on "
              "its next start")
        return 0
    print(f"\nroom call is live  {url}")
    print(f"  paste this in the call chat   {room_link(url)}")
    print("  everyone who opens it lands in the same room; the first visit after a quiet spell "
          "takes a few seconds while the server starts")
    print(f"  stop it   uv run modal app stop --env {environment} {APP_NAME}")
    return 0


def library_add(scene_xml: Path, *, label: str, environment: str = "main",
                credentials: tuple[str, str] | None = None, profile: str | None = None,
                stage: str = "scene", thumb: Path | None = None) -> int:
    """`roomcall SCENE --library`: build the scene here and add it to the hosted link's room
    library (`add_to_library`) — no `modal deploy`, and the link's default room stays what it
    is. A running container lists it within a poll or two."""
    import modal

    name = scene_name(label)
    if len(name) > NAME_CHARS:
        raise SystemExit(f"a library room's name is at most {NAME_CHARS} characters: give a "
                         f"shorter --label than {label!r}")
    with _modal_credentials(credentials, profile):
        volume = modal.Volume.from_name(SCENES_VOLUME, environment_name=environment,
                                        create_if_missing=True)
        print(f"→ Building {label} ({stage}) and adding it to the room library on the Modal "
              f"Volume {SCENES_VOLUME}")
        try:
            published = add_to_library(Path(scene_xml), name=name, volume=volume, label=label,
                                       stage=stage, thumb=thumb)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    print(f"  version {published.version}  {published.sent / 1e6:.1f} MB")
    print(f"\n{name} is in the room library: anyone in a call can pick it under Rooms within "
          f"~{2 * POLL_SECONDS:.0f} s; the link's default room is unchanged")
    print(f"  its photoreal room   uv run litereality splatroom <scan> --label {name} "
          "--publish --upload-only")
    print(f"  take it out again    uv run litereality roomcall --library-remove {name}")
    return 0


def library_remove(name: str, *, environment: str = "main",
                   credentials: tuple[str, str] | None = None, profile: str | None = None) -> int:
    """`roomcall --library-remove NAME`: take a room out of the hosted link's library."""
    import modal

    with _modal_credentials(credentials, profile):
        volume = modal.Volume.from_name(SCENES_VOLUME, environment_name=environment,
                                        create_if_missing=True)
        try:
            removed = remove_from_library(name, volume=volume)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    if not removed:
        print(f"the room library has no room named {name!r}", file=sys.stderr)
        return 1
    print(f"{name} is out of the room library; calls already in it stay until they pick "
          "another room")
    return 0
