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
from contextlib import contextmanager
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
)

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
    """What `publish` did: the version folder CURRENT now names, and the bytes sent."""

    version: str
    sent: int


# Files a version folder holds beside the scene's own (see `server.prebuilt_files`).
_PREBUILT = ("scene.glb", "scene.json", "roomcall.json")


def publish(scene_xml: Path | None, *, name: str, volume, label: str | None = None,
            stage: str = "scene", prebuild: bool = True) -> Published:
    """Upload a scene into `volume` as a new version of the room `name` and make it the one served.

    Three steps, in this order, each finished before the next starts:
    1. one batch with the version's folder — the scene's files plus, with `prebuild`, its geometry
       built here (`server.prebuilt_files`; it also proves the scene compiles);
    2. a second batch with `CURRENT` naming that folder: a batch commits as a whole when it
       returns, so a container that sees the new CURRENT always finds the whole folder;
    3. pruning older versions of `name`, keeping `KEEP_VERSIONS` (this one and the one served
       before it, which a running container may still have rooms on), whatever CURRENT names by
       then, and any folder uploaded since the one served before (another publish of the room
       may be about to name it: `_prune`). A prune that fails is reported, never fatal: the
       version is already live.

    `scene_xml=None` switches the link back to the demo room without uploading anything. `volume`
    is a `modal.Volume`; anything with its `batch_upload`, `read_file`, `listdir` and
    `remove_file` works, which is how the tests run offline.
    """
    if scene_xml is None:
        with volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO(DEMO.encode()), f"/{CURRENT}")
        return Published(DEMO, 0)
    scene_xml = Path(scene_xml)
    files = scene_files(scene_xml)
    clash = sorted(str(rel) for _local, rel in files if str(rel) in _PREBUILT)
    if clash:
        raise ValueError(f"the scene names files the room call keeps beside it: {', '.join(clash)}")
    shown = label.replace("_", " ") if label else name.replace("_", " ").replace("-", " ")
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
    return Published(folder, sent)


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
    not written CURRENT yet; removing it would leave CURRENT naming nothing. Newest by the
    listing's mtime; the folders in `keep` count first. Returns what was removed."""
    prefix = scene_name(name)[:NAME_CHARS]
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
              avatars: str = "modal"):
    """The room call app the container serves: the Volume's current version, else the demo room,
    watching the Volume (`VolumeWatch`; `reload` is the Volume's own `reload`) for the next one.
    `freeze` (the container: the app is the process) exempts startup objects from the garbage
    collector (`server.freeze_startup_objects`).

    Full bodies are built on the GPU function in the container's own Modal environment, at most
    `avatars.GPU_DAILY_CAP` a day across restarts (`ModalDictBudget`), the stand-in otherwise.
    Nothing here touches Modal: the budget and the function are looked up by the first build."""
    from litereality_agent.room_ops.roomcall.avatars import (
        GPU_DAILY_CAP,
        AvatarConfig,
        ModalDictBudget,
    )
    from litereality_agent.room_ops.roomcall.server import create_app, freeze_startup_objects

    if avatars == "modal":
        bodies = AvatarConfig(mode="modal", environment=None,
                              budget=ModalDictBudget(GPU_DAILY_CAP))
    else:
        bodies = AvatarConfig(mode=avatars)
    watch = VolumeWatch(root, reload=reload, poll_seconds=poll_seconds, scratch=scratch)
    ref = served_ref(root) or watch.demo()
    app = create_app(ref.scene, label=ref.label, stage=ref.stage, version=ref.id,
                     prebuilt=ref.prebuilt, web_dir=web_dir, watch=watch, avatars=bodies)
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
