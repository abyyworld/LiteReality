"""The room call as a permanent public link: the server hosted on Modal, the scene on a Volume.

Run locally, the room call needs a tunnel before anyone else on the call can reach it, and the
link dies with the laptop. Hosted, it is one https address that lasts: whoever is on the Zoom, Meet
or Teams call clicks it in the chat and is in the room — webcam allowed, because it is https.

    uv run litereality roomcall run/tea_room --deploy     # upload the scene, deploy, print the link

What runs where:
- The scene (scene.xml plus the meshes and textures it names) is uploaded once into the Modal
  Volume `SCENES_VOLUME`, under its own folder, and the file `CURRENT` at the Volume's root names the
  folder to serve. Uploading another scene and redeploying switches the link to it.
- `deploy/modal/roomcall/app.py` serves `build_app` from ONE CPU container (`max_containers=1`) that
  takes every connection concurrently — rooms live in that process's memory, so a second container
  would split one call's people into two rooms that cannot see each other. Each WebSocket is one
  Modal input that lasts as long as the function's timeout allows, and the container scales to
  zero once the last person has left and the idle window has passed.

This module is what both halves share: the upload (run on the laptop) and the app factory (run in
the container). It imports MuJoCo, FastAPI and Modal only inside the functions that need them, so
the deploy file can read its constants without either.
"""

from __future__ import annotations

import io
import os
import re
import secrets
import subprocess
import sys
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

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
# Elements whose `file` MuJoCo resolves against the compiler's meshdir (else assetdir, else the
# model's folder), against its texturedir, or against the including file's folder.
_MESH_LIKE = {"mesh", "hfield", "skin"}


def scene_name(label: str) -> str:
    """A folder name for the Volume: the label with anything unsafe collapsed to '-'."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")[:64]
    return name if name and name != DEMO else "room"


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


def publish(scene_xml: Path | None, *, name: str, volume) -> int:
    """Upload a scene into `volume` under `name` and make it the one served. Returns bytes sent.

    `scene_xml=None` switches the link back to the demo room without uploading anything. `volume` is
    a `modal.Volume` (anything with its `batch_upload` works, which is how the tests run offline).
    """
    sent = 0
    with volume.batch_upload(force=True) as batch:
        if scene_xml is not None:
            for local, relative in scene_files(Path(scene_xml)):
                batch.put_file(local, f"/{name}/{relative}")
                sent += local.stat().st_size
        batch.put_file(io.BytesIO((name if scene_xml is not None else DEMO).encode()),
                       f"/{CURRENT}")
    return sent


def served_scene(root: Path) -> tuple[Path, str] | None:
    """The scene `CURRENT` names under the mounted Volume, or None for the demo room."""
    marker = Path(root) / CURRENT
    try:
        name = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not _NAME.match(name) or name == DEMO:
        return None
    scene = Path(root) / name / "scene.xml"
    return (scene, name) if scene.is_file() else None


def build_app(root: Path, *, web_dir: Path | None = None, scratch: Path | None = None):
    """The room call app the container serves: the Volume's current scene, else the demo room."""
    import tempfile

    from litereality_agent.room_ops.roomcall.demo import write_demo_scene
    from litereality_agent.room_ops.roomcall.server import create_app

    current = served_scene(root)
    if current is None:
        # The container's own disk, not the Volume: the demo is rebuilt on every start, and writing
        # it to the Volume would race a laptop uploading a real scene.
        folder = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="roomcall-demo-"))
        return create_app(write_demo_scene(folder), label="demo room", web_dir=web_dir)
    scene, name = current
    return create_app(scene, label=name.replace("_", " ").replace("-", " "), web_dir=web_dir)


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
           credentials: tuple[str, str] | None = None, profile: str | None = None) -> int:
    """Upload the scene (or select the demo), deploy the app, print the link to share."""
    import modal

    app_file = Path(repo_root) / DEPLOY_FILE
    if not app_file.is_file():
        raise SystemExit(f"{DEPLOY_FILE} not found under {repo_root}: deploy from a source checkout")
    name = scene_name(label)
    if scene_xml is not None:
        # Compile it here first: a scene that does not load locally will not load in the
        # container either, and finding out there means a link that only shows an error.
        from litereality_agent.room_ops.roomcall.sim import load_model

        load_model(Path(scene_xml))
    with _modal_credentials(credentials, profile) as env:
        volume = modal.Volume.from_name(SCENES_VOLUME, environment_name=environment,
                                        create_if_missing=True)
        if scene_xml is not None:
            print(f"→ Uploading {name} to the Modal Volume {SCENES_VOLUME}")
            sent = publish(Path(scene_xml), name=name, volume=volume)
            print(f"  {sent / 1e6:.1f} MB")
        else:
            print("→ Serving the demo room")
            publish(None, name=DEMO, volume=volume)
        print(f"→ Deploying {APP_NAME} ({DEPLOY_FILE})")
        result = subprocess.run(
            [sys.executable, "-m", "modal", "deploy", "--env", environment, str(app_file)],
            env=env, check=False, text=True,
        )
        if result.returncode != 0:
            print("deploying the room call failed; see the Modal output above.", file=sys.stderr)
            return result.returncode
        url = modal.Function.from_name(APP_NAME, FUNCTION_NAME,
                                       environment_name=environment).get_web_url()
    print(f"\nroom call is live  {url}")
    print(f"  paste this in the call chat   {room_link(url)}")
    print("  everyone who opens it lands in the same room; the first visit after a quiet spell "
          "takes a few seconds while the server starts")
    print(f"  stop it   uv run modal app stop --env {environment} {APP_NAME}")
    return 0
