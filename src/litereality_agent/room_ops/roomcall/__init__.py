"""The room call: people on a Zoom / Meet / Teams call (or none) open one link and share a room.

A LiteReality-scanned room (the MuJoCo `scene.xml` from `litereality stage simulate --from-seed`)
or a built-in demo room is simulated ONCE on the server, per room id, and streamed to every
browser in it: everyone walks around as an avatar with their webcam on its face and can grab,
drag and throw objects, open doors and drawers, and sees the same thing. The call carries the
voice; this adds the shared room. It can also run inside the Zoom client as a Zoom App.

    litereality roomcall [SCENE] [--demo] [--port 8790] [--watch]
    litereality roomcall [SCENE] --deploy      # a permanent https link on Modal (`hosted`)
    litereality roomcall [SCENE] --publish     # a new version of that link, live, no redeploy
    python -m litereality_agent.room_ops.roomcall [SCENE] [--demo] [--port 8790]

A room's scene can improve during the call — the quick RoomPlan room seconds after the scan, the
full pipeline's room later. Each is a new VERSION of the same link, swapped into every room live
with what people did carried over: `--watch` does it here whenever the scene's files change,
`--publish` on the hosted link.

Parts: `sim` (the shared physics), `geometry` (scene -> glb + manifest for the browser),
`server` (HTTP + WebSocket), `web/` (the client), `demo` (the built-in room), `hosted` (the same
server on Modal, so a call needs no tunnel), `avatars` and `avatar_format` (full-body people:
each person's body, built from their photos, drawn with Spark and driven live).

Under `room_ops` because it reads a scene file and knows nothing about run layout. This module
stays import-light — only argparse wiring — so building the `litereality` parser never pulls in
MuJoCo or FastAPI.
"""

from __future__ import annotations

import argparse

DEFAULT_PORT = 8790
# What a version may say it is (shown to people, and on the wire as the "scene" message's stage):
# RoomPlan boxes, per-object meshes, the full pipeline's room, any other scene, the demo.
STAGES = ("quick", "objects", "full", "scene", "demo")
# How full bodies are made (`avatars.MODES`, repeated here so the parser needs no numpy).
AVATAR_MODES = ("off", "standin", "modal")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The room call's arguments, shared by `litereality roomcall` and `python -m`."""
    parser.add_argument(
        "scene", nargs="?", metavar="SCENE",
        help="a scene.xml, or a folder holding scene.xml, mujoco_seed/scene.xml, "
             "mujoco/scene.xml or quick/scene.xml, the first found (default: the demo room)",
    )
    parser.add_argument("--demo", action="store_true",
                        help="serve the built-in demo room (also what no SCENE means)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to listen on (0.0.0.0 to reach it from the LAN)")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help="where to start looking for a free port; taken ones are skipped",
    )
    parser.add_argument("--label", help="room name shown to people (default: the scene's folder)")
    parser.add_argument("--no-open", dest="open_browser", action="store_false",
                        help="do not open a browser")
    parser.add_argument(
        "--watch", action="store_true",
        help="serve here and swap the scene in live, for everyone in the room, whenever its "
             "files change (a folder is re-resolved, so quick/ moves on to mujoco_seed/)",
    )
    parser.add_argument(
        "--stage", choices=[s for s in STAGES if s != "demo"],
        help="what this version of the room is (default: from its folder — quick/ is quick, "
             "mujoco_seed/ and mujoco/ are full, anything else scene)",
    )
    # `deploy` is False, "publish" or "deploy": truthy for both, so a caller that passes Modal
    # credentials only `if args.deploy` covers --publish as well.
    remote = parser.add_mutually_exclusive_group()
    remote.add_argument(
        "--deploy", action="store_const", const="deploy", default=False,
        help="instead of serving here, upload the scene to your Modal workspace and deploy it as "
             "a permanent https link for the call chat (needs `uv sync --extra modal`)",
    )
    remote.add_argument(
        "--publish", dest="deploy", action="store_const", const="publish",
        help="upload the scene as a new version of the already deployed link, without "
             "redeploying: everyone in it switches over within seconds, nobody leaves",
    )
    parser.add_argument("--modal-env", help="Modal environment to deploy into (default: main)")
    parser.add_argument(
        "--avatars", choices=AVATAR_MODES, default="standin",
        help="full-body people, from each person's turnaround or photos: 'standin' builds a "
             "simple body tinted from the photos here (default); 'modal' builds a photoreal one "
             "with LHM++ on your Modal GPU (deploy deploy/modal/avatar/app.py; the stand-in "
             "whenever that fails); 'off' keeps everyone a capsule or hologram. The hosted link "
             "always uses 'modal'",
    )


def run(args: argparse.Namespace, *, credentials: tuple[str, str] | None = None,
        profile: str | None = None, environment: str | None = None) -> int:
    """Serve here, or with `--deploy` / `--publish` put the scene on Modal. `credentials`/`profile`
    are the `.env` ones `litereality roomcall` passes in; without them Modal uses its own active
    profile."""
    if getattr(args, "deploy", False):
        if getattr(args, "watch", False):
            raise SystemExit("--watch serves here; for the hosted link, --publish each new "
                             "version (it goes live in seconds)")
        return _deploy(args, credentials=credentials, profile=profile,
                       environment=args.modal_env or environment or "main")
    from litereality_agent.room_ops.roomcall import server

    if getattr(args, "avatars", "standin") != "modal":
        return server.run(args)
    # Bodies on the GPU: this process calls the Modal function itself, with the `.env` token
    # (set before the client is imported, which is when it reads its configuration).
    import os

    if credentials:
        os.environ["MODAL_TOKEN_ID"], os.environ["MODAL_TOKEN_SECRET"] = credentials
    if profile:
        os.environ["MODAL_PROFILE"] = profile
    try:
        import modal  # noqa: F401
    except ImportError:
        raise SystemExit("--avatars modal needs the Modal client: run `uv sync --extra modal`, "
                         "or use --avatars standin") from None
    return server.run(args, modal_env=args.modal_env or environment or "main")


def _deploy(args: argparse.Namespace, *, credentials, profile, environment: str) -> int:
    from pathlib import Path

    from litereality_agent import repo_root
    from litereality_agent.room_ops.roomcall import hosted, server

    try:
        import modal  # noqa: F401
    except ImportError:
        raise SystemExit("deploying needs the Modal client: run `uv sync --extra modal`") from None
    if args.scene and not args.demo:
        try:
            scene = server.resolve_scene(args.scene)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        label = args.label or server.scene_label(Path(scene))
        stage = getattr(args, "stage", None) or server.scene_stage(Path(scene))
    else:
        scene, label, stage = None, "demo room", "demo"
    return hosted.deploy(scene, label=label, repo_root=repo_root(), environment=environment,
                         credentials=credentials, profile=profile, stage=stage,
                         redeploy=args.deploy != "publish")
