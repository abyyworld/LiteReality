"""The room call: people on a Zoom / Meet / Teams call (or none) open one link and share a room.

A LiteReality-scanned room (the MuJoCo `scene.xml` from `litereality stage simulate --from-seed`)
or a built-in demo room is simulated ONCE on the server, per room id, and streamed to every
browser in it: everyone walks around as an avatar with their webcam on its face and can grab,
drag and throw objects, open doors and drawers, and sees the same thing. The call carries the
voice; this adds the shared room. It can also run inside the Zoom client as a Zoom App.

    litereality roomcall [SCENE] [--demo] [--port 8790]
    litereality roomcall [SCENE] --deploy      # a permanent https link on Modal (`hosted`)
    python -m litereality_agent.room_ops.roomcall [SCENE] [--demo] [--port 8790]

Parts: `sim` (the shared physics), `geometry` (scene -> glb + manifest for the browser),
`server` (HTTP + WebSocket), `web/` (the client), `demo` (the built-in room), `hosted` (the same
server on Modal, so a call needs no tunnel).

Under `room_ops` because it reads a scene file and knows nothing about run layout. This module
stays import-light — only argparse wiring — so building the `litereality` parser never pulls in
MuJoCo or FastAPI.
"""

from __future__ import annotations

import argparse

DEFAULT_PORT = 8790


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The room call's arguments, shared by `litereality roomcall` and `python -m`."""
    parser.add_argument(
        "scene", nargs="?", metavar="SCENE",
        help="a scene.xml, or a folder holding scene.xml, mujoco_seed/scene.xml or "
             "mujoco/scene.xml (default: the demo room)",
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
        "--deploy", action="store_true",
        help="instead of serving here, upload the scene to your Modal workspace and deploy it as "
             "a permanent https link for the call chat (needs `uv sync --extra modal`)",
    )
    parser.add_argument("--modal-env", help="Modal environment to deploy into (default: main)")


def run(args: argparse.Namespace, *, credentials: tuple[str, str] | None = None,
        profile: str | None = None, environment: str | None = None) -> int:
    """Serve here, or with `--deploy` publish to Modal. `credentials`/`profile` are the `.env`
    ones `litereality roomcall` passes in; without them Modal uses its own active profile."""
    if getattr(args, "deploy", False):
        return _deploy(args, credentials=credentials, profile=profile,
                       environment=args.modal_env or environment or "main")
    from litereality_agent.room_ops.roomcall import server

    return server.run(args)


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
    else:
        scene, label = None, "demo room"
    return hosted.deploy(scene, label=label, repo_root=repo_root(), environment=environment,
                         credentials=credentials, profile=profile)
