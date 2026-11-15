"""The room call: people on a Zoom / Meet / Teams call (or none) open one link and share a room.

A LiteReality-scanned room (the MuJoCo `scene.xml` from `litereality stage simulate --from-seed`)
or a built-in demo room is simulated ONCE on the server, per room id, and streamed to every
browser in it: everyone walks around as an avatar with their webcam on its face and can grab,
drag and throw objects, open doors and drawers, and sees the same thing. The call carries the
voice; this adds the shared room. It can also run inside the Zoom client as a Zoom App.

    litereality roomcall [SCENE] [--demo] [--port 8790] [--watch]
    litereality roomcall [SCENE] --deploy      # a permanent https link on Modal (`hosted`)
    litereality roomcall [SCENE] --publish     # a new version of that link, live, no redeploy
    litereality roomcall SCENE --library       # a ready room people can pick on that link
    python -m litereality_agent.room_ops.roomcall [SCENE] [--demo] [--port 8790]

Someone on the call can also send a scan straight into their room from the phone that made it
(Rooms, then Add a room you scanned, in the page): its quick room is built in seconds and swapped
into that room alone, then, where `--uploads realistic`, the realistic objects follow version by
version (`scans`). On the hosted link, Rooms also lists the ready rooms of its library
(`--library`): anyone in a call moves everyone in their call room into one in a click.

A room's scene can improve during the call — the quick RoomPlan room seconds after the scan, the
full pipeline's room later. Each is a new VERSION of the same link, swapped into every room live
with what people did carried over: `--watch` does it here whenever the scene's files change,
`--publish` on the hosted link.

Parts: `sim` (the shared physics), `geometry` (scene -> glb + manifest for the browser),
`server` (HTTP + WebSocket), `web/` (the client), `demo` (the built-in room), `hosted` (the same
server on Modal, so a call needs no tunnel), `avatars` and `avatar_format` (full-body people:
each person's body, built from their photos, drawn with Spark and driven live), `scans` (a scan
sent into a room during the call, built by the CLI in child processes).

Under `room_ops` because it reads a scene file and knows nothing about run layout. This module
stays import-light — only argparse wiring — so building the `litereality` parser never pulls in
MuJoCo or FastAPI.
"""

from __future__ import annotations

import argparse

DEFAULT_PORT = 8790
# The hosted link's room library (`hosted.add_to_library`): what `--library` and
# `--library-remove` put in `args.deploy`, beside "deploy" and "publish".
LIBRARY_ADD = "library"
LIBRARY_REMOVE = "library-remove"
# What a version may say it is (shown to people, and on the wire as the "scene" message's stage):
# RoomPlan boxes, per-object meshes, the full pipeline's room, any other scene, the demo.
STAGES = ("quick", "objects", "full", "scene", "demo")
# How full bodies are made (`avatars.MODES`, repeated here so the parser needs no numpy).
AVATAR_MODES = ("off", "standin", "modal")
# What a scan sent into a room becomes (`scans.MODES`, repeated for the same reason).
UPLOAD_MODES = ("off", "quick", "realistic")


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
    # `deploy` is False, "publish", "deploy", "library" or "library-remove": truthy for all of
    # them, so a caller that passes Modal credentials only `if args.deploy` covers each.
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
    remote.add_argument(
        "--library", dest="deploy", action="store_const", const=LIBRARY_ADD,
        help="add the scene to the deployed link's room library instead: a ready room anyone in "
             "a call can move their call room into (Rooms in the page), named after --label; "
             "the link's default room stays as it is",
    )
    remote.add_argument(
        "--library-remove", metavar="NAME", action=_LibraryRemove, default=None,
        help="take the room NAME out of the deployed link's room library",
    )
    parser.add_argument(
        "--thumb", metavar="IMG.jpg", default=None,
        help="with --library: the JPEG (at most 1 MB) the room is shown with in the room picker",
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
    parser.add_argument(
        "--uploads", choices=UPLOAD_MODES, default=None,
        help="what a scan someone sends into their room during the call (Rooms: a .zip of "
             "the scan's folder, or its room.usdz) becomes, in that room only: 'quick' its quick "
             "room in seconds; 'realistic' then realistic objects from TRELLIS on Modal, swapped "
             "in as they land; 'off' no uploads. Default: realistic when Modal credentials are "
             "configured (or with --fake-objects), else quick. The hosted link uses realistic",
    )
    parser.add_argument(
        "--splat", metavar="DIR", default=None,
        help="the photoreal room: a splat bundle folder (`litereality splatroom` writes "
             "run/<scan>/splat); everyone sees the real room as Gaussian splats, each movable "
             "object's splat moving with it, and every new snapshot swaps in live. Default: "
             "run/<scan>/splat when serving a scan that has one",
    )
    parser.add_argument(
        "--fake-objects", action="store_true",
        help="with --uploads realistic: stand-in textured boxes instead of TRELLIS (offline, "
             "free), to try the whole flow on this machine",
    )


class _LibraryRemove(argparse.Action):
    """`--library-remove NAME`: the name, and `deploy` set like the other Modal actions so the
    `.env` credentials are passed on for it too."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        namespace.deploy = LIBRARY_REMOVE


def run(args: argparse.Namespace, *, credentials: tuple[str, str] | None = None,
        profile: str | None = None, environment: str | None = None) -> int:
    """Serve here, or with `--deploy` / `--publish` put the scene on Modal. `credentials`/`profile`
    are the `.env` ones `litereality roomcall` passes in; without them Modal uses its own active
    profile."""
    # `--realistic` publishes every version after the quick room as the link's default room:
    # with --library that would replace the default, not add to the library. Refused before
    # anything is uploaded.
    if getattr(args, "realistic", False) and getattr(args, "deploy", False) in (LIBRARY_ADD,
                                                                                LIBRARY_REMOVE):
        raise SystemExit("--library adds one finished scene: build the room first (e.g. "
                         "`litereality realroom`), then add its objects/scene.xml with --library")
    if getattr(args, "thumb", None) and getattr(args, "deploy", False) != LIBRARY_ADD:
        raise SystemExit("--thumb is the picture of a room added with --library")
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
    if args.deploy == LIBRARY_REMOVE:
        if args.scene or args.demo:
            raise SystemExit("--library-remove takes a room name, not a scene")
        return hosted.library_remove(args.library_remove, environment=environment,
                                     credentials=credentials, profile=profile)
    if args.deploy == LIBRARY_ADD and (args.demo or not args.scene):
        raise SystemExit("--library adds a scanned room: give its SCENE (the demo room is "
                         "always there as the default)")
    if args.scene and not args.demo:
        try:
            scene = server.resolve_scene(args.scene)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        label = args.label or server.scene_label(Path(scene))
        stage = getattr(args, "stage", None) or server.scene_stage(Path(scene))
    else:
        scene, label, stage = None, "demo room", "demo"
    if args.deploy == LIBRARY_ADD:
        return hosted.library_add(scene, label=label, environment=environment,
                                  credentials=credentials, profile=profile, stage=stage,
                                  thumb=Path(args.thumb) if args.thumb else None)
    return hosted.deploy(scene, label=label, repo_root=repo_root(), environment=environment,
                         credentials=credentials, profile=profile, stage=stage,
                         redeploy=args.deploy != "publish")
