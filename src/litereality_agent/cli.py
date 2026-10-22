"""Public ``litereality`` command line.

The command surface deliberately mirrors the core architecture: run a pipeline,
run one stage, inspect a scene package, or invoke one model directly.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import threading
from pathlib import Path

from litereality_agent import console
from litereality_agent.pipeline import PipelineRunner, RunContext
from litereality_agent.pipeline.stages import STAGES
from litereality_agent.room_ops import roomcall
from litereality_agent.settings import load_settings

SCAN_MARKERS = ("room.usdz", "roomplan/room.usdz")

# Stages that CONSUME the scene package rather than producing it. Starting work here means the
# package must already exist — and then which package is the entire question.
AUTHORING_STAGES = frozenset({"author", "publish"})


def looks_like_scan_dir(path: str | os.PathLike[str]) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (
        any((candidate / marker).is_file() for marker in SCAN_MARKERS)
        or bool(next(candidate.glob("frame_*.jpg"), None))
    )


def resolve_scan(value: str) -> str:
    """Resolve a scan folder for small standalone callers retained in the package."""
    raw = value.rstrip("/")
    candidate = Path(raw).expanduser()
    if os.sep not in raw and not raw.startswith("."):
        return raw
    if not candidate.is_dir():
        raise SystemExit(f"no such scan folder: {candidate}")
    if not looks_like_scan_dir(candidate):
        raise SystemExit(f"{candidate} does not look like a RoomPlan capture")
    candidate = candidate.resolve()
    os.environ["LR_SCANS_DIR"] = str(candidate.parent)
    return candidate.name


def _print_results(results) -> int:
    failed = False
    marks = {"completed": "✓", "reused": "↻", "skipped": "⊘", "failed": "✗"}
    for result in results:
        detail = result.error or "; ".join(result.warnings)
        suffix = f" — {detail}" if detail else ""
        print(
            f"{marks[result.status.value]} {result.stage:<12} "
            f"{result.status.value:<9} {result.duration_seconds:7.1f}s{suffix}"
        )
        failed |= result.status.value == "failed" and result.details.get("fatal", True)
    return int(failed)


def _context(args) -> RunContext:
    return RunContext.resolve(args.target, output_root=getattr(args, "output_root", None))


def _require_scene_package(target: str, context: RunContext, stage: str) -> None:
    """Insist that an authoring stage is handed the scene package, not the capture it came from.

    A capture folder resolves to a perfectly valid context — same scan name, same default output
    root — so `stage author scans/<name>` appears to work. It stops appearing to work the moment
    `--output-root` differs from the default, at which point the stage authors one tree while you
    sit watching another, and nothing says so. The package path cannot be wrong in that way.

    The two failures want different answers, so they get them: a package that does not exist yet
    means scene init has not run, and saying so here beats the stage failing several steps later
    on a missing seed room.
    """
    package = context.scene_dir / "scene.json"
    try:
        given_is_package = Path(target).expanduser().resolve() == context.scene_dir.resolve()
    except OSError:
        given_is_package = False
    if package.is_file() and given_is_package:
        return

    red, bold, dim, off = (console.colour(c) for c in ("red", "bold", "dim", "off"))
    scene, capture = console.short(context.scene_dir), console.short(context.capture_dir)
    lines = [
        f"\n{red}✗ {stage} runs on the scene package, not the capture{off}\n",
        f"   {dim}given   {off} {target}",
        f"   {dim}expected{off} {scene}{dim}  — the tree scene init produces{off}\n",
    ]
    if package.is_file():
        lines += ["Point it at the package instead:\n",
                  f"   {bold}uv run litereality stage {stage} {scene}{off}\n"]
    else:
        lines += [f"{dim}{scene}/scene.json does not exist — scene init has not produced it yet.{off}\n",
                  f"   {bold}uv run litereality run {capture} --through seed{off}",
                  f"   {bold}uv run litereality stage {stage} {scene}{off}\n"]
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(2)


def _author_options(args) -> dict:
    # `--polish` deliberately does NOT include the model-driven quality pass. Refinement and
    # materials each produce something the room did not have before; QC re-examines what the
    # authoring pass already rendered and compared, so it is the longest agent pass on the run for
    # the smallest marginal change — and nothing downstream reads its output. It stays available
    # as `--quality-pass` for a room being prepared to be looked at rather than simulated.
    polish = getattr(args, "polish", False)
    opts = {
        "refine_objects": polish or getattr(args, "refine_objects", False),
        "materials": polish or getattr(args, "materials", False),
        "quality_pass": getattr(args, "quality_pass", False),
    }
    # Only pass what was actually given, so the stage keeps its own defaults for the rest.
    for name, key in (("author_profile", "profile"), ("author_steps", "step_budget"),
                      ("author_turns", "max_turns"), ("repair_rounds", "repair_rounds"),
                      ("author_provider", "provider"), ("repair_steps", "repair_steps"), ("repair_seconds", "repair_seconds")):
        value = getattr(args, name, None)
        if value is not None:
            opts[key] = value
    return opts


def _reconstruct_options(args) -> dict:
    """Options for the reconstruct stage.

    `chair_qc` gates `mesh_qc.chair_repair`, which is the only thing that looks at what TRELLIS
    actually produced: it flags fused floor slabs, squat blobs and floating fragments, then repairs
    a bad cluster by regenerating its reference image and re-running TRELLIS. The stage has always
    forwarded `--chair-qc` to `scene_init.flow`, but nothing here ever set it, so every chair went
    into a room ungated. It is ON by default — the checks are local geometry and cost nothing, and
    only a cluster that FAILS pays for a regeneration.
    """
    return {"chair_qc": getattr(args, "chair_qc", True)}


def _add_reconstruct_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--chair-qc", dest="chair_qc", action="store_true", default=True,
        help="QC generated chair meshes and regenerate the ones that fail (default)",
    )
    group.add_argument(
        "--no-chair-qc", dest="chair_qc", action="store_false",
        help="skip chair mesh QC — a bad chair then ships as-is",
    )


def _add_author_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--polish",
        action="store_true",
        help="run object refinement and materials after authoring "
             "(add --quality-pass for the model-driven QC pass as well)",
    )
    parser.add_argument("--refine-objects", action="store_true")
    parser.add_argument("--materials", action="store_true")
    parser.add_argument(
        "--quality-pass",
        action="store_true",
        help="model-driven QC pass — not included in --polish",
    )
    # Exposed for SHORT RUNS, not lowered as a default. Authoring spends its opening steps reading
    # the capture and measuring surfaces before it edits anything, so a small budget does not give
    # a rougher room, it gives an unfinished one — on Office-Elliott the first edit landed at step
    # 19 and the walls were not squared up until the sixties. A truncated run now says so in the
    # stage summary, which is what makes a low number safe to ask for.
    parser.add_argument(
        "--author-steps", type=int, default=None, metavar="N",
        help="tool-call budget for the authoring session (open default 300); "
             "a low value returns an unfinished room and says so",
    )
    parser.add_argument(
        "--author-profile", choices=["base", "detail", "simulation", "open"],
        help="authoring brief: base (materials + wall fixtures), detail (multi-part fixtures), "
             "simulation (adds real lighting, small objects, and support declarations), "
             "open (a goal brief with the full evidence pack + measurement helpers and the validation "
             "gate; default profile, using the codex/gpt-6 harness)",
    )
    parser.add_argument("--author-turns", type=int, help="authoring hard turn backstop (default 140)")
    parser.add_argument("--author-provider", choices=["claude", "codex"],
                        help="author and repair harness (default Codex/GPT-6)")
    parser.add_argument("--repair-rounds", type=int, choices=range(4), default=2,
                        help="bounded support-first repair rounds (default 2; 0 checks only)")
    parser.add_argument("--repair-steps", type=int, default=40, help="tool calls per repair (1..150)")
    parser.add_argument("--repair-seconds", type=int, default=600, help="wall seconds per repair (1..1800)")


def _simulate_options(args) -> dict:
    return {
        "shake": getattr(args, "shake", False),
        "from_seed": getattr(args, "from_seed", False),
        "reuse_meshes": getattr(args, "reuse_meshes", False),
        "no_decompose": getattr(args, "no_decompose", False),
    }


def _add_simulate_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shake", action="store_true",
        help="after exporting, step the scene under a rising ground acceleration and report what "
             "moved — the measurement that says whether the room's supports are true",
    )
    parser.add_argument(
        "--reuse-meshes", action="store_true",
        help="keep the collider meshes already exported and only rewrite scene.xml",
    )
    parser.add_argument(
        "--from-seed", action="store_true",
        help="export the room straight out of scene_init instead of the authored one — the shell "
             "plus the reconstructed objects, with no materials, fixtures or props",
    )
    parser.add_argument(
        "--no-decompose", action="store_true",
        help="skip convex decomposition for objects with no compiled physics — faster, but a "
             "concave body then collides as its hull (a table becomes a solid block)",
    )


def _add_publish_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compare-frames", type=int, default=None,
        help="real-vs-render comparison pairs to build at the end of publish; 0 to skip (default 6)",
    )


def _publish_options(args) -> dict:
    return {"compare_frames": getattr(args, "compare_frames", None)}


def _add_live_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live",
        action="store_true",
        help="watch the room and the agent's trace in a browser while this runs; the viewer keeps "
             "serving after it finishes (ctrl-c to stop)",
    )
    parser.add_argument(
        "--live-port", type=int, default=8770,
        help="where to start looking for a free port; taken ones are skipped",
    )
    parser.add_argument(
        "--live-no-bake",
        action="store_true",
        help="live viewer rebuilds geometry only — keeps up with rapid saves, but procedural "
             "SHELL materials render flat",
    )


def _block_until_interrupt() -> None:
    """Park the main thread so the daemon HTTP server keeps serving. Named so tests can replace it —
    otherwise every `--live` test would hang here waiting for a ctrl-c that never comes."""
    threading.Event().wait()


@contextlib.contextmanager
def _live_viewer(args, context):
    """Serve the room beside the work when `--live` is passed, and keep serving after it ends.

    Yields a callback the caller uses to say how the run turned out. Sharing one `RunContext` with
    the stage is the whole point — a separately launched viewer can be pointed at a different
    `--output-root` than the run it is meant to be watching, and then silently shows nothing.

    Deliberately does NOT require the room to exist first. The stage this wraps is the thing that
    creates it — `author` copies the seed room in as its first act — so demanding it up front made
    `--live` work only on the SECOND authoring run of a scene, which is the run you least need to
    watch. Every layer below already copes with a room that is not there yet: the watcher ignores a
    missing `Room.py` and fires the moment one appears, the server answers `no build yet`, and the
    page says "waiting for a build". Standalone `litereality live` keeps the hard check, because
    nothing in that command is going to produce a room.
    """
    if not getattr(args, "live", False):
        yield lambda _note: None
        return

    from litereality_agent.pipeline.realism_authoring import live

    bake = not args.live_no_bake
    room, server, url = live.start(context, port=args.live_port, bake=bake)
    live.describe(context, url, bake)
    print()
    try:
        yield room.finish
    finally:
        bold, cyan, off = (console.colour(c) for c in ("bold", "cyan", "off"))
        # Same burial problem as the opening banner, at the other end: this lands under whatever
        # the stage printed last, and it is the line you need in order to go and look at the room.
        print(f"\n   {bold}{cyan}▶ viewer still serving at {url}{off}"
              f"   {console.colour('dim')}ctrl-c to stop{off}\n")
        try:
            _block_until_interrupt()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            room.stop()
            server.shutdown()
            server.server_close()


def _run(args) -> int:
    context = _context(args)
    # Only when the run STARTS at authoring. A full `run scans/<name>` legitimately begins with a
    # capture and produces the package on the way through.
    if args.from_stage in AUTHORING_STAGES:
        _require_scene_package(args.target, context, args.from_stage)
    with _live_viewer(args, context) as finished:
        results = PipelineRunner().run(
            context,
            start=args.from_stage,
            through=args.through,
            force=set(args.force or ()),
            strict=args.strict,
            options={"ingest": {"use_dino": getattr(args, "use_dino", False)},
                     "reconstruct": _reconstruct_options(args),
                     "author": _author_options(args), "publish": _publish_options(args),
                     "simulate": _simulate_options(args)},
        )
        rc = _print_results(results)
        finished("run finished" if not rc else "run failed")
    return rc


def _stage(args) -> int:
    context = _context(args)
    if args.stage in AUTHORING_STAGES:
        _require_scene_package(args.target, context, args.stage)
    with _live_viewer(args, context) as finished:
        result = PipelineRunner().run_stage(
            context,
            args.stage,
            force=args.force,
            strict=args.strict,
            options={
                "skip_image_generation": args.skip_image_generation,
                "use_dino": getattr(args, "use_dino", False),
                **_reconstruct_options(args),
                **_author_options(args),
                **_publish_options(args),
                **_simulate_options(args),
            },
        )
        rc = _print_results([result])
        finished(f"{args.stage} finished" if not rc else f"{args.stage} failed")
    return rc


def _live(args) -> int:
    from litereality_agent.pipeline.realism_authoring import live

    return live.serve(
        args.target, port=args.port, host=args.host, poll=args.poll,
        output_root=args.output_root, bake=args.bake, bake_resolution=args.bake_resolution,
    )


def _view(args) -> int:
    """Walk a published room. `publish` knows which file it wrote; `room_ops.walk` serves any glb —
    resolving one run's tree to the other's argument is this layer's whole job."""
    from litereality_agent.pipeline.room_qc import publish
    from litereality_agent.room_ops import walk

    context = _context(args)
    return walk.serve(
        publish.viewable_room(context), context.scan,
        port=args.port, host=args.host, open_browser=args.open_browser,
        subtitle="published reconstruction",
    )


def _quick_target(value: str, output_root: str | None = None) -> tuple[Path, RunContext] | None:
    """The scan a quick room is built from, and the run it belongs to — or None if ``value``
    does not name a scan with a RoomPlan export. A scan folder, a room.usdz in one, a scene
    package that records its capture, or a bare scan name under ``$LR_SCANS_DIR`` all qualify."""
    from litereality_agent.pipeline.quickroom import find_room_usdz

    raw = Path(value).expanduser()
    if raw.is_file() and raw.suffix == ".usdz":
        raw = raw.parent
    # The scanner's `roomplan/` subfolder is part of the scan, not the scan: its parent is.
    if raw.is_dir() and raw.name == "roomplan" and (raw / "room.usdz").is_file():
        raw = raw.parent
    try:
        context = RunContext.resolve(raw if raw.exists() else value, output_root=output_root)
    except ValueError:
        return None
    if find_room_usdz(context.capture_dir) is None:
        return None
    return context.capture_dir, context


def _quickroom(args) -> int:
    """Build the quick room — RoomPlan boxes as furniture templates with the capture's colours and
    estimated physics — for one scan, in seconds, and say what came out."""
    import json
    import time

    from litereality_agent.pipeline.quickroom import REPORT_NAME, write_quick_scene

    found = _quick_target(args.target, args.output_root)
    if found is None:
        raise SystemExit(f"{args.target}: no RoomPlan room.usdz found (a scan folder holding "
                         "room.usdz or roomplan/room.usdz, or a scan name under $LR_SCANS_DIR)")
    scan_dir, context = found
    out = Path(args.out).expanduser() if args.out else context.scene_dir / "quick"
    started = time.perf_counter()
    scene = write_quick_scene(scan_dir, out, label=args.label, colours=args.colours,
                              textures=args.textures)
    seconds = time.perf_counter() - started
    report = json.loads((out / REPORT_NAME).read_text())
    bodies = report["bodies"].values()
    joints = [b["joint"] for b in bodies]
    bold, dim, off = (console.colour(c) for c in ("bold", "dim", "off"))
    colours = report["colours"]
    layout = report["layout"]
    print(f"{bold}quick room{off}  {report['label']}  {dim}{seconds:.1f} s{off}")
    print(f"  scene    {console.short(scene)}")
    print(f"  bodies   {len(joints)}: {joints.count('free')} free, {joints.count('hinge')} hinged, "
          f"{joints.count('static')} static · {report['walls']} walls · "
          f"{len(report['openings'])} openings")
    print("  colours  " + (f"from {colours['frames_used']} capture frames"
                            if colours["source"] == "frames"
                            else f"by category ({colours['note'] or 'no frames'})"))
    changes = [f"merged {name} <- {'+'.join(members)}"
               for name, members in (layout.get("merged") or {}).items()]
    if layout.get("moved"):
        changes.append("moved " + ", ".join(layout["moved"]))
    if layout.get("error"):
        changes.append(f"repair failed: {layout['error']}")
    if layout.get("skipped"):
        changes.append(f"repair skipped ({layout['skipped']})")
    if changes:
        print(f"  layout   {' · '.join(changes)}")
    for warning in report.get("warnings") or ():
        print(f"  {console.colour('yellow')}warning{off}  {warning}")
    if report.get("pinned"):
        print(f"  pinned   {', '.join(report['pinned'])} {dim}(could not be separated){off}")
    if report["initial_overlaps"]:
        print(f"  overlaps {len(report['initial_overlaps'])} deeper than 5 mm at t=0")
    print(f"  {dim}walk it with others:{off} uv run litereality roomcall {console.short(scene)}")
    return 0


def _quick_room_for_scan(args) -> None:
    """`litereality roomcall <scan>`: when SCENE names a capture rather than a scene, build its
    quick room first and serve that, so a room is on the call seconds after it was scanned.

    A scene that already resolves is left alone, and so is a scan whose run already holds the
    pipeline's own export — that is the better room — unless the scan is NEWER than that export:
    a re-scan under the same name is a different room, and the old export would put everyone in
    the room as it was. Otherwise the quick room is written into ``run/<scan>/quick/`` and the RUN
    folder is what is served when the room call resolves it there, so `--watch` moves on to
    ``mujoco_seed/`` by itself the moment the full export lands. Which room is served, and why, is
    printed.
    """
    scene = getattr(args, "scene", None)
    if not scene or getattr(args, "demo", False):
        return
    from litereality_agent.room_ops.roomcall import server

    if not str(scene).endswith(".usdz"):
        try:
            server.resolve_scene(scene)
            return
        except ValueError:
            pass
    found = _quick_target(str(scene))
    if found is None:
        return                       # not a scan either: the room call says what it looked for
    import json
    import time

    from litereality_agent.pipeline.quickroom import REPORT_NAME, find_room_usdz, write_quick_scene

    scan_dir, context = found
    run_dir = context.scene_dir
    try:
        existing = server.resolve_scene(run_dir)
    except ValueError:
        existing = None
    usdz = find_room_usdz(scan_dir)
    rescanned = (existing is not None and usdz is not None
                 and usdz.stat().st_mtime > existing.stat().st_mtime)
    if existing is not None and existing.parent.name != "quick" and not rescanned:
        print(f"room call  {context.scan}: serving the pipeline's room {console.short(existing)} "
              "(newer than the scan)")
        args.scene = str(run_dir)
    else:
        started = time.perf_counter()
        quick = write_quick_scene(scan_dir, run_dir / "quick", label=context.scan,
                                  textures=getattr(args, "textures", True))
        why = ("the scan is newer than the pipeline's room "
               f"{console.short(existing)}, which is left unserved"
               if rescanned and existing.parent.name != "quick" else
               "there is no pipeline room for this scan yet")
        print(f"room call  {context.scan}: serving the quick room {console.short(quick)}, built "
              f"in {time.perf_counter() - started:.1f} s — {why}")
        report = json.loads((quick.parent / REPORT_NAME).read_text())
        for warning in report.get("warnings") or ():
            print(f"  {console.colour('yellow')}warning{console.colour('off')}  {warning}")
        try:
            served = server.resolve_scene(run_dir)
        except ValueError:
            served = None
        # The run folder only while it still resolves to this quick room: once it holds an older
        # full export, it would resolve (and `--watch` would move) to that instead.
        args.scene = str(run_dir if served == quick else quick)
    if not getattr(args, "label", None):
        args.label = context.scan


def _modal_kwargs(settings) -> dict:
    """The `.env` Modal login, the way `roomcall --deploy` passes it on."""
    return {"credentials": None if settings.modal_profile else settings.modal_credentials(),
            "profile": settings.modal_profile, "environment": settings.modal_environment}


def _object_generator(args, settings=None):
    """TRELLIS on the `.env`'s Modal workspace — or, with `--fake`, the offline stand-in."""
    from litereality_agent.pipeline import realroom

    if getattr(args, "fake", False):
        return realroom.FakeGenerator(delay=float(getattr(args, "fake_delay", 0.0) or 0.0))
    settings = settings or load_settings()
    if not settings.modal_configured():
        raise SystemExit("generating objects needs TRELLIS on Modal: set MODAL_TOKEN_ID and "
                         "MODAL_TOKEN_SECRET (or MODAL_PROFILE) in .env and deploy it once with "
                         "`uv run litereality setup` — or try the flow offline with --fake")
    try:
        import modal  # noqa: F401
    except ImportError:
        raise SystemExit("generating objects needs the Modal client: run "
                         "`uv sync --extra modal`") from None
    kwargs = _modal_kwargs(settings)
    return realroom.ModalTrellisGenerator(
        app_name=settings.modal_trellis_app, function_name=settings.modal_trellis_function,
        environment_name=kwargs["environment"], profile=kwargs["profile"],
        credentials=kwargs["credentials"])


def _estimate_line(generator, objects: int, passes) -> str:
    from litereality_agent.pipeline.realroom import generate

    if generator.name == "fake":
        return "stand-in boxes (--fake): no GPU, no cost"
    price = load_settings().modal_gpu_price_per_hour
    minutes, dollars = generate.estimate(objects, passes, price_per_hour=price)
    return (f"TRELLIS on Modal, up to {generate.CONTAINERS} H100s: ~{minutes:.0f} min to the last "
            f"final, ~${dollars:.2f} of GPU time (an estimate, see `realroom.generate.estimate`)")


def _articulated_source(args, settings=None):
    """Infinigen's openable furniture for the objects stage (`realroom.articulated`), or None.

    ``--articulated`` asks for it, ``--no-articulated`` refuses it; without either it is on only
    with ``$LR_ARTICULATED=1`` (minutes of CPU per room the first time: never by surprise). It runs
    in a local Infinigen env when ``$LR_INFINIGEN_PYTHON`` names that env's python, else — with
    ``--fake`` — as the offline stand-in, else on Modal (``litereality-infinigen``,
    deploy/modal/infinigen). Both variables are read from the shell, not `.env`."""
    from litereality_agent.models.articulated import infinigen
    from litereality_agent.pipeline.realroom import articulated

    choice = getattr(args, "articulated", None)
    local = os.environ.get(infinigen.PYTHON_ENV, "").strip()
    wanted = os.environ.get("LR_ARTICULATED", "").strip().lower() in ("1", "true", "yes", "on")
    if choice is False or (choice is None and not wanted):
        return None
    fake = bool(getattr(args, "fake", False))
    modal_kwargs = None
    if not local and not fake:
        settings = settings or load_settings()
        if not settings.modal_configured():
            raise SystemExit("--articulated needs Infinigen: a local env ($LR_INFINIGEN_PYTHON, "
                             "see deploy/modal/infinigen/README.md) or Modal credentials in "
                             ".env and `modal deploy deploy/modal/infinigen/app.py` once")
        modal_kwargs = _modal_kwargs(settings)
    runner = infinigen.runner_from_env(fake=fake, modal_kwargs=modal_kwargs)
    try:
        runner.check()
    except Exception as exc:  # noqa: BLE001 — said once here rather than once per object
        message = f"Infinigen ({runner.name}) is not available: {type(exc).__name__}: {exc}"
        if choice:
            raise SystemExit(message) from None
        print(f"  {console.colour('yellow')}{message}{console.colour('off')}; "
              "openable furniture skipped", file=sys.stderr)
        return None
    drawers = [h.strip() for h in (getattr(args, "drawers", "") or "").split(",") if h.strip()]
    return articulated.ArticulatedSource(runner, drawers=drawers)


def _plan_articulated(room, source) -> list:
    """Claim the openable objects before the crops are cut; print what will be built."""
    if source is None:
        return []
    source.plan(room)
    if source.targets:
        print(f"  {console.colour('dim')}openable{console.colour('off')} "
              f"{len(source.targets)} with {source.runner.name}: {source.describe()}")
    return [source]


def _report_objects(room, progress) -> None:
    from litereality_agent.pipeline.realroom import build

    bold, dim, off = (console.colour(c) for c in ("bold", "dim", "off"))
    have = build.counts(room)
    print(f"{bold}objects{off}  {have['final']} final, {have['preview']} preview, "
          f"{have['template']} still templates  {dim}{progress.seconds:.0f} s, "
          f"{progress.versions} version(s){off}")
    print(f"  scene    {console.short(room.scene_path)}")
    opened = sorted(h for h, s in progress.installed.items() if s == "infinigen")
    if opened:
        print(f"  openable {len(opened)} from Infinigen: {', '.join(opened)}")
    for key, error in sorted(progress.failed.items()):
        print(f"  {console.colour('yellow')}failed{off}   {key}: {error}")
    for handle, donor in sorted(progress.borrowed.items()):
        print(f"  {dim}twin     {handle} looks like {donor} (its own view was too partial){off}")
    for handle, why in sorted(progress.skipped.items()):
        print(f"  {dim}template {handle}: {why}{off}")


def _realroom(args) -> int:
    """`litereality realroom <scan>`: the objects stage, offline — every object generated (one
    pass) and fitted into the quick room, written as one scene."""
    from litereality_agent.pipeline import realroom

    found = _quick_target(args.target, args.output_root)
    if found is None:
        raise SystemExit(f"{args.target}: no RoomPlan room.usdz found (a scan folder holding "
                         "room.usdz or roomplan/room.usdz, or a scan name under $LR_SCANS_DIR)")
    scan_dir, context = found
    out = Path(args.out).expanduser() if args.out else context.scene_dir / realroom.DEFAULT_DIR
    generator = _object_generator(args)
    room = realroom.RealRoom(scan_dir, out, label=args.label or context.scan,
                             colours=args.colours, textures=args.textures, rebuild=True)
    room.write(check=False)
    extras = _plan_articulated(room, _articulated_source(args))
    room.make_crops()
    print(f"realroom  {context.scan}: {len(room.generated_handles())} objects, pass "
          f"{args.pass_name} — {_estimate_line(generator, len(room.crops), [args.pass_name])}")
    progress = realroom.generate_room(room, generator, passes=[args.pass_name],
                                      min_interval=float("inf"), reuse=not args.fresh,
                                      extras=extras)
    _report_objects(room, progress)
    print(f"  {console.colour('dim')}walk it with others:{console.colour('off')} "
          f"uv run litereality roomcall {console.short(room.scene_path)}")
    return 0


def _realistic_roomcall(args) -> int:
    """`litereality roomcall <scan> --realistic`: the quick room on the call at once, then every
    object generated (preview, then final) and swapped in live as it lands.

    Served here, the run's ``objects/scene.xml`` is watched and each version written there is
    swapped in (`--watch` is implied). With ``--publish`` / ``--deploy`` the quick room goes up
    first, and every version after it is published to the same link (`MIN_INTERVAL` apart)."""
    import time

    from litereality_agent.pipeline import realroom

    found = _quick_target(str(args.scene)) if args.scene and not args.demo else None
    if found is None:
        raise SystemExit("--realistic needs a scan (a folder holding room.usdz or "
                         "roomplan/room.usdz, or a scan name under $LR_SCANS_DIR)")
    scan_dir, context = found
    out = context.scene_dir / realroom.DEFAULT_DIR
    label = args.label or context.scan
    settings = load_settings()
    generator = _object_generator(args, settings)
    source = _articulated_source(args, settings)
    started = time.perf_counter()
    room = realroom.RealRoom(scan_dir, out, label=label,
                             textures=getattr(args, "textures", True), rebuild=True)
    scene = room.write(check=False)
    print(f"room call  {context.scan}: the quick room in {time.perf_counter() - started:.1f} s; "
          "each object is swapped in live as it is generated")
    args.scene, args.label = str(scene), label
    passes = ["preview", "final"]

    def generate(on_version=None):
        extras = _plan_articulated(room, source)
        room.make_crops()
        print(f"  objects  {len(room.crops)}: "
              f"{_estimate_line(generator, len(room.crops), passes)}")
        progress = realroom.generate_room(room, generator, passes=passes, on_version=on_version,
                                          extras=extras)
        _report_objects(room, progress)
        return progress

    if args.deploy:
        modal_kwargs = _modal_kwargs(settings)
        args.stage = "quick"
        rc = roomcall.run(args, **modal_kwargs)
        if rc:
            return rc

        def publish(path, _info) -> None:
            args.scene, args.deploy, args.stage = str(path), "publish", realroom.STAGE
            roomcall.run(args, **modal_kwargs)

        generate(publish)
        return 0
    args.watch = True
    args.stage = args.stage or realroom.STAGE

    def background() -> None:
        try:
            generate()
        except Exception as exc:  # noqa: BLE001 — the room stays up on what it has
            print(f"  objects  stopped: {type(exc).__name__}: {exc}", file=sys.stderr)

    worker = threading.Thread(target=background, daemon=True, name="realroom")
    args._realroom_worker = worker            # for tests: the thread generating the versions
    worker.start()
    return roomcall.run(args)


def _roomcall(args) -> int:
    """Share a simulated room with the people on a video call. Takes a scene file or folder, not a
    run: the room call is a room operation and knows nothing about run layout. A scan is the one
    exception, resolved HERE (this layer knows run layout): its quick room is built first."""
    if getattr(args, "realistic", False):
        return _realistic_roomcall(args)
    _quick_room_for_scan(args)
    # The `.env` Modal token is needed to deploy, and to build full bodies on the GPU from here.
    if not args.deploy and getattr(args, "avatars", "standin") != "modal":
        return roomcall.run(args)
    settings = load_settings()
    return roomcall.run(
        args,
        credentials=None if settings.modal_profile else settings.modal_credentials(),
        profile=settings.modal_profile,
        environment=settings.modal_environment,
    )


def _inspect(args) -> int:
    from litereality_agent.room_ops import manifest

    package = manifest.require(args.target)
    print(package.summary())
    required, optional = package.check()
    for path in optional:
        print(f"  ~ missing optional: {path}")
    for path in required:
        print(f"  ! missing required: {path}")
    return int(bool(required))


def _setup(args) -> int:
    from litereality_agent import repo_root
    from litereality_agent.runtimes import setup

    settings = load_settings()
    return setup.run(
        repo_root(),
        environment=args.env or settings.modal_environment,
        profile=args.profile or settings.modal_profile,
        credentials=None if args.profile else settings.modal_credentials(),
        skip_deploy=args.skip_deploy,
    )


def _generate_3d(args) -> int:
    from litereality_agent.models.registry import gen3d_from_settings

    service = gen3d_from_settings(load_settings())
    output = args.out or str(Path(args.image).with_suffix(".glb"))
    result = service.reconstruct(
        [args.image], out_dir=str(Path(output).resolve().parent),
        asset_id=Path(output).stem, seed=args.seed, simplify=args.simplify,
    )
    print(result)
    return int(not result or not Path(result).is_file())


def _add_articulated_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--articulated", dest="articulated", action="store_true", default=None,
        help="openable furniture from Infinigen-Sim: cabinets, a dishwasher, the door and "
             "windows built with real hinges and drawers, sized to the scan (in "
             "$LR_INFINIGEN_PYTHON's env, else on Modal; default off, or on with "
             "$LR_ARTICULATED=1)")
    parser.add_argument("--no-articulated", dest="articulated", action="store_false",
                        help="keep every object's template (or TRELLIS mesh)")
    parser.add_argument("--drawers", metavar="HANDLES", default="",
                        help="with --articulated: storage units that are drawer chests, "
                             "comma-separated (e.g. Storage3)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="litereality", description="LiteReality-Agent")
    commands = parser.add_subparsers(dest="command", required=True)
    stages = [stage.name for stage in STAGES]

    run = commands.add_parser("run", help="run the reconstruction pipeline")
    run.add_argument("target", metavar="SCAN_OR_SCENE")
    run.add_argument("--from", dest="from_stage", choices=stages)
    run.add_argument("--through", choices=stages)
    run.add_argument("--force", action="append", choices=stages)
    run.add_argument("--strict", action="store_true")
    run.add_argument("--output-root")
    run.add_argument("--use-dino", action="store_true",
                     help="Refine capture crops with GroundingDINO during ingest.")
    _add_reconstruct_options(run)
    _add_author_options(run)
    _add_live_options(run)
    _add_publish_options(run)
    _add_simulate_options(run)
    run.set_defaults(handler=_run)

    stage = commands.add_parser("stage", help="run exactly one pipeline stage")
    stage.add_argument("stage", choices=stages)
    stage.add_argument("target", metavar="SCAN_OR_SCENE")
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--strict", action="store_true")
    stage.add_argument("--skip-image-generation", action="store_true")
    stage.add_argument("--output-root")
    stage.add_argument("--use-dino", action="store_true",
                       help="Refine capture crops with GroundingDINO during ingest.")
    _add_reconstruct_options(stage)
    _add_author_options(stage)
    _add_live_options(stage)
    _add_publish_options(stage)
    _add_simulate_options(stage)
    stage.set_defaults(handler=_stage)

    view = commands.add_parser("view", help="walk a published room in the browser")
    view.add_argument("target", metavar="SCAN_OR_SCENE")
    view.add_argument(
        "--port", type=int, default=8780,
        help="where to start looking for a free port; taken ones are skipped",
    )
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--output-root")
    view.add_argument("--no-open", dest="open_browser", action="store_false")
    view.set_defaults(handler=_view)

    call = commands.add_parser(
        "roomcall", help="share a simulated room with the people on a video call"
    )
    roomcall.add_arguments(call)
    call.add_argument(
        "--realistic", action="store_true",
        help="with a scan: the quick room at once, then every object generated with TRELLIS on "
             "Modal (a fast preview, then the final) and swapped in live as it lands; with "
             "--publish/--deploy each version goes to the hosted link",
    )
    call.add_argument("--no-textures", dest="textures", action="store_false",
                      help="flat colours only; skip the wall/floor/ceiling photos")
    call.add_argument("--fake", action="store_true",
                      help="with --realistic: stand-in textured boxes instead of TRELLIS "
                           "(offline, free) to try the flow")
    call.add_argument("--fake-delay", type=float, default=1.0, metavar="S",
                      help=argparse.SUPPRESS)
    _add_articulated_options(call)
    call.set_defaults(handler=_roomcall)

    real = commands.add_parser(
        "realroom",
        help="a scan's objects stage offline: every object's photo -> TRELLIS mesh, fitted into "
             "the quick room",
    )
    real.add_argument("target", metavar="SCAN",
                      help="a scan folder (room.usdz or roomplan/room.usdz), or a scan name "
                           "under $LR_SCANS_DIR")
    real.add_argument("--out", metavar="DIR",
                      help="where to write it (default: run/<scan>/objects)")
    real.add_argument("--output-root")
    real.add_argument("--label", help="the room's name (default: the scan's)")
    real.add_argument("--pass", dest="pass_name", choices=["preview", "final"], default="final",
                      help="preview: TRELLIS at 512 (fast); final: 1024 cascade (default)")
    real.add_argument("--fake", action="store_true",
                      help="stand-in textured boxes instead of TRELLIS: offline, free")
    real.add_argument("--fresh", action="store_true",
                      help="generate again even where a mesh for the same photo is cached")
    real.add_argument("--no-colours", dest="colours", action="store_false",
                      help="the quick room in category colours (skip reading the frames)")
    real.add_argument("--no-textures", dest="textures", action="store_false",
                      help="flat colours only; skip the wall/floor/ceiling photos")
    _add_articulated_options(real)
    real.set_defaults(handler=_realroom, fake_delay=0.0)

    quick = commands.add_parser(
        "quickroom",
        help="build a scan's quick room in seconds: RoomPlan boxes as furniture, real colours, "
             "estimated physics",
    )
    quick.add_argument("target", metavar="SCAN",
                       help="a scan folder (room.usdz or roomplan/room.usdz), or a scan name "
                            "under $LR_SCANS_DIR")
    quick.add_argument("--out", metavar="DIR", help="where to write it (default: run/<scan>/quick)")
    quick.add_argument("--output-root")
    quick.add_argument("--label", help="the room's name (default: the scan's)")
    quick.add_argument("--no-colours", dest="colours", action="store_false",
                       help="category colours only; skip reading the capture's frames")
    quick.add_argument("--no-textures", dest="textures", action="store_false",
                       help="flat colours only; skip the wall/floor/ceiling photos")
    quick.set_defaults(handler=_quickroom)

    live = commands.add_parser("live", help="watch a room being authored, with the agent's trace")
    live.add_argument("target", metavar="SCAN_OR_SCENE")
    live.add_argument(
        "--port", type=int, default=8770,
        help="where to start looking for a free port; taken ones are skipped",
    )
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--poll", type=float, default=1.0, help="seconds between Room.py checks")
    live.add_argument("--output-root")
    live.add_argument(
        "--no-bake", dest="bake", action="store_false",
        help="geometry only — faster, but procedural SHELL materials render flat",
    )
    live.add_argument("--bake-resolution", type=int, default=1024)
    live.set_defaults(handler=_live)

    setup = commands.add_parser(
        "setup", help="authenticate Modal and deploy the litereality model app"
    )
    setup.add_argument("--profile", help="Modal profile to use instead of the active one")
    setup.add_argument("--env", help="Modal environment (default: MODAL_ENVIRONMENT)")
    setup.add_argument(
        "--skip-deploy", action="store_true", help="authenticate and record the profile only"
    )
    setup.set_defaults(handler=_setup)

    scene = commands.add_parser("scene", help="inspect a generated scene package")
    scene_subcommands = scene.add_subparsers(dest="scene_command", required=True)
    inspect = scene_subcommands.add_parser("inspect")
    inspect.add_argument("target", nargs="?", default=None)
    inspect.set_defaults(handler=_inspect)

    model = commands.add_parser("model", help="invoke one configured model")
    model_subcommands = model.add_subparsers(dest="model_command", required=True)
    generate = model_subcommands.add_parser("generate-3d")
    generate.add_argument("image")
    generate.add_argument("--out")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--simplify", type=float, default=0.95)
    generate.set_defaults(handler=_generate_3d)

    evaluate = commands.add_parser(
        "evaluate",
        help="the paper's numbers for a room: physics audit per stage and, with --render, "
             "visual fidelity against the scan's own photos",
    )
    evaluate.add_argument(
        "targets", nargs="+", metavar="SCENE_OR_RUN",
        help="a scene.xml, a folder holding one, a run folder (run/<scan>: every stage it has — "
             "quick/, objects/, mujoco_seed/) or a scan name; several are compared in one table")
    evaluate.add_argument("--scan", help="the capture to compare renders with (a scan folder or "
                                         "a name under $LR_SCANS_DIR); default: what the scene's "
                                         "build recorded")
    evaluate.add_argument("--out", metavar="DIR",
                          help="where the reports go (default: run/<scan>/eval, or eval/ beside "
                               "the scene)")
    evaluate.add_argument("--output-root")
    evaluate.add_argument("--render", action="store_true",
                          help="render the room from the scan's camera poses in headless "
                               "Chromium (needs `uv run --with playwright`) and compare with the "
                               "photos: PSNR/SSIM overall, on walls/floor, on objects")
    evaluate.add_argument("--size", default="320x240", metavar="WxH",
                          help="the resolution images are compared at (default 320x240)")
    evaluate.add_argument("--max-views", type=int, default=None, metavar="N",
                          help="compare at most N of the capture's frames, evenly spread")
    evaluate.add_argument("--holdout", type=int, default=None, metavar="K",
                          help="with --render: also score each photo against a quick room "
                               "rebuilt WITHOUT it (K folds of the capture's frames; default 4, "
                               "0 to skip). The quick room's walls are made of these photos, so "
                               "the as-built numbers are training views")
    evaluate.add_argument("--holdout-split", choices=("interleaved", "block"),
                          default="interleaved",
                          help="folds of every K-th frame (default), or K contiguous stretches "
                               "of the trajectory (stricter)")
    evaluate.add_argument("--fast", action="store_true",
                          help="shorter physics checks (2 s stability, one timing run, no "
                               "concurrency or server-capacity test, no hold-out unless "
                               "--holdout): for a quick look, not for the paper's numbers")
    evaluate.set_defaults(handler=_evaluate)
    return parser


def _is_capture(path: Path) -> bool:
    """A scan folder (or its room.usdz) rather than a scene, a stage folder or a run folder."""
    from litereality_agent.pipeline.evaluate.report import SCENE_NAME, STAGE_FOLDERS
    from litereality_agent.pipeline.quickroom import find_room_usdz

    if path.is_file():
        return path.suffix == ".usdz"
    if not path.is_dir() or (path / SCENE_NAME).is_file():
        return False
    if any((path / folder / SCENE_NAME).is_file() for folder in STAGE_FOLDERS):
        return False
    return find_room_usdz(path) is not None


def _run_root(path: Path) -> Path:
    """The run folder a target belongs to: itself for a run folder, the parent of its stage
    folder for a stage or a scene file in one."""
    from litereality_agent.pipeline.evaluate.report import SCENE_NAME, STAGE_FOLDERS

    folder = path if path.is_dir() else path.parent
    if path.is_dir() and not (path / SCENE_NAME).is_file():
        return folder
    return folder.parent if folder.name in STAGE_FOLDERS else folder


def _evaluate(args) -> int:
    """`litereality evaluate`: the per-stage reports and the comparison tables.

    Each target is compared with ITS OWN capture. A scan name or a scan folder is resolved here
    (this layer knows the run layout): it means that scan's run folder, compared with that scan.
    A scene, stage or run folder is compared with the capture its build recorded. ``--scan``
    names one capture for every target, because the caller said so (the report flags a scene
    whose build recorded another)."""
    from litereality_agent.pipeline import evaluate as evaluation

    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--size {args.size}: expected WxH, e.g. 320x240") from None
    scan = None
    if args.scan:
        found = _quick_target(args.scan, args.output_root)
        if found is None:
            raise SystemExit(f"--scan {args.scan}: no RoomPlan scan there (a folder holding "
                             "room.usdz, or a scan name under $LR_SCANS_DIR)")
        scan = found[0]
    targets: list[tuple[Path, Path | None]] = []
    for target in args.targets:
        path = Path(target).expanduser()
        own = scan
        if not path.exists() or _is_capture(path):
            found = _quick_target(target, args.output_root)
            if found is None:
                raise SystemExit(f"{target}: no such scene, run folder or scan")
            capture, context = found
            path, own = context.scene_dir, scan or capture
            if not path.is_dir():
                raise SystemExit(f"{target}: a scan, but its run folder {path} does not exist "
                                 f"yet; build a stage first (`litereality quickroom {target}`)")
        try:
            evaluation.find_scenes(path)
        except ValueError as exc:
            raise SystemExit(f"{target}: {exc}") from None
        targets.append((path, own))
    roots = {_run_root(path.resolve()) for path, _own in targets}
    if args.out:
        out = Path(args.out).expanduser()
    elif len(roots) == 1:
        out = roots.pop() / "eval"
    else:
        out = Path(os.path.commonpath([str(r) for r in roots])) / "eval"
    renderer = None
    if args.render:
        try:
            renderer = evaluation.PlaywrightRenderer()
        except evaluation.RendererUnavailable as exc:
            raise SystemExit(str(exc)) from None
    holdout = args.holdout if args.holdout is not None else (0 if args.fast else 4)
    options = ({"stability_seconds": 2.0, "cost_repeats": 1, "measure_concurrency": False}
               if args.fast else {})
    result = evaluation.evaluate(targets, out, render=args.render, renderer=renderer,
                                 size=(w, h), max_views=args.max_views, physics_options=options,
                                 holdout=holdout if args.render else 0,
                                 holdout_split=args.holdout_split)
    bold, dim, yellow, off = (console.colour(c) for c in ("bold", "dim", "yellow", "off"))
    print()
    print((out / "comparison.md").read_text(encoding="utf-8"))
    for report in result["reports"]:
        print(f"  {bold}{report['name']}{off}  {console.short(out / report['name'] / 'report.md')}"
              + ("" if report.get("fidelity") else f"  {dim}({report['fidelity_skipped']}){off}"))
        if report.get("scan_mismatch"):
            print(f"    {yellow}scan mismatch{off}: {report['scan_mismatch']}")
    print(f"  {dim}tables:{off} {console.short(out / 'comparison.md')} "
          f"{dim}(+ .json, .tex){off}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_settings().apply_environment()
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
