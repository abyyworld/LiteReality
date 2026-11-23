"""splatroom — the scanned room as photoreal Gaussian splats, growing live, cut to the physics.

    stage "instant"  the scan's photos + LiDAR as fused surfels: no GPU, ~8 s (models/splat/rgbd)
    stage "trained"  gsplat's trainer on Modal, a snapshot every few hundred steps, each sharper

Every stage is split (`split.split`) against the room's MuJoCo scene: the static room, and one
splat per movable body in that body's own frame, so the browser hangs each object's splat under
the body the server moves — grab the real chair and the real chair moves.

What it writes, in ``run/<scan>/splat/`` (`DEFAULT_DIR`):

    splat.json                 the current bundle: {"stage", "step", "room": file,
                               "bodies": {name: file}, "counts", "created", "view", "capture"},
                               rewritten atomically ("view": where a newcomer first stands;
                               "capture": where the phone was, for the browser's walk zone)
    room-<hash>.spz            files are named by their content, so a browser caches them forever
    body-<name>-<hash>.spz     and a bundle never points at a half-written file
    report.json                timings, splat counts and the trainer's held-out PSNR/SSIM/LPIPS
    raw.spz.bin                the latest trained snapshot, uncut and uncleaned (`SavedTrainer`)
    raw.clean.spz.bin          the same after `clean` (with ``clean=True`` or the agreement filter):
                               what was cut and served

Two optional improvements for the views the scan barely saw (both off by default):

    dense depth  a `refiner` (PromptDA on Modal, `models.splat.promptda`; cached in the scan's
                 ``depth_dense/``) turns each 256 x 192 LiDAR map into a 960-wide metric map. The
                 instant room is built from it, the training set draws more points from it, and the
                 trainer may fit its depth and normals (options "dense_depth", "normal_lambda").
                 Uncached, it is fetched while the LiDAR room is already on screen: the first
                 room is never later than without it. Its GPU is woken at the start, beside the
                 trainer's, and training waits for it at most `DEPTH_DEADLINE` seconds from then:
                 past that the room trains on the LiDAR, and the depth still lands in the cache
                 for the next run. When it arrives during the run, the dense instant room is
                 fused while the training set is sent and the trainer starts, not before them:
                 the trainer never waits for it.
    clean        every snapshot loses its faint splats, the ones outside RoomPlan's room and the
                 floaters the depth maps see through (`models.splat.clean`) before it is cut.
    agreement    ("support_views" in `clean_options`, on its own or with `clean`) every snapshot
                 keeps only the splats at least that many training photos saw on their surface:
                 the speckle a thin scan leaves where one or two photos looked goes.

A re-cut (`SavedTrainer`) replays `RAW` through the same rules: a trained room is cleaned again
with new rules on a laptop, no GPU.

    litereality splatroom <scan> [--scene run/<scan>/objects/scene.xml] [--instant-only]
                         [--depth promptda] [--depth-loss-dense] [--normal-lambda F] [--clean]
                         [--support-views N] [--recut --no-masks]
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import mujoco

from litereality_agent.models.splat import clean as splat_clean
from litereality_agent.models.splat import dataset as splat_dataset
from litereality_agent.models.splat import promptda, rgbd, spz
from litereality_agent.models.splat import views as splat_views
from litereality_agent.pipeline.splatroom.split import Split, corrected_scene, split

DEFAULT_DIR = "splat"
BUNDLE = "splat.json"
REPORT = "report.json"
SPLAT_SCENE = "scene_splat.xml"  # the physics with the bodies moved, beside the scene cut against
RAW = "raw.spz.bin"            # the latest snapshot uncut (not a bundle file: never served)
RAW_CLEAN = "raw.clean.spz.bin"  # the same cleaned, as it was cut (`clean`, agreement)
INSTANT_VOXEL = 0.016          # ~640k splats, ~6 MB for tea_room: on screen in seconds
# Points per photo drawn from dense depth: up to this many, but no more in all than the trainer's
# splat budget (MCMC never prunes its starting points, so 20000 x 300 photos would start it at 6M).
DENSE_POINTS_MAX = 20_000
CAP_MAX = 800_000              # the trainer's default splat budget (deploy/modal/splat "cap_max")
TEST_EVERY = 8                 # the trainer's default hold-out ("test_every")
# Capture spots a bundle carries (the browser keeps walkers near them): ~1 per photo of a thin
# scan, a spread of a long one; a few kB of JSON either way.
CAPTURE_MAX = 200
# Seconds from the start of a build that training waits for dense depth. A warm 60-photo scan
# takes ~17 s, a cold one ~32 s; no free GPU, or a stalled upload, must not hold the H100 for the
# Modal call's own 10-minute limit while the LiDAR would have trained the room in ~75 s.
DEPTH_DEADLINE = 90.0
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def dense_points_for(photos: int, cap_max: int = CAP_MAX) -> int:
    """The dense training set's points per photo when none is asked for: the splat budget shared
    out over the photos, between the LiDAR's own `POINTS_PER_FRAME` and `DENSE_POINTS_MAX`."""
    share = int(cap_max) // max(1, photos)
    return int(min(DENSE_POINTS_MAX, max(splat_dataset.POINTS_PER_FRAME, share)))


def wants_dense_maps(options: dict | None) -> bool:
    """Whether trainer `options` ask for the dense depth maps in the dataset (`depths/`)."""
    options = options or {}
    return bool(options.get("dense_depth")) or float(options.get("normal_lambda") or 0) > 0


def room_bounds(scan_dir: Path, log: Callable[[str], None] = print):
    """RoomPlan's room (`clean.Room`) for the walls rule, or None (no room.usdz, unreadable)."""
    from litereality_agent.pipeline.quickroom.roomplan import find_room_usdz, parse_room

    usdz = find_room_usdz(scan_dir)
    if usdz is None:
        return None
    try:
        return splat_clean.room_from_plan(parse_room(usdz))
    except Exception as exc:  # noqa: BLE001 — carving still runs without the walls
        log(f"  clean: no room bounds ({type(exc).__name__}: {exc}); carving only")
        return None


def _name(prefix: str, data: bytes) -> str:
    return f"{prefix}-{hashlib.sha256(data).hexdigest()[:16]}.spz"


def capture_positions(frames, most: int = CAPTURE_MAX) -> list[list[float]]:
    """Where the phone was for each photo, in the room's frame (Z up, as "view"), rounded to the
    centimetre: at most `most`, evenly spaced in frame order (`clean.spread`). The browser keeps
    walkers within reach of these spots, where the splat has photos behind it."""
    picked = splat_clean.spread(list(frames), most)
    return [[round(float(v), 2) for v in f.c2w[:3, 3]] for f in picked]


def write_bundle(out_dir: Path, parts: Split, *, stage: str, step: int | None = None,
                 antialiased: bool = True, view: dict | None = None,
                 capture: list | None = None) -> dict:
    """Write one bundle (files first, then `splat.json` atomically) and drop older files. The
    previous bundle's files are kept one bundle longer: an upload of it may still be reading them
    (`BundleUploader`), and a browser may still be fetching them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = read_bundle(out_dir)
    keep: set[str] = set()
    room_bytes = spz.encode(parts.room, antialiased=antialiased)
    room = _name("room", room_bytes)
    (out_dir / room).write_bytes(room_bytes)
    keep.add(room)
    bodies = {}
    for name, splats in sorted(parts.bodies.items()):
        data = spz.encode(splats, antialiased=antialiased)
        file = _name(f"body-{_SAFE.sub('_', name)}", data)
        (out_dir / file).write_bytes(data)
        bodies[name] = file
        keep.add(file)
    bundle = {"stage": stage, "step": step, "room": room, "bodies": bodies,
              "counts": {"room": len(parts.room), **parts.counts},
              "bytes": sum((out_dir / f).stat().st_size for f in keep),
              "created": round(time.time(), 3)}
    if view is not None:
        bundle["view"] = view          # where a newcomer first stands (`views.best_view`)
    if capture:
        bundle["capture"] = capture    # the capture spots (`capture_positions`)
    tmp = out_dir / (BUNDLE + ".tmp")
    tmp.write_text(json.dumps(bundle, indent=1))
    os.replace(tmp, out_dir / BUNDLE)
    spare = set()
    if previous is not None:
        spare = {previous["room"], *previous.get("bodies", {}).values()}
    for old in out_dir.glob("*.spz"):
        if old.name not in keep and old.name not in spare:
            old.unlink(missing_ok=True)
    return bundle


def add_view(out_dir: Path, scan_dir: Path) -> dict | None:
    """Give the bundle in `out_dir` its first view (`views.best_view`) and its capture spots
    (`capture_positions`) where it has none yet, for bundles written before either existed.
    Rewrites `splat.json` atomically when something was added; returns the bundle."""
    bundle = read_bundle(out_dir)
    if bundle is None or ("view" in bundle and "capture" in bundle):
        return bundle
    frames = rgbd.load_frames(scan_dir)
    added = False
    if "view" not in bundle:
        view = splat_views.best_view(frames)   # reads every photo: only when it is missing
        if view is not None:
            bundle["view"], added = view, True
    if "capture" not in bundle and frames:
        bundle["capture"], added = capture_positions(frames), True
    if not added:
        return bundle
    tmp = out_dir / (BUNDLE + ".tmp")
    tmp.write_text(json.dumps(bundle, indent=1))
    os.replace(tmp, out_dir / BUNDLE)
    return bundle


def read_bundle(out_dir: Path) -> dict | None:
    try:
        bundle = json.loads((Path(out_dir) / BUNDLE).read_text())
    except (OSError, ValueError):
        return None
    return bundle if isinstance(bundle, dict) and isinstance(bundle.get("room"), str) else None


def build(scan_dir: Path, out_dir: Path, scene_xml: Path, *, trainer=None,
          options: dict | None = None, instant_only: bool = False,
          log: Callable[[str], None] = print,
          on_bundle: Callable[[dict], None] | None = None,
          segment: Callable[[dict], dict] | None = None,
          on_scene: Callable[[Path], None] | None = None,
          move_bodies: bool = True, refiner=None, dense_points: int | None = None,
          clean: bool = False, clean_options: dict | None = None,
          depth_deadline: float = DEPTH_DEADLINE) -> dict:
    """The instant room, then (with a `trainer`) each trained snapshot, each written as a bundle
    the moment it exists. Returns the report (also written to `REPORT`).

    The first split snaps the free bodies onto the splat; every later one reuses those
    corrections, and `corrected_scene` writes the physics with the bodies moved to match
    (``scene_splat.xml`` beside `scene_xml`: serve that one). With `segment` (the SAM 2 function,
    `models.splat.remote.segment_call`), each body's outline in the photos is fetched while the GPU trains, and
    every bundle after it arrives is cut by the photos' masks (`masks`).

    With a `refiner` (`promptda.ModalDepthRefiner`: ``refine(frames, width) -> {frame index:
    metres}``) the scan's depth is made dense once (see the module docstring); `dense_points` is
    then the training set's points per photo (default `dense_points_for`). Dense trainer losses
    asked for in `options` are dropped, with a line in the report, when no dense depth came back
    (or none within `depth_deadline` seconds of the start: ``report["depth"]["fallback"]``).
    With `clean`, every snapshot goes through `clean.clean` (keywords `clean_options`) first.
    ``clean_options["support_views"]`` (the agreement filter) works with or without `clean`:
    without it, agreement is the only rule that runs (the faint, outside and carving rules are
    switched off), so each rule can be tried alone."""
    import threading

    from litereality_agent.pipeline.splatroom import masks as object_masks

    t0 = time.perf_counter()
    training = trainer is not None and not instant_only
    if training and hasattr(trainer, "warm"):
        trainer.warm()                 # the GPU boots while the instant room is computed
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    frames = rgbd.load_frames(scan_dir)
    if not frames:
        raise ValueError(f"{scan_dir}: no frames with a pose, a photo and depth")
    first_view = splat_views.best_view(frames)
    capture = capture_positions(frames)
    report: dict = {"scan": str(scan_dir), "scene": str(scene_xml), "frames": len(frames),
                    "view": first_view, "capture": len(capture),
                    "depth": {"source": "promptda" if refiner is not None else "lidar"},
                    "bundles": []}
    options = dict(options or {})
    # A re-cut holds out what the training run held out (--no-holdout, a --trainer-option),
    # not what this command line would: otherwise photos it trained on lose their vote.
    trained_options = (trainer.report.get("options") or {}) if isinstance(
        trainer, SavedTrainer) else {}
    test_every = int(options.get("test_every", trained_options.get("test_every", TEST_EVERY)))
    clean_kw = dict(clean_options or {})
    support_views = int(clean_kw.get("support_views") or 0)
    cleaning = clean or support_views > 0
    if clean:
        clean_kw.setdefault("bounds", room_bounds(scan_dir, log))
    elif cleaning:
        # Agreement alone: the other rules stay off unless `clean` asks for them too.
        clean_kw.update(min_opacity=0.0, bounds=None, carve_votes=0)
    if cleaning:
        report["clean"] = {"margin": clean_kw.get("margin", splat_clean.MARGIN),
                           "carve_frames": clean_kw.get("carve_frames",
                                                        splat_clean.CARVE_FRAMES),
                           "min_opacity": clean_kw.get("min_opacity", splat_clean.MIN_OPACITY),
                           "carve_votes": clean_kw.get("carve_votes", splat_clean.CARVE_VOTES),
                           "support_views": support_views,
                           "room_bounds": clean_kw["bounds"] is not None,
                           "test_every": test_every}

    state: dict = {"corrections": None, "masks": None, "dense": None}
    clean_kw.setdefault("depth_cache", {})     # each frame's carving depth, read once per build

    def cleaned(splats: spz.Splats, dense: dict | None,
                image_names=None) -> tuple[spz.Splats, dict]:
        """`splats` through `clean.clean` (when on), carving and checking agreement with the
        training photos only: the trainer scores its held-out photos, so their depth must not
        decide what stays."""
        if not cleaning:
            return splats, {}
        out, stats = splat_clean.clean(splats, frames, dense=dense or None,
                                       test_every=test_every, image_names=image_names,
                                       **clean_kw)
        log(f"  clean: {stats['kept']:,} of {stats['input']:,} kept ({stats['faint']:,} faint, "
            f"{stats['outside']:,} outside, {stats['carved']:,} carved, "
            f"{stats['unsupported']:,} unsupported), {stats['seconds']} s")
        return out, {"clean": stats}

    publishing = threading.RLock()     # the dense instant room is published from its own thread

    def publish(splats: spz.Splats, stage: str, step: int | None, extra: dict) -> None:
        with publishing:
            _publish(splats, stage, step, extra)

    def _publish(splats: spz.Splats, stage: str, step: int | None, extra: dict) -> None:
        t = time.perf_counter()
        # Before the masks: boxes only, and no snapping when masks are on their way (the photos
        # will say where each object is). The first split with masks fixes the corrections.
        waiting = segment is not None and state["masks"] is None
        parts = split(splats, model, corrections=state["corrections"], masks=state["masks"],
                      snap_bodies=not waiting, move_bodies=move_bodies)
        if state["corrections"] is None and not waiting:
            state["corrections"] = parts.corrections
            # An earlier run's moved physics may still be the served one; with nothing moved
            # this time the bodies are cut at the scanned poses, so serve those again.
            earlier = Path(scene_xml).with_name(SPLAT_SCENE)
            moved_before = earlier.is_file() and earlier.resolve() != Path(scene_xml).resolve()
            fixed = corrected_scene(Path(scene_xml), parts.corrections or {},
                                    out_name=SPLAT_SCENE)
            report["corrections"] = parts.corrections
            report["corrected_scene"] = str(fixed) if fixed else None
            if fixed:
                log(f"  {len(parts.corrections)} object(s) moved to where the "
                    f"{'photos' if state['masks'] is not None else 'splat'} show them: {fixed}")
                if on_scene:
                    on_scene(fixed)
            elif moved_before and on_scene:
                log(f"  no object moved this time: the physics as scanned again ({scene_xml})")
                on_scene(Path(scene_xml))
        bundle = write_bundle(out_dir, parts, stage=stage, step=step, view=first_view,
                              capture=capture)
        entry = {"stage": stage, "step": step, "at": round(time.perf_counter() - t0, 2),
                 "split_write_s": round(time.perf_counter() - t, 2), "splats": len(splats),
                 "bodies": len(parts.bodies), "masks": state["masks"] is not None,
                 "mb": round(bundle["bytes"] / 1e6, 2), **extra}
        report["bundles"].append(entry)
        log(f"  {stage}{'' if step is None else f' step {step}'}: {len(splats):,} splats, "
            f"{len(parts.bodies)} objects cut out{' by photo masks' if entry['masks'] else ''}, "
            f"{entry['mb']} MB, at {entry['at']} s")
        if on_bundle:
            on_bundle(bundle)

    def fuse(dense: dict | None, image_names=None) -> tuple[spz.Splats, dict]:
        # Cleaned once: the masks' re-cuts below publish this same room again. Before the
        # training set exists its carving cannot know which photos it drops: by name alone.
        splats = rgbd.instant_splats(frames, voxel=INSTANT_VOXEL, dense=dense or None)
        splats, extra = cleaned(splats, dense, image_names)
        if dense:
            extra["depth"] = "promptda"
        return splats, extra

    def show_instant(splats: spz.Splats, extra: dict) -> None:
        with publishing:
            state["instant"] = (splats, extra)
            publish(splats, "instant", None, extra)

    def instant_room(dense: dict | None) -> None:
        show_instant(*fuse(dense, report.get("images")))

    def dense_instant_in_background(dense: dict) -> None:
        """Fuse the dense instant room beside the training set's upload and the trainer's start:
        the trainer must not wait ~12 s for a room its first snapshot soon replaces. Published
        from its thread the moment it is fused, never once a trained room is up; the event loop
        waits for it (`join_dense_instant`) only before the first snapshot, by when it is done."""
        names = report.get("images")

        def run() -> None:
            try:
                room = fuse(dense, names)
            except Exception as exc:  # noqa: BLE001 — the LiDAR room stays on screen
                log(f"  dense instant room failed ({type(exc).__name__}: {exc}); "
                    "the LiDAR one stays")
                return
            with publishing:
                if any(b["stage"] == "trained" for b in report["bundles"]):
                    return             # a trained room is already better than this one
                show_instant(*room)
                report["instant_dense_s"] = report["bundles"][-1]["at"]
        thread = threading.Thread(target=run, daemon=True, name="dense-instant")
        thread.start()
        state["dense_instant"] = thread

    def join_dense_instant() -> None:
        thread = state.pop("dense_instant", None)
        if thread is not None:
            thread.join()

    def instant_again() -> None:
        with publishing:
            splats, extra = state["instant"]
            publish(splats, "instant", None, extra)

    def write_raw(name: str, splats: spz.Splats) -> None:
        raw = out_dir / name
        tmp = raw.with_name(raw.name + ".tmp")
        tmp.write_bytes(spz.encode(splats))
        os.replace(tmp, raw)

    if isinstance(trainer, SavedTrainer) and not instant_only:
        report["images"] = trainer.report.get("images")   # the photos it was trained on

    refining = None
    refined: dict = {}                 # the refine thread's result, read only once it is done
    if refiner is not None:
        def refine() -> None:
            t = time.perf_counter()
            try:
                got = refiner.refine(frames, width=splat_dataset.TRAIN_WIDTH) or {}
            except Exception as exc:  # noqa: BLE001 — refine never should; LiDAR carries on
                log(f"  dense depth failed ({type(exc).__name__}: {exc}); LiDAR only")
                got = {}
            refined.update(dense=got, report={
                "frames": len(got), "s": round(time.perf_counter() - t, 2),
                **dict(getattr(refiner, "last_stats", None) or {})})

        def take_refined() -> None:
            state["dense"] = refined["dense"]
            report["depth"].update(refined["report"])
            if not state["dense"]:
                report["depth"]["fallback"] = "lidar"

        # Already refined (the scan's depth_dense/ cache): no wait, the first room is dense.
        # Otherwise the LiDAR room goes on screen while PromptDA works, and is replaced.
        if all(promptda.is_cached(f, splat_dataset.TRAIN_WIDTH) for f in frames):
            refine()
            take_refined()
            report["depth"]["cached"] = True
        else:
            if hasattr(refiner, "warm"):
                refiner.warm()         # its GPU boots while the scan is read and fused
            refining = threading.Thread(target=refine, daemon=True, name="dense-depth")
            refining.start()

    instant_room(state["dense"])
    report["instant_s"] = report["bundles"][-1]["at"]

    mask_thread = None
    if segment is not None:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)                 # the scanned poses: what SAM 2 is shown
        from litereality_agent.room_ops.roomcall.model_info import body_name, movable_body_ids

        bodies = {body_name(model, b): b for b in sorted(movable_body_ids(model))}
        views = object_masks.plan_views(frames, model, data, bodies)

        def fetch() -> None:
            t = time.perf_counter()
            try:
                from litereality_agent.pipeline.splatroom.split import _static_boxes

                # Walls included: a SAM candidate lying mostly on the wall is not the object.
                static = _static_boxes(model, data, set(bodies.values()))
                got = object_masks.request(views, segment, static_boxes=static)
            except Exception as exc:  # noqa: BLE001 — the boxes still cut the room
                report["masks_error"] = f"{type(exc).__name__}: {exc}"
                log(f"  photo masks failed ({report['masks_error']}); cutting by boxes")
                state["failed"] = True
                return
            report["masks"] = {"views": len(views), "masks": len(got),
                               "gpu_s": round(got.seconds, 1),
                               "s": round(time.perf_counter() - t, 1)}
            state["masks"] = got
            log(f"  photo masks: {len(got)} from {len(views)} photos in "
                f"{report['masks']['s']} s")

        mask_thread = threading.Thread(target=fetch, daemon=True)
        mask_thread.start()

    if refining is not None:
        refining.join(timeout=max(0.0, depth_deadline - (time.perf_counter() - t0)))
        if refining.is_alive():
            # Train on the LiDAR now; the thread carries on and fills the cache for next time.
            state["dense"] = {}
            report["depth"].update(fallback="timeout", deadline_s=depth_deadline)
            log(f"  dense depth not back within {depth_deadline:.0f} s: training on the LiDAR")
        else:
            take_refined()
            report["depth"]["at"] = round(time.perf_counter() - t0, 2)
        if state["dense"] and not training:
            instant_room(state["dense"])
            report["instant_dense_s"] = report["bundles"][-1]["at"]
    if mask_thread is not None and not training:
        mask_thread.join()
        mask_thread = None
        if state["masks"] is None:
            segment = None                     # failed: snap by the boxes after all
        instant_again()

    if training:
        t = time.perf_counter()
        if isinstance(trainer, SavedTrainer):
            tar = b""
        else:
            tar = training_set(frames, state["dense"] or {}, options, dense_points, report, log)
        report["dataset_s"] = round(time.perf_counter() - t, 2)
        report["dataset_mb"] = round(len(tar) / 1e6, 1)
        if refining is not None and state["dense"]:
            # Only now: fused beside the set's build, the two shared one core (the GIL) and the
            # trainer started ~8 s later; beside the upload and the GPU's start it waits on the
            # network, not on the fuse.
            dense_instant_in_background(state["dense"])
        report["options"] = dict(options)
        log(f"  training set: {len(frames)} photos, {report['dataset_mb']} MB; "
            f"training on Modal ({trainer.name})")
        for event in trainer.stream(tar, options):
            if event["kind"] == "snapshot":
                join_dense_instant()   # on screen before the first trained room, not after it
            if mask_thread is not None and event["kind"] == "snapshot":
                mask_thread.join()              # cut every trained room by the photos
                mask_thread = None
                if state["masks"] is None:
                    segment = None
                # Where the objects really are comes from the LiDAR room, whose splats lie on
                # the surfaces; trained splats are free to float. It is published again, cut by
                # the photos, just before the trained one.
                instant_again()
            kind = event["kind"]
            if kind == "started":
                report["trainer_started_s"] = round(time.perf_counter() - t0, 2)
                report["gpu"] = event.get("gpu")
                if event.get("depths"):         # dense depth maps the trainer found in the tar
                    report["trainer_depths"] = event["depths"]
                log(f"  trainer up on {event.get('gpu')} at {report['trainer_started_s']} s")
            elif kind == "snapshot":
                splats = spz.decode(event["spz"])
                if not event.get("replayed"):
                    # Uncut and uncleaned, for evaluation and for cutting again (`SavedTrainer`).
                    write_raw(RAW, splats)
                splats, extra = cleaned(splats, state["dense"], report.get("images"))
                if cleaning:
                    write_raw(RAW_CLEAN, splats)
                else:                   # an earlier run's: evaluate must not score it as this one
                    (out_dir / RAW_CLEAN).unlink(missing_ok=True)
                publish(splats, "trained", int(event["step"]),
                        {"trainer_s": round(float(event["seconds"]), 1), **extra,
                         **({"replayed": True} if event.get("replayed") else {})})
            elif kind == "done":
                report["stats"] = event.get("stats") or {}
                report["trainer_s"] = round(float(event.get("seconds") or 0), 1)
                s = report["stats"]
                if s:
                    log(f"  held-out photos: PSNR {s.get('psnr', float('nan')):.2f} dB, "
                        f"SSIM {s.get('ssim', float('nan')):.3f}, "
                        f"LPIPS {s.get('lpips', float('nan')):.3f}")
            elif kind == "error":
                report["error"] = event.get("error")
                report["trainer_log"] = (event.get("log") or "")[-4000:]
                log(f"  training failed: {event.get('error')}")
        join_dense_instant()           # done before the report is written
    if mask_thread is not None:
        # The trainer ended without a snapshot (it failed, or never started): the masks are still
        # used, or the boxes snap the bodies, so the room left behind is never the unsnapped
        # first cut; and the thread is done before it can touch the report being written.
        mask_thread.join()
        mask_thread = None
        if state["masks"] is None:
            segment = None
        instant_again()
    if isinstance(trainer, SavedTrainer) and not instant_only:
        # A re-cut trains nothing: the report keeps what the training run measured.
        for key in ("dataset_s", "dataset_mb", "trainer_started_s", "gpu", "trainer_s",
                    "dataset", "options"):
            if key in trainer.report:
                report[key] = trainer.report[key]
        report["recut"] = True
    report["total_s"] = round(time.perf_counter() - t0, 2)
    (out_dir / REPORT).write_text(json.dumps(report, indent=1))
    return report


def training_set(frames: list, dense: dict, options: dict, dense_points: int | None,
                 report: dict, log: Callable[[str], None] = print) -> bytes:
    """The trainer's dataset tar, from dense depth where there is some (`dataset.build`).

    Dense losses asked for in `options` are taken out of it (in place) when no photo has a dense
    map to fit, rather than failing the run: the trainer refuses them without ``depths/``. The
    photos it holds go in ``report["images"]`` (they set the trainer's hold-out order) and its
    make-up in ``report["dataset"]``."""
    maps = wants_dense_maps(options)
    per_photo = splat_dataset.POINTS_PER_FRAME
    test_every = int(options.get("test_every", TEST_EVERY))
    if dense:
        # Shared over the photos that seed it: the held-out ones give no points.
        seeding = len(frames) - (-(-len(frames) // test_every) if test_every > 0 else 0)
        per_photo = int(dense_points or dense_points_for(seeding,
                                                         options.get("cap_max", CAP_MAX)))
    # The trainer's hold-out photos give no starting points: they score the room.
    ds = splat_dataset.build(frames, points_per_frame=per_photo, dense=dense or None,
                             depth_maps=maps and bool(dense),
                             test_every=test_every)
    depth_maps = sum(1 for n in ds.files if n.startswith(splat_dataset.DEPTH_DIR + "/"))
    if maps and not depth_maps:
        dropped = {k: options.pop(k) for k in ("dense_depth", "dense_depth_lambda",
                                               "normal_lambda") if k in options}
        report["dense_losses_dropped"] = dropped
        log("  no dense depth to fit: training without the dense depth and normal losses")
    report["images"] = sorted(n[len("images/"):] for n in ds.files if n.startswith("images/"))
    report["dataset"] = {"photos": ds.images, "held_out": getattr(ds, "held_out", 0),
                         "points": ds.points,
                         "points_per_photo": per_photo, "dense_photos": ds.dense_images,
                         "depth_maps": depth_maps}
    return ds.tar()


class SavedTrainer:
    """The last trained snapshot in `out_dir` (`RAW`), replayed as if the GPU had just sent it:
    re-cut a room (new masks, new cutting rules) without training it again."""

    name = "saved snapshot"

    def __init__(self, out_dir: Path) -> None:
        self.raw = Path(out_dir) / RAW
        try:
            self.report = json.loads((Path(out_dir) / REPORT).read_text())
        except (OSError, ValueError):
            self.report = {}
        if not self.raw.is_file():
            raise FileNotFoundError(f"{self.raw}: no trained snapshot to re-cut (train it first)")
        self.data = self.raw.read_bytes()            # now: the run itself rewrites the folder

    def stream(self, _dataset_tar: bytes, _options: dict | None = None):
        trained = [b for b in self.report.get("bundles", []) if b.get("stage") == "trained"]
        step = trained[-1]["step"] if trained else 0
        yield {"kind": "started", "seconds": 0.0, "gpu": self.report.get("gpu")}
        yield {"kind": "snapshot", "step": step, "splats": None, "spz": self.data, "replayed": True,
               "seconds": (trained[-1].get("trainer_s") if trained else None) or 0.0}
        yield {"kind": "done", "seconds": self.report.get("trainer_s") or 0.0,
               "stats": self.report.get("stats") or {}}


class BundleUploader:
    """Uploads bundles in the background, newest first: `submit` returns at once, so training and
    cutting never wait for the network, and a bundle superseded before its turn is skipped (the
    room call only ever needs the newest). `close` waits for the last one. `upload(bundle)` does
    the work (`upload_bundle` with the Volume and the room's name)."""

    def __init__(self, upload: Callable[[dict], int], log: Callable[[str], None] = print) -> None:
        import threading

        self._upload, self._log = upload, log
        self._next: dict | None = None
        self._closed = False
        self._cv = threading.Condition()
        self.sent = self.skipped = 0
        self.errors: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="bundle-upload")
        self._thread.start()

    def submit(self, bundle: dict) -> None:
        with self._cv:
            if self._next is not None:
                self.skipped += 1
            self._next = bundle
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._next is None and not self._closed:
                    self._cv.wait()
                if self._next is None:
                    return
                bundle, self._next = self._next, None
            t = time.perf_counter()
            try:
                sent = self._upload(bundle)
                self.sent += 1
                self._log(f"    published {bundle.get('stage')}"
                          f"{'' if bundle.get('step') is None else ' step ' + str(bundle['step'])}"
                          f" ({sent / 1e6:.1f} MB, {time.perf_counter() - t:.1f} s): live within"
                          " ~10 s")
            except Exception as exc:  # noqa: BLE001 — the next bundle may well get through
                self.errors.append(f"{type(exc).__name__}: {exc}")
                self._log(f"    publishing failed ({type(exc).__name__}: {exc})")

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify()
        self._thread.join()


def upload_bundle(volume, name: str, out_dir: Path, bundle: dict) -> int:
    """Put one bundle on the hosted room call's Volume, under ``splats/<name>/``: the SPZ files
    it names first (one batch; files already there are skipped, their names are their content),
    then ``splat.json`` in a second batch, so a container never reads a bundle naming a file that
    is not there yet; then files no longer named are removed. Returns the bytes sent."""
    import io

    from litereality_agent.room_ops.roomcall.splats import SPLATS_DIR

    folder = f"/{SPLATS_DIR}/{name}"
    try:
        present = {Path(e.path).name for e in volume.listdir(folder)}
    except Exception:  # noqa: BLE001 — a first upload: nothing there yet
        present = set()
    wanted = [bundle["room"], *bundle["bodies"].values()]
    sent = 0
    missing = [f for f in wanted if f not in present]
    if missing:
        with volume.batch_upload(force=True) as batch:
            for file in missing:
                batch.put_file(out_dir / file, f"{folder}/{file}")
                sent += (out_dir / file).stat().st_size
    data = (out_dir / BUNDLE).read_bytes()
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(data), f"{folder}/{BUNDLE}")
    for stale in present - set(wanted) - {BUNDLE}:
        if stale.endswith(".spz"):
            try:
                volume.remove_file(f"{folder}/{stale}")
            except Exception:  # noqa: BLE001 — a leftover file costs only storage
                pass
    return sent + len(data)
