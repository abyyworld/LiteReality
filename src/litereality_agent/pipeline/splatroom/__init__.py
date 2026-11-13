"""splatroom — the scanned room as photoreal Gaussian splats, growing live, cut to the physics.

    stage "instant"  the scan's photos + LiDAR as fused surfels: no GPU, ~8 s (models/splat/rgbd)
    stage "trained"  gsplat's trainer on Modal, a snapshot every few hundred steps, each sharper

Every stage is split (`split.split`) against the room's MuJoCo scene: the static room, and one
splat per movable body in that body's own frame, so the browser hangs each object's splat under
the body the server moves — grab the real chair and the real chair moves.

What it writes, in ``run/<scan>/splat/`` (`DEFAULT_DIR`):

    splat.json                 the current bundle: {"stage", "step", "room": file,
                               "bodies": {name: file}, "counts", "created"}, rewritten atomically
    room-<hash>.spz            files are named by their content, so a browser caches them forever
    body-<name>-<hash>.spz     and a bundle never points at a half-written file
    report.json                timings, splat counts and the trainer's held-out PSNR/SSIM/LPIPS

    litereality splatroom <scan> [--scene run/<scan>/objects/scene.xml] [--instant-only]
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

from litereality_agent.models.splat import dataset as splat_dataset
from litereality_agent.models.splat import rgbd, spz
from litereality_agent.pipeline.splatroom.split import Split, corrected_scene, split

DEFAULT_DIR = "splat"
BUNDLE = "splat.json"
REPORT = "report.json"
RAW = "raw.spz.bin"            # the latest snapshot uncut (not a bundle file: never served)
INSTANT_VOXEL = 0.016          # ~640k splats, ~6 MB for tea_room: on screen in seconds
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _name(prefix: str, data: bytes) -> str:
    return f"{prefix}-{hashlib.sha256(data).hexdigest()[:16]}.spz"


def write_bundle(out_dir: Path, parts: Split, *, stage: str, step: int | None = None,
                 antialiased: bool = True) -> dict:
    """Write one bundle (files first, then `splat.json` atomically) and drop older files."""
    out_dir.mkdir(parents=True, exist_ok=True)
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
    tmp = out_dir / (BUNDLE + ".tmp")
    tmp.write_text(json.dumps(bundle, indent=1))
    os.replace(tmp, out_dir / BUNDLE)
    for old in out_dir.glob("*.spz"):
        if old.name not in keep:
            old.unlink(missing_ok=True)
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
          move_bodies: bool = True) -> dict:
    """The instant room, then (with a `trainer`) each trained snapshot, each written as a bundle
    the moment it exists. Returns the report (also written to `REPORT`).

    The first split snaps the free bodies onto the splat; every later one reuses those
    corrections, and `corrected_scene` writes the physics with the bodies moved to match
    (``scene_splat.xml`` beside `scene_xml`: serve that one). With `segment` (the SAM 2 function,
    `models.splat.remote.segment_call`), each body's outline in the photos is fetched while the GPU trains, and
    every bundle after it arrives is cut by the photos' masks (`masks`)."""
    import threading

    from litereality_agent.pipeline.splatroom import masks as object_masks

    t0 = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    frames = rgbd.load_frames(scan_dir)
    if not frames:
        raise ValueError(f"{scan_dir}: no frames with a pose, a photo and depth")
    report: dict = {"scan": str(scan_dir), "scene": str(scene_xml), "frames": len(frames),
                    "bundles": []}

    state: dict = {"corrections": None, "masks": None}

    def publish(splats: spz.Splats, stage: str, step: int | None, extra: dict) -> None:
        t = time.perf_counter()
        # Before the masks: boxes only, and no snapping when masks are on their way (the photos
        # will say where each object is). The first split with masks fixes the corrections.
        waiting = segment is not None and state["masks"] is None
        parts = split(splats, model, corrections=state["corrections"], masks=state["masks"],
                      snap_bodies=not waiting, move_bodies=move_bodies)
        if state["corrections"] is None and not waiting:
            state["corrections"] = parts.corrections
            fixed = corrected_scene(Path(scene_xml), parts.corrections or {})
            report["corrections"] = parts.corrections
            report["corrected_scene"] = str(fixed) if fixed else None
            if fixed:
                log(f"  {len(parts.corrections)} object(s) moved to where the "
                    f"{'photos' if state['masks'] is not None else 'splat'} show them: {fixed}")
                if on_scene:
                    on_scene(fixed)
        bundle = write_bundle(out_dir, parts, stage=stage, step=step)
        if stage == "trained" and not extra.get("replayed"):
            # The trained snapshot, uncut, for evaluation and for cutting again (`SavedTrainer`).
            raw = out_dir / RAW
            raw.with_suffix(".tmp").write_bytes(spz.encode(splats))
            os.replace(raw.with_suffix(".tmp"), raw)
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

    instant = rgbd.instant_splats(frames, voxel=INSTANT_VOXEL)
    publish(instant, "instant", None, {})
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
                from litereality_agent.pipeline.splatroom.split import _geom_boxes

                static = _geom_boxes(model, data, [b for b in range(1, model.nbody)
                                                   if b not in set(bodies.values())])
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
        if trainer is None or instant_only:
            mask_thread.join()
            if state["masks"] is None:
                segment = None                     # failed: snap by the boxes after all
            publish(instant, "instant", None, {})

    if trainer is not None and not instant_only:
        t = time.perf_counter()
        tar = b"" if isinstance(trainer, SavedTrainer) else splat_dataset.build(frames).tar()
        report["dataset_s"] = round(time.perf_counter() - t, 2)
        report["dataset_mb"] = round(len(tar) / 1e6, 1)
        log(f"  training set: {len(frames)} photos, {report['dataset_mb']} MB; "
            f"training on Modal ({trainer.name})")
        for event in trainer.stream(tar, options):
            if mask_thread is not None and event["kind"] == "snapshot":
                mask_thread.join()              # cut every trained room by the photos
                mask_thread = None
                if state["masks"] is None:
                    segment = None
                # Where the objects really are comes from the LiDAR room, whose splats lie on
                # the surfaces; trained splats are free to float. It is published again, cut by
                # the photos, just before the trained one.
                publish(instant, "instant", None, {})
            kind = event["kind"]
            if kind == "started":
                report["trainer_started_s"] = round(time.perf_counter() - t0, 2)
                report["gpu"] = event.get("gpu")
                log(f"  trainer up on {event.get('gpu')} at {report['trainer_started_s']} s")
            elif kind == "snapshot":
                splats = spz.decode(event["spz"])
                publish(splats, "trained", int(event["step"]),
                        {"trainer_s": round(float(event["seconds"]), 1),
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
    report["total_s"] = round(time.perf_counter() - t0, 2)
    (out_dir / REPORT).write_text(json.dumps(report, indent=1))
    return report


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
        yield {"kind": "started", "seconds": 0.0, "gpu": None}
        yield {"kind": "snapshot", "step": step, "seconds": 0.0, "splats": None,
               "spz": self.data, "replayed": True}
        yield {"kind": "done", "seconds": 0.0, "stats": self.report.get("stats") or {}}


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
