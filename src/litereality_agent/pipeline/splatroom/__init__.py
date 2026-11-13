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
from litereality_agent.pipeline.splatroom.split import Split, split

DEFAULT_DIR = "splat"
BUNDLE = "splat.json"
REPORT = "report.json"
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
          on_bundle: Callable[[dict], None] | None = None) -> dict:
    """The instant room, then (with a `trainer`) each trained snapshot, each written as a bundle
    the moment it exists. Returns the report (also written to `REPORT`)."""
    t0 = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    frames = rgbd.load_frames(scan_dir)
    if not frames:
        raise ValueError(f"{scan_dir}: no frames with a pose, a photo and depth")
    report: dict = {"scan": str(scan_dir), "scene": str(scene_xml), "frames": len(frames),
                    "bundles": []}

    def publish(splats: spz.Splats, stage: str, step: int | None, extra: dict) -> None:
        t = time.perf_counter()
        parts = split(splats, model)
        bundle = write_bundle(out_dir, parts, stage=stage, step=step)
        entry = {"stage": stage, "step": step, "at": round(time.perf_counter() - t0, 2),
                 "split_write_s": round(time.perf_counter() - t, 2), "splats": len(splats),
                 "bodies": len(parts.bodies), "mb": round(bundle["bytes"] / 1e6, 2), **extra}
        report["bundles"].append(entry)
        log(f"  {stage}{'' if step is None else f' step {step}'}: {len(splats):,} splats, "
            f"{len(parts.bodies)} objects cut out, {entry['mb']} MB, at {entry['at']} s")
        if on_bundle:
            on_bundle(bundle)

    instant = rgbd.instant_splats(frames, voxel=INSTANT_VOXEL)
    publish(instant, "instant", None, {})
    report["instant_s"] = report["bundles"][-1]["at"]

    if trainer is not None and not instant_only:
        t = time.perf_counter()
        tar = splat_dataset.build(frames).tar()
        report["dataset_s"] = round(time.perf_counter() - t, 2)
        report["dataset_mb"] = round(len(tar) / 1e6, 1)
        log(f"  training set: {len(frames)} photos, {report['dataset_mb']} MB; "
            f"training on Modal ({trainer.name})")
        for event in trainer.stream(tar, options):
            kind = event["kind"]
            if kind == "started":
                report["trainer_started_s"] = round(time.perf_counter() - t0, 2)
                report["gpu"] = event.get("gpu")
                log(f"  trainer up on {event.get('gpu')} at {report['trainer_started_s']} s")
            elif kind == "snapshot":
                splats = spz.decode(event["spz"])
                publish(splats, "trained", int(event["step"]),
                        {"trainer_s": round(float(event["seconds"]), 1)})
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
