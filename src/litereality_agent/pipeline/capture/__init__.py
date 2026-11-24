"""capture — a recording from another capture app, imported as a LiteReality scan.

    import_stray   a Stray Scanner recording (folder or .zip) -> a scan folder the photoreal
                   pipeline reads as it reads the LiteReality Scanner's (`models/capture/stray`)

The LiteReality Scanner saves ~51 photos a room; Stray Scanner keeps the whole 60 fps walk. This
picks the training set from it (the sharpest frame of each stretch, near-duplicate poses
dropped, `--frames` of them) and, with ``align_to``, lines the recording up with the room's
LiteReality Scanner scan (`models/capture/align`: yaw about gravity + a translation, refused when
the fit's residual is above `max_residual`, too little of the recording fits, or another heading
fits nearly as well), so the imported scan shares that scan's ARKit world:
its RoomPlan ``room.usdz`` (copied in), and so the physics scene already built from it under
``run/<room>/``.

What it writes, in ``out_dir``:

    frame_00000.jpg/.json, depth_00000.png, conf_00000.png    the chosen frames, in time order
    pointcloud.pcd           their confident depth fused on a 2 cm grid (ARKit world)
    scan_label.json          "Stray Scanner - <recording>" (the target's roomType kept)
    scene_alignment.json     the target's, with this cloud's bounds and the import's transform
    room.usdz, roomplan/     copied from ``align_to`` (what quickroom / splatroom look for)
    import_report.json       what was kept and why, sharpness, the alignment, timings
"""

from __future__ import annotations

import json
import shutil
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from litereality_agent.models.capture import align as aligner
from litereality_agent.models.capture import stray

REPORT = "import_report.json"
SCAN_FILES = ("frame_*.jpg", "frame_*.json", "depth_*.png", "conf_*.png")
USDZ_CANDIDATES = ("room.usdz", "roomplan/room.usdz")
LOW_INLIERS = 0.7         # below this the import warns


def find_room_usdz(scan_dir: Path) -> Path | None:
    return next((scan_dir / n for n in USDZ_CANDIDATES if (scan_dir / n).is_file()), None)


def scan_cloud(scan_dir: Path, voxel: float = stray.PCD_VOXEL) -> np.ndarray:
    """A LiteReality scan's surface in its ARKit world: its pointcloud.pcd and its frames'
    confident depth, fused on a `voxel` grid."""
    from litereality_agent.models.splat import rgbd

    scan_dir = Path(scan_dir)
    parts = []
    pcd = scan_dir / "pointcloud.pcd"
    if pcd.is_file():
        parts.append(stray.read_pcd(pcd))
    to_arkit = np.linalg.inv(rgbd.ARKIT_TO_Z_UP)
    for f in rgbd.load_frames(scan_dir):
        frame = stray.StrayFrame(f.index, f"{f.index:05d}", 0.0, to_arkit @ f.c2w, f.K, f.depth,
                                 f.conf)
        parts.append(stray.frame_points(frame, f.size))
    if not parts:
        raise FileNotFoundError(f"{scan_dir}: no pointcloud.pcd and no frames with depth")
    return stray.voxel_downsample(np.concatenate(parts).astype(np.float64), voxel)


def _clear(out_dir: Path) -> None:
    for pattern in SCAN_FILES:
        for p in out_dir.glob(pattern):
            p.unlink()
    for name in ("pointcloud.pcd", "scene_alignment.json", "scan_label.json", REPORT,
                 "room.usdz"):
        (out_dir / name).unlink(missing_ok=True)
    shutil.rmtree(out_dir / "roomplan", ignore_errors=True)
    # Anything cached from the old frames (PromptDA's dense depth) belongs to them.
    shutil.rmtree(out_dir / "depth_dense", ignore_errors=True)


def _bounds(points: np.ndarray) -> dict:
    if len(points) == 0:
        return {}
    lo, hi = points.min(0), points.max(0)
    return {"max": [round(float(v), 5) for v in hi], "min": [round(float(v), 5) for v in lo],
            "size": [round(float(v), 5) for v in hi - lo]}


def import_stray(recording: Path | str, out_dir: Path | str, *, align_to: Path | str | None = None,
                 frames: int = stray.TARGET_FRAMES,
                 max_residual: float = aligner.MAX_RESIDUAL,
                 min_inliers: float = aligner.MIN_INLIERS, overwrite: bool = False,
                 accept_ambiguous: bool = False, allow_frame_mismatch: bool = False,
                 min_move: float = stray.MIN_MOVE, min_turn_deg: float = stray.MIN_TURN_DEG,
                 exclude: Iterable[int] | None = None, exclude_span: float = 0.0,
                 picker: str = "laplacian", max_blur_px: float | None = stray.MAX_BLUR_PX,
                 window_grow: float = stray.BLUR_WINDOW_GROW, score_width: int | None = None,
                 log: Callable[[str], None] = print) -> dict:
    """Import a Stray Scanner recording into `out_dir` (see the module docstring). Returns the
    report (also written to `REPORT`). Raises ValueError, before anything is written, when:
    rgb.mp4's frames cannot be paired with odometry rows by their PTS (`stray.pair_video`;
    `allow_frame_mismatch` pairs them from the first anyway); the alignment's residual is above
    `max_residual` metres or fewer than `min_inliers` of the recording's points lie on the
    target's surface (a wrong room can fit some walls well); or a heading more than
    `aligner.AMBIGUITY_SEPARATION` degrees away fits the walls nearly as well
    (`Alignment.ambiguous`; `accept_ambiguous` takes the best anyway). The residual is not what
    tells a wrong heading from the right one: any converged fit has a low one.

    `min_move` / `min_turn_deg` are the near-duplicate thresholds (`stray.select_frames`).
    `exclude` names recording frames (odometry ``frame`` numbers, as `frame_json` keeps them in
    ``stray.row``) never to train on, with every frame within `exclude_span` seconds of one: an
    evaluation set held out of every import, so the views it scores stay unseen.

    `picker` "laplacian" (the default) keeps the sharpest frame of each window by the Laplacian at
    `stray.SCORE_WIDTH`; "blur" scores every frame at full resolution (`score_width`, default the
    video's own), estimates its motion blur in pixels at 1920 relative to the frames around it
    (`stray.blur_estimate`) and keeps the least blurred frame of each window or `window_grow`
    windows either side, never one blurred by more than `max_blur_px` (None: no limit)."""
    if picker not in ("laplacian", "blur"):
        raise ValueError(f"picker must be 'laplacian' or 'blur', not {picker!r}")
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    out = Path(out_dir).expanduser()
    target = Path(align_to).expanduser() if align_to else None
    if out.exists() and any(out.glob("frame_*.json")):
        if not overwrite:
            raise ValueError(f"{out} already holds a scan (frame_*.json); pick another --out or "
                             "pass --overwrite")
    if target is not None:
        if not target.is_dir():
            raise FileNotFoundError(f"--align-to {target}: no such scan folder")
        if target.resolve() == out.resolve():
            raise ValueError("--out must not be the scan it is aligned to")

    with stray.open_recording(recording) as rec:
        warnings = list(rec.warnings)
        log(f"recording {rec.name}: {len(rec.frames)} poses, video {rec.size[0]}x{rec.size[1]}")
        t = time.perf_counter()
        width = int(score_width or (stray.FULL_SCORE_WIDTH if picker == "blur"
                                    else stray.SCORE_WIDTH))
        scan = stray.scan_video(
            rec.video, rec.size, width=width, progress=lambda n: log(f"  scored {n} frames ..."))
        scores = scan.scores
        timings["score_s"] = time.perf_counter() - t
        pairing = stray.pair_video(scan.pts, [f.row for f in rec.frames],
                                   [f.timestamp for f in rec.frames], n_video_hint=len(scores),
                                   allow_mismatch=allow_frame_mismatch)
        if pairing.note:
            warnings.append(pairing.note)
            log(f"  {pairing.note}")
        listed = [f for f in rec.frames if f.row in pairing.video_of_row]
        if not listed:
            raise ValueError(f"{rec.name}: no odometry row has a video frame")
        video_of = [pairing.video_of_row[f.row] for f in listed]
        times = np.array([f.timestamp for f in listed])
        poses = np.stack([f.c2w for f in listed])
        sharp = scores[video_of]
        has_depth = np.array([f.depth is not None for f in listed])
        tracking = stray.tracking_ok(times, poses)
        held = excluded_mask(listed, times, exclude, exclude_span)
        if exclude and not held.any():
            warnings.append("--exclude named no frame of this recording")
        scored_width = min(width, rec.size[0])
        blur = (stray.blur_estimate(times, sharp, scored_width=scored_width)
                if picker == "blur" else None)
        sel = stray.select_frames(times, poses, sharp, has_depth & tracking & ~held,
                                  target=frames, min_move=min_move, min_turn_deg=min_turn_deg,
                                  blur_px=blur, max_blur_px=max_blur_px if blur is not None
                                  else None, window_grow=window_grow)
        sel.stats["score_width"] = scored_width
        sel.stats.update({"no_depth": int((~has_depth).sum()),
                          "tracking_dropped": int((~tracking).sum()),
                          "min_move": min_move, "min_turn_deg": min_turn_deg})
        if exclude:
            sel.stats.update({"excluded_named": len(set(int(r) for r in exclude)),
                              "excluded_span_s": exclude_span,
                              "excluded_frames": int(held.sum())})
        if not len(sel.rows):
            raise ValueError(f"{rec.name}: no usable frame (with depth, steady tracking)")
        chosen = [listed[i] for i in sel.rows]
        s = sel.stats["sharpness"]
        log(f"  {len(scores)} frames scored in {timings['score_s']:.1f} s; kept {len(chosen)} "
            f"(target {frames}; {sel.stats['duplicates_dropped']} near-duplicates, "
            f"{sel.stats['blurry_dropped']} blurry, {sel.stats['tracking_dropped']} around "
            f"tracking jumps); sharpness median {s['kept_median']} kept vs {s['all_median']} all")

        transform = np.eye(4)
        alignment = None
        if target is not None:
            t = time.perf_counter()
            log(f"aligning to {target} ...")
            source_cloud = stray.fuse_points(chosen, rec.size)
            alignment = aligner.align(source_cloud, scan_cloud(target), log=log)
            timings["align_s"] = time.perf_counter() - t
            log(f"  yaw {alignment.yaw_deg:.2f} deg, translation "
                f"{np.round(alignment.translation, 3).tolist()} m, residual "
                f"{alignment.residual * 100:.1f} cm (point-to-plane median), "
                f"{alignment.inliers:.0%} of the recording on the scan's surface")
            log(f"  heading margin: {alignment.describe_margin()}")
            if not (alignment.residual <= max_residual and alignment.inliers >= min_inliers):
                raise ValueError(
                    f"alignment to {target} is not trustworthy: residual "
                    f"{alignment.residual * 100:.1f} cm (limit {max_residual * 100:.1f}), "
                    f"{alignment.inliers:.0%} of the recording within "
                    f"{aligner.FINAL_MATCH * 100:.0f} cm of its surface (need "
                    f"{min_inliers:.0%}). Is it the same room? Nothing was written; "
                    "--max-residual / --min-inliers relax the limits")
            if alignment.ambiguous:
                msg = (f"alignment to {target} is ambiguous: {alignment.describe_margin()}. A "
                       "rectangular room, little shared surface or rearranged furniture can "
                       "make a turned fit look as good (its residual is low too).")
                if not accept_ambiguous:
                    raise ValueError(msg + " Nothing was written; check the room, record more of "
                                     "it, or pass --accept-ambiguous to take the best fit")
                warnings.append(msg + " Taken anyway (--accept-ambiguous): check it in the "
                                "viewer")
            if alignment.inliers < LOW_INLIERS:
                warnings.append(f"only {alignment.inliers:.0%} of the recording lies on "
                                f"{target.name}'s surface: check the room in the viewer")
            transform = alignment.transform

        if overwrite and out.exists():
            _clear(out)
        out.mkdir(parents=True, exist_ok=True)
        t = time.perf_counter()
        speed, turn = stray.velocities(times, poses)
        position = {f.row: i for i, f in enumerate(listed)}
        by_video = {pairing.video_of_row[f.row]: (j, f) for j, f in enumerate(chosen)}
        pending: deque = deque()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for k, rgb in stray.decode_frames(rec.video, sorted(by_video), rec.size):
                j, f = by_video[k]
                i = position[f.row]
                c2w = transform @ f.c2w
                stem = f"{j:05d}"
                stray.write_json(out / f"frame_{stem}.json", stray.frame_json(
                    j, c2w, f.K, f.timestamp, speed[i], turn[i],
                    {"recording": rec.name, "frame": f.name, "row": int(f.row),
                     "video_frame": int(k),
                     "sharpness": round(float(sharp[i]), 3),
                     **({"blur_px1920": round(float(blur[i]), 2)} if blur is not None else {})}))
                stray.write_depth(f, out / f"depth_{stem}.png")
                if f.conf is not None:
                    shutil.copyfile(f.conf, out / f"conf_{stem}.png")
                pending.append(pool.submit(_save_jpeg, rgb, out / f"frame_{stem}.jpg"))
                while len(pending) > 8:
                    pending.popleft().result()
            for fut in pending:
                fut.result()
        timings["frames_s"] = time.perf_counter() - t

        t = time.perf_counter()
        cloud = stray.fuse_points(chosen, rec.size, transform=transform)
        stray.write_pcd(out / "pointcloud.pcd", cloud)
        timings["pointcloud_s"] = time.perf_counter() - t
        name = rec.name

    label = stray.scan_label(name, name)
    alignment_doc = {"coordinateSystem": "ARKit world coordinates shared by camera poses and "
                                         "pointcloud.pcd",
                     "unit": "meters"}
    if target is not None:
        usdz = find_room_usdz(target)
        if usdz is None:
            warnings.append(f"{target} has no RoomPlan room.usdz: quickroom / splatroom need one")
        if (target / "room.usdz").is_file():
            shutil.copyfile(target / "room.usdz", out / "room.usdz")
        if (target / "roomplan").is_dir():
            shutil.copytree(target / "roomplan", out / "roomplan", dirs_exist_ok=True)
        with_label = target / "scan_label.json"
        if with_label.is_file():
            try:
                theirs = json.loads(with_label.read_text(encoding="utf-8"))
                if isinstance(theirs, dict) and theirs.get("roomType"):
                    label["roomType"] = theirs["roomType"]
            except (OSError, ValueError):
                pass
        theirs_alignment = target / "scene_alignment.json"
        if theirs_alignment.is_file():
            try:
                loaded = json.loads(theirs_alignment.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    alignment_doc = loaded
            except (OSError, ValueError):
                pass
        alignment_doc["note"] = (
            f"Imported from the Stray Scanner recording {name} and moved into {target.name}'s "
            "ARKit world (yaw about Y + translation, `strayImport.transform`), so the copied "
            "RoomPlan room.usdz, the camera poses and pointcloud.pcd share one frame.")
    else:
        alignment_doc["note"] = (f"The Stray Scanner recording {name}'s own ARKit world (no "
                                 "RoomPlan scan to line up with: --align-to adds one).")
    alignment_doc["pointCloudBounds"] = _bounds(cloud)
    alignment_doc["strayImport"] = {
        "recording": str(recording), "alignedTo": str(target) if target else None,
        "transform": [[round(float(v), 6) for v in r] for r in transform],
        "residual_m": round(alignment.residual, 4) if alignment else None,
        "heading_margin": round(alignment.margin, 3) if alignment else None}
    stray.write_json(out / "scan_label.json", label)
    stray.write_json(out / "scene_alignment.json", alignment_doc)

    timings["total_s"] = time.perf_counter() - t0
    report = {"recording": str(recording), "out": str(out),
              "video": {"size": list(rec.size), "frames": int(len(scores)),
                        "odometry_rows": int(max(f.row for f in rec.frames) + 1),
                        "rows_without_frame": {"start": pairing.rows_before,
                                               "end": pairing.rows_after}},
              "selection": sel.stats, "frames": len(chosen),
              "alignment": alignment.summary() if alignment else None,
              "pointcloud_points": int(len(cloud)), "warnings": warnings,
              "seconds": {k: round(v, 2) for k, v in timings.items()}}
    stray.write_json(out / REPORT, report)
    for w in warnings:
        log(f"  warning: {w}")
    return report


def excluded_mask(listed, times: np.ndarray, exclude: Iterable[int] | None,
                  span: float = 0.0) -> np.ndarray:
    """Which of `listed` (StrayFrames, time order) are held out: the frames `exclude` names by
    odometry row, and every frame within `span` seconds of one."""
    held = np.zeros(len(listed), bool)
    rows = {int(r) for r in exclude or ()}
    if not rows:
        return held
    named = np.array([f.row in rows for f in listed])
    for t in np.asarray(times, float)[named]:
        held |= np.abs(times - t) <= span
    return held | named


def read_frame_list(path: Path | str) -> list[int]:
    """Odometry rows from a text file: the first whitespace-separated token of each line that is
    not blank or a ``#`` comment (``001234`` or ``1234``)."""
    rows = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        token = line.split("#", 1)[0].split()
        if token:
            rows.append(int(token[0]))
    return rows


def _save_jpeg(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image

    tmp = path.with_suffix(".tmp.jpg")
    Image.fromarray(rgb).save(tmp, quality=stray.JPEG_QUALITY)
    tmp.replace(path)
