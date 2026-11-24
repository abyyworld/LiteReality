"""A second evaluation set for the atrium Stray comparison: frames at RANDOM times (run once).

The first set (eval_frames.txt, pick_eval.py) took the sharpest pool frame of each stretch, and
the pool import had itself already kept the sharpest frame of every ~3: its views are the
recording's sharpest moments, sharper than what the trainer saw (results/diagnosis/blur). This
set draws N times uniformly at random over the recording (seeded, at least MIN_GAP_S apart) and
takes the frame nearest each time that has depth and steady tracking, whatever its blur, so the
score is what a viewer sees on the walk's typical frame.

It writes, beside eval_frames.txt and in its format:

    eval_frames_random.txt    odometry row, the frame's name in POOL_OUT, time (s), est. blur
                              (px at 1920, `stray.blur_estimate` at full resolution)
    eval_frames_all.txt       both sets' rows: what a NEW training import excludes
                              (`import-stray --exclude results/stray_quality/eval_frames_all.txt
                              --exclude-span 0.2`)

and the frames themselves into POOL_OUT (photo, pose, depth, confidence), posed in the first
pool's world (its `strayImport.transform`, copied with its scene_alignment.json), so score.py
maps them into a variant's world exactly as it maps the first set. Training scans imported before
this set existed may hold frames near these: score.py reports how many and scores the unseen ones
apart.

    uv run --frozen python results/stray_quality/pick_eval_random.py [N] [SEED]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RECORDING = Path("~/LiteReality-scans/stray_atrium_raw/901a305fdf").expanduser()
POOL = Path("~/LiteReality-scans/stray_atrium_pool").expanduser()
POOL_OUT = Path("~/LiteReality-scans/stray_atrium_pool_random").expanduser()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261124
MIN_GAP_S = 1.0            # two eval frames at least this far apart in time
EDGE_S = 0.5               # and not in the recording's first / last half second


def draw_times(t0: float, t1: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """`n` times uniform on [t0, t1], at least MIN_GAP_S apart (rejection, then sorted)."""
    out: list[float] = []
    for _ in range(100_000):
        if len(out) == n:
            break
        t = float(rng.uniform(t0, t1))
        if all(abs(t - u) >= MIN_GAP_S for u in out):
            out.append(t)
    if len(out) < n:
        raise ValueError(f"cannot place {n} times {MIN_GAP_S} s apart in {t1 - t0:.1f} s")
    return np.sort(np.asarray(out))


def main() -> None:
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from litereality_agent.models.capture import stray
    from litereality_agent.pipeline.capture import _save_jpeg, read_frame_list

    rec = stray.read_recording(RECORDING)
    scan = stray.scan_video(rec.video, rec.size, width=stray.FULL_SCORE_WIDTH,
                            progress=lambda k: print(f"  scored {k} frames", flush=True))
    pairing = stray.pair_video(scan.pts, [f.row for f in rec.frames],
                               [f.timestamp for f in rec.frames], n_video_hint=len(scan.scores))
    listed = [f for f in rec.frames if f.row in pairing.video_of_row]
    video_of = np.array([pairing.video_of_row[f.row] for f in listed])
    times = np.array([f.timestamp for f in listed])
    poses = np.stack([f.c2w for f in listed])
    blur = stray.blur_estimate(times, scan.scores[video_of],
                               scored_width=min(stray.FULL_SCORE_WIDTH, rec.size[0]))
    usable = np.array([f.depth is not None for f in listed]) & stray.tracking_ok(times, poses)
    rng = np.random.default_rng(SEED)
    wanted = draw_times(times[0] + EDGE_S, times[-1] - EDGE_S, N, rng)
    ok = np.flatnonzero(usable)
    picks = sorted({int(ok[np.argmin(np.abs(times[ok] - t))]) for t in wanted})

    T = np.asarray(json.loads((POOL / "scene_alignment.json").read_text())
                   ["strayImport"]["transform"], float)
    if POOL_OUT.exists():
        shutil.rmtree(POOL_OUT)
    POOL_OUT.mkdir(parents=True)
    shutil.copyfile(POOL / "scene_alignment.json", POOL_OUT / "scene_alignment.json")
    speed, turn = stray.velocities(times, poses)
    by_video = {int(video_of[i]): (j, i) for j, i in enumerate(picks)}
    stems = {}
    for k, rgb in stray.decode_frames(rec.video, sorted(by_video), rec.size):
        j, i = by_video[k]
        f = listed[i]
        stem = f"{j:05d}"
        stray.write_json(POOL_OUT / f"frame_{stem}.json", stray.frame_json(
            j, T @ f.c2w, f.K, f.timestamp, speed[i], turn[i],
            {"recording": rec.name, "frame": f.name, "row": int(f.row), "video_frame": int(k),
             "sharpness": round(float(scan.scores[k]), 3),
             "blur_px1920": round(float(blur[i]), 2), "picked": "random time"}))
        stray.write_depth(f, POOL_OUT / f"depth_{stem}.png")
        if f.conf is not None:
            shutil.copyfile(f.conf, POOL_OUT / f"conf_{stem}.png")
        _save_jpeg(rgb, POOL_OUT / f"frame_{stem}.jpg")
        stems[i] = f"frame_{stem}"

    t0 = times[0]
    lines = [f"# {len(picks)} evaluation frames at RANDOM times (seed {SEED}, >= {MIN_GAP_S} s "
             f"apart) of {RECORDING.name}, in {POOL_OUT} (pick_eval_random.py).",
             "# Not chosen by sharpness: blur_px is each frame's estimated motion blur at 1920.",
             "# New train scans: import-stray ... --exclude results/stray_quality/"
             "eval_frames_all.txt --exclude-span 0.2",
             "# row     pool_frame    t_s     blur_px"]
    for i in picks:
        lines.append(f"{int(listed[i].row):06d}  {stems[i]}  {times[i] - t0:6.2f}  "
                     f"{blur[i]:7.2f}")
    (HERE / "eval_frames_random.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = read_frame_list(HERE / "eval_frames.txt")
    both = sorted(set(first) | {int(listed[i].row) for i in picks})
    (HERE / "eval_frames_all.txt").write_text(
        "# Both evaluation sets (eval_frames.txt, eval_frames_random.txt): odometry rows never "
        "to train on.\n" + "".join(f"{r:06d}\n" for r in both), encoding="utf-8")
    b = blur[picks]
    print(f"{len(picks)} random-time eval frames -> {POOL_OUT}; blur px median {np.median(b):.2f}, "
          f"p90 {np.percentile(b, 90):.2f} (recording median {np.median(blur):.2f})")


if __name__ == "__main__":
    main()
