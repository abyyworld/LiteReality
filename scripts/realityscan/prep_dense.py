"""DENSE frames for RealityScan: (nearly) every frame of a walk, so consecutive frames overlap
heavily and the alignment can chain through the feature-poor stretches that split the 400-frame
sets into 9-25 components.

  stray  the Stray recording 901a305fdf (rgb.mp4 + ARKit odometry): every --step-th odometry row
         at 1920 x 1440, the 29 eval rows ALWAYS included (and marked), frames at ARKit tracking
         jumps dropped, then the blurriest --drop share of the rest (sharpness relative to its
         +-1 s neighbourhood, so a plain wall is not mistaken for blur). Names r<row>.jpg (the
         odometry row) and r<row>_eval.jpg for eval frames.
  video  the iPhone Camera clip atrium_video.MOV (4K HEVC HLG, -90 display rotation): every
         --step-th frame at 1080 x 1920 portrait (display rotation applied, HLG read as its
         SDR-compatible signal, as `import-video` reads it), blurriest --drop share dropped.
         Names v<frame>.jpg (the decoded-stream index).

Both reuse models/capture's own decoders (stray.scan_video / decode_frames, video.probe /
scan_video / decode_frames) and write <out>/images/*.jpg + <out>/selection.json.

    uv run --frozen python scripts/realityscan/prep_dense.py stray --step 2 --out ~/LiteReality-scans/rs_dense_stray
    uv run --frozen python scripts/realityscan/prep_dense.py video --step 2 --out ~/LiteReality-scans/rs_dense_video
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
RAW = Path("~/LiteReality-scans/stray_atrium_raw/901a305fdf").expanduser()
VIDEO = Path("~/LiteReality-scans/atrium_video.MOV").expanduser()
EVAL = REPO / "results/stray_quality/eval_frames.txt"


def eval_rows(path: Path = EVAL) -> dict[int, str]:
    out = {}
    for line in path.read_text().splitlines():
        tok = line.split("#", 1)[0].split()
        if len(tok) >= 2:
            out[int(tok[0])] = tok[1]
    return out


def relative_sharpness(times: np.ndarray, sharp: np.ndarray, span: float = 1.0) -> np.ndarray:
    """sharpness / the 90th percentile within +-span s: blur, not texture."""
    lo = np.searchsorted(times, times - span, side="left")
    hi = np.searchsorted(times, times + span, side="right")
    ref = np.array([np.percentile(sharp[a:b], 90) for a, b in zip(lo, hi)])
    return sharp / np.maximum(ref, 1e-6)


def _fresh(out: Path) -> Path:
    img = out / "images"
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    return img


def prep_stray(a) -> None:
    from PIL import Image

    from litereality_agent.models.capture import stray as S

    rec = S.read_recording(RAW)
    scan = S.scan_video(rec.video, rec.size)
    rows = np.array([f.row for f in rec.frames])
    times = np.array([f.timestamp for f in rec.frames])
    poses = np.stack([f.c2w for f in rec.frames])
    pairing = S.pair_video(scan.pts, rows, times, n_video_hint=len(scan.scores))
    ok_track = S.tracking_ok(times, poses)
    sharp = np.array([scan.scores[pairing.video_of_row[r]] if r in pairing.video_of_row
                      else np.nan for r in rows])
    has_video = np.isfinite(sharp)
    rel = np.full(len(rows), np.nan)
    rel[has_video] = relative_sharpness(times[has_video], sharp[has_video])
    ev = eval_rows()
    is_eval = np.isin(rows, list(ev))
    assert is_eval.sum() == len(ev), (is_eval.sum(), len(ev))
    cand = (rows % a.step == 0) & has_video & ok_track & ~is_eval
    thr = np.nanpercentile(rel[cand], 100 * a.drop)
    keep = cand & (rel > thr)
    chosen = np.flatnonzero(keep | is_eval)
    print(f"{len(rows)} rows; step {a.step}: {int(((rows % a.step == 0)).sum())}; tracking-jump "
          f"rows {int((~ok_track).sum())}; blur cut rel<{thr:.3f} drops "
          f"{int(cand.sum() - (cand & keep).sum())}; kept {len(chosen)} ({int(is_eval.sum())} eval)")
    out = a.out.expanduser()
    img = _fresh(out)
    want = {pairing.video_of_row[int(rows[i])]: i for i in chosen}
    frames = {}
    t0 = times[0]
    for k, rgb in S.decode_frames(rec.video, list(want), rec.size):
        i = want[k]
        r = int(rows[i])
        name = f"r{r:06d}{'_eval' if is_eval[i] else ''}.jpg"
        Image.fromarray(rgb).save(img / name, quality=a.quality)
        frames[name] = {"row": r, "video_frame": int(k), "time_s": round(float(times[i] - t0), 4),
                        "timestamp": float(times[i]), "sharpness": round(float(sharp[i]), 2),
                        "rel_sharpness": round(float(rel[i]), 3), "eval": bool(is_eval[i]),
                        "pool_frame": ev.get(r)}
    (out / "selection.json").write_text(json.dumps(
        {"source": str(RAW), "step": a.step, "drop": a.drop, "rel_threshold": float(thr),
         "pairing_note": pairing.note, "tracking_jump_rows": int((~ok_track).sum()),
         "eval_frames": str(EVAL), "frames": dict(sorted(frames.items()))}, indent=1))
    print(f"{len(frames)} frames -> {img}")


def prep_video(a) -> None:
    from PIL import Image

    from litereality_agent.models.capture import video as V

    info = V.probe(VIDEO)
    print(info.summary())
    scan = V.scan_video(info)
    n = len(scan.scores)
    rel = relative_sharpness(scan.times, scan.scores)
    cand = np.arange(n) % a.step == 0
    thr = np.percentile(rel[cand], 100 * a.drop)
    chosen = np.flatnonzero(cand & (rel > thr))
    size = V.scaled_size(info.size, 1920)
    print(f"{n} frames; step {a.step}: {int(cand.sum())}; blur cut drops "
          f"{int(cand.sum() - len(chosen))}; kept {len(chosen)} at {size}")
    out = a.out.expanduser()
    img = _fresh(out)
    frames = {}
    for k, rgb in V.decode_frames(info, chosen, size):
        name = f"v{k:05d}.jpg"
        Image.fromarray(rgb).save(img / name, quality=a.quality)
        frames[name] = {"video_frame": int(k), "time_s": round(float(scan.times[k]), 4),
                        "sharpness": round(float(scan.scores[k]), 2),
                        "rel_sharpness": round(float(rel[k]), 3)}
    (out / "selection.json").write_text(json.dumps(
        {"video": str(VIDEO), "info": info.summary(), "photo_size": list(size), "step": a.step,
         "drop": a.drop, "rel_threshold": float(thr), "frames": dict(sorted(frames.items()))},
        indent=1))
    print(f"{len(frames)} frames -> {img}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["stray", "video"])
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.10)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    (prep_stray if a.which == "stray" else prep_video)(a)


if __name__ == "__main__":
    main()
