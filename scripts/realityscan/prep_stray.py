"""The Stray (atrium, recording 901a305fdf) frames RealityScan aligns: ~400 sharp frames spread
over the recording, the 29 fixed eval frames INCLUDED (so they get RealityScan poses) and marked.

Source: the pool import ~/LiteReality-scans/stray_atrium_pool (`import-stray --frames 800
--min-move 0.01 --min-turn 0.5`, models/capture/stray.py: the sharpest frame per window of the raw
recording, 1920 x 1440 JPEG, ARKit poses moved into atrium_1059's world). The eval frames were
picked from it (results/stray_quality/eval_frames.txt), so its poses are the ones score.py uses.
The other frames: from the pool frames more than 0.2 s from every eval frame (the scorer's
`--exclude-span 0.2` protocol), the sharpest of each of n - 29 equal runs of them.

    uv run --frozen python scripts/realityscan/prep_stray.py [--n 400] [--out ~/LiteReality-scans/rs_stray]

Writes <out>/rs_input/images/frame_XXXXX.jpg (the pool's files, pool names) and
<out>/rs_input/selection.json ({name: {"row", "time_s", "sharpness", "eval"}}).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
POOL = Path("~/LiteReality-scans/stray_atrium_pool").expanduser()
EVAL = REPO / "results/stray_quality/eval_frames.txt"
SPAN_S = 0.2


def eval_stems(path: Path = EVAL) -> dict[str, int]:
    out = {}
    for line in path.read_text().splitlines():
        tok = line.split("#", 1)[0].split()
        if len(tok) >= 2:
            out[tok[1]] = int(tok[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("~/LiteReality-scans/rs_stray"))
    a = ap.parse_args()
    ev = eval_stems()
    rows = []
    for m in sorted(POOL.glob("frame_*.json")):
        d = json.loads(m.read_text())
        rows.append({"name": m.stem + ".jpg", "row": int(d["stray"]["row"]),
                     "time_s": float(d["time"]), "sharpness": float(d["stray"]["sharpness"]),
                     "eval": m.stem in ev})
    t0 = rows[0]["time_s"]
    for r in rows:
        r["time_s"] = round(r["time_s"] - t0, 4)
    t_eval = np.array([r["time_s"] for r in rows if r["eval"]])
    assert len(t_eval) == len(ev), (len(t_eval), len(ev))
    cand = [r for r in rows if not r["eval"]
            and np.abs(t_eval - r["time_s"]).min() > SPAN_S]
    k = a.n - len(t_eval)
    # k runs of consecutive candidates (the pool is already one sharp frame per short window,
    # so equal runs are about equal stretches of the walk), the sharpest of each.
    edges = np.linspace(0, len(cand), k + 1).round().astype(int)
    picks = [max(range(lo, hi), key=lambda i: cand[i]["sharpness"])
             for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]
    chosen = sorted([r for r in rows if r["eval"]] + [cand[i] for i in picks],
                    key=lambda r: r["time_s"])
    out = a.out.expanduser() / "rs_input"
    img = out / "images"
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    for r in chosen:
        shutil.copy2(POOL / r["name"], img / r["name"])
    sel = {r["name"]: {k2: r[k2] for k2 in ("row", "time_s", "sharpness", "eval")} for r in chosen}
    (out / "selection.json").write_text(json.dumps(
        {"source": str(POOL), "eval_frames": str(EVAL), "exclude_span_s": SPAN_S,
         "candidates_outside_span": len(cand), "frames": sel}, indent=1))
    print(f"{len(chosen)} frames ({sum(r['eval'] for r in chosen)} eval, "
          f"{len(chosen) - len(t_eval)} train) of {len(rows)} pool frames -> {img}")


if __name__ == "__main__":
    main()
