"""Pick the fixed evaluation frames for the atrium Stray splat comparison (run once).

Reads the pool import (every candidate frame: `import-stray --frames 800 --min-move 0.01
--min-turn 0.5`), cuts the recording into N equal stretches of time and takes the sharpest pool
frame of each stretch's middle half (so two picks are never adjacent). Writes eval_frames.txt:
odometry row (what `import-stray --exclude` reads), the pool frame it is, time, sharpness.

    uv run --frozen python results/stray_quality/pick_eval.py [POOL_SCAN] [N]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POOL = Path(sys.argv[1] if len(sys.argv) > 1 else "~/LiteReality-scans/stray_atrium_pool").expanduser()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def main() -> None:
    metas = sorted(POOL.glob("frame_*.json"))
    rows = []
    for m in metas:
        d = json.loads(m.read_text())
        rows.append((m.stem, int(d["stray"]["row"]), float(d["time"]), float(d["stray"]["sharpness"])))
    t0 = rows[0][2]
    times = np.array([r[2] - t0 for r in rows])
    edges = np.linspace(0, times[-1], N + 1)
    picks = []
    for a, b in zip(edges[:-1], edges[1:]):
        q = (b - a) / 4
        inside = [i for i, t in enumerate(times) if a + q <= t <= b - q]
        if not inside:
            continue
        picks.append(max(inside, key=lambda i: rows[i][3]))
    out = HERE / "eval_frames.txt"
    lines = [f"# {len(picks)} evaluation frames of {POOL} (recording 901a305fdf), never trained on.",
             "# Train scans: import-stray ... --exclude results/stray_quality/eval_frames.txt "
             "--exclude-span 0.2",
             "# row     pool_frame    t_s     sharpness"]
    for i in picks:
        stem, row, _t, sharp = rows[i]
        lines.append(f"{row:06d}  {stem}  {times[i]:6.2f}  {sharp:8.1f}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out, len(picks))


if __name__ == "__main__":
    main()
