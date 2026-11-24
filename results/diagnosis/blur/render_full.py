"""Render the v2 uncut splat (raw.spz.bin) at FULL photo resolution (1920x1440, no box filter) at a
few eval views and at the nearest TRAINING views, with the same Spark 2.3.1 / SwiftShader page the
scorer uses, and save render + photo PNGs so their sharpness can be compared at the resolution a
viewer sees (the scorer only looks at 480x360).

    uv run --frozen --with playwright python results/diagnosis/blur/render_full.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "results/stray_quality"))
import score  # noqa: E402  (eval_views, the scorer's own pose mapping)

from litereality_agent.pipeline.evaluate import fidelity as fd  # noqa: E402

SCAN = Path("~/LiteReality-scans/stray_atrium_q600").expanduser()
OUT_DIR = REPO / "run/atrium_stray_q/v2"
FULL = (1920, 1440)
EVAL_ROWS = [int(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else "1219,2010,224".split(","))]
OUT = HERE / "renders"
OUT.mkdir(exist_ok=True)


def train_views():
    out = []
    for j in sorted(SCAN.glob("frame_*.json")):
        m = json.loads(j.read_text())
        out.append((j, m))
    return out


def main() -> None:
    views, _ = score.eval_views(SCAN)
    ev = [v for v in views if v.index in EVAL_ROWS]
    tv = train_views()
    pos = np.array([fd.camera_pose(m["cameraPoseARFrame"])[:3, 3] for _, m in tv])
    rot = [fd.camera_pose(m["cameraPoseARFrame"])[:3, :3] for _, m in tv]
    cams, names, photos = [], [], []
    for v in ev:
        cams.append(fd.three_camera(v, FULL)); names.append(f"eval_{v.index}"); photos.append(v.image)
        d = np.linalg.norm(pos - v.pose[:3, 3], axis=1)
        ang = np.array([np.degrees(np.arccos(np.clip((np.trace(R.T @ v.pose[:3, :3]) - 1) / 2, -1, 1)))
                        for R in rot])
        k = int(np.argmin(d / 0.05 + ang / 5.0))
        j, m = tv[k]
        stem = j.stem
        tview = fd.View(int(m["stray"]["row"]), SCAN / f"{stem}.jpg",
                        np.asarray(m["intrinsics"], float).reshape(3, 3), FULL,
                        fd.camera_pose(m["cameraPoseARFrame"]))
        cams.append(fd.three_camera(tview, FULL)); names.append(f"train_{tview.index}"); photos.append(tview.image)
        print(f"eval {v.index} -> train row {tview.index} ({stem}): {d[k]*100:.1f} cm, {ang[k]:.1f} deg", flush=True)
    t = time.perf_counter()
    renders = fd.SplatRenderer(timeout_s=3600).render((OUT_DIR / "raw.spz.bin").read_bytes(), cams, FULL)
    print(f"render {time.perf_counter() - t:.0f} s for {len(cams)} views at {FULL}", flush=True)
    meta = []
    for name, photo, rgba in zip(names, photos, renders, strict=True):
        rgb, cover = fd.composite(rgba, FULL)
        Image.fromarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)).save(OUT / f"{name}_render.png")
        with Image.open(photo) as im:
            im.convert("RGB").save(OUT / f"{name}_photo.png")
        meta.append({"name": name, "photo": str(photo), "coverage": round(float(cover.mean()), 4)})
    (OUT / "renders.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
