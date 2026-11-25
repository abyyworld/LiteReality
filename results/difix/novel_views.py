"""Off-path views of trained atrium splats, side by side: where the phone never stood.

    uv run --frozen --with playwright python results/difix/novel_views.py OUT.jpg \
        live=run/atrium_stray_q/v2/raw.spz.bin C=.../C.spz B=.../B.spz [--views 12]

The views are the Difix rounds' targets (scripts/difix/poses.py: ~/LiteReality-scans/difix/poses/
novel_poses.npy, last round), an even spread of them, drawn upright (the photos lie on their
side) by the room call's own renderer: Spark 2.3.1 in headless Chromium on Metal with the call's
SparkRenderer options (`pipeline/evaluate/fidelity.SplatRenderer`), each uncut splat as trained,
at 640 x 480 (2x supersampled). Same cameras for every splat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "results" / "stray_quality"))
import score  # noqa: E402

POSES = Path.home() / "LiteReality-scans/difix/poses/novel_poses.npy"
SIZE = (640, 480)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("splats", nargs="+", help="NAME=PATH")
    ap.add_argument("--views", type=int, default=12)
    args = ap.parse_args()
    fd = score._fd()
    goals = np.load(POSES)[-1]
    pick = np.linspace(0, len(goals) - 1, args.views).round().astype(int)
    big = (SIZE[0] * score.SCALE, SIZE[1] * score.SCALE)
    cams = []
    for k, v in enumerate(pick):
        eye = goals[v][:3, 3]
        pose = score.look_at(eye, eye + goals[v][:3, 2])
        view = fd.View(k, Path(f"novel_{v}"), score.WANDER_K, score.WANDER_FULL, pose)
        cams.append(fd.three_camera(view, big))
    renderer = fd.SplatRenderer(timeout_s=900, gl="metal", settle_ms=1000)
    cols = []
    for item in args.splats:
        name, path = item.split("=", 1)
        renders = renderer.render(Path(path).read_bytes(), cams, big)
        tiles = []
        for v, rgba in zip(pick, renders, strict=True):
            rgb, _ = fd.composite(rgba, SIZE)
            img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            ImageDraw.Draw(img).text((8, 8), f"{name}  view {v}", fill=(255, 40, 40))
            tiles.append(img)
        cols.append(tiles)
        print(f"{name}: {len(tiles)} views", flush=True)
    sheet = Image.new("RGB", (SIZE[0] * len(cols), SIZE[1] * len(pick)))
    for c, tiles in enumerate(cols):
        for r, img in enumerate(tiles):
            sheet.paste(img, (c * SIZE[0], r * SIZE[1]))
    sheet.save(args.out, quality=88)
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
