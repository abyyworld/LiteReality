"""Side-by-side photo | v2 | best arm for 4 eval views (one with the door), at 960 x 720 from a 1:1
1920 x 1440 render (Spark 2.3.1, the scorer's renderer), with a 2x crop row of each panel's centre.

    uv run --frozen --with playwright --with lpips==0.1.4 python results/fix_v4/retrain/make_compare.py \
        BEST_LABEL run/atrium_stray_q/<best> [--out results/fix_v4/compare.jpg]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "results" / "stray_quality"))
import score as sc  # noqa: E402

PANEL = (960, 720)
FULL = (1920, 1440)


def door_centre(plan) -> np.ndarray:
    for op in plan.openings.values():
        if "door" in op.handle.lower() or "door" in str(getattr(op, "kind", "")).lower():
            wall = plan.walls[op.wall]
            mid = np.asarray(wall.start) + wall.along * op.offset
            return np.array([mid[0], mid[1], plan.floor_z + op.sill + op.height / 2.0])
    raise SystemExit("no door in the RoomPlan")


def pick_views(views, plan, fd) -> list:
    door = door_centre(plan)
    best, best_d = None, 1e9
    for v in views:
        uv, depth = fd.project(door[None], v, FULL)
        if depth[0] <= 0.3:
            continue
        d = float(np.hypot(*(uv[0] - np.array(FULL) / 2)))
        if d < best_d:
            best, best_d = v, d
    if best is None:
        raise SystemExit("no eval view sees the door")
    rest = [v for v in views if v is not best]
    others = [rest[i] for i in np.linspace(0, len(rest) - 1, 3).round().astype(int)]
    return [best, *others]


def label(img: Image.Image, text: str) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    d.text((8, 7), text, fill=(255, 255, 255))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("best_label")
    ap.add_argument("best_dir")
    ap.add_argument("--base-label", default="v2 (served)")
    ap.add_argument("--base-dir", default="run/atrium_stray_q/v2")
    ap.add_argument("--out", default="results/fix_v4/compare.jpg")
    args = ap.parse_args()
    fd = sc._fd()
    from litereality_agent.pipeline.quickroom.roomplan import parse_room

    base, best = Path(args.base_dir), Path(args.best_dir)
    scan = Path(sc.load_run(best)["scan"]).expanduser()
    plan = parse_room(scan)
    views, _ = sc.eval_views(scan)
    picks = pick_views(views, plan, fd)
    renderer = fd.SplatRenderer(timeout_s=3600)
    cols: dict[str, list] = {}
    for name, d in (("base", base), ("best", best)):
        # Each variant in its own world (each import fitted its own alignment): its own views.
        s = Path(sc.load_run(d)["scan"]).expanduser()
        vs, _ = sc.eval_views(s)
        by = {v.index: v for v in vs}
        cams = [fd.three_camera(by[p.index], FULL) for p in picks]
        cams = [c for c in cams for _ in (0, 1)]                  # settle: keep the second draw
        renders = renderer.render((d / "raw.spz.bin").read_bytes(), cams, FULL)[1::2]
        cols[name] = [fd.composite(r, FULL)[0] for r in renders]
    rows = []
    for i, v in enumerate(picks):
        real = fd.load_photo(v.image, FULL)
        trip = [real, cols["base"][i], cols["best"][i]]
        names = [f"photo row {v.index}{' (door)' if i == 0 else ''}", args.base_label,
                 args.best_label]
        top, crop = [], []
        for arr, n in zip(trip, names, strict=True):
            im = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
            p = fd.psnr(real, sc.colour_fit(arr, real)) if arr is not real else None
            top.append(label(im.resize(PANEL, Image.BOX),
                             n + ("" if p is None else f"   PSNR_cc {p:.1f} dB @1920")))
            cx, cy = FULL[0] // 2, FULL[1] // 2
            c = im.crop((cx - 480, cy - 180, cx + 480, cy + 180))        # 1:1 pixels, centre
            crop.append(label(c, n + "  centre crop, 1:1 at 1920"))
        rows.append((top, crop))
    W = PANEL[0] * 3
    H = sum(PANEL[1] + 360 for _ in rows)
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    y = 0
    for top, crop in rows:
        for k, im in enumerate(top):
            canvas.paste(im, (k * PANEL[0], y))
        y += PANEL[1]
        for k, im in enumerate(crop):
            canvas.paste(im, (k * PANEL[0], y))
        y += 360
    out = Path(args.out)
    canvas.save(out, quality=88)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"views": [v.index for v in picks], "door_view": picks[0].index,
         "base": str(base), "best": str(best)}, indent=1) + "\n")
    print(out, [v.index for v in picks])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
