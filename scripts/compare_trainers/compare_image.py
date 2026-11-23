"""Same-angle comparison of every trainer x pose-source run -> results/trainers/compare.jpg (+ .json).

Columns: (photo), then one column per run. Rows:
* the 3 shared 'wander' cameras of results/stray_quality/score.py (WANDER, room frame, the
  recording's own lens), rendered from each run's uncut raw.spz.bin (no photo exists there);
* fixed eval frames (never trained on) at their ARKit pose, exactly as score.py renders them,
  with the photo in the first column and the per-view PSNR / LPIPS in each label.
Same renderer for every tile: Spark 2.3.1 / three.js in headless Chromium (SwiftShader), SH 1.
Wander tiles are rendered at 2x and box-filtered to the tile size, like the eval tiles.

    uv run --frozen --with playwright --with lpips==0.1.4 python scripts/compare_trainers/compare_image.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_trainer import REPO, load_scorer  # noqa: E402

RUNS = [
    ("gsplat | ARKit | full walk 188 (v0)", "run/atrium_stray_q/v0"),
    ("LichtFeld MRNF30k | ARKit | full walk 188", "run/atrium_trainers/lfs_mrnf30k_nocal"),
    ("gsplat | ARKit | RS piece 55", "run/realityscan/arkit_stray_gs5k"),
    ("gsplat | RealityScan | RS piece 55", "run/realityscan/rs_stray_gs5k"),
    ("LichtFeld MCMC30k | ARKit | RS piece 55", "run/realityscan/arkit_stray_pure_lfs"),
    ("LichtFeld MCMC30k | RealityScan | RS piece 55", "run/realityscan/rs_stray_pure_lfs"),
]
EVAL_ROWS = [224, 1149]          # 224: inside RealityScan's 60-frame piece; 1149: outside it


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=EVAL_ROWS)
    ap.add_argument("--out", type=Path, default=REPO / "results/trainers/compare.jpg")
    a = ap.parse_args(argv)
    s = load_scorer()
    fd = s._fd()
    w, h = s.SIZE
    big = (w * s.SCALE, h * s.SCALE)
    lp = s.Lpips()
    renderer = fd.SplatRenderer(timeout_s=900)
    names = list(s.WANDER)
    n_rows = len(names) + len(a.rows)
    canvas = Image.new("RGB", (w * (len(RUNS) + 1), h * n_rows), (0, 0, 0))
    wander_views = [fd.View(k, Path(n), s.WANDER_K, s.WANDER_FULL,
                            s.look_at(s.WANDER[n]["eye"], s.WANDER[n]["look"]))
                    for k, n in enumerate(names)]
    for r, n in enumerate(names):
        tile = Image.new("RGB", (w, h), (30, 30, 30))
        canvas.paste(s._label(tile, f"wander: {s.WANDER[n]['what']}", size=13), (0, r * h))
    out_json = {"wander": s.WANDER, "eval_rows": a.rows, "runs": {}}
    for c, (label, d) in enumerate(RUNS, start=1):
        out_dir = REPO / d
        report = json.loads((out_dir / "report.json").read_text())
        views, _ = s.eval_views(Path(report["scan"]).expanduser())
        by_row = {v.index: v for v in views}
        sel = [by_row[r] for r in a.rows]
        cams = [fd.three_camera(v, big) for v in wander_views + sel]
        renders = renderer.render((out_dir / "raw.spz.bin").read_bytes(), cams, big)
        per = []
        for r, rgba in enumerate(renders):
            render, cover = fd.composite(rgba, s.SIZE)
            if r < len(names):
                text = f"{label}"
                per.append({"wander": names[r], "coverage": round(float(cover.mean()), 3)})
            else:
                v = sel[r - len(names)]
                real = fd.load_photo(v.image, s.SIZE)
                p, l_ = fd.psnr(real, render), lp(real, render)
                per.append({"row": v.index, "psnr": round(p, 2), "lpips": round(l_, 3)})
                text = f"{label}  PSNR {p:.1f} LPIPS {l_:.3f}"
                if c == 1:
                    canvas.paste(s._label(s._to_img(real), f"photo, eval row {v.index}"),
                                 (0, r * h))
            canvas.paste(s._label(s._to_img(render), text, size=12), (c * w, r * h))
        out_json["runs"][label] = {"dir": d, "views": per}
        print(label, per, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out, quality=88)
    a.out.with_suffix(".json").write_text(json.dumps(out_json, indent=1) + "\n")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
