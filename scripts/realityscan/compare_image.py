"""Same-angle comparison image for the RealityScan experiments -> results/realityscan/compare.jpg.

Rows: eval frames (fixed Stray eval set, never trained on), each at its ARKit pose exactly as
results/stray_quality/score.py renders it (same Spark/SwiftShader renderer, same per-import
transform). Columns: the photo, then each run. Labels carry the per-view PSNR / LPIPS.

    uv run --frozen --with playwright --with lpips==0.1.4 python scripts/realityscan/compare_image.py \\
        --rows 32 224 981 1495 \\
        --run "ARKit full walk gsplat (v2)=run/atrium_stray_q/v2" \\
        --run "ARKit piece gsplat=run/realityscan/arkit_stray_gs5k" ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_rs import REPO, load_scorer  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", required=True)
    ap.add_argument("--run", action="append", required=True, help="label=run_dir")
    ap.add_argument("--out", type=Path, default=REPO / "results/realityscan/compare.jpg")
    a = ap.parse_args(argv)
    s = load_scorer()
    fd = s._fd()
    w, h = s.SIZE
    big = (w * s.SCALE, h * s.SCALE)
    lp = s.Lpips()
    renderer = fd.SplatRenderer(timeout_s=600)
    runs = [r.split("=", 1) for r in a.run]
    canvas = Image.new("RGB", (w * (len(runs) + 1), h * len(a.rows)), (0, 0, 0))
    out_json = {"rows": a.rows, "runs": {}}
    for c, (label, d) in enumerate(runs, start=1):
        out_dir = REPO / d
        report = json.loads((out_dir / "report.json").read_text())
        views, _ = s.eval_views(Path(report["scan"]).expanduser())
        by_row = {v.index: v for v in views}
        sel = [by_row[r] for r in a.rows]
        renders = renderer.render((out_dir / "raw.spz.bin").read_bytes(),
                                  [fd.three_camera(v, big) for v in sel], big)
        per = []
        for r, (v, rgba) in enumerate(zip(sel, renders, strict=True)):
            real = fd.load_photo(v.image, s.SIZE)
            render, _cover = fd.composite(rgba, s.SIZE)
            p, l_ = fd.psnr(real, render), lp(real, render)
            per.append({"row": v.index, "psnr": round(p, 2), "lpips": round(l_, 3)})
            if c == 1:
                canvas.paste(s._label(s._to_img(real), f"photo, eval row {v.index}"), (0, r * h))
            canvas.paste(s._label(s._to_img(render), f"{label}  PSNR {p:.1f} LPIPS {l_:.3f}",
                                  size=14), (c * w, r * h))
        out_json["runs"][label] = {"dir": d, "views": per}
        print(label, per, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out, quality=88)
    a.out.with_suffix(".json").write_text(json.dumps(out_json, indent=1) + "\n")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
