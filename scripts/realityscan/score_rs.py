"""Score a RealityScan-pose training run with the shared scorer, results/stray_quality/score.py,
unchanged: same 29 fixed Stray eval frames, same Spark/SwiftShader renderer, same metrics. Its
outputs are redirected to results/realityscan/scores/ (as scripts/compare_trainers/score_trainer.py
does), so nothing is written into the stray_quality workflow's files.

    --poses arkit   (default) the eval frames at their ARKit poses and K, the original photos:
                    exactly what every other variant is scored at. The run's splat must be in
                    the pool import's world (report.json "scan" = the pool, so score.py's
                    per-import transform is the identity).
    --poses rs      the eval frames at RealityScan's OWN poses for them (they were in the
                    alignment, never in training), mapped by the same similarity as the training
                    cameras, RS's undistorted K, and the photos undistorted with RS's lens
                    (build_dataset.py --eval-rs: <scan>/eval_rs/). This removes the ARKit-vs-RS
                    camera disagreement from the score, so it measures the splat, not the
                    registration of one camera set onto the other.

    uv run --frozen --with playwright --with lpips==0.1.4 python scripts/realityscan/score_rs.py \\
        eval rs_stray_gs run/realityscan/rs_stray_gs [--poses rs --eval-rs ~/LiteReality-scans/rs_stray/eval_rs]
    uv run --frozen python scripts/realityscan/score_rs.py table rs_stray_gs ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "realityscan" / "scores"


def load_scorer():
    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("stray_score",
                                                  REPO / "results/stray_quality/score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OUT.mkdir(parents=True, exist_ok=True)
    mod.HERE = OUT                      # outputs only; EVAL_FRAMES stays the shared fixed set
    return mod


def rs_eval_views(mod, eval_dir: Path):
    """score.py's `eval_views` with the eval frames' RealityScan poses (same rows, same order)."""
    def views(scan, pool=mod.POOL):
        fd = mod._fd()
        by_row = {}
        for j in eval_dir.glob("frame_*.json"):
            d = json.loads(j.read_text())
            by_row[int(d["row"])] = (j, d)
        out = []
        for row, _stem in mod.eval_rows():
            if row not in by_row:
                raise SystemExit(f"eval row {row} has no RealityScan pose in {eval_dir} "
                                 "(not registered): cannot score at RS poses")
            j, d = by_row[row]
            out.append(fd.View(row, j.with_suffix(".jpg"), np.array(d["K"], float),
                               tuple(d["size"]), np.array(d["pose_room_gl"], float)))
        return out, {"poses": "RealityScan eval poses", "dir": str(eval_dir)}
    return views


def main(argv: list[str]) -> int:
    s = load_scorer()
    if argv[0] == "table":
        s.table(argv[1:])
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("eval",))
    ap.add_argument("variant")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--poses", choices=("arkit", "rs"), default="arkit")
    ap.add_argument("--eval-rs", type=Path, default=Path("~/LiteReality-scans/rs_stray/eval_rs"))
    ap.add_argument("--subset", action="store_true",
                    help="only the eval frames RealityScan registered (those in --eval-rs), in "
                         "either pose mode: a partial reconstruction cannot be judged on views it "
                         "never saw")
    a = ap.parse_args(argv)
    out_dir = a.out_dir if a.out_dir.is_absolute() else REPO / a.out_dir
    variant = a.variant
    if a.subset:
        keep = {int(json.loads(j.read_text())["row"])
                for j in a.eval_rs.expanduser().glob("frame_*.json")}
        rows = s.eval_rows
        s.eval_rows = lambda *x: [r for r in rows(*x) if r[0] in keep]
        variant += f"_sub{len(keep)}"
    if a.poses == "rs":
        s.eval_views = rs_eval_views(s, a.eval_rs.expanduser())
        variant += "_rspose"
    s.score(variant, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
