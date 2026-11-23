"""Score a trainer comparison run with the shared scorer, results/stray_quality/score.py, unchanged:
same 29 fixed eval frames, same Spark/SwiftShader renderer, same metrics. Only its output folder
is redirected to results/trainers/ so this never writes into the stray_quality workflow's files.

    uv run --frozen --with playwright --with lpips==0.1.4 python scripts/compare_trainers/score_trainer.py \\
        eval gsplat_v0 run/atrium_stray_q/v0
    uv run --frozen --with playwright --with lpips==0.1.4 python scripts/compare_trainers/score_trainer.py \\
        run lfs_mcmc30k run/atrium_trainers/lfs_mcmc30k --raw   # eval + raw wander
    uv run --frozen python scripts/compare_trainers/score_trainer.py table gsplat_v0 lfs_mcmc30k
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "trainers"


def load_scorer():
    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("stray_score",
                                                  REPO / "results/stray_quality/score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OUT.mkdir(parents=True, exist_ok=True)
    mod.HERE = OUT                      # outputs only; EVAL_FRAMES stays the shared fixed set
    return mod


def main(argv: list[str]) -> int:
    s = load_scorer()
    cmd, rest = argv[0], argv[1:]
    if cmd == "table":
        s.table(rest)
        return 0
    variant, out_dir = rest[0], Path(rest[1])
    out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
    if cmd in ("run", "eval"):
        s.score(variant, out_dir)
    if cmd in ("run", "wander"):
        s.wander(variant, out_dir, raw="--raw" in rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
