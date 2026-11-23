"""Control run: our gsplat trainer (the deployed `litereality-splat` `train`, unchanged) on the SAME
tar LichtFeld trained on (run_lichtfeld.py's dataset_q206.tar.gz), with every depth loss off, so the
gsplat-vs-LichtFeld gap can be split into "trainer" and "LiDAR depth supervision" (v0 had both the
sparse LiDAR depth loss and PromptDA dense depth + normal losses; LichtFeld has no metric depth).

    uv run --frozen --env-file .env python scripts/compare_trainers/run_gsplat_nodepth.py gsplat_nodepth5k
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
RUNS = REPO / "run" / "atrium_trainers"


def main() -> int:
    import modal

    variant = sys.argv[1] if len(sys.argv) > 1 else "gsplat_nodepth5k"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    tar = (RUNS / "dataset_q206.tar.gz").read_bytes()
    meta = json.loads((RUNS / "dataset_q206.tar.json").read_text())
    options = {"steps": steps, "snapshots": [steps], "depth_lambda": 0.0, "dense_depth": False,
               "normal_lambda": 0.0}
    train = modal.Function.from_name("litereality-splat", "train")
    out = RUNS / variant
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    report = {"scan": str(Path("~/LiteReality-scans/stray_atrium_q206").expanduser()),
              "trainer": "gsplat 1.5.3 (deploy/modal/splat, depth losses off)",
              "frames": meta["frames"], "dataset": meta["dataset"], "images": meta["images"],
              "train_width": 960, "options": options}
    for ev in train.remote_gen(tar, options):
        kind = ev.get("kind")
        if kind == "started":
            report["gpu"] = ev.get("gpu")
            report["trainer_started_s"] = ev["seconds"]
        elif kind == "snapshot":
            (out / "raw.spz.bin").write_bytes(ev["spz"])
            report["bundles"] = [{"stage": "trained", "step": ev["step"], "splats": ev["splats"],
                                  "mb": round(len(ev["spz"]) / 1e6, 2)}]
        elif kind == "done":
            report["stats"] = ev.get("stats") or {}
            report["timing"] = ev.get("timing")
            report["trainer_s"] = round(ev["seconds"] - report.get("trainer_started_s", 0), 1)
        elif kind == "error":
            print(ev["error"], ev.get("log", "")[-3000:])
            return 1
    report["total_s"] = round(time.perf_counter() - t0, 1)
    report["cost_est_usd"] = round(report["trainer_s"] * 3.95 / 3600, 3)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report.get(k) for k in ("gpu", "trainer_s", "total_s", "stats")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
