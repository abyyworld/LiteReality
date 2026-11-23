"""Collect every scored run into results/realityscan/table.json and print the README's markdown table.

All metrics come from score.py's per-view rows (results/realityscan/scores/*_scores.json for the
RealityScan experiments, results/stray_quality and results/trainers for the ARKit baselines), so
the 29-frame means and the means over the 5 eval frames inside RealityScan's piece (rows 32, 116,
224, 308, 981) are computed from the same renders.

    uv run --frozen python scripts/realityscan/make_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SC = REPO / "results" / "realityscan" / "scores"
PIECE_ROWS = {32, 116, 224, 308, 981}

# (label, capture, poses, frames, trainer, scores json (29 views, ARKit poses), RS-pose 5-view json)
RUNS = [
    ("v0", "Stray walk", "ARKit", "188 (full walk)", "gsplat 5k, LiDAR/PromptDA dense depth",
     REPO / "results/stray_quality/v0_scores.json", None),
    ("v2", "Stray walk", "ARKit", "637 (full walk)", "gsplat 10k, dense depth (current best)",
     REPO / "results/stray_quality/v2_scores.json", None),
    ("lfs_q206", "Stray walk", "ARKit", "188 (full walk)", "LichtFeld mcmc 30k",
     REPO / "results/trainers/lfs_mcmc30k_nocal_scores.json", None),
    ("arkit_gs5k", "Stray RS piece", "ARKit", "55", "gsplat 5k, 960 px, no dense depth",
     SC / "arkit_stray_gs5k_e29_scores.json", None),
    ("rs_gs5k", "Stray RS piece", "RealityScan", "55", "gsplat 5k, 960 px, no dense depth",
     SC / "rs_stray_gs5k_e29_scores.json", SC / "rs_stray_gs5k_sub5_rspose_scores.json"),
    ("rs_gs5k_w1920", "Stray RS piece", "RealityScan", "55", "gsplat 5k, 1920 px, no dense depth",
     SC / "rs_stray_gs5k_w1920_e29_scores.json",
     SC / "rs_stray_gs5k_w1920_sub5_rspose_scores.json"),
    ("arkit_lfs", "Stray RS piece", "ARKit", "55", "LichtFeld mcmc 30k, 960 px",
     SC / "arkit_stray_pure_lfs_e29_scores.json", None),
    ("rs_lfs", "Stray RS piece", "RealityScan", "55", "LichtFeld mcmc 30k, 960 px",
     SC / "rs_stray_pure_lfs_e29_scores.json", SC / "rs_stray_pure_lfs_sub5_rspose_scores.json"),
]
RUN_DIRS = {"arkit_gs5k": "arkit_stray_gs5k", "rs_gs5k": "rs_stray_gs5k",
            "rs_gs5k_w1920": "rs_stray_gs5k_w1920", "arkit_lfs": "arkit_stray_pure_lfs",
            "rs_lfs": "rs_stray_pure_lfs"}


def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.mean(v)), 4) if v else None


def main() -> int:
    out = []
    for label, cap, poses, frames, trainer, f29, frs in RUNS:
        if not f29.is_file():
            print(f"missing {f29}")
            continue
        d = json.loads(f29.read_text())
        rows = d["per_view"]
        sub = [r for r in rows if r["row"] in PIECE_ROWS]
        t = d["train"]
        if label in RUN_DIRS:
            rep = json.loads((REPO / "run/realityscan" / RUN_DIRS[label] / "report.json").read_text())
            cost, train_s, gpu = rep["cost_est_usd"], rep["trainer_s"], rep["gpu"]
        else:
            cost, train_s, gpu = t["cost_est_usd"], t["trainer_s"], t["gpu"]
        e = {"label": label, "capture": cap, "poses": poses, "frames": frames, "trainer": trainer,
             "views": len(rows),
             "all29": {k: mean(rows, k) for k in ("psnr", "psnr_cc", "ssim", "lpips", "coverage")},
             "piece5_arkit_pose": {k: mean(sub, k) for k in ("psnr", "psnr_cc", "ssim", "lpips")},
             "trainer_heldout": t.get("trainer_heldout"), "trainer_s": round(train_s, 1),
             "gpu": gpu, "cost_est_usd": cost, "scores": str(f29.relative_to(REPO))}
        if frs and frs.is_file():
            r5 = json.loads(frs.read_text())["per_view"]
            e["piece5_rs_pose"] = {k: mean(r5, k) for k in ("psnr", "psnr_cc", "ssim", "lpips")}
        out.append(e)
    (REPO / "results/realityscan/table.json").write_text(json.dumps(out, indent=1) + "\n")

    def f(x, n=2):
        return "-" if x is None else f"{x:.{n}f}"

    print("| capture | poses | frames | trainer | 29 eval: PSNR / SSIM / LPIPS | 5 in-piece eval "
          "(ARKit poses) | 5 in-piece eval (own RS poses) | train s | GPU $ |")
    print("|---|---|---|---|---|---|---|---|---|")
    for e in out:
        a, p = e["all29"], e["piece5_arkit_pose"]
        r = e.get("piece5_rs_pose")
        rs = f"{f(r['psnr'])} / {f(r['ssim'], 3)} / {f(r['lpips'], 3)}" if r else "-"
        print(f"| {e['capture']} | {e['poses']} | {e['frames']} | {e['trainer']} | "
              f"{f(a['psnr'])} / {f(a['ssim'], 3)} / {f(a['lpips'], 3)} | "
              f"{f(p['psnr'])} / {f(p['ssim'], 3)} / {f(p['lpips'], 3)} | {rs} | "
              f"{e['trainer_s']:.0f} | {e['cost_est_usd']:.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
