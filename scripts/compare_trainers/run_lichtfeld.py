"""Train the atrium with LichtFeld Studio on Modal (deploy/modal/lichtfeld) on EXACTLY the dataset our
gsplat run v0 trained on, and write a run dir that results/stray_quality/score.py scores unchanged.

Input: the scan v0 used (~/LiteReality-scans/stray_atrium_q206: imported with
`--exclude results/stray_quality/eval_frames.txt --exclude-span 0.2`), turned into the trainer tar
by the pipeline's own `splatroom.training_set` (ARKit poses, PromptDA-dense LiDAR seed points from
the scan's cache, every 8th photo listed with no points and held out): same poses, same seeds,
same hold-out as v0. No depth maps (LichtFeld's depth loss is scale-invariant only).

    uv run --frozen python scripts/compare_trainers/run_lichtfeld.py lfs_mcmc30k --iterations 30000
    uv run --frozen python scripts/compare_trainers/run_lichtfeld.py lfs_mcmc5k --iterations 5000 \\
        --extra=--no-calibration-correction

Writes run/atrium_trainers/<variant>/{raw.spz.bin, splat.ply, report.json, lfs.log}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
SCAN = Path("~/LiteReality-scans/stray_atrium_q206").expanduser()
RUNS = REPO / "run" / "atrium_trainers"
TAR = RUNS / "dataset_q206.tar.gz"
GPU_RATE = {"L40S": 1.95 / 3600, "A100": 2.10 / 3600, "L4": 0.80 / 3600, "H100": 3.95 / 3600}


def dataset_tar() -> tuple[bytes, dict]:
    """v0's trainer tar, rebuilt (and cached) from the scan."""
    meta_path = TAR.with_suffix(".json")
    if TAR.exists() and meta_path.exists():
        return TAR.read_bytes(), json.loads(meta_path.read_text())
    from litereality_agent.models.splat import dataset, promptda, rgbd
    from litereality_agent.pipeline import splatroom

    frames = rgbd.load_frames(SCAN)
    dense = {}
    for f in frames:
        d = promptda.load_dense(f, dataset.TRAIN_WIDTH)
        if d is not None:
            dense[f.index] = d
    report: dict = {}
    tar = splatroom.training_set(frames, dense, {}, None, report, log=print)
    meta = {"frames": len(frames), "dense_cached": len(dense), "dataset": report["dataset"],
            "images": report["images"], "mb": round(len(tar) / 1e6, 1)}
    RUNS.mkdir(parents=True, exist_ok=True)
    TAR.write_bytes(tar)
    meta_path.write_text(json.dumps(meta, indent=1))
    return tar, meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("variant")
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--max-cap", type=int, default=1_000_000)
    p.add_argument("--strategy", default="mcmc")
    p.add_argument("--extra", action="append", default=[], help="more LichtFeld flags")
    p.add_argument("--xvfb", action="store_true")
    p.add_argument("--dataset-only", action="store_true")
    a = p.parse_args()
    tar, meta = dataset_tar()
    v0 = json.loads((REPO / "run/atrium_stray_q/v0/report.json").read_text())
    print(f"dataset: {meta['dataset']} ({meta['mb']} MB); v0 had {v0['dataset']}")
    if a.dataset_only:
        return 0
    from litereality_agent.models.splat import ply, spz

    options = {"strategy": a.strategy, "iterations": a.iterations, "max_cap": a.max_cap,
               "extra": [x for e in a.extra for x in e.split()], "xvfb": a.xvfb}
    t0 = time.perf_counter()
    import pickle
    import subprocess

    out = RUNS / a.variant
    out.mkdir(parents=True, exist_ok=True)
    pkl = out / "result.pkl"
    # An ephemeral `modal run` of the app (nothing deployed); the result comes back pickled.
    subprocess.run(["modal", "run", "deploy/modal/lichtfeld/app.py", "--tar", str(TAR),
                    "--options", json.dumps(options), "--out", str(pkl)], check=True,
                   env={**os.environ, "MODAL_IMAGE_BUILDER_VERSION": "2025.06"})
    res = pickle.loads(pkl.read_bytes())
    pkl.unlink()
    wall = time.perf_counter() - t0
    (out / "lfs.log").write_text(res["log"])
    for name, text in res.get("eval_files", {}).items():
        dest = out / "lfs_out" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    print(f"code {res['code']}, {res['seconds']}, gpu {res['gpu']}, files {res['files'][:30]}")
    if res["code"] != 0 or "ply" not in res:
        print(res["log"][-5000:])
        return 1
    (out / "splat.ply").write_bytes(res["ply"])
    splats = ply.read(res["ply"], max_sh_degree=1)      # the room call's export degree
    # LichtFeld trains the classic (non-mip) 3DGS rasteriser unless --enable-mip.
    mip = "--enable-mip" in options["extra"]
    raw = spz.encode(splats, antialiased=mip)
    (out / "raw.spz.bin").write_bytes(raw)
    train_s = float(res["seconds"]["train"])
    gpu = res["gpu"]
    rate = next((r for k, r in GPU_RATE.items() if k in gpu), GPU_RATE["L40S"])
    report = {
        "scan": str(SCAN), "trainer": f"LichtFeld Studio {res['lfs']}", "frames": meta["frames"],
        "dataset": meta["dataset"], "images": meta["images"], "train_width": 960,
        "options": {"steps": a.iterations, **options}, "cmd": res["cmd"], "gpu": gpu,
        "trainer_s": train_s, "total_s": round(wall, 1), "seconds": res["seconds"],
        "cost_est_usd": round(train_s * rate, 3), "stats": _stats(res),
        "bundles": [{"stage": "trained", "step": a.iterations, "splats": len(splats),
                     "mb": round(len(raw) / 1e6, 2)}],
    }
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("trainer_s", "total_s", "cost_est_usd", "stats")}))
    print(f"{len(splats):,} splats -> {out}")
    return 0


def _stats(res: dict) -> dict:
    """LichtFeld's own held-out numbers (its metrics.csv, last row; its own evaluator, so not
    directly comparable with gsplat's: the shared scorer is the comparison)."""
    import csv

    text = res.get("eval_files", {}).get("metrics.csv")
    if not text:
        return {}
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return {}
    last = rows[-1]
    out = {k: float(last[k]) for k in ("psnr", "ssim", "lpips") if last.get(k)}
    if last.get("num_gaussians"):
        out["num_GS"] = int(float(last["num_gaussians"]))
    return out


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
