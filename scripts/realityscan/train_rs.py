"""Train a splat on a RealityScan dataset (build_dataset.py) and write a run dir that score_rs.py
(results/stray_quality/score.py) reads: <run>/{raw.spz.bin, report.json}.

    gsplat     the deployed `litereality-splat` trainer (the same `train` generator
               models/splat/remote.ModalSplatTrainer streams for pipeline/splatroom, called
               through deploy/modal/realityscan/train_splat.py), gsplat 1.5.3 MCMC on an H100, `--steps` (5000, as
               the v0 baseline) and otherwise its defaults: sparse depth loss against the
               dataset's points (here RealityScan's SfM points), bilateral grid, antialiased.
               No dense depth: these datasets have no LiDAR / PromptDA maps.
    lichtfeld  LichtFeld Studio on Modal (deploy/modal/lichtfeld, an ephemeral `modal run` as
               scripts/compare_trainers/run_lichtfeld.py does): MCMC, `--iterations` 30000,
               --no-calibration-correction (that workflow's lfs_mcmc30k_nocal).

The splat is in the room frame of atrium_1059's ARKit world, the world of the Stray pool import,
so report.json's "scan" is the pool: score.py's per-import transform is the identity.

    uv run --frozen --env-file .env python scripts/realityscan/train_rs.py rs_stray_gs \\
        ~/LiteReality-scans/rs_stray/rs_w960.tar.gz --trainer gsplat
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

from rs_common import POOL, REPO

RUNS = REPO / "run" / "realityscan"
GPU_RATE = {"L40S": 1.95 / 3600, "A100": 2.10 / 3600, "L4": 0.80 / 3600, "H100": 3.95 / 3600}


def gsplat(tar_path: Path, steps: int, out: Path, log, batch_size: int = 1) -> dict:
    """The deployed trainer through a `modal run` entrypoint (deploy/modal/realityscan/
    train_splat.py: `Function.from_name("litereality-splat", "train").remote_gen`), so the Modal
    CLI's own login is used, as for every other Modal call of these experiments."""
    # batch_size 1: gsplat stacks a batch's depth-loss points, and SfM points per photo vary
    # (our LiDAR datasets give every photo the same count, so they can batch 4).
    opts = {"steps": steps, "snapshots": [steps], "batch_size": batch_size}
    pkl = out / "events.pkl"
    subprocess.run(["modal", "run", "deploy/modal/realityscan/train_splat.py", "--tar",
                    str(tar_path), "--options", json.dumps(opts), "--out", str(pkl)],
                   check=True, cwd=REPO,
                   env={**os.environ, "MODAL_IMAGE_BUILDER_VERSION": "2025.06"})
    events = pickle.loads(pkl.read_bytes())
    pkl.unlink()
    res = {"options": opts}
    for ev in events:
        k = ev.get("kind")
        if k == "started":
            res["gpu"] = ev.get("gpu")
        elif k == "snapshot":
            res.update(spz=ev["spz"], step=ev["step"], splats=ev.get("splats"))
        elif k == "done":
            res["stats"] = ev.get("stats") or {}
            res["trainer_s"] = float(ev.get("seconds") or 0)
            res["timing"] = ev.get("timing")
            log(f"done in {ev.get('seconds')} s: {res['stats']}")
        elif k == "error":
            raise SystemExit(f"trainer error: {ev.get('error')}\n{(ev.get('log') or '')[-4000:]}")
    if "spz" not in res:
        raise SystemExit("no snapshot came back")
    return res


def lichtfeld(tar_path: Path, iterations: int, out: Path, log) -> dict:
    from litereality_agent.models.splat import ply, spz

    options = {"strategy": "mcmc", "iterations": iterations, "max_cap": 1_000_000,
               "extra": ["--no-calibration-correction"], "xvfb": False}
    pkl = out / "result.pkl"
    subprocess.run(["modal", "run", "deploy/modal/lichtfeld/app.py", "--tar", str(tar_path),
                    "--options", json.dumps(options), "--out", str(pkl)], check=True, cwd=REPO,
                   env={**os.environ, "MODAL_IMAGE_BUILDER_VERSION": "2025.06"})
    r = pickle.loads(pkl.read_bytes())
    pkl.unlink()
    (out / "lfs.log").write_text(r["log"])
    if r["code"] != 0 or "ply" not in r:
        raise SystemExit(f"LichtFeld failed: {r['log'][-4000:]}")
    splats = ply.read(r["ply"], max_sh_degree=1)
    stats = {}
    text = r.get("eval_files", {}).get("metrics.csv")
    if text:
        import csv
        rows = list(csv.DictReader(text.splitlines()))
        if rows:
            stats = {k: float(rows[-1][k]) for k in ("psnr", "ssim", "lpips") if rows[-1].get(k)}
    log(f"LichtFeld {r.get('lfs')}: {len(splats)} splats, {r['seconds']}, {r['gpu']}")
    return {"options": options, "spz": spz.encode(splats, antialiased=False),
            "splats": len(splats), "step": iterations, "gpu": r["gpu"],
            "trainer_s": float(r["seconds"]["train"]), "stats": stats, "cmd": r.get("cmd"),
            "lfs": r.get("lfs")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("tar", type=Path)
    ap.add_argument("--trainer", choices=("gsplat", "lichtfeld"), default="gsplat")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--iterations", type=int, default=30000)
    a = ap.parse_args()
    tar_path = a.tar.expanduser()
    meta = json.loads(tar_path.with_name(tar_path.name.replace(".tar.gz", ".json")).read_text())
    out = RUNS / a.variant
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if a.trainer == "gsplat":
        r = gsplat(tar_path, a.steps, out, print)
        trainer = "gsplat 1.5.3 (deploy/modal/splat)"
    else:
        r = lichtfeld(tar_path, a.iterations, out, print)
        trainer = f"LichtFeld Studio {r.get('lfs')}"
    wall = time.perf_counter() - t0
    (out / "raw.spz.bin").write_bytes(r["spz"])
    gpu = r.get("gpu") or ""
    rate = next((v for k, v in GPU_RATE.items() if k in gpu), GPU_RATE["H100"])
    report = {
        "scan": str(POOL), "dataset_tar": str(tar_path), "trainer": trainer,
        "frames": meta["photos"], "dataset": {k: meta[k] for k in
                                              ("photos", "held_out", "points", "poses", "lens")},
        "train_width": meta["long"], "options": {"steps": r["step"], **r["options"]},
        "gpu": gpu, "trainer_s": r["trainer_s"], "total_s": round(wall, 1),
        "cost_est_usd": round(r["trainer_s"] * rate, 3), "stats": r.get("stats") or {},
        "bundles": [{"stage": "trained", "step": r["step"], "splats": r.get("splats"),
                     "mb": round(len(r["spz"]) / 1e6, 2)}],
        "note": "RealityScan research run (INTERNAL ONLY); splat in the pool import's room frame",
    }
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("trainer_s", "total_s", "cost_est_usd", "stats")}))
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
