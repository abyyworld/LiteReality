"""Convert ViPE's pose/intrinsics npz (downloaded from volume litereality-poses /vipe/) to the benchmark format.

    uv run python scripts/poses/vipe/convert.py RESULTS_DIR

RESULTS_DIR holds status.json and out_<mode>/{pose,intrinsics}/atrium.npz. Writes
~/LiteReality-scans/poses_bench/atrium/poses_vipe.json (own intrinsics) and poses_vipe_fixedk.json.
ViPE's pose npz is OpenCV camera-to-world, which is the benchmark convention, so poses pass through.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Modal GPU list prices per hour (CPU/memory add a few cents)
GPU_USD_PER_H = {"L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100": 2.50, "H100": 3.95}


def usd_per_s(gpu_name: str) -> float:
    for key, price in GPU_USD_PER_H.items():
        if key in gpu_name:
            return price / 3600
    return 4.0 / 3600


HALF = 2
OUT_DIR = Path.home() / "LiteReality-scans/poses_bench/atrium"
LICENCE = ("ViPE code Apache-2.0 (DROID-SLAM parts BSD-3-Clause) / weights: DROID-SLAM droid.pth "
           "(BSD-3-Clause repo), GeoCalib pinhole CC BY 4.0 (own-K run only), MoGe-2 ViT-L MIT")


def main(res: Path) -> None:
    status = json.loads((res / "status.json").read_text())
    names = [Path(n).stem + ".jpg" for n in status["names"]]
    fn_secs = status["function_seconds"]
    total_runs = sum(r["seconds"] for r in status["runs"].values())
    for mode, out_name in (("own", "poses_vipe.json"), ("fixedk", "poses_vipe_fixedk.json")):
        run = status["runs"].get(mode)
        if run is None or run["returncode"] != 0:
            print(mode, "missing or failed", run)
            continue
        pose = np.load(res / f"out_{mode}/pose/atrium.npz")
        intr = np.load(res / f"out_{mode}/intrinsics/atrium.npz")
        c2w, inds = pose["data"], pose["inds"]
        k = intr["data"]  # [N, 4]: fx fy cx cy at the 540 x 960 input
        assert len(names) == len(c2w) == len(k), (len(names), c2w.shape, k.shape)
        fx_all = k[:, 0] * HALF
        fy_all = k[:, 1] * HALF
        fx = float(np.median(fx_all))
        frames = {}
        for i, m in zip(inds, c2w):
            if np.all(np.isfinite(m)):
                frames[names[int(i)]] = {"c2w": np.asarray(m, float).round(6).tolist()}
        # GPU time: this mode's share of the container's lifetime (frame prep shared pro rata)
        gpu_s = fn_secs * run["seconds"] / total_runs
        notes = (f"ViPE commit {status['commit']}, pipeline=pose_only, init.instance=null (no Track-Anything/"
                 f"GroundingDINO/SAM), keyframe_depth=moge2-l, frames fed at 540x960 (fx scaled x2). "
                 f"fx={fx:.1f} fy={np.median(fy_all):.1f} cx={np.median(k[:, 2]) * HALF:.1f} "
                 f"cy={np.median(k[:, 3]) * HALF:.1f} at 1080x1920; one shared K for all frames. ")
        if mode == "fixedk":
            notes += "Intrinsics fixed: GeoCalib replaced by fx=fy=1221.9 (frames.json fx_from_35mm_px), not optimised."
        else:
            notes += "Intrinsics from GeoCalib init, refined in ViPE's bundle adjustment."
        notes += f" Metric scale from MoGe-2 keyframe depth. Modal {status['gpu']}."
        doc = {"method": "vipe" if mode == "own" else "vipe_fixedk", "licence": LICENCE, "metric": True,
               "fx": round(fx, 2), "frames": frames, "seconds": run["seconds"], "gpu_seconds": round(gpu_s, 1),
               "cost_usd": round(gpu_s * usd_per_s(status["gpu"]), 3), "notes": notes}
        (OUT_DIR / out_name).write_text(json.dumps(doc))
        print(out_name, len(frames), "frames, fx", round(fx, 1), "seconds", run["seconds"])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
