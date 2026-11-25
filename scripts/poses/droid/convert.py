"""Convert DROID-SLAM trajectories (from run_droid.py, on volume litereality-poses /droid/) to the
benchmark's poses_<method>.json.

    uv run python scripts/poses/droid/convert.py TRAJ.npz STATUS.json TAG OUT.json [--rate 0.80]

DROID's Droid.terminate returns camera-to-world poses as [tx ty tz qx qy qz qw] (lietorch SE3,
camera axes x right, y down, z forward: OpenCV), up to an unknown scale (monocular).
"""

import argparse
import json
from pathlib import Path

import numpy as np

LICENCE = ("BSD-3-Clause (princeton-vl/DROID-SLAM; lietorch BSD-3-Clause; pytorch_scatter MIT) / "
           "droid.pth linked from the BSD-3-Clause DROID-SLAM README, no separate weight terms "
           "(trained on TartanAir, CC BY 4.0)")


def quat_to_rot(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("status")
    ap.add_argument("tag")
    ap.add_argument("out")
    ap.add_argument("--rate", type=float, default=0.80, help="GPU $/hour (L4)")
    a = ap.parse_args()
    d = np.load(a.traj)
    status = json.loads(Path(a.status).read_text())
    run = status["runs"][a.tag]
    traj, names = d["traj"], [str(n) for n in d["names"]]
    assert len(traj) == len(names), (traj.shape, len(names))
    frames, bad = {}, []
    for n, row in zip(names, traj):
        if not np.all(np.isfinite(row)):
            bad.append(n)
            continue
        m = np.eye(4)
        m[:3, :3] = quat_to_rot(row[3:7])
        m[:3, 3] = row[:3]
        frames[n] = {"c2w": [[round(float(v), 7) for v in r] for r in m]}
    gpu_seconds = float(run["seconds"])
    out = {
        "method": "droid-slam" if a.tag == "f1221.9" else f"droid-slam-{a.tag}",
        "licence": LICENCE,
        "metric": False,
        "fx": float(run["fx"]),
        "frames": frames,
        "seconds": round(float(d["seconds"]), 1),
        "gpu_seconds": round(gpu_seconds, 1),
        "cost_usd": round(gpu_seconds / 3600 * a.rate, 4),
        "notes": (f"DROID-SLAM {status['commit'][:7]} monocular, sync mode on one {status['gpu']}, demo.py "
                  f"settings with stride 1 and buffer 1024; fixed pinhole fx=fy={run['fx']:g}, cx=540, cy=960 "
                  f"(DROID does not refine intrinsics, fx is the input). Network input "
                  f"{int(d['image_size'][1])}x{int(d['image_size'][0])}. {int(d['keyframes'])} keyframes; "
                  f"global BA (backend 7+12 iters over proximity edges) at the end, then DROID's "
                  f"PoseTrajectoryFiller for the other frames. {len(bad)} non-finite poses dropped. "
                  f"gpu_seconds = this run's share of the container (frame unpack and image build excluded)."),
    }
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "frames"} | {"posed": len(frames)}, indent=1))


if __name__ == "__main__":
    main()
