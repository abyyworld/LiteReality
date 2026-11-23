"""A dense RealityScan Stray alignment's world -> the room's world (atrium_1059's ARKit world, the
world of the Stray pool import and so of score.py's 29 eval poses), written as <scan>/rs_to_arkit.json
in the format build_dataset.py reads (rs_common.save_sim3).

    raw ARKit -> pool world   rigid: the pool import moved the recording's raw odometry into
                              atrium_1059's world; fitted here from the pool frames' rows (residual
                              printed; it must be ~0)
    RS -> raw ARKit           Sim(3): sim3.robust_sim3 of the largest component's camera centres
                              plus a point 0.1 m (in ARKit metres, scaled into RS units) along each
                              camera's optical axis and its down axis, onto the same rows' ARKit
                              cameras: centres AND orientations are fitted (a centre-only fit
                              leaves the rotation weak on a short or straight walk)

    uv run --frozen python scripts/realityscan/dense_to_room.py <run dir> <frames dir> --scan <scan dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rs_common import POOL, angle_deg, read_model, save_sim3, stats  # noqa: E402
from dense_report import arkit_rows  # noqa: E402


def raw_to_pool() -> tuple[np.ndarray, dict]:
    from litereality_agent.models.capture import sim3
    ar = arkit_rows()
    src, dst = [], []
    for j in sorted(POOL.glob("frame_*.json")):
        d = json.loads(j.read_text())
        r = int(d["stray"]["row"])
        if r in ar:
            m = np.array(d["cameraPoseARFrame"], float).reshape(4, 4)
            src.append(ar[r][1])
            dst.append(m[:3, 3])
    fit = sim3.robust_sim3(np.array(src), np.array(dst))
    return fit["T"], {"frames": len(src), "scale": fit["scale"],
                      "median_residual_m": fit["median_residual"]}


def main() -> None:
    from litereality_agent.models.capture import sim3

    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("frames", type=Path)
    ap.add_argument("--scan", type=Path, required=True)
    a = ap.parse_args()
    run = a.run.expanduser()
    sel = json.loads((a.frames.expanduser() / "selection.json").read_text())["frames"]
    imgs, cams, pts = read_model(next(run.rglob("images.txt")).parent)
    ar = arkit_rows()
    names = [n for n in sorted(imgs) if n in sel]
    C = np.array([imgs[n].C for n in names])
    Ca = np.array([ar[sel[n]["row"]][1] for n in names])
    s0 = sim3.robust_sim3(C, Ca)["scale"]
    L = 0.1 / s0          # 0.1 m in RS units
    src = np.concatenate([C, C + L * np.array([imgs[n].R[:, 2] for n in names]),
                          C + L * np.array([imgs[n].R[:, 1] for n in names])])
    dst = np.concatenate([Ca, Ca + 0.1 * np.array([ar[sel[n]["row"]][0][:, 2] for n in names]),
                          Ca + 0.1 * np.array([ar[sel[n]["row"]][0][:, 1] for n in names])])
    fit = sim3.robust_sim3(src, dst)
    T_rs_raw = fit["T"]
    T_raw_pool, q = raw_to_pool()
    T = T_raw_pool @ T_rs_raw
    s = sim3.sim3_scale(T_rs_raw)
    Rw = T_rs_raw[:3, :3] / s
    pos = np.linalg.norm(sim3.apply(T_rs_raw, C) - Ca, axis=1)
    rot = [angle_deg(Rw @ imgs[n].R, ar[sel[n]["row"]][0]) for n in names]
    meta = {"how": "robust_sim3 of the dense RealityScan run's camera centres + optical/down axis "
                   "points onto the raw ARKit odometry of the same rows, then the raw -> pool "
                   "(atrium_1059 ARKit world) rigid fit",
            "run": str(run), "frames": len(names), "median_residual_m": fit["median_residual"],
            "pos_err_m": stats(pos), "rot_err_deg": stats(rot), "raw_to_pool": q}
    save_sim3(a.scan.expanduser() / "rs_to_arkit.json", T, meta)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
