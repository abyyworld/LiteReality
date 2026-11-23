"""RealityScan vs ARKit camera poses for the Stray atrium frames (prep_stray.py -> RealityScan).

RealityScan's world is arbitrary in scale, rotation and origin, so its cameras are mapped onto
ARKit's (the pool import: atrium_1059's ARKit world, metres) by a robust similarity
(`sim3.robust_sim3`: Umeyama with Huber reweighting) over every frame both have, fitted to the
camera centres plus points on each camera's axes (so orientations count too; the centre-only fit
is reported beside it). Then per frame:

    pos_err_m        |s R c_rs + t - c_arkit|                      (ATE after Sim(3))
    rot_err_deg      angle(R R_rs, R_arkit)                         (the Sim(3)'s rotation)
    rot_err_rotfit   the same with the frame rotation fitted to the orientations instead
    rpe              over ~1 s steps: relative rotation difference and relative translation
                     difference (RS scaled to metres) between the two trajectories, which a
                     slow drift of either does not accumulate into

plus the intrinsics (RealityScan's self-calibration against ARKit's per-frame K), all frames and
the 29 eval frames apart. The similarity goes to <scan>/rs_to_arkit.json for build_dataset.py.

    uv run --frozen python scripts/realityscan/stray_poses.py [--scan ~/LiteReality-scans/rs_stray]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rs_common import REPO, angle_deg, arkit_pool, read_model, save_sim3, stats

OUT = REPO / "results/realityscan/stray"


def main() -> None:
    from litereality_agent.models.capture import sim3

    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=Path("~/LiteReality-scans/rs_stray"))
    ap.add_argument("--tag", default="", help="suffix of the result file")
    a = ap.parse_args()
    scan = a.scan.expanduser()
    sel = json.loads((scan / "rs_input/selection.json").read_text())["frames"]
    imgs, cams, pts = read_model(scan / "rs_out/colmap/sparse/0")
    ar = arkit_pool(sorted(imgs))
    names = sorted(n for n in imgs if n in ar)
    src = np.array([imgs[n].C for n in names])
    dst = np.array([ar[n][1] for n in names])
    centre_fit = sim3.robust_sim3(src, dst)
    # The similarity used: centres AND orientations. Each camera contributes its centre and three
    # points on its axes (0.5 m along x and y, 1 m ahead; RS side scaled by the centre fit), so a
    # rotation that turns the cameras wrong is penalised as much as one that moves them: on a
    # walk of a few metres the centres alone leave the frame rotation ~1-2 deg loose.
    s0 = centre_fit["scale"]
    aug_s, aug_d = [], []
    for n in names:
        for k, d in ((None, 0.0), (0, 0.5), (1, 0.5), (2, 1.0)):
            aug_s.append(imgs[n].C + (imgs[n].R[:, k] * d / s0 if k is not None else 0))
            aug_d.append(ar[n][1] + (ar[n][0][:, k] * d if k is not None else 0))
    fit = sim3.robust_sim3(np.array(aug_s), np.array(aug_d))
    s, R, t = fit["scale"], fit["R"], fit["t"]
    # The frame rotation that best maps RS orientations onto ARKit's (chordal L2 mean).
    U, _, Vt = np.linalg.svd(sum(ar[n][0] @ imgs[n].R.T for n in names))
    Ro = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    per = []
    for n in names:
        c = s * R @ imgs[n].C + t
        per.append({"name": n, "eval": bool(sel.get(n, {}).get("eval")),
                    "time_s": sel.get(n, {}).get("time_s"),
                    "pos_err_m": float(np.linalg.norm(c - ar[n][1])),
                    "rot_err_deg": angle_deg(R @ imgs[n].R, ar[n][0]),
                    "rot_err_rotfit_deg": angle_deg(Ro @ imgs[n].R, ar[n][0]),
                    "fx_rs": float(cams[imgs[n].camera_id].params[0]),
                    "fx_arkit": float(ar[n][2][0, 0])})
    per.sort(key=lambda p: p["time_s"])
    # Relative pose error over ~1 s.
    order = [p["name"] for p in per]
    times = np.array([p["time_s"] for p in per])
    rpe_rot, rpe_pos = [], []
    for i, n in enumerate(order):
        j = int(np.searchsorted(times, times[i] + 1.0))
        if j >= len(order):
            break
        m = order[j]
        dR_rs = imgs[n].R.T @ imgs[m].R
        dR_ar = ar[n][0].T @ ar[m][0]
        rpe_rot.append(angle_deg(dR_rs, dR_ar))
        d_rs = imgs[n].R.T @ (imgs[m].C - imgs[n].C) * s     # in camera n's frame, metres
        d_ar = ar[n][0].T @ (ar[m][1] - ar[n][1])
        rpe_pos.append(float(np.linalg.norm(d_rs - d_ar)))
    ev = [p for p in per if p["eval"]]
    K_ar = np.median([ar[n][2] for n in names], axis=0)
    cam_rows = {}
    for cid, c in cams.items():
        cam_rows[cid] = {"model": c.model, "w": c.w, "h": c.h, "fx": c.params[0],
                         "fy": c.params[1], "cx": c.params[2], "cy": c.params[3],
                         "dist": [float(v) for v in c.params[4:]]}
    first = next(iter(cams.values()))
    centre_extent = float(np.linalg.norm(dst - dst.mean(0), axis=1).max())
    res = {
        "registered": len(imgs), "input": len(sel), "eval_input": sum(v["eval"] for v in sel.values()),
        "eval_registered": len(ev), "compared": len(names), "rs_points": int(len(pts["xyz"])),
        "sim3": {"how": "robust Umeyama on camera centres + axis points (pose-aware)",
                 "scale_rs_to_m": s, "median_residual_m": fit["median_residual"],
                 "vs_orientation_only_rotation_deg": angle_deg(R, Ro),
                 "centre_only": {"scale": s0, "median_residual_m": centre_fit["median_residual"],
                                 "rotation_vs_used_deg": angle_deg(centre_fit["R"], R),
                                 "pos_err_m": stats([np.linalg.norm(
                                     s0 * centre_fit["R"] @ imgs[n].C + centre_fit["t"]
                                     - ar[n][1]) for n in names]),
                                 "rot_err_deg": stats([angle_deg(centre_fit["R"] @ imgs[n].R,
                                                                 ar[n][0]) for n in names])}},
        "arkit_extent_m": centre_extent,
        "pos_err_m": stats([p["pos_err_m"] for p in per]),
        "rot_err_deg": stats([p["rot_err_deg"] for p in per]),
        "rot_err_rotfit_deg": stats([p["rot_err_rotfit_deg"] for p in per]),
        "eval_pos_err_m": stats([p["pos_err_m"] for p in ev]) if ev else None,
        "eval_rot_err_deg": stats([p["rot_err_deg"] for p in ev]) if ev else None,
        "rpe_1s_rot_deg": stats(rpe_rot), "rpe_1s_pos_m": stats(rpe_pos),
        "intrinsics": {"n_cameras": len(cams), "rs": cam_rows if len(cams) <= 3 else
                       {"first": cam_rows[next(iter(cam_rows))]},
                       "arkit_median": {"fx": K_ar[0, 0], "fy": K_ar[1, 1],
                                        "cx_colmap": K_ar[0, 2] + 0.5,
                                        "cy_colmap": K_ar[1, 2] + 0.5},
                       "fx_rel_diff": float(first.params[0] / K_ar[0, 0] - 1)},
        "frames": per,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"pose_vs_arkit{a.tag}.json").write_text(json.dumps(res, indent=1))
    save_sim3(scan / "rs_to_arkit.json", fit["T"],
              {"how": "robust_sim3 of RealityScan camera centres + axis points onto the pool "
                      "import's ARKit cameras (atrium_1059's ARKit world)", "frames": len(names),
               "median_residual_m": fit["median_residual"]})
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=1))


if __name__ == "__main__":
    main()
