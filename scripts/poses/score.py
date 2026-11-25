"""Score one method's camera poses for the benchmark frames against the LiDAR reference poses.

    uv run python scripts/poses/score.py GT.json POSES.json [--out SCORE.json]

Both files: {"frames": {"00012.jpg": {"c2w": 4x4 camera-to-world, OpenCV axes (x right, y down,
z forward)}, ...}, "fx": pixels at the benchmark frames' size (one shared focal), ...}. A method
need not pose every frame; frames it leaves out count as unregistered.

The method's world is put on the reference's by a similarity (Umeyama on the camera centres of
the frames both have, refit on the best 90% to resist a few wild frames). Reported:
  registered          share of all benchmark frames the method posed
  compared            frames posed by both
  ate_rmse_m, ate_median_m, ate_p90_m   camera-centre error after the similarity
  rot_median_deg, rot_p90_deg           camera-orientation error after its rotation
  scale               the similarity's scale: 1 = the method's units are metres (when it claims
                      metric scale, |scale - 1| is its scale error)
  fx_error_pct        the method's focal against the reference focal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t with dst ~ s R src + t (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    r = u @ sign @ vt
    var = (a ** 2).sum() / len(src)
    s = float(np.trace(np.diag(d) @ sign) / var)
    return s, r, mu_d - s * r @ mu_s


def score(gt: dict, est: dict, total: int) -> dict:
    names = sorted(set(gt["frames"]) & set(est["frames"]))
    out = {"method": est.get("method"), "total": total,
           "registered": round(len(est["frames"]) / total, 4), "compared": len(names)}
    if len(names) < 3:
        return {**out, "error": "fewer than 3 frames in common with the reference"}
    G = np.array([gt["frames"][n]["c2w"] for n in names], float)
    E = np.array([est["frames"][n]["c2w"] for n in names], float)
    cg, ce = G[:, :3, 3], E[:, :3, 3]
    s, r, t = umeyama(ce, cg)
    err = np.linalg.norm((s * (r @ ce.T)).T + t - cg, axis=1)
    keep = err <= np.quantile(err, 0.9)
    if keep.sum() >= 3:
        s, r, t = umeyama(ce[keep], cg[keep])
        err = np.linalg.norm((s * (r @ ce.T)).T + t - cg, axis=1)
    rot = []
    for g, e in zip(G, E):
        d = g[:3, :3].T @ (r @ e[:3, :3])
        rot.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
    rot = np.array(rot)
    out.update({"ate_rmse_m": round(float(np.sqrt((err ** 2).mean())), 4),
                "ate_median_m": round(float(np.median(err)), 4),
                "ate_p90_m": round(float(np.quantile(err, 0.9)), 4),
                "rot_median_deg": round(float(np.median(rot)), 3),
                "rot_p90_deg": round(float(np.quantile(rot, 0.9)), 3),
                "scale": round(s, 4)})
    if est.get("fx") and gt.get("fx"):
        out["fx"] = round(float(est["fx"]), 1)
        out["fx_error_pct"] = round(100 * (float(est["fx"]) / float(gt["fx"]) - 1), 2)
    for k in ("seconds", "gpu_seconds", "cost_usd", "metric", "licence"):
        if k in est:
            out[k] = est[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("poses")
    ap.add_argument("--frames", default=str(Path.home() / "LiteReality-scans/poses_bench/atrium/frames.json"))
    ap.add_argument("--out")
    args = ap.parse_args()
    total = len(json.loads(Path(args.frames).read_text())["frames"])
    res = score(json.loads(Path(args.gt).read_text()), json.loads(Path(args.poses).read_text()), total)
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
