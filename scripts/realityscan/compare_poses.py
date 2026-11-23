"""RealityScan (COLMAP text export) vs ARKit camera poses for the same frames.

    uv run --frozen python scripts/realityscan/compare_poses.py \
        results/realityscan/smoke/colmap/sparse/0 ~/LiteReality-scans/stray_atrium [--json out.json]

ARKit poses come from the import's frame_*.json ("cameraPoseARFrame": camera-to-world, row-major,
ARKit camera axes x right / y up / z back). RealityScan's frame is arbitrary in scale, rotation and
origin, so its camera centres are mapped onto ARKit's with a similarity (Umeyama) fit first; then
per frame: centre error (m) and rotation error (deg), plus the fitted scale and focal lengths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0])  # ARKit camera axes -> OpenCV/COLMAP (x right, y down, z fwd)


def qvec_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def read_colmap_text(sparse: Path) -> tuple[dict, dict]:
    """-> ({name: (R_c2w, centre, camera_id)}, {camera_id: (model, w, h, params)})."""
    cams = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            p = line.split()
            cams[int(p[0])] = (p[1], int(p[2]), int(p[3]), [float(v) for v in p[4:]])
    imgs = {}
    lines = [ln for ln in (sparse / "images.txt").read_text().splitlines()
             if not ln.startswith("#")]
    for line in lines[0::2]:
        p = line.split()
        if len(p) < 10:
            continue
        R = qvec_to_R(np.array([float(v) for v in p[1:5]]))  # world -> camera
        t = np.array([float(v) for v in p[5:8]])
        imgs[p[9]] = (R.T, -R.T @ t, int(p[8]))
    return imgs, cams


def read_arkit(scan: Path, names: list[str]) -> dict:
    out = {}
    for n in names:
        j = scan / (Path(n).stem + ".json")
        if j.exists():
            m = np.array(json.loads(j.read_text())["cameraPoseARFrame"], float).reshape(4, 4)
            out[n] = (m[:3, :3] @ FLIP, m[:3, 3])
    return out


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising |dst - (s R src + t)|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def compare(sparse: Path, scan: Path) -> dict:
    rs, cams = read_colmap_text(sparse)
    ar = read_arkit(scan, sorted(rs))
    names = [n for n in sorted(rs) if n in ar]
    if len(names) < 3:
        raise SystemExit(f"only {len(names)} frames in common")
    src = np.array([rs[n][1] for n in names])
    dst = np.array([ar[n][1] for n in names])
    s, R, t = umeyama(src, dst)
    per = []
    for n in names:
        c = s * R @ rs[n][1] + t
        Rrs = R @ rs[n][0]
        dR = Rrs.T @ ar[n][0]
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        per.append({"name": n, "pos_err_m": float(np.linalg.norm(c - ar[n][1])),
                    "rot_err_deg": float(ang)})
    # Rotation residual with the frame rotation fitted to the orientations instead (the chordal
    # mean of R_arkit R_rs^T): on a short baseline the centre fit's rotation is poorly determined.
    U, _, Vt = np.linalg.svd(sum(ar[n][0] @ rs[n][0].T for n in names))
    Ro = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    for p, n in zip(per, names):
        dR = (Ro @ rs[n][0]).T @ ar[n][0]
        p["rot_err_deg_rotfit"] = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    frame_angle = float(np.degrees(np.arccos(np.clip((np.trace(Ro.T @ R) - 1) / 2, -1, 1))))
    pe = np.array([p["pos_err_m"] for p in per])
    re = np.array([p["rot_err_deg"] for p in per])
    extent = float(np.linalg.norm(dst - dst.mean(0), axis=1).max())
    return {"registered": len(rs), "compared": len(names), "scale_rs_to_m": float(s),
            "arkit_extent_m": extent,
            "pos_rmse_m": float(np.sqrt((pe ** 2).mean())), "pos_median_m": float(np.median(pe)),
            "rot_median_deg": float(np.median(re)), "rot_max_deg": float(re.max()),
            "rot_median_deg_rotfit": float(np.median([p["rot_err_deg_rotfit"] for p in per])),
            "rot_max_deg_rotfit": float(max(p["rot_err_deg_rotfit"] for p in per)),
            "centre_vs_orientation_frame_deg": frame_angle,
            "cameras": {k: {"model": v[0], "w": v[1], "h": v[2], "fx": v[3][0], "fy": v[3][1],
                            "cx": v[3][2], "cy": v[3][3], "dist": v[3][4:]}
                        for k, v in list(cams.items())[:1]},
            "n_cameras": len(cams), "frames": per}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sparse", type=Path)
    ap.add_argument("scan", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    r = compare(a.sparse, a.scan.expanduser())
    summary = {k: v for k, v in r.items() if k != "frames"}
    print(json.dumps(summary, indent=1))
    if a.json:
        a.json.write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
