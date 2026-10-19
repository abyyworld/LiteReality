"""colorize_pcd.py — colorize the scanner's clean pointcloud.pcd by projecting the RGB frames onto
it (occlusion-aware via the depth maps, confidence-gated). Saves run/<scan>/<scan>_pcd_colored.ply.

The scanner pcd is clean geometry but has no color; each point gets the average RGB from every frame
in which it is actually visible (in-image, in front, depth-consistent, high-confidence).

  uv run python scripts/capture/colorize_pcd.py [--conf 2] [--tol 0.06] [<scan> ...]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import numpy as np
import open3d as o3d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL = os.environ.get("LITEREALITY_FINAL") or os.path.join(REPO, "run")
SCANS = os.environ.get("LR_SCANS_DIR") or os.path.join(REPO, "scans_uploaded")


def discover_scenes() -> list[str]:
    """Every scan under $LR_SCANS_DIR that ships a pointcloud.pcd to colorize."""
    if not os.path.isdir(SCANS):
        return []
    return sorted(d for d in os.listdir(SCANS)
                  if os.path.isfile(os.path.join(SCANS, d, "pointcloud.pcd")))


def colorize(scan: str, conf_min: int, tol: float) -> str | None:
    pcd_path = os.path.join(SCANS, scan, "pointcloud.pcd")
    if not os.path.exists(pcd_path):
        print(f"  ✗ {scan}: no pointcloud.pcd")
        return None
    pcd = o3d.io.read_point_cloud(pcd_path)
    P = np.asarray(pcd.points)                       # N x 3 world
    if not len(P):
        print(f"  ✗ {scan}: empty pcd")
        return None
    Ph = np.concatenate([P, np.ones((len(P), 1))], axis=1).T   # 4 x N
    csum = np.zeros((len(P), 3))
    cnt = np.zeros(len(P))
    R = os.path.join(FINAL, scan, "scene_init", "obj_stage", "object_init", "input", "rgbd", scan)
    ids = sorted(int(re.search(r"(\d+)", os.path.basename(p)).group(1))
                 for p in glob.glob(os.path.join(R, "extrinsic", "*.npy")))
    used = 0
    for i in ids:
        try:
            K = np.load(os.path.join(R, "intrinsic", f"intrinsic_{i}.npy"))
            pose = np.load(os.path.join(R, "extrinsic", f"extrinsic_{i}.npy"))
            depth = cv2.imread(os.path.join(R, "depth", f"frame_{i}.jpg"), cv2.IMREAD_UNCHANGED)
            bgr = cv2.imread(os.path.join(R, "image", f"frame_{i}.jpg"), cv2.IMREAD_UNCHANGED)
        except Exception:
            continue
        if depth is None or bgr is None:
            continue
        H, W = depth.shape[:2]
        conf = cv2.imread(os.path.join(SCANS, scan, f"conf_{i:05d}.png"), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pc = np.linalg.inv(pose) @ Ph                # 4 x N camera space
        z = pc[2]
        front = z > 1e-6
        u = np.full(len(P), -1.0)
        v = np.full(len(P), -1.0)
        u[front] = fx * pc[0][front] / z[front] + cx
        v[front] = fy * pc[1][front] / z[front] + cy
        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        inim = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        idx = np.where(inim)[0]
        if not len(idx):
            continue
        du, dv = ui[idx], vi[idx]
        dmap = depth[dv, du].astype(np.float32) / 1000.0     # metres at that pixel
        ok = (dmap > 0) & (np.abs(dmap - z[idx]) < tol)      # occlusion / depth consistency
        if conf is not None and conf.shape[:2] == depth.shape[:2]:
            ok &= conf[dv, du] >= conf_min
        sel = idx[ok]
        if not len(sel):
            continue
        cols = rgb[vi[sel], ui[sel]].astype(np.float64) / 255.0
        csum[sel] += cols
        cnt[sel] += 1
        used += 1
    colored = cnt > 0
    frac = colored.mean() if len(colored) else 0
    P2 = P[colored]
    C2 = (csum[colored] / cnt[colored, None])
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(P2)
    out_pcd.colors = o3d.utility.Vector3dVector(np.clip(C2, 0, 1))
    out_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    out = os.path.join(FINAL, scan, f"{scan}_pcd_colored.ply")
    o3d.io.write_point_cloud(out, out_pcd)
    print(f"  ✓ {scan}: {used} frames · {len(P):,} pcd pts · {frac*100:.0f}% colored -> {len(P2):,} -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--conf", type=int, default=2, help="min ARKit confidence for a coloring pixel")
    ap.add_argument("--tol", type=float, default=0.06, help="depth-consistency tol (m) for occlusion")
    a = ap.parse_args()
    scenes = a.scenes or discover_scenes()
    if not scenes:
        print(f"no scans with a pointcloud.pcd under {SCANS} — pass scan names explicitly")
        return 2
    print(f"== colorize pcd for {len(scenes)} scene(s) · conf>={a.conf} tol={a.tol}m ==")
    ok = 0
    for s in scenes:
        try:
            if colorize(s, a.conf, a.tol):
                ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {s}: {type(e).__name__}: {e}")
    print(f"== done: {ok}/{len(scenes)} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
