"""build_pointclouds.py — fuse each scene's per-frame RGB + depth + intrinsics + poses into ONE
colorful world-space point cloud with Open3D, saved to run/<scan>/<scan>_pointcloud.ply.

Data (per scan, under scene_init/obj_stage/object_init/input/rgbd/<scan>/):
  image/frame_N.jpg      RGB, uint8, 1440x1920 (4:3)
  depth/frame_N.jpg      depth, uint16 millimetres, 192x256 (4:3)  -> depth_scale=1000
  intrinsic/intrinsic_N.npy   3x3 depth-camera intrinsic (at 256x192)
  extrinsic/extrinsic_N.npy   4x4 camera->world pose  (so world->camera = inv(pose))

Usage:
  uv run python scripts/capture/build_pointclouds.py [--voxel 0.015] [--stride 1] [--trunc 5.0] [<scan> ...]
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

def rgbd_dir(scan: str) -> str:
    return os.path.join(FINAL, scan, "scene_init", "obj_stage", "object_init", "input", "rgbd", scan)


def discover_scenes() -> list[str]:
    """Every scene under $LITEREALITY_FINAL that has an extracted rgbd tree to fuse."""
    if not os.path.isdir(FINAL):
        return []
    return sorted(d for d in os.listdir(FINAL) if os.path.isdir(rgbd_dir(d)))


def frame_ids(R: str) -> list[int]:
    def ids(pat, sub):
        return {int(re.search(r"(\d+)", os.path.basename(p)).group(1)) for p in glob.glob(os.path.join(R, sub, pat))}
    common = ids("frame_*.jpg", "image") & ids("frame_*.jpg", "depth") \
        & ids("intrinsic_*.npy", "intrinsic") & ids("extrinsic_*.npy", "extrinsic")
    return sorted(common)


def build(scan: str, voxel: float, stride: int, trunc: float, conf_min: int) -> str | None:
    R = rgbd_dir(scan)
    ids = frame_ids(R)
    if not ids:
        print(f"  ✗ {scan}: no complete RGBD frames at {R}")
        return None
    agg = o3d.geometry.PointCloud()
    used = 0
    dropped_lowconf = 0
    for i in ids[::stride]:
        depth = cv2.imread(os.path.join(R, "depth", f"frame_{i}.jpg"), cv2.IMREAD_UNCHANGED)
        bgr = cv2.imread(os.path.join(R, "image", f"frame_{i}.jpg"), cv2.IMREAD_UNCHANGED)
        if depth is None or bgr is None:
            continue
        H, W = depth.shape[:2]
        # CONFIDENCE FILTER: ARKit conf map (0 low / 1 med / 2 high) at the same 256x192 res. Zero
        # out depth below conf_min so those noisy pixels are ignored (depth==0 is skipped downstream).
        if conf_min > 0:
            conf = cv2.imread(os.path.join(SCANS, scan, f"conf_{i:05d}.png"), cv2.IMREAD_UNCHANGED)
            if conf is not None and conf.shape[:2] == depth.shape[:2]:
                depth = depth.copy()
                lowmask = conf < conf_min
                dropped_lowconf += int(lowmask.sum())
                depth[lowmask] = 0
        rgb = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        K = np.load(os.path.join(R, "intrinsic", f"intrinsic_{i}.npy"))
        pose = np.load(os.path.join(R, "extrinsic", f"extrinsic_{i}.npy"))  # camera -> world
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.uint16)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=trunc, convert_rgb_to_intensity=False)
        # Open3D's extrinsic is world->camera; result lands in world coords.
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr, np.linalg.inv(pose))
        if len(pcd.points):
            agg += pcd
            used += 1
    if not len(agg.points):
        print(f"  ✗ {scan}: fused 0 points")
        return None
    n0 = len(agg.points)
    agg = agg.voxel_down_sample(voxel)
    agg, _ = agg.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    agg.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    out = os.path.join(FINAL, scan, f"{scan}_pointcloud.ply")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    o3d.io.write_point_cloud(out, agg)
    print(f"  ✓ {scan}: {used}/{len(ids)} frames · conf>={conf_min} (dropped {dropped_lowconf/1e6:.1f}M low-conf px) "
          f"-> {n0:,} pts -> {len(agg.points):,} after {voxel*100:.1f}cm voxel  ->  {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*",
                    help="scan names (default: every scene with an extracted rgbd tree)")
    ap.add_argument("--voxel", type=float, default=0.008, help="voxel size m (density)")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--trunc", type=float, default=5.0, help="max depth m")
    ap.add_argument("--conf", type=int, default=2, help="min ARKit depth confidence 0/1/2 (2=high only)")
    a = ap.parse_args()
    scenes = a.scenes or discover_scenes()
    if not scenes:
        print(f"no scenes with an extracted rgbd tree under {FINAL} — pass scan names explicitly")
        return 2
    print(f"== point clouds for {len(scenes)} scene(s) · voxel={a.voxel*100:.1f}cm stride={a.stride} conf>={a.conf} ==")
    ok = 0
    for s in scenes:
        try:
            if build(s, a.voxel, a.stride, a.trunc, a.conf):
                ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {s}: {type(e).__name__}: {e}")
    print(f"== done: {ok}/{len(scenes)} point clouds written ==")
    return 0 if ok == len(scenes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
