"""build_meshes.py — fuse each scene's per-frame RGB + depth + intrinsics + poses into ONE colored
TRIANGLE MESH with Open3D TSDF volume integration (KinectFusion-style), saved to
run/<scan>/<scan>_mesh.ply (+ _mesh.glb for the viewer).

Why TSDF instead of meshing the point cloud: we have the full RGBD sequence (depth + RGB + intrinsic
+ camera->world pose + ARKit confidence per frame), so we integrate every depth map into a truncated
signed-distance volume and marching-cubes out a watertight, colored surface. That fuses multi-view
depth (denoises + fills) far better than Poisson/ball-pivoting on a fused point cloud.

Data (per scan, under scene_init/obj_stage/object_init/input/rgbd/<scan>/): identical to build_pointclouds.py
  image/frame_N.jpg      RGB uint8 1440x1920 -> resized to depth res
  depth/frame_N.jpg      depth uint16 mm 192x256   (depth_scale=1000)
  intrinsic/intrinsic_N.npy   3x3 depth-camera intrinsic (at 256x192)
  extrinsic/extrinsic_N.npy   4x4 camera->world pose (world->camera = inv(pose))
  scans_uploaded/<scan>/conf_NNNNN.png   ARKit confidence 0/1/2 (same 256x192)

Usage:
  uv run python scripts/capture/build_meshes.py [--voxel 0.015] [--sdf-trunc 0.06] [--trunc 4.0]
                                   [--conf 1] [--stride 1] [--decimate 300000] [<scan> ...]
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
        return {int(re.search(r"(\d+)", os.path.basename(p)).group(1))
                for p in glob.glob(os.path.join(R, sub, pat))}
    common = ids("frame_*.jpg", "image") & ids("frame_*.jpg", "depth") \
        & ids("intrinsic_*.npy", "intrinsic") & ids("extrinsic_*.npy", "extrinsic")
    return sorted(common)


def _save_glb(mesh: o3d.geometry.TriangleMesh, path: str) -> bool:
    """Write a .glb carrying vertex colors, via trimesh (Open3D's glb writer drops vertex colors)."""
    try:
        import trimesh
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)
        if not len(v) or not len(f):
            return False
        vc = np.asarray(mesh.vertex_colors)
        kw = {}
        if len(vc) == len(v):
            kw["vertex_colors"] = (np.clip(vc, 0, 1) * 255).astype(np.uint8)
        vn = np.asarray(mesh.vertex_normals)
        if len(vn) == len(v):
            kw["vertex_normals"] = vn
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False, **kw)
        tm.export(path)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    (glb export skipped: {type(e).__name__}: {e})")
        return False


def build(scan: str, voxel: float, sdf_trunc: float, trunc: float, conf_min: int,
          stride: int, decimate: int, min_cluster_frac: float) -> str | None:
    R = rgbd_dir(scan)
    ids = frame_ids(R)
    if not ids:
        print(f"  ✗ {scan}: no complete RGBD frames at {R}")
        return None

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    used = 0
    dropped_lowconf = 0
    for i in ids[::stride]:
        depth = cv2.imread(os.path.join(R, "depth", f"frame_{i}.jpg"), cv2.IMREAD_UNCHANGED)
        bgr = cv2.imread(os.path.join(R, "image", f"frame_{i}.jpg"), cv2.IMREAD_UNCHANGED)
        if depth is None or bgr is None:
            continue
        H, W = depth.shape[:2]
        # CONFIDENCE FILTER: ARKit conf map (0 low / 1 med / 2 high). Zero out depth below conf_min so
        # noisy pixels aren't integrated (depth==0 is ignored by TSDF).
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
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.uint16))),
            depth_scale=1000.0, depth_trunc=trunc, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intr, np.linalg.inv(pose))  # extrinsic = world->camera
        used += 1

    if not used:
        print(f"  ✗ {scan}: integrated 0 frames")
        return None

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    n_v0, n_t0 = len(mesh.vertices), len(mesh.triangles)
    if not n_t0:
        print(f"  ✗ {scan}: TSDF produced an empty mesh (try lower --conf or larger --voxel)")
        return None

    # drop tiny disconnected floaters (keep clusters >= min_cluster_frac of the largest)
    tri_ids, counts, _ = mesh.cluster_connected_triangles()
    tri_ids = np.asarray(tri_ids)
    counts = np.asarray(counts)
    if len(counts):
        keep = counts >= max(50, int(counts.max() * min_cluster_frac))
        remove = np.logical_not(keep[tri_ids])
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()

    # optional decimation to keep the .glb viewer-friendly
    if decimate and len(mesh.triangles) > decimate:
        mesh = mesh.simplify_quadric_decimation(int(decimate))
        mesh.compute_vertex_normals()

    out_dir = os.path.join(FINAL, scan)
    os.makedirs(out_dir, exist_ok=True)
    ply = os.path.join(out_dir, f"{scan}_mesh.ply")
    o3d.io.write_triangle_mesh(ply, mesh, write_vertex_colors=True, write_vertex_normals=True)
    glb = os.path.join(out_dir, f"{scan}_mesh.glb")
    glb_ok = _save_glb(mesh, glb)

    print(f"  ✓ {scan}: {used}/{len(ids)} frames · conf>={conf_min} "
          f"(dropped {dropped_lowconf/1e6:.1f}M low-conf px) -> "
          f"mesh {len(mesh.vertices):,}v / {len(mesh.triangles):,}t "
          f"(raw {n_v0:,}v/{n_t0:,}t)  ->  {ply}" + (f" + {os.path.basename(glb)}" if glb_ok else ""))
    return ply


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", help="scan names (default: all 11)")
    ap.add_argument("--voxel", type=float, default=0.015, help="TSDF voxel length m (surface detail)")
    ap.add_argument("--sdf-trunc", type=float, default=0.06, help="TSDF truncation m (~4x voxel)")
    ap.add_argument("--trunc", type=float, default=4.0, help="max integrated depth m")
    ap.add_argument("--conf", type=int, default=1, help="min ARKit depth confidence 0/1/2 (1=med+high)")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--decimate", type=int, default=300000, help="target triangle count for the mesh (0=off)")
    ap.add_argument("--min-cluster-frac", type=float, default=0.02,
                    help="drop connected components smaller than this fraction of the largest")
    a = ap.parse_args()
    scenes = a.scenes or discover_scenes()
    if not scenes:
        print(f"no scenes with an extracted rgbd tree under {FINAL} — pass scan names explicitly")
        return 2
    print(f"== TSDF meshes for {len(scenes)} scene(s) · voxel={a.voxel*100:.1f}cm "
          f"sdf_trunc={a.sdf_trunc*100:.1f}cm conf>={a.conf} stride={a.stride} ==")
    ok = 0
    for s in scenes:
        try:
            if build(s, a.voxel, a.sdf_trunc, a.trunc, a.conf, a.stride, a.decimate, a.min_cluster_frac):
                ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {s}: {type(e).__name__}: {e}")
    print(f"== done: {ok}/{len(scenes)} meshes written ==")
    return 0 if ok == len(scenes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
