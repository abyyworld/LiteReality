"""RealityScan's reconstruction of the ordinary phone video, moved into the atrium's ARKit world
(atrium_1059's, the world of the Stray pool import and so of the 29 eval frames' poses).

The video has no ARKit, so nothing pairs its cameras with ARKit's: the similarity comes from the
geometry. RealityScan's sparse points against the Stray pool import's LiDAR cloud
(pointcloud.pcd), by the video importer's own tools (models/capture/sim3, as `import-video`):

    scale guess  RealityScan's unit is arbitrary: first ~metres from the cameras' median distance
                 to what they see (2 m), then, after levelling, the ratio of the cameras' median
                 height above the floor (2nd percentile of the points' height) to the Stray
                 walk's (same person, same room: a rough guess; `align_sim3` searches 0.7-1.4 x
                 around it)
    gravity      `sim3.estimate_up` from the cameras' mean up (horizontal surfaces' normals,
                 perpendicular to the vertical ones); `level_transform` turns it onto +Y
    sim3         `sim3.align_sim3`: coarse plan-view correlation over scales and yaw, then
                 point-to-plane ICP over yaw, translation and scale; the residual (median
                 point-to-plane distance), inlier share and heading margin are reported

As a cross-check, the same video frames' poses from the MapAnything import
(~/LiteReality-scans/atrium_phone, aligned to atrium_1059 the same way) are compared, frame by
frame where both picked the same video frame.

    uv run --frozen python scripts/realityscan/video_register.py [--scan ~/LiteReality-scans/rs_video]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rs_common import FLIP, POOL, REPO, angle_deg, read_model, save_sim3, stats

OUT = REPO / "results/realityscan/video"
MAPANY = Path("~/LiteReality-scans/atrium_phone").expanduser()


def main() -> None:
    from litereality_agent.models.capture import sim3
    from litereality_agent.models.capture import video as V

    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, default=Path("~/LiteReality-scans/rs_video"))
    ap.add_argument("--init", choices=("search", "priors"), default="search",
                    help="priors: start from the similarity onto the prior poses (the MapAnything "
                         "import's, same frames) and refine by ICP only")
    ap.add_argument("--tag", default="", help="suffix of the results/realityscan/video files")
    a = ap.parse_args()
    scan = a.scan.expanduser()
    sel = json.loads((scan / "rs_input/selection.json").read_text())["frames"]
    imgs, cams, pts = read_model(scan / "rs_out/colmap/sparse/0")
    names = sorted(imgs)
    s_total = None
    C = np.array([imgs[n].C for n in names])
    Rs = np.array([imgs[n].R for n in names])
    X = pts["xyz"]
    target = V.scan_cloud(POOL)
    floor_ar = float(np.percentile(target[:, 1], 2))
    if a.init == "priors":
        T, al, up_stats, scale_guess = _from_priors(imgs, names, sel, X, target, sim3, V)
    else:
        # ~metres: cameras see things ~2 m away in a room.
        depth = []
        id_index = {int(i): k for k, i in enumerate(pts["id"])}
        for n in names[::5]:
            im = imgs[n]
            ok = im.point_ids >= 0
            P = X[[id_index[int(i)] for i in im.point_ids[ok] if int(i) in id_index]]
            if len(P):
                depth.append(np.median((P - im.C) @ im.R[:, 2]))
        s_depth = 2.0 / float(np.median(depth))
        S = np.diag([s_depth] * 3 + [1.0])
        Xm, Cm = X * s_depth, C * s_depth
        guess = -Rs[:, :, 1].mean(0)                       # an OpenCV camera's up is -y
        up, up_stats = sim3.estimate_up(Xm, guess)
        level = sim3.level_transform(up, origin=Cm[0], forward=Rs[0][:, 2])
        L = level @ S
        Xl, Cl = sim3.apply(L, X), sim3.apply(L, C)
        floor_rs = float(np.percentile(Xl[:, 1], 2))
        h_rs = float(np.median(Cl[:, 1]) - floor_rs)
        pool_c = []
        for j in sorted(POOL.glob("frame_*.json")):
            pool_c.append(np.array(json.loads(j.read_text())["cameraPoseARFrame"], float)[[3, 7, 11]])
        pool_c = np.array(pool_c)
        h_ar = float(np.median(pool_c[:, 1]) - floor_ar)
        s_h = h_ar / h_rs
        H = np.diag([s_h] * 3 + [1.0])
        src = sim3.apply(H, Xl)
        # Far / stray points (through the windows, mismatches) would blow up the plan-view grids of
        # the coarse search: keep what lies within CLOUD_MAX_DEPTH of some camera, as import-video.
        from scipy.spatial import cKDTree
        near = cKDTree(sim3.apply(H, Cl)).query(src)[0] <= V.CLOUD_MAX_DEPTH
        src = src[near]
        print(f"{near.sum()} of {len(near)} points within {V.CLOUD_MAX_DEPTH} m of a camera")
        print(f"scale guess: depth {s_depth:.4f} x height {s_h:.4f} (camera {h_rs:.2f} m above the "
              f"floor -> {h_ar:.2f} m); gravity {up_stats.get('from_guess_deg')} deg from guess")
        al = sim3.align_sim3(src, target, log=print)
        T = al.transform @ H @ L                           # RS world -> ARKit world
        s_total = sim3.sim3_scale(T)
        print(json.dumps({k: v for k, v in al.summary().items() if k != "tries"}, indent=1))
        scale_guess = {"depth": s_depth, "height": s_h, "rs_camera_height_m": h_rs,
                       "stray_camera_height_m": h_ar}
        al = {k: v for k, v in al.summary().items() if k != "tries"}

    s_total = sim3.sim3_scale(T)

    # Cross-check against the MapAnything import of the same video (same world).
    ma = {}
    for j in sorted(MAPANY.glob("frame_*.json")):
        d = json.loads(j.read_text())
        m = np.array(d["cameraPoseARFrame"], float).reshape(4, 4)
        ma[int(d["video"]["video_frame"])] = (m[:3, :3] @ FLIP, m[:3, 3])
    pairs = []
    for n in names:
        vf = sel[n]["video_frame"]
        if vf in ma:
            c = sim3.apply(T, imgs[n].C[None])[0]
            Rw = T[:3, :3] / s_total @ imgs[n].R
            pairs.append({"name": n, "video_frame": vf,
                          "pos_m": float(np.linalg.norm(c - ma[vf][1])),
                          "rot_deg": angle_deg(Rw, ma[vf][0]), "c_rs": c, "c_ma": ma[vf][1],
                          "R_rs": Rw, "R_ma": ma[vf][0]})
    cross = None
    if len(pairs) >= 3:
        fit = sim3.robust_sim3(np.array([p["c_rs"] for p in pairs]),
                               np.array([p["c_ma"] for p in pairs]))
        after = [float(np.linalg.norm(fit["scale"] * fit["R"] @ p["c_rs"] + fit["t"] - p["c_ma"]))
                 for p in pairs]
        cross = {"pairs": len(pairs),
                 "as_registered": {"pos_m": stats([p["pos_m"] for p in pairs]),
                                   "rot_deg": stats([p["rot_deg"] for p in pairs])},
                 "after_own_sim3": {"pos_m": stats(after), "scale_ma_over_rs": fit["scale"],
                                    "rot_deg": stats([angle_deg(fit["R"] @ p["R_rs"], p["R_ma"])
                                                      for p in pairs])},
                 "mapanything_K_fx": json.loads(next(MAPANY.glob("frame_*.json")).read_text())
                 ["intrinsics"][0]}
    cam_rows = {cid: {"model": c.model, "w": c.w, "h": c.h, "fx": c.params[0], "fy": c.params[1],
                      "cx": c.params[2], "cy": c.params[3],
                      "dist": [float(v) for v in c.params[4:]]} for cid, c in list(cams.items())[:3]}
    res = {"registered": len(imgs), "input": len(sel), "rs_points": int(len(X)),
           "n_cameras": len(cams), "cameras": cam_rows,
           "init": a.init, "scale_guess": scale_guess,
           "gravity": up_stats, "alignment": al,
           "scale_rs_to_m": s_total, "vs_mapanything": cross,
           "camera_height_m_after": stats(sim3.apply(T, C)[:, 1] - floor_ar)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"registration{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    save_sim3(scan / "rs_to_arkit.json", T,
              {"how": "RealityScan sparse points -> Stray pool pointcloud.pcd: levelled "
                      "(estimate_up), scale guess, sim3.align_sim3 (yaw + translation + scale ICP)",
               "init": a.init, "residual_m": al["residual_m"], "inliers": al["inliers"],
               "ambiguous": al.get("ambiguous")})
    print(json.dumps({k: v for k, v in res.items() if k not in ("alignment",)}, indent=1,
                     default=float))



def _fit_quality(src: np.ndarray, T: np.ndarray, tgt, tree, normals, sim3) -> dict:
    """Median point-to-plane distance of the points within 10 cm of the target, the share of
    points within 10 cm (overlap) and within 3 cm (inliers)."""
    p = sim3.apply(T, src)
    d, idx = tree.query(p, distance_upper_bound=0.10, workers=-1)
    ok = np.isfinite(d)
    pp = np.abs(np.einsum("ij,ij->i", normals[idx[ok]], p[ok] - tgt[idx[ok]]))
    return {"residual_m": float(np.median(pp)) if ok.any() else None,
            "overlap": float(ok.mean()), "inliers": float((d < 0.03).mean()),
            "points": int(len(src))}


def _from_priors(imgs, names, sel, X, target, sim3, V):
    """RS world -> ARKit world, starting from the similarity onto the prior cameras (same video
    frames in the MapAnything import, already in atrium_1059's world), then point-to-plane ICP
    over yaw, translation and scale of RS's points onto the Stray pool's LiDAR cloud. The global
    search (`align_sim3`, gravity from the prior frame) is run too, as a check."""
    from scipy.spatial import cKDTree

    from litereality_agent.models.capture import align as aligner
    from litereality_agent.models.capture.stray import voxel_downsample

    ma = {}
    for j in sorted(MAPANY.glob("frame_*.json")):
        d = json.loads(j.read_text())
        m = np.array(d["cameraPoseARFrame"], float).reshape(4, 4)
        ma[int(d["video"]["video_frame"])] = m[:3, 3]
    both = [n for n in names if sel[n]["video_frame"] in ma]
    fit = sim3.robust_sim3(np.array([imgs[n].C for n in both]),
                           np.array([ma[sel[n]["video_frame"]] for n in both]))
    T0 = fit["T"]
    C = sim3.apply(T0, np.array([imgs[n].C for n in names]))
    src = sim3.apply(T0, X)
    near = cKDTree(C).query(src)[0] <= V.CLOUD_MAX_DEPTH
    src = src[near]
    tgt = voxel_downsample(target, 0.02).astype(np.float64)
    tree = cKDTree(tgt)
    normals = aligner.estimate_normals(tgt, tree)
    src_ds = voxel_downsample(src, 0.02).astype(np.float64)
    q0 = _fit_quality(src_ds, np.eye(4), tgt, tree, normals, sim3)
    D = sim3.icp_sim_yaw(src_ds, tgt, tree, normals, np.eye(4))
    q1 = _fit_quality(src_ds, D, tgt, tree, normals, sim3)
    glob_ = sim3.align_sim3(src_ds, target)
    moved = sim3.apply(D, C) - C
    check = sim3.apply(glob_.transform, C) - sim3.apply(D, C)
    print(f"priors fit: scale {fit['scale']:.4f}, {len(both)} cameras, median residual "
          f"{fit['median_residual']:.3f} m; ICP: {q0} -> {q1}; cameras moved median "
          f"{np.median(np.linalg.norm(moved, axis=1)):.3f} m; global search agrees within "
          f"{np.median(np.linalg.norm(check, axis=1)):.3f} m (inliers {glob_.inliers:.2f})")
    al = {"how": "Sim(3) onto the MapAnything prior cameras, then ICP (yaw, translation, scale) "
                 "of RS points onto the Stray pool LiDAR cloud",
          "prior_fit": {"scale": fit["scale"], "cameras": len(both),
                        "median_residual_m": fit["median_residual"]},
          "before_icp": q0, **q1, "icp_scale": sim3.sim3_scale(D),
          "icp_yaw_deg": float(np.degrees(np.arctan2(D[0, 2], D[0, 0]))),
          "icp_camera_shift_m": float(np.median(np.linalg.norm(moved, axis=1))),
          "global_search": {k: v for k, v in glob_.summary().items() if k != "tries"},
          "global_vs_icp_camera_m": float(np.median(np.linalg.norm(check, axis=1)))}
    return D @ T0, al, None, None


if __name__ == "__main__":
    main()
