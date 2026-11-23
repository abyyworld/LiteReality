"""How close an imported phone video's poses and depth are to a LiDAR recording of the same room.

Both scans must already share one world (both aligned to the room's LiteReality Scanner scan);
`evaluate` checks that first (`same_world`) and refuses otherwise. The reference is ground truth:
ARKit poses and LiDAR depth (a Stray Scanner import aligned to ~1 cm). The two recordings are
different walks, so no video frame has a twin. The measures:

    reprojection  the headline pose + intrinsics measure, what photo-consistency (and so a splat)
              depends on: SIFT matches between each video photo and its nearest reference photos
              (RANSAC fundamental matrix, then kept only if a free-pose PnP with the frame's K
              agrees), the reference keypoints lifted to 3D with the LiDAR depth, projected with
              the video frame's pose AND K: the pixel distance to the video keypoint (per-frame
              median, then median / p90 over frames). Its floor is the reference's own
              photo / pose / depth consistency: a LiDAR scan scored this way against the Stray
              reference gets tens of pixels, not one.
    focal     the same 2D-3D matches, every frame re-posed by PnP for a grid of focal lengths:
              the focal with the lowest summed truncated reprojection error is the one the
              ground truth supports (independent of the import's poses).
    geometry  each video frame's confident depth registered onto the reference's fused LiDAR
              surface (point-to-plane ICP, 6 DoF, from where the import put it): the correction
              to its camera centre and rotation. This mixes pose, depth and K errors (a frame's
              cloud is lifted with its own K), so it is a geometry REGISTRATION error, not the
              pose error. Every frame counts: one that does not register (too few points within
              the capture radius) is a failure, counted at `ICP_START` (a lower bound on its
              error) in the all-frames statistics; one whose points constrain fewer than 6 DoF
              (a wall and the floor slide along their meeting line) is `degenerate`: its
              correction misses the unconstrained part, so it is reported apart.
    depth     the reference's fused LiDAR points projected into each video camera (z-buffered at
              the depth map's grid): |video depth - LiDAR depth| over the pixels both have,
              median per frame and over all, absolute and relative; `depth_raw_metric` is the
              same with the alignment's scale undone (the model's own metric depth, what a user
              without a LiDAR scan to align to gets).
    cloud     the video's whole fused cloud ICP'd onto the LiDAR (yaw + translation + scale): how
              far the import put the room as a whole.
    nearest_viewpoint  each video frame's nearest reference frame (camera centre distance +
              0.3 m per radian of view-direction difference) within `MATCH_DIST` and `MATCH_DEG`:
              indicative only. The two walks differ, the pairing is chosen with the estimated
              pose (biased low) and frames beyond the limits are left out (`unpaired`), so it
              is not a bound on the error.

`side_by_side` writes photo | video depth | LiDAR depth | |error| panels for a few frames.
`python -m litereality_agent.models.capture.video_eval SCAN REFERENCE` runs it on any two scans
in one world (e.g. a LiDAR scan against another for the method's floor).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from litereality_agent.models.capture import stray

MATCH_DIST = 0.5            # metres
MATCH_DEG = 25.0
ANGLE_WEIGHT = 0.3          # metres per radian
MIN_POINTS = 300
ICP_START, ICP_FINAL = 0.30, 0.03
# A frame's ICP is degenerate when the weakest direction of its (unit-balanced) 6x6 normal
# matrix carries less than this share of the strongest one.
DEGENERATE_RATIO = 0.02
REF_VOXEL = 0.01
MIN_OVERLAP = 0.2           # share of a frame's valid pixels with a LiDAR depth to count it
SAME_WORLD_DIST = 0.10      # metres: a video point this near the reference surface is on it
MIN_SAME_WORLD = 0.3        # share of the video cloud on the reference surface for one world
REPROJ_REFS = 3             # reference photos matched per video frame
REPROJ_DIST, REPROJ_DEG = 1.5, 45.0
REPROJ_MIN_POINTS = 10
PNP_TRUNCATE = 10.0         # px: reprojection errors are capped here in the focal search
FOCAL_FACTORS = tuple(float(v) for v in np.exp(np.linspace(np.log(0.6), np.log(2.0), 37)))


def load_frames(scan_dir: Path | str) -> tuple[list[stray.StrayFrame], tuple[int, int]]:
    """A scan's frames as StrayFrames with ARKit-world poses (OpenGL camera) and its photo size.
    `name` is the files' stem (frame_<name>.jpg), `row` the frame index."""
    from litereality_agent.models.splat import rgbd

    to_arkit = np.linalg.inv(rgbd.ARKIT_TO_Z_UP)
    frames, size = [], None
    for f in rgbd.load_frames(scan_dir):
        stem = Path(f.depth).stem[len("depth_"):]
        frames.append(stray.StrayFrame(f.index, stem, 0.0, to_arkit @ f.c2w, f.K, f.depth,
                                       f.conf))
        size = f.size
    if not frames:
        raise FileNotFoundError(f"{scan_dir}: no frames with depth")
    return frames, size


def photo_path(frame: stray.StrayFrame) -> Path:
    return Path(frame.depth).with_name(f"frame_{frame.name}.jpg")


def _angle(Ra, Rb):
    tr = np.einsum("...ij,...ij->...", Ra, Rb)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


# Quarter turns about the optical axis (z of an OpenGL camera). A phone video is usually
# shot upright (portrait frames) while ARKit / Stray store the sensor's landscape frame, so the
# same view differs by a 90-degree roll that is only an image-orientation convention.
_ROLLS = [np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
          for a in np.deg2rad([0.0, 90.0, 180.0, 270.0])]


def rotation_difference_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle between two camera rotations, ignoring a quarter-turn roll of the image."""
    return float(np.rad2deg(min(_angle(Ra @ Q, Rb) for Q in _ROLLS)))


def _nearest(f: stray.StrayFrame, rc: np.ndarray, rf: np.ndarray):
    d = np.linalg.norm(rc - f.c2w[:3, 3], axis=1)
    ang = np.arccos(np.clip(rf @ -f.c2w[:3, 2], -1.0, 1.0))
    return d, ang


def match_frames(video: list[stray.StrayFrame], ref: list[stray.StrayFrame]) -> list[dict]:
    """Nearest-viewpoint pairs (see the module docstring: indicative, not a bound)."""
    rc = np.stack([f.c2w[:3, 3] for f in ref])
    rf = np.stack([-f.c2w[:3, 2] for f in ref])
    out = []
    for i, f in enumerate(video):
        d, ang = _nearest(f, rc, rf)
        j = int(np.argmin(d + ANGLE_WEIGHT * ang))
        rot = rotation_difference_deg(f.c2w[:3, :3], ref[j].c2w[:3, :3])
        if d[j] <= MATCH_DIST and rot <= MATCH_DEG:
            out.append({"video": i, "ref": j, "position_m": float(d[j]), "rotation_deg": rot})
    return out


def _so3(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def register(src: np.ndarray, tgt: np.ndarray, tree, normals: np.ndarray, *,
             iterations: int = 30, start: float = ICP_START,
             final: float = ICP_FINAL) -> dict:
    """Point-to-plane ICP (rotation about the points' centroid + translation) of `src` onto the
    `tgt` surface: {"T" (4x4 moving src onto it), "matched" (points within the last match
    distance), "constraint" (smallest / largest eigenvalue of the last iteration's 6x6 normal
    matrix, the rotation columns divided by the points' RMS radius so both are in metres;
    near 0 = a direction the surface does not pin down), "residual_m" (median |plane distance|)}."""
    from litereality_agent.models.capture.align import _tukey

    T = np.eye(4)
    matched, constraint, residual = 0, 0.0, float("inf")
    for it in range(iterations):
        dist_max = max(final, start * (final / start) ** (it / max(iterations * 0.6, 1)))
        p = src @ T[:3, :3].T + T[:3, 3]
        d, idx = tree.query(p, distance_upper_bound=dist_max, workers=-1)
        ok = np.isfinite(d)
        matched = int(ok.sum())
        if matched < 30:
            constraint = 0.0
            break
        p, q, n = p[ok], tgt[idx[ok]], normals[idx[ok]]
        c = p.mean(0)
        r = np.einsum("ij,ij->i", n, p - q)
        J = np.concatenate([np.cross(p - c, n), n], 1)
        w = _tukey(r, dist_max)
        JW = J * w[:, None]
        H = JW.T @ J
        step = np.linalg.solve(H + 1e-6 * np.eye(6), -JW.T @ r)
        L = max(float(np.sqrt(np.mean(np.sum((p - c) ** 2, axis=1)))), 1e-3)
        S = np.diag([1.0 / L] * 3 + [1.0] * 3)
        ev = np.linalg.eigvalsh(S @ H @ S)
        constraint = float(ev[0] / max(ev[-1], 1e-300))
        residual = float(np.median(np.abs(r)))
        R = _so3(step[:3])
        D = np.eye(4)
        D[:3, :3] = R
        D[:3, 3] = c - R @ c + step[3:]
        T = D @ T
        if it > iterations * 0.6 and np.linalg.norm(step) < 1e-6:
            break
    return {"T": T, "matched": matched, "constraint": constraint, "residual_m": residual}


def icp_6dof(src: np.ndarray, tgt: np.ndarray, tree, normals: np.ndarray, *,
             iterations: int = 30, start: float = ICP_START,
             final: float = ICP_FINAL) -> tuple[np.ndarray, int]:
    """`register`'s transform and matched count."""
    r = register(src, tgt, tree, normals, iterations=iterations, start=start, final=final)
    return r["T"], r["matched"]


def frame_check(points: np.ndarray, c2w: np.ndarray, tgt: np.ndarray, tree,
                normals: np.ndarray) -> dict:
    """One frame's world points registered onto a surface: the correction to its camera
    (position_m, rotation_deg) and a status: "ok", "degenerate" (fewer than 6 DoF pinned down,
    the correction misses part of the error) or "failed" (too few points on the surface within
    the capture radius: its error is at least about `ICP_START`)."""
    if len(points) < MIN_POINTS:
        return {"status": "failed", "reason": "too few depth points", "points": int(len(points))}
    with np.errstate(all="ignore"):
        r = register(np.asarray(points, np.float64), tgt, tree, normals)
    if r["matched"] < MIN_POINTS:
        return {"status": "failed", "reason": "did not register", "points": r["matched"]}
    E, c = r["T"], c2w[:3, 3]
    return {"status": "ok" if r["constraint"] >= DEGENERATE_RATIO else "degenerate",
            "points": r["matched"], "constraint": round(r["constraint"], 4),
            "position_m": float(np.linalg.norm(E[:3, :3] @ c + E[:3, 3] - c)),
            "rotation_deg": float(np.rad2deg(_angle(E[:3, :3], np.eye(3)))),
            "residual_m": round(r["residual_m"], 4)}


def project_depth(points: np.ndarray, c2w_gl: np.ndarray, K: np.ndarray,
                  photo_size: tuple[int, int], grid: tuple[int, int]) -> np.ndarray:
    """World points rendered as a z-buffered depth map (metres, 0 = none) on a `grid` (w, h)
    covering the photo, for an OpenGL camera -> world pose and photo intrinsics `K`."""
    w, h = grid
    sx, sy = w / photo_size[0], h / photo_size[1]
    fx, fy = K[0, 0] * sx, K[1, 1] * sy
    cx, cy = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
    w2c = np.linalg.inv(c2w_gl)
    with np.errstate(all="ignore"):         # macOS Accelerate matmul raises spurious
        pc = points @ w2c[:3, :3].T + w2c[:3, 3]   # FP flags on finite data
    z = -pc[:, 2]
    ok = z > 0.1
    pc, z = pc[ok], z[ok]
    u = np.round(fx * pc[:, 0] / z + cx).astype(int)
    v = np.round(-fy * pc[:, 1] / z + cy).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[ok], v[ok], z[ok]
    depth = np.full(h * w, np.inf)
    np.minimum.at(depth, v * w + u, z)
    depth[~np.isfinite(depth)] = 0.0
    return depth.reshape(h, w)


def project_points(points: np.ndarray, c2w_gl: np.ndarray, K: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Photo pixels (n, 2) of world points for an OpenGL camera -> world pose, and which are in
    front of the camera."""
    w2c = np.linalg.inv(c2w_gl)
    with np.errstate(all="ignore"):
        pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = -pc[:, 2]
    zs = np.where(z > 1e-6, z, 1.0)
    uv = np.stack([K[0, 0] * pc[:, 0] / zs + K[0, 2], -K[1, 1] * pc[:, 1] / zs + K[1, 2]], 1)
    return uv, z > 0.05


def lift_pixels(frame: stray.StrayFrame, size: tuple[int, int], uv: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """World points of photo pixels `uv` (n, 2) from the frame's depth map (nearest cell,
    confident, off depth edges): (points (m, 3), the kept rows of uv)."""
    from litereality_agent.models.splat.rgbd import MAX_DEPTH, MIN_DEPTH, off_silhouettes

    depth, conf = stray.read_depth(frame)
    h, w = depth.shape
    ok_map = np.isfinite(depth) & (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    if conf is not None:
        ok_map &= conf >= stray.MIN_CONFIDENCE
    ok_map &= off_silhouettes(np.where(np.isfinite(depth), depth, 0.0))
    gu = np.round((uv[:, 0] + 0.5) * w / size[0] - 0.5).astype(int)
    gv = np.round((uv[:, 1] + 0.5) * h / size[1] - 0.5).astype(int)
    inside = (gu >= 0) & (gu < w) & (gv >= 0) & (gv < h)
    keep = np.zeros(len(uv), bool)
    keep[inside] = ok_map[gv[inside], gu[inside]]
    z = depth[gv[keep], gu[keep]].astype(np.float64)
    K = frame.K
    u, v = uv[keep, 0], uv[keep, 1]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z, -(v - K[1, 2]) / K[1, 1] * z, -z], 1)
    with np.errstate(all="ignore"):
        return cam @ frame.c2w[:3, :3].T + frame.c2w[:3, 3], keep


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def overlapping(f: stray.StrayFrame, vsize, ref: list[stray.StrayFrame], rsize, rc, rf, *,
                refs: int = REPROJ_REFS, rng=None, samples: int = 300) -> list[int]:
    """Up to `refs` reference frames (within `REPROJ_DIST` / `REPROJ_DEG`) that see the most of
    what video frame `f` sees: its depth points (a sample) projected into each, the share that
    lands in the photo. At least half must, or the frame is left out."""
    d, ang = _nearest(f, rc, rf)
    cand = np.nonzero((d <= REPROJ_DIST) & (np.rad2deg(ang) <= REPROJ_DEG))[0]
    if len(cand) == 0:
        return []
    with np.errstate(all="ignore"):
        pts = stray.frame_points(f, vsize, stride=2)
    if len(pts) < 30:
        return []
    rng = rng or np.random.default_rng(0)
    pts = pts[rng.choice(len(pts), min(samples, len(pts)), replace=False)].astype(np.float64)
    share = []
    for j in cand:
        uv, front = project_points(pts, ref[j].c2w, ref[j].K)
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < rsize[0]) & (uv[:, 1] >= 0) \
            & (uv[:, 1] < rsize[1])
        share.append(float(inside.mean()))
    order = np.argsort(share)[::-1]
    return [int(cand[k]) for k in order[:refs] if share[k] >= 0.5]


def correspondences(vid: list[stray.StrayFrame], vsize, ref: list[stray.StrayFrame], rsize, *,
                    refs: int = REPROJ_REFS, log=None) -> dict[int, dict]:
    """Per video frame: its SIFT keypoints matched in up to `refs` nearby reference photos,
    the reference side lifted to world points by the LiDAR depth: {i: {"uv" (n, 2) video photo
    px, "xyz" (n, 3) world}}."""
    from litereality_agent.models.capture import selfcal

    rc = np.stack([f.c2w[:3, 3] for f in ref])
    rf = np.stack([-f.c2w[:3, 2] for f in ref])
    ref_feats: dict[int, tuple] = {}
    out: dict[int, dict] = {}
    scale = max(vsize) / selfcal.FEATURE_LONG
    rng = np.random.default_rng(0)
    for i, f in enumerate(vid):
        near = overlapping(f, vsize, ref, rsize, rc, rf, refs=refs, rng=rng)
        if not near:
            continue
        fv = selfcal.features(_load_rgb(photo_path(f)))
        uvs, xyzs = [], []
        for j in near:
            if j not in ref_feats:
                ref_feats[j] = selfcal.features(_load_rgb(photo_path(ref[j])))
            xa, xb, F = selfcal.match(fv, ref_feats[j], scale=scale)
            if F is None or len(xa) < 8:
                continue
            xyz, keep = lift_pixels(ref[j], rsize, xb)
            uvs.append(xa[keep])
            xyzs.append(xyz)
        if uvs and sum(len(u) for u in uvs) >= REPROJ_MIN_POINTS:
            out[i] = {"uv": np.concatenate(uvs), "xyz": np.concatenate(xyzs), "refs": near}
        if log and (i + 1) % 50 == 0:
            log(f"  features: {i + 1}/{len(vid)} video frames")
    return out


def reprojection(vid: list[stray.StrayFrame], corr: dict[int, dict]) -> dict:
    """Pixel error of the lifted reference points projected with each video frame's pose and K,
    over the matches a free-pose PnP (RANSAC, 8 px, the frame's own K) keeps: wrong feature
    matches are left out without trusting the frame's pose."""
    rows = []
    for i, c in corr.items():
        X = np.ascontiguousarray(c["xyz"], np.float64)
        x = np.ascontiguousarray(c["uv"], np.float64)
        if len(X) < REPROJ_MIN_POINTS:
            continue
        verified = _pnp(X, x, vid[i].K, PNP_TRUNCATE)
        if verified is None or len(verified[3]) < REPROJ_MIN_POINTS:
            continue
        inl = verified[3]
        uv, front = project_points(X[inl], vid[i].c2w, vid[i].K)
        err = np.linalg.norm(uv - x[inl], axis=1)[front]
        if len(err) >= REPROJ_MIN_POINTS:
            rows.append({"frame": i, "points": int(len(err)), "matches": int(len(X)),
                         "median_px": float(np.median(err))})
    med = [r["median_px"] for r in rows]
    return {"frames": len(rows), "frames_without_matches": len(vid) - len(rows),
            "median_px": round(float(np.median(med)), 2) if med else None,
            "p90_px": round(float(np.percentile(med, 90)), 2) if med else None,
            "points": int(sum(r["points"] for r in rows)),
            "verified_share": round(sum(r["points"] for r in rows)
                                    / max(sum(r["matches"] for r in rows), 1), 3),
            "note": "per-frame median over PnP-verified matches, then median / p90 over frames",
            "per_frame": rows}


def _pnp(X: np.ndarray, x: np.ndarray, K: np.ndarray, truncate: float):
    """PnP (RANSAC, then LM on its inliers) of world points X onto pixels x: (mean truncated
    reprojection error over every match, OpenCV rvec, tvec, inliers) or None."""
    import cv2

    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, None, iterationsCount=300,
                                             reprojectionError=8.0, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 6:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(X[inl.ravel()], x[inl.ravel()], K, None, rvec, tvec)
    if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()):
        return None
    proj, _ = cv2.projectPoints(X, rvec, tvec, K, None)
    e = np.linalg.norm(proj.reshape(-1, 2) - x, axis=1)
    e = np.where(np.isfinite(e), e, np.inf)
    return float(np.mean(np.minimum(e, truncate))), rvec, tvec, inl.ravel(), e


def focal_search(vid: list[stray.StrayFrame], corr: dict[int, dict], *,
                 factors=FOCAL_FACTORS, truncate: float = PNP_TRUNCATE) -> dict:
    """The focal (fx = fy, the scan's principal point) that best explains the 2D-3D matches
    when every frame is re-posed by PnP: the mean over frames of the mean truncated
    reprojection error, over a grid around the scan's fx. At the best focal each well-fitting
    frame's PnP pose is the LiDAR's view of where that camera was: `pose_vs_pnp` compares the
    scan's poses with it (the actual camera pose error, independent of the scan's depth)."""
    import cv2

    usable = {i: c for i, c in corr.items() if len(c["uv"]) >= 12}
    if not usable:
        return {"focal": None}
    fx0 = float(np.median([vid[i].K[0, 0] for i in usable]))
    focals = fx0 * np.asarray(factors)
    costs = np.zeros(len(focals))
    data = {i: (np.ascontiguousarray(c["xyz"], np.float64), np.ascontiguousarray(c["uv"],
                                                                                np.float64))
            for i, c in usable.items()}

    def K_at(i, f):
        K0 = vid[i].K
        return np.array([[f, 0.0, K0[0, 2]], [0.0, f, K0[1, 2]], [0.0, 0.0, 1.0]])

    for k, f in enumerate(focals):
        for i, (X, x) in data.items():
            r = _pnp(X, x, K_at(i, f), truncate)
            costs[k] += truncate if r is None else r[0]
    costs /= len(data)
    best = int(np.argmin(costs))
    at_scan = int(np.argmin(np.abs(focals - fx0)))

    rows = []
    for i, (X, x) in data.items():
        r = _pnp(X, x, K_at(i, focals[best]), truncate)
        if r is None:
            continue
        _, rvec, tvec, inl, e = r
        if len(inl) < 20 or float(np.median(e[inl])) > 3.0:
            continue
        R, _ = cv2.Rodrigues(rvec)
        c = -R.T @ tvec.ravel()
        R_gl = (np.diag([1.0, -1.0, -1.0]) @ R).T
        rows.append({"frame": i, "position_m": float(np.linalg.norm(c - vid[i].c2w[:3, 3])),
                     "rotation_deg": float(np.rad2deg(_angle(R_gl, vid[i].c2w[:3, :3])))})
    p_med, p_p90 = _stats(rows, "position_m")
    return {"focal": round(float(focals[best]), 1), "scan_fx": round(fx0, 1),
            "ratio_to_scan": round(float(focals[best] / fx0), 4), "frames": len(data),
            "truncate_px": truncate,
            "cost_at_best_px": round(float(costs[best]), 3),
            "cost_at_scan_fx_px": round(float(costs[at_scan]), 3),
            "at_grid_edge": best in (0, len(focals) - 1),
            "pose_vs_pnp": {"frames": len(rows), "position_median_m": p_med,
                            "position_p90_m": p_p90,
                            "rotation_median_deg": _stats(rows, "rotation_deg")[0],
                            "note": "scan pose vs the PnP pose on LiDAR-lifted matches at the "
                                    "best focal (frames with >= 20 inliers, median < 3 px)"},
            "curve": [[round(float(f), 1), round(float(c), 3)]
                      for f, c in zip(focals[::2], costs[::2])]}


def _cloud(frames, size):
    with np.errstate(all="ignore"):
        return stray.fuse_points(frames, size).astype(np.float64)


def same_world(video_cloud: np.ndarray, ref_tree) -> float:
    """Share of the video's points within `SAME_WORLD_DIST` of the reference surface."""
    if len(video_cloud) == 0:
        return 0.0
    d, _ = ref_tree.query(video_cloud, distance_upper_bound=SAME_WORLD_DIST, workers=-1)
    return float(np.isfinite(d).mean())


def _aligned_to(scan_dir: Path) -> tuple[bool, str | None]:
    """(is a video import, the scan it was aligned to) from scene_alignment.json."""
    try:
        doc = json.loads((Path(scan_dir) / "scene_alignment.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, None
    vi = doc.get("videoImport") if isinstance(doc, dict) else None
    if not isinstance(vi, dict):
        return False, None
    return True, vi.get("alignedTo")


def _import_scale(scan_dir: Path) -> float | None:
    try:
        doc = json.loads((Path(scan_dir) / "scene_alignment.json").read_text(encoding="utf-8"))
        return float(doc["videoImport"]["scale"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def cloud_fit(video_cloud: np.ndarray, ref_cloud: np.ndarray, tree, normals) -> dict:
    """The video's fused cloud ICP'd onto the reference (yaw + translation + scale) from where
    the import put it: how far off the room lands as a whole."""
    from litereality_agent.models.capture import align as aligner
    from litereality_agent.models.capture import sim3

    src = stray.voxel_downsample(video_cloud, 0.03).astype(np.float64)
    if len(src) > sim3.SOURCE_POINTS:
        src = src[np.random.default_rng(0).choice(len(src), sim3.SOURCE_POINTS, replace=False)]
    with np.errstate(all="ignore"):
        before = aligner._measure(np.eye(4), src, ref_cloud, tree, normals)
        T = sim3.icp_sim_yaw(src, ref_cloud, tree, normals, np.eye(4), start=0.15)
        after = aligner._measure(T, src, ref_cloud, tree, normals)
    c = src.mean(0)
    return {"scale_correction": round(sim3.sim3_scale(T) - 1.0, 5),
            "yaw_deg": round(float(np.rad2deg(aligner._yaw_of(T))), 3),
            "centroid_shift_m": round(float(np.linalg.norm(sim3.apply(T, c[None])[0] - c)), 4),
            "residual_before_m": round(before["residual"], 4),
            "residual_after_m": round(after["residual"], 4),
            "inliers_before": round(before["inliers"], 3)}


def _stats(values, key=None):
    v = [r[key] for r in values] if key else list(values)
    if not v:
        return None, None
    return round(float(np.median(v)), 4), round(float(np.percentile(v, 90)), 4)


def evaluate(video_scan: Path | str, reference_scan: Path | str, *, viz_dir: Path | None = None,
             viz_count: int = 6, reprojection_check: bool = True, log=print) -> dict:
    from scipy.spatial import cKDTree

    from litereality_agent.models.capture.align import estimate_normals

    is_video, aligned_to = _aligned_to(Path(video_scan))
    if is_video and not aligned_to:
        raise ValueError(f"{video_scan} is a video import in its own world (no --align-to): it "
                         "does not share a world with any reference scan")
    vid, vsize = load_frames(video_scan)
    ref, rsize = load_frames(reference_scan)
    log(f"evaluating {len(vid)} frames of {video_scan} against {len(ref)} of {reference_scan}")
    ref_cloud = stray.fuse_points(ref, rsize, voxel=REF_VOXEL).astype(np.float64)
    tree = cKDTree(ref_cloud)
    with np.errstate(all="ignore"):
        normals = estimate_normals(ref_cloud, tree)
    video_cloud = _cloud(vid, vsize)
    shared = same_world(video_cloud, tree)
    if shared < MIN_SAME_WORLD:
        raise ValueError(f"only {shared:.0%} of {video_scan}'s points lie within "
                         f"{SAME_WORLD_DIST * 100:.0f} cm of {reference_scan}'s surface: the two "
                         "scans are not in one world (align both to the same scan first)")

    matches = match_frames(vid, ref)
    scale = _import_scale(Path(video_scan))

    pose_rows, depth_rows = [], []
    for i, f in enumerate(vid):
        depth, conf = stray.read_depth(f)
        grid = (depth.shape[1], depth.shape[0])
        with np.errstate(all="ignore"):
            pts = stray.frame_points(f, vsize, stride=1, max_depth=8.0).astype(np.float64)
        pose_rows.append({"frame": i, **frame_check(pts, f.c2w, ref_cloud, tree, normals)})
        lidar = project_depth(ref_cloud, f.c2w, f.K, vsize, grid)
        ok = (depth > 0) & (lidar > 0) & ((conf >= 2) if conf is not None else True)
        valid = int((depth > 0).sum())
        if valid and ok.sum() >= MIN_OVERLAP * valid and ok.sum() >= 50:
            err = np.abs(depth[ok] - lidar[ok])
            row = {"frame": i, "overlap": round(float(ok.sum() / valid), 3),
                   "abs_median_m": float(np.median(err)),
                   "rel_median": float(np.median(err / lidar[ok]))}
            if scale:
                raw = np.abs(depth[ok] / scale - lidar[ok])
                row["raw_abs_median_m"] = float(np.median(raw))
                row["raw_rel_median"] = float(np.median(raw / lidar[ok]))
            depth_rows.append(row)

    good = [r for r in pose_rows if r["status"] == "ok"]
    degen = [r for r in pose_rows if r["status"] == "degenerate"]
    failed = [r for r in pose_rows if r["status"] == "failed"]
    # every frame: a failure counts at the capture radius, a degenerate fit as measured (both
    # lower bounds on their true error)
    all_pos = [r.get("position_m", ICP_START) if r["status"] != "failed" else ICP_START
               for r in pose_rows]
    g_med, g_p90 = _stats(good, "position_m")
    a_med, a_p90 = _stats(all_pos)
    p90_censored = bool(failed) and a_p90 is not None and a_p90 >= ICP_START - 1e-9

    result = {
        "video_frames": len(vid), "reference_frames": len(ref),
        "reference_points": int(len(ref_cloud)), "same_world_share": round(shared, 3),
        "nearest_viewpoint": {
            "pairs": len(matches), "unpaired": len(vid) - len(matches),
            "position_median_m": _stats(matches, "position_m")[0],
            "rotation_median_deg": _stats(matches, "rotation_deg")[0],
            "limits": {"distance_m": MATCH_DIST, "rotation_deg": MATCH_DEG},
            "note": "indicative only: two different walks, paired by the estimated pose, "
                    "pairs beyond the limits left out"},
        "geometry_registration": {
            "note": "each frame's depth (lifted with its own K) ICP'd onto the LiDAR surface: "
                    "pose, depth and K errors mixed; not the pose error",
            "frames": len(vid), "ok": len(good), "degenerate": len(degen),
            "failed": len(failed), "capture_radius_m": ICP_START,
            "position_median_m": g_med, "position_p90_m": g_p90,
            "rotation_median_deg": _stats(good, "rotation_deg")[0],
            "all_frames": {"position_median_m": a_med, "position_p90_m": a_p90,
                           "p90_is_lower_bound": p90_censored,
                           "note": f"failed frames counted at {ICP_START} m, degenerate ones as "
                                   "measured: lower bounds"},
            "degenerate_position_median_m": _stats(degen, "position_m")[0],
            "degenerate_ratio": DEGENERATE_RATIO},
        "depth_vs_lidar": {"frames": len(depth_rows),
                           "abs_median_m": _stats(depth_rows, "abs_median_m")[0],
                           "rel_median": _stats(depth_rows, "rel_median")[0],
                           "overlap_median": _stats(depth_rows, "overlap")[0],
                           "note": "after the alignment's scale fit to the --align-to scan"},
        "cloud_vs_lidar": cloud_fit(video_cloud, ref_cloud, tree, normals),
    }
    if scale and depth_rows:
        result["depth_raw_metric"] = {
            "import_scale": round(scale, 5), "model_scale_error": round(1.0 / scale - 1.0, 4),
            "abs_median_m": _stats(depth_rows, "raw_abs_median_m")[0],
            "rel_median": _stats(depth_rows, "raw_rel_median")[0],
            "note": "the model's own metric depth (the alignment's scale undone) at the aligned "
                    "poses: what an import without --align-to gets"}
    if reprojection_check:
        corr = correspondences(vid, vsize, ref, rsize, log=log)
        rep = reprojection(vid, corr)
        result["reprojection"] = {k: v for k, v in rep.items() if k != "per_frame"}
        result["focal_vs_lidar"] = focal_search(vid, corr)
    if viz_dir is not None and depth_rows:
        best = sorted(depth_rows, key=lambda r: -r["overlap"])
        chosen = sorted(r["frame"] for r in best[:max(viz_count * 3, viz_count)])
        pick = [chosen[int(k)] for k in np.linspace(0, len(chosen) - 1, min(viz_count,
                                                                           len(chosen)))]
        result["visualisations"] = [str(side_by_side(vid[i], vsize, ref_cloud, Path(viz_dir)))
                                    for i in sorted(set(pick))]
    result["per_frame"] = {"geometry": pose_rows}
    return result


def _colour(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    import matplotlib

    cmap = matplotlib.colormaps["turbo"]
    x = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (cmap(x)[..., :3] * 255).astype(np.uint8)
    rgb[depth <= 0] = 0
    return rgb


def side_by_side(frame: stray.StrayFrame, photo_size: tuple[int, int], ref_cloud: np.ndarray,
                 out_dir: Path) -> Path:
    """photo | video depth | LiDAR depth (same colour scale) | |error| (0..0.3 m)."""
    from PIL import Image

    image = photo_path(frame)
    depth, _conf = stray.read_depth(frame)
    h, w = depth.shape
    lidar = project_depth(ref_cloud, frame.c2w, frame.K, photo_size, (w, h))
    both = (depth > 0) & (lidar > 0)
    vals = np.concatenate([depth[depth > 0], lidar[lidar > 0]])
    lo, hi = (np.percentile(vals, [2, 98]) if len(vals) else (0.0, 5.0))
    err = np.where(both, np.abs(depth - lidar), 0.0)
    scale = 2
    panels = []
    with Image.open(image) as im:
        panels.append(np.asarray(im.convert("RGB").resize((w * scale, h * scale))))
    for arr in (_colour(depth, lo, hi), _colour(lidar, lo, hi), _colour(err, 0.0, 0.3)):
        panels.append(np.asarray(Image.fromarray(arr).resize((w * scale, h * scale),
                                                            Image.Resampling.NEAREST)))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"depth_{frame.name}.png"
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)
    return path


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Score a scan's poses / depth / K against a LiDAR "
                                             "scan of the same room in the same world.")
    ap.add_argument("scan")
    ap.add_argument("reference")
    ap.add_argument("--json", help="write the result here")
    ap.add_argument("--viz-dir")
    ap.add_argument("--no-reprojection", action="store_true")
    a = ap.parse_args(argv)
    result = evaluate(Path(a.scan).expanduser(), Path(a.reference).expanduser(),
                      viz_dir=Path(a.viz_dir) if a.viz_dir else None,
                      reprojection_check=not a.no_reprojection)
    text = json.dumps({k: v for k, v in result.items() if k != "per_frame"}, indent=2)
    print(text)
    if a.json:
        Path(a.json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
