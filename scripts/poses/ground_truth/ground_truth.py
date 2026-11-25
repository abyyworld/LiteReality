"""Reference camera poses (gt.json) for the pose benchmark's frames, from the room's LiDAR recording.

    uv run python scripts/poses/ground_truth/ground_truth.py [--cache DIR] [--workers 8]

Every benchmark frame (1080 x 1920 portrait, ~/LiteReality-scans/poses_bench/atrium/frames) is
localised against the iPhone LiDAR recording of the same room (Stray Scanner, ARKit poses + LiDAR
depth, aligned to atrium_1059 to about 1 cm):

  1. initial pose   the 2026-11-23 MapAnything import (atrium_phone, aligned to atrium_1059),
                    interpolated by source frame, used ONLY to choose LiDAR photos. Its rotations
                    agree with the final poses to a few degrees but its positions are 0.6-1 m off
                    (its fx is ~35% short, so its cameras sit too close), so pass 1 picks
                    references by view direction + F-verified SIFT match counts (retrieval), plus
                    ASIFT (OpenCV AffineFeature) matches for frames SIFT cannot pose: ASIFT only
                    supplies the pose guesses that choose references and gate later matching;
                    its half-resolution points never pose a final frame (they jitter by cm).
                    The SDR benchmark video has one duplicated frame (found by decoding it), so
                    its frame indices run one ahead of the original video's after it
                    (`original_index`).
  2. matches        SIFT (OpenCV, full resolution, RootSIFT, mutual nearest neighbours + ratio
                    test, MAGSAC fundamental matrix) between the frame and each reference photo;
                    the reference keypoints lifted to 3-D with the reference's confident LiDAR
                    depth (bilinear inside one surface, off depth edges) and its ARKit pose.
  3. later passes   references re-chosen geometrically (the fused LiDAR cloud rendered at the
                    LiDAR-PnP pose, checked for occlusion in each candidate photo) and matching
                    guided by that pose (40 px window); LiDAR points also carried along the video
                    (`propagate`: a posed frame's inlier keypoints keep their LiDAR 3-D points,
                    neighbouring frames matched to them reuse them; never triangulated).
  4. focal          ONE shared fx (= fy) for all frames: every frame re-posed by PnP-RANSAC + LM
                    for a grid of focals; the focal with the lowest mean truncated reprojection
                    error over all frames wins (curve reported), refined on a fine grid, then a
                    joint (fx, cx, cy, k1) check with the inlier sets fixed.
  5. poses          PnP-RANSAC + LM at that fx, principal point at the image centre; a frame is
                    kept with >= `MIN_INLIERS` distinct image points among the inliers and a
                    median inlier reprojection error < `MAX_MEDIAN_PX`, then dropped if it is more
                    than 3 cm / 1 deg off the interpolation of its kept neighbours.

Writes (in the benchmark folder): gt.json (OpenCV c2w, the room frame = atrium_1059's ARKit world,
Y up), gt_report.json, gt_checks/*.jpg (LiDAR depth edges and inlier matches drawn on the frame)
and poses_mapanything_old.json (the control: the old import's own poses for benchmark frames
within one source frame of an import frame).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

SCANS = Path.home() / "LiteReality-scans"
BENCH = SCANS / "poses_bench" / "atrium"
REF_SCANS = [SCANS / "stray_atrium_q600", SCANS / "stray_atrium"]
INIT_SCAN = SCANS / "atrium_phone"

SIFT_FEATURES = 8000
RATIO = 0.8
F_THRESHOLD = 1.5            # px (full resolution) MAGSAC fundamental-matrix threshold
MIN_PAIR_MATCHES = 15
REF_DIST, REF_DEG = 2.5, 70.0
MIN_SHARE = 0.15
RETRIEVAL_FEATURES = 1500
ASIFT_FEATURES = 3000         # per simulated view
ASIFT_KID = 1_000_000         # ASIFT keypoint ids are offset so they never meet SIFT ones
RETRIEVE_DEG, RETRIEVE_DIST, RETRIEVE_MIN = 40.0, 3.0, 15
PNP_RANSAC_PX = 4.0
INLIER_PX = 3.0
TRUNCATE_PX = 6.0
MIN_INLIERS = 40
MAX_MEDIAN_PX = 2.0
FOCAL_GRID = tuple(float(v) for v in np.exp(np.linspace(np.log(800), np.log(2400), 56)))
CV_GL = np.diag([1.0, -1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------------------------
# data

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def duplicated_frames(video: Path, size=(135, 240)) -> list[int]:
    """SDR frame indices that repeat the previous frame exactly (a conversion's inserted frame)."""
    w, h = size
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                        f"scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True, check=True)
    v = np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w).astype(np.float32)
    d = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2))
    # an exact repeat is ~0; the stillest genuine neighbours differ by > 0.4 grey levels
    return [int(i) + 1 for i in np.nonzero(d < 0.05)[0]]


def original_index(sdr_index: int, dups: list[int]) -> int:
    """The original (HDR) video frame an SDR frame shows."""
    return sdr_index - sum(1 for d in dups if d <= sdr_index)


def load_bench():
    meta = json.loads((BENCH / "frames.json").read_text())
    return meta, meta["frames"]


def load_refs():
    """Reference frames from both Stray imports of recording 901a305fdf, in the first import's
    world (the second is moved by the constant transform between the two imports' alignments)."""
    from litereality_agent.models.capture import video_eval

    refs, rows_seen, T, sizes = [], {}, None, {}
    for k, scan in enumerate(REF_SCANS):
        frames, size = video_eval.load_frames(scan)
        sizes[k] = size
        row_of = {}
        for f in frames:
            j = json.loads((scan / f"frame_{f.name}.json").read_text())
            row_of[f.name] = int(j["stray"]["row"])
        if k == 0:
            for f in frames:
                rows_seen[row_of[f.name]] = f.c2w
                refs.append({"key": f"q600_{f.name}", "scan": str(scan), "c2w": f.c2w, "K": f.K,
                             "depth": str(f.depth), "conf": str(f.conf) if f.conf else None,
                             "image": str(video_eval.photo_path(f)), "size": size,
                             "row": row_of[f.name]})
            continue
        common = [f for f in frames if row_of[f.name] in rows_seen]
        Ts = np.stack([rows_seen[row_of[f.name]] @ np.linalg.inv(f.c2w) for f in common])
        T = Ts[0]
        spread = float(np.abs(Ts - T).max())
        for f in frames:
            if row_of[f.name] in rows_seen:
                continue
            refs.append({"key": f"s206_{f.name}", "scan": str(scan), "c2w": T @ f.c2w, "K": f.K,
                         "depth": str(f.depth), "conf": str(f.conf) if f.conf else None,
                         "image": str(video_eval.photo_path(f)), "size": size,
                         "row": row_of[f.name]})
    info = {"frames": len(refs), "from_q600": sum(r["key"].startswith("q600") for r in refs),
            "from_stray_atrium": sum(r["key"].startswith("s206") for r in refs),
            "stray_atrium_to_q600": {"translation_m": np.round(T[:3, 3], 4).tolist(),
                                     "rotation_deg": round(float(np.degrees(np.arccos(
                                         np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))), 4),
                                     "max_spread": spread}}
    return refs, info


def load_init():
    """The old import's frames: original video index, OpenGL c2w (atrium_1059 ARKit world), K."""
    from litereality_agent.models.capture import video_eval

    frames, size = video_eval.load_frames(INIT_SCAN)
    out = []
    for f in frames:
        j = json.loads((INIT_SCAN / f"frame_{f.name}.json").read_text())
        out.append({"v": int(j["video"]["video_frame"]), "c2w": f.c2w, "K": f.K, "name": f.name})
    out.sort(key=lambda r: r["v"])
    return out, size


def interpolate_poses(vs: np.ndarray, poses: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Poses (4x4) at indices q by slerp / lerp between the given (sorted) ones; clamped at ends."""
    from scipy.spatial.transform import Rotation, Slerp

    slerp = Slerp(vs.astype(float), Rotation.from_matrix(poses[:, :3, :3]))
    qc = np.clip(q.astype(float), vs[0], vs[-1])
    out = np.tile(np.eye(4), (len(q), 1, 1))
    out[:, :3, :3] = slerp(qc).as_matrix()
    for d in range(3):
        out[:, d, 3] = np.interp(qc, vs, poses[:, d, 3])
    return out


# ---------------------------------------------------------------------------------------------
# depth

def read_depth(ref: dict):
    from PIL import Image

    from litereality_agent.models.splat.rgbd import MAX_DEPTH, MIN_DEPTH, off_silhouettes

    with Image.open(ref["depth"]) as im:
        depth = np.asarray(im, dtype=np.float32) / 1000.0
    ok = np.isfinite(depth) & (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    if ref["conf"]:
        with Image.open(ref["conf"]) as im:
            c = np.asarray(im)
        if c.shape == depth.shape:
            ok &= c >= 2
    ok &= off_silhouettes(np.where(np.isfinite(depth), depth, 0.0))
    return depth, ok


def lift(ref: dict, uv: np.ndarray, depth=None, ok=None):
    """World points of reference-photo pixels: depth bilinear when all four cells are confident
    and off edges, else the nearest confident cell; (points, kept mask)."""
    if depth is None:
        depth, ok = read_depth(ref)
    h, w = depth.shape
    W, H = ref["size"]
    gx = (uv[:, 0] + 0.5) * w / W - 0.5
    gy = (uv[:, 1] + 0.5) * h / H - 0.5
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx_, fy_ = gx - x0, gy - y0
    inside = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    z = np.full(len(uv), np.nan)
    xi, yi = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    o4 = ok[yi, xi] & ok[yi, xi + 1] & ok[yi + 1, xi] & ok[yi + 1, xi + 1] & inside
    d = (depth[yi, xi] * (1 - fx_) * (1 - fy_) + depth[yi, xi + 1] * fx_ * (1 - fy_)
         + depth[yi + 1, xi] * (1 - fx_) * fy_ + depth[yi + 1, xi + 1] * fx_ * fy_)
    z[o4] = d[o4]
    nu, nv = np.round(gx).astype(int), np.round(gy).astype(int)
    inn = (nu >= 0) & (nu < w) & (nv >= 0) & (nv < h) & ~o4
    nn = np.zeros(len(uv), bool)
    nn[inn] = ok[nv[inn], nu[inn]]
    z[nn] = depth[nv[nn], nu[nn]]
    keep = np.isfinite(z)
    K = ref["K"]
    u, v, zk = uv[keep, 0], uv[keep, 1], z[keep]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * zk, -(v - K[1, 2]) / K[1, 1] * zk, -zk], 1)
    c2w = ref["c2w"]
    return cam @ c2w[:3, :3].T + c2w[:3, 3], keep


def fused_cloud(refs: list[dict], voxel: float, stride: int = 2) -> np.ndarray:
    from litereality_agent.models.capture import stray

    pts = []
    for r in refs:
        depth, ok = read_depth(r)
        h, w = depth.shape
        sub = np.zeros_like(ok)
        sub[::stride, ::stride] = True
        vv, uu = np.nonzero(ok & sub)
        W, H = r["size"]
        uv = np.stack([(uu + 0.5) * W / w - 0.5, (vv + 0.5) * H / h - 0.5], 1)
        p, _ = lift(r, uv, depth, ok)
        pts.append(p)
        if len(pts) >= 60:
            pts = [stray.voxel_downsample(np.concatenate(pts), voxel)]
    return stray.voxel_downsample(np.concatenate(pts), voxel).astype(np.float64)


def project_gl(points, c2w_gl, K):
    w2c = np.linalg.inv(c2w_gl)
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = -pc[:, 2]
    zs = np.where(z > 1e-6, z, 1.0)
    uv = np.stack([K[0, 0] * pc[:, 0] / zs + K[0, 2], -K[1, 1] * pc[:, 1] / zs + K[1, 2]], 1)
    return uv, z


def render_depth(points, c2w_gl, K, size, grid):
    """z-buffered depth (0 = none) of `points` on a `grid` (w, h) covering a `size` photo."""
    W, H = size
    w, h = grid
    uv, z = project_gl(points, c2w_gl, K)
    gu = np.floor((uv[:, 0] + 0.5) * w / W).astype(int)
    gv = np.floor((uv[:, 1] + 0.5) * h / H).astype(int)
    ok = (z > 0.1) & (gu >= 0) & (gu < w) & (gv >= 0) & (gv < h)
    d = np.full(w * h, np.inf)
    np.minimum.at(d, gv[ok] * w + gu[ok], z[ok])
    d[~np.isfinite(d)] = 0
    return d.reshape(h, w)


# ---------------------------------------------------------------------------------------------
# reference choice

def bench_K(fx, size, cx=None, cy=None):
    W, H = size
    return np.array([[fx, 0, (W - 1) / 2 if cx is None else cx],
                     [0, fx, (H - 1) / 2 if cy is None else cy], [0, 0, 1.0]])


class RefIndex:
    def __init__(self, refs, cloud):
        self.refs = refs
        self.cloud = cloud
        self.centres = np.stack([r["c2w"][:3, 3] for r in refs])
        self.dirs = np.stack([-r["c2w"][:3, 2] for r in refs])
        self.depths = [read_depth(r) for r in refs]

    def choose(self, c2w_gl, K, size, n, rng):
        """Up to n reference indices that see most of the view (occlusion-checked), spread."""
        dep = render_depth(self.cloud, c2w_gl, K, size, (54, 96))
        vv, uu = np.nonzero(dep > 0)
        if len(vv) < 30:
            return [], []
        W, H = size
        uv = np.stack([(uu + 0.5) * W / 54 - 0.5, (vv + 0.5) * H / 96 - 0.5], 1)
        z = dep[vv, uu]
        cam = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0] * z, -(uv[:, 1] - K[1, 2]) / K[1, 1] * z,
                        -z], 1)
        pts = cam @ c2w_gl[:3, :3].T + c2w_gl[:3, 3]
        if len(pts) > 400:
            pts = pts[rng.choice(len(pts), 400, replace=False)]
        c, f = c2w_gl[:3, 3], -c2w_gl[:3, 2]
        dist = np.linalg.norm(self.centres - c, axis=1)
        ang = np.degrees(np.arccos(np.clip(self.dirs @ f, -1, 1)))
        cand = np.nonzero((dist <= REF_DIST) & (ang <= REF_DEG))[0]
        shares = []
        for j in cand:
            r = self.refs[j]
            depth, ok = self.depths[j]
            h, w = depth.shape
            puv, pz = project_gl(pts, r["c2w"], r["K"])
            Wr, Hr = r["size"]
            gu = np.round((puv[:, 0] + 0.5) * w / Wr - 0.5).astype(int)
            gv = np.round((puv[:, 1] + 0.5) * h / Hr - 0.5).astype(int)
            ins = (pz > 0.1) & (gu >= 0) & (gu < w) & (gv >= 0) & (gv < h)
            vis = np.zeros(len(pts), bool)
            dz = depth[gv[ins], gu[ins]]
            vis[ins] = np.abs(dz - pz[ins]) < 0.08 + 0.05 * pz[ins]
            shares.append(float(vis.mean()))
        order = np.argsort(shares)[::-1]
        chosen, chosen_share = [], []
        for k in order:
            if shares[k] < MIN_SHARE or len(chosen) >= n:
                break
            j = cand[k]
            near = any(np.linalg.norm(self.centres[j] - self.centres[i]) < 0.10
                       and np.degrees(np.arccos(np.clip(self.dirs[j] @ self.dirs[i], -1, 1))) < 6
                       for i in chosen)
            if not near:
                chosen.append(int(j))
                chosen_share.append(round(shares[k], 3))
        return chosen, chosen_share


# ---------------------------------------------------------------------------------------------
# features / matching (worker processes)

def _cv2():
    import cv2

    cv2.setNumThreads(1)
    return cv2


def feature_job(args):
    image, out = args
    if Path(out).is_file():
        return out
    cv2 = _cv2()
    grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    kps, desc = cv2.SIFT_create(nfeatures=SIFT_FEATURES).detectAndCompute(grey, None)
    pts = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
    if desc is None:
        desc = np.zeros((0, 128), np.float32)
    np.savez(out + ".tmp.npz", pts=pts, desc=np.clip(np.round(desc), 0, 255).astype(np.uint8))
    os.replace(out + ".tmp.npz", out)
    return out


def small_feature_job(args):
    """SIFT at half resolution (long side 960, 1500 features) for choosing reference photos."""
    image, out = args
    if Path(out).is_file():
        return out
    cv2 = _cv2()
    grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    h, w = grey.shape
    k = 960 / max(h, w)
    small = cv2.resize(grey, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    kps, desc = cv2.SIFT_create(nfeatures=RETRIEVAL_FEATURES).detectAndCompute(small, None)
    pts = np.array([kp.pt for kp in kps], np.float32).reshape(-1, 2)
    pts = (pts + 0.5) / k - 0.5
    if desc is None:
        desc = np.zeros((0, 128), np.float32)
    np.savez(out + ".tmp.npz", pts=pts, desc=np.clip(np.round(desc), 0, 255).astype(np.uint8))
    os.replace(out + ".tmp.npz", out)
    return out


def asift_feature_job(args):
    """Affine-simulated SIFT (OpenCV AffineFeature, ASIFT) at half resolution: survives the
    large viewpoint tilts between the low, close video and the standing LiDAR walk."""
    image, out = args
    if Path(out).is_file():
        return out
    cv2 = _cv2()
    grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(grey, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    af = cv2.AffineFeature_create(cv2.SIFT_create(nfeatures=ASIFT_FEATURES), maxTilt=4)
    kps, desc = af.detectAndCompute(small, None)
    pts = np.array([kp.pt for kp in kps], np.float32).reshape(-1, 2)
    pts = (pts + 0.5) * 2.0 - 0.5
    if desc is None:
        desc = np.zeros((0, 128), np.float32)
    np.savez(out + ".tmp.npz", pts=pts, desc=np.clip(np.round(desc), 0, 255).astype(np.uint8))
    os.replace(out + ".tmp.npz", out)
    return out


def retrieval_job(args):
    """F-verified one-to-one match counts of a frame against candidate reference photos."""
    bench_small, items = args
    cv2 = _cv2()
    pa, da = _feats(bench_small)
    out = []
    for j, path in items:
        pb, db = _feats(path)
        if len(da) < 8 or len(db) < 8:
            out.append((j, 0))
            continue
        ia, ib = _unguided(cv2, pa, da, pb, db)
        if len(ia) < 12:
            out.append((j, 0))
            continue
        F, mask = _fmat(cv2, pa[ia], pb[ib], 3.0, 0.999, 5000)
        out.append((j, 0 if mask is None or F is None else int(mask.sum())))
    return out


def retrieve(names, guesses, refs, index, bench_small, ref_small, pool, n_refs):
    """Reference photos for frames without a usable pose guess: candidates by view direction
    only (the old import's ROTATIONS agree with PnP to a few degrees; its positions do not),
    ranked by F-verified SIFT matches."""
    jobs = []
    for n in names:
        f = -guesses[n][:3, 2]
        ang = np.degrees(np.arccos(np.clip(index.dirs @ f, -1, 1)))
        dist = np.linalg.norm(index.centres - guesses[n][:3, 3], axis=1)
        cand = np.nonzero((ang <= RETRIEVE_DEG) & (dist <= RETRIEVE_DIST))[0]
        jobs.append((bench_small[n], [(int(j), ref_small[j]) for j in cand]))
    res = list(pool.map(retrieval_job, jobs, chunksize=1))
    sel = {}
    for n, r in zip(names, res):
        r = sorted(r, key=lambda t: -t[1])
        chosen, counts = [], []
        for j, c in r:
            if c < RETRIEVE_MIN or len(chosen) >= n_refs[n]:
                break
            near = any(np.linalg.norm(index.centres[j] - index.centres[i]) < 0.10 and
                       index.dirs[j] @ index.dirs[i] > np.cos(np.radians(6)) for i in chosen)
            if not near:
                chosen.append(j)
                counts.append(c)
        sel[n] = (chosen, counts)
    return sel


def _feats(path):
    """Keypoints and RootSIFT descriptors (L1-normalised, square-rooted: Hellinger distance)."""
    with np.load(path) as d:
        pts, desc = d["pts"].astype(np.float64), d["desc"].astype(np.float32)
    desc = np.sqrt(desc / np.maximum(desc.sum(1, keepdims=True), 1e-9))
    return pts, desc.astype(np.float32)


def _fmat(cv2, a, b, thr, conf, iters):
    """MAGSAC fundamental matrix; (None, None) when it fails (degenerate sets can assert)."""
    try:
        F, mask = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, thr, conf, iters)
    except cv2.error:
        return None, None
    if F is None or mask is None or F.shape != (3, 3):
        return None, None
    return F, mask


def _unguided(cv2, pa, da, pb, db):
    """Mutual nearest neighbours passing Lowe's ratio test (one-to-one)."""
    if max(len(da), len(db)) > 9000:          # ASIFT-sized sets: approximate (FLANN kd-trees)
        bf = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        knn = bf.knnMatch(da, db, k=2)
        back = [m[0] for m in bf.knnMatch(db, da, k=1) if m]
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        knn = bf.knnMatch(da, db, k=2)
        back = bf.match(db, da)
    back_of = np.full(len(db), -1)
    for m in back:
        back_of[m.queryIdx] = m.trainIdx
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < RATIO * m[1].distance
            and back_of[m[0].trainIdx] == m[0].queryIdx]
    return np.array([m.queryIdx for m in good], int), np.array([m.trainIdx for m in good], int)


def _guided(pa, da, pb, db, ref, guide):
    """Matches restricted to where the reference keypoint's LiDAR point projects in the frame
    at a guessed pose (from earlier PnP onto LiDAR points, never from the import): within
    `radius` px, ratio test among those candidates, one-to-one."""
    from scipy.spatial import cKDTree

    c2w_gl, K, radius, size = guide
    xyz, ok = lift(ref, pb)
    ids = np.nonzero(ok)[0]
    if len(ids) < 8:
        return np.zeros(0, int), np.zeros(0, int)
    uv, z = project_gl(xyz, c2w_gl, K)
    vis = (z > 0.1) & (uv[:, 0] > -radius) & (uv[:, 0] < size[0] + radius) \
        & (uv[:, 1] > -radius) & (uv[:, 1] < size[1] + radius)
    ids, uv = ids[vis], uv[vis]
    if len(ids) < 8:
        return np.zeros(0, int), np.zeros(0, int)
    sp = cKDTree(pa).sparse_distance_matrix(cKDTree(uv), radius, output_type="ndarray")
    if len(sp) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    i, jj = sp["i"].astype(int), sp["j"].astype(int)
    d = np.empty(len(i), np.float32)
    for s in range(0, len(i), 200000):
        d[s:s + 200000] = np.linalg.norm(da[i[s:s + 200000]] - db[ids[jj[s:s + 200000]]], axis=1)
    o = np.lexsort((d, i))
    i, jj, d = i[o], jj[o], d[o]
    first = np.r_[True, i[1:] != i[:-1]]
    fi = np.nonzero(first)[0]
    second = np.full(len(fi), np.inf, np.float32)
    has2 = np.r_[fi[1:], len(i)] - fi > 1
    second[has2] = d[fi[has2] + 1]
    keep = d[fi] < RATIO * second
    qa, qb, qd = i[fi][keep], ids[jj[fi][keep]], d[fi][keep]
    o = np.argsort(qd)                       # one-to-one: each reference keypoint once
    _, u = np.unique(qb[o], return_index=True)
    sel = o[u]
    return qa[sel], qb[sel]


def match_job(args):
    """One benchmark frame against its chosen references: (uv, xyz, kp id, ref id) arrays.
    `guide` None: mutual-NN + ratio; else guided by a LiDAR-derived pose guess."""
    bench_feat, ref_items, guide = args
    cv2 = _cv2()
    pa, da = _feats(bench_feat)
    uvs, xyzs, kids, rids, per_ref = [], [], [], [], []
    if len(da) < 8:
        return None
    for j, ref, ref_feat in ref_items:
        pb, db = _feats(ref_feat)
        if len(db) < 8:
            per_ref.append([j, 0, 0])
            continue
        ia, ib = _unguided(cv2, pa, da, pb, db) if guide is None else \
            _guided(pa, da, pb, db, ref, guide)
        if len(ia) < MIN_PAIR_MATCHES:
            per_ref.append([j, len(ia), 0])
            continue
        F, mask = _fmat(cv2, pa[ia], pb[ib], F_THRESHOLD, 0.9999, 10000)
        if F is None or mask is None or F.shape != (3, 3):
            per_ref.append([j, len(ia), 0])
            continue
        keep = mask.ravel().astype(bool)
        if keep.sum() < MIN_PAIR_MATCHES:
            per_ref.append([j, len(ia), int(keep.sum())])
            continue
        ia, ib = ia[keep], ib[keep]
        xyz, ok = lift(ref, pb[ib])
        per_ref.append([j, len(keep), int(keep.sum()), int(ok.sum())])
        uvs.append(pa[ia][ok])
        xyzs.append(xyz)
        kids.append(ia[ok])
        rids.append(np.full(int(ok.sum()), j))
    if not uvs:
        return {"uv": np.zeros((0, 2)), "xyz": np.zeros((0, 3)), "kid": np.zeros(0, int),
                "rid": np.zeros(0, int), "per_ref": per_ref}
    return {"uv": np.concatenate(uvs), "xyz": np.concatenate(xyzs), "kid": np.concatenate(kids),
            "rid": np.concatenate(rids), "per_ref": per_ref}


# ---------------------------------------------------------------------------------------------
# PnP

def pnp(X, x, K, dist=None, ransac_px=PNP_RANSAC_PX, inlier_px=INLIER_PX):
    """PnP-RANSAC + LM, re-inliered at `inlier_px` and refined again: dict or None."""
    cv2 = _cv2()
    if len(X) < 6:
        return None
    X = np.ascontiguousarray(X, np.float64)
    x = np.ascontiguousarray(x, np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, dist, iterationsCount=2000,
                                             reprojectionError=ransac_px, confidence=0.9999,
                                             flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 6:
        return None
    inl = inl.ravel()
    for _ in range(3):
        rvec, tvec = cv2.solvePnPRefineLM(X[inl], x[inl], K, dist, rvec, tvec)
        proj, _ = cv2.projectPoints(X, rvec, tvec, K, dist)
        e = np.linalg.norm(proj.reshape(-1, 2) - x, axis=1)
        e = np.where(np.isfinite(e), e, np.inf)
        R, _ = cv2.Rodrigues(rvec)
        front = (X @ R.T + tvec.ravel())[:, 2] > 0.05
        new = np.nonzero((e < inlier_px) & front)[0]
        if len(new) < 6:
            return None
        if np.array_equal(new, inl):
            break
        inl = new
    return {"rvec": rvec.ravel(), "tvec": tvec.ravel(), "inl": inl, "err": e}


def c2w_cv(rvec, tvec):
    cv2 = _cv2()
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ np.asarray(tvec, np.float64).ravel()
    return T


def position_sigma(X, x, K, rvec, tvec, sigma_px):
    """1-sigma camera-centre uncertainty (m, sqrt of the covariance trace) from the PnP Jacobian."""
    cv2 = _cv2()
    _, J = cv2.projectPoints(X, rvec, tvec, K, None)
    J = J[:, :6]
    try:
        cov_p = sigma_px ** 2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return float("inf")
    def centre(p):
        R, _ = cv2.Rodrigues(p[:3])
        return -R.T @ p[3:]
    p0 = np.concatenate([np.ravel(rvec), np.ravel(tvec)])
    Jc = np.zeros((3, 6))
    for k in range(6):
        d = np.zeros(6)
        d[k] = 1e-6
        Jc[:, k] = (centre(p0 + d) - centre(p0 - d)) / 2e-6
    return float(np.sqrt(np.trace(Jc @ cov_p @ Jc.T)))


def distinct(uv):
    """Distinct image points: one per 2 px cell (a corner found by SIFT and ASIFT, or matched in
    several reference photos, counts once)."""
    if len(uv) == 0:
        return 0
    return int(len(np.unique(np.floor(np.asarray(uv) / 2.0).astype(np.int64), axis=0)))


def frame_stats(res, uv):
    inl = res["inl"]
    e = res["err"][inl]
    return {"inliers": int(len(inl)), "distinct_inliers": distinct(uv[inl]),
            "median_px": float(np.median(e)), "mean_trunc_px": float(np.mean(np.minimum(
                res["err"], TRUNCATE_PX)))}


def focal_job(args):
    """One frame over a list of focals: (cost per focal, distinct inliers per focal)."""
    X, x, kid, focals, size = args
    costs, inls = [], []
    for f in focals:
        r = pnp(X, x, bench_K(f, size))
        if r is None:
            costs.append(TRUNCATE_PX)
            inls.append(0)
        else:
            costs.append(float(np.mean(np.minimum(r["err"], TRUNCATE_PX))))
            inls.append(distinct(x[r["inl"]]))
    return costs, inls


def focal_search(corr, size, pool, focals):
    names = [n for n, c in corr.items() if c is not None and len(c["uv"]) >= 30]
    jobs = [(corr[n]["xyz"], corr[n]["uv"], corr[n]["kid"], list(focals), size) for n in names]
    res = list(pool.map(focal_job, jobs, chunksize=4))
    C = np.array([r[0] for r in res])
    inl_n = np.array([r[1] for r in res])
    cost = C.mean(0)
    best = int(np.argmin(cost))
    return {"focals": [round(float(f), 2) for f in focals], "cost_px": cost.round(4).tolist(),
            "distinct_inliers": inl_n.sum(0).tolist(),
            "frames_ok": ((C < TRUNCATE_PX) & (inl_n >= MIN_INLIERS)).sum(0).tolist(),
            "frames": len(names), "best": round(float(focals[best]), 2),
            "best_by_inliers": round(float(focals[int(np.argmax(inl_n.sum(0)))]), 2),
            "at_grid_edge": best in (0, len(focals) - 1)}


def refine_intrinsics(corr, poses, size, fx0):
    """(fx, cx, cy, k1) minimising the robust inlier reprojection over all frames, each frame
    re-refined by LM with its inlier set fixed. Diagnostic for the principal point / distortion."""
    from scipy.optimize import minimize

    cv2 = _cv2()
    W, H = size
    data = []
    for n, p in poses.items():
        c = corr[n]
        inl = p["inl"]
        data.append((np.ascontiguousarray(c["xyz"][inl]), np.ascontiguousarray(c["uv"][inl]),
                     p["rvec"].reshape(3, 1), p["tvec"].reshape(3, 1)))

    def cost(q, use_k1):
        fx, cx, cy = q[0], q[1], q[2]
        k1 = q[3] if use_k1 else 0.0
        K = bench_K(fx, size, cx, cy)
        dist = np.array([k1, 0, 0, 0, 0], np.float64)
        tot, cnt = 0.0, 0
        for X, x, r, t in data:
            r2, t2 = cv2.solvePnPRefineLM(X, x, K, dist, r.copy(), t.copy())
            pr, _ = cv2.projectPoints(X, r2, t2, K, dist)
            e = np.linalg.norm(pr.reshape(-1, 2) - x, axis=1)
            tot += float(np.sum(np.where(e < 2, e ** 2 / 2, 2 * e - 2)))   # Huber, 2 px
            cnt += len(e)
        return tot / max(cnt, 1)

    c0 = ((W - 1) / 2, (H - 1) / 2)
    base = cost([fx0, *c0], False)
    r3 = minimize(lambda q: cost(q, False), [fx0, *c0], method="Nelder-Mead",
                  options={"xatol": 0.2, "fatol": 1e-5, "maxiter": 400,
                           "initial_simplex": [[fx0, *c0], [fx0 * 1.02, *c0],
                                               [fx0, c0[0] + 15, c0[1]], [fx0, c0[0], c0[1] + 15]]})
    r4 = minimize(lambda q: cost(q, True), [*r3.x, 0.0], method="Nelder-Mead",
                  options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 600,
                           "initial_simplex": [[*r3.x, 0.0], [r3.x[0] * 1.02, *r3.x[1:], 0.0],
                                               [r3.x[0], r3.x[1] + 15, r3.x[2], 0.0],
                                               [r3.x[0], r3.x[1], r3.x[2] + 15, 0.0],
                                               [*r3.x, 0.05]]})
    fx_only = minimize(lambda q: cost([q[0], *c0], False), [fx0], method="Nelder-Mead",
                       options={"xatol": 0.2, "fatol": 1e-6, "maxiter": 100})
    return {"frames": len(data), "huber_cost_at_start": round(base, 4),
            "fx_only_centre_pp": {"fx": round(float(fx_only.x[0]), 2),
                                  "cost": round(float(fx_only.fun), 4)},
            "fx_cx_cy": {"fx": round(float(r3.x[0]), 2), "cx": round(float(r3.x[1]), 2),
                         "cy": round(float(r3.x[2]), 2), "cost": round(float(r3.fun), 4)},
            "fx_cx_cy_k1": {"fx": round(float(r4.x[0]), 2), "cx": round(float(r4.x[1]), 2),
                            "cy": round(float(r4.x[2]), 2), "k1": round(float(r4.x[3]), 5),
                            "cost": round(float(r4.fun), 4)},
            "centre": [c0[0], c0[1]],
            "note": "Huber(2 px) mean of squared inlier reprojection, inlier sets fixed, every "
                    "frame re-refined by LM per evaluation"}


# ---------------------------------------------------------------------------------------------
# pipeline

def build_correspondences(names, guesses, K_guess, size, index, refs, bench_feat, ref_feat, pool,
                          n_refs, rng, radius=None, selection=None):
    """`radius` None: unguided matching; else {name: px or None} guided by `guesses`.
    `selection` {name: (ref indices, scores)} overrides the geometric choice for those frames."""
    jobs, chosen = [], {}
    for n in names:
        if selection is not None and n in selection:
            sel, share = selection[n]
        else:
            sel, share = index.choose(guesses[n], K_guess, size, n_refs[n], rng)
        chosen[n] = {"refs": [refs[j]["key"] for j in sel], "share": share}
        rad = None if radius is None else radius.get(n)
        guide = None if rad is None else (guesses[n], K_guess, rad, size)
        jobs.append((bench_feat[n], [(j, refs[j], ref_feat[j]) for j in sel], guide))
    res = list(pool.map(match_job, jobs, chunksize=2))
    corr = {}
    for n, r in zip(names, res):
        corr[n] = r
        if r is not None:
            chosen[n]["per_ref"] = r.pop("per_ref")
    return corr, chosen


def pose_all(corr, K, pool=None):
    out = {}
    for n, c in corr.items():
        if c is None or len(c["uv"]) < 6:
            continue
        r = pnp(c["xyz"], c["uv"], K)
        if r is None:
            continue
        r.update(frame_stats(r, c["uv"]))
        out[n] = r
    return out


def pair_job(args):
    """Mutual RootSIFT + MAGSAC F matches between two benchmark frames (cached)."""
    fa, fb, out = args
    if Path(out).is_file():
        return out
    cv2 = _cv2()
    pa, da = _feats(fa)
    pb, db = _feats(fb)
    ia, ib = _unguided(cv2, pa, da, pb, db) if len(da) >= 8 and len(db) >= 8 else ([], [])
    ia, ib = np.asarray(ia, int), np.asarray(ib, int)
    if len(ia) >= MIN_PAIR_MATCHES:
        F, mask = _fmat(cv2, pa[ia], pb[ib], F_THRESHOLD, 0.9999, 10000)
        keep = np.zeros(len(ia), bool) if mask is None else mask.ravel().astype(bool)
        ia, ib = ia[keep], ib[keep]
    else:
        ia, ib = ia[:0], ib[:0]
    np.savez(out + ".tmp.npz", ia=ia, ib=ib)
    os.replace(out + ".tmp.npz", out)
    return out


def _accept(r):
    return r is not None and r["distinct_inliers"] >= MIN_INLIERS and \
        r["median_px"] < MAX_MEDIAN_PX


def propagate(names, corr, P, pairs, pts, K, max_sweeps=30):
    """Poses from LiDAR points carried along the video: a posed frame's PnP-inlier keypoints
    keep their LiDAR-lifted 3-D points; a neighbouring frame matched to those keypoints gets
    the same 3-D points (2-D association through the video, the 3-D points always the LiDAR
    ones, never triangulated, so nothing drifts). Each frame is posed by PnP on its direct
    LiDAR matches plus the carried ones and kept by the same rule; repeated until nothing
    changes. Returns ({name: PnP result}, {name: combined correspondences})."""
    kp3d, res, comb = {}, {}, {}
    for n, r in P.items():
        if _accept(r):
            c = corr[n]
            inl = r["inl"]
            d = {}
            for k, x in zip(c["kid"][inl], c["xyz"][inl]):
                d.setdefault(int(k), x)
            kp3d[n] = d
            res[n] = dict(r, direct_inliers=r["distinct_inliers"], carried_inliers=0)
            comb[n] = dict(c, carried=np.zeros(len(c["uv"]), bool))
    nbrs = {n: [] for n in names}
    for (a, b), (ia, ib) in pairs.items():
        nbrs[a].append((b, ia, ib))
        nbrs[b].append((a, ib, ia))
    for sweep in range(max_sweeps):
        changed = 0
        for n in (names if sweep % 2 == 0 else names[::-1]):
            uv, xyz, kid = [], [], []
            for m, mine, theirs in nbrs[n]:
                if m not in kp3d or not len(mine):
                    continue
                d = kp3d[m]
                sel = [k for k in range(len(theirs)) if int(theirs[k]) in d]
                if not sel:
                    continue
                uv.append(pts[n][mine[sel]])
                xyz.append(np.stack([d[int(theirs[k])] for k in sel]))
                kid.append(mine[sel])
            if not uv:
                continue
            c = corr.get(n)
            nd = 0 if c is None else len(c["uv"])
            U = np.concatenate(([c["uv"]] if nd else []) + uv)
            X = np.concatenate(([c["xyz"]] if nd else []) + xyz)
            kids = np.concatenate(([c["kid"]] if nd else []) + kid).astype(int)
            carried = np.r_[np.zeros(nd, bool), np.ones(len(U) - nd, bool)]
            r = pnp(X, U, K)
            if r is None:
                continue
            r.update(frame_stats(r, U))
            if not _accept(r) or r["distinct_inliers"] <= res.get(n, {}).get(
                    "distinct_inliers", 0) + 2:
                continue
            inl = r["inl"]
            r["direct_inliers"] = distinct(U[inl][~carried[inl]])
            r["carried_inliers"] = distinct(U[inl][carried[inl]])
            dct = {}
            for k, x in zip(kids[inl], X[inl]):
                dct.setdefault(int(k), x)
            kp3d[n] = dct
            res[n] = r
            comb[n] = {"uv": U, "xyz": X, "kid": kids, "carried": carried}
            changed += 1
        log(f"  propagation sweep {sweep + 1}: {changed} frames (re)posed, {len(res)} posed")
        if not changed:
            break
    return res, comb


def guesses_from(poses_cv, names, order, init_guess):
    """OpenGL c2w guesses: the frame's own PnP pose, else interpolated between the nearest posed
    frames (within 90 source frames on both sides), else the old import's."""
    posed = [n for n in names if n in poses_cv]
    if len(posed) < 2:
        return dict(init_guess)
    idx = np.array([order[n] for n in posed])
    P = np.stack([poses_cv[n] @ CV_GL for n in posed])
    out = {}
    for n in names:
        if n in poses_cv:
            out[n] = poses_cv[n] @ CV_GL
            continue
        i = order[n]
        lo, hi = idx[idx < i], idx[idx > i]
        if len(lo) and len(hi) and i - lo.max() <= 90 and hi.min() - i <= 90:
            out[n] = interpolate_poses(idx, P, np.array([i]))[0]
        else:
            out[n] = init_guess[n]
    return out


def texture_of(path):
    cv2 = _cv2()
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return float(np.median(m)), float(np.mean(m))


def overlay(name, image, c2w_cv_, K, cloud, corr, res, out):
    """The frame with the LiDAR cloud's depth edges (magenta, rendered at the reference pose and
    fx) and the PnP inliers (green: keypoint, red line: to where its LiDAR point projects)."""
    cv2 = _cv2()
    img = cv2.imread(image)
    H, W = img.shape[:2]
    gw, gh = W // 6, H // 6
    dep = render_depth(cloud, c2w_cv_ @ CV_GL, K, (W, H), (gw, gh)).astype(np.float32)
    for _ in range(3):                      # fill pin-holes from the nearest neighbour
        big = np.where(dep > 0, dep, 1e9).astype(np.float32)
        filled = cv2.erode(big, np.ones((3, 3), np.uint8))
        dep = np.where(dep > 0, dep, np.where(filled < 1e8, filled, 0)).astype(np.float32)
    tint = np.zeros_like(img)
    valid = dep > 0
    if valid.any():
        lo, hi = np.percentile(dep[valid], [2, 98])
        col = cv2.applyColorMap(np.clip((dep - lo) / max(hi - lo, 1e-3) * 255, 0, 255)
                                .astype(np.uint8), cv2.COLORMAP_TURBO)
        col[~valid] = 0
        tint = cv2.resize(col, (W, H), interpolation=cv2.INTER_NEAREST)
    ld = np.log(np.where(valid, dep, 1.0))
    edge = np.zeros_like(valid)
    edge[:, 1:] |= (np.abs(np.diff(ld, axis=1)) > 0.10) & valid[:, 1:] & valid[:, :-1]
    edge[1:, :] |= (np.abs(np.diff(ld, axis=0)) > 0.10) & valid[1:, :] & valid[:-1, :]
    edge_big = cv2.resize(edge.astype(np.uint8) * 255, (W, H), interpolation=cv2.INTER_LINEAR)
    base = cv2.addWeighted(img, 0.7, tint, 0.3, 0)
    base[edge_big > 100] = (255, 0, 255)
    rv, tv = res["rvec"], res["tvec"]
    proj, _ = cv2.projectPoints(corr["xyz"][res["inl"]], rv, tv, K, None)
    for (u, v), (pu, pv) in zip(corr["uv"][res["inl"]], proj.reshape(-1, 2)):
        cv2.line(base, (int(u), int(v)), (int(pu), int(pv)), (0, 0, 255), 1)
        cv2.circle(base, (int(round(u)), int(round(v))), 3, (0, 255, 0), 1)
    cv2.putText(base, f"{name} inliers {res['distinct_inliers']} median {res['median_px']:.2f}px",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    side = np.concatenate([img, base], axis=1)
    side = cv2.resize(side, (side.shape[1] * 2 // 3, side.shape[0] * 2 // 3),
                      interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), side, [cv2.IMWRITE_JPEG_QUALITY, 88])


CONSIST_M, CONSIST_DEG = 0.03, 1.0


def consistency_filter(names, order, poses_cv):
    """Frames to drop: repeatedly the one deviating most (> CONSIST_M or > CONSIST_DEG) from
    the slerp/lerp of its kept neighbours, both within 8 source frames."""
    kept = [n for n in names if n in poses_cv]
    dropped = {}
    while True:
        worst, worst_score = None, 1.0
        for a, b, c in zip(kept, kept[1:], kept[2:]):
            ia, ib, ic = order[a], order[b], order[c]
            if ib - ia > 8 or ic - ib > 8:
                continue
            P = interpolate_poses(np.array([ia, ic]), np.stack([poses_cv[a], poses_cv[c]]),
                                  np.array([ib]))[0]
            dp = float(np.linalg.norm(P[:3, 3] - poses_cv[b][:3, 3]))
            d = P[:3, :3].T @ poses_cv[b][:3, :3]
            dr = float(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
            score = max(dp / CONSIST_M, dr / CONSIST_DEG)
            if score > worst_score:
                worst, worst_score = b, score
                worst_dev = (round(dp, 4), round(dr, 3))
        if worst is None:
            return dropped
        dropped[worst] = {"position_m": worst_dev[0], "rotation_deg": worst_dev[1]}
        kept.remove(worst)


def unchecked(names, order, kept):
    """Kept frames the consistency check could not test (no kept frame within 8 source frames
    on one side or both)."""
    k = [n for n in names if n in kept]
    out = []
    for i, n in enumerate(k):
        lo = i > 0 and order[n] - order[k[i - 1]] <= 8
        hi = i + 1 < len(k) and order[k[i + 1]] - order[n] <= 8
        if not (lo and hi):
            out.append(n)
    return {"frames": len(out), "names": out}


def smoothness(names, order, poses_cv):
    """Deviation of each kept pose from the time-interpolation of its kept neighbours (both
    within 8 source frames): an upper bound on per-frame jitter (it includes real motion)."""
    pos, rot = [], []
    posed = [n for n in names if n in poses_cv]
    for a, b, c in zip(posed, posed[1:], posed[2:]):
        ia, ib, ic = order[a], order[b], order[c]
        if ic - ia > 16 or ib - ia > 8 or ic - ib > 8:
            continue
        P = interpolate_poses(np.array([ia, ic]), np.stack([poses_cv[a], poses_cv[c]]),
                              np.array([ib]))[0]
        pos.append(float(np.linalg.norm(P[:3, 3] - poses_cv[b][:3, 3])))
        d = P[:3, :3].T @ poses_cv[b][:3, :3]
        rot.append(float(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1)))))
    if not pos:
        return {"triplets": 0}
    return {"triplets": len(pos), "position_median_m": round(float(np.median(pos)), 4),
            "position_p90_m": round(float(np.percentile(pos, 90)), 4),
            "rotation_median_deg": round(float(np.median(rot)), 3),
            "note": "|pose - interpolation of its neighbours (<= 8 source frames each side)|: "
                    "noise plus real acceleration, so an upper bound on jitter"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "litereality_gt_cache"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--refs", type=int, default=4)
    ap.add_argument("--checks", type=int, default=6)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--neighbours", type=int, default=4,
                    help="frame pairs along the video: each frame with the next N")
    ap.add_argument("--every", type=int, default=1, help="test on every n-th frame only")
    ap.add_argument("--out", default=str(BENCH), help="where gt.json etc. are written")
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cache = Path(args.cache)
    (cache / "feat").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    meta, bench = load_bench()
    bench = bench[::args.every]
    size = tuple(meta["size"])
    names = [f["name"] for f in bench]
    dups = duplicated_frames(Path(meta["video"]))
    orig = {f["name"]: original_index(f["source_index"], dups) for f in bench}
    order = {f["name"]: f["source_index"] for f in bench}
    log(f"{len(names)} benchmark frames; SDR duplicated frames at {dups}")

    refs, ref_info = load_refs()
    log(f"{len(refs)} LiDAR reference frames")
    init, init_size = load_init()
    init_v = np.array([r["v"] for r in init])
    init_P = np.stack([r["c2w"] for r in init])
    ig = interpolate_poses(init_v, init_P, np.array([orig[n] for n in names]))
    init_guess = {n: ig[k] for k, n in enumerate(names)}
    gap = {n: int(np.min(np.abs(init_v - orig[n]))) for n in names}

    # features
    bench_feat = {n: str(cache / "feat" / f"bench_{n[:-4]}.npz") for n in names}
    ref_feat = [str(cache / "feat" / f"ref_{r['key']}.npz") for r in refs]
    jobs = [(str(BENCH / "frames" / n), bench_feat[n]) for n in names] + \
           [(r["image"], ref_feat[j]) for j, r in enumerate(refs)]
    bench_small = {n: str(cache / "feat" / f"small_bench_{n[:-4]}.npz") for n in names}
    ref_small = [str(cache / "feat" / f"small_ref_{r['key']}.npz") for r in refs]
    small_jobs = [(str(BENCH / "frames" / n), bench_small[n]) for n in names] + \
                 [(r["image"], ref_small[j]) for j, r in enumerate(refs)]
    with ProcessPoolExecutor(args.workers) as pool:
        log(f"SIFT on {len(jobs)} images")
        list(pool.map(feature_job, jobs, chunksize=4))
        list(pool.map(small_feature_job, small_jobs, chunksize=4))
        bench_af = {n: str(cache / "feat" / f"asift_bench_{n[:-4]}.npz") for n in names}
        ref_af = [str(cache / "feat" / f"asift_ref_{r['key']}.npz") for r in refs]
        log("ASIFT features")
        list(pool.map(asift_feature_job,
                      [(str(BENCH / "frames" / n), bench_af[n]) for n in names] +
                      [(r["image"], ref_af[j]) for j, r in enumerate(refs)], chunksize=2))
        log("fusing the LiDAR cloud")
        cloud = fused_cloud(refs, 0.03)
        index = RefIndex(refs, cloud)

        # pass 1: references retrieved by F-verified SIFT matches among the LiDAR photos facing
        # the way the old import says the frame faced (its positions are 0.6-1 m off: its fx is
        # ~35% short, so its cameras sit too close); unguided matching
        fx_init = float(np.median([r["K"][0, 0] for r in init])) * size[0] / init_size[0]
        K_guess = bench_K(fx_init, size)
        sel1 = retrieve(names, init_guess, refs, index, bench_small, ref_small, pool,
                        {n: args.refs for n in names})
        log(f"retrieval: {sum(1 for v in sel1.values() if v[0])} frames with references")
        corr, chosen = build_correspondences(names, init_guess, K_guess, size, index, refs,
                                             bench_feat, ref_feat, pool,
                                             {n: args.refs for n in names}, rng,
                                             selection=sel1)
        # ASIFT supplement for frames SIFT could not pose at a plausible focal: references from
        # retrieval plus the LiDAR photos that overlap the import's own view (pose AND its fx)
        P0 = pose_all(corr, bench_K(1550.0, size))
        weak = [n for n in names if not _accept(P0.get(n))]
        corr_a = {}
        jobs_a = []
        for n in weak:
            geo, _ = index.choose(init_guess[n], K_guess, size, args.refs, rng)
            sel = list(dict.fromkeys(list(sel1[n][0]) + geo))[:2 * args.refs]
            jobs_a.append((bench_af[n], [(j, refs[j], ref_af[j]) for j in sel], None))
        log(f"ASIFT matching for {len(weak)} frames")
        for n, r in zip(weak, pool.map(match_job, jobs_a, chunksize=1)):
            if r is not None and len(r["uv"]):
                r.pop("per_ref", None)
                r["kid"] = r["kid"] + ASIFT_KID
                corr_a[n] = r

        def full(c_sift):
            out_ = {}
            for n in names:
                parts = [x for x in (c_sift.get(n), corr_a.get(n)) if x is not None and len(x["uv"])]
                out_[n] = None if not parts else {
                    k: np.concatenate([x[k] for x in parts]) for k in ("uv", "xyz", "kid", "rid")}
            return out_

        corr_sift = corr
        corr = full(corr_sift)
        fs = focal_search(corr, size, pool, FOCAL_GRID)
        fx = fs["best"]
        P = pose_all(corr, bench_K(fx, size))
        kept = {n: c2w_cv(r["rvec"], r["tvec"]) for n, r in P.items() if _accept(r)}
        log(f"pass 1 with ASIFT: {sum(1 for n in weak if n in kept)} of {len(weak)} weak frames")
        passes = [{"pass": 1, "asift_frames": len(corr_a), "guided": False, "focal": fx, "kept_direct": len(kept)}]
        log(f"pass 1: focal {fx}, kept {len(kept)}")
        fs1 = fs

        # frame-to-frame matches along the video (for carrying LiDAR points)
        pair_paths = {}
        for k, n in enumerate(names):
            for m in names[k + 1:k + 1 + args.neighbours]:
                pair_paths[(n, m)] = str(cache / "feat" / f"pair_{n[:-4]}_{m[:-4]}.npz")
        log(f"matching {len(pair_paths)} frame pairs along the video")
        list(pool.map(pair_job, [(bench_feat[a_], bench_feat[b_], p_)
                                 for (a_, b_), p_ in pair_paths.items()], chunksize=4))
        pairs = {}
        for key_, p_ in pair_paths.items():
            with np.load(p_) as d:
                pairs[key_] = (d["ia"], d["ib"])
        pts = {n: _feats(bench_feat[n])[0] for n in names}

        # later passes: carry LiDAR points along the video, then re-choose references and
        # match guided by those LiDAR-PnP poses (posed: 40 px; interpolated: 120 px)
        for it in range(2, args.passes + 1):
            Pp, _ = propagate(names, corr, P, pairs, pts, bench_K(fx, size))
            posed = {n: c2w_cv(r["rvec"], r["tvec"]) for n, r in Pp.items()}
            guesses = guesses_from(posed, names, order, init_guess)
            interp = {n for n in names if n not in posed and
                      not np.allclose(guesses[n], init_guess[n])}
            radius = {n: 40.0 if n in posed else (120.0 if n in interp else None)
                      for n in names}
            n_refs = {n: args.refs if n in kept else args.refs + 2 for n in names}
            unguided = {n: sel1[n] for n in names if radius[n] is None}
            corr_new, chosen_new = build_correspondences(
                names, guesses, bench_K(fx, size), size, index, refs, bench_feat, ref_feat,
                pool, n_refs, rng, radius=radius, selection=unguided)
            # a frame keeps whichever pass's direct matches pose it with more distinct inliers
            P_new = pose_all(corr_new, bench_K(fx, size))
            P_old = pose_all(corr_sift, bench_K(fx, size))
            for n in names:
                a_ = P_old.get(n, {}).get("distinct_inliers", 0)
                b_ = P_new.get(n, {}).get("distinct_inliers", 0)
                if b_ >= a_ and corr_new.get(n) is not None:
                    corr_sift[n], chosen[n] = corr_new[n], chosen_new[n]
            corr = full(corr_sift)
            fs = focal_search(corr, size, pool, FOCAL_GRID)
            fx = fs["best"]
            P = pose_all(corr, bench_K(fx, size))
            kept = {n: c2w_cv(r["rvec"], r["tvec"]) for n, r in P.items() if _accept(r)}
            passes.append({"pass": it, "guided": True, "focal": fx, "kept_direct": len(kept),
                           "posed_with_carried_before_matching": len(posed),
                           "guided_own": sum(1 for v in radius.values() if v == 40.0),
                           "guided_interpolated": sum(1 for v in radius.values() if v == 120.0)})
            log(f"pass {it}: focal {fx}, kept direct {len(kept)} (carried poses {len(posed)})")
        # final: SIFT only (full-resolution keypoints; ASIFT only helped choose references and
        # the guess that gated the guided matching, its half-resolution points never pose a
        # final frame)
        corr = {n: c for n, c in corr_sift.items()}
        fs2 = focal_search(corr, size, pool, FOCAL_GRID)
        b = fs2["best"]
        fine = tuple(b * np.exp(np.linspace(np.log(0.95), np.log(1.05), 41)))
        fs_fine = focal_search(corr, size, pool, fine)
        fx = fs_fine["best"]
        log(f"final focal (SIFT only) coarse {b}, fine {fx}")
        P_direct = pose_all(corr, bench_K(fx, size))
        P, corr_all = propagate(names, corr, P_direct, pairs, pts, bench_K(fx, size))
    good = {n: r for n, r in P.items() if _accept(r)}
    # consistency along the video: a pose more than CONSIST_M / CONSIST_DEG from the
    # interpolation of its kept neighbours (each within 8 source frames: real acceleration
    # there is millimetres) is dropped, worst first
    rejected = consistency_filter(names, order, {n: c2w_cv(r["rvec"], r["tvec"])
                                                 for n, r in good.items()})
    for n in rejected:
        good.pop(n)
    log(f"consistency check dropped {len(rejected)}: {[n for n in rejected]}")
    direct_only = {n for n in good if good[n]["carried_inliers"] == 0}
    direct_strong = {n for n in good if good[n]["direct_inliers"] >= MIN_INLIERS}
    log(f"kept {len(good)} / {len(names)} ({len(direct_strong)} with >= {MIN_INLIERS} direct "
        f"LiDAR inliers)")
    intr = refine_intrinsics(corr_all, good, size, fx)
    corr = corr_all
    log(f"intrinsics check: {intr['fx_cx_cy']} {intr['fx_cx_cy_k1']}")

    K = bench_K(fx, size)
    frames_out, per_frame = {}, {}
    for n in names:
        row = {"orig_index": orig[n], "source_index": order[n], "init_gap_frames": gap[n],
               "refs": chosen.get(n, {}).get("refs", []),
               "matches": 0 if corr.get(n) is None else int(len(corr[n]["uv"]))}
        r = P.get(n)
        if r is not None:
            row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                        if k in ("inliers", "distinct_inliers", "median_px", "mean_trunc_px",
                                 "direct_inliers", "carried_inliers")})
        rd = P_direct.get(n)
        row["direct_only_pnp"] = None if rd is None else {
            "distinct_inliers": rd["distinct_inliers"], "median_px": round(rd["median_px"], 3)}
        if n in good:
            c = corr[n]
            inl = r["inl"]
            sig = position_sigma(c["xyz"][inl], c["uv"][inl], K, r["rvec"], r["tvec"],
                                 max(1.4826 * r["median_px"], 0.3))
            row["sigma_pos_m"] = round(sig, 4)
            row["kept"] = True
            frames_out[n] = {"c2w": np.round(c2w_cv(r["rvec"], r["tvec"]), 6).tolist(),
                             "inliers": r["distinct_inliers"],
                             "direct_inliers": r["direct_inliers"],
                             "carried_inliers": r["carried_inliers"],
                             "median_px": round(r["median_px"], 3),
                             "sigma_pos_m": round(sig, 4)}
        else:
            row["kept"] = False
            if n in rejected:
                row["dropped_inconsistent"] = rejected[n]
        row["texture_median_grad"], row["texture_mean_grad"] = texture_of(
            str(BENCH / "frames" / n))
        per_frame[n] = row

    poses_cv = {n: np.array(v["c2w"]) for n, v in frames_out.items()}
    seconds = round(time.time() - t0, 1)
    gt = {"method": "lidar_reference",
          "licence": "OpenCV SIFT/PnP (Apache-2.0) / no weights",
          "metric": True, "fx": round(float(fx), 2),
          "cx": (size[0] - 1) / 2, "cy": (size[1] - 1) / 2, "size": list(size),
          "world": "atrium_1059 ARKit world (Y up, metres), the frame of stray_atrium_q600",
          "frames": frames_out, "seconds": seconds, "gpu_seconds": 0.0, "cost_usd": 0.0,
          "notes": (f"Each frame PnP-RANSAC + LM on SIFT matches to up to {args.refs} LiDAR "
                    f"(Stray) photos lifted to 3-D with their LiDAR depth and ARKit pose; one "
                    f"shared fx by focal search; kept with >= {MIN_INLIERS} distinct inliers "
                    f"(< {INLIER_PX} px) and median < {MAX_MEDIAN_PX} px, and consistent with "
                    f"neighbouring frames (3 cm / 1 deg). ASIFT and the old import only chose "
                    f"reference photos. Frames short of direct "
                    f"matches also use LiDAR points carried along the video from neighbouring "
                    f"frames' inliers (per-frame direct_inliers / carried_inliers). c2w OpenCV "
                    f"axes.")}
    (OUT / "gt.json").write_text(json.dumps(gt, indent=1))

    # texture coverage
    tex = np.array([per_frame[n]["texture_median_grad"] for n in names])
    bins = np.empty(len(tex), int)
    bins[np.argsort(tex, kind="stable")] = np.arange(len(tex)) * 4 // len(tex)
    coverage = []
    for k in range(4):
        sel = [n for n, bb in zip(names, bins) if bb == k]
        if not sel:
            continue
        coverage.append({"quartile": k + 1,
                         "median_grad_range": [round(float(tex[bins == k].min()), 2),
                                               round(float(tex[bins == k].max()), 2)],
                         "frames": len(sel), "kept": sum(per_frame[n]["kept"] for n in sel),
                         "kept_ge40_direct": sum(1 for n in sel if n in direct_strong)})
    low = [n for n in names if per_frame[n]["texture_median_grad"] < 2.0]

    # time coverage (seconds of the video without a reference pose)
    kept_idx = sorted(order[n] for n in good)
    gaps = np.diff([0] + kept_idx + [meta["decoded"] - 1]) / meta["fps"]
    med = [per_frame[n]["median_px"] for n in good]
    inls = [per_frame[n]["distinct_inliers"] for n in good]
    sig = [per_frame[n]["sigma_pos_m"] for n in good]

    report = {
        "frames": len(names), "kept": len(good), "kept_share": round(len(good) / len(names), 4),
        "kept_with_ge40_direct_lidar_inliers": len(direct_strong),
        "kept_without_carried_points": len(direct_only),
        "carried_note": "direct = the frame's own SIFT matches to LiDAR photos; carried = LiDAR "
                        "points whose 2-D track came through neighbouring video frames (still "
                        "LiDAR-lifted 3-D points, PnP per frame, no triangulation)",
        "fx": round(float(fx), 2), "fx_vs_35mm_tag_1221.9_pct": round(100 * (fx / 1221.9 - 1), 2),
        "fx_vs_old_selfcal_1604_pct": round(100 * (fx / 1604.47 - 1), 2),
        "principal_point": [(size[0] - 1) / 2, (size[1] - 1) / 2],
        "focal_search": {"pass1": fs1, "last_pass_coarse": fs2, "last_pass_fine": fs_fine,
                         "cost": f"mean over frames of the mean over ALL matches of min(reprojection,"
                                 f" {TRUNCATE_PX} px), each frame re-posed by PnP at that focal"},
        "intrinsics_check": intr,
        "passes": passes,
        "inliers": {"distinct_median": float(np.median(inls)), "distinct_p10":
                    float(np.percentile(inls, 10))} if inls else None,
        "reprojection_median_px": {"median": round(float(np.median(med)), 3),
                                   "p90": round(float(np.percentile(med, 90)), 3)} if med else None,
        "position_sigma_m": {"median": round(float(np.median(sig)), 4),
                             "p90": round(float(np.percentile(sig, 90)), 4),
                             "note": "PnP covariance from the inlier residuals: measurement noise "
                                     "only, not LiDAR depth / ARKit pose / alignment bias"}
        if sig else None,
        "smoothness": smoothness(names, order, poses_cv),
        "consistency_dropped": rejected,
        "consistency_rule": f"dropped when > {CONSIST_M} m or > {CONSIST_DEG} deg from the "
                            "interpolation of its kept neighbours (each <= 8 source frames away)",
        "unchecked_isolated": unchecked(names, order, set(poses_cv)),
        "texture_coverage": {"metric": "median Sobel gradient magnitude of the grey frame at half "
                                       "size (low = plain walls / ceiling)",
                             "quartiles": coverage,
                             "low_texture_frames(<2.0)": len(low),
                             "low_texture_kept": sum(per_frame[n]["kept"] for n in low)},
        "time_coverage": {"longest_gap_s": round(float(gaps.max()), 3),
                          "gaps_over_1s": int((gaps > 1).sum())},
        "references": ref_info,
        "sdr_duplicated_frames": dups,
        "frame_index_note": "benchmark source_index counts frames of the SDR conversion; it has "
                            "the duplicated frames listed, so the original video's index is "
                            "source_index minus the duplicates at or before it",
        "seconds": seconds, "gpu_seconds": 0.0, "cost_usd": 0.0,
        "per_frame": per_frame,
    }

    # overlays
    checks = OUT / "gt_checks"
    checks.mkdir(exist_ok=True)
    if good:
        cloud1 = fused_cloud(refs, 0.01, stride=1)
        gn = sorted(good, key=lambda n: order[n])
        tex_good = np.array([per_frame[n]["texture_median_grad"] for n in gn])
        picks = set(gn[int(k)] for k in np.linspace(0, len(gn) - 1, max(args.checks - 2, 1)))
        picks.add(gn[int(np.argmin(tex_good))])
        picks.add(max(gn, key=lambda n: per_frame[n]["median_px"]))
        report["checks"] = []
        for n in sorted(picks):
            out = checks / f"gt_{n}"
            overlay(n, str(BENCH / "frames" / n), poses_cv[n], K, cloud1, corr[n], good[n], out)
            report["checks"].append(str(out))
    (OUT / "gt_report.json").write_text(json.dumps(report, indent=1))

    # control: the old import's own poses
    fx_old = float(np.median([r["K"][0, 0] for r in init]))
    sx = size[0] / init_size[0]
    ctrl = {}
    by_v = {r["v"]: r for r in init}
    for n in names:
        best = None
        for dv in (0, -1, 1):
            if orig[n] + dv in by_v:
                best = by_v[orig[n] + dv]
                break
        if best is not None:
            ctrl[n] = {"c2w": np.round(best["c2w"] @ CV_GL, 6).tolist(),
                       "old_frame": best["name"], "source_offset": int(best["v"] - orig[n])}
    old = {"method": "mapanything_old_import",
           "licence": "MapAnything Apache-2.0 code / facebook/map-anything-apache weights "
                      "Apache-2.0 (per the 2026-11-23 import)",
           "metric": True, "fx": round(fx_old * sx, 2), "frames": ctrl,
           "seconds": 0.0, "gpu_seconds": 0.0, "cost_usd": 0.0,
           "notes": f"Control: the 2026-11-23 atrium_phone import's own poses (aligned to "
                    f"atrium_1059 with its scale fit), for benchmark frames within one source "
                    f"frame of an import frame (original-video index; SDR duplicate(s) {dups} "
                    f"accounted for). fx = median import fx x {sx:.4f}."}
    (OUT / "poses_mapanything_old.json").write_text(json.dumps(old, indent=1))
    log(f"done in {seconds} s: kept {len(good)}, fx {fx:.1f}, control frames {len(ctrl)}")


if __name__ == "__main__":
    main()
