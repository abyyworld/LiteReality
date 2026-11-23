"""The focal length of a moving camera from its photos alone (no LiDAR, no metadata).

A feed-forward geometry model can get the focal length wrong (MapAnything put an iPhone video's
fx about 33% short) while its poses and depth stay self-consistent, so nothing downstream sees
it. This checks it with features only:

    features   SIFT on each photo at `FEATURE_LONG` px (grey), matched by Lowe's ratio test and
               filtered by a RANSAC fundamental matrix (`match`).
    criterion  Mendonca-Cipolla: with the right K, E = K^T F K is an essential matrix, whose two
               non-zero singular values are equal. For a grid of focal lengths (square pixels,
               principal point at the image centre) the cost (s1 - s2) / (s1 + s2) of every
               pair is computed; the focal with the lowest median cost over pairs wins. Each
               pair's own best focal is kept too (their median and spread say how consistent
               the pairs are).

It needs translation between the two photos (a pure rotation gives no F) and a scene that is not
a single plane, so `pairs_by_rotation` picks pairs that turned `MIN_TURN`..`MAX_TURN` degrees
and moved, by the model's poses (their focal error barely affects that choice).
"""

from __future__ import annotations

import numpy as np

FEATURE_LONG = 1440          # px: SIFT runs at this long side
SIFT_FEATURES = 4000
RATIO = 0.8
F_THRESHOLD = 1.0            # px at FEATURE_LONG: RANSAC fundamental-matrix inlier distance
MIN_F_INLIERS = 40
MIN_TURN, MAX_TURN = 3.0, 25.0
MIN_BASELINE = 0.05          # metres (model world) between the two cameras
MAX_PAIRS = 60
FOCAL_GRID = tuple(float(v) for v in np.exp(np.linspace(np.log(0.35), np.log(2.5), 121)))


def _cv2():
    import cv2

    return cv2


def features(rgb_or_grey: np.ndarray, long_side: int = FEATURE_LONG):
    """SIFT keypoints (n, 2) in the INPUT image's pixels (integer pixel centres) and descriptors."""
    cv2 = _cv2()
    img = rgb_or_grey
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    k = min(1.0, long_side / max(w, h))
    small = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA) \
        if k < 1.0 else img
    kps, desc = cv2.SIFT_create(nfeatures=SIFT_FEATURES).detectAndCompute(small, None)
    if desc is None or not kps:
        return np.zeros((0, 2)), np.zeros((0, 128), np.float32)
    sx, sy = w / small.shape[1], h / small.shape[0]
    pts = np.array([kp.pt for kp in kps], np.float64)
    pts = np.stack([(pts[:, 0] + 0.5) * sx - 0.5, (pts[:, 1] + 0.5) * sy - 0.5], 1)
    return pts, desc


def match(a, b, *, ratio: float = RATIO, f_threshold: float | None = None,
          scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Matched points of two `features` results that pass the ratio test and a RANSAC
    fundamental matrix (`f_threshold` px in the input pixels; default `F_THRESHOLD` x `scale`):
    (pts_a, pts_b, F) with F None when too few matched."""
    cv2 = _cv2()
    (pa, da), (pb, db) = a, b
    if len(da) < 8 or len(db) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2)), None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < 15:
        return np.zeros((0, 2)), np.zeros((0, 2)), None
    ia = np.array([m.queryIdx for m in good])
    ib = np.array([m.trainIdx for m in good])
    xa, xb = pa[ia], pb[ib]
    thr = F_THRESHOLD * scale if f_threshold is None else f_threshold
    F, mask = cv2.findFundamentalMat(xa, xb, cv2.FM_RANSAC, thr, 0.999, 5000)
    if F is None or F.shape != (3, 3) or mask is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), None
    keep = mask.ravel().astype(bool)
    return xa[keep], xb[keep], F


def essential_cost(F: np.ndarray, focal: float, centre: tuple[float, float]) -> float:
    """Mendonca-Cipolla: (s1 - s2) / (s1 + s2) of K^T F K; 0 for a true essential matrix."""
    K = np.array([[focal, 0.0, centre[0]], [0.0, focal, centre[1]], [0.0, 0.0, 1.0]])
    s = np.linalg.svd(K.T @ F @ K, compute_uv=False)
    return float((s[0] - s[1]) / max(s[0] + s[1], 1e-300))


def pairs_by_rotation(c2w: np.ndarray, *, min_turn: float = MIN_TURN, max_turn: float = MAX_TURN,
                      min_baseline: float = MIN_BASELINE, max_pairs: int = MAX_PAIRS,
                      ) -> list[tuple[int, int]]:
    """Up to `max_pairs` (i, j), i < j, spread over the sequence, whose cameras turned
    `min_turn`..`max_turn` degrees and moved at least `min_baseline` between them."""
    n = len(c2w)
    cands = []
    for i in range(n):
        for j in range(i + 1, min(n, i + 25)):
            R = c2w[i, :3, :3].T @ c2w[j, :3, :3]
            ang = np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            base = np.linalg.norm(c2w[j, :3, 3] - c2w[i, :3, 3])
            if min_turn <= ang <= max_turn and base >= min_baseline:
                cands.append((i, j))
                break                       # the first j far enough from i
    if len(cands) <= max_pairs:
        return cands
    pick = np.linspace(0, len(cands) - 1, max_pairs).round().astype(int)
    return [cands[k] for k in sorted(set(pick))]


def estimate_focal(images, pairs: list[tuple[int, int]], *, grid_factors=FOCAL_GRID,
                   reference_focal: float | None = None, log=None) -> dict:
    """The focal length (in the images' pixels) that makes the pairs' fundamental matrices most
    nearly essential. `images` is indexable (index -> RGB / grey array, all one size, loaded
    lazily); `grid_factors` scale `reference_focal` (default: the long side, ~53 deg FOV).
    Returns {"focal", "per_pair_median", "per_pair_iqr", "pairs", "curve"} or {"focal": None}."""
    feats: dict[int, tuple] = {}

    def feat(i):
        if i not in feats:
            feats[i] = features(images[i])
        return feats[i]

    size = None
    Fs = []
    for i, j in pairs:
        a, b = feat(i), feat(j)
        img = images[i]
        size = (img.shape[1], img.shape[0])
        scale = max(size) / FEATURE_LONG
        xa, xb, F = match(a, b, scale=scale)
        if F is not None and len(xa) >= MIN_F_INLIERS:
            Fs.append(F)
    if not Fs or size is None:
        return {"focal": None, "pairs": len(Fs), "pairs_tried": len(pairs)}
    centre = ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0)
    f0 = float(reference_focal or max(size))
    focals = f0 * np.asarray(grid_factors)
    costs = np.array([[essential_cost(F, f, centre) for f in focals] for F in Fs])
    total = np.median(costs, axis=0)
    best = int(np.argmin(total))
    per_pair = focals[np.argmin(costs, axis=1)]
    q1, q3 = np.percentile(per_pair, [25, 75])
    out = {"focal": float(focals[best]), "per_pair_median": float(np.median(per_pair)),
           "per_pair_iqr": [float(q1), float(q3)], "pairs": len(Fs), "pairs_tried": len(pairs),
           "image_size": list(size), "at_grid_edge": best in (0, len(focals) - 1),
           "curve": [[round(float(f), 1), round(float(c), 5)]
                     for f, c in zip(focals[::6], total[::6])]}
    if log:
        log(f"  self-calibration: fx {out['focal']:.0f} px over {len(Fs)} pairs (per-pair median "
            f"{out['per_pair_median']:.0f}, IQR {q1:.0f}-{q3:.0f})")
    return out
