"""One gravity-aligned ARKit world onto another: a yaw about Y and a translation.

Two recordings of one room (a Stray Scanner walk and the room's LiteReality Scanner scan) each
live in their own ARKit world: both metric, both with +Y against gravity, each with its origin
and heading where its session started. So the map between them is a rotation about Y and a
translation — four numbers — and that is all this ever solves for (a free 6-DoF fit could only
add a tilt that is not there).

    coarse   heights: the two clouds' height histograms (floor, table tops, ceiling) slid over each
             other give the vertical offset, whatever the heading. Plan: for every heading in
             `YAW_STEP` steps, the source's plan-view density (points per 5 cm cell, square-
             rooted: walls stack many points in one cell, floors one) is cross-correlated with the
             target's by FFT, giving the best shift for that heading and its score.
    refine   the coarse score's peaks at least `CANDIDATE_SEPARATION` apart (an ICP basin is
             20-30 degrees wide, so closer peaks refine to the same heading) plus the quarter
             turns of the best one (a rectangular room's competing headings) each go through
             point-to-plane ICP restricted to yaw + translation (scipy's KD-tree, target normals
             from local PCA, a Tukey-weighted 4 x 4 solve per step, the match distance shrinking
             from 30 to 4 cm). Each is scored by `wall_inliers`: the share of the source's
             vertical-surface points (walls, furniture sides) within `FINAL_MATCH` of the target.
             Floor and ceiling match at any heading, so counting them only narrows the gap
             between the right heading and a turned one.
    verdict  the best and the best fit more than `AMBIGUITY_SEPARATION` degrees from it are both
             refined on every source point; `ambiguous` when the winner's score is not clearly
             above the runner-up's (`AMBIGUITY_MARGIN`, `AMBIGUITY_OUTLIERS`). This is the check
             that tells a wrong heading from the right one: in a real room a heading 90 or 180
             degrees off still reaches ~1.5 cm residual and half the points as inliers.
    residual the median point-to-plane distance from each source point to the target surface,
             over the source points that have target within `OVERLAP` (what the recordings share).
             ICP minimises exactly this, so every converged fit has a low one: it only rejects
             fits that did not converge. A wrong room can still match a wall or two exactly, so
             the share of source points on the target surface (`inliers`) is a second check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from litereality_agent.models.capture.stray import voxel_downsample

YAW_STEP = 2.0              # degrees between the headings the coarse search tries
COARSE_CELL = 0.05          # metres: the plan-view grid
HEIGHT_BIN = 0.02           # metres: the height histogram
MAX_HEIGHT_SHIFT = 3.0      # metres
CANDIDATES = 6              # coarse peaks refined (plus the best one's quarter turns)
CANDIDATE_SEPARATION = 35.0  # degrees between coarse peaks (non-maximum suppression)
REFINED_SAME = 5.0          # degrees: refined headings this close are one basin
# Ambiguity: the best fit vs the best one more than AMBIGUITY_SEPARATION degrees away, by
# wall_inliers. Ambiguous when the gap is below AMBIGUITY_MARGIN AND the runner-up leaves fewer
# than AMBIGUITY_OUTLIERS times as many wall points unexplained (99% vs 87% is decisive: 1% vs
# 13% off the surface; 50% vs 45% is not).
AMBIGUITY_SEPARATION = 15.0
AMBIGUITY_MARGIN = 0.15
AMBIGUITY_OUTLIERS = 2.0
WALL_NORMAL_Y = 0.5         # |normal.y| below this: a vertical surface
MIN_WALL_POINTS = 300       # fewer: rank by all inliers instead
SOURCE_POINTS = 30000       # source points ICP uses (a random subset of the 3 cm cloud)
SCREEN_POINTS = 8000        # of those, the ones each candidate heading is screened with
OVERLAP = 0.10              # metres: a source point with target this near is in the shared part
FINAL_MATCH = 0.04          # metres: ICP's last match distance
MAX_RESIDUAL = 0.05         # metres: above this the fit is not trusted
# Nor below this fraction of source points within FINAL_MATCH of the target: two different
# rectangular rooms can share a few walls exactly (a low residual over a small overlap).
MIN_INLIERS = 0.4


@dataclass
class Alignment:
    transform: np.ndarray            # 4x4: source world -> target world
    yaw_deg: float
    translation: np.ndarray          # (3,) metres
    residual: float                  # median point-to-plane distance (m) over the overlap
    residual_point: float            # median nearest-neighbour distance (m) over the overlap
    overlap: float                   # fraction of source points within OVERLAP of the target
    inliers: float                   # fraction within FINAL_MATCH
    score: float = 0.0               # what headings are ranked by (wall_inliers, or inliers)
    score_kind: str = "wall_inliers"
    runner_up: dict | None = None    # the best fit > AMBIGUITY_SEPARATION degrees away
    margin: float = 1.0              # score - runner_up's score
    ambiguous: bool = False
    candidates: list[dict] = field(default_factory=list)
    seconds: float = 0.0

    def describe_margin(self) -> str:
        if self.runner_up is None:
            return f"no other heading converged ({self.score_kind} {self.score:.0%})"
        r = self.runner_up
        return (f"best yaw {self.yaw_deg:.1f} deg has {self.score_kind} {self.score:.0%}, the "
                f"next heading ({r['yaw_deg']:.1f} deg) {r['score']:.0%}: margin "
                f"{self.margin:.2f} (clear at >= {AMBIGUITY_MARGIN}, or when the next one leaves "
                f">= {AMBIGUITY_OUTLIERS:g}x as many wall points off the surface)")

    def summary(self) -> dict:
        return {"yaw_deg": round(self.yaw_deg, 3),
                "translation_m": [round(float(v), 4) for v in self.translation],
                "residual_m": round(self.residual, 4),
                "residual_point_m": round(self.residual_point, 4),
                "overlap": round(self.overlap, 3), "inliers": round(self.inliers, 3),
                "score": round(self.score, 3), "score_kind": self.score_kind,
                "runner_up": self.runner_up, "margin": round(self.margin, 3),
                "ambiguous": self.ambiguous,
                "transform": [[round(float(v), 6) for v in row] for row in self.transform],
                "candidates": self.candidates, "seconds": round(self.seconds, 2)}


def yaw_transform(yaw: float, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    """4x4: rotate `yaw` radians about +Y (right-handed: +z turns toward +x), then translate."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    T[:3, 3] = t
    return T


def _yaw_of(T: np.ndarray) -> float:
    return float(np.arctan2(T[0, 2], T[0, 0]))


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------------------------
# Coarse
# --------------------------------------------------------------------------------------------

def height_offset(src: np.ndarray, tgt: np.ndarray, *, bin_size: float = HEIGHT_BIN,
                  max_shift: float = MAX_HEIGHT_SHIFT) -> float:
    """ty such that src.y + ty best matches tgt.y, from the height histograms."""
    lo = min(src[:, 1].min(), tgt[:, 1].min()) - max_shift
    hi = max(src[:, 1].max(), tgt[:, 1].max()) + max_shift
    edges = np.arange(lo, hi + bin_size, bin_size)
    hs = np.sqrt(np.histogram(src[:, 1], edges)[0].astype(float))
    ht = np.sqrt(np.histogram(tgt[:, 1], edges)[0].astype(float))
    n = len(hs)
    corr = np.fft.irfft(np.fft.rfft(ht, 2 * n) * np.conj(np.fft.rfft(hs, 2 * n)), 2 * n)
    shifts = np.r_[np.arange(n), np.arange(-n, 0)]
    allowed = np.abs(shifts * bin_size) <= max_shift
    k = np.argmax(np.where(allowed, corr, -np.inf))
    return float(shifts[k] * bin_size)


def _plan_image(xz: np.ndarray, origin: np.ndarray, shape: tuple[int, int],
                cell: float) -> np.ndarray:
    ij = np.floor((xz - origin) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    img = np.zeros(shape, np.float64)
    np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1.0)
    return np.sqrt(img)


def coarse_candidates(src: np.ndarray, tgt: np.ndarray, *, yaw_step: float = YAW_STEP,
                      cell: float = COARSE_CELL, count: int = CANDIDATES,
                      separation: float = CANDIDATE_SEPARATION) -> list[dict]:
    """The best `count` distinct headings, each with its best plan shift and the height offset:
    [{"yaw": radians, "t": (3,), "score": normalised correlation}], best first."""
    ty = height_offset(src, tgt)
    t_xz = tgt[:, [0, 2]]
    t_origin = t_xz.min(0) - cell
    t_shape = tuple((np.ceil((t_xz.max(0) - t_origin) / cell) + 2).astype(int))
    A = _plan_image(t_xz, t_origin, t_shape, cell)
    s_xz = src[:, [0, 2]]
    radius = float(np.linalg.norm(s_xz - s_xz.mean(0), axis=1).max())
    b_side = int(np.ceil(2 * radius / cell)) + 3
    N = (t_shape[0] + b_side, t_shape[1] + b_side)
    FA = np.fft.rfft2(A, N)
    norm_a = np.linalg.norm(A)
    centre = s_xz.mean(0)
    yaws = np.deg2rad(np.arange(0.0, 360.0, yaw_step))
    results = []
    for yaw in yaws:
        c, s = np.cos(yaw), np.sin(yaw)
        # yaw_transform on (x, z): x' = c x + s z, z' = -s x + c z
        rel = s_xz - centre
        rot = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], 1)
        b_origin = np.array([-radius - cell, -radius - cell])
        B = _plan_image(rot, b_origin, (b_side, b_side), cell)
        corr = np.fft.irfft2(FA * np.conj(np.fft.rfft2(B, N)), N)
        k = np.unravel_index(np.argmax(corr), corr.shape)
        d = np.array([k[0] if k[0] < t_shape[0] else k[0] - N[0],
                      k[1] if k[1] < t_shape[1] else k[1] - N[1]], float)
        # A source point at plan position p (rotated, relative to the centre) lands on target
        # cell (p - b_origin)/cell + d: its target position is t_origin + (p - b_origin) + d cell.
        shift = t_origin - b_origin + d * cell
        # The translation for the full transform: R (x - centre) + shift = R x + (shift - R centre)
        rc = np.array([c * centre[0] + s * centre[1], -s * centre[0] + c * centre[1]])
        t = np.array([shift[0] - rc[0], ty, shift[1] - rc[1]])
        score = float(corr[k] / max(norm_a * np.linalg.norm(B), 1e-9))
        results.append({"yaw": float(yaw), "t": t, "score": score})
    order = sorted(range(len(results)), key=lambda i: -results[i]["score"])
    picked: list[dict] = []
    for i in order:
        r = results[i]
        if all(_yaw_gap(r["yaw"], p["yaw"]) >= separation for p in picked):
            picked.append(r)
        if len(picked) >= count:
            break
    # A rectangular room's competing headings are the best one's quarter turns: always tried.
    best = order[0]
    for quarter in (1, 2, 3):
        j = int(round(best + quarter * 90.0 / yaw_step)) % len(results)
        r = results[j]
        if all(_yaw_gap(r["yaw"], p["yaw"]) >= 1.0 for p in picked):
            picked.append({**r, "quarter_turn": quarter})
    return picked


def _yaw_gap(a: float, b: float) -> float:
    """Degrees between two headings given in radians, 0..180."""
    return abs((np.rad2deg(a - b) + 180.0) % 360.0 - 180.0)


# --------------------------------------------------------------------------------------------
# Refine
# --------------------------------------------------------------------------------------------

def estimate_normals(points: np.ndarray, tree, k: int = 16) -> np.ndarray:
    """Unit normals from the PCA of each point's `k` nearest neighbours."""
    k = min(k, len(points))
    _, nbr = tree.query(points, k=k, workers=-1)
    nb = points[nbr]                                         # (n, k, 3)
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]


def _tukey(r: np.ndarray, c: float) -> np.ndarray:
    w = np.zeros_like(r)
    inside = np.abs(r) < c
    w[inside] = (1 - (r[inside] / c) ** 2) ** 2
    return w


def icp_yaw(src: np.ndarray, tgt: np.ndarray, tree, normals: np.ndarray, T0: np.ndarray, *,
            iterations: int = 40, start: float = 0.30, final: float = FINAL_MATCH) -> np.ndarray:
    """Point-to-plane ICP over yaw about Y and a translation only, from `T0`."""
    T = T0.copy()
    for it in range(iterations):
        dist_max = max(final, start * (final / start) ** (it / max(iterations * 0.6, 1)))
        p = _apply(T, src)
        d, idx = tree.query(p, distance_upper_bound=dist_max, workers=-1)
        ok = np.isfinite(d)
        if ok.sum() < 30:
            break
        p, q, n = p[ok], tgt[idx[ok]], normals[idx[ok]]
        r = np.einsum("ij,ij->i", n, p - q)
        # d(R(dθ) p)/dθ at 0 = (p_z, 0, -p_x): the Jacobian row is [n·that, n_x, n_y, n_z].
        J = np.stack([n[:, 0] * p[:, 2] - n[:, 2] * p[:, 0], n[:, 0], n[:, 1], n[:, 2]], 1)
        w = _tukey(r, dist_max)
        JW = J * w[:, None]
        H = JW.T @ J + 1e-6 * np.eye(4)
        step = np.linalg.solve(H, -JW.T @ r)
        T = yaw_transform(step[0], step[1:]) @ T
        if it > iterations * 0.6 and abs(step[0]) < 1e-6 and np.linalg.norm(step[1:]) < 1e-5:
            break
    return T


def _measure(T: np.ndarray, src: np.ndarray, tgt: np.ndarray, tree,
             normals: np.ndarray, wall: np.ndarray | None = None) -> dict:
    """Overlap, inliers and residual of `src` moved by `T`; `score` ranks headings: the inliers
    among the `wall` (vertical-surface) points, or among all of them when `wall` is None."""
    p = _apply(T, src)
    d, idx = tree.query(p, distance_upper_bound=OVERLAP, workers=-1)
    ok = np.isfinite(d)
    if not ok.any():
        return {"overlap": 0.0, "inliers": 0.0, "score": 0.0, "residual": float("inf"),
                "residual_point": float("inf")}
    plane = np.abs(np.einsum("ij,ij->i", normals[idx[ok]], p[ok] - tgt[idx[ok]]))
    near = np.where(ok, d, np.inf) < FINAL_MATCH
    return {"overlap": float(ok.mean()), "inliers": float(np.mean(near)),
            "score": float(np.mean(near[wall])) if wall is not None else float(np.mean(near)),
            "residual": float(np.median(plane)), "residual_point": float(np.median(d[ok]))}


def align(source: np.ndarray, target: np.ndarray, *, yaw_step: float = YAW_STEP,
          candidates: int = CANDIDATES, seed: int = 0, log=None) -> Alignment:
    """The yaw + translation taking the `source` cloud (n, 3, ARKit world, Y up) onto `target`."""
    # numpy's Accelerate matmul on macOS raises spurious divide/overflow flags on finite input.
    with np.errstate(all="ignore"):
        return _align(source, target, yaw_step=yaw_step, candidates=candidates, seed=seed,
                      log=log)


def _align(source, target, *, yaw_step, candidates, seed, log) -> Alignment:
    from scipy.spatial import cKDTree

    t0 = time.perf_counter()
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    if len(source) < 100 or len(target) < 100:
        raise ValueError(f"too few points to align ({len(source)} source, {len(target)} target)")
    coarse_src = voxel_downsample(source, COARSE_CELL).astype(np.float64)
    coarse_tgt = voxel_downsample(target, COARSE_CELL).astype(np.float64)
    found = coarse_candidates(coarse_src, coarse_tgt, yaw_step=yaw_step, count=candidates)

    tgt = voxel_downsample(target, 0.02).astype(np.float64)
    tree = cKDTree(tgt)
    normals = estimate_normals(tgt, tree)
    src = voxel_downsample(source, 0.03).astype(np.float64)
    if len(src) > SOURCE_POINTS:
        src = src[np.random.default_rng(seed).choice(len(src), SOURCE_POINTS, replace=False)]

    # Vertical surfaces carry the heading; floor and ceiling fit any heading. Rotating about Y
    # keeps a normal's y, so which source points are "wall" does not depend on the candidate.
    src_normals = estimate_normals(src, cKDTree(src))
    wall_all = np.abs(src_normals[:, 1]) < WALL_NORMAL_Y
    use_wall = int(wall_all.sum()) >= MIN_WALL_POINTS
    kind = "wall_inliers" if use_wall else "inliers"

    # Every heading is screened on a subset; the best and the runner-up are refined on all of it.
    pick = np.random.default_rng(seed + 1).choice(len(src), min(len(src), SCREEN_POINTS),
                                                  replace=False)
    screen, wall_screen = src[pick], wall_all[pick]
    if use_wall and wall_screen.sum() < MIN_WALL_POINTS // 3:
        use_wall, kind = False, "inliers"
    tried: list[dict] = []
    fits: list[tuple[float, float, np.ndarray]] = []         # (score, -residual, T) per basin
    for cand in found:
        T = icp_yaw(screen, tgt, tree, normals, yaw_transform(cand["yaw"], cand["t"]),
                    iterations=25)
        m = _measure(T, screen, tgt, tree, normals, wall_screen if use_wall else None)
        yaw = _yaw_of(T)
        entry = {"coarse_yaw_deg": round(float(np.rad2deg(cand["yaw"])), 1),
                 "coarse_score": round(cand["score"], 4),
                 "quarter_turn": cand.get("quarter_turn", 0),
                 "yaw_deg": round(float(np.rad2deg(yaw)), 2),
                 "score": round(m["score"], 3), "inliers": round(m["inliers"], 3),
                 "residual_m": round(m["residual"], 4)}
        tried.append(entry)
        if log:
            log(f"  heading {entry['coarse_yaw_deg']:6.1f} deg -> {entry['yaw_deg']:7.2f}: "
                f"{m['score']:.0%} {kind}, {m['inliers']:.0%} inliers, residual "
                f"{m['residual'] * 100:.1f} cm")
        same = next((i for i, f in enumerate(fits)
                     if _yaw_gap(_yaw_of(f[2]), yaw) < REFINED_SAME), None)
        key = (m["score"], -m["residual"], T)
        if same is None:
            fits.append(key)
        elif key[:2] > fits[same][:2]:
            fits[same] = key
    fits.sort(key=lambda f: f[:2], reverse=True)
    best_T = fits[0][2]
    runner = next((f for f in fits[1:]
                   if _yaw_gap(_yaw_of(f[2]), _yaw_of(best_T)) > AMBIGUITY_SEPARATION), None)

    wall = wall_all if use_wall else None
    T = icp_yaw(src, tgt, tree, normals, best_T, start=0.08)
    m = _measure(T, src, tgt, tree, normals, wall)
    runner_doc, margin, ambiguous = None, 1.0, False
    if runner is not None:
        T2 = icp_yaw(src, tgt, tree, normals, runner[2], start=0.08)
        m2 = _measure(T2, src, tgt, tree, normals, wall)
        if m2["score"] > m["score"]:                 # the full cloud decides between the two
            T, m, T2, m2 = T2, m2, T, m
        if _yaw_gap(_yaw_of(T2), _yaw_of(T)) > AMBIGUITY_SEPARATION:
            margin = m["score"] - m2["score"]
            outliers = (1.0 - m2["score"]) / max(1.0 - m["score"], 1e-9)
            ambiguous = margin < AMBIGUITY_MARGIN and outliers < AMBIGUITY_OUTLIERS
            runner_doc = {"yaw_deg": round(float(np.rad2deg(_yaw_of(T2))), 2),
                          "translation_m": [round(float(v), 4) for v in T2[:3, 3]],
                          "score": round(m2["score"], 3), "inliers": round(m2["inliers"], 3),
                          "residual_m": round(m2["residual"], 4)}
    return Alignment(transform=T, yaw_deg=float(np.rad2deg(_yaw_of(T))),
                     translation=T[:3, 3].copy(), residual=m["residual"],
                     residual_point=m["residual_point"], overlap=m["overlap"],
                     inliers=m["inliers"], score=m["score"], score_kind=kind,
                     runner_up=runner_doc, margin=float(margin), ambiguous=bool(ambiguous),
                     candidates=tried, seconds=time.perf_counter() - t0)
