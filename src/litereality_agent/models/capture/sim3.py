"""Similarity alignment for recordings with no ARKit world: gravity, then yaw + translation + scale.

A phone video reconstructed by a feed-forward model (`models/capture/video`) has poses in the
model's own world: metric only as far as the model's scale estimate goes, with no gravity and no
heading. Two steps put it in the room's LiteReality Scanner ARKit world:

    gravity   `estimate_up`: the direction that is the normal of the horizontal surfaces (floor,
              ceiling, table tops) and perpendicular to the vertical ones (walls, furniture
              sides). The cloud's PCA normals near the current guess (horizontal) and near
              perpendicular to it (vertical) give M = mean(n n^T | horizontal) - mean(n n^T |
              vertical); its top eigenvector is the new guess, three rounds with a shrinking
              window. The first guess is the cameras' mean "up" (a person filming holds the
              phone roughly level). `level_transform` then turns that direction onto +Y.
    sim3      `align_sim3`: the scale first (`coarse_scales`: the plan-view correlation of
              `align.coarse_candidates` over a log grid of scales, best few kept), then at each
              of those `align.align` (yaw about Y + translation, the Stray importer's search), each
              fit refined by point-to-plane ICP over yaw, translation AND scale (about the
              matched points' centroid). The fit with the most source wall points on the
              target's surface wins (then the lowest residual), as in `align`; `align` is run
              once more at the winning scale for its heading verdict (`ambiguous`).

`umeyama` / `robust_sim3` are the closed-form Sim(3) from point pairs (used to chain model passes
and in tests); `apply_sim3_to_pose` moves a camera pose by a similarity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from litereality_agent.models.capture import align as aligner
from litereality_agent.models.capture.stray import voxel_downsample

# The scale search: the coarse plan / height correlation (`align.coarse_candidates`) at every
# scale of SCALE_GRID (log-spaced around the model's metric scale); the best SCALE_STARTS of
# them, at least SCALE_SEPARATION apart (relative), each go through the yaw search and the
# yaw + translation + scale ICP. A box room fits a floor and two walls at ANY scale (scaling
# about their corner keeps them), so ICP alone cannot find the scale: the coarse correlation,
# which needs every wall in place at once, picks it.
SCALE_GRID = tuple(float(v) for v in np.exp(np.linspace(np.log(0.7), np.log(1.4), 25)))
SCALE_STARTS = 3
SCALE_SEPARATION = 0.06
SOURCE_POINTS = 30000
UP_ROUNDS = ((30.0, 30.0), (15.0, 15.0), (8.0, 8.0))   # (horizontal, vertical) windows, degrees
UP_VOXEL = 0.03
# `scale_spread`: fits at the chosen heading scoring within this of the best count as equally good.
SPREAD_SCORE_WINDOW = 0.03
SCALE_SPREAD_WARN = 0.01


# --------------------------------------------------------------------------------------------
# Closed form
# --------------------------------------------------------------------------------------------

def umeyama(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None
            ) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted least-squares similarity: (s, R, t) minimising sum w |s R src + t - dst|^2."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    a, b = src - mu_s, dst - mu_d
    cov = (b * w[:, None]).T @ a
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var = float(w @ np.sum(a * a, axis=1))
    s = float(np.trace(np.diag(D) @ S) / max(var, 1e-12))
    return s, R, mu_d - s * R @ mu_s


def robust_sim3(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None, *,
                iterations: int = 8) -> dict:
    """`umeyama` with Huber reweighting (1.5 x the median residual): robust to a minority of
    bad pairs. Returns {"scale", "R", "t", "T" (4x4 with s R), "median_residual", "points"}."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    base = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64)
    w = base.copy()
    for _ in range(iterations):
        s, R, t = umeyama(src, dst, w)
        r = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        k = max(1.5 * float(np.median(r)), 1e-9)
        w = base * np.where(r <= k, 1.0, k / np.maximum(r, 1e-12))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t
    return {"scale": s, "R": R, "t": t, "T": T, "median_residual": float(np.median(r)),
            "points": int(len(src))}


def sim3_scale(T: np.ndarray) -> float:
    return float(np.cbrt(np.linalg.det(np.asarray(T)[:3, :3])))


def apply_sim3_to_pose(T: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """A camera -> world pose moved by the similarity `T` (s R in its 3x3): the rotation stays
    orthonormal, the camera centre moves (and scales) with the points."""
    s = sim3_scale(T)
    out = np.eye(4)
    out[:3, :3] = (T[:3, :3] / s) @ c2w[:3, :3]
    out[:3, 3] = T[:3, :3] @ c2w[:3, 3] + T[:3, 3]
    return out


def apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):     # Accelerate's matmul flags spurious FP errors on macOS
        return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------------------------
# Gravity
# --------------------------------------------------------------------------------------------

def rotation_onto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit direction `a` onto `b`."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def estimate_up(points: np.ndarray, guess: np.ndarray, *, voxel: float = UP_VOXEL,
                rounds=UP_ROUNDS) -> tuple[np.ndarray, dict]:
    """The world's up direction from the cloud's horizontal and vertical surfaces (see the module
    docstring), starting from `guess` and keeping its sign."""
    with np.errstate(all="ignore"):      # Accelerate's matmul flags spurious overflows on macOS
        return _estimate_up(points, guess, voxel, rounds)


def _estimate_up(points, guess, voxel, rounds):
    from scipy.spatial import cKDTree

    g = np.asarray(guess, float) / np.linalg.norm(guess)
    pts = voxel_downsample(np.asarray(points, np.float64), voxel).astype(np.float64)
    stats = {"points": int(len(pts)), "rounds": []}
    if len(pts) < 200:
        stats["note"] = "too few points: the cameras' mean up is used"
        return g, stats
    normals = aligner.estimate_normals(pts, cKDTree(pts))
    normals = normals[np.isfinite(normals).all(1)]
    for horiz_deg, vert_deg in rounds:
        dots = normals @ g
        horiz = np.abs(dots) > np.cos(np.deg2rad(horiz_deg))
        vert = np.abs(dots) < np.sin(np.deg2rad(vert_deg))
        if horiz.sum() < 30 and vert.sum() < 30:
            break
        M = np.zeros((3, 3))
        if horiz.sum():
            M += normals[horiz].T @ normals[horiz] / horiz.sum()
        if vert.sum():
            M -= normals[vert].T @ normals[vert] / vert.sum()
        _, vecs = np.linalg.eigh(M)
        new = vecs[:, -1]
        new = new if new @ g >= 0 else -new
        stats["rounds"].append({"horizontal": int(horiz.sum()), "vertical": int(vert.sum()),
                                "moved_deg": round(float(np.rad2deg(np.arccos(
                                    np.clip(new @ g, -1, 1)))), 3)})
        g = new
    stats["from_guess_deg"] = round(float(np.rad2deg(np.arccos(np.clip(
        g @ (np.asarray(guess, float) / np.linalg.norm(guess)), -1, 1)))), 3)
    return g, stats


def level_transform(up: np.ndarray, origin: np.ndarray | None = None,
                    forward: np.ndarray | None = None) -> np.ndarray:
    """4x4 rotation (and shift) taking `up` onto +Y. With `origin`, that point goes to 0; with
    `forward` (a direction, e.g. the first camera's view), its horizontal part is turned onto -Z,
    the way ARKit sets up its world at the start of a session."""
    T = np.eye(4)
    T[:3, :3] = rotation_onto(up, [0.0, 1.0, 0.0])
    if forward is not None:
        f = T[:3, :3] @ np.asarray(forward, float)
        if np.hypot(f[0], f[2]) > 1e-6:
            yaw = np.arctan2(f[0], f[2]) - np.pi    # angle turning (f_x, f_z) onto (0, -1)
            T = aligner.yaw_transform(-yaw) @ T
    if origin is not None:
        T[:3, 3] = -(T[:3, :3] @ np.asarray(origin, float))
    return T


# --------------------------------------------------------------------------------------------
# Yaw + translation + scale
# --------------------------------------------------------------------------------------------

def icp_sim_yaw(src: np.ndarray, tgt: np.ndarray, tree, normals: np.ndarray, T0: np.ndarray, *,
                iterations: int = 40, start: float = 0.30,
                final: float = aligner.FINAL_MATCH) -> np.ndarray:
    """Point-to-plane ICP over yaw about Y, a scale (about the matched points' centroid) and a
    translation, from `T0` (4x4, may hold a scale)."""
    T = T0.copy()
    for it in range(iterations):
        dist_max = max(final, start * (final / start) ** (it / max(iterations * 0.6, 1)))
        p = apply(T, src)
        d, idx = tree.query(p, distance_upper_bound=dist_max, workers=-1)
        ok = np.isfinite(d)
        if ok.sum() < 30:
            break
        p, q, n = p[ok], tgt[idx[ok]], normals[idx[ok]]
        c = p.mean(0)
        rel = p - c
        r = np.einsum("ij,ij->i", n, p - q)
        J = np.stack([n[:, 0] * rel[:, 2] - n[:, 2] * rel[:, 0],
                      np.einsum("ij,ij->i", n, rel), n[:, 0], n[:, 1], n[:, 2]], 1)
        w = aligner._tukey(r, dist_max)
        JW = J * w[:, None]
        step = np.linalg.solve(JW.T @ J + 1e-6 * np.eye(5), -JW.T @ r)
        A = (1.0 + step[1]) * aligner.yaw_transform(step[0])[:3, :3]
        D = np.eye(4)
        D[:3, :3] = A
        D[:3, 3] = c - A @ c + step[2:]
        T = D @ T
        if (it > iterations * 0.6 and abs(step[0]) < 1e-6 and abs(step[1]) < 1e-6
                and np.linalg.norm(step[2:]) < 1e-5):
            break
    return T


@dataclass
class SimAlignment:
    transform: np.ndarray          # 4x4 similarity: source world -> target world (s R | t)
    scale: float
    yaw_deg: float
    translation: np.ndarray
    residual: float                # median point-to-plane distance over the overlap (m)
    residual_point: float
    overlap: float
    inliers: float
    score: float
    score_kind: str
    ambiguous: bool
    margin: float
    runner_up: dict | None
    tries: list[dict] = field(default_factory=list)
    scale_spread: float = 0.0      # relative spread of the scales near-equal fits converged to
    seconds: float = 0.0

    def summary(self) -> dict:
        return {"scale": round(self.scale, 5), "yaw_deg": round(self.yaw_deg, 3),
                "translation_m": [round(float(v), 4) for v in self.translation],
                "residual_m": round(self.residual, 4),
                "residual_point_m": round(self.residual_point, 4),
                "overlap": round(self.overlap, 3), "inliers": round(self.inliers, 3),
                "score": round(self.score, 3), "score_kind": self.score_kind,
                "ambiguous": self.ambiguous, "margin": round(self.margin, 3),
                "runner_up": self.runner_up, "scale_spread": round(self.scale_spread, 4),
                "transform": [[round(float(v), 6) for v in row] for row in self.transform],
                "tries": self.tries, "seconds": round(self.seconds, 2)}


def coarse_scales(source: np.ndarray, target: np.ndarray, *, grid=SCALE_GRID,
                  count: int = SCALE_STARTS, separation: float = SCALE_SEPARATION) -> list[dict]:
    """The `count` best scales of `grid` by the coarse plan correlation's best score (each scale's
    best heading), at least `separation` apart: [{"scale", "score"}], best first."""
    src = voxel_downsample(np.asarray(source, np.float64), aligner.COARSE_CELL).astype(np.float64)
    tgt = voxel_downsample(np.asarray(target, np.float64), aligner.COARSE_CELL).astype(np.float64)
    rows = []
    for s in grid:
        found = aligner.coarse_candidates(src * s, tgt, count=1)
        rows.append({"scale": float(s), "score": float(max(c["score"] for c in found))})
    picked: list[dict] = []
    for r in sorted(rows, key=lambda r: -r["score"]):
        if all(abs(np.log(r["scale"] / p["scale"])) >= separation for p in picked):
            picked.append(r)
        if len(picked) >= count:
            break
    return picked


def align_sim3(source: np.ndarray, target: np.ndarray, *, scales=None, seed: int = 0,
               log=None) -> SimAlignment:
    """The similarity (yaw about Y, translation, scale) taking the gravity-aligned `source` cloud
    onto `target` (both Y up). See the module docstring."""
    with np.errstate(all="ignore"):
        return _align_sim3(source, target, scales=scales, seed=seed, log=log)


def _align_sim3(source, target, *, scales, seed, log) -> SimAlignment:
    from scipy.spatial import cKDTree

    t0 = time.perf_counter()
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    tgt = voxel_downsample(target, 0.02).astype(np.float64)
    tree = cKDTree(tgt)
    normals = aligner.estimate_normals(tgt, tree)
    src = voxel_downsample(source, 0.03).astype(np.float64)
    if len(src) > SOURCE_POINTS:
        src = src[np.random.default_rng(seed).choice(len(src), SOURCE_POINTS, replace=False)]
    src_normals = aligner.estimate_normals(src, cKDTree(src))
    wall = np.abs(src_normals[:, 1]) < aligner.WALL_NORMAL_Y
    use_wall = int(wall.sum()) >= aligner.MIN_WALL_POINTS

    if scales is None:
        coarse = coarse_scales(source, target)
        scales = [c["scale"] for c in coarse]
        if log:
            log("  coarse scales: " + ", ".join(f"{c['scale']:.3f} ({c['score']:.3f})"
                                                for c in coarse))
    best = None
    tries = []
    for s0 in scales:
        S = np.diag([s0, s0, s0, 1.0])
        try:
            fit = aligner.align(source * s0, target, seed=seed)
        except ValueError as exc:
            tries.append({"start_scale": round(float(s0), 4), "error": str(exc)})
            continue
        T = icp_sim_yaw(src, tgt, tree, normals, fit.transform @ S)
        m = aligner._measure(T, src, tgt, tree, normals, wall if use_wall else None)
        entry = {"start_scale": round(float(s0), 4), "scale": round(sim3_scale(T), 5),
                 "yaw_deg": round(float(np.rad2deg(aligner._yaw_of(T))), 2),
                 "score": round(m["score"], 3), "inliers": round(m["inliers"], 3),
                 "residual_m": round(m["residual"], 4), "yaw_only_ambiguous": fit.ambiguous}
        tries.append(entry)
        if log:
            log(f"  start scale {s0:.2f}: scale {entry['scale']:.4f}, yaw {entry['yaw_deg']:.1f} "
                f"deg, {m['score']:.0%} {'wall_inliers' if use_wall else 'inliers'}, residual "
                f"{m['residual'] * 100:.1f} cm")
        key = (m["score"], -m["residual"])
        if best is None or key > best[0]:
            best = (key, T, m, fit)
    if best is None:
        raise ValueError("no starting scale could be aligned: " + "; ".join(
            str(t.get("error")) for t in tries))
    _, T, m, fit = best
    ambiguous, margin, runner_up = bool(fit.ambiguous), float(fit.margin), fit.runner_up
    # The heading verdict at the scale found: at a start scale 10-15% off, the right heading and
    # a turned one both fit poorly and look alike, so `fit.ambiguous` says little.
    s_best = sim3_scale(T)
    try:
        final = aligner.align(source * s_best, target, seed=seed)
    except ValueError:
        final = None
    if final is not None:
        T2 = icp_sim_yaw(src, tgt, tree, normals,
                         final.transform @ np.diag([s_best, s_best, s_best, 1.0]))
        m2 = aligner._measure(T2, src, tgt, tree, normals, wall if use_wall else None)
        tries.append({"start_scale": round(s_best, 5), "scale": round(sim3_scale(T2), 5),
                      "yaw_deg": round(float(np.rad2deg(aligner._yaw_of(T2))), 2),
                      "score": round(m2["score"], 3), "inliers": round(m2["inliers"], 3),
                      "residual_m": round(m2["residual"], 4),
                      "yaw_only_ambiguous": final.ambiguous, "verdict": True})
        if aligner._yaw_gap(aligner._yaw_of(T), aligner._yaw_of(T2)) <= aligner.REFINED_SAME:
            # The same heading, so `final`'s verdict is about the transform kept, refined or not.
            if m2["score"] >= m["score"] - 0.01:    # the same basin, refined at the right scale
                T, m = T2, m2
            ambiguous, margin = bool(final.ambiguous), float(final.margin)
            runner_up = final.runner_up
        else:
            # The re-run at the found scale preferred another heading, which fits worse after
            # ICP: `final`'s verdict is about that heading, not T's. T stays; it is judged
            # against that heading with `align`'s own rule.
            margin = m["score"] - m2["score"]
            outliers = (1.0 - m2["score"]) / max(1.0 - m["score"], 1e-9)
            ambiguous = bool(margin < aligner.AMBIGUITY_MARGIN
                             and outliers < aligner.AMBIGUITY_OUTLIERS)
            runner_up = {"yaw_deg": round(float(np.rad2deg(aligner._yaw_of(T2))), 2),
                         "translation_m": [round(float(v), 4) for v in T2[:3, 3]],
                         "score": round(m2["score"], 3), "inliers": round(m2["inliers"], 3),
                         "residual_m": round(m2["residual"], 4),
                         "note": "the heading the re-run at the found scale preferred"}
    return SimAlignment(
        transform=T, scale=sim3_scale(T), yaw_deg=float(np.rad2deg(aligner._yaw_of(T))),
        translation=T[:3, 3].copy(), residual=m["residual"], residual_point=m["residual_point"],
        overlap=m["overlap"], inliers=m["inliers"], score=m["score"],
        score_kind="wall_inliers" if use_wall else "inliers", ambiguous=ambiguous,
        margin=float(margin), runner_up=runner_up, tries=tries,
        scale_spread=scale_spread(tries, T, m["score"]), seconds=time.perf_counter() - t0)


def scale_spread(tries: list[dict], T: np.ndarray, score: float, *,
                 score_window: float = SPREAD_SCORE_WINDOW) -> float:
    """(max - min) / T's scale over the scales the ICP runs converged to at T's heading with a
    score within `score_window` of T's (and T's own): the wall-inlier score plateaus over a range
    of scales, so this says how loosely the alignment pins the scale down."""
    yaw = aligner._yaw_of(T)
    s = [float(t["scale"]) for t in tries if "scale" in t
         and aligner._yaw_gap(np.deg2rad(t["yaw_deg"]), yaw) <= aligner.REFINED_SAME
         and t["score"] >= score - score_window]
    s.append(sim3_scale(T))
    return float((max(s) - min(s)) / sim3_scale(T))
