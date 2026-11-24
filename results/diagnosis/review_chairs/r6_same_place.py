"""Did the dark and teal chairs move between atrium_1059 (scan LiDAR, conf>=2) and the Stray splat?
Centroid of points in a band and disc around each chair, both captures, same frame."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.home() / "LiteReality/results/diagnosis/chairs"))
from common import load_spz, B
from litereality_agent.models.splat import rgbd
OUT = Path.home() / "LiteReality/results/diagnosis/review_chairs"; FLOOR = -1.2677
P = []
with np.errstate(all="ignore"):
    for f in rgbd.load_frames(Path.home() / "LiteReality-scans/atrium_1059"):
        d, c = rgbd.read_lidar(f); H, W = d.shape
        sx, sy = W / f.size[0], H / f.size[1]
        fx, fy, cx, cy = f.K[0, 0] * sx, f.K[1, 1] * sy, f.K[0, 2] * sx, f.K[1, 2] * sy
        v, u = np.mgrid[0:H, 0:W]; ok = (d > 0.1) & (d < 3.5) & ((c >= 2) if c is not None else True)
        z = d[ok]; x = (u[ok] - cx) / fx * z; y = (v[ok] - cy) / fy * z
        P.append((np.stack([x, -y, -z, np.ones_like(z)], 1) @ f.c2w.T)[:, :3])
S = np.concatenate(P); S = S[np.isfinite(S).all(1)]
raw = load_spz(B / "raw.spz.bin"); R = raw["pos"][raw["op"] > 0.3].astype(np.float64); del raw
tests = {"dark chair backrest (floor+0.80..1.05, 0.25 m disc)": ((-0.707, 0.416), 0.80, 1.05, 0.25),
         "teal chair seat (floor+0.40..0.60, 0.30 m disc)": ((-1.10, 0.12), 0.40, 0.60, 0.30)}
res = {}
for k, (c, lo, hi, r) in tests.items():
    row = {}
    for nm, Q in (("atrium_1059 LiDAR", S), ("Stray splat", R)):
        s = (Q[:, 2] > FLOOR + lo) & (Q[:, 2] < FLOOR + hi) & (np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1]) < r)
        row[nm] = {"n": int(s.sum()), "centroid_xy": Q[s, :2].mean(0).round(3).tolist() if s.any() else None}
    a, b = row["atrium_1059 LiDAR"]["centroid_xy"], row["Stray splat"]["centroid_xy"]
    row["centroid shift m"] = round(float(np.hypot(a[0] - b[0], a[1] - b[1])), 3)
    res[k] = row
json.dump(res, open(OUT / "r6_same_place.json", "w"), indent=1); print(json.dumps(res, indent=1))
