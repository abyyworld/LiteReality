"""Height histogram of atrium_1059 LiDAR points within 0.2 m (xy) of the scanned Chair1 centre vs the
Stray dark chair centre, all confidences and conf>=2. Shows whether a chair stood at Chair1 in the scan."""
import json, numpy as np
from pathlib import Path
from litereality_agent.models.splat import rgbd
OUT = Path.home() / "LiteReality/results/diagnosis/review_chairs"; FLOOR = -1.2677
frames = rgbd.load_frames(Path.home() / "LiteReality-scans/atrium_1059")
P, C = [], []
with np.errstate(all="ignore"):
    for f in frames:
        d, c = rgbd.read_lidar(f); H, W = d.shape
        sx, sy = W / f.size[0], H / f.size[1]
        fx, fy, cx, cy = f.K[0, 0] * sx, f.K[1, 1] * sy, f.K[0, 2] * sx, f.K[1, 2] * sy
        v, u = np.mgrid[0:H, 0:W]; ok = (d > 0.1) & (d < 3.5)
        z = d[ok]; x = (u[ok] - cx) / fx * z; y = (v[ok] - cy) / fy * z
        cam = np.stack([x, -y, -z, np.ones_like(z)], 1); P.append((cam @ f.c2w.T)[:, :3])
        C.append(c[ok] if c is not None else np.full(len(z), 2))
P = np.concatenate(P); C = np.concatenate(C); ok = np.isfinite(P).all(1); P, C = P[ok], C[ok]
edges = np.array([0, 0.04, 0.15, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.05])
res = {"height bins above floor (m)": edges.tolist()}
for k, xy in {"Chair1 scanned centre": (-0.3345, 0.8706), "Stray dark chair": (-0.707, 0.416), "Chair2 scanned centre": (-1.1924, 0.1971)}.items():
    r = np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1]) < 0.2
    h = P[r, 2] - FLOOR
    res[k] = {"all": np.histogram(h, edges)[0].tolist(), "conf>=2": np.histogram(h[C[r] >= 2], edges)[0].tolist()}
json.dump(res, open(OUT / "r4_chair1_column.json", "w"), indent=1); print(json.dumps(res))
