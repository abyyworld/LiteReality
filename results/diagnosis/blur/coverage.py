"""(5) Coverage per surface: sample points every 5 cm on the RoomPlan surfaces (door, the two
window openings, the rest of the walls, ceiling (at the LiDAR ceiling, z 1.40 = 2.67 m; RoomPlan says 2.7-2.94), open floor, floor under the desk, desk top) and,
for every TRAINING camera of v2 (q600, 637), test frustum + LiDAR-depth occlusion (visible if the
point is no more than 10 cm behind the frame's own depth at that pixel; no depth -> not counted).
Per surface: views per point, closest view distance, finest sampling at the 1440-px training
width (mm per pixel), and the blur of the sharpest view of the point.
Writes coverage.json, coverage.png."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from litereality_agent.pipeline.evaluate import fidelity as fd
from litereality_agent.pipeline.quickroom.roomplan import parse_room
HERE = Path(__file__).resolve().parent
Q = Path.home() / "LiteReality-scans/stray_atrium_q600"
plan = parse_room(Q)
fz = plan.floor_z
STEP = 0.05
pts, lab = [], []
def add(P, name):
    pts.append(np.asarray(P)); lab.extend([name] * len(P))
# walls + openings
for w in plan.walls.values():
    s, e = np.array(w.start), np.array(w.end); Lw = np.linalg.norm(e - s); a = (e - s) / Lw
    n = np.array([-a[1], a[0]])
    for t in np.arange(STEP / 2, Lw, STEP):
        for h in np.arange(STEP / 2, w.height, STEP):
            name = "wall (other)"
            for op in plan.openings.values():
                if op.wall == w.name and abs(t - op.offset) < op.width / 2 and op.sill <= h <= op.sill + op.height:
                    name = "door" if op.kind == "door" else f"window ({w.name})"
            p = s + a * t
            add([[p[0], p[1], fz + h]], name)
# floor / ceiling polygon
from matplotlib.path import Path as MPath
poly = np.array([[0.5232, 2.6491], [1.9705, 0.8054], [-0.8075, -1.3755], [-2.2549, 0.4681]])
mp = MPath(poly)
xs = np.arange(poly[:, 0].min(), poly[:, 0].max(), STEP); ys = np.arange(poly[:, 1].min(), poly[:, 1].max(), STEP)
G = np.array([[x, y] for x in xs for y in ys]); G = G[mp.contains_points(G, radius=-0.02)]
tb = plan.objects["Table0"]; c = np.array(tb.center[:2]); yaw = np.radians(tb.yaw)
Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
loc = (G - c) @ Rz  # into table axes
under = (np.abs(loc[:, 0]) < tb.size[0] / 2) & (np.abs(loc[:, 1]) < tb.size[1] / 2)
add(np.c_[G[~under], np.full((~under).sum(), fz + 0.005)], "floor (open)")
add(np.c_[G[under], np.full(under.sum(), fz + 0.005)], "floor under desk")
add(np.c_[G[under], np.full(under.sum(), tb.center[2] + tb.size[2] / 2)], "desk top")
add(np.c_[G, np.full(len(G), 1.40)], "ceiling")
P = np.concatenate(pts); lab = np.array(lab)
print("points", len(P), {k: int((lab == k).sum()) for k in np.unique(lab)})
blur = np.load(HERE / "blur_arrays.npz")["Lest"]
nview = np.zeros(len(P), int); nblur4 = np.zeros(len(P), int); blursum = np.zeros(len(P)); dmin = np.full(len(P), np.inf); best_blur = np.full(len(P), np.inf)
f1440 = None
for j in sorted(Q.glob("frame_*.json")):
    m = json.loads(j.read_text()); pose = fd.camera_pose(m["cameraPoseARFrame"]); K = np.array(m["intrinsics"]).reshape(3, 3)
    f1440 = K[0, 0] * 1440 / 1920
    dpath = Q / f"depth_{j.stem[6:]}.png"
    if not dpath.is_file():
        continue
    D = np.asarray(Image.open(dpath), dtype=np.float32) / 1000.0; dh, dw = D.shape
    pc = (P - pose[:3, 3]) @ pose[:3, :3]
    x, y, z = pc[:, 0], -pc[:, 1], -pc[:, 2]
    ok = z > 0.1
    u = np.where(ok, K[0, 0] * x / np.maximum(z, 1e-6) + K[0, 2], -1); v = np.where(ok, K[1, 1] * y / np.maximum(z, 1e-6) + K[1, 2], -1)
    ok &= (u >= 0) & (u < 1920) & (v >= 0) & (v < 1440)
    iu = np.clip((u * dw / 1920).astype(int), 0, dw - 1); iv = np.clip((v * dh / 1440).astype(int), 0, dh - 1)
    d = D[iv, iu]
    vis = ok & (d > 0) & (z <= d + 0.10)
    dist = np.linalg.norm(P - pose[:3, 3], axis=1)
    nview += vis
    dmin = np.where(vis, np.minimum(dmin, dist), dmin)
    b = blur[int(m["stray"]["video_frame"])]
    best_blur = np.where(vis, np.minimum(best_blur, b), best_blur)
    nblur4 += vis & (b >= 4); blursum += vis * b
out = {"f_train_1440_px": round(float(f1440), 1), "visibility_rule": "in frustum, LiDAR depth valid, point <= depth + 10 cm", "surfaces": {}}
for k in np.unique(lab):
    s = lab == k; seen = s & (nview > 0)
    mm = 1000 * dmin[seen] / f1440
    out["surfaces"][k] = {"points": int(s.sum()), "frac_unseen": round(float(np.mean(nview[s] == 0)), 3),
                          "frac_lt3_views": round(float(np.mean(nview[s] < 3)), 3),
                          "views_median": float(np.median(nview[s])),
                          "closest_view_m_median": round(float(np.median(dmin[seen])), 2) if seen.any() else None,
                          "finest_mm_per_px_at_1440_median": round(float(np.median(mm)), 2) if seen.any() else None,
                          "sharpest_view_blur_px1920_median": round(float(np.median(best_blur[seen])), 2) if seen.any() else None,
                          "mean_view_blur_px1920_median": round(float(np.median(blursum[seen] / nview[seen])), 2) if seen.any() else None,
                          "frac_views_blur_ge_4px_median": round(float(np.median(nblur4[seen] / nview[seen])), 3) if seen.any() else None}
(HERE / "coverage.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
ks = list(out["surfaces"]); fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].barh(ks, [out["surfaces"][k]["views_median"] for k in ks], color="#1f6feb"); ax[0].set_xlabel("median # training views seeing a point")
ax[1].barh(ks, [out["surfaces"][k]["frac_lt3_views"] for k in ks], color="#d1242f"); ax[1].set_xlabel("fraction of points seen by < 3 training views")
fig.tight_layout(); fig.savefig(HERE / "coverage.png", dpi=110)
