"""Reviewer check: same top-down band, side by side: atrium_1059 LiDAR (the capture the physics
boxes come from) vs Stray splat (opaque splats). If chairs moved between captures, blobs differ."""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
import mujoco
sys.path.insert(0, str(Path.home() / "LiteReality/results/diagnosis/chairs"))
from common import load_spz
from litereality_agent.models.splat import rgbd
ROOT = Path.home() / "LiteReality"; B = ROOT / "run/atrium_stray_q/v2"; OUT = ROOT / "results/diagnosis/review_chairs"
FLOOR = -1.2677
frames = rgbd.load_frames(Path.home() / "LiteReality-scans/atrium_1059")
pts = []
with np.errstate(all="ignore"):
    for f in frames:
        d, c = rgbd.read_lidar(f); H, W = d.shape
        sx, sy = W / f.size[0], H / f.size[1]
        fx, fy, cx, cy = f.K[0, 0] * sx, f.K[1, 1] * sy, f.K[0, 2] * sx, f.K[1, 2] * sy
        v, u = np.mgrid[0:H, 0:W]; ok = (d > 0.1) & (d < 3.5)
        if c is not None: ok &= c >= 2
        z = d[ok]; x = (u[ok] - cx) / fx * z; y = (v[ok] - cy) / fy * z
        cam = np.stack([x, -y, -z, np.ones_like(z)], 1); pts.append((cam @ f.c2w.T)[:, :3])
S = np.concatenate(pts); S = S[np.isfinite(S).all(1)]
raw = load_spz(B / "raw.spz.bin"); R = raw["pos"][raw["op"] > 0.3]; del raw
m = mujoco.MjModel.from_xml_path(str(B / "physics/scene.xml")); dd = mujoco.MjData(m); mujoco.mj_forward(m, dd)
def outline(nm):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm); out = []
    for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
        if m.geom_size[g][0] < 0.05: continue
        Rm = dd.geom_xmat[g].reshape(3, 3); c0 = dd.geom_xpos[g]; h = m.geom_size[g]
        cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * h[:2]; out.append(cs @ Rm[:2, :2].T + c0[:2])
    return out
bands = [(0.05, 0.30, "legs/base floor+0.05..0.30"), (0.35, 0.65, "seat floor+0.35..0.65 (below desk top)"), (0.78, 1.05, "backrest top floor+0.78..1.05 (above desk top)")]
fig, axs = plt.subplots(2, 3, figsize=(24, 15))
xb = np.arange(-1.9, 1.2, 0.02); yb = np.arange(-0.5, 2.1, 0.02)
res = {}
for j, (lo, hi, t) in enumerate(bands):
    for i, (nm, P) in enumerate((("atrium_1059 LiDAR (scan)", S), ("Stray splat opaque (op>0.3)", R))):
        s = (P[:, 2] > FLOOR + lo) & (P[:, 2] < FLOOR + hi)
        H, _, _ = np.histogram2d(P[s, 0], P[s, 1], bins=[xb, yb])
        ax = axs[i, j]; ax.imshow(np.log1p(H).T, origin="lower", extent=[xb[0], xb[-1], yb[0], yb[-1]], cmap="magma")
        for c in ("Chair0", "Chair1", "Chair2", "Table0"):
            for o in outline(c): ax.plot(o[:, 0], o[:, 1], "c--", lw=0.8)
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, c); ax.text(dd.xpos[b][0], dd.xpos[b][1], c, color="cyan", fontsize=8)
        for lab, xy in (("blue", (0.576, 1.337)), ("teal", (-1.018, 0.016)), ("dark", (-0.707, 0.416))):
            ax.plot(*xy, "g+", ms=14); ax.text(xy[0] + 0.03, xy[1] - 0.08, lab, color="lime")
        ax.set_title(f"{nm}\n{t}", fontsize=10); ax.set_aspect("equal")
        # quantify: points in 0.2 m disc around each Stray chair centre and around each scanned box origin
        for lab, xy in (("blue", (0.576, 1.337)), ("teal", (-1.018, 0.016)), ("dark", (-0.707, 0.416)),
                        ("Chair0 scanned", tuple(dd.xpos[1][:2])), ("Chair1 scanned", tuple(dd.xpos[2][:2])), ("Chair2 scanned", tuple(dd.xpos[3][:2]))):
            res.setdefault(nm, {}).setdefault(t, {})[lab] = int((s & (np.hypot(P[:, 0] - xy[0], P[:, 1] - xy[1]) < 0.2)).sum())
plt.tight_layout(); plt.savefig(OUT / "r2_scan_vs_stray.png", dpi=70)
json.dump(res, open(OUT / "r2_scan_vs_stray.json", "w"), indent=1); print(json.dumps(res, indent=1))
