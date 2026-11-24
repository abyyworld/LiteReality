"""Reviewer check: where were the chairs during the atrium_1059 scan? Back-project the scan's own
LiDAR depth (no splat, no projection drawings) and overlay the scanned RoomPlan boxes (scene.xml)
plus the Stray splat chair clusters (from s6 sets centres)."""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
import mujoco
from litereality_agent.models.splat import rgbd
ROOT = Path.home() / "LiteReality"; B = ROOT / "run/atrium_stray_q/v2"; OUT = ROOT / "results/diagnosis/review_chairs"
frames = rgbd.load_frames(Path.home() / "LiteReality-scans/atrium_1059")
pts = []
for f in frames:
    d, c = rgbd.read_lidar(f)
    H, W = d.shape
    sx, sy = W / f.size[0], H / f.size[1]
    fx, fy, cx, cy = f.K[0, 0] * sx, f.K[1, 1] * sy, f.K[0, 2] * sx, f.K[1, 2] * sy
    v, u = np.mgrid[0:H, 0:W]
    ok = (d > 0.1) & (d < 4.0)
    if c is not None: ok &= c >= 2
    z = d[ok]; x = (u[ok] - cx) / fx * z; y = (v[ok] - cy) / fy * z
    cam = np.stack([x, -y, -z, np.ones_like(z)], 1)        # OpenGL camera
    pts.append((cam @ f.c2w.T)[:, :3])
P = np.concatenate(pts); print("lidar points", len(P), "frames", len(frames))
zs = np.sort(P[:, 2]); hist, edges = np.histogram(P[:, 2], bins=400)
floor = edges[np.argmax(hist)]; print("floor (mode z)", round(float(floor), 3))
m = mujoco.MjModel.from_xml_path(str(B / "physics/scene.xml")); dd = mujoco.MjData(m); mujoco.mj_forward(m, dd)
ms = mujoco.MjModel.from_xml_path(str(B / "physics/scene_splat.xml")); ds = mujoco.MjData(ms); mujoco.mj_forward(ms, ds)
def outline(m, d, nm):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm); out = []
    for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
        R = d.geom_xmat[g].reshape(3, 3); c0 = d.geom_xpos[g]; h = m.geom_size[g]
        cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * h[:2]
        out.append(cs @ R[:2, :2].T + c0[:2])
    return out
bodies = [m.body(i).name for i in range(m.nbody)]
print("bodies", bodies)
band = (P[:, 2] > floor + 0.10) & (P[:, 2] < floor + 0.95)
Q = P[band]
fig, axs = plt.subplots(1, 2, figsize=(20, 9))
for ax, (lo, hi, t) in zip(axs, ((0.10, 0.95, "floor+0.10..0.95"), (0.10, 0.35, "floor+0.10..0.35 (legs, wheels, seat underside)"))):
    s = (P[:, 2] > floor + lo) & (P[:, 2] < floor + hi)
    ax.hist2d(P[s, 0], P[s, 1], bins=[np.arange(-2.4, 2.1, 0.02), np.arange(-1.5, 2.8, 0.02)], cmap="magma", vmax=40)
    for nm in ("Chair0", "Chair1", "Chair2", "Table0"):
        if nm not in bodies: continue
        for o in outline(m, dd, nm): ax.plot(o[:, 0], o[:, 1], "w--", lw=1)
        for o in outline(ms, ds, nm): ax.plot(o[:, 0], o[:, 1], "c-", lw=1)
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm); ax.text(*dd.xpos[b][:2], nm, color="w")
    for nm, xy in (("splat blue", (0.576, 1.337)), ("splat teal", (-1.018, 0.016)), ("splat dark", (-0.707, 0.416))):
        ax.plot(*xy, "g*", ms=14); ax.text(xy[0], xy[1] - 0.1, nm, color="lime")
    ax.set_aspect("equal"); ax.set_title(f"atrium_1059 LiDAR, z {t}; white dashed = scanned boxes, cyan = corrected; green star = Stray splat chair centre")
plt.tight_layout(); plt.savefig(OUT / "r1_scan_lidar_topdown.png", dpi=90)
# numbers: LiDAR point count inside each scanned chair box (xy, band), and near each splat chair centre
res = {"floor_z": float(floor)}
for nm in ("Chair0", "Chair1", "Chair2"):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
    R = dd.xmat[b].reshape(3, 3); x = dd.xpos[b]
    loc = (Q - x) @ R
    g = m.body_geomadr[b]
    res[nm] = {"origin": dd.xpos[b].round(3).tolist(), "yaw_deg": round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), 1),
               "geom_half": m.geom_size[g].round(3).tolist(), "geom_pos": m.geom_pos[g].round(3).tolist()}
for nm, xy in (("splat blue", (0.576, 1.337)), ("splat teal", (-1.018, 0.016)), ("splat dark", (-0.707, 0.416))):
    res[nm + " lidar pts within 0.25 m (xy, band)"] = int((np.hypot(Q[:, 0] - xy[0], Q[:, 1] - xy[1]) < 0.25).sum())
json.dump(res, open(OUT / "r1_scan_lidar.json", "w"), indent=1); print(json.dumps(res, indent=1))
