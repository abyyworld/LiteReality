"""Top-down maps of the chair height band: density of the uncut snapshot and who owns each splat."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

man = bundle()
mc, dc = model_data(B / "physics/scene_splat.xml")
mo, do = model_data(B / "physics/scene.xml")
room = load_spz(B / man["room"])
parts = {"room": room["pos"]}
ops = {"room": room["op"]}
cols = {"room": room["col"]}
for name, f in man["bodies"].items():
    s = load_spz(B / f)
    R, x = body_pose(mc, dc, name)
    parts[name] = (s["pos"].astype(np.float64) @ R.T + x).astype(np.float32)
    ops[name] = s["op"]; cols[name] = s["col"]
FLOOR = -1.2677
LO, HI = FLOOR + 0.06, FLOOR + 0.95          # chair band (seat ~0.45, back top ~0.89 above floor)
ext = [-2.4, 2.1, -1.5, 2.8]
cell = 0.02

def band(p):
    return (p[:, 2] > LO) & (p[:, 2] < HI)

def boxes(ax, m, d, names, ls, lw=1.2, color="k"):
    for nm in names:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; h = m.geom_size[g]
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX: continue
            cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * h[:2]
            w = cs @ R[:2, :2].T + c[:2]
            ax.plot(w[:, 0], w[:, 1], ls, color=color, lw=lw)
        ax.text(d.xpos[b][0], d.xpos[b][1], nm, fontsize=7, color=color)

chairs = ["Chair0", "Chair1", "Chair2"]
fig, axs = plt.subplots(1, 2, figsize=(18, 9))
allp = np.concatenate([parts[k][band(parts[k])] for k in parts])
allo = np.concatenate([ops[k][band(parts[k])] for k in parts])
H, xe, ye = np.histogram2d(allp[:, 0], allp[:, 1], bins=[np.arange(ext[0], ext[1], cell), np.arange(ext[2], ext[3], cell)], weights=allo)
axs[0].imshow(np.log1p(H.T), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="magma")
axs[0].set_title(f"uncut trained splat (room+bodies), opacity-weighted density, z in floor+[0.06,0.95] m\nwhite dashed = RoomPlan (atrium_1059) boxes, cyan = corrected (scene_splat.xml)")
boxes(axs[0], mo, do, chairs, "--", color="w")
boxes(axs[0], mc, dc, chairs, "-", color="c")
boxes(axs[0], mc, dc, ["Table0"], "-", color="y", lw=0.8)
colors = {"room": (0.6, 0.6, 0.6), "Chair0": (1, 0.2, 0.2), "Chair1": (0.1, 0.8, 0.1), "Chair2": (0.2, 0.4, 1), "Wall0_Door_0": (1, 0.8, 0)}
for k in parts:
    p = parts[k][band(parts[k]) & (ops[k] > 0.3)]
    axs[1].scatter(p[:, 0], p[:, 1], s=0.05 if k == "room" else 0.2, c=[colors[k]], label=f"{k} ({len(p)})", rasterized=True)
boxes(axs[1], mo, do, chairs, "--", color="k")
boxes(axs[1], mc, dc, chairs, "-", color="m")
boxes(axs[1], mc, dc, ["Table0"], "-", color="y", lw=0.8)
axs[1].legend(markerscale=20, fontsize=8); axs[1].set_aspect("equal")
axs[1].set_title("owner of each splat in the served bundle (opacity>0.3, same band)\nblack dashed = RoomPlan boxes, magenta = corrected")
for a in axs: a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
plt.tight_layout(); plt.savefig(OUT / "topdown_room.png", dpi=110)
np.savez_compressed(OUT / "band.npz", **{f"{k}_pos": parts[k][band(parts[k])] for k in parts}, **{f"{k}_op": ops[k][band(parts[k])] for k in parts}, **{f"{k}_col": cols[k][band(parts[k])] for k in parts})
print("saved")
