"""Top-down colour maps (splat colours, the highest opaque splat per 1.5 cm cell) in height slices,
to see how many real chairs the Stray splat has and where."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
z = np.load(OUT / "band.npz")
keys = ["room", "Chair0", "Chair1", "Chair2"]
P = np.concatenate([z[f"{k}_pos"] for k in keys]); O = np.concatenate([z[f"{k}_op"] for k in keys]); C = np.concatenate([z[f"{k}_col"] for k in keys])
mc, dc = model_data(B / "physics/scene_splat.xml"); mo, do = model_data(B / "physics/scene.xml")
FLOOR = -1.2677
ext = [-1.9, 1.2, -0.5, 2.0]; cell = 0.015
def topcolor(lo, hi):
    s = (P[:, 2] > FLOOR + lo) & (P[:, 2] < FLOOR + hi) & (O > 0.4)
    p, c = P[s], C[s]
    order = np.argsort(p[:, 2])                    # later (higher) wins
    p, c = p[order], c[order]
    nx, ny = int((ext[1]-ext[0])/cell), int((ext[3]-ext[2])/cell)
    ix = ((p[:, 0]-ext[0])/cell).astype(int); iy = ((p[:, 1]-ext[2])/cell).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    img = np.ones((ny, nx, 3)) * 1.0
    img[iy[ok], ix[ok]] = c[ok]
    return img
def boxes(ax, m, d, names, ls, color):
    for nm in names:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; h = m.geom_size[g]
            cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * h[:2]
            w = cs @ R[:2, :2].T + c[:2]; ax.plot(w[:, 0], w[:, 1], ls, color=color, lw=1)
        ax.text(d.xpos[b][0], d.xpos[b][1], nm, fontsize=8, color=color)
slices = [(0.06, 0.25, "legs/base"), (0.25, 0.42, "under seat"), (0.42, 0.62, "seat"), (0.62, 0.70, "arms/back below table top")]
fig, axs = plt.subplots(2, 2, figsize=(16, 13))
for ax, (lo, hi, t) in zip(axs.ravel(), slices):
    ax.imshow(topcolor(lo, hi), origin="lower", extent=ext)
    boxes(ax, mo, do, ["Chair0", "Chair1", "Chair2"], "--", "r")
    boxes(ax, mc, dc, ["Chair0", "Chair1", "Chair2"], "-", "g")
    boxes(ax, mc, dc, ["Table0"], ":", "orange")
    ax.set_title(f"floor+{lo:.2f}..{hi:.2f} m ({t}); red dashed scanned, green corrected"); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
plt.tight_layout(); plt.savefig(OUT / "topdown_colour_slices.png", dpi=100)
