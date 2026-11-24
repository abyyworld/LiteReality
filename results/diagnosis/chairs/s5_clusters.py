"""Find the real chairs in the uncut snapshot: 3D connected components of opaque, non-static,
above-floor splats (3 cm voxels), independent of the physics boxes."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
from common import *
from litereality_agent.pipeline.splatroom.split import _static_boxes, _inside_any
from litereality_agent.room_ops.roomcall.model_info import movable_body_ids
man = bundle()
mc, dc = model_data(B / "physics/scene_splat.xml")
raw = load_spz(B / "raw.spz.bin")
P = raw["pos"]; O = raw["op"]
FLOOR = -1.2677
sel = (P[:, 2] > FLOOR + 0.05) & (P[:, 2] < FLOOR + 1.05) & (O > 0.3) \
    & (P[:, 0] > -1.9) & (P[:, 0] < 1.3) & (P[:, 1] > -0.4) & (P[:, 1] < 2.1)
idx = np.flatnonzero(sel)
static = _static_boxes(mc, dc, sorted(movable_body_ids(mc)))
st = _inside_any(P[idx].astype(np.float64), static, 0.03)
idx = idx[~st]
print("candidates", len(idx))
v = 0.03
lo = P[idx].min(0)
g = ((P[idx] - lo) / v).astype(int)
shape = g.max(0) + 1
occ = np.zeros(shape, np.int32)
np.add.at(occ, tuple(g.T), 1)
lab, n = ndimage.label(occ >= 2, structure=np.ones((3, 3, 3)))
sizes = ndimage.sum(occ, lab, index=np.arange(1, n + 1))
lab_of = lab[tuple(g.T)]
order = np.argsort(-sizes)
print("components", n)
rows = []
for r in order[:15]:
    L = r + 1
    m = lab_of == L
    p = P[idx[m]]
    rows.append((L, int(m.sum()), p.mean(0), p.min(0), p.max(0)))
    print(f"comp {L}: {m.sum()} splats, centre {p.mean(0).round(2)}, x {p[:,0].min():.2f}..{p[:,0].max():.2f} y {p[:,1].min():.2f}..{p[:,1].max():.2f} z(above floor) {p[:,2].min()-FLOOR:.2f}..{p[:,2].max()-FLOOR:.2f}")
np.savez_compressed(OUT / "clusters.npz", idx=idx, lab=lab_of)
fig, ax = plt.subplots(figsize=(11, 9))
cmap = plt.get_cmap("tab10")
for k, (L, c, mu, mn, mx) in enumerate(rows[:10]):
    p = P[idx[lab_of == L]]
    ax.scatter(p[:, 0], p[:, 1], s=0.3, color=cmap(k), rasterized=True)
    ax.text(mu[0], mu[1], f"c{L} ({c})", fontsize=8)
for nm in ("Chair0", "Chair1", "Chair2"):
    b = mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_BODY, nm); ax.plot(*dc.xpos[b][:2], "k+", ms=12); ax.text(*dc.xpos[b][:2], nm)
ax.set_aspect("equal"); ax.set_title("3D connected components (3 cm voxels) of opaque non-static splats, floor+0.05..1.05 m; + = corrected body origin")
plt.savefig(OUT / "clusters_topdown.png", dpi=100)
