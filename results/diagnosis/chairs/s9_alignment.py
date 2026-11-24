"""(b) How well does the Stray splat sit on the scan's physics? Per wall box: the peak of the splats'
signed distance to the wall's inner face along its normal (within +-15 cm, over the wall's extent).
Floor: the floor splats' height against the physics floor top. Table: the desk top."""
import numpy as np, json
from common import *
man = bundle()
raw = load_spz(B / "raw.spz.bin"); P = raw["pos"].astype(np.float64); O = raw["op"]
m, d = model_data(B / "physics/scene_splat.xml")
out = {}
def peak(x):
    h, e = np.histogram(x, bins=np.arange(-0.15, 0.1501, 0.005)); i = np.argmax(h); return round(float((e[i] + e[i + 1]) / 2), 3), int(h.sum())
for g in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if m.geom_bodyid[g] != 0 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX: continue
    if not (nm.startswith("Wall") and nm.split("_")[-1].isdigit()): continue
    R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; h = m.geom_size[g]
    loc = (P - c) @ R
    inside = (np.abs(loc[:, 0]) < h[0] - 0.05) & (np.abs(loc[:, 2]) < h[2] - 0.05) & (np.abs(loc[:, 1]) < h[1] + 0.15) & (O > 0.3)
    if inside.sum() < 200: continue
    # the room side is towards the room centre
    sgn = np.sign((np.array([-0.14, 0.64, 0]) - c) @ R[:, 1])
    dist = sgn * loc[inside, 1] - h[1]          # 0 = on the inner face, + = into the room
    out[nm] = peak(dist)
fl = (np.abs(P[:, 2] + 1.2677) < 0.1) & (O > 0.3) & (np.hypot(P[:, 0] + 0.2, P[:, 1] - 0.4) < 1.2)
out["floor (splat z - physics floor top), m"] = peak(P[fl, 2] + 1.2677)
tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "Table0_top")
R = d.geom_xmat[tb].reshape(3, 3); c = d.geom_xpos[tb]; h = m.geom_size[tb]
loc = (P - c) @ R
on = (np.abs(loc[:, 0]) < h[0] - 0.1) & (np.abs(loc[:, 1]) < h[1] - 0.05) & (np.abs(loc[:, 2]) < 0.15) & (O > 0.3)
out["desk top (splat z - physics top face), m"] = peak(loc[on, 2] - h[2])
# desk front edge: splats on the desk top, their extent across the desk (local y) vs the physics box
on2 = (np.abs(loc[:, 0]) < h[0] - 0.1) & (np.abs(loc[:, 2] - h[2]) < 0.03) & (O > 0.3)
out["desk top across-desk extent: splat p1..p99 vs physics +-half"] = [round(float(np.percentile(loc[on2, 1], 1)), 3), round(float(np.percentile(loc[on2, 1], 99)), 3), round(float(h[1]), 3)]
print(json.dumps(out, indent=1)); json.dump(out, open(OUT.parent / "alignment_check.json", "w"), indent=1)
