"""Independent of the other agent's chair clusters: (a) where each body's splats sit in the world
(served pose), counted in 0.30 m xy discs around the three Stray chair centres, opacity>0.3 only;
(b) of all splats inside each corrected chair box (+2 cm), how many the served bundle left in the room."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.home() / "LiteReality/results/diagnosis/chairs"))
from common import load_spz, model_data, body_pose, bundle, B
import mujoco
from litereality_agent.pipeline.splatroom.split import _geom_boxes, _union_box
OUT = Path.home() / "LiteReality/results/diagnosis/review_chairs"; FLOOR = -1.2677
man = bundle(); mc, dc = model_data(B / "physics/scene_splat.xml")
centres = {"blue": (0.576, 1.337), "teal": (-1.018, 0.016), "dark": (-0.707, 0.416)}
res = {"(a) body splats (op>0.3) by nearest real chair, 0.30 m disc": {}}
W = {}
for nm in ("Chair0", "Chair1", "Chair2"):
    s = load_spz(B / man["bodies"][nm]); R, x = body_pose(mc, dc, nm)
    p = s["pos"].astype(np.float64) @ R.T + x; W[nm] = p
    op = s["op"] > 0.3; row = {"n_opaque": int(op.sum())}
    for k, c in centres.items():
        row[k] = int((op & (np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) < 0.30) & (p[:, 2] > FLOOR + 0.04)).sum())
    row["box origin to blue chair m"] = round(float(np.hypot(*(x[:2] - centres["blue"]))), 2)
    res["(a) body splats (op>0.3) by nearest real chair, 0.30 m disc"][nm] = row
room = load_spz(B / man["room"]); rp = room["pos"][: len(room["pos"]) - man["counts"].get("floor_fill", 0)].astype(np.float64)
rop = room["op"][: len(rp)]; del room
res["(b) splats inside corrected box+2cm (z > floor+0.04)"] = {}
for nm in ("Chair0", "Chair1", "Chair2"):
    b = mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_BODY, nm)
    boxes = [bx for bx in _geom_boxes(mc, dc, [b])]
    def inside(p):
        m = np.zeros(len(p), bool)
        for _b, Rg, c, half in boxes:
            loc = (p - c) @ Rg; m |= np.all(np.abs(loc) <= half + 0.02, 1)
        return m & (p[:, 2] > FLOOR + 0.04)
    ir = inside(rp); others = {o: int(inside(W[o]).sum()) for o in W}
    res["(b) splats inside corrected box+2cm (z > floor+0.04)"][nm] = {"in room": int(ir.sum()), "in room opaque": int((ir & (rop > 0.3)).sum()), **{f"in body {o}": v for o, v in others.items()}}
json.dump(res, open(OUT / "r5_ownership_simple.json", "w"), indent=1); print(json.dumps(res, indent=1))
