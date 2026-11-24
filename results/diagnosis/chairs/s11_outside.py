"""(c) Of each real chair's splats: how far outside its nearest chair box (served corrected poses)
they are, how many are faint (opacity<0.2) or big (sigma>5 cm), and how many are at floor level."""
import json, numpy as np
from common import *
import s6_fractions as s6
from litereality_agent.pipeline.splatroom.split import _union_box
P, O, S, L, true = s6.P, s6.O, s6.S, s6.L, s6.true
m, d = model_data(B / "physics/scene_splat.xml")
boxes = {n: _union_box(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)) for n in ("Chair0", "Chair1", "Chair2")}
FLOOR = -1.2677
out = {}
for k, ix in true.items():
    p = P[ix].astype(np.float64)
    best = np.full(len(p), np.inf); who = np.empty(len(p), object)
    for n, (_b, R, c, h) in boxes.items():
        dist = np.linalg.norm(np.maximum(np.abs((p - c) @ R) - h, 0), axis=1)
        better = dist < best; best[better] = dist[better]; who[better] = n
    r = {"n": len(ix),
         "inside box+2cm": round(float((best <= 0.02).mean()), 3),
         "2-10 cm outside": round(float(((best > 0.02) & (best <= 0.10)).mean()), 3),
         "10-30 cm outside": round(float(((best > 0.10) & (best <= 0.30)).mean()), 3),
         ">30 cm outside": round(float((best > 0.30).mean()), 3),
         "nearest box": {n: round(float((who == n).mean()), 3) for n in boxes},
         "faint (op<0.2)": round(float((O[ix] < 0.2).mean()), 3),
         "big (sigma>5cm)": round(float((S[ix].max(1) > 0.05).mean()), 3),
         "below floor+0.10": round(float((p[:, 2] < FLOOR + 0.10).mean()), 3),
         "chair top above floor (p99), m": round(float(np.percentile(p[:, 2], 99) - FLOOR), 3)}
    out[k] = r; print(k, json.dumps(r))
print("physics chair box top above floor:", round(float(max(c[2] + abs(R[2]) @ h for _b, R, c, h in boxes.values()) - FLOOR), 3))
json.dump(out, open(OUT / "outside_box.json", "w"), indent=1)
