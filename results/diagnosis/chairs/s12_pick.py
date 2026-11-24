"""What a click on each visible chair grabs: rays from real Stray camera positions through that
chair's opaque splats, first hit against the served physics (the client picks on these meshes,
app.js pick(): static geometry blocks, a movable body's hidden mesh is grabbed)."""
import json, numpy as np
from common import *
import s6_fractions as s6
from litereality_agent.models.splat import rgbd
P, O, true = s6.P, s6.O, s6.true
m, d = model_data(B / "physics/scene_splat.xml")
frames = {f.index: f for f in rgbd.load_frames(Path.home() / "LiteReality-scans/stray_atrium_q600")}
rng = np.random.default_rng(1)
geomid = np.zeros(1, np.int32)
out = {}
for k, ix in true.items():
    ix = ix[O[ix] > 0.5]
    tally = {}
    for fi in (12, 63, 67, 68, 254, 280, 400, 500):
        cam = frames[fi].c2w[:3, 3].astype(np.float64)
        for j in rng.choice(ix, min(300, len(ix)), replace=False):
            tgt = P[j].astype(np.float64); v = tgt - cam; dist = np.linalg.norm(v); v /= dist
            dd = mujoco.mj_ray(m, d, cam, v, None, 1, -1, geomid)
            if dd < 0 or dd > dist + 0.05:
                hit = "nothing before the splat (click ignored)"
            else:
                b = int(m.geom_bodyid[geomid[0]])
                hit = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) if b else "static (blocked)"
            tally[hit] = tally.get(hit, 0) + 1
    tot = sum(tally.values())
    out[k] = {h: round(c / tot, 3) for h, c in sorted(tally.items(), key=lambda t: -t[1])}
    print(k, out[k])
json.dump(out, open(OUT / "pick_what_a_click_grabs.json", "w"), indent=1)
