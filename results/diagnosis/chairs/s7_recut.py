"""Re-cut the same trained splat on the laptop WITHOUT the photo masks, to separate what the masks
did from what the boxes do. Ownership is traced by carrying each splat's index in its opacity
(split() decides ownership from positions only). Same real-chair sets as s6."""
import json, numpy as np
from common import *
from litereality_agent.models.splat.spz import Splats
from litereality_agent.pipeline.splatroom.split import split
import s6_fractions as s6            # noqa: re-runs s6's set-up (P, O, L, true sets)
P, O, L, true, names = s6.P, s6.O, s6.L, s6.true, s6.names
rep = json.loads((B / "report.json").read_text())
fixed = {k: {kk: vv for kk, vv in v.items() if kk in ("dx", "dy", "dyaw_deg")} for k, v in rep["corrections"].items()}
n = len(P)
sp = Splats(P, np.full((n, 3), 0.01, np.float32), np.tile(np.float32([0, 0, 0, 1]), (n, 1)),
            np.full((n, 3), 0.5, np.float32), np.arange(n, dtype=np.float32))
out = {}
def run(tag, **kw):
    m = mujoco.MjModel.from_xml_path(str(B / "physics/scene.xml"))
    res = split(sp, m, fill=False, **kw)
    owner = np.zeros(n, np.int8)
    for nm, s in res.bodies.items():
        owner[s.opacities.astype(np.int64)] = names.index(nm)
    d = {"corrections": res.corrections, "counts": res.counts, "true_chairs": {}}
    for k, ix in true.items():
        d["true_chairs"][k] = {names[i]: round(float((owner[ix] == i).mean()), 3) for i in range(4)}
    for i, nm in enumerate(names[1:4], 1):
        mine = owner == i
        inany = np.zeros(n, bool)
        for ix in true.values(): inany[ix] = True
        d.setdefault("body_junk_share", {})[nm] = round(float((mine & ~inany).sum() / max(mine.sum(), 1)), 3)
    out[tag] = d
    print(tag, json.dumps(d, default=str))
run("served (masks, as in bundle)", corrections=fixed) if False else None
out["served (masks)"] = {"true_chairs": {k: {nm: v for nm, v in s6.rows[k]["by_count"].items()} for k in true}}
run("boxes only, scanned poses (no snap, no masks)", snap_bodies=False)
run("boxes only, served corrected poses", corrections=fixed)
run("boxes only, snap (no masks)", snap_bodies=True)
json.dump(out, open(OUT / "recut_compare.json", "w"), indent=1, default=str)
