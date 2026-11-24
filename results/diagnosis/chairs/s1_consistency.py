"""Which pose were the body splats stored against? raw = room + bodies? """
import json, numpy as np
from scipy.spatial import cKDTree
from common import *
man = bundle()
raw = load_spz(B / "raw.spz.bin")
print("raw", len(raw["pos"]))
mc, dc = model_data(B / "physics/scene_splat.xml")
mo, do = model_data(B / "physics/scene.xml")
tree = cKDTree(raw["pos"])
for name, f in man["bodies"].items():
    s = load_spz(B / f)
    loc = s["pos"].astype(np.float64)
    print(name, len(loc), "body-frame centroid", loc.mean(0).round(3), "min z", loc[:, 2].min().round(3))
    for tag, (m, d) in (("corrected", (mc, dc)), ("scanned", (mo, do))):
        R, x = body_pose(m, d, name)
        w = loc @ R.T + x
        dist, _ = tree.query(w[:5000])
        print(f"   {tag:9s} pose -> world; median NN dist to raw {np.median(dist)*1000:.2f} mm, p95 {np.percentile(dist,95)*1000:.2f} mm")
room = load_spz(B / man["room"])
print("room", len(room["pos"]), "sum bodies+room-fill", len(room["pos"]) - man["counts"].get("floor_fill", 0) + sum(v for k, v in man["counts"].items() if k not in ("room", "floor_fill")))
