"""Was each spot seen by the atrium_1059 LiDAR at all? Floor points (z < floor+0.04) and object points
(floor+0.08..1.05) within 0.2 m (xy) of each scanned box centre and each Stray chair centre.
Also the RoomPlan chair objects' spacing / size / yaw regularity."""
import json, numpy as np
from pathlib import Path
from litereality_agent.models.splat import rgbd
from litereality_agent.pipeline.quickroom.roomplan import parse_room
OUT = Path.home() / "LiteReality/results/diagnosis/review_chairs"; FLOOR = -1.2677
frames = rgbd.load_frames(Path.home() / "LiteReality-scans/atrium_1059")
pts = []
with np.errstate(all="ignore"):
    for f in frames:
        d, c = rgbd.read_lidar(f); H, W = d.shape
        sx, sy = W / f.size[0], H / f.size[1]
        fx, fy, cx, cy = f.K[0, 0] * sx, f.K[1, 1] * sy, f.K[0, 2] * sx, f.K[1, 2] * sy
        v, u = np.mgrid[0:H, 0:W]; ok = (d > 0.1) & (d < 3.5)
        z = d[ok]; x = (u[ok] - cx) / fx * z; y = (v[ok] - cy) / fy * z
        cam = np.stack([x, -y, -z, np.ones_like(z)], 1); pts.append((cam @ f.c2w.T)[:, :3])
S = np.concatenate(pts); S = S[np.isfinite(S).all(1)]
plan = parse_room(Path.home() / "LiteReality-scans/atrium_1059/roomplan/room.usdz")
ch = {k: o for k, o in plan.objects.items() if o.category == "chair"}
res = {"roomplan_chairs": {k: {"center": o.center, "size": o.size, "yaw": o.yaw} for k, o in ch.items()}}
c = [np.array(ch[k].center[:2]) for k in sorted(ch)]
res["spacing_0to1"] = (c[1] - c[0]).round(4).tolist(); res["spacing_1to2"] = (c[2] - c[1]).round(4).tolist()
spots = {k: tuple(ch[k].center[:2]) for k in ch}
spots.update({"Stray blue": (0.576, 1.337), "Stray teal": (-1.018, 0.016), "Stray dark": (-0.707, 0.416)})
for k, xy in spots.items():
    r = np.hypot(S[:, 0] - xy[0], S[:, 1] - xy[1]) < 0.2
    res[k] = {"floor pts": int((r & (S[:, 2] < FLOOR + 0.04)).sum()),
              "object pts floor+0.08..1.05": int((r & (S[:, 2] > FLOOR + 0.08) & (S[:, 2] < FLOOR + 1.05)).sum()),
              "dist to nearest Stray chair centre m": round(float(min(np.hypot(xy[0] - a, xy[1] - b) for a, b in ((0.576, 1.337), (-1.018, 0.016), (-0.707, 0.416)))), 3)}
json.dump(res, open(OUT / "r3_scan_coverage.json", "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
