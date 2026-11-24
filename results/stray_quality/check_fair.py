"""Fairness check: no eval frame (by recording row) in any training scan, and the nearest
training camera to each eval camera (pool world -> each scan's world)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
POOL = Path("~/LiteReality-scans/stray_atrium_pool").expanduser()

def rows_poses(scan):
    out = {}
    for f in sorted(scan.glob("frame_*.json")):
        m = json.loads(f.read_text())
        out[int(m["stray"]["row"])] = (np.asarray(m["cameraPoseARFrame"], float).reshape(4, 4), m["time"])
    return out

def T(scan):
    return np.asarray(json.loads((scan / "scene_alignment.json").read_text())["strayImport"]["transform"], float)

ev = [tuple(line.split("#")[0].split()[:2])
      for line in (HERE / "eval_frames.txt").read_text().splitlines() if line.split("#")[0].strip()]
ev = [(int(r), s) for r, s in ev]
pool = rows_poses(POOL)
res = {"eval_frames": len(ev)}
for name in sys.argv[1:]:
    scan = Path(name).expanduser()
    tr = rows_poses(scan)
    M = T(scan) @ np.linalg.inv(T(POOL))
    overlap = sorted(set(tr) & {r for r, _ in ev})
    P = np.stack([p for p, _ in tr.values()])
    times = np.array([t for _, t in tr.values()])
    trows = np.array(list(tr))
    dmin, amin, tmin, rowgap, near = [], [], [], [], []
    for r, s in ev:
        e = M @ pool[r][0]
        d = np.linalg.norm(P[:, :3, 3] - e[:3, 3], axis=1)
        R = np.einsum("nij,ik->njk", P[:, :3, :3], e[:3, :3])
        ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        # "same view" = close in both: combined score
        if ((d < 0.05) & (ang < 5.0)).any():
            near.append(r)
        dmin.append(d.min())
        amin.append(ang[np.argmin(d)])
        tmin.append(np.abs(times - pool[r][1]).min())
        rowgap.append(np.abs(trows - r).min())
    res[scan.name] = {"train_frames": len(tr), "eval_rows_in_train": overlap,
                      "min_row_gap": int(min(rowgap)), "min_time_gap_s": round(float(min(tmin)), 3),
                      "nearest_train_cam_m": {"min": round(float(min(dmin)), 3), "median": round(float(np.median(dmin)), 3)},
                      "eval_rows_with_train_cam_within_5cm_5deg": near,
                      "angle_at_nearest_deg": {"min": round(float(min(amin)), 1), "median": round(float(np.median(amin)), 1)}}
print(json.dumps(res, indent=1))
(HERE / "fairness.json").write_text(json.dumps(res, indent=1) + "\n")
