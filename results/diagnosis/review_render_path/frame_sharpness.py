"""Sharpness of every Stray q600 photo (mean |Laplacian| of grey at 480x360, and the frame's
ratio to the 90th-percentile frame): how many training photos are motion-blurred. One photo at a time."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
S = Path.home() / "LiteReality-scans/stray_atrium_q600"
vals = []
for p in sorted(S.glob("frame_*.jpg")):
    with Image.open(p) as im:
        im.draft("L", (480, 360)); g = np.asarray(im.convert("L").resize((480, 360), Image.Resampling.BOX), float) / 255
    lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    vals.append((int(p.stem[6:]), float(lap.mean())))
v = np.array([x[1] for x in vals]); ref = np.percentile(v, 90)
# local relative sharpness: each frame vs the sharpest of its +-5 neighbours (same content)
loc = np.array([v[i] / v[max(0, i - 5):i + 6].max() for i in range(len(v))])
out = {"n": len(v), "lap_pct": {q: round(float(np.percentile(v, q)), 4) for q in (5, 25, 50, 75, 95)},
       "local_ratio_pct": {q: round(float(np.percentile(loc, q)), 3) for q in (5, 10, 25, 50)},
       "frac_below_0.7_of_local_best": round(float((loc < 0.7).mean()), 3),
       "frac_below_0.5_of_local_best": round(float((loc < 0.5).mean()), 3)}
(Path(__file__).parent / "frame_sharpness.json").write_text(json.dumps({**out, "per_frame": vals}, indent=1))
print(json.dumps(out, indent=1))
