import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from litereality_agent.models.splat import rgbd, spz

REPO = Path.home() / "LiteReality"
OUT = Path(sys.argv[1])
s = spz.decode((REPO / "run/atrium_stray_q/v4_train/raw.spz.bin").read_bytes())
rep = json.loads((REPO / "run/atrium_stray_q/v4/report.json").read_text())
floor = rep["instances"]["floor"]
P = s.positions.astype(np.float64)
op = s.opacities
objs = {
    "base887": ([-0.862, 0.418], 0.40, 0.051, 0.405, (0, 255, 255)),
    "teal_-1.17_0.14": ([-1.17, 0.137], 0.40, 0.439, 0.841, (40, 90, 255)),
    "A_-0.80_1.18": ([-0.803, 1.176], 0.35, 0.061, 0.685, (255, 220, 0)),
    "B_-0.38_1.57": ([-0.375, 1.574], 0.45, 0.177, 0.682, (255, 0, 255)),
    "blue_0.60_1.48": ([0.599, 1.48], 0.33, 0.137, 0.821, (255, 40, 40)),
    "dark_-0.69_0.61": ([-0.694, 0.612], 0.36, 0.416, 1.025, (40, 230, 40)),
}
sets = {}
for k, (c, r, b, t, col) in objs.items():
    m = (
        (np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < r)
        & (P[:, 2] > floor + b)
        & (P[:, 2] < floor + t)
        & (op > 0.3)
    )
    sets[k] = (P[m], col, np.array([c[0], c[1], floor + (b + t) / 2]))
    print(k, m.sum())
frames = rgbd.load_frames(Path.home() / "LiteReality-scans/stray_atrium_b3")
rng = np.random.default_rng(0)
for k in ("base887",):
    centre = sets[k][2]
    best = []
    for f in frames:
        w2c = np.linalg.inv(f.c2w)
        cam = w2c[:3, :3] @ centre + w2c[:3, 3]
        z = -cam[2]
        if z < 0.5:
            continue
        u = f.K[0, 0] * cam[0] / z + f.K[0, 2]
        v = -f.K[1, 1] * cam[1] / z + f.K[1, 2]
        W, H = 2 * f.K[0, 2], 2 * f.K[1, 2]
        d = np.hypot(u / W - 0.5, v / H - 0.5)
        best.append((d + 0.1 * abs(z - 2.2), f))
    best.sort(key=lambda x: x[0])
    for _, f in best[:2]:
        im = Image.open(f.image).convert("RGB")
        raw = im.copy()
        dr = ImageDraw.Draw(im)
        w2c = np.linalg.inv(f.c2w)
        for kk, (pts, col, _) in sets.items():
            if len(pts) > 4000:
                pts = pts[rng.choice(len(pts), 4000, replace=False)]
            cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
            z = -cam[:, 2]
            ok = z > 0.1
            u = f.K[0, 0] * cam[ok, 0] / z[ok] + f.K[0, 2]
            v = -f.K[1, 1] * cam[ok, 1] / z[ok] + f.K[1, 2]
            for a, b in zip(u[::2], v[::2]):
                dr.rectangle([a - 1, b - 1, a + 1, b + 1], fill=col)
        both = Image.new("RGB", (im.width * 2, im.height))
        both.paste(raw, (0, 0))
        both.paste(im, (im.width, 0))
        both = both.rotate(-90, expand=True)
        both.thumbnail((1400, 1400))
        both.save(OUT / f"look_{k}_f{f.index:05d}.jpg", quality=85)
        print("saved", k, f.index)
