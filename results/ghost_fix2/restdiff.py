"""The room at rest, published v2 against the re-cut bundle: the same views rendered by
callcheck.py's rest mode, per view the mean absolute pixel difference (max over RGB), the share of
pixels off by more than 40, and the share showing the bare scene background (223, 229, 232).
Noise floor: the first three views rendered twice in one bundle.

    uv run python results/ghost_fix2/restdiff.py RAW_FIX RAW_V2 VIEWS.json OUT.json SHEET.jpg
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FIX, V2, VIEWS, OUT, SHEET = (Path(a) for a in sys.argv[1:6])
views = json.loads(VIEWS.read_text())
BG = np.array([223, 229, 232])


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), np.int16)[70:]       # below the toolbar


def holes(a):
    return float((np.abs(a - BG).max(-1) <= 2).mean())


out, rows = {}, []
for k in views:
    a, b = load(FIX / f"rest_{k}.png"), load(V2 / f"rest_{k}.png")
    d = np.abs(a - b).max(-1)
    out[k] = {"mad": round(float(d.mean()), 2), "frac_gt40": round(float((d > 40).mean()), 4),
              "holes_fix2": round(holes(a), 4), "holes_v2": round(holes(b), 4)}
    rows.append((k, a, b, d))
for k in list(views)[:3]:
    for t, folder in (("fix2", FIX), ("v2", V2)):
        a, b = load(folder / f"rest_{k}.png"), load(folder / f"rest_{k}_again.png")
        d = np.abs(a - b).max(-1)
        out[f"noise_{t}_{k}"] = {"mad": round(float(d.mean()), 2),
                                 "frac_gt40": round(float((d > 40).mean()), 4)}
vals = [v for k, v in out.items() if not k.startswith("noise")]
out["summary"] = {"views": len(vals), "mad_mean": round(float(np.mean([v["mad"] for v in vals])), 2),
                  "frac_gt40_mean": round(float(np.mean([v["frac_gt40"] for v in vals])), 4),
                  "frac_gt40_max": max(v["frac_gt40"] for v in vals)}
OUT.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["summary"]))
worst = sorted(rows, key=lambda r: -out[r[0]]["frac_gt40"])[:6]
W, H = 640, 337
s = Image.new("RGB", (3 * W, H * len(worst)))
for i, (k, a, b, d) in enumerate(worst):
    for j, im in enumerate((b, a, np.stack([np.clip(d * 3, 0, 255)] * 3, -1))):
        t = Image.fromarray(im.astype(np.uint8)).resize((W, H))
        label = ["v2 published", "v2_chairs_fix2", "|diff| x3"][j]
        ImageDraw.Draw(t).text((8, 8), f"{k} {label} >40: {out[k]['frac_gt40']}", fill=(255, 255, 0))
        s.paste(t, (j * W, i * H))
s.save(SHEET, quality=85)
