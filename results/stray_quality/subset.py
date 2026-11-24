"""Stricter view of the fixed eval set: means over the eval frames that have NO training camera
within 5 cm and 5 deg in either training scan (fairness.json), plus paired per-view wins."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
fair = json.loads((HERE / "fairness.json").read_text())
near = set()
for k, v in fair.items():
    if isinstance(v, dict):
        near |= set(v["eval_rows_with_train_cam_within_5cm_5deg"])
variants = sys.argv[1:]
d = {v: {r["row"]: r for r in json.loads((HERE / f"{v}_scores.json").read_text())["per_view"]} for v in variants}
rows = sorted(d[variants[0]])
far = [r for r in rows if r not in near]
out = [f"Far subset: {len(far)} of {len(rows)} eval frames (no training camera within 5 cm & 5 deg).", "",
       "| variant | PSNR (far) | PSNR cc (far) | SSIM (far) | LPIPS (far) | views where PSNR cc beats " + variants[0] + " |",
       "|---|---|---|---|---|---|"]
for v in variants:
    def m(k, v=v):
        return np.mean([d[v][r][k] for r in far])

    wins = sum(d[v][r]["psnr_cc"] > d[variants[0]][r]["psnr_cc"] for r in rows)
    out.append(f"| {v} | {m('psnr'):.2f} | {m('psnr_cc'):.2f} | {m('ssim'):.3f} | {m('lpips'):.3f} | {wins}/{len(rows)} |")
text = "\n".join(out) + "\n"
(HERE / "subset.md").write_text(text)
print(text)
