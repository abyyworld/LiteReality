"""Independent recheck of export_path's Spark vs gsplat numbers (9 held-out frames, 960x720).
Own PSNR; tests whether gsplat's premultiplied colour was double-multiplied by alpha."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from litereality_agent.pipeline.evaluate import fidelity as fd
E = Path.home() / "LiteReality/results/diagnosis/export_path"
SCAN = Path.home() / "LiteReality-scans/stray_atrium_q600"
frames = json.loads((E / "variants_9views.json").read_text())["setup"]["frames"]
bg = np.asarray(fd.BACKGROUND, float) / 255
def psnr(a, b): return float(-10 * np.log10(np.mean((a - b) ** 2)))
rows = []
for f in frames:
    photo = fd.load_photo(SCAN / f"frame_{f:05d}.jpg", (960, 720))
    sp = np.asarray(Image.open(E / f"renders/base_{f:05d}.png").convert("RGBA"), float) / 255
    gs = np.asarray(Image.open(E / f"renders/gsplat_sh1_aa_{f:05d}.png").convert("RGBA"), float) / 255
    a_s, a_g = sp[..., 3:], gs[..., 3:]
    sp_rgb = sp[..., :3] * a_s + bg * (1 - a_s)
    gs_straight = gs[..., :3] * a_g + bg * (1 - a_g)      # what compare_gsplat did
    gs_premul = gs[..., :3] + bg * (1 - a_g)              # gsplat colour is already premultiplied
    # 8-bit precision of each image's own detail: mean |laplacian|
    lap = lambda x: np.abs(4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:]).mean()
    rows.append(dict(frame=f, spark=psnr(sp_rgb, photo), gs_as_scored=psnr(gs_straight, photo),
                     gs_premul=psnr(gs_premul, photo), spark_vs_gs=psnr(sp_rgb, gs_premul),
                     alpha_lt_099_spark=float((a_s < .99).mean()), alpha_lt_099_gs=float((a_g < .99).mean()),
                     lap_photo=float(lap(photo)), lap_spark=float(lap(sp_rgb)), lap_gs=float(lap(gs_premul))))
out = {"per_view": rows}
for k in rows[0]:
    if k != "frame": out[k] = round(float(np.mean([r[k] for r in rows])), 4)
d = np.array([r["gs_premul"] - r["spark"] for r in rows])
out["gs_premul_minus_spark"] = dict(mean=round(float(d.mean()), 3), se=round(float(d.std(ddof=1) / 3), 3), better=int((d > 0).sum()))
(Path(__file__).parent / "recheck.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "per_view"}, indent=1))
for r in rows: print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
