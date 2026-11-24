"""Score gsplat's render of raw.spz.bin against Spark's (same 9 held-out cameras, 960x720) and
the photos, with variants.py's metrics. -> gsplat_vs_spark.json, gsplat_vs_spark.jpg"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants as V
from litereality_agent.pipeline.evaluate import fidelity as fd
H = Path(__file__).resolve().parent
frames = json.loads((H / "variants_9views.json").read_text())["setup"]["frames"]
views = {v.index: v for v in fd.capture_views(V.SCAN)}
res = {}
tiles = []
for name in ("spark_base", "spark_msd3", "gsplat_sh1_aa", "gsplat_sh0_aa", "gsplat_sh1_classic", "gsplat_sh1_aa_clip01", "gsplat_sh1_aa_sparkpack"):
    rows = []
    for f in frames:
        file = H / "renders" / (f"base_{f:05d}.png" if name == "spark_base" else
                                f"msd3_{f:05d}.png" if name == "spark_msd3" else f"{name}_{f:05d}.png")
        rgba = np.asarray(Image.open(file).convert("RGBA"))
        r = V.score(rgba, views[f])
        sp, _ = fd.composite(np.asarray(Image.open(H / "renders" / f"base_{f:05d}.png").convert("RGBA")), V.SIZE)
        me, _ = fd.composite(rgba, V.SIZE)
        r["psnr_vs_spark_base_960"] = V.psnr(me, sp) if name != "spark_base" else None
        rows.append(r)
    keys = [k for k in rows[0] if k != "frame" and rows[0][k] is not None]
    res[name] = {"mean": {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}, "per_view": rows}
    print(name, json.dumps(res[name]["mean"]))
base = res["spark_base"]["per_view"]
for name in res:
    d = [a["psnr_960"] - b["psnr_960"] for a, b in zip(res[name]["per_view"], base)]
    res[name]["delta_psnr_960_vs_spark_base"] = {"mean": round(float(np.mean(d)), 3),
                                                 "se": round(float(np.std(d, ddof=1) / np.sqrt(len(d))), 3),
                                                 "better_views": int(sum(x > 0 for x in d))}
    print(name, res[name]["delta_psnr_960_vs_spark_base"])
(H / "gsplat_vs_spark.json").write_text(json.dumps(res, indent=1))
# picture: photo | Spark | gsplat for 3 frames, 1:1 centre crops
rows = []
for f in frames[::3]:
    photo = fd.load_photo(V.SCAN / f"frame_{f:05d}.jpg", V.SIZE)
    sp, _ = fd.composite(np.asarray(Image.open(H / f"renders/base_{f:05d}.png").convert("RGBA")), V.SIZE)
    gs, _ = fd.composite(np.asarray(Image.open(H / f"renders/gsplat_sh1_aa_{f:05d}.png").convert("RGBA")), V.SIZE)
    c = lambda x: x[200:520, 320:640]
    rows.append(np.concatenate([c(photo), c(sp), c(gs)], 1))
Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(H / "gsplat_vs_spark.jpg", quality=92)
