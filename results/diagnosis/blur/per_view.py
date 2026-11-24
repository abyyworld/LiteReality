"""(3) Per-eval-view error vs distance/angle to the nearest training camera, vs the eval photo's
own blur, and vs how many training views see it. All poses from the raw odometry (one ARKit world).
Writes per_view.json and per_view.png."""
import csv, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; R = HERE.parents[1]
REC = Path.home() / "LiteReality-scans/stray_atrium_raw/901a305fdf"
od = list(csv.DictReader(open(REC / "odometry.csv"), skipinitialspace=True))
xyz = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in od])
def qm(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
Rs = np.array([qm([float(r[k]) for k in ("qx", "qy", "qz", "qw")]) for r in od])
fwd = Rs[:, :, 2]  # OpenCV camera z (looking direction) in world
z = np.load(HERE / "blur_arrays.npz"); Lest = z["Lest"]; rel = z["rel"]
def train_rows(scan):
    return np.array([json.loads(j.read_text())["stray"]["video_frame"] for j in sorted((Path.home() / "LiteReality-scans" / scan).glob("frame_*.json"))])
sets = {"q600": train_rows("stray_atrium_q600"), "q206": train_rows("stray_atrium_q206")}
variants = {"v0": "q206", "v1": "q600", "v2": "q600", "v3": "q600"}
out = {}
rows_all = []
for v, s in variants.items():
    sc = json.loads((R / f"stray_quality/{v}_scores.json").read_text())["per_view"]
    tr = sets[s]
    for r in sc:
        e = r["row"]
        d = np.linalg.norm(xyz[tr] - xyz[e], axis=1)
        ang = np.degrees(np.arccos(np.clip(fwd[tr] @ fwd[e], -1, 1)))
        k = np.argmin(d / 0.05 + ang / 5)
        n_near = int(((d < 0.30) & (ang < 20)).sum())
        rows_all.append({"variant": v, "row": e, "psnr_cc": r["psnr_cc"], "lpips": r["lpips"], "ssim": r["ssim"],
                         "nearest_cm": round(float(d.min() * 100), 1), "nearest_ang_deg": round(float(ang.min()), 1),
                         "best_combo_cm": round(float(d[k] * 100), 1), "best_combo_deg": round(float(ang[k]), 1),
                         "n_train_within_30cm_20deg": n_near, "eval_photo_blur_px1920": round(float(Lest[e]), 2),
                         "nearest_train_blur_px1920": round(float(Lest[tr[k]]), 2)})
for v in variants:
    rr = [x for x in rows_all if x["variant"] == v]
    def c(a, b): return round(float(spearmanr([x[a] for x in rr], [x[b] for x in rr]).correlation), 3)
    out[v] = {"spearman": {f"{m} vs {f}": c(m, f) for m in ("psnr_cc", "lpips") for f in ("best_combo_cm", "best_combo_deg", "nearest_cm", "n_train_within_30cm_20deg", "eval_photo_blur_px1920", "nearest_train_blur_px1920")}}
v2 = [x for x in rows_all if x["variant"] == "v2"]
# simple linear fit psnr_cc ~ a + b*blur_train + c*dist
X = np.array([[1, x["nearest_train_blur_px1920"], x["best_combo_cm"], x["eval_photo_blur_px1920"]] for x in v2]); y = np.array([x["psnr_cc"] for x in v2])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
out["v2_linear_fit_psnr_cc"] = {"intercept": round(coef[0], 2), "per_px_nearest_train_blur": round(coef[1], 3), "per_cm_to_nearest_train": round(coef[2], 3), "per_px_eval_photo_blur": round(coef[3], 3),
                                "r2": round(1 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2), 3)}
out["v2_rows"] = v2
(HERE / "per_view.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "v2_rows"}, indent=1))
for x in sorted(v2, key=lambda x: x["psnr_cc"]): print(x)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, f, lab in zip(ax, ("best_combo_cm", "nearest_train_blur_px1920", "n_train_within_30cm_20deg"), ("distance to nearest training camera (cm)", "blur of that nearest training frame (px @1920)", "# training views within 30 cm & 20 deg")):
    a.scatter([x[f] for x in v2], [x["psnr_cc"] for x in v2], color="#1f6feb"); a.set_xlabel(lab); a.set_ylabel("v2 PSNR cc (dB)")
fig.tight_layout(); fig.savefig(HERE / "per_view.png", dpi=110)
