"""Same view, two sensors: each LiteReality Scanner still photo (atrium_1059, ARKit capturedImage JPEG)
paired with the nearest Stray TRAINING frame (q600, same world after yaw+translation alignment, within 35 cm and 15 deg;
15 cm and 8 deg). Laplacian energy per octave (1920 / 960 / 480) and a radial power spectrum ratio.
Writes matched_detail.json and matched_detail.png."""
import glob, json
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
S = Path.home() / "LiteReality-scans/atrium_1059"; Q = Path.home() / "LiteReality-scans/stray_atrium_q600"
def pose(j): return np.asarray(json.loads(Path(j).read_text())["cameraPoseARFrame"]).reshape(4, 4)
def L(g): return cv2.Laplacian(g, cv2.CV_32F, ksize=1).var()
def octs(g): return [L(g), L(cv2.resize(g, (960, 720), interpolation=cv2.INTER_AREA)), L(cv2.resize(g, (480, 360), interpolation=cv2.INTER_AREA))]
def grey(p): return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY).astype(np.float32)
def radial(g):
    g = g - g.mean(); win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2
    h, w = g.shape; yy, xx = np.indices(P.shape); r = np.hypot((yy - h / 2) / h, (xx - w / 2) / w)
    bins = np.linspace(0, 0.5, 49); idx = np.digitize(r.ravel(), bins)
    return np.array([P.ravel()[idx == k].mean() for k in range(1, 49)]), 0.5 * (bins[1:] + bins[:-1])
qs = sorted(Q.glob("frame_*.json")); qp = [pose(j) for j in qs]
qpos = np.array([p[:3, 3] for p in qp])
pairs = []; ratios = []; specs = []
for j in sorted(S.glob("frame_*.json")):
    P = pose(j); d = np.linalg.norm(qpos - P[:3, 3], axis=1)
    ang = np.array([np.degrees(np.arccos(np.clip((np.trace(P[:3, :3].T @ R[:3, :3]) - 1) / 2, -1, 1))) for R in qp])
    k = int(np.argmin(d / 0.35 + ang / 15))
    if d[k] > 0.35 or ang[k] > 15: continue
    a, b = grey(j.with_suffix(".jpg")), grey(qs[k].with_suffix(".jpg"))
    if a.shape != (1440, 1920) or b.shape != (1440, 1920): continue
    oa, ob = octs(a), octs(b)
    ratios.append([ob[i] / oa[i] for i in range(3)])
    sa, f = radial(a); sb, _ = radial(b); specs.append(sb / sa)
    pairs.append({"scanner": j.stem, "stray": qs[k].stem, "dist_cm": round(float(d[k] * 100), 1), "angle_deg": round(float(ang[k]), 1),
                  "lap_scanner": [round(float(x), 1) for x in oa], "lap_stray": [round(float(x), 1) for x in ob]})
ratios = np.array(ratios); specs = np.array(specs)
med = np.median(specs, 0)
out = {"pairs": len(pairs), "stray_over_scanner_lap_median": {"1920": round(float(np.median(ratios[:, 0])), 3), "960": round(float(np.median(ratios[:, 1])), 3), "480": round(float(np.median(ratios[:, 2])), 3)},
       "power_ratio_by_freq": {f"{fr*1920:.0f}px_cycles_per_width": round(float(m), 3) for fr, m in zip(f[::4], med[::4])},
       "freq_where_power_ratio_below_0.5_cyc_per_px": float(f[np.argmax(med < 0.5)]) if (med < 0.5).any() else None,
       "pair_list": pairs}
(HERE / "matched_detail.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "pair_list"}, indent=1))
fig, ax = plt.subplots(figsize=(7, 4))
for s in specs: ax.plot(f, s, color="#bbb", lw=0.5)
ax.plot(f, med, color="#1f6feb", lw=2, label=f"median of {len(specs)} matched pairs")
ax.axhline(1, color="k", lw=0.5); ax.set_yscale("log"); ax.set_xlabel("spatial frequency (cycles/pixel at 1920x1440); Nyquist of 960-px = 0.25, of 480-px = 0.125")
ax.set_ylabel("power: Stray video frame / scanner still"); ax.legend(); ax.set_title("Detail lost in the Stray video frames (same views)")
for x, lab in [(0.25, "960 px"), (0.125, "480 px"), (0.1875, "1440 px")]: ax.axvline(x, ls=":", color="#d1242f"); ax.text(x, 1.3, lab, fontsize=8, color="#d1242f")
fig.tight_layout(); fig.savefig(HERE / "matched_detail.png", dpi=110)
