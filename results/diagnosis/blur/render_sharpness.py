"""How soft is the v2 render compared with its photo, at the resolution a viewer sees?
For each full-res (1920x1440) Spark render in renders/ and its photo:
  * sigma_eq: the Gaussian blur (px @1920) that, applied to the photo, best matches the render
    (MSE after a per-image 3x4 colour fit, grey, central 80 %): "render ~= photo blurred by sigma".
  * PSNR cc at 1920x1440, 960x720, 480x360 (the scorer's size).
  * radial power spectrum ratio render/photo.
Plus the metric-blindness check: the photo blurred by sigma 1, 2, 3 px scored against itself at
1920 and at 480. Writes render_sharpness.json, render_spectrum.png, crops_*.png."""
import json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; R = HERE / "renders"
def load(p): return np.asarray(Image.open(p).convert("RGB"), np.float64) / 255
def area(x, w, h): return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)
def cfit(a, b):  # a -> b, 3x4 affine colour
    A = np.c_[a.reshape(-1, 3), np.ones(a.shape[0] * a.shape[1])]; M, *_ = np.linalg.lstsq(A, b.reshape(-1, 3), rcond=None)
    return np.clip(A @ M, 0, 1).reshape(a.shape)
def psnr(a, b): return float(10 * np.log10(1 / np.mean((a - b) ** 2)))
def radial(g):
    g = g - g.mean(); win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g * win))) ** 2; h, w = g.shape; yy, xx = np.indices(P.shape)
    r = np.hypot((yy - h / 2) / h, (xx - w / 2) / w); bins = np.linspace(0, 0.5, 49); idx = np.digitize(r.ravel(), bins)
    return np.array([P.ravel()[idx == k].mean() for k in range(1, 49)]), 0.5 * (bins[1:] + bins[:-1])
out = {"pairs": {}}
fig, ax = plt.subplots(figsize=(8, 4.5))
names = sorted({p.name.rsplit("_", 1)[0] for p in R.glob("*_render.png")})
for n in names:
    ph, rd = load(R / f"{n}_photo.png"), load(R / f"{n}_render.png")
    rc = cfit(rd, ph)
    gp, gr = cv2.cvtColor(ph.astype(np.float32), cv2.COLOR_RGB2GRAY), cv2.cvtColor(rc.astype(np.float32), cv2.COLOR_RGB2GRAY)
    H, W = gp.shape; cy, cx = slice(int(.1 * H), int(.9 * H)), slice(int(.1 * W), int(.9 * W))
    errs = {}
    for s in [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]:
        b = gp if s == 0 else cv2.GaussianBlur(gp, (0, 0), s)
        A = np.c_[b[cy, cx].ravel(), np.ones(b[cy, cx].size)]; m, *_ = np.linalg.lstsq(A, gr[cy, cx].ravel(), rcond=None)
        errs[s] = float(np.mean((A @ m - gr[cy, cx].ravel()) ** 2))
    s_eq = min(errs, key=errs.get)
    row = {"sigma_eq_px1920": s_eq, "mse_by_sigma": {str(k): round(v * 1e4, 3) for k, v in errs.items()},
           "psnr_cc_1920": round(psnr(rc, ph), 2), "psnr_cc_960": round(psnr(cfit(area(rd, 960, 720), area(ph, 960, 720)), area(ph, 960, 720)), 2),
           "psnr_cc_480": round(psnr(cfit(area(rd, 480, 360), area(ph, 480, 360)), area(ph, 480, 360)), 2)}
    L = lambda g: float(cv2.Laplacian(g, cv2.CV_32F, ksize=1).var())
    row["lap_render_over_photo"] = {str(w): round(L(area(gr, w, w * 3 // 4)) / L(area(gp, w, w * 3 // 4)), 3) for w in (1920, 960, 480)}
    sp, f = radial(gp); sr, _ = radial(gr); ratio = sr / sp
    row["power_ratio_half_at_cyc_per_px"] = float(f[np.argmax(ratio < 0.5)]) if (ratio < 0.5).any() else None
    ax.plot(f, ratio, label=n)
    out["pairs"][n] = row
    # crops: centre 480x360 at 1:1
    y0, x0 = H // 2 - 180, W // 2 - 240
    crop = np.concatenate([ph[y0:y0 + 360, x0:x0 + 480], rc[y0:y0 + 360, x0:x0 + 480]], 1)
    Image.fromarray((crop * 255).astype(np.uint8)).save(HERE / f"crop_{n}.png")
    print(n, json.dumps(row))
ax.axhline(1, color="k", lw=.5); ax.set_yscale("log"); ax.set_xlabel("cycles/pixel at 1920x1440 (480-px metric sees only < 0.125)")
ax.set_ylabel("power: v2 render / photo"); ax.legend(fontsize=8); ax.axvline(0.125, ls=":", color="#d1242f"); ax.axvline(0.1875, ls=":", color="#888")
ax.set_title("Spark render of v2 vs the photo, full resolution"); fig.tight_layout(); fig.savefig(HERE / "render_spectrum.png", dpi=110)
# metric blindness: blur a photo, score at 1920 vs 480
ph = load(R / "eval_1219_photo.png"); bl = {}
for s in (1, 2, 3):
    b = cv2.GaussianBlur(ph, (0, 0), s)
    bl[str(s)] = {"psnr_1920": round(psnr(b, ph), 2), "psnr_480": round(psnr(area(b, 480, 360), area(ph, 480, 360)), 2)}
out["photo_blurred_vs_itself"] = bl
print("blindness", bl)
(HERE / "render_sharpness.json").write_text(json.dumps(out, indent=1))
