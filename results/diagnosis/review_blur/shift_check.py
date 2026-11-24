"""Reviewer check: is the render misregistered against its own photo (global/local shift)?
For each render/photo pair at 1920x1440: global phase-correlation shift, cc-PSNR before/after
shifting, and a grid of local shifts (256 px tiles)."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
D = Path(__file__).resolve().parents[1] / "blur/renders"
def load(p): return np.asarray(Image.open(p).convert("RGB"), np.float64) / 255.0
def grey(a): return a @ np.array([0.299, 0.587, 0.114])
def phase(a, b):
    a = a - a.mean(); b = b - b.mean()
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fft2(a * w) * np.conj(np.fft.fft2(b * w)); F /= np.abs(F) + 1e-9
    r = np.fft.ifft2(F).real; i = np.unravel_index(np.argmax(r), r.shape)
    dy, dx = [int(v) if v < s // 2 else int(v) - s for v, s in zip(i, r.shape)]
    # subpixel parabola
    def sp(m, c, p):
        d = (m - 2 * c + p); return 0.0 if d == 0 else 0.5 * (m - p) / d
    y, x = i; H, W = r.shape
    sy = sp(r[(y-1)%H, x], r[y, x], r[(y+1)%H, x]); sx = sp(r[y, (x-1)%W], r[y, x], r[y, (x+1)%W])
    return dy + sy, dx + sx, float(r.max())
def cc_psnr(render, photo, m=None):
    X = render.reshape(-1, 3); Y = photo.reshape(-1, 3)
    if m is not None: X = X[m.ravel()]; Y = Y[m.ravel()]
    A = np.c_[X, np.ones(len(X))]; M, *_ = np.linalg.lstsq(A, Y, rcond=None)
    e = np.mean((A @ M - Y) ** 2); return 10 * np.log10(1 / e)
out = {}
for name in ["train_234", "train_2002", "train_1228", "eval_224", "eval_2010", "eval_1219"]:
    R = load(D / f"{name}_render.png"); P = load(D / f"{name}_photo.png")
    gR, gP = grey(R), grey(P)
    dy, dx, pk = phase(gP, gR)
    sh = np.roll(np.roll(R, int(round(dy)), 0), int(round(dx)), 1)
    m = np.zeros(gR.shape, bool); b = 40; m[b:-b, b:-b] = True
    base = cc_psnr(R, P, m); after = cc_psnr(sh, P, m)
    tiles = []
    T = 256
    for y0 in range(64, gP.shape[0] - T - 63, T):
        for x0 in range(64, gP.shape[1] - T - 63, T):
            a = gP[y0:y0+T, x0:x0+T]; c = gR[y0:y0+T, x0:x0+T]
            if a.std() < 0.03: continue
            ty, tx, tp = phase(a, c)
            if tp > 0.08: tiles.append((round(ty, 2), round(tx, 2), round(tp, 3)))
    mags = [float(np.hypot(t[0], t[1])) for t in tiles]
    out[name] = {"global_shift_px_dy_dx": [round(dy, 2), round(dx, 2)], "peak": round(pk, 3),
                 "cc_psnr_1920": round(base, 2), "cc_psnr_after_global_shift": round(after, 2),
                 "tiles_n": len(tiles), "tile_shift_median_px": round(float(np.median(mags)), 2) if mags else None,
                 "tile_shift_p90_px": round(float(np.percentile(mags, 90)), 2) if mags else None,
                 "tiles": tiles}
    print(name, {k: v for k, v in out[name].items() if k != "tiles"}, flush=True)
(Path(__file__).parent / "shift_check.json").write_text(json.dumps(out, indent=1))
