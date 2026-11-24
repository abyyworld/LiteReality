"""Reviewer check: is the coarse error a spatially varying colour field (what a per-image
bilateral grid absorbs in training and the exported splat lacks)? PSNR after a 3x4 colour fit per
tile (8x6 tiles = 240 px at 1920), vs one global fit; at 1920 and at 60 px wide."""
import json
from pathlib import Path
import numpy as np, cv2
from PIL import Image
D = Path(__file__).resolve().parents[1] / "blur/renders"
def load(p): return np.asarray(Image.open(p).convert("RGB"), np.float64) / 255
def fit(a, b):
    A = np.c_[a.reshape(-1, 3), np.ones(a.size // 3)]; M, *_ = np.linalg.lstsq(A, b.reshape(-1, 3), rcond=None)
    return np.clip(A @ M, 0, 1).reshape(a.shape)
def ps(e): return round(float(10 * np.log10(1 / e)), 2)
out = {}
for n in ["train_234", "train_2002", "train_1228", "eval_224", "eval_2010", "eval_1219"]:
    R, P = load(D / f"{n}_render.png"), load(D / f"{n}_photo.png")
    res = {}
    for w in (1920, 480):
        h = w * 3 // 4
        r = cv2.resize(R, (w, h), interpolation=cv2.INTER_AREA); p = cv2.resize(P, (w, h), interpolation=cv2.INTER_AREA)
        g = ps(np.mean((fit(r, p) - p) ** 2))
        T = w // 8; err = 0.0
        out_img = np.zeros_like(p)
        for y in range(0, h, T):
            for x in range(0, w, T):
                out_img[y:y+T, x:x+T] = fit(r[y:y+T, x:x+T], p[y:y+T, x:x+T])
        l = ps(np.mean((out_img - p) ** 2))
        # a perfect-but-blurred model, same local fit, for scale
        res[w] = {"global_cc": g, "tile_cc": l}
    out[n] = res; print(n, res, flush=True)
P = load(D / "eval_224_photo.png"); X = cv2.GaussianBlur(P, (0, 0), 2)
out["ref_photo_sigma2_tile_cc_1920"] = ps(np.mean((np.concatenate([np.concatenate([fit(X[y:y+240, x:x+240], P[y:y+240, x:x+240]) for x in range(0, 1920, 240)], 1) for y in range(0, 1440, 240)], 0) - P) ** 2))
print(out["ref_photo_sigma2_tile_cc_1920"])
(Path(__file__).parent / "local_cc.json").write_text(json.dumps(out, indent=1))
