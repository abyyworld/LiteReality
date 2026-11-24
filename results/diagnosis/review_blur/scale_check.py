"""Reviewer check: at what spatial scale is the render-vs-photo error? cc-PSNR (per-image 3x4
colour fit) of the 1920x1440 render/photo pairs after area-downsampling to coarser sizes. If the
error were fine blur / few-px misregistration, coarse sizes would score far higher."""
import json
from pathlib import Path
import numpy as np, cv2
from PIL import Image
D = Path(__file__).resolve().parents[1] / "blur/renders"
def load(p): return np.asarray(Image.open(p).convert("RGB"), np.float64) / 255
def cc(a, b):
    A = np.c_[a.reshape(-1, 3), np.ones(a.size // 3)]; M, *_ = np.linalg.lstsq(A, b.reshape(-1, 3), rcond=None)
    return float(10 * np.log10(1 / np.mean((np.clip(A @ M, 0, 1) - b.reshape(-1, 3)) ** 2)))
out = {}
for n in ["train_234", "train_2002", "train_1228", "eval_224", "eval_2010", "eval_1219"]:
    R, P = load(D / f"{n}_render.png"), load(D / f"{n}_photo.png")
    row = {}
    for w in (1920, 480, 240, 120, 60):
        h = w * 3 // 4
        r = cv2.resize(R, (w, h), interpolation=cv2.INTER_AREA) if w != 1920 else R
        p = cv2.resize(P, (w, h), interpolation=cv2.INTER_AREA) if w != 1920 else P
        row[w] = round(cc(r, p), 2)
    # same for a perfect model of the photo blurred by sigma 2 px and by a 4 px shift
    out[n] = row; print(n, row, flush=True)
P = load(D / "eval_224_photo.png")
for lab, X in [("photo_sigma2", cv2.GaussianBlur(P, (0, 0), 2)), ("photo_sigma8", cv2.GaussianBlur(P, (0, 0), 8)), ("photo_shift4", np.roll(P, 4, 1))]:
    row = {}
    for w in (1920, 480, 240, 120, 60):
        h = w * 3 // 4
        f = (lambda x: cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)) if w != 1920 else (lambda x: x)
        row[w] = round(cc(f(X[8:-8, 8:-8]), f(P[8:-8, 8:-8])), 2)
    out["ref_eval224_" + lab] = row; print(lab, row, flush=True)
(Path(__file__).parent / "scale_check.json").write_text(json.dumps(out, indent=1))
