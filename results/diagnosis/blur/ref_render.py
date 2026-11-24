"""Independent reference rasteriser (CPU, numpy, standard 3DGS/EWA maths as gsplat does it) for a
480x360 crop at FULL resolution of one view, from the same raw.spz.bin Spark draws. If this is as
soft as Spark's render, the softness is in the splats (training/data); if it is much sharper,
Spark or the export loses it.

  modes: "aa"   gsplat --antialiased: Sigma2d + 0.3 I, opacity * sqrt(det/det_dilated), 3 sigma
         "plain" Sigma2d + 0.3 I, no opacity compensation (what a viewer that ignores the SPZ
                antialiased flag draws)
Colours: SH degree 1 as stored. Writes ref_<name>_<mode>.png and ref_render.json.

    .venv/bin/python results/diagnosis/blur/ref_render.py train_234
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from litereality_agent.models.splat import spz
from litereality_agent.pipeline.evaluate import fidelity as fd

HERE = Path(__file__).resolve().parent
RUN = Path.home() / "LiteReality/run/atrium_stray_q/v2"
SH_C1 = 0.4886025119029199
CROP_W, CROP_H = 480, 360


def view_for(name: str):
    kind, row = name.split("_")
    meta = json.loads((HERE / "renders/renders.json").read_text()) if False else None
    if kind == "train":
        scan = Path.home() / "LiteReality-scans/stray_atrium_q600"
        for j in scan.glob("frame_*.json"):
            m = json.loads(j.read_text())
            if int(m["stray"]["row"]) == int(row):
                return fd.camera_pose(m["cameraPoseARFrame"]), np.array(m["intrinsics"]).reshape(3, 3)
    else:
        sys.path.insert(0, str(HERE.parents[1] / "stray_quality"))
        import score
        views, _ = score.eval_views(Path.home() / "LiteReality-scans/stray_atrium_q600")
        v = [v for v in views if v.index == int(row)][0]
        return v.pose, v.K
    raise KeyError(name)


def quat_to_R(q):  # x y z w
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def render(name: str, modes=("aa", "plain")):
    pose, K = view_for(name)
    s = spz.decode((RUN / "raw.spz.bin").read_bytes())
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x0, y0 = 1920 // 2 - CROP_W // 2, 1440 // 2 - CROP_H // 2
    R, t = pose[:3, :3], pose[:3, 3]
    W = np.diag([1.0, -1.0, -1.0]) @ R.T                      # world -> OpenCV camera
    pc = (s.positions - t) @ W.T
    z = pc[:, 2]
    keep = (z > 0.05) & (s.opacities > 1 / 255)
    u = fx * pc[:, 0] / np.maximum(z, 1e-6) + cx
    v = fy * pc[:, 1] / np.maximum(z, 1e-6) + cy
    keep &= (u > x0 - 400) & (u < x0 + CROP_W + 400) & (v > y0 - 400) & (v < y0 + CROP_H + 400)
    idx = np.flatnonzero(keep)
    Rg = quat_to_R(s.rotations[idx].astype(np.float64))
    S2 = s.scales[idx].astype(np.float64) ** 2
    Sig = np.einsum("nij,nj,nkj->nik", Rg, S2, Rg)
    Sc = np.einsum("ij,njk,lk->nil", W, Sig, W)
    zi, xi, yi = z[idx], pc[idx, 0], pc[idx, 1]
    J = np.zeros((len(idx), 2, 3))
    J[:, 0, 0] = fx / zi; J[:, 0, 2] = -fx * xi / zi ** 2
    J[:, 1, 1] = fy / zi; J[:, 1, 2] = -fy * yi / zi ** 2
    S2d = np.einsum("nij,njk,nlk->nil", J, Sc, J)
    det0 = S2d[:, 0, 0] * S2d[:, 1, 1] - S2d[:, 0, 1] ** 2
    S2d[:, 0, 0] += 0.3; S2d[:, 1, 1] += 0.3
    det = S2d[:, 0, 0] * S2d[:, 1, 1] - S2d[:, 0, 1] ** 2
    comp = np.sqrt(np.clip(det0 / det, 0, 1))
    inv = np.stack([S2d[:, 1, 1], -S2d[:, 0, 1], S2d[:, 0, 0]], 1) / det[:, None]
    rad = 3 * np.sqrt(np.maximum(S2d[:, 0, 0], S2d[:, 1, 1]))
    ui, vi = u[idx], v[idx]
    hit = (ui + rad > x0) & (ui - rad < x0 + CROP_W) & (vi + rad > y0) & (vi - rad < y0 + CROP_H)
    # colour with SH degree 1
    d = s.positions[idx] - t; d /= np.linalg.norm(d, axis=1, keepdims=True)
    col = s.colors[idx].astype(np.float64)
    if s.sh is not None:
        sh = s.sh[idx].astype(np.float64)
        col = col + SH_C1 * (-d[:, 1:2] * sh[:, 0] + d[:, 2:3] * sh[:, 1] - d[:, 0:1] * sh[:, 2])
    col = np.clip(col, 0, 1)
    order = np.flatnonzero(hit)[np.argsort(zi[hit])]
    print(f"{name}: {len(idx):,} near the crop, {len(order):,} touch it", flush=True)
    out = {}
    for mode in modes:
        t0 = time.time()
        C = np.zeros((CROP_H, CROP_W, 3)); T = np.ones((CROP_H, CROP_W))
        op = s.opacities[idx].astype(np.float64) * (comp if mode == "aa" else 1.0)
        for k in order:
            r = rad[k]
            ax0, ax1 = int(max(x0, np.floor(ui[k] - r))), int(min(x0 + CROP_W, np.ceil(ui[k] + r) + 1))
            ay0, ay1 = int(max(y0, np.floor(vi[k] - r))), int(min(y0 + CROP_H, np.ceil(vi[k] + r) + 1))
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            xs = np.arange(ax0, ax1) + 0.5 - ui[k]; ys = np.arange(ay0, ay1) + 0.5 - vi[k]
            a, b, c = inv[k]
            pw = -0.5 * (a * xs[None, :] ** 2 + 2 * b * xs[None, :] * ys[:, None] + c * ys[:, None] ** 2)
            al = np.minimum(0.99, op[k] * np.exp(pw))
            al[al < 1 / 255] = 0
            sl = (slice(ay0 - y0, ay1 - y0), slice(ax0 - x0, ax1 - x0))
            Tl = T[sl]
            C[sl] += (Tl * al)[..., None] * col[k]
            T[sl] = Tl * (1 - al)
        img = C + T[..., None] * (np.array(fd.BACKGROUND) / 255.0)
        Image.fromarray(np.clip(img * 255 + .5, 0, 255).astype(np.uint8)).save(HERE / f"ref_{name}_{mode}.png")
        out[mode] = {"seconds": round(time.time() - t0, 1), "mean_T_left": round(float(T.mean()), 4)}
        print(name, mode, out[mode], flush=True)
    return out, (x0, y0)


def compare(name, origin):
    x0, y0 = origin
    sl = (slice(y0, y0 + CROP_H), slice(x0, x0 + CROP_W))
    photo = np.asarray(Image.open(HERE / f"renders/{name}_photo.png").convert("RGB"), np.float64)[sl] / 255
    spark = np.asarray(Image.open(HERE / f"renders/{name}_render.png").convert("RGB"), np.float64)[sl] / 255

    def cfit(a, b):
        A = np.c_[a.reshape(-1, 3), np.ones(a.size // 3)]; M, *_ = np.linalg.lstsq(A, b.reshape(-1, 3), rcond=None)
        return np.clip(A @ M, 0, 1).reshape(a.shape)

    def psnr(a, b): return round(float(10 * np.log10(1 / np.mean((a - b) ** 2))), 2)

    def lap(a): return float(cv2.Laplacian(cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2GRAY), cv2.CV_32F, ksize=1).var())
    res = {"photo_lap": round(lap(photo) * 1e4, 2)}
    tiles = [photo, spark]
    for lab, img in [("spark", spark)] + [(m, np.asarray(Image.open(HERE / f"ref_{name}_{m}.png"), np.float64) / 255) for m in ("aa", "plain")]:
        res[lab] = {"psnr_cc_vs_photo": psnr(cfit(img, photo), photo), "lap_over_photo": round(lap(img) / lap(photo), 3)}
        if lab != "spark":
            res[lab]["psnr_vs_spark"] = psnr(img, spark)
            tiles.append(img)
    Image.fromarray((np.concatenate(tiles, 1) * 255).astype(np.uint8)).save(HERE / f"ref_compare_{name}.png")
    return res


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "train_234"
    timing, origin = render(name)
    res = compare(name, origin)
    res["timing"] = timing
    p = HERE / "ref_render.json"
    allr = json.loads(p.read_text()) if p.is_file() else {}
    allr[name] = res
    p.write_text(json.dumps(allr, indent=1))
    print(json.dumps(res, indent=1))
