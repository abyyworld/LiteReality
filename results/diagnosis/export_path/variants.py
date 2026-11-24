"""Trainer output -> SPZ -> Spark: render the trained v2 atrium with one setting changed at a time
and score each against the trainer's OWN held-out photos (every 8th frame of stray_atrium_q600).

    uv run --frozen --with playwright python results/diagnosis/export_path/variants.py info
    uv run --frozen --with playwright python results/diagnosis/export_path/variants.py run base app ...

Writes renders/<config>_<frame>.png and variants.json beside this file. CPU only (SwiftShader).
"""
from __future__ import annotations

import base64, gzip, io, json, struct, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

from litereality_agent.models.splat import spz
from litereality_agent.pipeline.evaluate import fidelity as fd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PAGE = HERE / "render_variant.html"
BUNDLE = REPO / "run/atrium_stray_q/v2"
SCAN = Path("~/LiteReality-scans/stray_atrium_q600").expanduser()
SIZE = (960, 720)
SMALL = (480, 360)
import os
OUT = HERE / os.environ.get("VARIANTS_OUT", "variants.json")
RENDERS = HERE / "renders"

# config -> (query string, spz transform)
CONFIGS = {
    "base": ("", None),                                   # = score.py / render_splat.html
    "app": ("tm=neutral&aa=1", None),                     # + app.js:170-174 (MSAA, Neutral tone map)
    "tm_only": ("tm=neutral", None),
    "msd8": (f"msd={np.sqrt(8)}", None),                  # Spark's default cut-off
    "msd3": ("msd=3", None),                              # gsplat's 3 sigma
    "flag_off": ("", "flag_off"),                         # SPZ antialiased flag cleared
    "sh0": ("", "sh0"),                                   # view-dependent SH dropped
    "blur0": ('sr={"blurAmount":0}', None),               # no 2D filter at all
    "classic": ('sr={"blurAmount":0,"preBlurAmount":0.3}', None),   # 3DGS dilation, no compensation
    "rgbwide": ('enc={"rgbMin":-0.5,"rgbMax":1.5,"lnScaleMin":-12,"lnScaleMax":9,"sh1Max":1,"sh2Max":1,"sh3Max":1}', None),
}


def spz_variant(kind: str | None) -> bytes:
    data = (BUNDLE / "raw.spz.bin").read_bytes()
    if kind is None:
        return data
    raw = bytearray(gzip.decompress(data))
    if kind == "flag_off":
        raw[14] = 0                                       # header byte 14: flags
        return gzip.compress(bytes(raw), 6, mtime=0)
    if kind == "sh0":
        _m, _v, n, degree, *_ = struct.unpack_from("<IIIBBBB", raw)
        k = spz.sh_dim(degree)
        raw[12] = 0
        end = 16 + n * (9 + 1 + 3 + 3 + 4)
        return gzip.compress(bytes(raw[:end]), 6, mtime=0)
    raise ValueError(kind)


def held_out_views(count: int) -> list:
    from litereality_agent.models.splat.rgbd import load_frames
    frames = load_frames(SCAN)
    held = {f.index for i, f in enumerate(frames) if i % 8 == 0}
    views = [v for v in fd.capture_views(SCAN) if v.index in held]
    picks = np.linspace(0, len(views) - 1, count + 2).round().astype(int)[1:-1]
    return [views[i] for i in picks], len(held)


def render(query: str, spz_bytes: bytes, cameras: list[dict], size, passes=6) -> tuple[list, dict]:
    from playwright.sync_api import sync_playwright
    out, info = [], {}
    with fd.serve_page(spz_bytes, page_file=PAGE, route_name="/scene.spz",
                       kind="application/octet-stream") as base, sync_playwright() as pw:
        browser = pw.chromium.launch(args=list(fd.CHROMIUM_ARGS), headless=True)
        try:
            page = browser.new_page(viewport={"width": size[0], "height": size[1]})
            page.set_default_timeout(1_800_000)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base}/?{query}")
            page.wait_for_function("window.renderReady === true", timeout=60_000)
            if cameras:
                for i in range(0, len(cameras), 3):
                    urls = page.evaluate("job => window.renderViews(job)",
                                         {"glb": "scene.spz", "width": size[0], "height": size[1],
                                          "views": cameras[i:i + 3], "passes": passes})
                    for url in urls:
                        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as im:
                            out.append(np.asarray(im.convert("RGBA")))
            info = page.evaluate("u => window.sparkInfo(u)", "scene.spz")
            info["errors"] = errors
        finally:
            browser.close()
    return out, info


# ── metrics ─────────────────────────────────────────────────────────────────────────────────────

def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def cc_affine(render, real):                              # score.py's 3x4 fit
    x = render.reshape(-1, 3)
    A = np.c_[x, np.ones(len(x))]
    M, *_ = np.linalg.lstsq(A, real.reshape(-1, 3), rcond=None)
    return np.clip(A @ M, 0, 1).reshape(render.shape)


def cc_gsplat(img, ref, num_iters=5, eps=0.5 / 255):      # gsplat examples/utils.color_correct (multinerf)
    im = img.reshape(-1, 3).astype(np.float64)
    rf = ref.reshape(-1, 3).astype(np.float64)
    unclipped = lambda z: (z >= eps) & (z <= 1 - eps)
    mask0 = unclipped(im)
    for _ in range(num_iters):
        a = [im[:, c:c + 1] * im[:, c:] for c in range(3)] + [im, np.ones((len(im), 1))]
        a = np.concatenate(a, 1)
        warp = []
        for c in range(3):
            m = mask0[:, c] & unclipped(im[:, c]) & unclipped(rf[:, c])
            w, *_ = np.linalg.lstsq(np.where(m[:, None], a, 0), np.where(m, rf[:, c], 0), rcond=None)
            warp.append(w)
        im = np.clip(a @ np.stack(warp, 1), 0, 1)
    return im.reshape(img.shape)


def hf_energy(img):
    g = img.mean(2)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.mean(lap ** 2))


def box(img, size):
    h, w = img.shape[:2]
    fy, fx = h // size[1], w // size[0]
    return img.reshape(size[1], fy, size[0], fx, 3).mean(axis=(1, 3))


def score(rgba, view):
    real = fd.load_photo(view.image, SIZE)
    render, cov = fd.composite(rgba, SIZE)
    rs, ps = box(render, SMALL), box(real, SMALL)
    return {
        "frame": view.index, "coverage": round(float(cov.mean()), 4),
        "psnr_960": psnr(render, real), "psnr_480": psnr(rs, ps),
        "psnr_cc_affine_480": psnr(cc_affine(rs, ps), ps),
        "psnr_cc_gsplat_480": psnr(cc_gsplat(rs, ps), ps),
        "psnr_cc_gsplat_960": psnr(cc_gsplat(render, real), real),
        "ssim_480": fd.ssim(ps, rs),
        "sharpness_ratio_960": hf_energy(render) / hf_energy(real),   # <1: smoother than the photo
        "sharpness_ratio_480": hf_energy(rs) / hf_energy(ps),
    }


def main():
    cmd = sys.argv[1]
    if cmd == "info":
        small = (BUNDLE / "body-Chair0-32ebba88587e0254.spz").read_bytes()
        for q in ("", "tm=neutral&aa=1"):
            _, info = render(q, small, [], (64, 48))
            print(json.dumps(info, indent=1))
        return
    count = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--views=")), 3))
    names = [a for a in sys.argv[2:] if not a.startswith("--")]
    views, n_held = held_out_views(count)
    cams = [fd.three_camera(v, SIZE) for v in views]
    RENDERS.mkdir(exist_ok=True)
    doc = json.loads(OUT.read_text()) if OUT.exists() else {}
    doc["setup"] = {"bundle": str(BUNDLE), "scan": str(SCAN), "held_out_total": n_held,
                    "frames": [v.index for v in views], "size": SIZE, "small": SMALL,
                    "renderer": "Chromium (SwiftShader) + three 0.180 + Spark 2.3.1"}
    for name in names:
        query, kind = CONFIGS[name]
        t = time.perf_counter()
        imgs, info = render(query, spz_variant(kind), cams, SIZE)
        rows = []
        for v, rgba in zip(views, imgs):
            Image.fromarray(rgba).save(RENDERS / f"{name}_{v.index:05d}.png")
            rows.append(score(rgba, v))
        mean = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0] if k != "frame"}
        doc.setdefault("configs", {})[name] = {"query": query, "spz": kind, "mean": mean,
                                               "per_view": rows, "seconds": round(time.perf_counter() - t, 1),
                                               "three": info.get("three"), "errors": info.get("errors")}
        OUT.write_text(json.dumps(doc, indent=1))
        print(name, json.dumps(mean), flush=True)



def pack_check(stride=111):
    """Spark's own in-memory copy of the room (PackedSplats, 16 bytes a splat) against the SPZ."""
    data = (BUNDLE / "raw.spz.bin").read_bytes()
    from playwright.sync_api import sync_playwright
    with fd.serve_page(data, page_file=PAGE, route_name="/scene.spz",
                       kind="application/octet-stream") as base, sync_playwright() as pw:
        browser = pw.chromium.launch(args=list(fd.CHROMIUM_ARGS), headless=True)
        try:
            page = browser.new_page(viewport={"width": 64, "height": 48})
            page.set_default_timeout(600_000)
            page.goto(f"{base}/")
            page.wait_for_function("window.renderReady === true", timeout=60_000)
            info = page.evaluate("([u, s]) => window.packInfo(u, s)", ["scene.spz", stride])
        finally:
            browser.close()
    s = spz.decode(data)
    a = np.asarray(info.pop("samples"), dtype=np.float64)
    idx = a[:, 0].astype(int)
    pos, sc, col, op = s.positions[idx], s.scales[idx], s.colors[idx], s.opacities[idx]
    dpos = np.linalg.norm(a[:, 1:4] - pos, axis=1)
    rel_s = np.abs(np.log(np.maximum(a[:, 4:7], 1e-9) / np.maximum(sc, 1e-9)))
    dcol = np.abs(a[:, 7:10] - np.clip(col, 0, 1)) * 255
    dop = np.abs(a[:, 10] - op) * 255
    r = np.linalg.norm(pos, axis=1)
    pct = lambda x: {p: round(float(np.percentile(x, p)), 6) for p in (50, 90, 99, 100)}
    res = {**info, "sampled": len(idx),
           "centre_err_m": pct(dpos), "centre_err_m_by_radius": {
               f"{lo}-{hi} m": pct(dpos[(r >= lo) & (r < hi)]) for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 99))
               if ((r >= lo) & (r < hi)).any()},
           "scale_abs_log_err": pct(rel_s), "colour_err_grey_levels": pct(dcol),
           "colour_clipped_by_spark_frac": float(((col < 0) | (col > 1)).any(1).mean()),
           "opacity_err_levels": pct(dop)}
    (HERE / "pack_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if sys.argv[1:2] == ["pack"]:
    pack_check()
    sys.exit(0)


if __name__ == "__main__":
    main()
