"""Score the Eyeful Tower splats as the BROWSER draws them (Spark 2.3.1), against the trainer.

The trainer (gsplat 1.5.3, scripts/eyeful/train_eyeful.py) reports PSNR / SSIM / LPIPS over the 71
camera-17 test images of office1b, rendered by gsplat's CUDA rasteriser from its own float
splats. This script renders the exported SPZ of each run with Spark at the very same test cameras,
in headless Chromium (`pipeline/evaluate`'s render_splat.html: three.js 0.180 + Spark 2.3.1, the
room call's renderer), and scores it with the trainer's own metric code (torchmetrics, as gsplat's
simple_trainer: PSNR data range 1, SSIM data range 1 on RGB, LPIPS AlexNet normalize=True), so the
two sets of numbers differ only by the path: SPZ quantisation + Spark's rasteriser.

Cameras (scripts/eyeful/fetch_eval.py stages them): COLMAP world->camera R, t (OpenCV axes) and
PINHOLE K at the photo's own size, 1:1 (no supersampling: gsplat renders 1:1 too). A run whose
splats are in gsplat's renormalised world (arm A) carries the 4x4 COLMAP->splat similarity in its
result.json "world_transform"; the camera is moved by it (its scale taken out of the rotation).
The render page wants a camera->world pose in OpenGL axes (x right, y up, looking down -z) in the
SPZ's own frame; it turns mesh and camera by the same Z-up->Y-up matrix, so any frame works.
Background black (gsplat's eval background); `coverage` says how much of each view Spark drew.

    uv run --frozen --with playwright --with lpips==0.1.4 --with torchmetrics==1.6.0 \\
        python scripts/eyeful/spark_score.py trainer A_30000          # metric check, all 71
    ... spark_score.py spark A_30000 [--views N] [--gl metal|swiftshader] [--settle]
    ... spark_score.py sheet A_30000 B_5000 B_1250                   # results/eyeful/compare.jpg

Local inputs (fetched from the Volume `litereality-eyeful`, not in the repo):
    ~/LiteReality-scans/eyeful_eval/spark_eval/{test_cams.json, gt/*.jpg}
    ~/LiteReality-scans/eyeful_eval/runs/<run>/renders/val_step<N>_<i>.jpg   (gt | render, q92)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "eyeful"
WORK = Path("~/LiteReality-scans/eyeful_eval").expanduser()
CAMS = WORK / "spark_eval" / "test_cams.json"
GT_DIR = WORK / "spark_eval" / "gt"
SHEET_VIEWS = (0, 20, 40, 60)        # the step-2 render picks
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
GL_ARGS = {
    # Chromium's GPU path on this Mac (ANGLE on Metal): what a browser on a laptop does.
    "metal": ("--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist",
              "--enable-gpu-rasterization"),
    # The software path every earlier score used (pipeline/evaluate CHROMIUM_ARGS).
    "swiftshader": ("--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                    "--ignore-gpu-blocklist"),
}


PAGE = Path(__file__).with_name("render_splat_eyeful.html")
SQRT5 = float(np.sqrt(5.0))
# How Spark draws. "call" is the room call's own SparkRenderer (room_ops/roomcall/web/app.js:
# maxStdDev sqrt(5), every other option Spark's default: blurAmount 0.3 with opacity compensation,
# maxPixelRadius 512, clipXY 1.4, radial sort, float16 centres) and its near plane 0.05. Each
# later preset changes one more thing toward what gsplat did in training, cumulatively, so the
# steps split the trainer-vs-browser gap; "matched" is the last.
#   std3     maxStdDev 3 (gsplat cuts a splat at 3 sigma)
#   blur     arm A only (trained without --antialiased): gsplat adds 0.3 px^2 to the 2D covariance
#            WITHOUT opacity compensation = Spark preBlurAmount 0.3, blurAmount 0. Arm B
#            (--antialiased) already matches Spark's default (0.3 with compensation).
#   limits   no splat size cap (maxPixelRadius 1e5), minAlpha 1/255 and near plane 0.01 as
#            gsplat, depth (not radial) sort. clipXY stays 1.4: Spark does not clamp the EWA
#            Jacobian as gsplat does (tx/tz within 1.3 tan(fov/2)), so a splat centred off screen
#            near the camera blows up to cover the view once it is not culled (clipXY 100: 15.6
#            dB on 7 A views; 3: -1.7 dB, 2: -1.1, 1.4: -0.68, 1.15: -0.79 against the trainer)
#   matched  + float32 splat centres (SplatMesh extSplats + SparkRenderer accumExtSplats)
CALL = {"maxStdDev": SQRT5, "ext": False, "near": 0.05}
PRESETS = {
    "call": dict(CALL),
    "std3": {**CALL, "maxStdDev": 3.0},
    "blur": {**CALL, "maxStdDev": 3.0, "match_blur": True},
    "limits": {**CALL, "maxStdDev": 3.0, "match_blur": True, "maxPixelRadius": 1e5,
               "clipXY": 1.4, "minAlpha": 1 / 255, "near": 0.01, "sortRadial": False},
    "matched": {**CALL, "maxStdDev": 3.0, "match_blur": True, "maxPixelRadius": 1e5,
                "clipXY": 1.4, "minAlpha": 1 / 255, "near": 0.01, "sortRadial": False,
                "ext": True},
}
FAR = 1000.0


def spark_options(preset: str, arm: str) -> dict:
    """The page's `spark` job options for a preset and a training arm (A: no antialiasing)."""
    o = dict(PRESETS[preset])
    if o.pop("match_blur", False) and arm == "A":
        o["preBlurAmount"], o["blurAmount"] = 0.3, 0.0
    return o


def render_page(spz: bytes, cameras: list[dict], size: tuple[int, int], opts: dict, gl: str,
                settle_ms: int, timeout_s: float = 3600.0) -> tuple[list[np.ndarray], list]:
    """Draw `cameras` (fidelity.three_camera dicts) with render_splat_eyeful.html in headless
    Chromium; one browser for the whole list. Returns RGBA arrays and per-view frame counts."""
    import base64
    import io

    from playwright.sync_api import sync_playwright

    fd = _fd()
    out, frames, errors = [], [], []
    with fd.serve_page(spz, page_file=PAGE, route_name="/scene.spz",
                       kind="application/octet-stream") as base, sync_playwright() as pw:
        browser = pw.chromium.launch(args=list(GL_ARGS[gl]), headless=True)
        try:
            version = browser.version
            page = browser.new_page(viewport={"width": size[0], "height": size[1]})
            page.set_default_timeout(int(timeout_s * 1000))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{base}/")
            page.wait_for_function("window.renderReady === true", timeout=240000)
            for i in range(0, len(cameras), fd.BATCH):
                job = {"glb": "scene.spz", "width": size[0], "height": size[1],
                       "views": cameras[i:i + fd.BATCH], "spark": opts, "settleMs": settle_ms}
                res = page.evaluate("job => window.renderViews(job)", job)
                frames += res["frames"]
                for url in res["images"]:
                    png = base64.b64decode(url.split(",", 1)[1])
                    with Image.open(io.BytesIO(png)) as im:
                        out.append(np.asarray(im.convert("RGBA")))
        finally:
            browser.close()
    if errors:
        print("page errors:", errors[:3], flush=True)
    return out, frames, version


def _fd():
    sys.path.insert(0, str(REPO / "src"))
    from litereality_agent.pipeline.evaluate import fidelity
    return fidelity


def ssim_torchmetrics(a: np.ndarray, b: np.ndarray) -> float:
    """torchmetrics' StructuralSimilarityIndexMeasure(data_range=1.0) of two HxWx3 images, in
    float64 numpy: per channel, separable 11-tap Gaussian (sigma 1.5, taps exp(-x^2 / 2 sigma^2)
    at x = -5..5, normalised), C1 = 0.01^2, C2 = 0.03^2, variances E[x^2] - E[x]^2, the map
    cropped by the 5-pixel border it pads (so the padding mode does not matter), then the mean.
    Agrees with torchmetrics (float32) to within 3e-4 on four A test views, which is float32
    rounding in E[x^2] - E[x]^2."""
    from scipy.ndimage import correlate1d

    x = np.arange(-5, 6, dtype=np.float64)
    k = np.exp(-0.5 * (x / 1.5) ** 2)
    k /= k.sum()

    def blur(z: np.ndarray) -> np.ndarray:
        return correlate1d(correlate1d(z, k, axis=0, mode="mirror"), k, axis=1, mode="mirror")

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for c in range(3):
        p, t = np.asarray(a[..., c], np.float64), np.asarray(b[..., c], np.float64)
        mp, mt = blur(p), blur(t)
        vp, vt, cov = blur(p * p) - mp * mp, blur(t * t) - mt * mt, blur(p * t) - mp * mt
        m = ((2 * mp * mt + c1) * (2 * cov + c2)) / ((mp * mp + mt * mt + c1) * (vp + vt + c2))
        vals.append(m[5:-5, 5:-5])
    return float(np.mean(vals))


class Metrics:
    """gsplat 1.5.3 simple_trainer's eval metrics, per image (float RGB in [0, 1], HxWx3): PSNR
    (data range 1), SSIM (torchmetrics' definition, `ssim_torchmetrics`), LPIPS (torchmetrics,
    AlexNet, normalize=True).

    PSNR and SSIM are computed in float64 numpy on the CPU. On this Mac torchmetrics' SSIM took
    ~15 s an image on the CPU, and on MPS it returned garbage now and then (SSIM 1.66, 1.41, ...;
    PSNR off by 2.5 dB once) under memory pressure, though standalone rechecks agreed with the
    CPU. LPIPS runs on MPS twice and is recomputed on the CPU if the two disagree."""

    def __init__(self) -> None:
        import torch
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True
                                                           ).eval().to(self.device)
        self._cpu_lpips = None
        self.fallbacks = 0

    def _t(self, img: np.ndarray, device: str):
        t = self.torch
        x = t.from_numpy(np.ascontiguousarray(img, np.float32)).permute(2, 0, 1)[None]
        return x.contiguous().to(device)

    def _lpips(self, real: np.ndarray, render: np.ndarray, cpu: bool = False) -> float:
        t = self.torch
        if cpu and self._cpu_lpips is None:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self._cpu_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex",
                                                                    normalize=True).eval()
        net, dev = (self._cpu_lpips, "cpu") if cpu else (self.lpips, self.device)
        with t.no_grad():
            v = float(net(self._t(render, dev), self._t(real, dev)))
        net.reset()
        return v

    def __call__(self, real: np.ndarray, render: np.ndarray, only_psnr: bool = False) -> dict:
        render = np.clip(np.asarray(render, np.float64), 0.0, 1.0)
        real = np.asarray(real, np.float64)
        mse = float(np.mean((render - real) ** 2))
        out = {"psnr": 10.0 * float(np.log10(1.0 / max(mse, 1e-12)))}
        if only_psnr:
            return out
        out["ssim"] = ssim_torchmetrics(render, real)
        first, second = self._lpips(real, render), self._lpips(real, render)
        if not (np.isfinite(first) and 0.0 <= first <= 2.0 and abs(first - second) < 1e-5):
            print(f"    MPS LPIPS disagrees ({first} vs {second}): recomputing on the CPU",
                  flush=True)
            self.fallbacks += 1
            first = self._lpips(real, render, cpu=True)
        out["lpips"] = first
        return out


def load_cams() -> list[dict]:
    return json.loads(CAMS.read_text())["views"]


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), np.float32) / 255.0


def trainer_render(run: str, i: int, width: int) -> np.ndarray:
    """The trainer's own render of test image i: the right half of its gt | render canvas."""
    files = sorted((WORK / "runs" / run / "renders").glob(f"val_step*_{i:04d}.jpg"))
    if not files:
        raise FileNotFoundError(f"{run}: no trainer render {i}")
    canvas = load_rgb(files[-1])
    if canvas.shape[1] != 2 * width:
        raise ValueError(f"{files[-1]}: {canvas.shape[1]} px wide, want 2 x {width}")
    return canvas[:, width:]


def run_info(run: str) -> dict:
    return json.loads((RESULTS / run / "result.json").read_text())


def pose_in_splat_frame(cam: dict, world: np.ndarray) -> np.ndarray:
    """Camera -> splat-frame pose, OpenGL axes, from COLMAP world->camera (OpenCV axes)."""
    R, t = np.asarray(cam["R"], float), np.asarray(cam["t"], float)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ t
    s = float(np.cbrt(np.linalg.det(world[:3, :3])))
    move = world.copy()
    out = move @ c2w
    out[:3, :3] /= s                     # a similarity: keep the rotation orthonormal
    return out @ CV_TO_GL


def subset(n: int | None, total: int) -> list[int]:
    if not n or n >= total:
        return list(range(total))
    picks = set(np.linspace(0, total - 1, n).round().astype(int).tolist())
    picks |= {i for i in SHEET_VIEWS if i < total}
    return sorted(picks)


def cmd_trainer(run: str) -> dict:
    """Metrics of the trainer's JPEG renders over all test images: checks this script's metric
    code (and the q92 JPEG) against the trainer's own float numbers."""
    cams, metrics = load_cams(), Metrics()
    rows = []
    for i, cam in enumerate(cams):
        real = load_rgb(GT_DIR / cam["name"])
        rows.append({"i": i, "name": cam["name"], **metrics(real, trainer_render(run, i,
                                                                               cam["width"]))})
    info = run_info(run)
    step = max(info["stats"], key=lambda k: int(k.split("step")[1]))
    out = {"run": run, "views": len(rows), "trainer_reported": info["stats"][step],
           "trainer_jpeg_recomputed": {k: round(float(np.mean([r[k] for r in rows])), 4)
                                       for k in ("psnr", "ssim", "lpips")},
           "metric_cpu_fallbacks": metrics.fallbacks, "per_view": rows}
    (RESULTS / "spark").mkdir(parents=True, exist_ok=True)
    (RESULTS / "spark" / f"{run}_trainer_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(run, "reported", {k: round(out["trainer_reported"][k], 4)
                            for k in ("psnr", "ssim", "lpips")},
          "recomputed from JPEG", out["trainer_jpeg_recomputed"])
    return out


def cmd_spark(run: str, spz_name: str, views: int | None, gl: str, settle: bool,
              chunk: int, prerotate: bool = False, tag: str = "", preset: str = "legacy",
              settle_ms: int = 1000, overrides: list[str] | None = None) -> dict:
    """preset "legacy": pipeline/evaluate's render_splat.html exactly (the call's settings, 6
    frames a view, so a view can be read before its sort is back); any PRESETS name:
    render_splat_eyeful.html with those settings, each view drawn until settled (`settle_ms`
    and three identical frames)."""
    fd = _fd()
    opts = None if preset == "legacy" else spark_options(preset, run_info(run)["arm"])
    for kv in overrides or ():
        k, _, v = kv.partition("=")
        opts[k] = json.loads(v)

    class Renderer(fd.SplatRenderer):
        def _args(self) -> list[str]:
            return list(GL_ARGS[gl])

    cams, info = load_cams(), run_info(run)
    world = np.asarray(info.get("world_transform") or np.eye(4), float)
    spz_path = Path(spz_name) if Path(spz_name).is_absolute() else RESULTS / run / spz_name
    # --prerotate: the SPZ was turned by inv(Z_UP_TO_THREE) (scripts/eyeful/spz_variant.py), so
    # the page's Z-up -> Y-up turn brings it back to the training frame: move the camera too.
    pre = np.linalg.inv(fd.Z_UP_TO_THREE) if prerotate else np.eye(4)
    spz_bytes = spz_path.read_bytes()
    idx = subset(views, len(cams))
    metrics = Metrics()
    renderer = Renderer(timeout_s=3600)
    name = (f"{run}_{Path(spz_name).stem}_{gl}" + ("" if preset == "legacy" else f"_{preset}")
            + (f"_{tag}" if tag else ""))
    settle_log: list = []
    keep = WORK / "spark" / name
    keep.mkdir(parents=True, exist_ok=True)
    rows, render_s = [], 0.0
    for at in range(0, len(idx), chunk):
        part = idx[at:at + chunk]
        size = (cams[part[0]]["width"], cams[part[0]]["height"])
        if any((cams[i]["width"], cams[i]["height"]) != size for i in part):
            raise ValueError("a chunk mixes image sizes")
        three = []
        for i in part:
            c = cams[i]
            K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)
            view = fd.View(i, GT_DIR / c["name"], K, size, pre @ pose_in_splat_frame(c, world))
            cam = fd.three_camera(view, size)
            if opts is not None:
                proj = fd.projection_matrix(*fd.intrinsics_at(view, size), size,
                                            near=opts["near"], far=FAR)
                cam["projection"] = proj.T.reshape(-1).tolist()
            three.append(cam)
        if settle:
            three = [v for v in three for _ in (0, 1)]
        t0 = time.perf_counter()
        if opts is None:
            renders = renderer.render(spz_bytes, three, size)
        else:
            renders, frames, version = render_page(spz_bytes, three, size, opts, gl, settle_ms)
            renderer.version = version
            settle_log.extend(frames)
        render_s += time.perf_counter() - t0
        if settle:
            renders = renders[1::2]
        for i, rgba in zip(part, renders, strict=True):
            c = cams[i]
            real = load_rgb(GT_DIR / c["name"])
            spark, cover = fd.composite(rgba, size, background=(0, 0, 0))
            spark = spark.astype(np.float32)
            train = trainer_render(run, i, c["width"])
            row = {"i": i, "name": c["name"], "coverage": round(float(cover.mean()), 4)}
            row.update({f"spark_{k}": v for k, v in metrics(real, spark).items()})
            row.update({f"trainer_{k}": v for k, v in metrics(real, train).items()})
            row["spark_vs_trainer_psnr"] = metrics(train, spark, only_psnr=True)["psnr"]
            row["mean_rgb_spark"] = [round(float(x), 4) for x in spark.reshape(-1, 3).mean(0)]
            row["mean_rgb_trainer"] = [round(float(x), 4) for x in train.reshape(-1, 3).mean(0)]
            rows.append(row)
            Image.fromarray((np.clip(spark, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
                keep / f"{i:04d}.jpg", quality=92)
            print(f"  {run} view {i}: spark {row['spark_psnr']:.2f} dB, trainer "
                  f"{row['trainer_psnr']:.2f} dB, spark vs trainer {row['spark_vs_trainer_psnr']:.2f}"
                  f" dB, coverage {row['coverage']:.3f}", flush=True)
        del renders

    def mean(key):
        return round(float(np.mean([r[key] for r in rows])), 4)

    summary = {k: mean(k) for k in rows[0] if isinstance(rows[0][k], float)}
    summary["gap_psnr_db"] = round(summary["spark_psnr"] - summary["trainer_psnr"], 4)
    summary["gap_ssim"] = round(summary["spark_ssim"] - summary["trainer_ssim"], 4)
    summary["gap_lpips"] = round(summary["spark_lpips"] - summary["trainer_lpips"], 4)
    step = max(info["stats"], key=lambda k: int(k.split("step")[1]))
    out = {"run": run, "spz": spz_name, "spz_mb": round(spz_path.stat().st_size / 1e6, 1),
           "splats": info.get("splats"), "renderer": f"{renderer.name} {renderer.version}",
           "gl": gl, "settle": settle, "prerotate": prerotate, "preset": preset,
           "page": "pipeline/evaluate/render_splat.html" if opts is None else PAGE.name,
           "spark_options": opts, "settle_ms": None if opts is None else settle_ms,
           "settle_frames": settle_log, "views": len(rows), "view_indices": idx,
           "size": [cams[idx[0]]["width"], cams[idx[0]]["height"]], "render_scale": 1,
           "background": "black", "render_s": round(render_s, 1),
           "trainer_reported_all71": {k: info["stats"][step][k] for k in ("psnr", "ssim", "lpips")},
           "summary": summary, "metric_cpu_fallbacks": metrics.fallbacks, "per_view": rows}
    (RESULTS / "spark").mkdir(parents=True, exist_ok=True)
    name += "_settle" if settle else ""
    (RESULTS / "spark" / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{name}: {len(rows)} views, Spark PSNR {summary['spark_psnr']:.3f} vs trainer "
          f"{summary['trainer_psnr']:.3f} (gap {summary['gap_psnr_db']:+.3f} dB); SSIM "
          f"{summary['spark_ssim']:.4f} vs {summary['trainer_ssim']:.4f}; LPIPS "
          f"{summary['spark_lpips']:.4f} vs {summary['trainer_lpips']:.4f}; render {render_s:.0f} s")
    return out


def _label(img: Image.Image, text: str, size: int = 34) -> Image.Image:
    sys.path.insert(0, str(REPO / "src"))
    from litereality_agent.fonts import font

    d = ImageDraw.Draw(img)
    f = font(size)
    box = d.textbbox((10, 8), text, font=f)
    d.rectangle((box[0] - 6, box[1] - 5, box[2] + 6, box[3] + 5), fill=(0, 0, 0))
    d.text((10, 8), text, fill=(255, 255, 255), font=f)
    return img


def cmd_sheet(entries: list[str], scale: float = 0.25) -> Path:
    """compare.jpg: one row per test view in SHEET_VIEWS; ground truth, then for each run its
    trainer render and its Spark render, each labelled with its PSNR."""
    cams = load_cams()
    loaded = []
    for e in entries:                  # "<run>" or "<run>:<spark json stem>"
        run, _, stem = e.partition(":")
        stem = stem or f"{run}_splats_sh3_metal"
        doc = json.loads((RESULTS / "spark" / f"{stem}.json").read_text())
        loaded.append((run, stem, {r["i"]: r for r in doc["per_view"]}))
    w, h = cams[0]["width"], cams[0]["height"]
    tw, th = int(w * scale), int(h * scale)
    cols = 1 + 2 * len(loaded)
    canvas = Image.new("RGB", (tw * cols, th * len(SHEET_VIEWS)), (0, 0, 0))

    def tile(arr_or_path, text):
        im = (Image.open(arr_or_path).convert("RGB") if isinstance(arr_or_path, Path)
              else Image.fromarray((np.clip(arr_or_path, 0, 1) * 255 + 0.5).astype(np.uint8)))
        return _label(im.resize((tw, th), Image.Resampling.LANCZOS), text, size=15)

    for r, i in enumerate(SHEET_VIEWS):
        c = cams[i]
        canvas.paste(tile(GT_DIR / c["name"], f"ground truth {c['name']}"), (0, r * th))
        for k, (run, stem, rows) in enumerate(loaded):
            row = rows[i]
            canvas.paste(tile(trainer_render(run, i, c["width"]),
                              f"{run} gsplat {row['trainer_psnr']:.2f} dB"),
                         ((1 + 2 * k) * tw, r * th))
            canvas.paste(tile(WORK / "spark" / stem.removesuffix("_settle") / f"{i:04d}.jpg",
                              f"{run} Spark {row['spark_psnr']:.2f} dB"),
                         ((2 + 2 * k) * tw, r * th))
    path = RESULTS / "compare.jpg"
    canvas.save(path, quality=88)
    print(path, canvas.size)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("trainer")
    s.add_argument("run")
    s = sub.add_parser("spark")
    s.add_argument("run")
    s.add_argument("--spz", default="splats_sh3.spz")
    s.add_argument("--views", type=int, default=None, help="an even spread of N test views "
                   "(plus the sheet's four); default all 71")
    s.add_argument("--gl", choices=sorted(GL_ARGS), default="metal")
    s.add_argument("--settle", action="store_true", help="draw each view twice, keep the second")
    s.add_argument("--chunk", type=int, default=8, help="views per browser launch")
    s.add_argument("--prerotate", action="store_true",
                   help="the SPZ is pre-turned by inv(Z_UP_TO_THREE) (spz_variant.py --prerotate)")
    s.add_argument("--tag", default="", help="suffix for the output name")
    s.add_argument("--preset", default="legacy", choices=["legacy", *PRESETS],
                   help="how Spark draws (see PRESETS); legacy = pipeline/evaluate's page as is")
    s.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                   help="presets: override one page option, e.g. --set maxPixelRadius=512")
    s.add_argument("--settle-ms", type=int, default=1000,
                   help="presets: draw each view at least this long, then until 3 frames agree")
    s = sub.add_parser("sheet")
    s.add_argument("entries", nargs="+")
    a = p.parse_args(argv)
    if a.cmd == "trainer":
        cmd_trainer(a.run)
    elif a.cmd == "spark":
        cmd_spark(a.run, a.spz, a.views, a.gl, a.settle, a.chunk, a.prerotate, a.tag,
                  a.preset, a.settle_ms, a.set)
    else:
        cmd_sheet(a.entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
