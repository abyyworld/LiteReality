"""Camera poses, intrinsics and metric depth for a phone video's frames: MapAnything on one GPU.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file .env modal deploy deploy/modal/geometry/app.py
    uv run litereality import-video <video.MOV> --out <scan> --align-to <scan>     # calls it

Its own app, `litereality-geometry`, never included into `litereality` (deploy/modal/app.py), so
deploying it never touches the pipeline's models. Standalone: it imports nothing from this repo
(no __init__.py here, like deploy/modal/depth).

The model. MapAnything (Keetha et al., 3DV 2026; github facebookresearch/map-anything, code
Apache-2.0) in its Apache-2.0 weights variant, HF `facebook/map-anything-apache` (not gated; the
default `facebook/map-anything` is CC-BY-NC and is NOT used). One feed-forward pass over N
images returns, per view, an OpenCV camera-to-world pose, pinhole intrinsics, z-depth and a
confidence, all in metres ("universal metric 3D reconstruction"). Run as its README says:
`MapAnything.from_pretrained(...)`, `preprocess_inputs(views)` and `model.infer(views,
memory_efficient_inference=True, use_amp=True, amp_dtype="bf16", apply_mask=True,
mask_edges=True)`. The README quotes up to 2000 views on 140 GB in memory-efficient mode
(~70 MB a view at 518 px); a 200-view pass fits an L40S (48 GB) with room to spare.

Input size. `load_images` / `preprocess_inputs` with their defaults ("fixed_mapping", 518 set)
resize every image to the 518-px size closest to the batch's aspect (16:9 -> 518 x 294, 9:16 ->
294 x 518, 4:3 -> 518 x 392): scale s = max(tw / W, th / H) (+1e-8), LANCZOS to floor(W s) x
floor(H s), then a centred crop of ((rw - tw) // 2, (rh - th) // 2). This function does exactly
that itself (with the library's own `crop_resize_if_necessary` and `find_closest_aspect_ratio`)
so that it knows the crop, and returns it per frame (`prep`): the client maps the model's
intrinsics and depth back onto its full-resolution photo with it.

`reconstruct(frames, options)`:

    frames   [JPEG bytes], one per view, all the same size, in capture order
    options  {"max_views": int (views per forward pass, default 320),
              "anchors": int (views shared by every pass when there are several, default 32),
              "shared_intrinsics": bool (default True: one calibration pass over
                                         `calib_views` frames, then every view is given the
                                         median K: a video has one lens),
              "calib_views": int (default 48),
              "intrinsics": 3x3 (optional: the camera's known K in the pixels of the JPEGs
                            sent; every view is conditioned on it and there is no calibration
                            pass. The returned K is still the model's own, recovered from its
                            predicted rays: on the atrium video it moved only part of the way
                            towards the K given, 282 -> 330 px for 422 given)}
    returns  {"arrays": .npz bytes holding
                  c2w    (N, 4, 4) float64  camera -> world, OpenCV camera (x right, y down,
                                            z forward), metres, world = view 0's camera
                  K      (N, 3, 3) float64  intrinsics at the model's resolution (model_size)
                  depth  (N, h, w) float16  z-depth in metres, 0 where MapAnything masked it
                                            (ambiguous or an edge)
                  conf_log (N, h, w) float16  log(conf - 1) of MapAnything's confidence
                                            (conf = 1 + exp(x), so this is x: higher is surer;
                                            conf itself rounds to 1.0 in float16 for a third
                                            of a plain room's pixels and loses its order)
              "model_size": [w, h], "prep": [{"size": [W, H], "resized": [rw, rh],
              "crop": [left, top]}] per frame, "passes": [...], "intrinsics_shared": 3x3 or
              None, "metric_scaling": per-pass medians, "seconds": {...}, "gpu": str,
              "model": "<repo>@<revision>", "code": "<commit>", "convention": str}

More views than `max_views`: every pass is the `anchors` views spread evenly over the whole
video (frame 0 first, so each pass's world is frame 0's camera) plus one contiguous block of the
rest. Pass 0 is the reference; every other pass is mapped onto it by the Sim(3) (scale,
rotation, translation) that best takes its anchors' per-pixel 3D points onto pass 0's for the
same pixels of the same images (confident pixels, Umeyama with Huber reweighting), and its
poses and depths are moved by it. `reconstruct([])` only loads the model (a warm-up).

The weights (4.9 GB) and the code are baked into the image at pinned revisions, the weights
checked by SHA-256, and the image build loads the model and runs one CPU forward pass, so a
broken pin fails the deploy rather than the first video.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import modal
import numpy as np

APP_NAME = "litereality-geometry"        # NOT included into the `litereality` app
FUNCTION_NAME = "reconstruct"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
MAPANYTHING_REPO = "https://github.com/facebookresearch/map-anything"
# main as of 2026-08-07 (pyproject version 1.1.4).
MAPANYTHING_COMMIT = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
WEIGHTS_REPO = "facebook/map-anything-apache"     # Apache-2.0 (facebook/map-anything is NC)
WEIGHTS_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
WEIGHTS_SHA256 = "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201"
WEIGHTS_DIR = Path("/opt/map-anything-apache")
# First free wins. ~5 GB of weights + ~70 MB a view: an L40S takes ~500 views a pass.
GPUS = ["H100", "L40S"]
TIMEOUT_SECONDS = 1200
MAX_FRAMES = 1000
DEFAULT_MAX_VIEWS = 320
DEFAULT_ANCHORS = 32
DEFAULT_CALIB_VIEWS = 48
RESOLUTION_SET = 518
CONVENTION = ("c2w: camera -> world 4x4, OpenCV camera (x right, y down, z forward), metres; "
              "world = view 0's camera (MapAnything's reference view). K: OpenCV pinhole in "
              "pixels of the model-size image, pixel centres at integer coordinates (the "
              "library converts to COLMAP's +0.5 only inside its crop code); depth: z along the "
              "optical axis, metres, 0 = masked")

OPTION_KEYS = {"max_views", "anchors", "shared_intrinsics", "calib_views", "intrinsics"}


def _fetch_weights() -> None:
    """Image build: the pinned weights, checked, then the model built and run once on CPU."""
    import hashlib
    import shutil

    from huggingface_hub import hf_hub_download

    for name in ("config.json", "model.safetensors"):
        hf_hub_download(WEIGHTS_REPO, name, revision=WEIGHTS_REVISION, local_dir=WEIGHTS_DIR)
    shutil.rmtree(WEIGHTS_DIR / ".cache", ignore_errors=True)
    digest = hashlib.sha256()
    with open(WEIGHTS_DIR / "model.safetensors", "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError(f"model.safetensors sha256 {digest.hexdigest()} != {WEIGHTS_SHA256}")

    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs
    from PIL import Image

    # Builds the DINOv2 trunk through torch.hub (cached into the image here) and loads the
    # weights strictly.
    model = MapAnything.from_pretrained(str(WEIGHTS_DIR)).eval()
    rng = np.random.default_rng(0)
    imgs = [Image.fromarray(rng.integers(0, 255, (294, 518, 3), dtype=np.uint8))
            for _ in range(2)]
    views = preprocess_inputs([{"img": im} for im in imgs])
    with torch.inference_mode():
        preds = model.infer(views, memory_efficient_inference=False, use_amp=False)
    assert len(preds) == 2
    assert tuple(preds[0]["depth_z"].shape) == (1, 294, 518, 1), preds[0]["depth_z"].shape
    assert tuple(preds[1]["camera_poses"].shape) == (1, 4, 4)
    print("mapanything ok", float(preds[1]["depth_z"].float().mean()))


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
                 index_url=TORCH_INDEX)
    .pip_install(f"mapanything @ git+{MAPANYTHING_REPO}@{MAPANYTHING_COMMIT}",
                 "numpy<2.3", "huggingface_hub>=0.26")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .run_function(_fetch_weights, cpu=8, memory=32768, timeout=3600)
    .env({"HF_HUB_OFFLINE": "1"})
)
app = modal.App(APP_NAME)

_model = None


# --------------------------------------------------------------------------------------------
# Pure helpers (numpy only; tested locally without a GPU)
# --------------------------------------------------------------------------------------------

def parse_options(options: dict | None) -> dict:
    options = dict(options or {})
    unknown = set(options) - OPTION_KEYS
    if unknown:
        raise ValueError(f"unknown options {sorted(unknown)}")
    out = {"max_views": int(options.get("max_views", DEFAULT_MAX_VIEWS)),
           "anchors": int(options.get("anchors", DEFAULT_ANCHORS)),
           "shared_intrinsics": bool(options.get("shared_intrinsics", True)),
           "calib_views": int(options.get("calib_views", DEFAULT_CALIB_VIEWS))}
    if not 2 <= out["max_views"] <= 2000:
        raise ValueError(f"max_views {out['max_views']} is outside 2..2000")
    if not 2 <= out["anchors"] < out["max_views"]:
        raise ValueError(f"anchors {out['anchors']} must be in 2..max_views-1")
    if out["calib_views"] < 2:
        raise ValueError("calib_views must be >= 2")
    out["intrinsics"] = None
    if options.get("intrinsics") is not None:
        K = np.asarray(options["intrinsics"], dtype=np.float64)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("intrinsics must be a finite 3x3 with positive focal lengths")
        out["intrinsics"] = K.tolist()
    return out


def image_K_to_model(K: np.ndarray, prep: dict) -> np.ndarray:
    """Intrinsics in the pixels of the image sent -> at the model's crop (`prep`): the resize
    per axis (pixel centres), then the crop's shift."""
    W, H = prep["size"]
    rw, rh = prep["resized"]
    left, top = prep["crop"]
    ax, ay = rw / W, rh / H
    out = np.array(K, dtype=np.float64).copy()
    out[0, 0] *= ax
    out[0, 1] *= ax
    out[1, 1] *= ay
    out[0, 2] = (out[0, 2] + 0.5) * ax - 0.5 - left
    out[1, 2] = (out[1, 2] + 0.5) * ay - 0.5 - top
    return out


def prep_geometry(size: tuple[int, int], target: tuple[int, int]) -> dict:
    """What `crop_resize_if_necessary(image, resolution=target)` does to a `size` image (no
    intrinsics given): LANCZOS to floor(W s) x floor(H s), s = max(tw / W, th / H) + 1e-8, then
    the centred crop. Mirrors map-anything@MAPANYTHING_COMMIT utils/cropping.py."""
    W, H = (int(v) for v in size)
    tw, th = (int(v) for v in target)
    s = max(tw / W, th / H) + 1e-8
    rw, rh = int(np.floor(W * s)), int(np.floor(H * s))
    return {"size": [W, H], "resized": [rw, rh], "crop": [(rw - tw) // 2, (rh - th) // 2]}


def plan_passes(n: int, max_views: int, anchors: int) -> list[list[int]]:
    """Frame indices of each forward pass. One pass when they fit; else every pass starts with
    the same `anchors` frames (spread over the whole video, frame 0 first) followed by one
    contiguous block of the others."""
    if n <= max_views:
        return [list(range(n))]
    anchor = sorted({int(round(v)) for v in np.linspace(0, n - 1, anchors)})
    rest = [i for i in range(n) if i not in set(anchor)]
    block = max_views - len(anchor)
    count = -(-len(rest) // block)
    size = -(-len(rest) // count)           # even blocks
    return [anchor + rest[k * size:(k + 1) * size] for k in range(count)]


def umeyama(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None
            ) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted least-squares similarity: (s, R, t) minimising sum w |s R src + t - dst|^2."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    a, b = src - mu_s, dst - mu_d
    cov = (b * w[:, None]).T @ a
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var = float(w @ np.sum(a * a, axis=1))
    s = float(np.trace(np.diag(D) @ S) / max(var, 1e-12))
    t = mu_d - s * R @ mu_s
    return s, R, t


def robust_sim3(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None, *,
                iterations: int = 8) -> dict:
    """`umeyama` with Huber reweighting (threshold 1.5 x the median residual each round): the
    Sim(3) taking `src` points onto `dst`, robust to a minority of bad correspondences."""
    base = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64)
    w = base.copy()
    for _ in range(iterations):
        s, R, t = umeyama(src, dst, w)
        r = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        k = max(1.5 * float(np.median(r)), 1e-6)
        w = base * np.where(r <= k, 1.0, k / np.maximum(r, 1e-12))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t
    return {"scale": s, "R": R, "t": t, "T": T,
            "median_residual": float(np.median(r)), "points": int(len(src))}


def apply_sim3_to_pose(T: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """A camera-to-world pose moved by the similarity `T` (4x4 with s R in its 3x3): the
    rotation stays orthonormal, the centre moves with the scale."""
    s = np.cbrt(np.linalg.det(T[:3, :3]))
    out = np.eye(4)
    out[:3, :3] = (T[:3, :3] / s) @ c2w[:3, :3]
    out[:3, 3] = T[:3, :3] @ c2w[:3, 3] + T[:3, 3]
    return out


def _pack(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


# --------------------------------------------------------------------------------------------
# GPU side
# --------------------------------------------------------------------------------------------

def _load_model():
    global _model
    if _model is None:
        from mapanything.models import MapAnything

        _model = MapAnything.from_pretrained(str(WEIGHTS_DIR)).to("cuda").eval()
    return _model


def _decode(frames: list) -> tuple[list, list[dict], tuple[int, int]]:
    from mapanything.utils.cropping import crop_resize_if_necessary
    from mapanything.utils.image import find_closest_aspect_ratio
    from PIL import Image

    pils = []
    for k, data in enumerate(frames):
        with Image.open(io.BytesIO(data)) as im:
            pils.append(im.convert("RGB"))
        if pils[-1].size != pils[0].size:
            raise ValueError(f"frame {k} is {pils[-1].size}, frame 0 {pils[0].size}: one size")
    W, H = pils[0].size
    target = tuple(int(v) for v in find_closest_aspect_ratio(W / H, RESOLUTION_SET))
    prep = prep_geometry((W, H), target)
    out = []
    for im in pils:
        small = crop_resize_if_necessary(im, resolution=target)[0]
        if small.size != target:
            raise RuntimeError(f"crop gave {small.size}, expected {target}")
        out.append(small)
    return out, [dict(prep) for _ in out], target


def _infer(model, images: list, K=None) -> list[dict]:
    """One forward pass; CPU numpy per view."""
    import torch
    from mapanything.utils.image import preprocess_inputs

    views = []
    for im in images:
        v = {"img": im}
        if K is not None:
            v["intrinsics"] = torch.tensor(K, dtype=torch.float32)
        views.append(v)
    views = preprocess_inputs(views)
    with torch.inference_mode():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True,
                            amp_dtype="bf16", apply_mask=True, mask_edges=True)
    out = []
    for p in preds:
        mask = p["mask"][0, ..., 0].bool()
        out.append({
            "c2w": p["camera_poses"][0].double().cpu().numpy(),
            "K": p["intrinsics"][0].double().cpu().numpy(),
            "depth": torch.where(mask, p["depth_z"][0, ..., 0], 0).float().cpu().numpy(),
            "conf": p["conf"][0].float().cpu().numpy(),
            "pts3d": p["pts3d"][0].float().cpu().numpy(),
            "mask": mask.cpu().numpy(),
            "scale": float(p["metric_scaling_factor"].reshape(-1)[0]),
        })
    del preds
    torch.cuda.empty_cache()
    return out


def _chain(ref: dict[int, dict], cur: dict[int, dict], anchors: list[int],
           rng: np.random.Generator, max_points: int = 400_000) -> dict:
    """The Sim(3) taking pass `cur` onto pass `ref`, from the anchors' shared pixels."""
    src, dst, w = [], [], []
    for a in anchors:
        r, c = ref[a], cur[a]
        ok = r["mask"] & c["mask"]
        if ok.sum() < 50:
            continue
        thr_r = np.median(r["conf"][ok])
        thr_c = np.median(c["conf"][ok])
        ok &= (r["conf"] >= thr_r) & (c["conf"] >= thr_c)
        src.append(c["pts3d"][ok])
        dst.append(r["pts3d"][ok])
        w.append(np.minimum(r["conf"][ok], c["conf"][ok]))
    if not src:
        raise RuntimeError("no confident anchor pixels shared between passes")
    src, dst, w = np.concatenate(src), np.concatenate(dst), np.concatenate(w)
    if len(src) > max_points:
        pick = rng.choice(len(src), max_points, replace=False)
        src, dst, w = src[pick], dst[pick], w[pick]
    return robust_sim3(src, dst, w)


@app.function(
    name=FUNCTION_NAME,
    image=image,
    gpu=GPUS,
    timeout=TIMEOUT_SECONDS,
    scaledown_window=60,
    max_containers=2,
    memory=65536,
    cpu=8,
)
def reconstruct(frames: list, options: dict | None = None) -> dict:
    import torch

    t0 = time.perf_counter()
    opts = parse_options(options)
    if len(frames) > MAX_FRAMES:
        raise ValueError(f"{len(frames)} frames in one call, over {MAX_FRAMES}")
    model = _load_model()
    seconds = {"load": time.perf_counter() - t0}
    gpu = torch.cuda.get_device_name(0)
    if not frames:
        return {"frames": 0, "gpu": gpu, "seconds": seconds}
    torch.cuda.reset_peak_memory_stats()

    t = time.perf_counter()
    images, prep, target = _decode(frames)
    n = len(images)
    seconds["decode"] = time.perf_counter() - t

    K_shared = None
    passes_doc = []
    if opts["intrinsics"] is not None:
        K_shared = image_K_to_model(np.asarray(opts["intrinsics"]), prep[0])
    elif opts["shared_intrinsics"] and n > 2:
        t = time.perf_counter()
        pick = sorted({int(round(v)) for v in np.linspace(0, n - 1, min(opts["calib_views"], n))})
        calib = _infer(model, [images[i] for i in pick])
        Ks = np.stack([c["K"] for c in calib])
        K_shared = np.array([[np.median(Ks[:, 0, 0]), 0.0, np.median(Ks[:, 0, 2])],
                             [0.0, np.median(Ks[:, 1, 1]), np.median(Ks[:, 1, 2])],
                             [0.0, 0.0, 1.0]])
        spread = float(np.std(Ks[:, 0, 0]) / max(np.median(Ks[:, 0, 0]), 1e-9))
        seconds["calibration"] = time.perf_counter() - t
        passes_doc.append({"kind": "calibration", "views": len(pick),
                           "seconds": round(seconds["calibration"], 2),
                           "fx_relative_spread": round(spread, 4)})
        del calib

    plan = plan_passes(n, opts["max_views"], opts["anchors"])
    anchors = plan[0][:opts["anchors"]] if len(plan) > 1 else []
    rng = np.random.default_rng(0)
    results: dict[int, dict] = {}
    ref: dict[int, dict] | None = None
    scales = []
    t_inf = time.perf_counter()
    for k, idx in enumerate(plan):
        t = time.perf_counter()
        preds = dict(zip(idx, _infer(model, [images[i] for i in idx], K_shared)))
        doc = {"kind": "reconstruct", "views": len(idx), "seconds": 0.0,
               "metric_scaling": round(float(np.median([p["scale"] for p in preds.values()])), 5)}
        scales.append(doc["metric_scaling"])
        if k == 0:
            ref = preds
        else:
            sim = _chain(ref, preds, anchors, rng)
            doc["sim3_to_pass0"] = {"scale": round(sim["scale"], 6),
                                    "median_residual_m": round(sim["median_residual"], 4),
                                    "points": sim["points"]}
            for i, p in preds.items():
                p["c2w"] = apply_sim3_to_pose(sim["T"], p["c2w"])
                p["depth"] = p["depth"] * sim["scale"]
        for i, p in preds.items():
            if i not in results:                      # anchors keep pass 0's
                results[i] = {key: p[key] for key in ("c2w", "K", "depth", "conf")}
        doc["seconds"] = round(time.perf_counter() - t, 2)
        passes_doc.append(doc)
    seconds["inference"] = time.perf_counter() - t_inf

    order = list(range(n))
    arrays = _pack(
        c2w=np.stack([results[i]["c2w"] for i in order]).astype(np.float64),
        K=np.stack([results[i]["K"] for i in order]).astype(np.float64),
        depth=np.stack([results[i]["depth"] for i in order]).astype(np.float16),
        conf_log=np.stack([np.log(np.maximum(results[i]["conf"] - 1.0, 1e-30))
                           for i in order]).astype(np.float16),
    )
    seconds["total"] = time.perf_counter() - t0
    return {
        "frames": n, "arrays": arrays, "model_size": list(target), "prep": prep,
        "passes": passes_doc, "intrinsics_shared": None if K_shared is None else K_shared.tolist(),
        "intrinsics_given": opts["intrinsics"] is not None,
        "metric_scaling": scales, "convention": CONVENTION, "gpu": gpu,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "model": f"{WEIGHTS_REPO}@{WEIGHTS_REVISION}", "code": MAPANYTHING_COMMIT,
        "options": opts, "seconds": {k: round(v, 2) for k, v in seconds.items()},
    }
