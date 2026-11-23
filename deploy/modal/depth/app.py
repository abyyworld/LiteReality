"""Dense metric depth for a scan's photos: Prompt Depth Anything (PromptDA) on one Modal GPU.

    uv run modal deploy --env main deploy/modal/depth/app.py          # the `refine` function
    uv run litereality splatroom <scan> --depth promptda              # uses it (cached per scan)

Its own app, `litereality-depth`, never included into `litereality` (deploy/modal/app.py), so
deploying or redeploying it never touches the pipeline's models. Standalone: it imports nothing
from this repo.

PromptDA (Lin et al., CVPR 2025; code and weights Apache-2.0) takes one RGB photo and the
phone's own 256 x 192 ARKit LiDAR map as a "prompt" and returns metric depth at the photo's
resolution. Run exactly as its README's example does (`PromptDA.from_pretrained(...)`,
`model.predict(image, prompt_depth)`):

    image          RGB float in 0..1, 1 x 3 x H x W, its long side shrunk to at most 1008 px and
                   both sides rounded down to a multiple of 14 (`load_image`, cv2 INTER_AREA)
    prompt_depth   the LiDAR map in metres as written (uint16 mm / 1000), 1 x 1 x 192 x 256, not
                   resized or masked: the model normalises it by its own min / max
    returns        metres, 1 x 1 x H x W at the image's size

`refine(frames, options)`:

    frames   [{"name": str, "image": JPEG bytes, "depth": PNG bytes (uint16 mm, the LiDAR map),
               "conf": PNG bytes or None, "size": [W, H] the photo as captured (optional)}]
    options  {"width": int (default 960)}
    returns  [{"name", "depth": PNG bytes, uint16 millimetres at width x round(width x H / W)
               (960 x 720 for a 4:3 scan; 0 = invalid), "ms": GPU milliseconds for that frame}]

W x H is the item's "size" when given, else the JPEG's own size. A client may send the photo
shrunk to PromptDA's input (sides rounded down to 14 px): 1920 x 1080 becomes 1008 x 560, whose
aspect would give 960 x 533, not the 960 x 540 the captured photo has.

A frame that cannot be decoded is left out of the reply (the client keeps its LiDAR). "conf" is
accepted but unused: PromptDA was trained on raw ARKit depth, and the client's agreement check
(`rgbd.dense_mask`) uses the confidence instead. `refine([])` only boots a container: sent early,
it hides the cold start.

The weights (1.36 GB, HF depth-anything/prompt-depth-anything-vitl) and the code (github
DepthAnything/PromptDA) are baked into the image at pinned revisions, the weights checked by
SHA-256, and the image build loads the model and runs one CPU forward pass, so a broken pin fails
the deploy rather than the first scan.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import modal

APP_NAME = "litereality-depth"           # NOT included into the `litereality` app
FUNCTION_NAME = "refine"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
PROMPTDA_REPO = "https://github.com/DepthAnything/PromptDA"
# main as of 2026-01-29 ("update promptda++": README and paper only; the code is unchanged since
# 3f770c7, 2025-02-03, "Resolve path to dinov2").
PROMPTDA_COMMIT = "9161547ca7151e14fcc8f490ad3cc02fa9437483"
PROMPTDA_DIR = Path("/opt/PromptDA")
WEIGHTS_REPO = "depth-anything/prompt-depth-anything-vitl"
WEIGHTS_REVISION = "573ab70c94c27d96f0bc70f81f670d2b0a62d571"
WEIGHTS_SHA256 = "4b3a3e0a60ea68e719cad3b2e22da8c95ce78f9559ee04824c064a2a06b0f10f"
WEIGHTS_DIR = Path("/opt/weights")
WEIGHTS = WEIGHTS_DIR / "model.ckpt"
# Any of these (first free wins): the 340M ViT-L fits all three, and an L4 is the cheapest.
GPUS = ["L4", "A10G", "L40S"]
MAX_CONTAINERS = 4
TIMEOUT_SECONDS = 600
MAX_FRAMES = 64                          # per call; the client sends 16
MAX_SIDE, PATCH = 1008, 14               # PromptDA's README `load_image` defaults
# Photos per forward pass. The vendored DINOv2 runs plain attention without xFormers (~2 GB of
# attention per 1008 x 756 photo at fp32), so 4 fit a 24 GB L4; an out-of-memory retries one by one.
MICRO_BATCH = 4
DEFAULT_WIDTH = 960

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url=TORCH_INDEX)
    # PromptDA's own pins where it has them (numpy, opencv); the rest is what its inference path
    # imports (promptda.promptda, model/, utils/logger, the vendored DINOv2 hub).
    .pip_install("numpy==1.26.4", "opencv-python-headless==4.9.0.80", "Pillow==10.4.0",
                 "huggingface_hub==0.26.5", "termcolor==2.5.0")
    .env({"PYTHONPATH": str(PROMPTDA_DIR), "HF_HUB_DISABLE_TELEMETRY": "1"})
    .run_commands(
        # `pip install -e .` (the README) installs nothing: its setup.py looks for packages under
        # src/, which the repo does not have. The module is used from the checkout (PYTHONPATH),
        # which is also where it finds its vendored DINOv2 (torchhub/).
        f"git init -q {PROMPTDA_DIR} && cd {PROMPTDA_DIR}"
        f" && git fetch -q --depth 1 {PROMPTDA_REPO} {PROMPTDA_COMMIT}"
        " && git checkout -q FETCH_HEAD",
        f"python -c \"from huggingface_hub import hf_hub_download;"
        f" hf_hub_download('{WEIGHTS_REPO}', 'model.ckpt', revision='{WEIGHTS_REVISION}',"
        f" local_dir='{WEIGHTS_DIR}')\" && rm -rf {WEIGHTS_DIR}/.cache /root/.cache/huggingface",
        f"echo '{WEIGHTS_SHA256}  {WEIGHTS}' | sha256sum -c -",
        # The model builds from the pinned code and checkpoint (strict state-dict load) and one
        # forward pass has the README's shapes: 1 x 1 x H x W metres at the image's size.
        "python -c \"import torch; from promptda.promptda import PromptDA;"
        f" m = PromptDA.from_pretrained('{WEIGHTS}').eval();"
        " d = m.predict(torch.rand(1, 3, 56, 70), torch.rand(1, 1, 12, 16) + 1.0);"
        " assert tuple(d.shape) == (1, 1, 56, 70), d.shape; print('promptda ok', float(d.mean()))\"",
    )
)
app = modal.App(APP_NAME)

_model = None


def _load_model():
    global _model
    if _model is None:
        from promptda.promptda import PromptDA

        _model = PromptDA.from_pretrained(str(WEIGHTS)).to("cuda").eval()
    return _model


def _width(options: dict | None) -> int:
    options = options or {}
    unknown = set(options) - {"width"}
    if unknown:
        raise ValueError(f"unknown options {sorted(unknown)}")
    width = int(options.get("width", DEFAULT_WIDTH))
    if not 64 <= width <= 4096:
        raise ValueError(f"width {width} is outside 64..4096")
    return width


def _ensure_multiple_of(x: float, multiple_of: int = PATCH) -> int:
    return int(x // multiple_of * multiple_of)


def _decode(item: dict):
    """One frame -> (photo as H x W x 3 float 0..1 at PromptDA's input size, LiDAR metres, the
    photo's captured (width, height): its "size", else the JPEG's)."""
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(item["image"])) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(io.BytesIO(item["depth"])) as im:
        lidar = np.asarray(im).astype(np.float32) / 1000.0
    if lidar.ndim != 2 or not np.isfinite(lidar).all() or lidar.max() <= 0:
        raise ValueError(f"depth map {lidar.shape} is not a LiDAR map")
    h, w = rgb.shape[:2]
    # The README's `load_image`: shrink so the long side is at most 1008, both sides a multiple
    # of 14. A photo already within 1008 (the client sends 1008 x 756) is only rounded to 14:
    # DINOv2's patch embedding refuses any other size.
    scale = min(1.0, (MAX_SIDE // PATCH * PATCH) / max(h, w))
    size = (_ensure_multiple_of(w * scale), _ensure_multiple_of(h * scale))
    if size != (w, h):
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    captured = item.get("size")
    if captured is not None:
        cw, ch = (int(v) for v in captured)
        if cw <= 0 or ch <= 0:
            raise ValueError(f"size {captured} is not a photo's")
        w, h = cw, ch
    return rgb, lidar, (w, h)


def _png_mm(depth) -> bytes:
    import cv2
    import numpy as np

    mm = np.where(np.isfinite(depth) & (depth > 0), np.round(depth * 1000.0), 0)
    ok, buf = cv2.imencode(".png", np.clip(mm, 0, 65535).astype(np.uint16),
                           [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def _predict(model, rgbs: list, lidars: list):
    """PromptDA on photos of one size and LiDAR maps of one size: B x H x W metres (CPU)."""
    import numpy as np
    import torch

    image = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).to("cuda")
    prompt = torch.from_numpy(np.stack(lidars))[:, None].to("cuda")
    return model.predict(image, prompt)[:, 0]


@app.function(
    name=FUNCTION_NAME,
    image=image,
    gpu=GPUS,
    timeout=TIMEOUT_SECONDS,
    scaledown_window=60,          # a scan's batches arrive together; nothing to keep warm after
    max_containers=MAX_CONTAINERS,
    memory=16384,
    cpu=4,
)
def refine(frames: list, options: dict | None = None) -> list:
    import torch
    import torch.nn.functional as F

    width = _width(options)
    if len(frames) > MAX_FRAMES:
        raise ValueError(f"{len(frames)} frames in one call, over {MAX_FRAMES}")
    model = _load_model()
    if not frames:
        return []                                # a warm-up call

    decoded = []
    for item in frames:
        try:
            decoded.append((item["name"], *_decode(item)))
        except Exception as exc:  # noqa: BLE001 — one bad frame never sinks the batch
            print(f"skipping {item.get('name') if isinstance(item, dict) else item!r}: {exc}")
    # Same-size photos (and LiDAR maps) share a forward pass.
    groups: dict[tuple, list] = {}
    for entry in decoded:
        groups.setdefault((entry[1].shape, entry[2].shape), []).append(entry)

    out = []
    for group in groups.values():
        step = MICRO_BATCH
        i = 0
        while i < len(group):
            chunk = group[i:i + step]
            t0 = time.perf_counter()
            try:
                depth = _predict(model, [c[1] for c in chunk], [c[2] for c in chunk])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if step == 1:
                    raise
                step = 1
                continue
            for (name, _rgb, _lidar, (w, h)), d in zip(chunk, depth):
                size = (int(round(width * h / w)), width)
                d = F.interpolate(d[None, None], size=size, mode="bilinear",
                                  align_corners=False)[0, 0]
                out.append({"name": name, "depth": _png_mm(d.float().cpu().numpy()), "ms": 0.0})
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / len(chunk)
            for entry in out[-len(chunk):]:
                entry["ms"] = round(ms, 1)
            i += len(chunk)
    return out
