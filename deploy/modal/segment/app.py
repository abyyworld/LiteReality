"""Object masks for cutting a room splat: SAM 2.1 on one Modal GPU, photos + boxes in, masks out.

    uv run modal deploy --env main deploy/modal/segment/app.py
    (called by `litereality splatroom`, pipeline/splatroom/masks.py)

Its own app, `litereality-segment`, never included into `litereality`.

`segment(request)`: ``{"frames": [{"id": str, "jpeg": bytes, "boxes": {name: [x0, y0, x1, y1]},
"negatives": {name: [[x, y], ...]}, "positives": {name: [[x, y], ...]}}]}`` (negatives optional:
points that are NOT the object, e.g. the table top a tucked chair's box also covers; positives
optional: points that ARE, e.g. the centre of a body already put on its object)
-> ``{"masks": {frame id: {name: PNG (1-bit, the photo's size)}}, "seconds", "gpu"}`` or
``{"error"}``. With ``"multimask": true`` each name maps to SAM 2's three candidate masks instead,
``{"pngs": [PNG, PNG, PNG], "scores": [float, float, float]}``, for the caller to choose from (a
box over a tucked chair also holds the table, and SAM's best guess is often the table). Each box is an object's scanned 3D box projected into that photo; SAM 2's image
predictor turns it into the object's own outline, all of a photo's boxes in one batch on one image
embedding. Model: facebook/sam2.1-hiera-large (Apache-2.0), baked into the image.
"""

from __future__ import annotations

import io
import time

import modal

APP_NAME = "litereality-segment"
FUNCTION_NAME = "segment"
SAM2_REPO = "https://github.com/facebookresearch/sam2"
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MODEL_ID = "facebook/sam2.1-hiera-large"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
# Any of these, cheapest first (an L4 alone sat queued with no container for minutes).
GPU = ["L4", "A10G", "L40S"]
MAX_FRAMES = 400
MAX_BOXES = 64
MAX_JPEG = 4 * 1024 * 1024

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url=TORCH_INDEX)
    .pip_install("huggingface_hub==0.26.5", "Pillow==10.4.0", "numpy==1.26.4")
    .run_commands(
        # The CUDA extension only speeds up mask post-processing; without it SAM 2 is unchanged.
        f"SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation 'git+{SAM2_REPO}@{SAM2_COMMIT}'",
        "python -c \"from sam2.sam2_image_predictor import SAM2ImagePredictor as P;"
        f" P.from_pretrained('{MODEL_ID}', device='cpu')\"",
    )
)
app = modal.App(APP_NAME)
_PREDICTOR = None


def _predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        _PREDICTOR = SAM2ImagePredictor.from_pretrained(MODEL_ID, device="cuda")
    return _PREDICTOR


def _check(request) -> list[dict]:
    frames = request.get("frames") if isinstance(request, dict) else None
    if not isinstance(frames, list) or not 0 < len(frames) <= MAX_FRAMES:
        raise ValueError("frames: a list of 1..%d" % MAX_FRAMES)
    for f in frames:
        if not isinstance(f.get("id"), str) or not isinstance(f.get("jpeg"), bytes) \
                or len(f["jpeg"]) > MAX_JPEG or not isinstance(f.get("boxes"), dict) \
                or len(f["boxes"]) > MAX_BOXES:
            raise ValueError("each frame: {id, jpeg (bytes), boxes {name: [x0, y0, x1, y1]}}")
    return frames


@app.function(name=FUNCTION_NAME, image=image, gpu=GPU, timeout=900, scaledown_window=60,
              max_containers=2)
def segment(request: dict) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    t0 = time.perf_counter()
    try:
        frames = _check(request)
    except (ValueError, TypeError, AttributeError) as exc:
        return {"error": str(exc)}
    predictor = _predictor()
    out: dict[str, dict[str, bytes]] = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for f in frames:
            names = [n for n, b in f["boxes"].items() if isinstance(b, (list, tuple)) and len(b) == 4]
            if not names:
                continue
            rgb = np.asarray(Image.open(io.BytesIO(f["jpeg"])).convert("RGB"))
            predictor.set_image(rgb)
            boxes = np.asarray([f["boxes"][n] for n in names], dtype=np.float32)
            negatives = f.get("negatives") or {}
            positives = f.get("positives") or {}
            prompts = {n: [(p, 1) for p in (positives.get(n) or [])]
                       + [(p, 0) for p in (negatives.get(n) or [])] for n in names}
            most = max((len(v) for v in prompts.values()), default=0)
            points = labels = None
            if most:
                # Padded to the same count per box with label -1 (SAM's "no point").
                points = np.zeros((len(names), most, 2), np.float32)
                labels = np.full((len(names), most), -1, np.int32)
                for i, n in enumerate(names):
                    for k, ((x, y), label) in enumerate(prompts[n][:most]):
                        points[i, k], labels[i, k] = (x, y), label
            many = bool(request.get("multimask"))
            masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels,
                                                 box=boxes, multimask_output=many)
            masks = np.asarray(masks).reshape(len(names), -1, rgb.shape[0], rgb.shape[1])
            scores = np.asarray(scores, dtype=np.float32).reshape(len(names), -1)
            out[f["id"]] = {}
            for name, ms, sc in zip(names, masks, scores):
                pngs = []
                for m in (ms if many else ms[:1]):
                    buf = io.BytesIO()
                    Image.fromarray((m > 0).astype(np.uint8) * 255).convert("1").save(
                        buf, format="PNG", optimize=True)
                    pngs.append(buf.getvalue())
                out[f["id"]][name] = ({"pngs": pngs, "scores": [float(x) for x in sc]}
                                      if many else pngs[0])
    return {"masks": out, "seconds": time.perf_counter() - t0,
            "gpu": torch.cuda.get_device_name(0)}
