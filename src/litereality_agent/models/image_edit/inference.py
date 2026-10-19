"""Canonical Qwen-Image-Edit-2509 inference for reference images.

Runs INSIDE the Modal container (``deploy/modal/image_edit/app.py``): the evidence sheet and the
reference prompt come in, one PNG render comes out. It is the open alternative to the OpenAI and
Gemini image APIs — ``Qwen/Qwen-Image-Edit-2509`` is Apache-2.0 and ungated, so neither a paid
image key nor a Hugging Face token is needed, only GPU time on the caller's own Modal workspace.

The module must stay importable on a machine with no torch or diffusers: the client imports it for
the request contract (``parse_request`` and the defaults), so every heavy import is inside the
function that needs it.
"""

from __future__ import annotations

import base64
import binascii
import gc
import io
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2509"
# The model card's own settings: 40 steps with true CFG at 4.0. Steps are the cost knob — each one
# is a full pass of the 20B transformer (two with true CFG) — so they are the first thing to lower.
DEFAULT_STEPS = 40
DEFAULT_TRUE_CFG = 4.0
# The model card's own value. The pipeline runs true CFG only when true_cfg_scale > 1 and a
# negative_prompt is passed (it checks "is not None", so "" would do the same); " " just matches the
# reference settings without steering away from anything.
NEGATIVE_PROMPT = " "

MAX_STEPS = 100
MIN_TRUE_CFG = 1.0
MAX_TRUE_CFG = 20.0
# torch.Generator.manual_seed takes a 64-bit value; staying non-negative and below 2**63 keeps the
# seed meaningful on every torch build and in JSON.
MAX_SEED = 2**63 - 1

# Loaded pipelines by model id. A warm container serves every later request from here, which is the
# whole point of keeping it up for a scaledown window: loading is minutes, an edit is about one.
_PIPELINES: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class EditRequest:
    """One validated edit: the decoded input image plus the knobs the pipeline is called with."""

    image: bytes
    prompt: str
    model: str = DEFAULT_MODEL
    steps: int = DEFAULT_STEPS
    true_cfg_scale: float = DEFAULT_TRUE_CFG
    # Only the contract's default. The reference step draws a fresh seed for every call (see
    # image_gen._call_modal), since a fixed one makes every forced regeneration identical.
    seed: int = 0


def _is_int(value: Any) -> bool:
    # bool is an int subclass; `steps=True` is a caller bug, not one step.
    return isinstance(value, int) and not isinstance(value, bool)


def parse_request(request: Any) -> EditRequest:
    """Validate the JSON-safe transport payload, raising ValueError with what is wrong.

    Kept pure (no torch, no PIL) so it runs on the client before a request is sent — a malformed
    request then fails locally instead of waking an H100 to reject it — and again in the container,
    which is the contract boundary.
    """
    if not isinstance(request, dict):
        raise ValueError(f"image_edit request must be a dict, got {type(request).__name__}")

    encoded = request.get("image_b64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image_edit request needs image_b64 (the base64-encoded input image)")
    try:
        image = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"image_edit image_b64 is not valid base64: {exc}") from exc
    if not image:
        raise ValueError("image_edit image_b64 decodes to an empty image")

    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("image_edit request needs a non-empty prompt")

    model = request.get("model")
    if model is None:
        model = DEFAULT_MODEL
    if not isinstance(model, str) or not model.strip():
        raise ValueError("image_edit model must be a non-empty Hugging Face model id")

    steps = request.get("steps", DEFAULT_STEPS)
    if not _is_int(steps) or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"image_edit steps must be an int in 1..{MAX_STEPS}, got {steps!r}")

    cfg = request.get("true_cfg_scale", DEFAULT_TRUE_CFG)
    if (
        isinstance(cfg, bool)
        or not isinstance(cfg, (int, float))
        or not math.isfinite(cfg)
        or not MIN_TRUE_CFG <= cfg <= MAX_TRUE_CFG
    ):
        raise ValueError(
            f"image_edit true_cfg_scale must be a number in [{MIN_TRUE_CFG:g}, {MAX_TRUE_CFG:g}], "
            f"got {cfg!r}"
        )

    seed = request.get("seed", 0)
    if not _is_int(seed) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"image_edit seed must be a non-negative int, got {seed!r}")

    return EditRequest(
        image=image,
        prompt=prompt,
        model=model.strip(),
        steps=steps,
        true_cfg_scale=float(cfg),
        seed=seed,
    )


def load_pipeline(model_id: str = DEFAULT_MODEL):
    """Return the bf16 pipeline for ``model_id`` on CUDA, loading it once per container.

    ``HF_HOME`` points at the Modal Volume, so the first load in a fresh workspace downloads ~58 GB
    (the 20B transformer plus the Qwen2.5-VL-7B text encoder) and every later container reads it
    from there.
    """
    cached = _PIPELINES.get(model_id)
    if cached is not None:
        return cached

    import torch
    from diffusers import QwenImageEditPlusPipeline

    if _PIPELINES:
        # One pipeline at a time: two ~58 GB bf16 pipelines do not fit in one 80 GB H100, so a
        # request for a different model id evicts the resident one instead of running out of VRAM.
        _PIPELINES.clear()
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[image_edit] loading {model_id} (bf16, downloads to $HF_HOME on first run)", flush=True)
    started = time.monotonic()
    pipeline = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipeline.to("cuda")
    # A tqdm bar per edit is pure noise in the Modal logs; the handler reports the timing instead.
    pipeline.set_progress_bar_config(disable=True)
    print(f"[image_edit] {model_id} loaded in {time.monotonic() - started:.1f}s", flush=True)
    _PIPELINES[model_id] = pipeline
    return pipeline


def edit(
    pipeline,
    image_bytes: bytes,
    prompt: str,
    *,
    steps: int = DEFAULT_STEPS,
    true_cfg_scale: float = DEFAULT_TRUE_CFG,
    seed: int = 0,
) -> bytes:
    """Run one edit of ``image_bytes`` under ``prompt`` and return the render as PNG bytes.

    The output size follows the input's aspect ratio at ~1 MP, so the 1024x1024 evidence sheet comes
    back as a 1024x1024 render, which is what ``normalize_reference`` expects.
    """
    import torch
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    options: dict[str, Any] = {}
    if true_cfg_scale > 1:
        # Passing the negative prompt is what turns true CFG on. At a scale of 1 the pipeline would
        # ignore it and only log a warning, so leave it out and run the one-pass path.
        options["negative_prompt"] = NEGATIVE_PROMPT
    result = pipeline(
        image=[image],
        prompt=prompt,
        true_cfg_scale=true_cfg_scale,
        num_inference_steps=steps,
        # The model card passes 1.0. 2509 is not guidance-distilled, so the pipeline ignores it
        # (with a log line); it is kept so the call matches the reference settings exactly.
        guidance_scale=1.0,
        num_images_per_prompt=1,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        **options,
    )
    buffer = io.BytesIO()
    result.images[0].save(buffer, format="PNG")
    return buffer.getvalue()


def hosted_handler(
    request: dict, *, loader: Callable[[str], Any] | None = None
) -> dict:
    """Serve one hosted request: validate, load (or reuse) the pipeline, edit, encode.

    ``loader`` lets the Modal wrapper commit its weights Volume after the first load; it defaults
    to ``load_pipeline``. Bad input raises ValueError, which Modal returns to the client as the
    call's exception, so a failure is never mistaken for an image.

    ``elapsed_s`` covers the whole handler, model load included, because that is the GPU time the
    request was billed for; ``load_s`` says how much of it was the load (about 0 when warm).
    """
    started = time.monotonic()
    parsed = parse_request(request)
    load = loader or load_pipeline
    pipeline = load(parsed.model)
    loaded = time.monotonic()
    png = edit(
        pipeline,
        parsed.image,
        parsed.prompt,
        steps=parsed.steps,
        true_cfg_scale=parsed.true_cfg_scale,
        seed=parsed.seed,
    )
    finished = time.monotonic()
    return {
        "png_b64": base64.b64encode(png).decode("ascii"),
        "model": parsed.model,
        "steps": parsed.steps,
        "true_cfg_scale": parsed.true_cfg_scale,
        "seed": parsed.seed,
        "load_s": round(loaded - started, 3),
        "elapsed_s": round(finished - started, 3),
    }
