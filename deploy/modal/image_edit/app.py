"""Modal deployment wrapper for Qwen-Image-Edit-2509 reference images.

Serves ``LR_IMAGE_PROVIDER=modal``: the reference step sends the evidence sheet and the prompt
here instead of to OpenAI or Gemini, so it needs no paid image key and spends Modal credit instead.
Inference lives in ``litereality_agent.models.image_edit.inference``; this file only supplies a GPU
and a persistent weight cache. The model is Apache-2.0 and ungated, so no Hugging Face account or
secret is required. See README.md.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "litereality-image-edit"  # included into `litereality` by deploy/modal/app.py
FUNCTION_NAME = "image_edit"
MODEL_VOLUME = "litereality-image-edit-models"
WEIGHTS_ROOT = Path("/models")

# Every dependency is a prebuilt wheel, so unlike TRELLIS this image builds on CPU in minutes and
# never attaches a GPU to the builder. torch 2.8.0's PyPI wheel bundles its CUDA runtime, and
# torchvision 0.23.0 is the release built against it (it pins torch==2.8.0); the Qwen2.5-VL
# processor needs torchvision. diffusers 0.36.0 is the release whose QwenImageEditPlusPipeline this
# code was written against, and transformers is pinned to the same 4.57.5 the TRELLIS image uses.
# huggingface_hub stays below 1.0 because that transformers release requires it.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "diffusers==0.36.0",
        "transformers==4.57.5",
        "accelerate>=1.0,<2",
        "pillow>=10",
        "huggingface_hub[hf_xet]>=0.34,<1.0",
    )
    .env(
        {
            # On the Volume, so the ~58 GB download happens once per workspace, not per container.
            "HF_HOME": str(WEIGHTS_ROOT / "huggingface"),
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_python_source("litereality_agent")
)
weights = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)
_committed: set[str] = set()


def _load_pipeline(model_id: str):
    """Load (or reuse) the pipeline, committing the Volume after the first load in a container.

    The first load in a fresh workspace is what writes the weights into the Volume. Committing right
    then, rather than on container exit, means a container that is later killed or times out still
    leaves the download behind for the next one instead of paying for it again.
    """
    from litereality_agent.models.image_edit.inference import load_pipeline

    pipeline = load_pipeline(model_id)
    if model_id not in _committed:
        weights.commit()
        _committed.add(model_id)
    return pipeline


@app.function(
    name=FUNCTION_NAME,
    image=image,
    # ~58 GB of bf16 weights (the 20B transformer plus the Qwen2.5-VL-7B text encoder) has to sit
    # in VRAM at once; an 80 GB H100 holds it with room for 1024px activations, a 40-48 GB card
    # does not.
    gpu="H100",
    volumes={str(WEIGHTS_ROOT): weights},
    # The first call in a workspace downloads ~58 GB, loads it and only then edits, all inside this
    # one input. A limit is billed only while it runs, so it is generous rather than tight: cutting
    # that first call off would throw the load away and start the next one from the top.
    timeout=60 * 60,
    # A failure here is an OOM, a timeout, or a bad request: retrying any of them re-bills the H100
    # for the same failure. The client makes exactly one attempt for the same reason.
    retries=0,
    # References arrive in bursts (objects, then chairs, then openings) and a cold load reads
    # ~58 GB, so idle a minute between bursts rather than reload; this is the only idle time billed.
    scaledown_window=60,
    # Serialize cold starts so two containers cannot race to populate the Volume, and cap spend on
    # the free tier: concurrent reference calls queue for this one container (queueing is not
    # billed) rather than each loading its own copy of the model.
    max_containers=1,
)
def image_edit(request: dict) -> dict:
    """Edit one evidence sheet into a clean reference render using model-owned inference code."""
    from litereality_agent.models.image_edit.inference import hosted_handler

    return hosted_handler(request, loader=_load_pipeline)
