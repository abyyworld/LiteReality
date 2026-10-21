"""Modal deployment wrapper for the canonical TRELLIS.2 implementation.

Deploys into whichever workspace the active ``MODAL_PROFILE`` names; the weights are ungated, so
no Hugging Face account or secret is required. See README.md.

Up to `MAX_CONTAINERS` GPU containers serve requests at once (the room call's objects stage sends
every object of a room together). That is safe because no GPU container ever WRITES the shared
weights Volume:

* the weights are downloaded once, by `populate` — a CPU function, one container at a time,
  idempotent — on the first request that finds them missing (or by hand, see README.md);
* a GPU container loads them with Hugging Face offline (``local_files_only``) and assembles its
  model folder (the patched pipeline.json + a ``ckpts`` link into the snapshot) under /tmp, never
  on the Volume, and commits nothing;
* flex_gemm's autotune cache is per container in /tmp, seeded from the Volume's copy if one exists.

The request is ``{"image_b64", "seed", "simplify", "texture_size", "pipeline_type", "id"?}`` and
the response ``{"glb_b64", "id"?, "seconds"}``, or ``{"error", "id"?}`` for a request that cannot
be served (`parse_request`); ``models/trellis/modal.py`` is the client.

Keeping containers warm: ``scaledown_window`` keeps an idle container `SCALEDOWN_SECONDS` after its
last request, which covers the gap between one object's preview and its final pass and a re-run
right after. To have a container already up when a call starts (no cold start, ~1-2 min), raise
the minimum for the call and put it back afterwards — an override lasts until the next deploy,
and a container kept up costs H100 time whether or not anyone uses it:

    import modal
    f = modal.Function.from_name("litereality", "trellis")
    f.update_autoscaler(min_containers=1)       # at call start
    f.update_autoscaler(min_containers=0)       # when the call is over
"""

from __future__ import annotations

import base64
import binascii
import os
import shutil
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "litereality-trellis"  # included into `litereality` by deploy/modal/app.py
FUNCTION_NAME = "trellis"
POPULATE_FUNCTION = "trellis_populate"
MODEL_VOLUME = "litereality-trellis-models"
REMOTE_REPO_ROOT = Path("/root")
WEIGHTS_ROOT = Path("/models")
MODEL_ID = "microsoft/TRELLIS.2-4B"   # models.trellis.inference.DEFAULT_MODEL
# Written by `populate` once the snapshot and the Volume's model folder are complete.
POPULATED_MARKER = WEIGHTS_ROOT / ".trellis2-4b-populated"
# A GPU container's own model folder and flex_gemm cache: container disk, never the Volume.
LOCAL_MODEL_DIR = Path("/tmp/trellis2_4b_local")
FLEX_GEMM_CACHE = Path("/tmp/flex_gemm/autotune_cache.json")
VOLUME_FLEX_GEMM_CACHE = WEIGHTS_ROOT / ".flex_gemm" / "autotune_cache.json"

# Four H100s at most: a room's objects in parallel without an open-ended bill.
MAX_CONTAINERS = 4
# Idle seconds before a container stops. Each object's final is queued the moment its preview is
# back, so a busy room never leaves a container idle in between; this is the tail after the last
# request (at most MAX_CONTAINERS x this of H100 time, ~$0.5 at list price) and lets a re-run
# within two minutes skip the cold start.
SCALEDOWN_SECONDS = 120
PIPELINE_TYPES = ("512", "1024", "1024_cascade", "1536_cascade")
TEXTURE_SIZES = (256, 4096)
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_ID_CHARS = 128

# TRELLIS.2's image encoder is the gated ``facebook/dinov3-vitl16-pretrain-lvd1689m``. The DINOv3
# licence grants the right to redistribute the trained weights provided the licence travels with
# them, so we bake an ungated mirror into the image instead of requiring every deployer to hold an
# HF account. Pinned by commit and verified by digest: a mirror owner can force-push or delete the
# repo, and four independent uploads of these weights agree on this digest.
DINOV3_DIR = Path("/opt/dinov3")
DINOV3_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
DINOV3_SHA256 = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"

deployment_root = Path(__file__).resolve().parent
image = (
    modal.Image.from_dockerfile(
        deployment_root / "Dockerfile",
        context_dir=deployment_root,
    )
    .run_commands("python3.10 -m pip install --no-cache-dir 'setuptools>=64' wheel")
    .run_commands(
        "bash -lc 'cd /root/TRELLIS.2 && "
        ". ./setup.sh --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec'",
        # Upstream checks nvidia-smi before installing its native extensions. Scope the
        # billable builder GPU to this cached layer instead of the complete Docker build.
        gpu="H100",
        env={"TORCH_CUDA_ARCH_LIST": "9.0"},
    )
    # TRELLIS.2 reaches into DINOv3 internals that changed in Transformers 5.x.
    # Keep this after the cached native-extension layer so the compatibility pin
    # remains a CPU-only image operation and cannot trigger another GPU build.
    .pip_install("transformers==4.57.5")
    # CPU-only, and deliberately after the cached GPU layer so pulling 1.2GB of weights can never
    # trigger another H100 build. Fails the build on a digest mismatch rather than at inference.
    .run_commands(
        f"python3.10 -c \"from huggingface_hub import snapshot_download; \
snapshot_download('{DINOV3_REPO}', revision='{DINOV3_REVISION}', local_dir='{DINOV3_DIR}')\"",
        f"echo '{DINOV3_SHA256}  {DINOV3_DIR}/model.safetensors' | sha256sum -c -",
    )
    .env(
        {
            "LR_REPO_ROOT": str(REMOTE_REPO_ROOT),
            "LITEREALITY_WEIGHTS": str(WEIGHTS_ROOT),
            "HF_XET_HIGH_PERFORMANCE": "1",
            # resolve_dinov3() reads this first; with it set, pipeline.json is repointed at the
            # local copy and loading never reaches the gated repo, so no HF token is needed.
            "LITEREALITY_DINOV3": str(DINOV3_DIR),
            # Per container (runtime_env only sets a default): containers sharing one autotune
            # file on the Volume would overwrite each other's.
            "FLEX_GEMM_AUTOTUNE_CACHE_PATH": str(FLEX_GEMM_CACHE),
        }
    )
    .add_local_python_source("litereality_agent")
)
weights = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)
_pipeline = None


# ── requests ─────────────────────────────────────────────────────────────────────────────────────


def parse_request(request: dict) -> dict:
    """The request's values, checked: ValueError says what is wrong with it. Unknown keys are
    ignored so a newer client can talk to this function."""
    if not isinstance(request, dict):
        raise ValueError("the request must be a dict")
    rid = request.get("id")
    if rid is not None and (not isinstance(rid, str) or len(rid) > MAX_ID_CHARS):
        raise ValueError(f"id must be a string of at most {MAX_ID_CHARS} characters")
    raw = request.get("image_b64")
    if not isinstance(raw, str) or not raw:
        raise ValueError("image_b64 is required")
    try:
        image = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("image_b64 is not valid base64") from None
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise ValueError(f"the image must be 1 byte to {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    pipeline_type = request.get("pipeline_type")
    if pipeline_type is not None and pipeline_type not in PIPELINE_TYPES:
        raise ValueError(f"pipeline_type must be one of {', '.join(PIPELINE_TYPES)} or null")
    try:
        seed = int(request.get("seed", 42))
        simplify = float(request.get("simplify", 0.95))
        texture_size = int(request.get("texture_size", 1024))
    except (TypeError, ValueError):
        raise ValueError("seed, simplify and texture_size must be numbers") from None
    lo, hi = TEXTURE_SIZES
    if not lo <= texture_size <= hi:
        raise ValueError(f"texture_size must be {lo}..{hi}")
    return {"id": rid, "image": image, "seed": seed,
            "decimation_target": _decimation_target(simplify), "texture_size": texture_size,
            "pipeline_type": pipeline_type}


def _decimation_target(simplify: float) -> int:
    """Translate the legacy ratio into the canonical launcher's face target."""
    ratio = min(max(float(simplify), 0.0), 1.0)
    return max(10_000, round(1_000_000 * (1.0 - ratio)))


def _with_id(response: dict, rid: str | None) -> dict:
    return {**response, "id": rid} if rid is not None else response


# ── weights ──────────────────────────────────────────────────────────────────────────────────────


def _snapshot(*, local_only: bool) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_ID, local_files_only=local_only))


def _go_offline() -> None:
    """No network for Hugging Face in this process from here on. Called before anything imports
    huggingface_hub or transformers in a GPU container: both read it once, at import."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _seed_flex_gemm_cache() -> None:
    """Start this container's autotune cache from the Volume's, when an earlier deployment left
    one there (read only; this container's own tuning stays in /tmp)."""
    if FLEX_GEMM_CACHE.exists() or not VOLUME_FLEX_GEMM_CACHE.is_file():
        return
    FLEX_GEMM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(VOLUME_FLEX_GEMM_CACHE, FLEX_GEMM_CACHE)
    except OSError:
        pass


def offline_model_dir() -> str:
    """This container's model folder, from the Volume's snapshot without touching the network
    or writing the Volume. Weights that are not there yet are populated first — by the CPU
    `populate` function, which runs one container at a time, so four GPU containers starting at
    once download them once — and the Volume reloaded to see them."""
    _go_offline()
    from litereality_agent.models.trellis.inference import assemble_local_model

    try:
        snap = _snapshot(local_only=True)
    except Exception as exc:  # noqa: BLE001 — LocalEntryNotFoundError and friends: not populated
        print(f"[TRELLIS] weights not on the Volume yet ({type(exc).__name__}); populating")
        populate.remote()
        weights.reload()
        snap = _snapshot(local_only=True)
    _seed_flex_gemm_cache()
    return str(assemble_local_model(snap, LOCAL_MODEL_DIR))


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        from litereality_agent.models.trellis.inference import load_pipeline

        _pipeline = load_pipeline(offline_model_dir())
    return _pipeline


@app.function(
    name=POPULATE_FUNCTION,
    image=image,
    cpu=2.0,
    memory=4096,
    volumes={str(WEIGHTS_ROOT): weights},
    timeout=60 * 60,
    retries=0,
    # One at a time: a second caller waits for the first and then finds the marker.
    max_containers=1,
)
def populate(force: bool = False) -> dict:
    """Download TRELLIS.2's weights into the Volume once (CPU only, idempotent) and commit."""
    from litereality_agent.models.trellis.inference import assemble_local_model

    weights.reload()
    if POPULATED_MARKER.is_file() and not force:
        try:
            return {"status": "present", "snapshot": str(_snapshot(local_only=True))}
        except Exception:  # noqa: BLE001 — a marker without its files: download again
            pass
    started = time.monotonic()
    snap = _snapshot(local_only=False)
    # The Volume's own model folder too, for anything that loads it as before.
    assemble_local_model(snap, WEIGHTS_ROOT / "trellis2_4b_local")
    POPULATED_MARKER.write_text(f"{MODEL_ID} {snap.name}\n")
    weights.commit()
    return {"status": "downloaded", "snapshot": str(snap),
            "seconds": round(time.monotonic() - started, 1)}


@app.function(
    name=FUNCTION_NAME,
    image=image,
    gpu="H100",
    volumes={str(WEIGHTS_ROOT): weights},
    timeout=20 * 60,
    retries=0,
    scaledown_window=SCALEDOWN_SECONDS,
    # Safe above one: no GPU container writes the Volume (see the module docstring).
    max_containers=MAX_CONTAINERS,
)
def generate(request: dict) -> dict:
    """Convert the transport payload into one GLB using model-owned inference code."""
    from litereality_agent.models.trellis.inference import generate as generate_glb

    try:
        req = parse_request(request)
    except ValueError as exc:
        rid = request.get("id") if isinstance(request, dict) else None
        return _with_id({"error": f"bad request: {exc}"}, rid if isinstance(rid, str) else None)
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as temporary_dir:
        temp = Path(temporary_dir)
        image_path = temp / "input.png"
        output_path = temp / "output.glb"
        image_path.write_bytes(req["image"])
        generate_glb(
            _load_pipeline(),
            image_path,
            output_path,
            seed=req["seed"],
            decimation_target=req["decimation_target"],
            texture_size=req["texture_size"],
            pipeline_type=req["pipeline_type"],
        )
        response = {"glb_b64": base64.b64encode(output_path.read_bytes()).decode("ascii"),
                    "seconds": round(time.monotonic() - started, 2)}
    return _with_id(response, req["id"])
