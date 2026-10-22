"""Full-body people for the room call: LHM++ on one Modal GPU, photos in, a skinned splat body out.

    uv run modal run --env main deploy/modal/avatar/app.py::populate   # the weights, once (CPU)
    uv run modal deploy --env main deploy/modal/avatar/app.py          # the `avatar` function
    uv run modal run --env main deploy/modal/avatar/app.py --images me.jpg --deployed  # try it

Its own app, `litereality-avatar`, never included into `litereality` (deploy/modal/app.py): the
room server looks it up by name (room_ops/roomcall/avatars.py) and builds its stand-in body
instead whenever this app is not deployed, times out or answers with an error, so deploying it
is optional and redeploying it never touches the pipeline's models.

The request is ``{"images_b64": [JPEG or PNG, 1..8], "mode": "turnaround" | "photo", "id"}`` and
the response ``{"avatar_b64", "seconds", "id", "stats"}``: one LRAV v1 file
(room_ops/roomcall/avatar_format.py) of up to 120k splats in the SMPL-X canonical T-pose with
per-splat skinning weights, which every browser in the call animates with Spark. A request that
cannot be served gets ``{"error", "id"}``; `handle` never raises. After a fault inside the model
the container stops taking inputs (its CUDA context may be broken) and the next request gets a
fresh one; a model that fails to LOAD is not retried for LOAD_RETRY_SECONDS, so requests in that
window fail in milliseconds instead of each rebuilding it on the GPU.

The model is LHM++ (github.com/aigc3d/LHM-plusplus) `LHMPP-700M-SMPLX-FREE`: any number of
pose-free photos, body shape predicted from them, ~1-2 s of GPU time per body (authors' figures),
8 GB of VRAM, so one L4 is plenty. Code Apache-2.0; the WEIGHTS are CC-BY-NC-4.0 and the SMPL-X
body model they need is non-commercial: fine for research, not for a product.

Weights follow TRELLIS's pattern (deploy/modal/trellis/app.py): the CPU function `populate`
downloads them into the `litereality-avatar-models` Volume once (Hugging Face, ModelScope as the
fallback), one container at a time; GPU containers read them offline, keep LHM++'s
`pretrained_models` link on their own disk, and never write the Volume, so two can run at once.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import secrets
import sys
import time
import traceback
from pathlib import Path

import modal
import numpy as np

from litereality_agent.models.avatar import lhm
from litereality_agent.models.avatar.lhm import MODES
from litereality_agent.room_ops.roomcall import avatar_format

APP_NAME = "litereality-avatar"          # NOT included into the `litereality` app
FUNCTION_NAME = "avatar"
POPULATE_FUNCTION = "avatar_populate"
MODEL_VOLUME = "litereality-avatar-models"
WEIGHTS_ROOT = Path("/models")
LHMPP_REPO = "https://github.com/aigc3d/LHM-plusplus"
LHMPP_COMMIT = "906b5d9fb967ab42efb92f6fa55bf22cac86b653"
LHMPP_DIR = Path("/opt/lhmpp")
MODEL_ID = "3DAIGC/LHMPP-700M-SMPLX-FREE"      # Hugging Face; ModelScope fallbacks below
PRIOR_ID = "3DAIGC/LHMPP-Prior"
MS_MODEL_ID = "Damo_XR_Lab/LHMPP-700M-SMPLX-FREE"
MS_PRIOR_ID = "Damo_XR_Lab/LHMPP-Prior"
MODEL_REVISION = PRIOR_REVISION = None         # unknown from here: populate logs the resolved
                                               # commit; pin it after the first real run
GPU = "L4"
MAX_CONTAINERS = 2
SCALEDOWN_SECONDS = 300
TIMEOUT_SECONDS = 600
LOAD_RETRY_SECONDS = 180                 # a failed model load is not retried for this long
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 4 * 1024 * 1024
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
JPEG_MAGIC, PNG_MAGIC = b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"

# On the Volume. PRIOR_DIR is LHM++'s "pretrained_models" bundle (SMPL-X, FLAME, the diffused
# skinning grid, its sample points, BiRefNet, ArcFace), MODEL_DIR the 700M checkpoint.
PRIOR_DIR = WEIGHTS_ROOT / "prior"
MODEL_DIR = WEIGHTS_ROOT / "LHMPP-700M-SMPLX-FREE"
POPULATED_MARKER = WEIGHTS_ROOT / ".lhmpp-populated"
# Every prior file LHM++'s inference opens by name (relative to ./pretrained_models) at the pinned
# commit for this model: populate refuses to finish without them, so a bundle laid out differently
# fails there with a listing, not at the first avatar with a cryptic model-load error.
# FLAME_NEUTRAL.pkl: with 10 betas and 100 expressions LHM++ grafts FLAME's expression space onto
# SMPL-X (base_skinning._get_expr_from_flame -> smplx.create(path, "flame", gender="neutral")).
REQUIRED_PRIOR = (
    "human_model_files/smplx/SMPLX_NEUTRAL.npz",
    "human_model_files/smplx/SMPLX_MALE.npz",
    "human_model_files/smplx/SMPLX_FEMALE.npz",
    "human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy",
    "human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl",
    "human_model_files/flame/FLAME_NEUTRAL.pkl",
    "human_model_files/flame/2019/generic_model.pkl",
    "voxel_grid/cano_1_volume.npz",
    "voxel_grid/human_prior_constrain.npz",
    "dense_sample_points/1_160000.ply",
    "arcface_resnet18.pth",
    lhm.BIREFNET_WEIGHTS,
)

# ── image ────────────────────────────────────────────────────────────────────────────────────────
# LHM++'s INSTALL.md recipe at LHMPP_COMMIT, made explicit and pinned. Nothing here was built from
# this machine (no Modal or Hugging Face access): the first `modal deploy` is its first build.
# No layer attaches a GPU: CUDA extensions compile for TORCH_CUDA_ARCH_LIST with FORCE_CUDA=1.
CUDA_IMAGE = "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
TORCH_HOME = "/opt/torch_home"
# Prebuilt CUDA wheels from the projects' GitHub releases, pinned by digest (pip checks the
# #sha256 fragment). The digests were computed from the release files on 2026-10-22.
GSPLAT_WHEEL = (
    "https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/"
    "gsplat-1.4.0%2Bpt23cu121-cp310-cp310-linux_x86_64.whl"
    "#sha256=40d4f9e16f97e2836d8b68fef483a37fdd72e3ccd708e9e0f226d3501a3da648"
)
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
    "flash_attn-2.6.3%2Bcu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    "#sha256=c50f5de67a8b75bcfcf4a34257622475f9b56d59da58a539476a21db9d5b9867"
)
PYG_INDEX = "https://data.pyg.org/whl/torch-2.3.0+cu121.html"
PYTORCH3D_INDEX = (
    "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html"
)
PYTORCH3D_COMMIT = "89653419d0973396f3eff1a381ba09a07fffc2ed"   # tag v0.7.7 (supports torch 2.3)
DGR_REPO = "https://github.com/ashawkey/diff-gaussian-rasterization.git"
DGR_COMMIT = "8829d14f814fccdaf840b7b0f3021a616583c0a1"         # main, `git ls-remote` 2026-10-22
DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
DINOV2_FILE = f"{TORCH_HOME}/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth"
# LHM++'s requirements.txt at LHMPP_COMMIT, filtered: without dna, gradio, spaces and rembg (the
# demo app and video tooling), the non-headless opencv_python, torch/torchvision/xformers (step 3),
# gsplat (step 5, a prebuilt wheel: PyPI's is a source build), chumpy (step 6b) and modelscope
# (step 11); every entry the file leaves unpinned is pinned here to a release checked on PyPI to
# exist for CPython 3.10 and to accept numpy 1.23.
LHMPP_REQUIREMENTS = (
    "einops==0.8.0", "roma==1.5.1", "accelerate==0.34.2", "smplx==0.1.28", "decord==0.6.0",
    "diffusers==0.32.0", "huggingface_hub==0.23.2", "imageio==2.34.1", "imageio-ffmpeg==0.5.1",
    "jaxtyping==0.2.38", "kiui==0.2.14", "kornia==0.7.2", "loguru==0.7.3", "lpips==0.1.4",
    "matplotlib==3.5.3", "megfile==4.1.0.post2", "numpy==1.23.0", "omegaconf==2.3.0",
    "open3d==0.19.0", "opencv-python-headless==4.10.0.84", "Pillow==10.4.0", "plyfile==1.0.3",
    "pygltflib==1.16.2", "pyrender==0.1.45", "PyYAML==6.0.1", "requests==2.32.3", "scipy==1.11.4",
    "scikit-image==0.22.0", "setuptools==74.0.0", "taming-transformers-rom1504==0.0.6",
    "timm==1.0.15", "tqdm==4.66.4", "transformers==4.41.2", "trimesh==4.4.9",
    "typeguard==2.13.3", "xatlas==0.0.9", "ffmpegio==0.10.0.post0",
)

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.10")
    # 1. Build environment. FORCE_CUDA + an explicit arch list let pointops, pytorch3d and
    #    diff-gaussian-rasterization compile without a GPU (A10/A100-class 8.6, L4 8.9, H100 9.0).
    #    TORCH_HOME is where torch.hub looks for the DINOv2 weights baked in at step 10.
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.6;8.9;9.0",
        "FORCE_CUDA": "1",
        "TORCH_HOME": TORCH_HOME,
        "MAX_JOBS": "4",
        "PIP_NO_CACHE_DIR": "1",
    })
    # 2. OS packages: git for the clones, a compiler toolchain and ninja for the CUDA extensions,
    #    the GL/X libraries opencv, open3d and pyrender load at import, ffmpeg for decord/imageio.
    .apt_install(
        "git", "curl", "ca-certificates", "build-essential", "ninja-build", "libgl1",
        "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "libgomp1", "ffmpeg",
    )
    # 3. PyTorch 2.3.0 for CUDA 12.1 and the matching xformers, from PyTorch's own index.
    .pip_install("torch==2.3.0", "torchvision==0.18.0", "torchaudio==2.3.0", index_url=TORCH_INDEX)
    .pip_install("xformers==0.0.26.post1", index_url=TORCH_INDEX)
    # 4. LHM++ itself at the pinned commit (its only submodule is an internal GitLab mirror that
    #    inference does not use, so no --recursive).
    .run_commands(
        f"git init -q {LHMPP_DIR} && cd {LHMPP_DIR} && git remote add origin {LHMPP_REPO}"
        f" && git fetch -q --depth 1 origin {LHMPP_COMMIT} && git checkout -q --detach FETCH_HEAD"
        f' && test "$(git rev-parse HEAD)" = {LHMPP_COMMIT}'
    )
    # 5. gsplat 1.4.0, prebuilt for torch 2.3 + cu121 (PyPI only has the source, which compiles
    #    its CUDA kernels at first import, on the GPU's clock).
    .run_commands(f"python -m pip install '{GSPLAT_WHEEL}'")
    # 6. LHM++'s Python requirements (filtered and pinned, see LHMPP_REQUIREMENTS); torch is
    #    restated so pip can never swap it for another build while resolving.
    .pip_install(*LHMPP_REQUIREMENTS, "torch==2.3.0", "torchvision==0.18.0",
                 extra_index_url=TORCH_INDEX)
    # 6b. chumpy: needed to unpickle FLAME's generic_model.pkl; its setup.py imports pip, so it
    #     cannot build in an isolated environment.
    .run_commands("python -m pip install --no-build-isolation chumpy==0.70")
    # 7. LHM++'s own CUDA op (lib/pointops -> the `pointops_cuda` extension).
    .run_commands(f"cd {LHMPP_DIR}/lib/pointops && python -m pip install --no-build-isolation .")
    # 8. Sparse conv + scatter for the point transformer, pytorch3d (the prebuilt wheel LHM++'s
    #    INSTALL.md uses, or the same release built from source if that index lacks it), and
    #    diff-gaussian-rasterization (imported by the renderer) at a pinned commit.
    .pip_install("spconv-cu121==2.3.8")
    .pip_install("torch-scatter==2.1.2+pt23cu121", find_links=PYG_INDEX)
    # pytorch3d's own requirements first, from PyPI: the --no-index install below cannot fetch them.
    .pip_install("fvcore==0.1.5.post20221221", "iopath==0.1.10")
    .run_commands(
        f"python -m pip install --no-index pytorch3d==0.7.7 -f {PYTORCH3D_INDEX}"
        " || python -m pip install --no-build-isolation"
        f" 'git+https://github.com/facebookresearch/pytorch3d.git@{PYTORCH3D_COMMIT}'"
    )
    .run_commands(f"python -m pip install --no-build-isolation 'git+{DGR_REPO}@{DGR_COMMIT}'")
    # 9. flash-attn 2.6.3 (the point transformer's attention; it falls back without it, slower).
    .run_commands(f"python -m pip install '{FLASH_ATTN_WHEEL}'")
    # 10. DINOv2 ViT-L/14 + 4 registers, LHM++'s image encoder: torch.hub would download it at
    #     the first model load, from a GPU container; baked in here instead and checked.
    .run_commands(
        f"mkdir -p {TORCH_HOME}/hub/checkpoints",
        f"curl -fsSL --retry 5 -o {DINOV2_FILE} {DINOV2_URL}",
        f"python -c \"import torch; sd = torch.load('{DINOV2_FILE}', map_location='cpu');"
        " assert tuple(sd['register_tokens'].shape) == (1, 4, 1024), 'not ViT-L/14 reg4'\"",
    )
    # 11. ModelScope, the weights' second source (used by `populate` only).
    .pip_install("modelscope==1.20.1")
    # The pins above must have survived every later step.
    .run_commands(
        "python -c \"import numpy, torch; assert numpy.__version__ == '1.23.0', numpy.__version__;"
        " assert torch.__version__.startswith('2.3.0'), torch.__version__\""
    )
    # 12. Our code: avatar_format (the file) and models.avatar.lhm (the inference).
    .add_local_python_source("litereality_agent")
)
weights = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)
_RUNNER = None
_LOAD_FAILURE: tuple[float, str] | None = None   # (time.monotonic(), "Type: message") of the last
                                                 # failed load in this container


# ── requests ─────────────────────────────────────────────────────────────────────────────────────


def _b64_chars(size: int) -> int:
    return 4 * ((size + 2) // 3)


def parse_request(request: dict) -> dict:
    """The request, checked: ``{"id", "mode", "images": [bytes, ...]}``. ValueError says what is
    wrong. Unknown keys are ignored so a newer room server can talk to this function, and the
    checks run before any GPU work so a malformed request never loads the model."""
    if not isinstance(request, dict):
        raise ValueError("the request must be a dict")
    rid = request.get("id")
    if not isinstance(rid, str) or not ID_PATTERN.fullmatch(rid):
        raise ValueError("id must be 1-64 of [A-Za-z0-9_-]")
    mode = request.get("mode")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    encoded = request.get("images_b64")
    if not isinstance(encoded, list) or not 1 <= len(encoded) <= MAX_IMAGES:
        raise ValueError(f"images_b64 must be a list of 1..{MAX_IMAGES} images")
    images = []
    for i, text in enumerate(encoded):
        if not isinstance(text, str) or not text:
            raise ValueError(f"image {i} is not a base64 string")
        if len(text) > _b64_chars(MAX_IMAGE_BYTES):
            raise ValueError(f"image {i} is over {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
        try:
            data = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError(f"image {i} is not valid base64") from None
        if not 1 <= len(data) <= MAX_IMAGE_BYTES:
            raise ValueError(f"image {i} must be 1 byte to {MAX_IMAGE_BYTES} bytes")
        if not (data.startswith(JPEG_MAGIC) or data.startswith(PNG_MAGIC)):
            raise ValueError(f"image {i} is not a JPEG or PNG")
        images.append(data)
    return {"id": rid, "mode": mode, "images": images}


def file_skin(joints: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The skin as browsers read it back from the file: avatar_format's 8-bit top-4 weights
    (`lhm.build`'s skin_codec, so the reported skin error is of what is shipped)."""
    packed = avatar_format.quantize_skin(joints, weights).astype(np.int64)
    return packed >> 8, (packed & 0xFF) / 255.0


def to_avatar(fields: dict, stats: dict) -> avatar_format.Avatar:
    """`lhm.build`'s numpy fields -> the room call's Avatar (skin quantised to 8-bit weights)."""
    return avatar_format.Avatar(
        positions=fields["positions"], scales=fields["scales"], rotations=fields["rotations"],
        colors=fields["colors"],
        skin=avatar_format.quantize_skin(fields["skin_joints"], fields["skin_weights"]),
        joint_names=list(fields["joint_names"]), parents=list(fields["parents"]),
        rest=fields["rest"], floor_y=float(fields["floor_y"]), kind="lhm", meta=dict(stats),
    )


def handle(request: dict, runner) -> dict:
    """One request -> one response dict; never raises, so the caller always gets an answer.

    `runner` is anything with `segment` and `run` (LHMRunner, or a test's fake), or a zero-argument
    callable returning one: the Modal function passes its lazy loader, so a bad request is refused
    before the model loads, and a failed load is an error reply rather than a crashed call.

    - a bad request: ``{"error": "bad request: ...", "id"}``;
    - photos that cannot make a body (`lhm.PhotoError`: no person in them, an undecodable image;
      or a body the file would refuse, AvatarFormatError): ``{"error": "<why>", "id"}``;
    - anything else is a fault, other ValueErrors included (a model or library error is not the
      photos' fault): logged here, ``{"error": "internal: <type>", "id"}``."""
    started = time.monotonic()
    given = request.get("id") if isinstance(request, dict) else None
    rid = given[:64] if isinstance(given, str) else None
    try:
        req = parse_request(request)
    except ValueError as exc:
        return {"error": f"bad request: {exc}", "id": rid}
    try:
        if callable(runner) and not hasattr(runner, "run"):
            runner = runner()
        fields, stats = lhm.build(runner, req["images"], req["mode"], skin_codec=file_skin)
        data = avatar_format.pack(to_avatar(fields, stats))
    except (lhm.PhotoError, avatar_format.AvatarFormatError) as exc:
        return {"error": str(exc), "id": rid}
    except Exception as exc:  # noqa: BLE001 - every fault becomes a reply, see the docstring
        traceback.print_exc()
        return {"error": f"internal: {type(exc).__name__}", "id": rid}
    return {"avatar_b64": base64.b64encode(data).decode("ascii"),
            "seconds": round(time.monotonic() - started, 2), "id": req["id"], "stats": stats}


# ── weights ──────────────────────────────────────────────────────────────────────────────────────


def _model_dir(root: Path) -> Path | None:
    """The checkpoint folder under `root` (config.json + model.safetensors), or None. Like LHM++'s
    own `_get_max_step_folder`: flat for LHM++, the newest step_* folder for an LHM-style tree.
    Exactly those two files: huggingface_hub 0.23's PyTorchModelHubMixin.from_pretrained opens
    only `model.safetensors` from a local folder (no pytorch_model.bin, no shards)."""
    def complete(path: Path) -> bool:
        return (path / "config.json").is_file() and (path / "model.safetensors").is_file()

    root = Path(root)
    if not root.is_dir():
        return None
    if complete(root):
        return root
    steps = [p for p in root.glob("step_*") if p.is_dir() and complete(p)]
    if not steps:
        return None

    def step(path: Path) -> int:
        tail = path.name.split("_", 1)[1]
        return int(tail) if tail.isdigit() else -1

    return max(steps, key=step)


def _missing_prior(root: Path) -> list[str]:
    return [name for name in REQUIRED_PRIOR if not (Path(root) / name).exists()]


def _populated() -> bool:
    return (POPULATED_MARKER.is_file() and not _missing_prior(PRIOR_DIR)
            and _model_dir(MODEL_DIR) is not None)


def _download(hf_id: str, ms_id: str, dest: Path, revision: str | None) -> str:
    """One repo into `dest` (plain files, no cache symlinks): Hugging Face at `revision`, else
    ModelScope (LHM++ publishes both). Returns where it came from, with the commit when known."""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import HfApi, snapshot_download

        sha = HfApi().model_info(hf_id, revision=revision).sha
        snapshot_download(hf_id, revision=sha, local_dir=str(dest))
        return f"hf:{hf_id}@{sha}"
    except Exception as exc:  # noqa: BLE001 - any Hub failure: try the mirror
        print(f"[avatar] Hugging Face {hf_id} failed ({type(exc).__name__}: {exc}); "
              f"trying ModelScope {ms_id}")
    from modelscope import snapshot_download as modelscope_download

    modelscope_download(ms_id, local_dir=str(dest))
    return f"modelscope:{ms_id}@master"


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
    """Download LHM++'s prior bundle and the 700M SMPLX-FREE weights into the Volume once (CPU,
    idempotent), check every file inference opens is there, write the marker and commit."""
    weights.reload()
    if _populated() and not force:
        return {"status": "present", "revisions": json.loads(POPULATED_MARKER.read_text())}
    started = time.monotonic()
    revisions = {"prior": _download(PRIOR_ID, MS_PRIOR_ID, PRIOR_DIR, PRIOR_REVISION),
                 "model": _download(MODEL_ID, MS_MODEL_ID, MODEL_DIR, MODEL_REVISION)}
    missing = _missing_prior(PRIOR_DIR)
    if missing:
        present = sorted(p.name for p in PRIOR_DIR.iterdir())
        raise RuntimeError(f"the prior bundle lacks {missing}; it holds {present}")
    if _model_dir(MODEL_DIR) is None:
        found = sorted(p.name for p in MODEL_DIR.iterdir()) if MODEL_DIR.is_dir() else []
        raise RuntimeError(f"no config.json + model.safetensors under {MODEL_DIR}; "
                           f"it holds {found}")
    POPULATED_MARKER.write_text(json.dumps(revisions, indent=1) + "\n")
    weights.commit()
    print(f"[avatar] weights ready: {revisions} (pin MODEL_REVISION / PRIOR_REVISION to these)")
    return {"status": "downloaded", "revisions": revisions,
            "seconds": round(time.monotonic() - started, 1)}


# ── the GPU function ─────────────────────────────────────────────────────────────────────────────


def _prepare_container() -> None:
    """Make this container ready to load LHM++ without the network and without writing the Volume.

    Weights that are not there yet are populated first by the CPU `populate` (one container at a
    time, so two GPU containers starting together download once) and the Volume reloaded. LHM++
    opens its prior files as ./pretrained_models/..., so the checkout's `pretrained_models` is a
    link to the Volume's bundle made on THIS container's disk, and the checkout is the working
    directory and first on sys.path (it imports itself as `core`, `engine`, `lib`)."""
    os.environ["HF_HUB_OFFLINE"] = "1"          # before anything imports huggingface_hub
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not _populated():
        print("[avatar] LHM++ weights not on the Volume yet; populating")
        populate.remote()
        weights.reload()
        if not _populated():
            raise RuntimeError("the LHM++ weights are still missing after populate")
    link = LHMPP_DIR / "pretrained_models"
    if link.is_symlink() and not link.exists():
        link.unlink()
    if not link.exists():
        link.symlink_to(PRIOR_DIR, target_is_directory=True)
    os.chdir(LHMPP_DIR)
    if str(LHMPP_DIR) not in sys.path:
        sys.path.insert(0, str(LHMPP_DIR))


class ModelLoadError(RuntimeError):
    """LHM++ did not load in this container, now or less than LOAD_RETRY_SECONDS ago."""


def _runner():
    """LHM++ and BiRefNet, loaded once per container (a warm container answers in seconds).

    A failed load is remembered: for LOAD_RETRY_SECONDS every call raises ModelLoadError at once
    instead of repeating the populate check and the whole model build on the GPU (most load
    failures, a missing prior file or a broken install, fail the same way every time), then the
    next call tries again."""
    global _RUNNER, _LOAD_FAILURE
    if _RUNNER is not None:
        return _RUNNER
    if _LOAD_FAILURE is not None:
        age = time.monotonic() - _LOAD_FAILURE[0]
        if age < LOAD_RETRY_SECONDS:
            raise ModelLoadError(f"{_LOAD_FAILURE[1]} ({age:.0f} s ago; retried after "
                                 f"{LOAD_RETRY_SECONDS} s)")
    try:
        _prepare_container()
        _RUNNER = lhm.LHMRunner(LHMPP_DIR, _model_dir(MODEL_DIR))
    except Exception as exc:
        _LOAD_FAILURE = (time.monotonic(), f"{type(exc).__name__}: {exc}"[:500])
        raise ModelLoadError(_LOAD_FAILURE[1]) from exc
    _LOAD_FAILURE = None
    return _RUNNER


def _retire_container() -> None:
    """After a fault inside a loaded model: drop it and take no more inputs here. A device-side
    assert or an illegal memory access leaves the process's CUDA context unusable, and a warm
    container that only answers errors would keep winning Modal's routing (each error renewing
    its idle window), so everyone would get stand-ins until it idled out. Modal lets this input
    finish, the container exits, and the next input starts a fresh one."""
    global _RUNNER
    _RUNNER = None
    try:
        import modal.experimental

        modal.experimental.stop_fetching_inputs()
    except Exception as exc:  # noqa: BLE001 - not in a Modal container (tests), or an older client
        print(f"[avatar] could not retire this container ({type(exc).__name__}: {exc})")


@app.function(
    name=FUNCTION_NAME,
    image=image,
    gpu=GPU,
    volumes={str(WEIGHTS_ROOT): weights},
    timeout=TIMEOUT_SECONDS,
    # An idle container stays this long: people in one call join minutes apart, and a warm
    # container skips the 1-2 min cold start. At most MAX_CONTAINERS x this of L4 time per burst.
    scaledown_window=SCALEDOWN_SECONDS,
    # Two at once is safe (no GPU container writes the Volume) and caps the bill.
    max_containers=MAX_CONTAINERS,
    # A failure is a bad photo, an OOM or a timeout: a retry re-bills the GPU for the same failure,
    # and the room server falls back to its stand-in body anyway.
    retries=0,
    memory=32768,
)
def avatar(request: dict) -> dict:
    """Photos of one person -> one LRAV body (see the module docstring for the contract)."""
    reply = handle(request, _runner)
    # A fault with the model loaded may have broken the GPU state: retire. A failed load leaves
    # _RUNNER None; that container stays and answers fast (see _runner).
    if _RUNNER is not None and str(reply.get("error", "")).startswith("internal:"):
        print("[avatar] a fault with the model loaded: retiring this container")
        _retire_container()
    return reply


# ── trying it from a laptop ──────────────────────────────────────────────────────────────────────


def _photo(path: Path) -> bytes:
    """A local photo as the room's browser would send it: upright, at most 1600 px, JPEG, no EXIF
    (so phone photos of any size work, and no GPS leaves the laptop)."""
    import io

    from PIL import Image, ImageOps

    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    rgb.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    rgb.save(out, format="JPEG", quality=88)
    return out.getvalue()


@app.local_entrypoint()
def main(images: str, mode: str = "photo", out: str = "avatar.lrav",
         deployed: bool = False) -> None:
    """`modal run deploy/modal/avatar/app.py --images a.jpg,b.jpg [--mode turnaround]
    [--out avatar.lrav] [--deployed]`: build one body, check it unpacks, write the file and print
    its stats. Without --deployed it runs this file as a temporary app (same image and Volume);
    with it, it calls the deployed `litereality-avatar` exactly as the room server does."""
    paths = [Path(p.strip()).expanduser() for p in images.split(",") if p.strip()]
    request = {"images_b64": [base64.b64encode(_photo(p)).decode("ascii") for p in paths],
               "mode": mode, "id": "try-" + secrets.token_hex(4)}
    fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME) if deployed else avatar
    result = fn.remote(request)
    if "error" in result:
        raise SystemExit(f"avatar failed: {result['error']}")
    data = base64.b64decode(result["avatar_b64"], validate=True)
    body = avatar_format.unpack(data)
    Path(out).write_bytes(data)
    print(json.dumps({"file": out, "bytes": len(data), "splats": body.count, "kind": body.kind,
                      "joints": len(body.joint_names), "head_splats": int(body.head_mask().sum()),
                      "seconds": result["seconds"], "stats": result["stats"]}, indent=1))
