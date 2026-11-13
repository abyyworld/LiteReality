"""Photoreal rooms: gsplat's own trainer on one Modal GPU, a scan in, splat snapshots out as it trains.

    uv run modal deploy --env main deploy/modal/splat/app.py                # the `train` function
    uv run modal run --env main deploy/modal/splat/app.py::trainer_help     # its options (CPU)
    uv run litereality splatroom <scan>                                     # train one room

Its own app, `litereality-splat`, never included into `litereality` (deploy/modal/app.py), so
deploying or redeploying it never touches the pipeline's models.

`train(dataset_tar, options)` is a GENERATOR (call it with ``.remote_gen``). The dataset is the
tar `litereality_agent.models.splat.dataset.build(...).tar()` writes: a COLMAP model of the scan
in the room's frame, made from ARKit's poses and the LiDAR (no COLMAP run). It runs gsplat
1.5.3's `examples/simple_trainer.py` unchanged (MCMC densification, LiDAR depth loss, a bilateral
grid per photo for the phone's exposure changes, antialiased rasterisation, the world NOT
renormalised so the splats stay in metres in the room's frame) and yields, in order:

    {"kind": "started", "seconds", "images", "gpu"}
    {"kind": "snapshot", "step", "seconds", "splats", "spz"}     # at each of options["snapshots"]
    {"kind": "done", "seconds", "stats"}                          # held-out PSNR / SSIM / LPIPS
    {"kind": "error", "error", "log"}                             # instead of "done"

Each snapshot is an SPZ file (models/splat/spz.py) that Spark draws as it is. The call shows the
instant LiDAR room until the first one (step 2000, ~80 s after the request for tea_room), then
sharpens with each. tea_room, 7000 steps on an H100: 146 s of training, ~$0.30.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "litereality-splat"           # NOT included into the `litereality` app
FUNCTION_NAME = "train"
CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
GSPLAT_VERSION = "1.5.3"
# The prebuilt wheel (PyPI's compiles the CUDA kernels at first import, on the GPU's clock).
GSPLAT_INDEX = "https://docs.gsplat.studio/whl/pt24cu124"
GSPLAT_REPO = "https://github.com/nerfstudio-project/gsplat"
GSPLAT_DIR = Path("/opt/gsplat")
TRAINER = GSPLAT_DIR / "examples" / "simple_trainer.py"
GPU = "H100"
MAX_CONTAINERS = 2
TIMEOUT_SECONDS = 1800
MAX_DATASET_BYTES = 400 * 1024 * 1024

DEFAULTS = {
    "steps": 7000,
    # Earlier snapshots are sparser than the instant LiDAR room the call already shows (tea_room:
    # step 300 is speckle, 1000 has holes), so the first one sent is 2000 (~50 s of training).
    "snapshots": [2000, 3500, 5000, 7000],
    "cap_max": 800_000,           # MCMC's splat budget: ~16 MB of SPZ at SH degree 1
    "sh_degree": 3,               # trained with; snapshots keep degree 1 (export_sh_degree)
    "export_sh_degree": 1,
    "test_every": 8,              # every 8th photo held out for PSNR/SSIM/LPIPS (paper numbers)
    "depth_lambda": 0.01,
    "bilateral_grid": True,
    "antialiased": True,
}

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.10")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",   # A100, A10, L4, H100: builds without a GPU
        "FORCE_CUDA": "1",
        "MAX_JOBS": "8",
        "PIP_NO_CACHE_DIR": "1",
    })
    # clang: fused-ssim and fused-bilagrid (the trainer's CUDA extensions) build with clang++.
    .apt_install("git", "build-essential", "clang", "ninja-build", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url=TORCH_INDEX)
    .pip_install("wheel==0.45.1", "setuptools==75.8.0")       # builds without isolation
    .pip_install(f"gsplat=={GSPLAT_VERSION}+pt24cu124", extra_index_url=GSPLAT_INDEX)
    # The trainer and its requirements at the same release (fused-ssim and fused-bilagrid are
    # CUDA extensions: built here against the torch above, hence no build isolation).
    # On a GPU: fused-ssim's build picks its CUDA arch from the device it sees and ignores
    # TORCH_CUDA_ARCH_LIST, so built without one it has no kernel the H100 can run.
    .run_commands(
        f"git clone -q --depth 1 --branch v{GSPLAT_VERSION} {GSPLAT_REPO} {GSPLAT_DIR}",
        f"cd {GSPLAT_DIR}/examples && python -m pip install --no-build-isolation -r requirements.txt",
        "python -c \"import torch, gsplat; assert torch.__version__.startswith('2.4.1');"
        f" assert gsplat.__version__.startswith('{GSPLAT_VERSION}'), gsplat.__version__\"",
        "python -c \"import torch; from fused_ssim import fused_ssim;"
        " x = torch.rand(1, 3, 64, 64, device='cuda'); print('fused_ssim', float(fused_ssim(x, x)))\"",
        # LPIPS (the held-out metric): its AlexNet weights, baked in rather than fetched per run.
        "python -c \"from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as L;"
        " L(net_type='alex')\"",
        gpu=GPU,
    )
    .add_local_python_source("litereality_agent")
)
app = modal.App(APP_NAME)


def _options(given: dict | None) -> dict:
    out = dict(DEFAULTS)
    for key, value in (given or {}).items():
        if key not in DEFAULTS:
            raise ValueError(f"unknown option {key!r}")
        out[key] = value
    out["steps"] = int(out["steps"])
    out["snapshots"] = sorted({int(s) for s in out["snapshots"] if 0 < int(s) <= out["steps"]}
                              | {out["steps"]})
    return out


def _command(data_dir: Path, result_dir: Path, opt: dict) -> list[str]:
    # tyro's flags, hyphenated (`trainer_help` prints them).
    cmd = [sys.executable, str(TRAINER), "mcmc",
           "--data-dir", str(data_dir), "--data-factor", "1", "--result-dir", str(result_dir),
           "--no-normalize-world-space", "--disable-viewer", "--disable-video",
           "--max-steps", str(opt["steps"]),
           "--save-ply", "--ply-steps", *map(str, opt["snapshots"]),
           "--save-steps", str(opt["steps"]),
           "--eval-steps", str(opt["steps"]),
           "--test-every", str(opt["test_every"]),
           "--sh-degree", str(opt["sh_degree"]),
           "--strategy.cap-max", str(int(opt["cap_max"])),
           "--depth-loss", "--depth-lambda", str(opt["depth_lambda"])]
    if opt["bilateral_grid"]:
        cmd.append("--use-bilateral-grid")
    if opt["antialiased"]:
        cmd.append("--antialiased")
    return cmd


def _settled(path: Path) -> bool:
    """A ply the trainer has finished writing: its size holds still for half a second."""
    try:
        size = path.stat().st_size
        time.sleep(0.5)
        return size > 0 and path.stat().st_size == size
    except OSError:
        return False


@app.function(image=image, timeout=120)
def trainer_help() -> str:
    """The trainer's own `--help` for the mcmc config (checks the flags `_command` uses)."""
    out = subprocess.run([sys.executable, str(TRAINER), "mcmc", "--help"], capture_output=True,
                         text=True, cwd=TRAINER.parent)
    print(out.stdout[-20000:] or out.stderr[-5000:])
    return out.stdout


@app.function(
    name=FUNCTION_NAME,
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    scaledown_window=60,          # nothing to keep warm: each room trains once
    max_containers=MAX_CONTAINERS,
    memory=32768,
    cpu=8,
)
def train(dataset_tar: bytes, options: dict | None = None):
    from litereality_agent.models.splat import ply, spz

    t0 = time.perf_counter()
    try:
        opt = _options(options)
        if len(dataset_tar) > MAX_DATASET_BYTES:
            raise ValueError(f"dataset is {len(dataset_tar) / 1e6:.0f} MB, over the limit")
    except (ValueError, TypeError) as exc:
        yield {"kind": "error", "error": str(exc), "log": ""}
        return
    work = Path(tempfile.mkdtemp(prefix="splat-"))
    data_dir, result_dir = work / "data", work / "result"
    log_path = work / "trainer.log"
    try:
        with tarfile.open(fileobj=io.BytesIO(dataset_tar)) as tf:
            for member in tf.getmembers():
                if not member.isfile() or member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"unexpected entry in the dataset: {member.name}")
            tf.extractall(data_dir)
        images = len(list((data_dir / "images").glob("*")))
        import torch
        yield {"kind": "started", "seconds": time.perf_counter() - t0, "images": images,
               "gpu": torch.cuda.get_device_name(0)}

        with open(log_path, "w") as log:
            proc = subprocess.Popen(_command(data_dir, result_dir, opt), cwd=TRAINER.parent,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            pending = list(opt["snapshots"])
            while pending:
                path = result_dir / "ply" / f"point_cloud_{pending[0] - 1}.ply"
                if not path.exists():
                    path = result_dir / "ply" / f"point_cloud_{pending[0]}.ply"
                if path.exists() and (_settled(path) or proc.poll() is not None):
                    step = pending.pop(0)
                    splats = ply.read(path.read_bytes(), max_sh_degree=opt["export_sh_degree"])
                    data = spz.encode(splats, antialiased=bool(opt["antialiased"]))
                    yield {"kind": "snapshot", "step": step, "seconds": time.perf_counter() - t0,
                           "splats": len(splats), "spz": data}
                    path.unlink(missing_ok=True)
                    continue
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
            code = proc.wait()
        log_tail = log_path.read_text(errors="replace")[-6000:]
        if code != 0 or pending:
            yield {"kind": "error", "error": f"trainer exited with {code}"
                   + (f"; no snapshot for steps {pending}" if pending else ""), "log": log_tail}
            return
        stats = {}
        for f in sorted((result_dir / "stats").glob("val_step*.json")):
            stats = json.loads(f.read_text())
        yield {"kind": "done", "seconds": time.perf_counter() - t0, "stats": stats,
               "log": log_tail[-1500:]}
    except Exception as exc:  # noqa: BLE001 — reported to the caller, never retried
        tail = log_path.read_text(errors="replace")[-6000:] if log_path.exists() else ""
        yield {"kind": "error", "error": f"{type(exc).__name__}: {exc}", "log": tail}
    finally:
        shutil.rmtree(work, ignore_errors=True)
