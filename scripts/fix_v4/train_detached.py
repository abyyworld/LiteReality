"""The splatroom trainer as a DETACHED Modal job: survives the local client losing its connection.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file .env modal run --detach --env main \
        scripts/fix_v4/train_detached.py --run atrium_v4

The deployed `train()` (deploy/modal/splat/app.py) is a streaming generator: when the local client
drops, the call is cancelled and the GPU job dies with it (results/fix_v4/retrain: three times near
step 1250 of 10000). Here, as in scripts/eyeful/train_eyeful.py, the job reads its input from a
Modal Volume and writes everything it makes back to it, so nothing depends on a held-open stream:

    /<run>/dataset.tar       IN: the tar `splatroom.training_set` builds (uploaded by build_v4.py)
    /<run>/options.json      IN: the trainer options (a preset plus overrides, app.py's `_options`)
    /<run>/progress.json     step, steps, it/s, seconds; rewritten every ~30 s
    /<run>/trainer.log       the trainer's own output (copied every ~30 s)
    /<run>/snap_<step>.spz   each snapshot, as app.py's train() would yield it (export SH degree)
    /<run>/point_cloud.ply   the final splats as the trainer saved them (float32, SH 3)
    /<run>/stats/*.json      gsplat's val_step*.json (held-out PSNR / SSIM / LPIPS)
    /<run>/done.json         last: exit code, stats, timings, splat count, the snapshot steps

The image is app.py's, layer for layer (its cached CUDA builds are reused), plus app.py itself as
`splat_app.py`, so the trainer command and the option checks are app.py's own (`_options`,
`_command`), not a copy. An ephemeral app only: nothing is deployed.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SPLAT = REPO / "deploy" / "modal" / "splat"

# ---- deploy/modal/splat/app.py's image, verbatim (same layers -> same cache) ----------------
CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
GSPLAT_VERSION = "1.5.3"
GSPLAT_INDEX = "https://docs.gsplat.studio/whl/pt24cu124"
GSPLAT_REPO = "https://github.com/nerfstudio-project/gsplat"
GSPLAT_DIR = Path("/opt/gsplat")
TRAINER = GSPLAT_DIR / "examples" / "simple_trainer.py"
PATCH = "/opt/litereality/patch_trainer.py"
GPU = "H100"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.10")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",
        "FORCE_CUDA": "1",
        "MAX_JOBS": "8",
        "PIP_NO_CACHE_DIR": "1",
    })
    .apt_install("git", "build-essential", "clang", "ninja-build", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url=TORCH_INDEX)
    .pip_install("wheel==0.45.1", "setuptools==75.8.0")
    .pip_install(f"gsplat=={GSPLAT_VERSION}+pt24cu124", extra_index_url=GSPLAT_INDEX)
    .run_commands(
        f"git clone -q --depth 1 --branch v{GSPLAT_VERSION} {GSPLAT_REPO} {GSPLAT_DIR}",
        f"cd {GSPLAT_DIR}/examples && python -m pip install --no-build-isolation -r requirements.txt",
        "python -c \"import torch, gsplat; assert torch.__version__.startswith('2.4.1');"
        f" assert gsplat.__version__.startswith('{GSPLAT_VERSION}'), gsplat.__version__\"",
        "python -c \"import torch; from fused_ssim import fused_ssim;"
        " x = torch.rand(1, 3, 64, 64, device='cuda'); print('fused_ssim', float(fused_ssim(x, x)))\"",
        "python -c \"from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as L;"
        " L(net_type='alex')\"",
        gpu=GPU,
    )
    .add_local_file(SPLAT / "litereality_dense.py", str(TRAINER.parent / "litereality_dense.py"),
                    copy=True)
    .add_local_file(SPLAT / "patch_trainer.py", PATCH, copy=True)
    .run_commands(f"python {PATCH} {TRAINER}",
                  f"cd {TRAINER.parent} && python -c 'import litereality_dense'")
    # ---- ours: app.py itself, for its `_options` and `_command` ----
    .add_local_file(SPLAT / "app.py", "/root/splat_app.py")
    .add_local_python_source("litereality_agent")
)

VOLUME_NAME = "litereality-v4-train"
MOUNT = Path("/vol")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("litereality-v4-train")

STEP_RE = re.compile(r"(\d+)/(\d+) \[[^\]]*?([\d.]+)(it/s|s/it)\]")


def _progress(text: str) -> dict:
    """The last tqdm line of the trainer's log: step, steps, rate."""
    last = None
    for m in STEP_RE.finditer(text.replace("\r", "\n")):
        last = m
    if last is None:
        return {}
    rate = float(last.group(3))
    return {"step": int(last.group(1)), "steps": int(last.group(2)),
            "it_s": rate if last.group(4) == "it/s" else (1.0 / rate if rate else 0.0)}


@app.function(image=image, gpu=GPU, volumes={MOUNT: volume}, cpu=8, memory=32768,
              scaledown_window=2, timeout=2700)
def train_job(run: str, max_train_seconds: int = 2100) -> dict:
    sys.path.insert(0, "/root")
    import splat_app  # deploy/modal/splat/app.py, added to the image above
    import torch

    from litereality_agent.models.splat import ply, spz

    t0 = time.perf_counter()
    out = MOUNT / run
    volume.reload()
    options = json.loads((out / "options.json").read_text())
    for stale in ["done.json", "progress.json", "point_cloud.ply", *[
            p.name for p in out.glob("snap_*.spz")]]:
        (out / stale).unlink(missing_ok=True)
    opt = splat_app._options(options)
    work = Path("/tmp/splat")
    data_dir, result_dir = work / "data", work / "result"
    with tarfile.open(fileobj=io.BytesIO((out / "dataset.tar").read_bytes())) as tf:
        for member in tf.getmembers():
            if not member.isfile() or member.name.startswith("/") or ".." in member.name:
                raise ValueError(f"unexpected entry in the dataset: {member.name}")
        tf.extractall(data_dir)
    images = len(list((data_dir / "images").glob("*")))
    depths = len(list((data_dir / "depths").glob("*.png")))
    cmd = splat_app._command(data_dir, result_dir, opt)
    gpu = torch.cuda.get_device_name(0)
    result = {"run": run, "gpu": gpu, "images": images, "depths": depths, "options": opt,
              "command": cmd[1:], "setup_s": round(time.perf_counter() - t0, 1),
              "snapshots": {}}
    print(f"[{run}] {gpu}; {images} photos, {depths} depth maps; {' '.join(cmd[2:])}", flush=True)
    log_path = work / "trainer.log"
    stop = threading.Event()
    lock = threading.Lock()

    def save_progress(final: bool = False) -> None:
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        prog = {**_progress(text), "seconds": round(time.perf_counter() - t0, 1),
                "train_s": round(time.perf_counter() - t1, 1), "gpu": gpu, "images": images,
                "depths": depths, "snapshots": sorted(result["snapshots"]), "final": final,
                "time": time.time()}
        with lock:
            if log_path.exists():
                shutil.copyfile(log_path, out / "trainer.log")
            (out / "progress.json").write_text(json.dumps(prog))
            volume.commit()
        print(f"[{run}] {json.dumps(prog)}", flush=True)

    def heartbeat() -> None:
        while not stop.wait(30):
            try:
                save_progress()
            except Exception as exc:  # noqa: BLE001
                print(f"[{run}] heartbeat: {exc}", flush=True)

    def export(step: int, path: Path) -> None:
        splats = ply.read(path.read_bytes(), max_sh_degree=opt["export_sh_degree"])
        data = spz.encode(splats, antialiased=bool(opt["antialiased"]))
        with lock:
            (out / f"snap_{step}.spz").write_bytes(data)
            volume.commit()
        result["snapshots"][step] = {"seconds": round(time.perf_counter() - t0, 1),
                                     "splats": len(splats), "mb": round(len(data) / 1e6, 2)}
        print(f"[{run}] snapshot {step}: {len(splats):,} splats, {len(data) / 1e6:.1f} MB",
              flush=True)

    t1 = time.perf_counter()
    hb = threading.Thread(target=heartbeat, daemon=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=TRAINER.parent, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        hb.start()
        save_progress()
        pending = list(opt["snapshots"])
        code = None
        while pending:
            if time.perf_counter() - t1 > max_train_seconds:
                proc.kill()
                code = "timeout"
                break
            path = result_dir / "ply" / f"point_cloud_{pending[0] - 1}.ply"
            if not path.exists():
                path = result_dir / "ply" / f"point_cloud_{pending[0]}.ply"
            if path.exists() and (splat_app._settled(path) or proc.poll() is not None):
                step = pending.pop(0)
                export(step, path)
                if pending:
                    path.unlink(missing_ok=True)
                else:
                    with lock:
                        shutil.copyfile(path, out / "point_cloud.ply")
                continue
            if proc.poll() is not None:
                break
            time.sleep(1.0)
        if code is None:
            try:
                code = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = "timeout after the last snapshot"
    stop.set()
    result["train_s"] = round(time.perf_counter() - t1, 1)
    result["exit"] = code
    result["missing_snapshots"] = pending
    (out / "stats").mkdir(exist_ok=True)
    result["stats"] = {}
    for f in sorted((result_dir / "stats").glob("*.json")):
        shutil.copyfile(f, out / "stats" / f.name)
        if f.name.startswith("val_step"):
            result["stats"] = json.loads(f.read_text())
    save_progress(final=True)
    result["log_tail"] = log_path.read_text(errors="replace").replace("\r", "\n")[-3000:]
    result["total_s"] = round(time.perf_counter() - t0, 1)
    (out / "done.json").write_text(json.dumps(result, indent=1))
    volume.commit()
    print(f"[{run}] done in {result['total_s']} s: exit {code}; stats {result['stats']}",
          flush=True)
    return {k: v for k, v in result.items() if k != "log_tail"}


@app.local_entrypoint()
def main(run: str = "atrium_v4", max_train_seconds: int = 2100):
    call = train_job.spawn(run, max_train_seconds)
    print(f"spawned {run}: {call.object_id}", flush=True)
    # Waited on only for the log; the app is detached, so a lost connection here never stops it.
    deadline = time.time() + max_train_seconds + 1500
    while time.time() < deadline:
        try:
            print(json.dumps(call.get(timeout=60), indent=1), flush=True)
            return
        except (modal.exception.RemoteError, modal.exception.ExecutionError,
                modal.exception.FunctionTimeoutError) as exc:
            print(f"job failed: {type(exc).__name__}: {exc}", flush=True)
            return
        except (TimeoutError, modal.exception.TimeoutError):
            continue
        except Exception as exc:  # noqa: BLE001 — network: the job carries on; ask again
            print(f"wait: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(30)
