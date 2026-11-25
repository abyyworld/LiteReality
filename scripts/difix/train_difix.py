"""RESEARCH ONLY: NVIDIA Difix3D's progressive update on the atrium, as detached Modal jobs.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file .env modal run --detach --env main \
        scripts/difix/train_difix.py --mode smoke            # Difix alone on renders of the live room
    ... --mode arms                                          # B (Difix) and C (control) in parallel

Volume litereality-difix, folder /atrium (uploaded first: scripts/difix/README.md):
    dataset.tar, options.json   the live v2 room's training set and options (scripts/difix/prepare.py)
    novel_poses.npy             [rounds, views, 4, 4] (scripts/difix/poses.py)
    live_v2.spz                 the live room's trained splat (run/atrium_stray_q/v2/raw.spz.bin)
Out:
    smoke/                      renders, references and Difix's repairs at each width; smoke.json
    B/, C/                      progress.json, trainer.log, point_cloud.ply, snap_10000.spz, stats/,
                                val/ (a quarter of the held-out photos beside their renders),
                                difix/ (B: each round's renders and repairs, rounds.json), done.json

Arms (the same dataset, options, seed and 10,000 steps as the live v2 room):
    C   the live room's training, again: the control
    B   C plus four Difix rounds at steps 6000, 7000, 8000, 9000 (patch_difix.py)

Difix (github.com/nv-tlabs/Difix3D, weights nvidia/difix_ref) is under NVIDIA's research /
evaluation-only licence. It lives only in this ephemeral app's image, never in the deployed
splat app, and no room trained with it may be served by the hosted room call.
"""

from __future__ import annotations

import io
import json
import os
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
# ---- Difix3D: pinned commit (main as read on 2026-11-25) ----
DIFIX_REPO_URL = "https://github.com/nv-tlabs/Difix3D"
DIFIX_DIR = "/opt/difix3d"
DIFIX_STEPS = [6000, 7000, 8000, 9000]

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
    # ---- research only: Difix and our loop around it ----
    .pip_install("diffusers==0.25.1", "transformers==4.38.0", "huggingface-hub==0.25.1",
                 "peft==0.9.0", "einops", "accelerate==0.27.2")
    .run_commands(
        f"git clone -q {DIFIX_REPO_URL} {DIFIX_DIR} && cd {DIFIX_DIR} && git log -1 --format=%H",
        "python -c \"from huggingface_hub import snapshot_download;"
        " print(snapshot_download('nvidia/difix_ref'))\"",
        f"cd {DIFIX_DIR} && python -c 'from src.pipeline_difix import DifixPipeline'",
    )
    .add_local_file(HERE / "litereality_difix.py", str(TRAINER.parent / "litereality_difix.py"),
                    copy=True)
    .add_local_file(HERE / "patch_difix.py", "/opt/litereality/patch_difix.py", copy=True)
    .run_commands(f"python /opt/litereality/patch_difix.py {TRAINER}",
                  f"cd {TRAINER.parent} && python -c 'import litereality_difix'")
    .add_local_file(SPLAT / "app.py", "/root/splat_app.py")
    .add_local_file(HERE / "poses.py", "/root/difix_poses.py")
    .add_local_python_source("litereality_agent")
)

volume = modal.Volume.from_name("litereality-difix", create_if_missing=True)
app = modal.App("litereality-difix-research")
MOUNT = Path("/vol")
RUN = "atrium"


def _extract(tar_path: Path, dest: Path, only=None) -> None:
    with tarfile.open(fileobj=io.BytesIO(tar_path.read_bytes())) as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        for m in members:
            if m.name.startswith("/") or ".." in m.name:
                raise ValueError(f"unexpected entry in the dataset: {m.name}")
        if only is not None:
            members = [m for m in members if only(m.name)]
        tf.extractall(dest, members=members)


def _cameras(data_dir: Path):
    """The first camera's K and size, and every photo's (name, camera-to-world), sorted."""
    import struct

    import numpy as np

    sys.path.insert(0, "/root")
    from difix_poses import read_images_bin

    raw = (data_dir / "sparse/0/cameras.bin").read_bytes()
    _cid, model, w, h = struct.unpack("<iiQQ", raw[8:32])
    params = struct.unpack("<4d", raw[32:64])
    fx, fy, cx, cy = params if model == 1 else (params[0], params[0], params[1], params[2])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    return K, int(w), int(h), read_images_bin((data_dir / "sparse/0/images.bin").read_bytes())


@app.function(image=image, gpu=GPU, volumes={MOUNT: volume}, cpu=8, memory=32768,
              scaledown_window=2, timeout=1800)
def smoke(widths: list[int], views: int = 8) -> dict:
    """Difix alone: renders of the live room from the last round's novel poses, repaired at each
    of `widths` (0 = native) with the nearest training photo as reference; time and memory."""
    import numpy as np
    import torch
    from gsplat import rasterization
    from PIL import Image

    sys.path.insert(0, str(TRAINER.parent))
    from litereality_difix import load_pipeline, nearest, repair

    from litereality_agent.models.splat import spz
    from litereality_agent.models.splat.spz import SH_C0

    t0 = time.perf_counter()
    src, out = MOUNT / RUN, MOUNT / RUN / "smoke"
    volume.reload()
    out.mkdir(parents=True, exist_ok=True)
    data = Path("/tmp/data")
    _extract(src / "dataset.tar", data, only=lambda n: n.startswith("sparse/"))
    K, w, h, cams = _cameras(data)
    held = np.arange(len(cams)) % 8 == 0
    train = [(n, c) for i, (n, c) in enumerate(cams) if not held[i]]
    train_c2w = np.stack([c for _, c in train])
    poses = np.load(src / "novel_poses.npy")[-1]
    pick = np.linspace(0, len(poses) - 1, views).round().astype(int)
    refs = {train[nearest(train_c2w, poses[i])][0] for i in pick}
    _extract(src / "dataset.tar", data, only=lambda n: n[len("images/"):] in refs)
    s = spz.decode((src / "live_v2.spz").read_bytes())
    dev = "cuda"
    sh0 = (torch.from_numpy(s.colors).float() - 0.5) / SH_C0
    coeffs = torch.cat([sh0[:, None], torch.from_numpy(s.sh).float()], 1).to(dev)
    means = torch.from_numpy(s.positions).float().to(dev)
    quats = torch.from_numpy(s.rotations[:, [3, 0, 1, 2]]).float().to(dev)    # wxyz
    scales = torch.from_numpy(s.scales).float().to(dev)
    opac = torch.from_numpy(s.opacities).float().to(dev)
    pipe_t = time.perf_counter()
    pipe = load_pipeline()
    load_s = time.perf_counter() - pipe_t
    report = {"views": len(pick), "size": [w, h], "load_s": round(load_s, 1), "widths": {}}
    for i, v in enumerate(pick):
        c2w = torch.from_numpy(poses[v]).float().to(dev)
        with torch.no_grad():
            img, alpha, _ = rasterization(means, quats, scales, opac, coeffs,
                                          torch.linalg.inv(c2w)[None],
                                          torch.from_numpy(K)[None].to(dev), w, h,
                                          sh_degree=1, rasterize_mode="antialiased")
        render = Image.fromarray((img[0].clamp(0, 1) * 255).byte().cpu().numpy())
        ref = Image.open(data / "images" / train[nearest(train_c2w, poses[v])][0]).convert("RGB")
        render.save(out / f"{i:02d}_render.jpg", quality=92)
        ref.save(out / f"{i:02d}_ref.jpg", quality=85)
        (out / f"{i:02d}_alpha.txt").write_text(f"{float((alpha[0, ..., 0] > 0.5).float().mean()):.3f}")
        for width in widths:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t = time.perf_counter()
            fixed = repair(pipe, render, ref, width)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            fixed.save(out / f"{i:02d}_fixed_{width or 'native'}.jpg", quality=92)
            rec = report["widths"].setdefault(str(width or "native"), {"s": [], "gb": []})
            rec["s"].append(round(dt, 3))
            rec["gb"].append(round(torch.cuda.max_memory_allocated() / 1e9, 2))
    for rec in report["widths"].values():
        rec["s_median"] = float(np.median(rec["s"][1:] or rec["s"]))      # the first warms up
        rec["gb_max"] = max(rec["gb"])
    report["total_s"] = round(time.perf_counter() - t0, 1)
    report["gpu"] = torch.cuda.get_device_name(0)
    (out / "smoke.json").write_text(json.dumps(report, indent=1))
    volume.commit()
    print(json.dumps(report), flush=True)
    return report


@app.function(image=image, gpu=GPU, volumes={MOUNT: volume}, cpu=8, memory=32768,
              scaledown_window=2, timeout=3000)
def train_arm(arm: str, difix_width: int = 0, max_train_seconds: int = 2400) -> dict:
    """One arm: the live room's training again (C), or with the Difix rounds (B)."""
    sys.path.insert(0, "/root")
    import splat_app
    import torch

    from litereality_agent.models.splat import ply, spz

    t0 = time.perf_counter()
    src, out = MOUNT / RUN, MOUNT / RUN / arm
    volume.reload()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    options = json.loads((src / "options.json").read_text())
    options.update({"steps": 10000, "snapshots": [10000]})
    opt = splat_app._options(options)
    work = Path("/tmp/splat")
    data_dir, result_dir = work / "data", work / "result"
    _extract(src / "dataset.tar", data_dir)
    cmd = splat_app._command(data_dir, result_dir, opt)
    if arm == "B":
        shutil.copyfile(src / "novel_poses.npy", data_dir / "novel_poses.npy")
        cmd += ["--difix-poses", str(data_dir / "novel_poses.npy"),
                "--difix-steps", *map(str, DIFIX_STEPS), "--difix-width", str(int(difix_width))]
    gpu = torch.cuda.get_device_name(0)
    result = {"arm": arm, "gpu": gpu, "options": opt, "command": cmd[2:],
              "setup_s": round(time.perf_counter() - t0, 1)}
    print(f"[{arm}] {gpu}: {' '.join(cmd[2:])}", flush=True)
    log_path = work / "trainer.log"
    stop = threading.Event()
    t1 = time.perf_counter()

    def save_progress(final: bool = False) -> None:
        sys.path.insert(0, str(REPO))
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        lines = [ln for ln in text.replace("\r", "\n").splitlines() if ln.strip()]
        prog = {"seconds": round(time.perf_counter() - t0, 1), "final": final, "time": time.time(),
                "last": lines[-1][-300:] if lines else ""}
        if log_path.exists():
            shutil.copyfile(log_path, out / "trainer.log")
        (out / "progress.json").write_text(json.dumps(prog))
        volume.commit()

    def heartbeat() -> None:
        while not stop.wait(30):
            try:
                save_progress()
            except Exception as exc:  # noqa: BLE001
                print(f"[{arm}] heartbeat: {exc}", flush=True)

    hb = threading.Thread(target=heartbeat, daemon=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=TRAINER.parent, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        hb.start()
        try:
            code = proc.wait(timeout=max_train_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = "timeout"
    stop.set()
    result["train_s"] = round(time.perf_counter() - t1, 1)
    result["exit"] = code
    plys = sorted((result_dir / "ply").glob("point_cloud_*.ply"))
    if plys:
        shutil.copyfile(plys[-1], out / "point_cloud.ply")
        splats = ply.read(plys[-1].read_bytes(), max_sh_degree=opt["export_sh_degree"])
        (out / "snap_10000.spz").write_bytes(spz.encode(splats,
                                                        antialiased=bool(opt["antialiased"])))
        result["splats"] = len(splats)
    (out / "stats").mkdir(exist_ok=True)
    for f in sorted((result_dir / "stats").glob("*.json")):
        shutil.copyfile(f, out / "stats" / f.name)
        if f.name.startswith("val_step"):
            result["stats"] = json.loads(f.read_text())
    renders = sorted((result_dir / "renders").glob("val_step*_*.png"))
    if renders:
        from PIL import Image

        (out / "val").mkdir(exist_ok=True)
        for p in renders[::4]:
            Image.open(p).convert("RGB").save(out / "val" / (p.stem + ".jpg"), quality=90)
    if (result_dir / "difix").is_dir():
        shutil.copytree(result_dir / "difix", out / "difix")
    result["log_tail"] = log_path.read_text(errors="replace").replace("\r", "\n")[-4000:]
    result["total_s"] = round(time.perf_counter() - t0, 1)
    (out / "done.json").write_text(json.dumps(result, indent=1))
    save_progress(final=True)
    print(f"[{arm}] done: exit {code}, {result.get('stats')}", flush=True)
    return {k: v for k, v in result.items() if k != "log_tail"}


@app.local_entrypoint()
def main(mode: str = "smoke", widths: str = "1024,0", difix_width: int = 0):
    if mode == "smoke":
        print(json.dumps(smoke.remote([int(w) for w in widths.split(",")]), indent=1))
        return
    if mode == "arms":
        calls = {arm: train_arm.spawn(arm, difix_width) for arm in ("C", "B")}
        for arm, call in calls.items():
            print(f"spawned {arm}: {call.object_id}", flush=True)
        for arm, call in calls.items():
            while True:
                try:
                    print(arm, json.dumps(call.get(timeout=60), indent=1), flush=True)
                    break
                except (TimeoutError, modal.exception.TimeoutError):
                    continue
                except Exception as exc:  # noqa: BLE001 — detached: the jobs carry on
                    print(f"{arm}: {type(exc).__name__}: {exc}", flush=True)
                    break
        return
    raise SystemExit(f"unknown mode {mode!r}")
