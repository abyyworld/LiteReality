"""Train gsplat 1.5.3 on a prepared Eyeful Tower scene, on one Modal H100, from the Volume.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run modal run --detach --env main \
        scripts/eyeful/train_eyeful.py --arms A,B --steps-b 10000

An ephemeral app only (never deployed). The image is deploy/modal/splat/app.py's trainer image,
layer for layer (so its cached CUDA builds are reused), plus one patch to gsplat's COLMAP reader
(patch_colmap_split.py: the official test list instead of every-8th, and a common centre crop of
the training photos for batches of more than one). The dataset is read from the Volume
`litereality-eyeful` (/<scene>/gsplat_2k, scripts/eyeful/prepare_eyeful.py), copied to local disk
first. Each arm writes to /<scene>/runs/<run name>/ on the Volume:

    result.json        command, timings, GPU, trainer metrics per eval step, splat count, sizes
    trainer.log        the trainer's own output
    stats/*.json       gsplat's val_step*.json (PSNR / SSIM / LPIPS over the 71 test images)
    point_cloud.ply    the final splats as the trainer saved them (its own frame)
    splats_sh3.spz     models/splat/spz.py at SH degree 3 (and splats_sh1.spz for arm B)
    renders/*.jpg      every test image, ground truth | render, JPEG q92 (the trainer's own)

Arms:
    A  gsplat's default 3DGS (`default` strategy), batch 1, 30k steps, SH 3, 2K, every flag at
       gsplat's default except --data-factor 1 (gsplat defaults to 4) and no viewer/video/tb.
    B  our room settings (deploy/modal/splat/app.py DEFAULTS as atrium_stray v2 used them): MCMC,
       batch 4, cap_max 800k, fused bilateral grid, antialiased, visible Adam, schedules scaled to
       the step count, no world renormalisation; depth losses off (Eyeful has no depth). Training
       photos are centre-cropped to a common size so a batch stacks.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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
COLMAP_PY = GSPLAT_DIR / "examples" / "datasets" / "colmap.py"
PATCH = "/opt/litereality/patch_trainer.py"
SPLIT_PATCH = "/opt/litereality/patch_colmap_split.py"
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
    # ---- ours: the explicit test list + common training crop ----
    .add_local_file(HERE / "patch_colmap_split.py", SPLIT_PATCH, copy=True)
    .run_commands(f"python {SPLIT_PATCH} {COLMAP_PY}",
                  f"python {SPLIT_PATCH} --eval-jpeg {TRAINER}",
                  "python -c \"import imageio, numpy as np;"
                  " imageio.imwrite('/tmp/t.jpg', np.zeros((8, 8, 3), np.uint8), quality=92)\"")
    .add_local_python_source("litereality_agent")
)

VOLUME_NAME = "litereality-eyeful"
MOUNT = Path("/vol")
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("litereality-eyeful-train")

SCHEDULE_STEPS = 7000   # app.py: MCMC growth / SH climb tuned for this many steps


def command(arm: str, data_dir: Path, result_dir: Path, steps: int) -> tuple[list[str], dict]:
    common = ["--data-dir", str(data_dir), "--data-factor", "1", "--result-dir", str(result_dir),
              "--disable-viewer", "--disable-video", "--tb-every", "0",
              "--max-steps", str(steps), "--save-ply", "--ply-steps", str(steps),
              "--save-steps", str(steps)]
    env = {}
    if arm == "A":
        evals = sorted({s for s in (7000, steps) if s <= steps})
        cmd = ["default", *common, "--eval-steps", *map(str, evals)]
    elif arm == "B":
        f = steps / SCHEDULE_STEPS
        cmd = ["mcmc", *common, "--no-normalize-world-space", "--batch-size", "4",
               "--eval-steps", str(steps), "--sh-degree", "3",
               "--strategy.cap-max", "800000", "--use-fused-bilagrid", "--antialiased",
               "--visible-adam",
               "--strategy.refine-start-iter", str(max(1, round(500 * f))),
               "--strategy.refine-every", str(max(1, round(100 * f))),
               "--sh-degree-interval", str(max(1, round(1000 * f)))]
        env["LR_TRAIN_CROP"] = "1"
    else:
        raise ValueError(arm)
    return [sys.executable, str(TRAINER), *cmd], env


def _copy_tree(src: Path, dst: Path, threads: int = 32) -> int:
    from concurrent.futures import ThreadPoolExecutor

    files = [p for p in src.rglob("*") if p.is_file()]
    for p in files:
        (dst / p.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda p: shutil.copyfile(p, dst / p.relative_to(src)), files))
    return sum(p.stat().st_size for p in files)


@app.function(image=image, gpu=GPU, volumes={MOUNT: volume}, cpu=4, memory=32768,
              scaledown_window=2, timeout=3000)
def train(arm: str, steps: int, scene: str = "office1b", name: str = "",
          max_train_seconds: int = 2400) -> dict:
    import numpy as np
    import torch

    from litereality_agent.models.splat import ply, spz

    t0 = time.perf_counter()
    name = name or f"{arm}_{steps}"
    out = MOUNT / scene / "runs" / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    work = Path("/tmp/eyeful")
    data_dir, result_dir = work / "data", work / "result"
    nbytes = _copy_tree(MOUNT / scene / "gsplat_2k", data_dir)
    t_copy = time.perf_counter() - t0
    cmd, env = command(arm, data_dir, result_dir, steps)
    log_path = out / "trainer.log"
    gpu = torch.cuda.get_device_name(0)
    print(f"[{name}] {gpu}; copied {nbytes / 1e9:.2f} GB in {t_copy:.0f} s; {' '.join(cmd[2:])}",
          flush=True)
    result = {"arm": arm, "name": name, "scene": scene, "steps": steps, "gpu": gpu,
              "command": cmd[1:], "env": env, "copy_s": round(t_copy, 1)}

    stop = threading.Event()

    def heartbeat():   # progress to the app log, and the log onto the Volume, every minute
        while not stop.wait(60):
            try:
                text = log_path.read_text(errors="replace")
                last = text.replace("\r", "\n").strip().splitlines()[-1:] or [""]
                print(f"[{name}] {time.perf_counter() - t0:.0f} s | {last[0][-160:]}", flush=True)
                volume.commit()
            except Exception as exc:  # noqa: BLE001
                print(f"[{name}] heartbeat: {exc}", flush=True)

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    t1 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=TRAINER.parent, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1", **env})
        try:
            code = proc.wait(timeout=max_train_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = "timeout"
    stop.set()
    result["train_s"] = round(time.perf_counter() - t1, 1)
    result["exit"] = code
    text = log_path.read_text(errors="replace")
    result["log_tail"] = text.replace("\r", "\n")[-3000:]
    # Trainer metrics at each eval step.
    result["stats"] = {}
    stats_dir = out / "stats"
    stats_dir.mkdir()
    for f in sorted((result_dir / "stats").glob("*.json")):
        shutil.copyfile(f, stats_dir / f.name)
        if f.name.startswith("val_step"):
            result["stats"][f.stem] = json.loads(f.read_text())
    # Splats: the trainer's ply, and our SPZ.
    plys = sorted((result_dir / "ply").glob("*.ply"), key=lambda p: p.stat().st_mtime)
    if plys:
        data = plys[-1].read_bytes()
        shutil.copyfile(plys[-1], out / "point_cloud.ply")
        result["ply_mb"] = round(len(data) / 1e6, 1)
        degrees = [3, 1] if arm == "B" else [3]
        for d in degrees:
            splats = ply.read(data, max_sh_degree=d)
            blob = spz.encode(splats, antialiased=(arm == "B"))
            (out / f"splats_sh{d}.spz").write_bytes(blob)
            result[f"spz_sh{d}_mb"] = round(len(blob) / 1e6, 1)
            result["splats"] = len(splats)
    # The world frame the splats are in: arm A's trainer renormalises the world (B does not).
    result["world_transform"] = np.eye(4).tolist()
    try:
        if arm != "A":
            raise StopIteration
        sys.path.insert(0, str(TRAINER.parent))
        from datasets.colmap import Parser  # noqa: E402

        p = Parser(str(data_dir), factor=1, normalize=(arm == "A"), test_every=8)
        result["world_transform"] = np.asarray(p.transform).tolist()
        result["scene_scale"] = float(p.scene_scale)
    except StopIteration:
        pass
    except Exception as exc:  # noqa: BLE001
        result["world_transform_error"] = repr(exc)
    # Test renders (ground truth | render), every one as JPEG.
    rdir = out / "renders"
    rdir.mkdir()
    final = sorted((result_dir / "renders").glob(f"val_step{steps - 1}_*.jpg"))
    for f in final:
        shutil.copyfile(f, rdir / f.name)
    result["renders"] = len(final)
    m = None
    for line in text.replace("\r", "\n").splitlines():
        if "it/s]" in line or "s/it]" in line:
            m = line
    result["last_progress"] = (m or "")[-200:]
    result["total_s"] = round(time.perf_counter() - t0, 1)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    volume.commit()
    print(f"[{name}] done in {result['total_s']} s: exit {code}; "
          f"{json.dumps({k: {kk: round(v, 4) for kk, v in s.items()} for k, s in result['stats'].items()})}",
          flush=True)
    return {k: v for k, v in result.items() if k != "log_tail"}


@app.local_entrypoint()
def main(arms: str = "A,B", steps_a: int = 30000, steps_b: int = 10000, scene: str = "office1b",
         tag: str = "", max_a: int = 2400, max_b: int = 1500):
    calls = {}
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        steps = steps_a if arm == "A" else steps_b
        name = f"{arm}_{steps}{tag}"
        calls[name] = train.spawn(arm, steps, scene, name, max_a if arm == "A" else max_b)
        print(f"spawned {name}: {calls[name].object_id}")
    for name, call in calls.items():
        r = call.get()
        print(json.dumps(r, indent=1))
