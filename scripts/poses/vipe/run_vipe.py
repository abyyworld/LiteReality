"""NVIDIA ViPE (pose_only pipeline, no instance segmentation) on the atrium benchmark frames, on Modal.

    # build the image and check the CUDA extension (cheap, short):
    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file ~/LiteReality/.env modal run --env main \
        scripts/poses/vipe/run_vipe.py::check
    # the pose runs, detached; results land on volume litereality-poses under /vipe/:
    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file ~/LiteReality/.env modal run --detach --env main \
        scripts/poses/vipe/run_vipe.py --modes own,fixedk

Then convert with scripts/poses/vipe/convert.py.

What runs (licences checked by hand, see convert.py's LICENCE string):
  - ViPE code at a pinned commit (Apache-2.0 NVIDIA code; DROID-SLAM parts BSD-3-Clause).
  - DROID-SLAM weights droid.pth (distributed by princeton-vl/DROID-SLAM, BSD-3-Clause).
  - GeoCalib pinhole weights (code Apache-2.0, weights CC BY 4.0), own-intrinsics mode only.
  - MoGe-2 ViT-L keyframe metric depth (code MIT, weights Ruicheng/moge-2-vitl "license: mit").
  - NOT run: Track-Anything / Segment-and-Track (AGPL), GroundingDINO, SAM (init.instance=null),
    UniDepth (CC BY-NC), Unik3D (CC BY-NC-SA), Depth Pro 2, MetricTrail weights, SuperPoint.
"""

import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import modal

VIPE_COMMIT = "d48641590f1431963d7b5363b7acf7dcbea6ab4d"
FIXED_FX = 1221.9  # frames.json fx_from_35mm_px, at 1080 x 1920
HALF = 2  # frames are fed at 540 x 960; ViPE's SLAM works at ~336 x 600 anyway

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0", "ffmpeg", "wget")
    .pip_install("setuptools>=77", "wheel", "ninja", "numpy")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        # nvidia-vipe's runtime deps minus gtsam-develop (only used by the long-sequence
        # pose graph; its pinned dev wheel is no longer on PyPI)
        "click", "einops", "gdown", "huggingface_hub", "hydra-core", "imageio[ffmpeg]", "kornia",
        "matplotlib", "omegaconf", "opencv-python-headless<5", "OpenEXR<3.3.0", "pydantic>=2,<3",
        "pillow", "python-pycg", "ray", "rerun-sdk", "safetensors", "scipy", "timm", "tqdm",
        "transformers>=4,<5", "viser",
    )
    .run_commands(
        f"git clone https://github.com/nv-tlabs/vipe.git /opt/vipe && cd /opt/vipe && git checkout {VIPE_COMMIT}",
        # the standalone Python's sysconfig links with clang++, which the CUDA image lacks
        "cd /opt/vipe && CC=gcc CXX=g++ LDSHARED='gcc -shared' LDCXXSHARED='g++ -shared' "
        "TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9' MAX_JOBS=8 pip install --no-build-isolation --no-deps .",
    )
    .add_local_file(Path(__file__).parent / "vipe_driver.py", "/opt/vipe/vipe_driver.py")
)
app = modal.App("poses-vipe", image=image)
vol = modal.Volume.from_name("litereality-poses")
OUT = Path("/vol/vipe")


@app.function(gpu=["L4", "A10G", "L40S"], cpu=2, memory=8192, timeout=600)
def check():
    import torch
    import vipe
    import vipe_ext  # noqa: F401

    print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    print("vipe", vipe.__file__)
    subprocess.run("cd /opt/vipe && git log -1 --format='%H %cd'", shell=True)


def _prepare_frames() -> Path:
    import cv2

    raw = Path("/tmp/raw")
    raw.mkdir(exist_ok=True)
    with tarfile.open("/vol/atrium/frames.tar") as tar:
        tar.extractall(raw)
    jpgs = sorted(p for p in raw.rglob("*.jpg") if not p.name.startswith("._"))  # skip macOS AppleDouble files
    assert len(jpgs) == 693, len(jpgs)
    dst = Path("/tmp/half/atrium")
    dst.mkdir(parents=True, exist_ok=True)
    for p in jpgs:
        im = cv2.imread(str(p))
        assert im is not None, p
        assert im.shape[:2] == (1920, 1080), (p, im.shape)
        im = cv2.resize(im, (1080 // HALF, 1920 // HALF), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / (p.stem + ".png")), im)
    print(f"prepared {len(jpgs)} frames at {1080 // HALF}x{1920 // HALF}")
    return dst


@app.function(gpu=["L4", "A10G", "L40S"], cpu=4, memory=16384, timeout=3 * 3600, volumes={"/vol": vol})
def run(modes: str = "own,fixedk"):
    import torch

    t_start = time.time()
    gpu_name = torch.cuda.get_device_name(0)
    print("gpu", gpu_name)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "cache"
    env = dict(os.environ, TORCH_HOME=str(cache / "torch"), HF_HOME=str(cache / "hf"), PYTHONUNBUFFERED="1")
    frames = _prepare_frames()
    names = sorted(p.name for p in frames.iterdir())
    status = {"commit": VIPE_COMMIT, "gpu": gpu_name, "n_frames": len(names), "input_size": [540, 960], "runs": {}}
    for mode in [m for m in modes.split(",") if m]:
        out = Path(f"/tmp/out_{mode}")
        cmd = [
            "python", "/opt/vipe/vipe_driver.py", "pipeline=pose_only", "streams=frame_dir_stream",
            f"streams.base_path={frames}", "pipeline.init.instance=null", f"pipeline.output.path={out}",
            "pipeline.output.save_artifacts=true", "pipeline.output.save_viz=false",
        ]
        run_env = dict(env)
        if mode == "fixedk":
            cmd.append("pipeline.init.intrinsics=gt")
            run_env["VIPE_FIXED_FX"] = str(FIXED_FX)
        t0 = time.time()
        log_path = OUT / f"log_{mode}.txt"
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, cwd="/opt/vipe", env=run_env, stdout=log, stderr=subprocess.STDOUT)
        secs = time.time() - t0
        dst = OUT / f"out_{mode}"
        if dst.exists():
            shutil.rmtree(dst)
        if out.exists():
            shutil.copytree(out, dst)
        status["runs"][mode] = {"returncode": proc.returncode, "seconds": round(secs, 1)}
        status["names"] = names
        status["function_seconds"] = round(time.time() - t_start, 1)
        (OUT / "status.json").write_text(json.dumps(status, indent=1))
        vol.commit()
        print(mode, proc.returncode, round(secs, 1))
    status["function_seconds"] = round(time.time() - t_start, 1)
    status["done"] = True
    (OUT / "status.json").write_text(json.dumps(status, indent=1))
    vol.commit()
    print(json.dumps({k: v for k, v in status.items() if k != "names"}))


@app.local_entrypoint()
def main(modes: str = "own,fixedk"):
    run.remote(modes)
