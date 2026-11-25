"""DROID-SLAM (monocular, fixed pinhole intrinsics) on the atrium benchmark frames, on Modal.

    # build the image and check the CUDA extensions (short):
    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file ~/LiteReality/.env modal run --env main \
        scripts/poses/droid/run_droid.py::check
    # the pose runs, detached; results land on volume litereality-poses under /droid/:
    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file ~/LiteReality/.env modal run --detach --env main \
        scripts/poses/droid/run_droid.py --fxs 1221.9,1604

Then convert with scripts/poses/droid/convert.py.

What runs (licences checked by hand, see convert.py's LICENCE string):
  - princeton-vl/DROID-SLAM at a pinned commit (BSD-3-Clause), with its submodules
    lietorch (BSD-3-Clause) and pytorch_scatter (MIT; installed from PyG's prebuilt wheel).
  - droid.pth, the weights the DROID-SLAM README links (Google Drive, no separate terms; trained
    on TartanAir, CC BY 4.0).
"""

import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import modal

DROID_COMMIT = "2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99"
WEIGHTS_GDRIVE_ID = "1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh"
CX, CY = 540.0, 960.0  # principal point at 1080 x 1920

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install("setuptools", "wheel", "ninja", "numpy<2.3")
    .pip_install("torch==2.7.0", "torchvision==0.22.0", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("torch-scatter", find_links="https://data.pyg.org/whl/torch-2.7.0+cu126.html")
    .pip_install("opencv-python-headless<5", "tqdm", "scipy", "gdown", "pyyaml")
    .run_commands(
        "git clone https://github.com/princeton-vl/DROID-SLAM.git /opt/DROID-SLAM"
        f" && cd /opt/DROID-SLAM && git checkout {DROID_COMMIT}"
        " && git submodule update --init --recursive thirdparty/lietorch",
        "cd /opt/DROID-SLAM/thirdparty/lietorch && CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='8.6;8.9' MAX_JOBS=8"
        " pip install --no-build-isolation --no-deps .",
        "cd /opt/DROID-SLAM && CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='8.6;8.9' MAX_JOBS=8 pip install --no-build-isolation --no-deps .",
        f"cd /opt/DROID-SLAM && gdown {WEIGHTS_GDRIVE_ID} -O droid.pth && sha256sum droid.pth",
    )
    .pip_install("matplotlib")  # factor_graph.py imports it
    .add_local_file(Path(__file__).parent / "droid_driver.py", "/opt/DROID-SLAM/droid_driver.py")
)
app = modal.App("poses-droid", image=image)
vol = modal.Volume.from_name("litereality-poses")
OUT = Path("/vol/droid")


@app.function(gpu="L4", cpu=2, memory=8192, timeout=600)
def check():
    import droid_backends  # noqa: F401
    import lietorch  # noqa: F401
    import torch  # first: the extensions need torch's shared libraries loaded
    import torch_scatter  # noqa: F401

    print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
    x = lietorch.SE3.Identity(2, device="cuda")
    print("lietorch ok", x.data.shape)
    subprocess.run("cd /opt/DROID-SLAM && git log -1 --format='%H %cd' && ls -la droid.pth && sha256sum droid.pth",
                   shell=True)


def _prepare_frames() -> Path:
    raw = Path("/tmp/raw")
    raw.mkdir(exist_ok=True)
    with tarfile.open("/vol/atrium/frames.tar") as tar:
        tar.extractall(raw)
    jpgs = sorted(p for p in raw.rglob("*.jpg") if not p.name.startswith("._"))  # skip macOS AppleDouble
    dst = Path("/tmp/frames")
    dst.mkdir(exist_ok=True)
    for p in jpgs:
        (dst / p.name).symlink_to(p)
    print(f"prepared {len(jpgs)} frames")
    assert len(jpgs) == len({p.name for p in jpgs}) == 693, len(jpgs)
    return dst


@app.function(gpu="L4", cpu=4, memory=16384, timeout=2 * 3600, volumes={"/vol": vol})
def run(fxs: str = "1221.9,1604"):
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = _prepare_frames()
    status = {"commit": DROID_COMMIT, "gpu": "L4", "n_frames": len(os.listdir(frames)), "runs": {}}
    for fx in [float(f) for f in fxs.split(",") if f]:
        tag = f"f{fx:g}"
        t0 = time.time()
        cmd = ["python", "/opt/DROID-SLAM/droid_driver.py", str(frames), str(fx), str(fx), str(CX), str(CY),
               str(OUT / f"traj_{tag}.npz")]
        with open(OUT / f"log_{tag}.txt", "w") as log:
            proc = subprocess.run(cmd, cwd="/opt/DROID-SLAM", stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, PYTHONUNBUFFERED="1"))
        status["runs"][tag] = {"fx": fx, "returncode": proc.returncode, "seconds": round(time.time() - t0, 1)}
        status["function_seconds"] = round(time.time() - t_start, 1)
        (OUT / "status.json").write_text(json.dumps(status, indent=1))
        vol.commit()
        print(tag, proc.returncode, round(time.time() - t0, 1), flush=True)
    status["function_seconds"] = round(time.time() - t_start, 1)
    status["done"] = True
    (OUT / "status.json").write_text(json.dumps(status, indent=1))
    vol.commit()
    print(json.dumps(status))


@app.local_entrypoint()
def main(fxs: str = "1221.9,1604"):
    run.remote(fxs)
