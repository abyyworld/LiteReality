"""LichtFeld Studio (github.com/MrNeRF/LichtFeld-Studio, GPL-3.0) on Modal: a research comparison
against our gsplat trainer (deploy/modal/splat). Never part of the served pipeline.

No Linux binaries are published (GitHub Releases has no assets; the Windows build is the paid
"Portal"), so it is built from source at a pinned tag, following upstream CI
(.github/workflows/ubuntu.yml): CUDA 12.8 devel on Ubuntu 24.04, Kitware CMake 4.4.0, GCC 14,
vcpkg in manifest mode, then `cmake --install` of the portable layout (bin/run_lichtfeld.sh plus
bundled libs). The image builder has one core, so the compile runs in a 16-core function
(`build`) and the install lands as a tarball on a Volume; `train` unpacks it on a GPU.

    # one-off build (CPU, ~1-2 h; vcpkg's binary cache on the Volume makes a retry quick)
    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file .env modal run --detach \\
        deploy/modal/lichtfeld/app.py::build
    # its --help and --version (L4, a minute)
    ... modal run deploy/modal/lichtfeld/app.py::smoke
    # training: scripts/compare_trainers/run_lichtfeld.py (builds our dataset tar, calls `train`)

`train(dataset_tar, options)`: the tar `litereality_agent.models.splat.dataset.build(...).tar()`
writes (images/ + sparse/0/*.bin, our ARKit poses and LiDAR seed points: the same input as the
gsplat runs) -> {"ply": bytes, "log": str, "seconds": {...}, "gpu": str, "cmd": [...]}.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "litereality-lichtfeld"
LFS_REPO = "https://github.com/MrNeRF/LichtFeld-Studio.git"
LFS_TAG = "v0.5.4.3"
LFS_COMMIT = "06c9a040677c0f302dc0e169ff5d5641a9f9c88a"
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
CMAKE_PIN = "4.4.0-0kitware1ubuntu24.04.1"       # CI's pin: 4.4.1 breaks check_symbol_exists
MIN_SM = 80                                        # PTX for SM >= 8.0 (A100, L4, L40S, H100)
VOLUME = "litereality-lichtfeld"
VOL = Path("/vol")
DIST_TAR = VOL / f"lfs-{LFS_TAG}-sm{MIN_SM}.tar.gz"
SRC = Path("/opt/src/LichtFeld-Studio")
VCPKG = Path("/opt/vcpkg")
DIST = Path("/opt/lfs")
GPU = ["L40S", "A100-40GB"]   # either: 40+ GB, SM >= 8.0; L40S first (cheaper)
# The function's Python is Modal's, in /usr/local/bin; the build's tools want the system python3.
BUILD_PATH = "/usr/local/cuda/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/sbin:/usr/local/bin"
MAX_DATASET_BYTES = 600 * 1024 * 1024

APT = [
    "git", "curl", "wget", "ca-certificates", "gpg", "unzip", "zip", "tar", "pkg-config",
    "gcc-14", "g++-14", "ninja-build", "nasm", "autoconf", "autoconf-archive", "automake",
    "libtool", "python3", "python3-dev", "python3-venv", "bison", "flex",
    "libxinerama-dev", "libxcursor-dev", "xorg-dev", "libglu1-mesa-dev", "libwayland-dev",
    "libxkbcommon-dev", "libegl-dev", "libdecor-0-dev", "libibus-1.0-dev", "libdbus-1-dev",
    "libsystemd-dev", "libgtk-3-dev", "xvfb", "xauth", "libvulkan1", "mesa-vulkan-drivers",
    "vulkan-tools",
]

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends ca-certificates gpg wget",
        "wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -"
        " > /usr/share/keyrings/kitware-archive-keyring.gpg",
        "echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg]"
        " https://apt.kitware.com/ubuntu/ noble main' > /etc/apt/sources.list.d/kitware.list",
        "apt-get update && apt-get install -y --no-install-recommends "
        f"cmake={CMAKE_PIN} cmake-data={CMAKE_PIN} " + " ".join(APT),
        "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100"
        " --slave /usr/bin/g++ g++ /usr/bin/g++-14",
        "rm -rf /var/lib/apt/lists/*",
    )
    .run_commands(
        f"git clone -q https://github.com/microsoft/vcpkg.git {VCPKG}",
        f"cd {VCPKG} && ./bootstrap-vcpkg.sh -disableMetrics && git rev-parse HEAD > /opt/vcpkg.commit",
        f"printf 'set(VCPKG_BUILD_TYPE release)\\n' >> {VCPKG}/triplets/x64-linux.cmake",
        f"git clone -q --recursive --branch {LFS_TAG} {LFS_REPO} {SRC}",
        f"cd {SRC} && test \"$(git rev-parse HEAD)\" = {LFS_COMMIT}",
        f"mkdir -p {VCPKG}/downloads && cp {SRC}/.github/vcpkg-downloads/*.patch {VCPKG}/downloads/",
    )
    .env({
        "VCPKG_ROOT": str(VCPKG), "CC": "gcc-14", "CXX": "g++-14", "CUDA_PATH": "/usr/local/cuda",
        "VCPKG_FORCE_SYSTEM_BINARIES": "1", "CMAKE_MAKE_PROGRAM": "/usr/bin/ninja",
        "VCPKG_DEFAULT_BINARY_CACHE": str(VOL / "vcpkg-cache"),
    })
)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App(APP_NAME)


def _run(cmd: str, log: Path, cwd: Path | None = None, env: dict | None = None) -> None:
    """One build step, its output appended to `log` (on the Volume) and echoed; raises on failure."""
    with open(log, "a") as fh:
        fh.write(f"\n$ {cmd}\n")
        fh.flush()
        proc = subprocess.Popen(["bash", "-lc", cmd], cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                env={**os.environ, "PATH": BUILD_PATH, **(env or {})})
        for line in proc.stdout:
            fh.write(line)
            print(line, end="")
        code = proc.wait()
        fh.write(f"[exit {code}]\n")
    if code:
        raise RuntimeError(f"`{cmd}` exited with {code}; log {log}")


@app.function(image=image, volumes={VOL: vol}, cpu=16.0, memory=32768, timeout=5 * 3600)
def build(force: bool = False) -> dict:
    """Configure (vcpkg installs ~40 ports) and build the portable install, tar it to the Volume."""
    if DIST_TAR.exists() and not force:
        return {"dist": str(DIST_TAR), "cached": True, "mb": DIST_TAR.stat().st_size / 1e6}
    (VOL / "vcpkg-cache").mkdir(parents=True, exist_ok=True)
    (VOL / "logs").mkdir(exist_ok=True)
    log = VOL / "logs" / f"build-{time.strftime('%Y%m%d-%H%M%S')}.log"
    t0 = time.perf_counter()
    jobs = os.cpu_count() or 16
    timing = {}
    try:
        _run("nproc; free -g || true; df -h /; cmake --version | head -1; gcc --version | head -1;"
             " nvcc --version | tail -2; cat /opt/vcpkg.commit", log)
        _run(f"cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release"
             f" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
             f" -DBUILD_PORTABLE=ON -DBUILD_CUDA_MIN_SM={MIN_SM}"
             f" -DBUILD_PYTHON_STUBS=OFF -DLFS_DEV_IMPORT_SOURCE_PYTHON=OFF"
             f" -DLFS_DEV_IMPORT_SOURCE_RESOURCES=OFF -DENABLE_COMPILER_CACHE=OFF"
             f" -DLFS_VCPKG_MAX_CONCURRENCY={jobs}", log, cwd=SRC)
        timing["configure_s"] = round(time.perf_counter() - t0)
        vol.commit()                      # the vcpkg binary cache, before the app compile
        _run(f"cmake --build build -j {jobs}", log, cwd=SRC,
             env={"CMAKE_BUILD_PARALLEL_LEVEL": str(jobs)})
        timing["build_s"] = round(time.perf_counter() - t0)
        _run(f"cmake --install build --prefix {DIST} && du -sh {DIST} && ls {DIST}/bin", log,
             cwd=SRC)
        _run(f"tar -C {DIST.parent} -czf {DIST_TAR}.part {DIST.name} && mv {DIST_TAR}.part {DIST_TAR}",
             log)
    finally:
        vol.commit()
    timing["total_s"] = round(time.perf_counter() - t0)
    return {"dist": str(DIST_TAR), "mb": round(DIST_TAR.stat().st_size / 1e6, 1), "log": str(log),
            "cpus": jobs, **timing}


def _unpack() -> Path:
    vol.reload()
    if not DIST_TAR.exists():
        raise FileNotFoundError(f"{DIST_TAR} missing: run `modal run ...lichtfeld/app.py::build`")
    if not (DIST / "bin" / "run_lichtfeld.sh").exists():
        subprocess.run(["tar", "-C", str(DIST.parent), "-xzf", str(DIST_TAR)], check=True)
    return DIST / "bin" / "run_lichtfeld.sh"


def _lfs(args: list[str], log: Path | None = None, timeout: int | None = None,
         env: dict | None = None) -> subprocess.CompletedProcess:
    exe = _unpack()
    out = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=timeout,
                         env={**os.environ, "NVIDIA_DRIVER_CAPABILITIES": "all", **(env or {})})
    if log:
        log.write_text(out.stdout + "\n--- stderr ---\n" + out.stderr)
    return out


@app.function(image=image, volumes={VOL: vol}, gpu=["L4", "A10G", "L40S"], timeout=900)
def smoke() -> dict:
    """--version and --help on a GPU, plus whether a Vulkan device is visible (vulkaninfo)."""
    res = {}
    for name, args in {"version": ["--version"], "help": ["--help"]}.items():
        p = _lfs(args, timeout=300)
        res[name] = {"code": p.returncode, "out": p.stdout[-6000:], "err": p.stderr[-3000:]}
    vk = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
    res["vulkaninfo"] = {"code": vk.returncode, "out": (vk.stdout + vk.stderr)[-3000:]}
    res["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[-2000:]
    print(res["version"], res["vulkaninfo"])
    return res


DEFAULTS = {
    "strategy": "mcmc",
    "iterations": 30000,
    "max_cap": 1_000_000,
    "test_every": 8,
    "eval": True,
    "sh_degree": 3,
    "extra": [],                  # further CLI flags, as given
    "xvfb": False,                # run under Xvfb (if --headless still wants a display)
}


@app.function(image=image, volumes={VOL: vol}, gpu=GPU, timeout=3600, memory=32768, cpu=8)
def train(dataset_tar: bytes, options: dict | None = None) -> dict:
    opt = dict(DEFAULTS)
    for k, v in (options or {}).items():
        if k not in DEFAULTS:
            raise ValueError(f"unknown option {k!r}")
        opt[k] = v
    if len(dataset_tar) > MAX_DATASET_BYTES:
        raise ValueError("dataset too large")
    t0 = time.perf_counter()
    exe = _unpack()
    seconds = {"unpacked": round(time.perf_counter() - t0, 2)}
    work = Path(tempfile.mkdtemp(prefix="lfs-"))
    data, out = work / "data", work / "out"
    with tarfile.open(fileobj=io.BytesIO(dataset_tar)) as tf:
        for m in tf.getmembers():
            if not m.isfile() or m.name.startswith("/") or ".." in m.name:
                raise ValueError(f"unexpected entry {m.name}")
        tf.extractall(data)
    cmd = [str(exe), "--headless", "-d", str(data), "-o", str(out),
           "--strategy", str(opt["strategy"]), "-i", str(int(opt["iterations"])),
           "--max-cap", str(int(opt["max_cap"])), "--sh-degree", str(int(opt["sh_degree"])),
           "--test-every", str(int(opt["test_every"])), "--export", "ply"]
    if opt["eval"]:
        cmd.append("--eval")
    cmd += [str(a) for a in opt["extra"]]
    env = {**os.environ, "NVIDIA_DRIVER_CAPABILITIES": "all"}
    if opt["xvfb"]:
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x720x24", *cmd]
    log_path = work / "lfs.log"
    t1 = time.perf_counter()
    with open(log_path, "w") as fh:
        code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                              timeout=3300).returncode
    seconds["train"] = round(time.perf_counter() - t1, 2)
    log = log_path.read_text(errors="replace")
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()) \
        if out.exists() else []
    plys = sorted(out.rglob("*.ply"), key=lambda p: p.stat().st_mtime)
    result = {"code": code, "cmd": cmd, "seconds": seconds, "files": files,
              "log": log[-60000:], "gpu": _gpu_name(), "lfs": f"{LFS_TAG} {LFS_COMMIT[:9]}",
              "options": opt}
    if plys:
        result["ply_name"] = str(plys[-1].relative_to(out))
        result["ply"] = plys[-1].read_bytes()
    # Eval summaries LichtFeld writes (json/csv/txt), small enough to return whole.
    result["eval_files"] = {str(p.relative_to(out)): p.read_text(errors="replace")[:200000]
                            for p in out.rglob("*") if p.is_file()
                            and p.suffix in (".json", ".csv", ".txt") and p.stat().st_size < 2e6}
    shutil.rmtree(work, ignore_errors=True)
    return result


def _gpu_name() -> str:
    p = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return p.stdout.strip()


@app.local_entrypoint()
def main(tar: str = "", options: str = "{}", out: str = "") -> None:
    """`modal run app.py --tar d.tar.gz --options '{...}' --out res.pkl`: one `train` call, its
    result pickled to `out` (scripts/compare_trainers/run_lichtfeld.py). No --tar: the build."""
    import json
    import pickle

    if not tar:
        print(build.remote())
        return
    res = train.remote(Path(tar).read_bytes(), json.loads(options))
    Path(out).write_bytes(pickle.dumps(res))
    print({k: res[k] for k in ("code", "seconds", "gpu")})
