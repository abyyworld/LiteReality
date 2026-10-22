"""Infinigen-Sim on Modal: openable furniture sized to a scanned box (models/articulated).

    uv run modal deploy --env main deploy/modal/infinigen/app.py     # once, and after updating

Its own app, ``litereality-infinigen``, never included into ``litereality`` (deploy/modal/app.py):
that app is the GPU models and is always deployed whole, while this is a CPU function with an
image of its own — Infinigen needs Python 3.11 and exactly ``bpy==4.2.0``, which no other image
here has — and redeploying one must not rebuild the other.

The image is the study's environment, pinned: Infinigen (princeton-vl/infinigen, BSD-3) cloned at
`INFINIGEN_COMMIT` and installed with its ``sim`` extra (bpy 4.2.0, CoACD, MuJoCo, trimesh), and
`models/articulated/infinigen_worker.py` — the same script a local env runs. Building it takes
~10-15 min once (scikit-image 0.19 compiles from source); later deploys reuse it.

The request is ``{"job": {"key", "factory", "size": [W, D, H], "seed", "options"}, "image_res"}``
(`parse_request`); the response ``{"key", "ok": true, "zip_b64", "seconds", "result"}`` — the
export folder zipped, without the baked maps nothing reads (`KEEP_TEXTURES`) — or ``{"key",
"ok": false, "error"}``. ``models/articulated/infinigen.py`` (`ModalRunner`) is the client: one
input per asset through ``Function.map``, so a room's assets build in parallel containers.

Cost: one asset is 10-90 s of 2 CPUs (Modal's CPU and memory rates make a whole room of seven a
few US cents) plus each container's start; idle containers stop after `SCALEDOWN_SECONDS`.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import modal

APP_NAME = "litereality-infinigen"
FUNCTION_NAME = "generate"
INFINIGEN_REPO = "https://github.com/princeton-vl/infinigen"
INFINIGEN_COMMIT = "224598d4a2ff95b4fc928e72c222598b7122c698"   # models.articulated.INFINIGEN_COMMIT
INFINIGEN_DIR = "/opt/infinigen"
WORKER = "/opt/litereality/infinigen_worker.py"


def _worker_local(here: Path) -> Path:
    """The worker script in the checkout this file is in (deploy/modal/infinigen/ -> the repo).

    Only the deploying machine reads it (``add_local_file`` is lazy). This module is ALSO
    imported inside every container, where Modal mounts it alone as ``/root/app.py`` with no
    repo around it, so the path must be computable there without raising, even if it does
    not exist."""
    parents = Path(here).resolve().parents
    repo = parents[3] if len(parents) > 3 else Path(here).resolve().parent
    return repo / "src" / "litereality_agent" / "models" / "articulated" / "infinigen_worker.py"


WORKER_LOCAL = _worker_local(Path(__file__))
# Must match infinigen_worker.FACTORIES (tests/test_articulated.py checks).
FACTORIES = ("door", "window", "cabinet", "drawer", "dishwasher", "refrigerator", "oven")
KEEP_TEXTURES = ("_DIFFUSE.png", "_TRANSMISSION.png")
MAX_CONTAINERS = 8           # a room's openable objects at once (tea_room has seven)
CPUS = 2.0                   # the export is mostly one core (Cycles bake + CoACD); two keep it busy
MEMORY_MB = 6144
TIMEOUT_SECONDS = 20 * 60    # the slowest healthy asset took 90 s; a stuck CoACD is cut off here
SCALEDOWN_SECONDS = 60
JOB_SECONDS = 15 * 60
IMAGE_RES = (64, 2048)
MAX_SIZE_M = 4.0
KEY = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # git for the clone; a compiler for scikit-image 0.19 (no cp311 wheel); the X11/GL client
    # libraries the bpy wheel links against even when it never opens a window.
    .apt_install("git", "build-essential", "libgl1", "libegl1", "libglib2.0-0", "libsm6",
                 "libxrender1", "libxi6", "libxkbcommon0", "libxxf86vm1", "libxfixes3",
                 "libxext6", "libx11-6")
    .run_commands(f"git clone {INFINIGEN_REPO} {INFINIGEN_DIR}",
                  f"git -C {INFINIGEN_DIR} checkout {INFINIGEN_COMMIT}")
    # The minimal install: no terrain, no OpenGL ground truth — the sim factories need neither.
    .run_commands(f"cd {INFINIGEN_DIR} && pip install --no-cache-dir -e '.[sim]'",
                  env={"INFINIGEN_MINIMAL_INSTALL": "True"})
    .env({"INFINIGEN_MINIMAL_INSTALL": "True"})
    .add_local_file(WORKER_LOCAL, WORKER)
)
app = modal.App(APP_NAME)


# ── requests ─────────────────────────────────────────────────────────────────────────────────────


def parse_request(request: dict) -> dict:
    """The job and texture size, checked: ValueError says what is wrong. Unknown keys are
    ignored so a newer client can talk to this function."""
    if not isinstance(request, dict) or not isinstance(request.get("job"), dict):
        raise ValueError("the request must be a dict with a 'job' dict")
    job = request["job"]
    key = job.get("key")
    if not isinstance(key, str) or not KEY.match(key):
        raise ValueError("job.key must be 1-128 of A-Z a-z 0-9 . _ -")
    if job.get("factory") not in FACTORIES:
        raise ValueError(f"job.factory must be one of {', '.join(FACTORIES)}")
    try:
        size = [float(v) for v in job.get("size")]
        seed = int(job.get("seed", 0))
        image_res = int(request.get("image_res", 512))
    except (TypeError, ValueError):
        raise ValueError("job.size (three numbers), job.seed and image_res must be numbers") \
            from None
    if len(size) != 3 or not all(0.01 <= v <= MAX_SIZE_M for v in size):
        raise ValueError(f"job.size must be three sides of 0.01-{MAX_SIZE_M:g} m")
    lo, hi = IMAGE_RES
    if not lo <= image_res <= hi:
        raise ValueError(f"image_res must be {lo}..{hi}")
    options = job.get("options") or {}
    if not isinstance(options, dict) or not all(
            isinstance(k, str) and isinstance(v, (str, int, float, bool)) for k, v in
            options.items()):
        raise ValueError("job.options must map names to plain values")
    return {"job": {"key": key, "factory": job["factory"], "size": size, "seed": seed,
                    "options": options}, "image_res": image_res}


def zip_export(folder: Path) -> bytes:
    """The job's folder zipped, without the maps the conversion never reads."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(Path(folder).rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".png" and not path.name.endswith(KEEP_TEXTURES):
                continue
            zf.write(path, path.relative_to(folder).as_posix())
    return buffer.getvalue()


def run_job(request: dict, *, python: str = sys.executable, worker: str = WORKER,
            timeout: float = JOB_SECONDS) -> dict:
    """One job through the worker in a fresh process (what `generate` does in the container)."""
    started = time.monotonic()
    try:
        parsed = parse_request(request)
    except ValueError as exc:
        key = request.get("job", {}).get("key") if isinstance(request, dict) and isinstance(
            request.get("job"), dict) else None
        return {"key": key, "ok": False, "error": f"bad request: {exc}"}
    job = parsed["job"]
    with tempfile.TemporaryDirectory(prefix="infinigen-") as tmp:
        jobs = Path(tmp) / "jobs.json"
        out = Path(tmp) / "out"
        jobs.write_text(json.dumps({"image_res": parsed["image_res"], "jobs": [job]}))
        env = {**os.environ, "BLENDER_USER_RESOURCES": str(Path(tmp) / "blender")}
        try:
            proc = subprocess.run([python, worker, str(jobs), str(out)], capture_output=True,
                                  text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"key": job["key"], "ok": False,
                    "error": f"no asset within {timeout:g} s", "seconds": timeout}
        folder = out / job["key"]
        try:
            result = json.loads((folder / "result.json").read_text())
        except (OSError, ValueError):
            tail = (proc.stderr or proc.stdout or "")[-600:]
            return {"key": job["key"], "ok": False, "seconds": time.monotonic() - started,
                    "error": f"the worker wrote no result (exit {proc.returncode}): {tail}"}
        trace = result.pop("traceback", "")
        if not result.get("ok"):
            error = str(result.get("error") or "failed")
            return {"key": job["key"], "ok": False, "seconds": time.monotonic() - started,
                    "error": f"{error} | {trace[-400:]}" if trace else error, "result": result}
        return {"key": job["key"], "ok": True, "seconds": round(time.monotonic() - started, 3),
                "zip_b64": base64.b64encode(zip_export(folder)).decode("ascii"),
                "result": result}


@app.function(image=image, cpu=CPUS, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS,
              max_containers=MAX_CONTAINERS, scaledown_window=SCALEDOWN_SECONDS)
def generate(request: dict) -> dict:
    """One articulated asset: an Infinigen sim factory sized to ``request["job"]``."""
    return run_job(request)
