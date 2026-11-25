"""COLMAP 4 global mapper (GLOMAP, merged into COLMAP 4) on the 693 benchmark frames.

Runs the official COLMAP docker image (colmap/colmap, COLMAP 4.3.0.dev0, BSD-3-Clause, built with
CUDA and ONNX Runtime): the pycolmap-cuda12 4.2.1 wheel is built WITHOUT ONNX, so it cannot run
ALIKED ("ALIKED feature extraction requires ONNX support").

    features : ALIKED n16rot (ONNX, COLMAP 3.13.0 release; ALIKED is BSD-3-Clause)
    matching : LightGlue for ALIKED (ONNX, COLMAP 3.13.0 release; LightGlue is Apache-2.0),
               sequential, overlap 20 (+ quadratic), loop detection with COLMAP's ALIKED vocab tree
    camera   : ONE shared SIMPLE_RADIAL camera, f initialised at 1221.9 px, cx, cy = 540, 960,
               refined by the global bundle adjustment
    mapper   : colmap global_mapper

DETACHED Modal job; all results on the volume litereality-poses under /colmap/<run>/:

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file ~/LiteReality/.env modal run --detach \
        --env main scripts/poses/colmap/run_colmap.py --run aliked_lg

    IN : /atrium/frames.tar
    OUT: progress.json, run.log, models.json, poses_raw.json, sparse/<i>/ (TXT), done.json (last)

fetch.py then writes ~/LiteReality-scans/poses_bench/atrium/poses_colmap.json.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path

import modal

COLMAP_IMAGE = "colmap/colmap:20261109.9263"
GPU = ["L4", "A10G", "T4"]  # first one free; L4/A10G were unavailable for 10+ min on 2026-11-25
FX0 = 1221.9
W, H = 1080, 1920
ALIKED_VOCAB = ("https://github.com/colmap/colmap/releases/download/3.13.0/"
                "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin;"
                "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin;"
                "8b2f9bdc44ca7204d8543bb3adab4c03ba9336c84ef41220b5007991036f075e")
SIFT_VOCAB = ("https://github.com/colmap/colmap/releases/download/3.11.1/"
              "vocab_tree_faiss_flickr100K_words256K.bin;vocab_tree_faiss_flickr100K_words256K.bin;"
              "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc")
CUDNN_LIB = "/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib"

image = (
    modal.Image.from_registry(COLMAP_IMAGE, add_python="3.12")
    # ONNX Runtime's CUDA provider needs cuDNN 9, which the CUDA *runtime* image lacks.
    .pip_install("nvidia-cudnn-cu12==9.10.2.21", "numpy")
    .env({"LD_LIBRARY_PATH": f"{CUDNN_LIB}:/usr/local/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
          # the CUDA 12.9 base image demands driver >= 12.9 via NVIDIA_REQUIRE_CUDA; skip that gate
          "NVIDIA_DISABLE_REQUIRE": "1"})
)
app = modal.App("poses-colmap", image=image)
vol = modal.Volume.from_name("litereality-poses")


@app.function(gpu=GPU, cpu=8, memory=32768, timeout=3 * 3600, volumes={"/vol": vol})
def colmap_job(run_name: str, features: str = "aliked", overlap: int = 20, max_features: int = 4096,
               db_from: str = "", mapper_args: str = ""):
    import traceback

    t = time.time()
    out = Path("/vol/colmap") / run_name
    out.mkdir(parents=True, exist_ok=True)
    try:
        _job(out, features, overlap, max_features, t, db_from, mapper_args)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        (out / "done.json").write_text(json.dumps({"ok": False, "error": tb,
                                                   "container_seconds": round(time.time() - t, 1)}))
        vol.commit()
        raise


def _job(out: Path, features: str, overlap: int, max_features: int, t_container: float,
         db_from: str, mapper_args: str):
    log = open(out / "run.log", "a")
    stages: dict = {}

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    def progress(stage, **kw):
        stages[stage] = round(time.time() - t_container, 1)
        (out / "progress.json").write_text(json.dumps({"stage": stage, "stages": stages, **kw}))
        say(f"{stage} {kw}")
        vol.commit()

    stop = threading.Event()

    def committer():
        while not stop.wait(60):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("commit failed", e, flush=True)

    threading.Thread(target=committer, daemon=True).start()

    def colmap(*args):
        cmd = ["colmap", *map(str, args)]
        say("$ " + shlex.join(cmd))
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            print(line, end="", flush=True)
        log.flush()
        if p.wait() != 0:
            raise RuntimeError(f"colmap {args[0]} exited {p.returncode}")
        return time.time() - t0

    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    say(f"gpu {gpu_name}")
    work = Path("/tmp/work")
    imgs = work / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    with tarfile.open("/vol/atrium/frames.tar") as tf:
        tf.extractall(work / "raw", filter="data")
    for p in sorted((work / "raw").rglob("*.jpg")):
        if not p.name.startswith("._"):  # macOS AppleDouble entries in the tar
            (imgs / p.name).symlink_to(p)
    names = sorted(p.name for p in imgs.iterdir())
    if len(names) != 693:
        raise RuntimeError(f"expected 693 frames, found {len(names)}")
    progress("untarred", n=len(names), first=names[0], last=names[-1])

    db = work / "database.db"
    aliked = features == "aliked"
    t0 = time.time()
    ext = ["feature_extractor", "--database_path", db, "--image_path", imgs,
           "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "SIMPLE_RADIAL",
           "--ImageReader.camera_params", f"{FX0},{W / 2},{H / 2},0",
           "--FeatureExtraction.use_gpu", 1]
    if aliked:
        ext += ["--FeatureExtraction.type", "ALIKED_N16ROT",
                "--AlikedExtraction.max_num_features", max_features]
    else:
        ext += ["--FeatureExtraction.type", "SIFT", "--SiftExtraction.max_num_features", max_features]
    if db_from:  # re-map only: reuse the features + matches of an earlier run
        shutil.copy(Path("/vol/colmap") / db_from / "database.db", db)
        t_ext = 0.0
    else:
        t_ext = colmap(*ext)
    progress("features", seconds=round(t_ext, 1), db_from=db_from)

    match = ["sequential_matcher", "--database_path", db, "--FeatureMatching.use_gpu", 1,
             "--FeatureMatching.type", "ALIKED_LIGHTGLUE" if aliked else "SIFT_BRUTEFORCE",
             "--SequentialMatching.overlap", overlap, "--SequentialMatching.quadratic_overlap", 1,
             "--SequentialMatching.loop_detection", 1,
             "--SequentialMatching.loop_detection_period", 10,
             "--SequentialMatching.loop_detection_num_images", 50,
             "--SequentialMatching.vocab_tree_path", ALIKED_VOCAB if aliked else SIFT_VOCAB]
    if db_from:
        t_match = 0.0
    else:
        t_match = colmap(*match)
        shutil.copy(db, out / "database.db")  # so the mapper can be re-run without re-matching
    progress("matched", seconds=round(t_match, 1))

    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    t_map = colmap("global_mapper", "--database_path", db, "--image_path", imgs,
                   "--output_path", sparse, "--GlobalMapper.ba_refine_focal_length", 1,
                   *shlex.split(mapper_args))
    wall = time.time() - t0
    progress("mapped", seconds=round(t_map, 1))

    import numpy as np

    def quat_to_r(qw, qx, qy, qz):
        return np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

    models = []
    for mdir in sorted(p for p in sparse.iterdir() if p.is_dir()):
        txt = out / "sparse" / mdir.name
        txt.mkdir(parents=True, exist_ok=True)
        colmap("model_converter", "--input_path", mdir, "--output_path", txt, "--output_type", "TXT")
        cams = {}
        for line in (txt / "cameras.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                f = line.split()
                cams[int(f[0])] = {"model": f[1], "w": int(f[2]), "h": int(f[3]),
                                   "params": [float(x) for x in f[4:]]}
        poses = {}
        lines = [x for x in (txt / "images.txt").read_text().splitlines() if not x.startswith("#")]
        for line in lines[::2]:
            f = line.split()
            if len(f) < 10:
                continue
            qw, qx, qy, qz, tx, ty, tz = map(float, f[1:8])
            r = quat_to_r(qw, qx, qy, qz)
            c2w = np.eye(4)
            c2w[:3, :3] = r.T
            c2w[:3, 3] = -r.T @ np.array([tx, ty, tz])
            poses[f[9]] = {"c2w": c2w.tolist(), "camera_id": int(f[8])}
        n_pts = sum(1 for x in (txt / "points3D.txt").read_text().splitlines()
                    if x and not x.startswith("#"))
        reg = sorted(poses)
        models.append({"id": mdir.name, "registered": len(reg), "points3D": n_pts,
                       "first": reg[0] if reg else None, "last": reg[-1] if reg else None,
                       "cameras": cams, "poses": poses})
        say(f"model {mdir.name}: {len(reg)} images, {n_pts} points, cameras {cams}")
    models.sort(key=lambda m: -m["registered"])
    (out / "models.json").write_text(json.dumps(
        [{k: v for k, v in m.items() if k != "poses"} for m in models], indent=1))

    best = models[0] if models else None
    container_s = time.time() - t_container
    raw = {"models": models, "seconds": round(wall, 1), "container_seconds": round(container_s, 1),
           "timing": {"features_s": round(t_ext, 1), "matching_s": round(t_match, 1),
                      "global_mapping_s": round(t_map, 1)},
           "colmap_image": COLMAP_IMAGE, "gpu": gpu_name, "features": features, "overlap": overlap,
           "max_features": max_features, "db_from": db_from, "mapper_args": mapper_args}
    (out / "poses_raw.json").write_text(json.dumps(raw))
    stop.set()
    progress("done", best=best and best["registered"], n_models=len(models))
    (out / "done.json").write_text(json.dumps({
        "ok": True, "models": [(m["id"], m["registered"]) for m in models],
        "container_seconds": round(container_s, 1), "wall_seconds": round(wall, 1)}))
    vol.commit()


@app.local_entrypoint()
def main(run: str = "aliked_lg", features: str = "aliked", overlap: int = 20,
         max_features: int = 4096, db_from: str = "", mapper_args: str = ""):
    call = colmap_job.spawn(run, features, overlap, max_features, db_from, mapper_args)
    print("spawned", call.object_id)


@app.function(gpu=GPU, cpu=2, memory=4096, timeout=600)
def smoke():
    """Does the image start on the GPU, see the driver, and load cuDNN? (modal run ...::smoke_main)"""
    subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv; nvidia-smi | head -4; "
                   "colmap -h | head -1; ls /usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib "
                   "| head -3", shell=True)


@app.local_entrypoint()
def smoke_main():
    smoke.remote()
