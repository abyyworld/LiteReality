"""Eyeful Tower scene -> a COLMAP dataset gsplat reads, built entirely inside a Modal container.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run modal run --env main \
        scripts/eyeful/prepare_eyeful.py --scene office1b

Nothing goes through this Mac: the container pulls the public S3 bucket (no credentials,
--no-sign-request equivalent) straight into the Modal Volume `litereality-eyeful`, then builds:

    /<scene>/raw/                      as published (S3 layout kept):
        cameras.json splits.json md5sums.txt
        colmap/sparse/0/{cameras,images,points3D}.bin   Metashape's COLMAP export, full-res px
        images-jpeg-2k/<cam>/<cam>_DSCxxxx.jpg          original (distorted) 2K JPEGs, reference
    /<scene>/gsplat_2k/                what the trainer reads (data_factor 1, test list below):
        images/<cam>_DSCxxxx.jpg       colmap/images_4 = the export's undistorted images at 1/4
        sparse/0/cameras.bin           PINHOLE per sensor, rescaled to images/ sizes
        sparse/0/images.bin            same poses; 2D observations rescaled the same way
        sparse/0/points3D.bin          Metashape tie points, unchanged
        test_images.txt train_images.txt   from splits.json (v2: every camera-17 image is test)
    /<scene>/prepare_summary.json      counts, sizes, camera models, frame check, timings

Re-running skips files already on the Volume with the right size.
"""

from __future__ import annotations

import json
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

BUCKET = "fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15"
ROOT = "EyefulTower"
VOLUME_NAME = "litereality-eyeful"
MOUNT = Path("/vol")

app = modal.App("litereality-eyeful-prepare")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "boto3==1.35.36", "numpy==1.26.4", "pillow==10.4.0"
)

CAMERA_PARAMS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
                 3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12)}


# ---------------------------------------------------------------- COLMAP binary I/O (numpy)
def read_cameras(path: Path) -> dict:
    data = path.read_bytes()
    (n,), off, cams = struct.unpack_from("<Q", data, 0), 8, {}
    for _ in range(n):
        cid, model, w, h = struct.unpack_from("<iiQQ", data, off)
        off += 24
        k = CAMERA_PARAMS[model][1]
        params = list(struct.unpack_from("<%dd" % k, data, off))
        off += 8 * k
        cams[cid] = {"model": model, "width": w, "height": h, "params": params}
    return cams


def write_cameras(path: Path, cams: dict) -> None:
    out = [struct.pack("<Q", len(cams))]
    for cid in sorted(cams):
        c = cams[cid]
        out.append(struct.pack("<iiQQ", cid, c["model"], c["width"], c["height"]))
        out.append(struct.pack("<%dd" % len(c["params"]), *c["params"]))
    path.write_bytes(b"".join(out))


def read_images(path: Path) -> list:
    import numpy as np

    data = path.read_bytes()
    (n,), off, ims = struct.unpack_from("<Q", data, 0), 8, []
    pt = np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack_from("<i7di", data, off)
        off += 64
        end = data.index(b"\x00", off)
        name = data[off:end].decode()
        off = end + 1
        (m,) = struct.unpack_from("<Q", data, off)
        off += 8
        pts = np.frombuffer(data, dtype=pt, count=m, offset=off).copy()
        off += 24 * m
        ims.append({"id": iid, "q": (qw, qx, qy, qz), "t": (tx, ty, tz), "camera_id": cid,
                    "name": name, "points2D": pts})
    return ims


def write_images(path: Path, ims: list) -> None:
    out = [struct.pack("<Q", len(ims))]
    for im in ims:
        out.append(struct.pack("<i7di", im["id"], *im["q"], *im["t"], im["camera_id"]))
        out.append(im["name"].encode() + b"\x00")
        out.append(struct.pack("<Q", len(im["points2D"])))
        out.append(im["points2D"].tobytes())
    path.write_bytes(b"".join(out))


def qvec_to_R(q):
    import numpy as np

    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def umeyama(src, dst):
    """Similarity s, R, t with dst ~ s R src + t."""
    import numpy as np

    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / a.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


# ---------------------------------------------------------------- download
def _wanted(key: str, scene: str) -> str | None:
    """Relative path under raw/ (or gsplat_2k/images/ for images_4) for keys we need."""
    rel = key[len(f"{ROOT}/{scene}/"):]
    if rel in ("cameras.json", "splits.json", "md5sums.txt", "colmap/md5sums.txt"):
        return "raw/" + rel
    if rel.startswith("colmap/sparse/0/") and rel.endswith(".bin"):
        return "raw/" + rel
    if rel.startswith("colmap/images_4/") and rel.endswith(".jpg"):
        return "gsplat_2k/images/" + rel.split("/")[-1]
    if rel.startswith("images-jpeg-2k/") and rel.endswith(".jpg"):
        return "raw/" + rel
    return None


@app.function(image=image, volumes={MOUNT: volume}, cpu=4, memory=4096, timeout=3600)
def prepare(scene: str, threads: int = 64) -> dict:
    import boto3
    import numpy as np
    from botocore import UNSIGNED
    from botocore.config import Config
    from PIL import Image

    t0 = time.time()
    base = MOUNT / scene
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                          max_pool_connections=threads, retries={"max_attempts": 8}))
    jobs = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=f"{ROOT}/{scene}/"):
        for obj in page.get("Contents", []):
            rel = _wanted(obj["Key"], scene)
            if rel:
                jobs.append((obj["Key"], base / rel, obj["Size"]))
    total_bytes = sum(j[2] for j in jobs)
    print(f"{len(jobs)} files, {total_bytes / 1e9:.2f} GB to fetch", flush=True)

    def fetch(job):
        key, dst, size = job
        if dst.exists() and dst.stat().st_size == size:
            return 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path("/tmp/dl") / key.replace("/", "_")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(tmp))
        shutil.move(str(tmp), dst)
        return size

    done = 0
    with ThreadPoolExecutor(threads) as pool:
        for i, n in enumerate(pool.map(fetch, jobs)):
            done += n
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}  {done / 1e9:.2f} GB  {time.time() - t0:.0f}s", flush=True)
    t_dl = time.time() - t0
    print(f"downloaded {done / 1e9:.2f} GB in {t_dl:.0f}s", flush=True)
    volume.commit()

    # ------------------------------------------------------------ build gsplat_2k/sparse/0
    raw_sparse = base / "raw/colmap/sparse/0"
    out = base / "gsplat_2k"
    (out / "sparse/0").mkdir(parents=True, exist_ok=True)
    cams = read_cameras(raw_sparse / "cameras.bin")
    ims = read_images(raw_sparse / "images.bin")
    img_dir = out / "images"
    on_disk = sorted(p.name for p in img_dir.iterdir())
    names = {im["name"] for im in ims}
    missing_files = sorted(names - set(on_disk))
    extra_files = sorted(set(on_disk) - names)

    # per-camera image size on disk (must be uniform within a camera)
    sizes_by_cam: dict[int, set] = {}
    for im in ims:
        if im["name"] in missing_files:
            continue
        with Image.open(img_dir / im["name"]) as f:
            sizes_by_cam.setdefault(im["camera_id"], set()).add(f.size)
    new_cams, cam_report = {}, {}
    for cid, c in cams.items():
        if cid not in sizes_by_cam:
            continue
        sz = sizes_by_cam[cid]
        assert len(sz) == 1, f"camera {cid} has several image sizes {sz}"
        (w, h), = sz
        sx, sy = w / c["width"], h / c["height"]
        name, k = CAMERA_PARAMS[c["model"]]
        assert name in ("PINHOLE", "SIMPLE_PINHOLE"), f"camera {cid} is {name}, expected pinhole"
        p = c["params"]
        if name == "PINHOLE":
            newp = [p[0] * sx, p[1] * sy, p[2] * sx, p[3] * sy]
        else:
            newp = [p[0] * sx, p[1] * sx, p[2] * sy]
        new_cams[cid] = {"model": c["model"], "width": w, "height": h, "params": newp}
        cam_report[cid] = {"model": name, "full_res": [c["width"], c["height"]], "2k": [w, h],
                           "scale": [round(sx, 6), round(sy, 6)],
                           "params_2k": [round(v, 3) for v in newp]}
    write_cameras(out / "sparse/0/cameras.bin", new_cams)
    for im in ims:
        c, n = cams[im["camera_id"]], new_cams.get(im["camera_id"])
        if n is not None and len(im["points2D"]):
            im["points2D"]["x"] *= n["width"] / c["width"]
            im["points2D"]["y"] *= n["height"] / c["height"]
    write_images(out / "sparse/0/images.bin", ims)
    shutil.copyfile(raw_sparse / "points3D.bin", out / "sparse/0/points3D.bin")
    (n_points,) = struct.unpack("<Q", (raw_sparse / "points3D.bin").read_bytes()[:8])

    # ------------------------------------------------------------ split
    splits = json.loads((base / "raw/splits.json").read_text())
    def to_name(s):  # "17/17_DSC0001" -> "17_DSC0001.jpg"
        return s.split("/")[-1] + ".jpg"

    test = sorted(to_name(s) for s in splits["test"])
    train = sorted(to_name(s) for s in splits["train"])
    (out / "test_images.txt").write_text("\n".join(test) + "\n")
    (out / "train_images.txt").write_text("\n".join(train) + "\n")
    test_cams = sorted({n.split("_")[0] for n in test})

    # ------------------------------------------------------------ raw 2K JPEG check + frame check
    jpeg2k = base / "raw/images-jpeg-2k"
    raw_counts, raw_sizes = {}, {}
    for cam_dir in sorted(jpeg2k.iterdir()):
        files = sorted(cam_dir.glob("*.jpg"))
        raw_counts[cam_dir.name] = len(files)
        with Image.open(files[0]) as f:
            raw_sizes[cam_dir.name] = f.size
    krt = json.loads((base / "raw/cameras.json").read_text())["KRT"]
    by_name = {k["cameraId"].split("/")[-1] + ".jpg": k for k in krt}
    src, dst = [], []
    for im in ims:
        k = by_name.get(im["name"])
        if k is None:
            continue
        T = np.array(k["T"], dtype=float).reshape(4, 4).T     # column-major world-to-camera
        dst.append(-T[:3, :3].T @ T[:3, 3])
        R = qvec_to_R(im["q"])
        src.append(-R.T @ np.array(im["t"]))
    src, dst = np.array(src), np.array(dst)
    s, R, t = umeyama(src, dst)
    resid = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    dist = {k["distortionModel"] for k in krt}
    dcoef = np.array([k["distortion"] for k in krt if k["cameraId"].startswith("17/")])

    summary = {
        "scene": scene,
        "volume": VOLUME_NAME,
        "dataset_dir": f"/{scene}/gsplat_2k",
        "raw_dir": f"/{scene}/raw",
        "files_fetched": len(jobs),
        "bytes_fetched": total_bytes,
        "download_seconds": round(t_dl, 1),
        "colmap": {
            "cameras": len(cams), "images_in_model": len(ims), "images_on_disk": len(on_disk),
            "missing_files": missing_files[:20], "n_missing_files": len(missing_files),
            "extra_files": extra_files[:20], "n_extra_files": len(extra_files),
            "points3D": n_points, "camera_models": sorted({v["model"] for v in cam_report.values()}),
            "per_camera": cam_report,
        },
        "split": {
            "train": len(train), "test": len(test), "test_cameras": test_cams,
            "test_missing_from_model": sorted(set(test) - names)[:20],
            "train_missing_from_model": sorted(set(train) - names)[:20],
            "in_model_but_in_neither_split": sorted(names - set(test) - set(train))[:20],
            "n_in_model_but_in_neither_split": len(names - set(test) - set(train)),
        },
        "raw_jpeg_2k": {"per_camera_count": raw_counts, "per_camera_size": raw_sizes,
                        "total": sum(raw_counts.values())},
        "cameras_json": {
            "entries": len(krt), "distortion_models": sorted(dist),
            "cam17_distortion_mean_abs_[k1,k2,p1,p2,k3]": np.abs(dcoef).mean(0).round(5).tolist()
            if len(dcoef) else None,
            "colmap_to_cameras_json_similarity": {
                "scale": float(s), "rms_center_residual_m": float(np.sqrt((resid ** 2).mean())),
                "max_center_residual_m": float(resid.max()), "matched": len(src),
                "R": R.round(6).tolist(), "t": t.round(6).tolist()},
        },
        "seconds_total": round(time.time() - t0, 1),
    }
    (base / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    return summary


@app.local_entrypoint()
def main(scene: str = "office1b", out: str = ""):
    summary = prepare.remote(scene)
    text = json.dumps(summary, indent=2)
    print(text)
    if out:
        Path(out).write_text(text)
