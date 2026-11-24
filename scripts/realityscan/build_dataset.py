"""A RealityScan alignment as a splat training set in our convention (models/splat/dataset.py):
a COLMAP BINARY model in the room's frame (atrium_1059's ARKit world turned Z up, metres), the
frame score.py renders the eval frames in.

    images/<name>.jpg     the photos, long side `--long` (960: 960 x 720 Stray / 540 x 960 video;
                          1920: full size), UNDISTORTED with RealityScan's lens (FULL_OPENCV ->
                          PINHOLE; cv2, cropped to valid pixels: alpha 0, same size)
    sparse/0/cameras.bin  one PINHOLE camera per photo
    sparse/0/images.bin   RealityScan's poses moved by <scan>/rs_to_arkit.json (stray_poses.py /
                          video_register.py), OpenCV camera, world -> camera; each photo's
                          observations of the kept points (undistorted, at the training size)
    sparse/0/points3D.bin RealityScan's sparse points, same similarity, within 6 m of a training camera; tracks cut to the seeding
                          photos, points seen by fewer than 2 of them dropped, colours re-sampled
                          from those photos only

The 29 eval frames (Stray) are left out entirely (they were only in the alignment so they get
RealityScan poses). The trainer's own hold-out (every `--test-every`-th photo by name, as
`dataset.build`) is listed with no observations: its pixels never seed the room it is scored on.

`--poses arkit` (Stray only) writes the CONTROL: the same photos, same points, but ARKit's poses
and per-frame K on the original (not undistorted) photos, so a training run differs from the RS
one only in the cameras.

With --eval-rs (Stray, RS poses) it also writes <scan>/eval_rs/: the eval frames undistorted at
full size + {K (OpenCV, first pixel centre 0), pose (camera -> room, OpenGL camera)} each, for
scoring at RealityScan's own eval poses (score_rs.py --poses rs).

    uv run --frozen python scripts/realityscan/build_dataset.py --scan ~/LiteReality-scans/rs_stray --long 960 --eval-rs
    uv run --frozen python scripts/realityscan/build_dataset.py --scan ~/LiteReality-scans/rs_stray --long 960 --poses arkit
    uv run --frozen python scripts/realityscan/build_dataset.py --scan ~/LiteReality-scans/rs_video --long 960
"""

from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

import numpy as np
from rs_common import ARKIT_TO_Z_UP, FLIP, arkit_pool, load_sim3, read_model

MAX_DEPTH = 6.0          # m: points farther from every training camera are dropped
MIN_POINTS = 20          # a seeding photo with fewer observations is dropped (gsplat's depth loss)
JPEG_QUALITY = 95


def undistorter(cam, cv2):
    """(map_x, map_y, newK OpenCV) for one RS camera at full size."""
    K = cam.K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    dist = cam.dist
    size = (cam.w, cam.h)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, 0.0, size)
    mx, my = cv2.initUndistortRectifyMap(K, dist, None, newK, size, cv2.CV_32FC1)
    return mx, my, newK, K, dist


def main() -> None:
    import cv2
    from PIL import Image as PImage

    from litereality_agent.models.splat import dataset as D

    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--long", type=int, default=960)
    ap.add_argument("--poses", choices=("rs", "arkit"), default="rs")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--eval-rs", action="store_true")
    a = ap.parse_args()
    scan = a.scan.expanduser()
    sel = json.loads((scan / "rs_input/selection.json").read_text())["frames"]
    imgs, cams, pts = read_model(scan / "rs_out/colmap/sparse/0")
    T = load_sim3(scan / "rs_to_arkit.json")
    s = float(np.cbrt(np.linalg.det(T[:3, :3])))
    Rw, t = T[:3, :3] / s, T[:3, 3]
    A = ARKIT_TO_Z_UP
    ar = arkit_pool(sorted(imgs)) if a.poses == "arkit" or a.eval_rs else {}
    if a.poses == "arkit":
        missing = [n for n in imgs if n not in ar and not sel[n].get("eval")]
        assert not missing, missing[:5]

    train = sorted(n for n in imgs if not sel[n].get("eval"))
    evals = sorted(n for n in imgs if sel[n].get("eval"))
    id_of = {v: k for k, v in pts["image_names"].items()}       # name -> RS image id
    X_room = (s * pts["xyz"] @ Rw.T + t) @ A.T
    und = {cid: undistorter(c, cv2) for cid, c in cams.items()} if a.poses == "rs" or a.eval_rs \
        else {}

    # Seeding photos: training photos minus the trainer's hold-out.
    held = D._held_out(train, a.test_every)
    seed_ids = {id_of[n] for n in train if n not in held}
    keep = np.array([np.sum(np.isin(tr[:, 0], list(seed_ids))) >= 2 for tr in pts["track"]])
    # Far points (seen through the windows, mismatched) would seed splats kilometres away (the
    # SPZ export refuses > 2 km): keep what lies within MAX_DEPTH m of a training camera.
    from scipy.spatial import cKDTree
    cams_room = np.array([(s * Rw @ imgs[n].C + t) @ A.T for n in train])
    far = cKDTree(cams_room).query(X_room)[0] > MAX_DEPTH
    keep &= ~far
    # Per seeding photo: its observations of kept points (RS point index, original xy).
    pid_index = {int(p): k for k, p in enumerate(pts["id"])}
    obs: dict[str, list] = {}
    for n in train:
        if n in held:
            obs[n] = []
            continue
        im = imgs[n]
        rows = [(pid_index[int(p)], k) for k, p in enumerate(im.point_ids)
                if p >= 0 and int(p) in pid_index and keep[pid_index[int(p)]]]
        obs[n] = rows
    dropped = [n for n in train if n not in held and len(obs[n]) < MIN_POINTS]
    train = [n for n in train if n not in dropped]
    # Seeding photos may have been dropped: recount which points still have 2 observers.
    count = np.zeros(len(pts["xyz"]), int)
    for n in train:
        for k, _ in obs[n]:
            count[k] += 1
    keep &= count >= 2

    variant = f"{a.poses}_w{a.long}"
    out = scan / variant
    files: dict[str, bytes] = {}
    cameras, images = [], []
    col_sum = np.zeros((len(X_room), 3))
    col_n = np.zeros(len(X_room))
    per_image_obs = []
    for image_id, n in enumerate(train, 1):
        im = imgs[n]
        cams[im.camera_id]  # every image's camera is known (KeyError otherwise)
        with PImage.open(scan / "rs_input/images" / n) as f:
            rgb = np.asarray(f.convert("RGB"))
        h0, w0 = rgb.shape[:2]
        k = a.long / max(w0, h0)
        size = (int(round(w0 * k)), int(round(h0 * k)))
        rows = [(kk, j) for kk, j in obs[n] if keep[kk]]
        xy0 = im.xy[[j for _, j in rows]] if rows else np.zeros((0, 2))
        idx = np.array([kk for kk, _ in rows], int)
        if len(idx):
            # Colour from this photo (original pixels at the RS observation).
            u = np.clip(np.round(xy0[:, 0] - 0.5).astype(int), 0, w0 - 1)
            v = np.clip(np.round(xy0[:, 1] - 0.5).astype(int), 0, h0 - 1)
            np.add.at(col_sum, idx, rgb[v, u].astype(float))
            np.add.at(col_n, idx, 1)
        if a.poses == "rs":
            mx, my, newK, K_cv, dist = und[im.camera_id]
            rgb = cv2.remap(rgb, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            Kc = newK.copy()
            Kc[0, 2] += 0.5
            Kc[1, 2] += 0.5
            if len(xy0):
                xy = cv2.undistortPoints((xy0 - 0.5).reshape(-1, 1, 2), K_cv, dist,
                                         P=newK).reshape(-1, 2) + 0.5
            else:
                xy = xy0
            R_room = A @ Rw @ im.R
            C_room = A @ (s * Rw @ im.C + t)
        else:
            Rar, Car, Kar = ar[n]
            Kc = Kar.copy()
            Kc[0, 2] += 0.5          # ARKit: first pixel centre 0; COLMAP: 0.5
            Kc[1, 2] += 0.5
            xy = xy0                 # RS observations on the original photo (not used by gsplat)
            R_room = A @ Rar
            C_room = A @ Car
        small = PImage.fromarray(rgb).resize(size, PImage.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=JPEG_QUALITY)
        files[f"images/{n}"] = buf.getvalue()
        sx, sy = size[0] / w0, size[1] / h0
        fx, fy, cx, cy = Kc[0, 0] * sx, Kc[1, 1] * sy, Kc[0, 2] * sx, Kc[1, 2] * sy
        cameras.append(struct.pack("<iiQQdddd", image_id, 1, *size, fx, fy, cx, cy))
        w2c_R = R_room.T
        w2c_t = -R_room.T @ C_room
        q = D._quat_wxyz(w2c_R)
        head = struct.pack("<i4d3di", image_id, *q, *w2c_t, image_id) + n.encode() + b"\0"
        per_image_obs.append((head, idx, xy * [sx, sy] if len(xy) else xy))

    kept_idx = np.flatnonzero(keep)
    new_id = {int(k): i + 1 for i, k in enumerate(kept_idx)}
    tracks: dict[int, list] = {}
    for image_id, (head, idx, xy) in enumerate(per_image_obs, 1):
        n_obs = len(idx)
        p2d = np.zeros(n_obs, dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
        if n_obs:
            p2d["x"], p2d["y"] = xy[:, 0], xy[:, 1]
            p2d["id"] = [new_id[int(k)] for k in idx]
            for j, k in enumerate(idx):
                tracks.setdefault(int(k), []).append((image_id, j))
        images.append(head + struct.pack("<Q", n_obs) + p2d.tobytes())
    colours = np.where(col_n[:, None] > 0, col_sum / np.maximum(col_n, 1)[:, None],
                       pts["rgb"]).round().clip(0, 255).astype(np.uint8)
    recs = []
    for k in kept_idx:
        tr = np.array(tracks[int(k)], dtype="<i4").reshape(-1)
        recs.append(struct.pack("<Q3d3BdQ", new_id[int(k)], *X_room[k], *colours[k],
                                float(pts["err"][k]), len(tr) // 2) + tr.tobytes())
    count_b = struct.pack("<Q", len(images))
    files["sparse/0/cameras.bin"] = count_b + b"".join(cameras)
    files["sparse/0/images.bin"] = count_b + b"".join(images)
    files["sparse/0/points3D.bin"] = struct.pack("<Q", len(recs)) + b"".join(recs)
    ds = D.Dataset(files=files, images=len(images), points=len(recs), size=size,
                   held_out=sum(1 for n in train if n in held))
    ds.write(out)
    tar = ds.tar()
    (scan / f"{variant}.tar.gz").write_bytes(tar)
    meta = {"variant": variant, "poses": a.poses, "long": a.long, "photos": len(images),
            "held_out": ds.held_out, "points": len(recs), "eval_excluded": len(evals),
            "dropped_few_points": dropped, "far_points_dropped": int(far.sum()), "registered": len(imgs), "input": len(sel),
            "size": list(size), "test_every": a.test_every,
            "images": [n for n in train], "mb": round(len(tar) / 1e6, 1),
            "lens": "undistorted (RealityScan FULL_OPENCV -> PINHOLE)" if a.poses == "rs"
            else "original photos, ARKit K"}
    (scan / f"{variant}.json").write_text(json.dumps(meta, indent=1))
    print({k: v for k, v in meta.items() if k not in ("images",)})

    if a.eval_rs and a.poses == "rs":
        ev_dir = scan / "eval_rs"
        ev_dir.mkdir(exist_ok=True)
        for n in evals:
            im = imgs[n]
            mx, my, newK, _, _ = und[im.camera_id]
            with PImage.open(scan / "rs_input/images" / n) as f:
                rgb = np.asarray(f.convert("RGB"))
            rgb = cv2.remap(rgb, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            PImage.fromarray(rgb).save(ev_dir / n, quality=JPEG_QUALITY)
            pose = np.eye(4)
            pose[:3, :3] = A @ Rw @ im.R @ FLIP          # OpenCV -> OpenGL camera
            pose[:3, 3] = A @ (s * Rw @ im.C + t)
            (ev_dir / (Path(n).stem + ".json")).write_text(json.dumps(
                {"K": newK.tolist(), "pose_room_gl": pose.tolist(), "size": [im_w(rgb), im_h(rgb)],
                 "row": sel[n]["row"]}))
        print(f"{len(evals)} eval frames (RS poses, undistorted) -> {ev_dir}")


def im_w(a):
    return int(a.shape[1])


def im_h(a):
    return int(a.shape[0])


if __name__ == "__main__":
    main()
