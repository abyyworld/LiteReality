"""Import a COLMAP export made OUTSIDE our Modal runs (RealityScan GUI on a Windows PC, RealityScan
Linux CLI on the user's machine, or plain COLMAP) of the Stray atrium frames, and turn it into our
pose format next to ARKit's. See results/trainers/realityscan_manual.md for how the export is made.

    uv run --frozen python scripts/compare_trainers/import_colmap_poses.py <export_dir> \\
        [--input ~/LiteReality-scans/rs_stray/rs_input] [--out ~/LiteReality-scans/rs_manual]

<export_dir> is searched for one COLMAP model (cameras/images/points3D, .bin OR .txt, standard
`sparse/0` or flat layout). Image names are matched by basename to the alignment input
(prep_stray.py: frame_XXXXX.jpg, pool names, 29 eval frames included and marked).

Writes <out>/ in the same shape as the Modal runs' scan folder, so the existing scripts take it:

    rs_input                 -> symlink to --input (photos + selection.json)
    rs_out/colmap/sparse/0/  the model as COLMAP TEXT (converted when the export is binary)
    rs_to_arkit.json         Sim(3) RS -> ARKit (pool) world: robust Umeyama on camera centres plus
                             points on each camera's axes (stray_poses.py's fit)
    poses/frame_XXXXX.json   OUR format, one per registered frame: "cameraPoseARFrame" = the RS
                             pose moved into the ARKit world, camera -> world, ARKit/OpenGL camera
                             axes (x right, y up, z back), row-major 4x4, exactly like the import's
                             frame_*.json; "intrinsics" = RS's K (row-major 3x3, ARKit pixel
                             convention = COLMAP cx/cy - 0.5, scaled to the input photo's width);
                             "colmap_camera" = the exported model + params
                             (lens distortion); plus the pool frame's time/stray fields
    pose_vs_arkit.json       per frame and summary: centre error (m), rotation error (deg),
                             fx vs ARKit, components are NOT detected (RS exports one component)

Then, unchanged:
    uv run --frozen python scripts/realityscan/stray_poses.py --scan <out> --tag _manual
    uv run --frozen python scripts/realityscan/build_dataset.py --scan <out> --long 960 --eval-rs
    uv run --frozen --env-file .env python scripts/realityscan/train_rs.py rs_manual_gs \\
        <out>/rs_w960.tar.gz --trainer gsplat
    uv run --frozen --with playwright --with lpips==0.1.4 python scripts/realityscan/score_rs.py \\
        eval rs_manual_gs run/realityscan/rs_manual_gs
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/realityscan"))
sys.path.insert(0, str(REPO / "src"))

from rs_common import FLIP, POOL, angle_deg, arkit_pool, read_model, save_sim3, stats  # noqa: E402

# COLMAP camera model id -> (name, number of params)
MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
          4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
          8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}


def find_model(root: Path) -> Path:
    hits = sorted({p.parent for ext in ("bin", "txt") for p in root.rglob(f"images.{ext}")
                   if (p.parent / f"cameras.{ext}").exists()})
    if not hits:
        raise SystemExit(f"no COLMAP model (cameras + images .bin/.txt) under {root}")
    if len(hits) > 1:
        print(f"several models, using the first: {[str(h) for h in hits]}", file=sys.stderr)
    return hits[0]


def _read(f, fmt: str):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def _cstr(f) -> str:
    b = bytearray()
    while (c := f.read(1)) not in (b"\x00", b""):
        b += c
    return b.decode()


def bin_to_text(src: Path, dst: Path) -> None:
    """COLMAP binary model -> text model (the format rs_common.read_model and the scripts read)."""
    dst.mkdir(parents=True, exist_ok=True)
    lines = ["# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    with open(src / "cameras.bin", "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            cid, mid, w, h = _read(f, "iiQQ")
            name, n = MODELS[mid]
            params = _read(f, "d" * n)
            lines.append(" ".join([str(cid), name, str(w), str(h)] + [repr(p) for p in params]))
    (dst / "cameras.txt").write_text("\n".join(lines) + "\n")
    lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# POINTS2D[] as (X, Y, POINT3D_ID)"]
    with open(src / "images.bin", "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            iid, *qt, cid = _read(f, "idddddddi")
            name = _cstr(f)
            n = _read(f, "Q")[0]
            obs = np.frombuffer(f.read(24 * n), dtype=[("x", "<f8"), ("y", "<f8"), ("p", "<i8")])
            lines.append(" ".join([str(iid)] + [repr(v) for v in qt] + [str(cid), name]))
            lines.append(" ".join(f"{float(o['x'])!r} {float(o['y'])!r} {int(o['p'])}" for o in obs))
    (dst / "images.txt").write_text("\n".join(lines) + "\n")
    lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"]
    pts = src / "points3D.bin"
    if pts.exists():
        with open(pts, "rb") as f:
            for _ in range(_read(f, "Q")[0]):
                pid, x, y, z, r, g, b, err = _read(f, "QdddBBBd")
                n = _read(f, "Q")[0]
                tr = _read(f, "ii" * n)
                lines.append(" ".join([str(pid), repr(x), repr(y), repr(z), str(r), str(g), str(b),
                                       repr(err)] + [str(v) for v in tr]))
    (dst / "points3D.txt").write_text("\n".join(lines) + "\n")


def copy_text(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for n in ("cameras.txt", "images.txt", "points3D.txt"):
        if (src / n).exists():
            (dst / n).write_bytes((src / n).read_bytes())
        elif n == "points3D.txt":
            (dst / n).write_text("# no points exported\n")


def rename_to_basenames(sparse: Path, known: set[str]) -> int:
    """RealityScan / COLMAP may store names with a folder prefix or another extension case:
    rewrite images.txt so every name is the matching input basename. Returns unmatched count."""
    by_stem = {Path(k).stem.lower(): k for k in known}
    out, miss = [], 0
    lines = (sparse / "images.txt").read_text().splitlines()
    body = [ln for ln in lines if not ln.startswith("#")]
    for head, obs in zip(body[0::2], body[1::2]):
        p = head.split(maxsplit=9)
        if len(p) == 10:
            stem = Path(p[9].replace("\\", "/")).stem.lower()
            if stem in by_stem:
                p[9] = by_stem[stem]
            else:
                miss += 1
        out += [" ".join(p), obs]
    (sparse / "images.txt").write_text("\n".join([ln for ln in lines if ln.startswith("#")] + out) + "\n")
    return miss


def fit_sim3(imgs, ar, names):
    """stray_poses.py's fit: robust Sim(3) on centres + axis points (orientations count too)."""
    from litereality_agent.models.capture import sim3

    src = np.array([imgs[n].C for n in names])
    dst = np.array([ar[n][1] for n in names])
    s0 = sim3.robust_sim3(src, dst)["scale"]
    aug_s, aug_d = [], []
    for n in names:
        for k, d in ((None, 0.0), (0, 0.5), (1, 0.5), (2, 1.0)):
            aug_s.append(imgs[n].C + (imgs[n].R[:, k] * d / s0 if k is not None else 0))
            aug_d.append(ar[n][1] + (ar[n][0][:, k] * d if k is not None else 0))
    return sim3.robust_sim3(np.array(aug_s), np.array(aug_d))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("export", type=Path, help="folder holding the COLMAP export")
    ap.add_argument("--input", type=Path, default=Path("~/LiteReality-scans/rs_stray/rs_input"),
                    help="the alignment input: images/ + selection.json (prep_stray.py)")
    ap.add_argument("--out", type=Path, default=Path("~/LiteReality-scans/rs_manual"))
    ap.add_argument("--pool", type=Path, default=POOL, help="import holding the ARKit frame_*.json")
    a = ap.parse_args()
    export, inp, out = (p.expanduser().resolve() for p in (a.export, a.input, a.out))
    sel = json.loads((inp / "selection.json").read_text())["frames"]

    model = find_model(export)
    sparse = out / "rs_out/colmap/sparse/0"
    if (model / "images.bin").exists() and not (model / "images.txt").exists():
        bin_to_text(model, sparse)
    else:
        copy_text(model, sparse)
    unmatched = rename_to_basenames(sparse, set(sel))
    link = out / "rs_input"
    if not link.exists():
        link.symlink_to(inp, target_is_directory=True)

    imgs, cams, pts = read_model(sparse)
    ar = arkit_pool(sorted(imgs), a.pool.expanduser())
    names = sorted(n for n in imgs if n in ar)
    if len(names) < 3:
        raise SystemExit(f"only {len(names)} registered frames match ARKit frames "
                         f"({len(imgs)} registered, {unmatched} names unmatched)")
    fit = fit_sim3(imgs, ar, names)
    s, R, t = fit["scale"], fit["R"], fit["t"]
    save_sim3(out / "rs_to_arkit.json", fit["T"],
              {"how": "robust_sim3 of camera centres + axis points onto the pool import's ARKit "
                      "cameras (import_colmap_poses.py)", "frames": len(names),
               "median_residual_m": fit["median_residual"], "export": str(model)})

    from PIL import Image as PImage

    with PImage.open(inp / "images" / names[0]) as f:
        photo_w = f.size[0]               # the input photos' width (1920 for the pool import)
    pose_dir = out / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    per = []
    for n in names:
        im = imgs[n]
        cam = cams[im.camera_id]
        R_w = R @ im.R                       # OpenCV camera -> ARKit world
        C_w = s * R @ im.C + t
        M = np.eye(4)
        M[:3, :3] = R_w @ FLIP               # back to ARKit camera axes (x right, y up, z back)
        M[:3, 3] = C_w
        K = cam.K.copy()
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5                        # COLMAP pixel centres -> ARKit's
        K[:2] *= photo_w / cam.w              # at the input photo's size (an export may be resized)
        src = json.loads((a.pool.expanduser() / (Path(n).stem + ".json")).read_text())
        rec = {"cameraPoseARFrame": [float(v) for v in M.ravel()],
               "intrinsics": [float(v) for v in K.ravel()],
               "colmap_camera": {"model": cam.model, "w": cam.w, "h": cam.h,
                                 "params": [float(v) for v in cam.params]},
               "source": "RealityScan/COLMAP export, Sim(3)-aligned to ARKit (rs_to_arkit.json)",
               "eval": bool(sel.get(n, {}).get("eval"))}
        for k in ("frame_index", "time", "stray"):
            if k in src:
                rec[k] = src[k]
        (pose_dir / (Path(n).stem + ".json")).write_text(json.dumps(rec, indent=1))
        per.append({"name": n, "eval": rec["eval"],
                    "pos_err_m": float(np.linalg.norm(C_w - ar[n][1])),
                    "rot_err_deg": angle_deg(R_w, ar[n][0]),
                    "fx_rs": float(K[0, 0]), "fx_arkit": float(ar[n][2][0, 0])})

    ev = [p for p in per if p["eval"]]
    dst = np.array([ar[n][1] for n in names])
    res = {"export": str(model), "input_frames": len(sel),
           "input_eval": sum(bool(v.get("eval")) for v in sel.values()),
           "registered": len(imgs), "compared": len(names), "unmatched_names": unmatched,
           "eval_registered": len(ev), "n_cameras": len(cams), "rs_points": int(len(pts["xyz"])),
           "scale_rs_to_m": float(s), "sim3_median_residual_m": float(fit["median_residual"]),
           "arkit_extent_m": float(np.linalg.norm(dst - dst.mean(0), axis=1).max()),
           "pos_err_m": stats([p["pos_err_m"] for p in per]),
           "rot_err_deg": stats([p["rot_err_deg"] for p in per]),
           "eval_pos_err_m": stats([p["pos_err_m"] for p in ev]) if ev else None,
           "eval_rot_err_deg": stats([p["rot_err_deg"] for p in ev]) if ev else None,
           "fx_rel_diff_median": float(np.median([p["fx_rs"] / p["fx_arkit"] - 1 for p in per])),
           "frames": per}
    (out / "pose_vs_arkit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=1))
    if len(imgs) < 0.9 * len(sel):
        print(f"WARNING: only {len(imgs)}/{len(sel)} frames registered (one component exported)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
