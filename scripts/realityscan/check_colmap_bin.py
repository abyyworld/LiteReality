"""Sanity check of a COLMAP binary model written by build_dataset.py: parse it independently and
report each photo's reprojection error of its observations (pose + K + undistortion consistency).

    uv run --frozen python scripts/realityscan/check_colmap_bin.py ~/LiteReality-scans/rs_stray/rs_w960
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

from rs_common import qvec_to_R


def read(root: Path):
    sp = root / "sparse/0"
    b = (sp / "cameras.bin").read_bytes()
    n = struct.unpack_from("<Q", b)[0]
    cams, off = {}, 8
    for _ in range(n):
        cid, model, w, h, fx, fy, cx, cy = struct.unpack_from("<iiQQdddd", b, off)
        off += struct.calcsize("<iiQQdddd")
        cams[cid] = (w, h, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    b = (sp / "images.bin").read_bytes()
    n = struct.unpack_from("<Q", b)[0]
    imgs, off = [], 8
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack_from("<i4d3di", b, off)
        off += struct.calcsize("<i4d3di")
        e = b.index(b"\0", off)
        name = b[off:e].decode()
        off = e + 1
        k = struct.unpack_from("<Q", b, off)[0]
        off += 8
        obs = np.frombuffer(b, dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")], count=k, offset=off)
        off += 24 * k
        imgs.append((name, qvec_to_R([qw, qx, qy, qz]), np.array([tx, ty, tz]), cid, obs))
    b = (sp / "points3D.bin").read_bytes()
    n = struct.unpack_from("<Q", b)[0]
    pts, off = {}, 8
    for _ in range(n):
        pid, x, y, z, r, g, bb, err, tl = struct.unpack_from("<Q3d3BdQ", b, off)
        off += struct.calcsize("<Q3d3BdQ") + 8 * tl
        pts[pid] = np.array([x, y, z])
    return cams, imgs, pts


def main() -> None:
    root = Path(sys.argv[1]).expanduser()
    cams, imgs, pts = read(root)
    errs, with_obs = [], 0
    for name, R, t, cid, obs in imgs:
        if not len(obs):
            continue
        with_obs += 1
        X = np.array([pts[int(i)] for i in obs["id"]])
        c = X @ R.T + t
        K = cams[cid][2]
        uv = (c @ K.T)[:, :2] / c[:, 2:3]
        e = np.linalg.norm(uv - np.c_[obs["x"], obs["y"]], axis=1)
        errs.append(np.median(e))
    print(f"{len(imgs)} images ({with_obs} with observations), {len(pts)} points, all finite: "
          f"{all(np.isfinite(p).all() for p in pts.values())}; per-image median reprojection px: "
          f"median {np.median(errs):.3f}, max {np.max(errs):.3f}")


if __name__ == "__main__":
    main()
