"""Shared helpers for the RealityScan experiments: its COLMAP text export, ARKit poses of the Stray
pool import, rotations, and the RS -> ARKit-world similarity files.

Conventions: every camera here is camera -> world with an OpenCV camera (x right, y down, z ahead),
as COLMAP. ARKit's `cameraPoseARFrame` is camera -> world with ARKit's camera (x right, y up,
z back): times FLIP. The room frame the splat trainer and score.py use is ARKit's world turned Z up
(`rgbd.ARKIT_TO_Z_UP`).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
POOL = Path("~/LiteReality-scans/stray_atrium_pool").expanduser()
FLIP = np.diag([1.0, -1.0, -1.0])
ARKIT_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def qvec_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


@dataclass
class Camera:
    model: str
    w: int
    h: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        """3x3, COLMAP convention (first pixel's centre at 0.5)."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        else:
            fx, fy, cx, cy = p[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    @property
    def dist(self) -> np.ndarray:
        """OpenCV's distortion vector (k1, k2, p1, p2, k3, k4, k5, k6) for cv2.undistort."""
        p = self.params
        if self.model == "FULL_OPENCV":
            return np.asarray(p[4:12], float)
        if self.model == "OPENCV":
            return np.r_[p[4:8], 0.0]
        if self.model in ("PINHOLE", "SIMPLE_PINHOLE"):
            return np.zeros(5)
        if self.model == "SIMPLE_RADIAL":
            return np.array([p[3], 0, 0, 0, 0.0])
        if self.model == "RADIAL":
            return np.array([p[3], p[4], 0, 0, 0.0])
        raise ValueError(f"camera model {self.model}")


@dataclass
class Image:
    name: str
    R: np.ndarray        # camera -> world rotation (OpenCV camera)
    C: np.ndarray        # camera centre in world
    camera_id: int
    xy: np.ndarray       # (n, 2) observations (COLMAP pixel convention)
    point_ids: np.ndarray  # (n,) -1 = none


def read_model(sparse: Path) -> tuple[dict[str, Image], dict[int, Camera], dict]:
    """COLMAP text model -> (images by name, cameras by id, points: {"id","xyz","rgb","err",
    "track": list of (image_id array)} as arrays)."""
    cams = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            p = line.split()
            cams[int(p[0])] = Camera(p[1], int(p[2]), int(p[3]), np.array(p[4:], float))
    imgs: dict[str, Image] = {}
    by_id: dict[int, str] = {}
    lines = [ln for ln in (sparse / "images.txt").read_text().splitlines()
             if not ln.startswith("#")]
    for head, obs in zip(lines[0::2], lines[1::2]):
        p = head.split()
        if len(p) < 10:
            continue
        Rw2c = qvec_to_R([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        o = np.array(obs.split(), float).reshape(-1, 3) if obs.strip() else np.zeros((0, 3))
        imgs[p[9]] = Image(p[9], Rw2c.T, -Rw2c.T @ t, int(p[8]), o[:, :2],
                           o[:, 2].astype(np.int64))
        by_id[int(p[0])] = p[9]
    ids, xyz, rgb, err, tracks = [], [], [], [], []
    for line in (sparse / "points3D.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        ids.append(int(p[0]))
        xyz.append([float(v) for v in p[1:4]])
        rgb.append([int(v) for v in p[4:7]])
        err.append(float(p[7]))
        tr = np.array(p[8:], np.int64).reshape(-1, 2)
        tracks.append(tr)
    points = {"id": np.array(ids, np.int64), "xyz": np.array(xyz, float).reshape(-1, 3),
              "rgb": np.array(rgb, np.uint8).reshape(-1, 3), "err": np.array(err, float),
              "track": tracks, "image_names": by_id}
    return imgs, cams, points


def arkit_pool(names, pool: Path = POOL) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """name -> (R c2w OpenCV, centre, K) from the pool import (atrium_1059's ARKit world)."""
    out = {}
    for n in names:
        j = pool / (Path(n).stem + ".json")
        if j.exists():
            d = json.loads(j.read_text())
            m = np.array(d["cameraPoseARFrame"], float).reshape(4, 4)
            out[n] = (m[:3, :3] @ FLIP, m[:3, 3], np.array(d["intrinsics"], float).reshape(3, 3))
    return out


def save_sim3(path: Path, T: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"T_rs_to_arkit": [[float(v) for v in r] for r in T],
                                "scale": float(np.cbrt(np.linalg.det(T[:3, :3]))), **meta},
                               indent=1))


def load_sim3(path: Path) -> np.ndarray:
    return np.array(json.loads(path.read_text())["T_rs_to_arkit"], float)


def stats(v) -> dict:
    v = np.asarray(v, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(v.mean()),
            "rmse": float(np.sqrt((v ** 2).mean())), "p90": float(np.percentile(v, 90)),
            "max": float(v.max())}
