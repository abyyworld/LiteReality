"""Dense sensor depth for gsplat's trainer: the photo's depth map in, two extra losses out.

Copied next to gsplat's `examples/simple_trainer.py` at image build (deploy/modal/splat/app.py)
and imported by the patched trainer (patch_trainer.py). Only used when the trainer runs with
`--dense-depth` or `--normal-lambda > 0`; without them the trainer is gsplat's own.

The dataset tar then carries `depths/<image name>.png` beside `images/<image name>`: uint16
millimetres at the training resolution, 0 where nothing was measured (PromptDA's refinement of
the phone's LiDAR, models/splat/promptda.py). `depths/<image stem>.png` is accepted too.

The loss maths is written once for both numpy and torch (plain slicing and arithmetic only), so
tests/test_splat_trainer_patch.py checks on a laptop the very code the H100 runs.
"""

from __future__ import annotations

import os

import numpy as np

# A pixel whose depth jumps by more than this fraction to a neighbour is an occlusion edge in the
# sensor depth: its finite-difference normal is meaningless there, so it gets none.
MAX_NORMAL_JUMP = 0.1


def depth_path(data_dir: str, image_name: str) -> str | None:
    """Where the photo's dense depth is in the dataset, or None when it has none."""
    for name in (image_name + ".png", os.path.splitext(image_name)[0] + ".png"):
        path = os.path.join(data_dir, "depths", name)
        if os.path.isfile(path):
            return path
    return None


def load_depth(path: str | None, size: tuple[int, int]) -> np.ndarray:
    """The depth map in metres (float32, H x W, 0 = none) at `size` = (width, height).

    A map of another size is resized nearest-neighbour (never blended across an edge); a missing
    or unreadable one is all zeros, so that photo trains on colour alone."""
    import cv2

    width, height = size
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED) if path else None
    if depth is None or depth.ndim != 2:
        return np.zeros((height, width), np.float32)
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.float32) / 1000.0


class DenseDepthData:
    """gsplat's training Dataset with each sample's dense depth added as ``"dense_depth"``.

    Wraps rather than subclasses, so this file needs no pycolmap: anything else the trainer asks
    of the dataset (``len``, ``parser``, ``indices``) is the wrapped one's."""

    def __init__(self, base, data_dir: str):
        self.base = base
        self.data_dir = data_dir
        self.missing = 0

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name):
        if name == "base":  # not set yet (unpickling in a spawned worker): no recursion
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getitem__(self, item: int) -> dict:
        import cv2
        import torch

        data = self.base[item]
        parser = self.base.parser
        index = self.base.indices[item]
        camera_id = parser.camera_ids[index]
        path = depth_path(self.data_dir, parser.image_names[index])
        if path is None:
            self.missing += 1
            if self.missing == 1:
                print(f"[dense depth] none for {parser.image_names[index]}: colour only there")
        # The photo as the trainer sees it: undistorted (cropped to its ROI) at imsize, then cut
        # to the random patch if any. The depth follows the same steps.
        params = parser.params_dict[camera_id]
        if len(params) > 0 and path is not None:
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            mapx, mapy = parser.mapx_dict[camera_id], parser.mapy_dict[camera_id]
            if raw is not None and raw.shape == mapx.shape:
                raw = cv2.remap(raw, mapx, mapy, cv2.INTER_NEAREST)
                x, y, w, h = parser.roi_undist_dict[camera_id]
                depth = raw[y: y + h, x: x + w].astype(np.float32) / 1000.0
            else:
                depth = load_depth(path, parser.imsize_dict[camera_id])
        else:
            depth = load_depth(path, parser.imsize_dict[camera_id])
        height, width = data["image"].shape[:2]
        if depth.shape != (height, width):
            # A random patch: its offset is how far the principal point moved.
            K0 = parser.Ks_dict[camera_id]
            x = int(round(float(K0[0, 2] - data["K"][0, 2])))
            y = int(round(float(K0[1, 2] - data["K"][1, 2])))
            depth = depth[y: y + height, x: x + width]
            if depth.shape != (height, width):
                depth = np.zeros((height, width), np.float32)
        data["dense_depth"] = torch.from_numpy(np.ascontiguousarray(depth))
        return data


def _pixel_grid(depth):
    """Pixel-centre coordinates (u, v), each shaped like one depth map, in depth's library."""
    height, width = depth.shape[-2:]
    if isinstance(depth, np.ndarray):  # (numpy 2 arrays have a .device too: ask the type)
        v, u = np.meshgrid(np.arange(height, dtype=depth.dtype),
                           np.arange(width, dtype=depth.dtype), indexing="ij")
    else:  # torch
        import torch

        v, u = torch.meshgrid(torch.arange(height, device=depth.device, dtype=depth.dtype),
                              torch.arange(width, device=depth.device, dtype=depth.dtype),
                              indexing="ij")
    # gsplat projects a point onto pixel (u, v) when it lands in [u, u+1): centre u + 0.5.
    return u + 0.5, v + 0.5


def _masked_mean(values, mask):
    """Mean of `values` where `mask` holds; 0 when it never does (numpy or torch)."""
    count = mask.sum()
    return (values * mask).sum() / (count + (count == 0))


def depth_normals(depth, Ks, max_jump: float | None = None):
    """Camera-space normals of depth maps by central differences, and where they are valid.

    depth: B x H x W z-depth in metres (0 = none); Ks: B x 3 x 3. Returns (nx, ny, nz), each
    B x (H-2) x (W-2) for the interior pixels, unit length and facing the camera (nz < 0 in
    OpenCV's x right, y down, z forward), and the mask of pixels whose four neighbours and self
    all have depth (and, given `max_jump`, none more than that fraction away from the centre)."""
    u, v = _pixel_grid(depth)
    fx, fy = Ks[:, 0, 0][:, None, None], Ks[:, 1, 1][:, None, None]
    cx, cy = Ks[:, 0, 2][:, None, None], Ks[:, 1, 2][:, None, None]
    X, Y, Z = (u - cx) / fx * depth, (v - cy) / fy * depth, depth
    # Central differences: along the row (a) and down the column (b).
    ax = X[:, 1:-1, 2:] - X[:, 1:-1, :-2]
    ay = Y[:, 1:-1, 2:] - Y[:, 1:-1, :-2]
    az = Z[:, 1:-1, 2:] - Z[:, 1:-1, :-2]
    bx = X[:, 2:, 1:-1] - X[:, :-2, 1:-1]
    by = Y[:, 2:, 1:-1] - Y[:, :-2, 1:-1]
    bz = Z[:, 2:, 1:-1] - Z[:, :-2, 1:-1]
    # b x a (not a x b) so the normal points back at the camera.
    nx, ny, nz = by * az - bz * ay, bz * ax - bx * az, bx * ay - by * ax
    # The epsilon is inside the root: sqrt's gradient at exactly 0 (no depth around a pixel) is
    # infinite, and masked out or not, inf x 0 would put NaN into every splat's gradient.
    norm = (nx * nx + ny * ny + nz * nz + 1e-20) ** 0.5
    centre = depth[:, 1:-1, 1:-1]
    sides = (depth[:, 1:-1, 2:], depth[:, 1:-1, :-2], depth[:, 2:, 1:-1], depth[:, :-2, 1:-1])
    valid = centre > 0
    for side in sides:
        valid = valid & (side > 0)
        if max_jump is not None:
            valid = valid & (abs(side - centre) <= max_jump * centre)
    return (nx / norm, ny / norm, nz / norm), valid


def dense_depth_l1(pred, dense):
    """Mean |rendered - sensor| depth in metres over the pixels both have (B x H x W each).

    Where nothing renders (expected depth 0) there is nothing for depth to move, so it is left to
    the colour loss rather than counted here."""
    valid = (dense > 0) & (pred > 0)
    return _masked_mean(abs(pred - dense), valid)


def depth_normal_loss(pred, dense, Ks, max_jump: float = MAX_NORMAL_JUMP):
    """DN-Splatter's normal term: 1 - cos(normal of the rendered depth, normal of the sensor's),
    both from finite differences in camera space, averaged over pixels both have normals.

    The sensor's occlusion edges are left out (`max_jump`); the render's are not, so a floater
    that tears the rendered surface where the sensor saw a smooth one is penalised."""
    (px, py, pz), pred_ok = depth_normals(pred, Ks)
    (gx, gy, gz), dense_ok = depth_normals(dense, Ks, max_jump=max_jump)
    return _masked_mean(1.0 - (px * gx + py * gy + pz * gz), pred_ok & dense_ok)
