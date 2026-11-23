"""Teach gsplat 1.5.3's `examples/simple_trainer.py` the dense-depth losses (litereality_dense.py).

    python patch_trainer.py /opt/gsplat/examples/simple_trainer.py      # at image build

Adds three trainer flags, all off by default so an unflagged run is gsplat's own:

    --dense-depth               load depths/<image>.png per photo; L1 of rendered expected depth
                                against it on the pixels both have (metres)
    --dense-depth-lambda 0.1    that loss's weight
    --normal-lambda 0.0         DN-Splatter's normal term: 1 - cos between the normals of the
                                rendered and the sensor depth (finite differences, camera space)

gsplat's sparse `--depth-loss` (the LiDAR points, in disparity) still works beside them, with one
fix: a point the render does not cover gets no loss instead of a NaN gradient (edit 8; the same
loss wherever the render covers the point). Evaluation (held-out PSNR / SSIM / LPIPS) is not
touched at all.

Each edit is anchored on exact text of the pinned file and must match exactly once: a different
gsplat fails the image build here, loudly, instead of training without the loss.
tests/test_splat_trainer_patch.py applies it to the real v1.5.3 file.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# LiteReality dense depth (deploy/modal/splat/patch_trainer.py)"

# (anchor, replacement): the anchor is kept and the new lines go after it unless the replacement
# restates it.
EDITS: list[tuple[str, str]] = [
    # 1. The helpers, imported beside gsplat's own example utils.
    (
        "from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed\n",
        "from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed\n"
        f"{MARKER}\n"
        "from litereality_dense import (DenseDepthData, dense_depth_l1, depth_normal_loss,\n"
        "                               sparse_disparity)\n",
    ),
    # 2. The flags (tyro: --dense-depth, --dense-depth-lambda, --normal-lambda).
    (
        "    # Weight for depth loss\n    depth_lambda: float = 1e-2\n",
        "    # Weight for depth loss\n    depth_lambda: float = 1e-2\n"
        f"    {MARKER}\n"
        "    # Dense depth loss: L1 of rendered expected depth against depths/<image>.png (metres)\n"
        "    dense_depth: bool = False\n"
        "    # Weight for the dense depth loss\n"
        "    dense_depth_lambda: float = 0.1\n"
        "    # Weight for the normal loss (rendered vs dense depth normals); 0 = off\n"
        "    normal_lambda: float = 0.0\n",
    ),
    # 3. The training set gains each photo's dense depth (the validation set never needs it).
    (
        "            load_depths=cfg.depth_loss,\n        )\n",
        "            load_depths=cfg.depth_loss,\n        )\n"
        f"        {MARKER}\n"
        "        if cfg.dense_depth or cfg.normal_lambda > 0.0:\n"
        "            self.trainset = DenseDepthData(self.trainset, cfg.data_dir)\n",
    ),
    # 4. The batch's dense depth onto the GPU.
    (
        '                depths_gt = data["depths"].to(device)  # [1, M]\n',
        '                depths_gt = data["depths"].to(device)  # [1, M]\n'
        f"            {MARKER}\n"
        '            dense_gt = data["dense_depth"].to(device) if "dense_depth" in data else None\n',
    ),
    # 5. Render expected depth whenever any depth loss needs it.
    (
        '                render_mode="RGB+ED" if cfg.depth_loss else "RGB",\n',
        '                render_mode="RGB+ED"\n'
        "                if (cfg.depth_loss or dense_gt is not None)\n"
        '                else "RGB",\n',
    ),
    # 6. The losses, before gsplat's sparse depth block (which overwrites `depths`).
    (
        "            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda\n",
        "            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda\n"
        f"            {MARKER}\n"
        "            if dense_gt is not None:\n"
        "                pred_depth = depths[..., 0]  # [B, H, W] expected z-depth\n"
        "                if cfg.dense_depth:\n"
        "                    densedepthloss = dense_depth_l1(pred_depth, dense_gt)\n"
        "                    loss += densedepthloss * cfg.dense_depth_lambda\n"
        "                if cfg.normal_lambda > 0.0:\n"
        "                    normalloss = depth_normal_loss(pred_depth, dense_gt, Ks)\n"
        "                    loss += normalloss * cfg.normal_lambda\n",
    ),
    # 7. Shown on the progress bar (the trainer log the Modal app returns).
    (
        '                desc += f"depth loss={depthloss.item():.6f}| "\n',
        '                desc += f"depth loss={depthloss.item():.6f}| "\n'
        f"            {MARKER}\n"
        "            if dense_gt is not None and cfg.dense_depth:\n"
        '                desc += f"dense depth loss={densedepthloss.item():.4f}| "\n'
        "            if dense_gt is not None and cfg.normal_lambda > 0.0:\n"
        '                desc += f"normal loss={normalloss.item():.4f}| "\n',
    ),
    # 8. gsplat's sparse depth loss without its NaN: a LiDAR point where the render covers no
    #    pixel (or only some of its bilinear footprint) gave 1 / 0 depth, whose gradient is NaN
    #    (litereality_dense.sparse_disparity). Unchanged wherever the render covers the point.
    (
        "                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))\n"
        "                disp_gt = 1.0 / depths_gt  # [1, M]\n",
        f"                {MARKER}\n"
        "                covered = F.grid_sample(\n"
        "                    (alphas > 0.0).float().permute(0, 3, 1, 2), grid, align_corners=True\n"
        "                ).squeeze(3).squeeze(1)  # [1, M]\n"
        "                disp, disp_gt = sparse_disparity(depths, covered, depths_gt)\n",
    ),
]


def patch(source: str) -> str:
    """The trainer's source with the dense-depth edits applied (ValueError if any anchor is not
    there exactly once, or the file is already patched)."""
    if MARKER in source:
        raise ValueError("simple_trainer.py is already patched")
    for anchor, replacement in EDITS:
        count = source.count(anchor)
        if count != 1:
            raise ValueError(f"anchor found {count} times (want 1): {anchor.strip()[:80]!r}")
        source = source.replace(anchor, replacement)
    compile(source, "simple_trainer.py", "exec")
    return source


def main(argv: list[str]) -> None:
    path = Path(argv[1])
    path.write_text(patch(path.read_text()))
    print(f"patched {path} ({len(EDITS)} edits)")


if __name__ == "__main__":
    main(sys.argv)
