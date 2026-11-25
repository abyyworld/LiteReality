"""RESEARCH ONLY. Teach our patched gsplat 1.5.3 trainer a Difix3D-style progressive update.

    python patch_difix.py /opt/gsplat/examples/simple_trainer.py     # after patch_trainer.py

At each of `--difix-steps`, the splats are rendered from that round's novel poses
(`--difix-poses`, a .npy of [rounds, views, 4, 4] camera-to-world matrices in the dataset's frame),
each render is repaired by NVIDIA's Difix (difix_ref, the nearest training photo as reference) and
the repaired views join training: from then on, with probability `--difix-prob`, a step trains on
a batch of them instead of the photos, its loss weighted `--difix-lambda`, only where the render
already had splats (alpha > 0.5: Difix repairs what is there, it does not fill holes). Those steps
skip the per-photo colour grid, the sparse LiDAR loss and the dense depth / normal losses (a novel
view has no photo, no LiDAR and no depth map). The loop itself is `litereality_difix.DifixLoop`.

Difix (code and weights) is NVIDIA's, under a research / evaluation-only licence: this patch is
applied only in scripts/difix/train_difix.py's image, never in the deployed splat app, and no room
trained with it may be served by the hosted room call.

Each edit is anchored on exact text of the trainer as patch_trainer.py leaves it and must match
exactly once.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# LiteReality Difix (scripts/difix/patch_difix.py; research only)"

EDITS: list[tuple[str, str]] = [
    # 1. The loop, beside our dense-depth helpers.
    (
        "                               depth_normal_loss, grid_regularisers, sparse_disparity)\n",
        "                               depth_normal_loss, grid_regularisers, sparse_disparity)\n"
        f"{MARKER}\n"
        "from litereality_difix import DifixLoop\n",
    ),
    # 2. The flags.
    (
        "    # Weight for the normal loss (rendered vs dense depth normals); 0 = off\n"
        "    normal_lambda: float = 0.0\n",
        "    # Weight for the normal loss (rendered vs dense depth normals); 0 = off\n"
        "    normal_lambda: float = 0.0\n"
        f"    {MARKER}\n"
        "    # .npy of [rounds, views, 4, 4] camera-to-world novel poses; empty = no Difix\n"
        "    difix_poses: str = \"\"\n"
        "    # The step each round runs at (one per round)\n"
        "    difix_steps: List[int] = field(default_factory=list)\n"
        "    # Share of steps trained on the repaired novel views once there are some\n"
        "    difix_prob: float = 0.3\n"
        "    # Weight of a novel-view step's loss\n"
        "    difix_lambda: float = 0.2\n"
        "    # Width Difix works at (its input is resized to it, its output back); 0 = native\n"
        "    difix_width: int = 0\n",
    ),
    # 3. The loop, made once the data and the splats exist.
    (
        "        trainloader_iter = iter(trainloader)\n\n        # Training loop.\n",
        "        trainloader_iter = iter(trainloader)\n"
        f"        {MARKER}\n"
        "        difix = DifixLoop(self, cfg) if cfg.difix_poses else None\n"
        "\n        # Training loop.\n",
    ),
    # 4. Each step: a round when due, then maybe a batch of repaired novel views.
    (
        "            except StopIteration:\n"
        "                trainloader_iter = iter(trainloader)\n"
        "                data = next(trainloader_iter)\n\n"
        "            camtoworlds = camtoworlds_gt = data[\"camtoworld\"].to(device)  # [1, 4, 4]\n",
        "            except StopIteration:\n"
        "                trainloader_iter = iter(trainloader)\n"
        "                data = next(trainloader_iter)\n"
        f"            {MARKER}\n"
        "            novel = None\n"
        "            if difix is not None:\n"
        "                difix.maybe_fix(step)\n"
        "                novel = difix.sample()\n"
        "                if novel is not None:\n"
        "                    data = novel\n"
        "            is_novel = novel is not None\n\n"
        "            camtoworlds = camtoworlds_gt = data[\"camtoworld\"].to(device)  # [1, 4, 4]\n",
    ),
    # 5. No LiDAR points for a novel view.
    (
        "            if cfg.depth_loss:\n"
        "                points = data[\"points\"].to(device)  # [1, M, 2]\n",
        f"            {MARKER}\n"
        "            if cfg.depth_loss and not is_novel:\n"
        "                points = data[\"points\"].to(device)  # [1, M, 2]\n",
    ),
    # 6. Render depth only when a depth loss will use it.
    (
        "                render_mode=\"RGB+ED\"\n"
        "                if (cfg.depth_loss or dense_gt is not None)\n",
        "                render_mode=\"RGB+ED\"\n"
        "                if ((cfg.depth_loss and not is_novel) or dense_gt is not None)\n",
    ),
    # 7. No per-photo colour grid: a novel view is no photo.
    (
        "            if cfg.use_bilateral_grid:\n"
        "                grid_y, grid_x = torch.meshgrid(\n",
        f"            {MARKER}\n"
        "            if cfg.use_bilateral_grid and not is_novel:\n"
        "                grid_y, grid_x = torch.meshgrid(\n",
    ),
    # 8. No sparse LiDAR loss for a novel view.
    (
        "            if cfg.depth_loss:\n"
        "                # query depths from depth map\n",
        f"            {MARKER}\n"
        "            if cfg.depth_loss and not is_novel:\n"
        "                # query depths from depth map\n",
    ),
    # 9. A novel view's loss at its weight (before the splat regularisers, which stay whole).
    (
        "            # regularizations\n"
        "            if cfg.opacity_reg > 0.0:\n",
        f"            {MARKER}\n"
        "            if is_novel:\n"
        "                loss = loss * cfg.difix_lambda\n"
        "            # regularizations\n"
        "            if cfg.opacity_reg > 0.0:\n",
    ),
    # 10. The progress bar: no LiDAR loss to show on a novel step.
    (
        "            if cfg.depth_loss:\n"
        "                desc += f\"depth loss={depthloss.item():.6f}| \"\n",
        f"            {MARKER}\n"
        "            if cfg.depth_loss and not is_novel:\n"
        "                desc += f\"depth loss={depthloss.item():.6f}| \"\n",
    ),
]


def patch(source: str) -> str:
    """The trainer's source (already through patch_trainer.py) with the Difix edits applied
    (ValueError if any anchor is not there exactly once, or the file is already patched)."""
    if MARKER in source:
        raise ValueError("simple_trainer.py is already Difix-patched")
    if "LiteReality dense depth" not in source:
        raise ValueError("apply deploy/modal/splat/patch_trainer.py first")
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
    print(f"Difix-patched {path} ({len(EDITS)} edits)")


if __name__ == "__main__":
    main(sys.argv)
