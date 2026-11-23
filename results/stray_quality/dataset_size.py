"""The trainer's dataset tar for a scan at a training width, built locally (no GPU), to check it
fits the deployed trainer's MAX_DATASET_BYTES (400 MB) before paying for a run. Reads the
scan's PromptDA cache (depth_dense/, made by an earlier `splatroom --depth promptda` run on the
same scan); frames without a cached map fall back to LiDAR points, as in a real run.

    uv run --frozen python results/stray_quality/dataset_size.py ~/LiteReality-scans/stray_atrium_q600 1920
"""

from __future__ import annotations

import sys
from pathlib import Path

from litereality_agent.models.splat import dataset, promptda, rgbd
from litereality_agent.pipeline import splatroom


def main() -> None:
    scan = Path(sys.argv[1]).expanduser()
    width = int(sys.argv[2]) if len(sys.argv) > 2 else dataset.TRAIN_WIDTH
    frames = rgbd.load_frames(scan)
    dense = {}
    for f in frames:
        d = promptda.load_dense(f, dataset.TRAIN_WIDTH)
        if d is not None:
            dense[f.index] = d
    dataset.train_width_override = width
    options = {"dense_depth": True, "normal_lambda": 0.05}
    report: dict = {}
    tar = splatroom.training_set(frames, dense, options, None, report, log=print)
    mb = len(tar) / 1e6
    limit = 400 * 1024 * 1024 / 1e6
    print(f"{scan.name} at {width}: {len(frames)} photos, {len(dense)} with dense depth, "
          f"tar {mb:.0f} MB (trainer limit {limit:.0f} MB: {'fits' if mb < limit else 'TOO BIG'})"
          f"; {report['dataset']}")


if __name__ == "__main__":
    main()
