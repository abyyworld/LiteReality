"""Build the atrium v2 training set (the very tar `litereality splatroom` gave the live v2 room)
without training: the pipeline runs as for v2 and its trainer is a stand-in that keeps the tar
and the options and stops.

    uv run --env-file .env python scripts/difix/prepare.py --out ~/LiteReality-scans/difix/v2_data

v2's flags (run/atrium_stray_q/v2/report.json): scan stray_atrium_q600, PromptDA depth (cached),
dense depth and normal (0.05) losses, train width 1440, 10,000 steps, the default preset, and
4,000 seed points a photo (2.23M; the pipeline has since lowered its default to fit `cap_max`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


class Keep:
    name = "keep the dataset (no training)"

    def __init__(self, out: Path) -> None:
        self.out = out

    def warm(self) -> None:
        pass

    def stream(self, dataset_tar: bytes, options: dict | None = None):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "dataset.tar").write_bytes(dataset_tar)
        (self.out / "options.json").write_text(json.dumps(options or {}, indent=1))
        print(f"    kept the dataset: {len(dataset_tar) / 1e6:.1f} MB, options {options}",
              flush=True)
        yield {"kind": "error", "error": "stand-in trainer: dataset kept, nothing trained",
               "log": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "LiteReality-scans/difix/v2_data"))
    ap.add_argument("--scan", default=str(Path.home() / "LiteReality-scans/stray_atrium_q600"))
    ap.add_argument("--scene", default=str(REPO / "run/atrium_stray_q/v2/physics/scene.xml"))
    args = ap.parse_args()
    os.chdir(REPO)
    out = Path(args.out).expanduser()
    from litereality_agent import cli

    cli._ephemeral_splat_trainer = lambda stack, modal_kw: Keep(out)
    try:
        cli.main(["splatroom", args.scan, "--scene", args.scene, "--out", str(out / "bundle"),
                  "--ephemeral-trainer", "--depth", "promptda", "--depth-loss-dense",
                  "--normal-lambda", "0.05", "--train-width", "1440",
                  "--dense-points", "4000"])
    except SystemExit:
        pass
    return 0 if (out / "dataset.tar").is_file() else 1


if __name__ == "__main__":
    sys.exit(main())
