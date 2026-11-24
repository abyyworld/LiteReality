"""Write a variant of an SPZ for the Spark diagnosis (scripts/eyeful/spark_score.py).

    uv run --frozen python scripts/eyeful/spz_variant.py IN.spz OUT.spz [--sh N] [--prerotate]

--sh N        keep SH up to degree N (0 = colour only)
--prerotate   turn the splats (positions, rotations and higher SH, `Splats.transformed`) by
              inv(Z_UP_TO_THREE), so the render page's Z-up -> Y-up group turn brings them back to
              the training frame: Spark then draws them with an identity world transform. With
              spark_score.py --prerotate the camera follows. Tells whether Spark evaluates SH in
              the mesh's own frame (a page turn then changes nothing) or in the world's.
"""

from __future__ import annotations

import argparse
import gzip
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from litereality_agent.models.splat import spz  # noqa: E402
from litereality_agent.pipeline.evaluate.fidelity import Z_UP_TO_THREE  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--sh", type=int, default=None)
    p.add_argument("--prerotate", action="store_true")
    a = p.parse_args()
    data = Path(a.src).read_bytes()
    flags = struct.unpack_from("<IIIBBBB", gzip.decompress(data)[:16])[5]
    s = spz.decode(data)
    if a.sh is not None and s.sh is not None:
        k = spz.sh_dim(a.sh)
        s = spz.Splats(s.positions, s.scales, s.rotations, s.colors, s.opacities,
                       s.sh[:, :k].copy() if k else None)
    if a.prerotate:
        s = s.transformed(np.linalg.inv(Z_UP_TO_THREE)[:3, :3], np.zeros(3))
    out = spz.encode(s, antialiased=bool(flags & spz.FLAG_ANTIALIASED))
    Path(a.dst).write_bytes(out)
    print(f"{a.dst}: {len(s)} splats, SH {s.sh_degree}, antialiased {bool(flags & 1)}, "
          f"{len(out) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
