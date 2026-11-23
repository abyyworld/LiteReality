"""ARKit pose PRIORS for the dense Stray frames (prep_dense.py output), as RealityScan XMP sidecars,
in the convention RealityScan itself writes (measured, not assumed: a no-prior run's own
-exportXMP against its COLMAP export of the same cameras, scripts/realityscan/dense_report.py's
run s2_def_dm2, 145 cameras, residual 1e-12):

    xcr:Position  = camera centre C_x in the XMP world
    xcr:Rotation  = world -> camera rotation, row-major, OpenCV camera (x right, y down, z ahead)
                    (= COLMAP's R for the same world)
    XMP world     = A . COLMAP-export world, A = [[1,0,0],[0,0,1],[0,-1,0]]: the XMP world is
                    Z up where the COLMAP export is Y down. So the export of a prior-placed
                    alignment comes out turned 90 degrees about x relative to the priors' frame:
                    a pure frame change, not an error.

The priors go in a Z-up world: ARKit's (gravity-aligned, Y up, metres) turned by ARKIT_TO_Z_UP,
so RealityScan's "up" is gravity. Each XMP also carries the shared calibration (one calibration
group and one lens/distortion group for the whole walk, focal 35 mm = ARKit fx * 36 / 1920,
Brown3, "initial"), so the per-component focal lengths of the earlier prior run cannot happen
through the sidecars. PosePrior "initial": the alignment may move them (by ~ the accuracies set
with sfmCameraPriorAccuracy* in the run). Eval frames get NO prior unless --eval-priors.

    uv run --frozen python scripts/realityscan/write_priors_dense.py ~/LiteReality-scans/rs_dense_stray --out ~/LiteReality-scans/rs_dense_stray/priors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
RAW = Path("~/LiteReality-scans/stray_atrium_raw/901a305fdf").expanduser()
FLIP = np.diag([1.0, -1.0, -1.0])
ARKIT_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xcr:Version="3" xcr:PosePrior="{pose_prior}" xcr:Coordinates="absolute"
       xcr:DistortionModel="brown3" xcr:FocalLength35mm="{f35:.6f}" xcr:Skew="0"
       xcr:AspectRatio="1" xcr:PrincipalPointU="0" xcr:PrincipalPointV="0"
       xcr:CalibrationPrior="initial" xcr:CalibrationGroup="0" xcr:DistortionGroup="0"
       xcr:InTexturing="1" xcr:InMeshing="1"
       xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#">
      <xcr:Rotation>{rot}</xcr:Rotation>
      <xcr:Position>{pos}</xcr:Position>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""


def main() -> None:
    from litereality_agent.models.capture import stray as S

    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval-priors", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="a prior on every k-th frame only")
    ap.add_argument("--pose-prior", default="initial", choices=("initial", "draft"),
                    help="initial: a georeferencing prior (each component is placed by its "
                         "cameras' positions); draft: a starting pose the alignment adjusts")
    a = ap.parse_args()
    sel = json.loads((a.frames.expanduser() / "selection.json").read_text())["frames"]
    rec = S.read_recording(RAW)
    by_row = {f.row: f for f in rec.frames}
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.xmp"):
        p.unlink()
    n = skipped = 0
    # One calibration group: one focal for all (the median of ARKit's per-frame fx; ARKit
    # refocuses, so its fx varies by ~1%).
    fxs = [float(by_row[d["row"]].K[0, 0]) for d in sel.values()]
    f35 = float(np.median(fxs)) * 36.0 / 1920.0
    for k, (name, d) in enumerate(sorted(sel.items(), key=lambda kv: kv[1]["row"])):
        if (d["eval"] and not a.eval_priors) or k % a.every:
            skipped += 1
            continue
        f = by_row[d["row"]]
        R_c2w = ARKIT_TO_Z_UP @ f.c2w[:3, :3] @ FLIP
        C = ARKIT_TO_Z_UP @ f.c2w[:3, 3]
        (out / (Path(name).stem + ".xmp")).write_text(XMP.format(
            f35=f35, pose_prior=a.pose_prior,
            rot=" ".join(f"{v:.9f}" for v in R_c2w.T.reshape(-1)),
            pos=" ".join(f"{v:.6f}" for v in C)))
        n += 1
    print(f"{n} priors -> {out} ({skipped} frames without; fx median {np.median(fxs):.1f}, focal35 {f35:.4f})")


if __name__ == "__main__":
    main()
