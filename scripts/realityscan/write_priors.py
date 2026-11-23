"""RealityScan camera pose PRIORS as XMP sidecars (frame_XXXXX.xmp beside frame_XXXXX.jpg, which
RealityScan reads when it adds the image), so an alignment that the images alone split into
many components is held together by the phone's own poses.

RealityScan (RealityCapture) XMP: xcr:Rotation is the world -> camera rotation, row-major, with
the camera looking down +z, y down (COLMAP's R); xcr:Position the camera centre. PosePrior
"initial": a starting value the alignment may move (by up to ~its prior accuracy, set with
sfmCameraPriorAccuracy* in app.py), not a locked pose. Coordinates "absolute" in the project's
local Euclidean system (metres here).

Sources:
    --from arkit       the Stray pool import's ARKit poses (atrium_1059's ARKit world)
    --from <scan dir>  any scan folder's frame_*.json cameraPoseARFrame (e.g. the MapAnything
                       video import ~/LiteReality-scans/atrium_phone)
Frames named in --skip-eval's selection.json as eval frames get NO prior: their RealityScan pose
then comes from the images only (and stays independent of ARKit for scoring).

    uv run --frozen python scripts/realityscan/write_priors.py ~/LiteReality-scans/rs_stray_prior/rs_input/images --from arkit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rs_common import FLIP, POOL

XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#"
       xcr:Version="3" xcr:PosePrior="initial" xcr:Coordinates="absolute"
       xcr:InTexturing="1" xcr:InMeshing="1">
      <xcr:Rotation>{rot}</xcr:Rotation>
      <xcr:Position>{pos}</xcr:Position>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path)
    ap.add_argument("--from", dest="src", default="arkit")
    ap.add_argument("--skip-eval", action="store_true", help="no prior for eval frames")
    a = ap.parse_args()
    images = a.images.expanduser()
    src = POOL if a.src == "arkit" else Path(a.src).expanduser()
    sel_path = images.parent / "selection.json"
    sel = json.loads(sel_path.read_text())["frames"] if sel_path.exists() else {}
    n = skipped = 0
    for img in sorted(images.glob("frame_*.jpg")):
        if a.skip_eval and sel.get(img.name, {}).get("eval"):
            skipped += 1
            continue
        j = src / (img.stem + ".json")
        if not j.exists():
            continue
        m = np.array(json.loads(j.read_text())["cameraPoseARFrame"], float).reshape(4, 4)
        R_c2w = m[:3, :3] @ FLIP
        R = R_c2w.T
        img.with_suffix(".xmp").write_text(XMP.format(
            rot=" ".join(f"{v:.9f}" for v in R.reshape(-1)),
            pos=" ".join(f"{v:.6f}" for v in m[:3, 3])))
        n += 1
    print(f"{n} priors written from {src} ({skipped} eval frames left without)")


if __name__ == "__main__":
    main()
