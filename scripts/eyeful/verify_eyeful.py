"""Check the prepared Eyeful dataset loads in the official COLMAP reader (pycolmap), on Modal.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run modal run --env main \
        scripts/eyeful/verify_eyeful.py --scene office1b

Loads /<scene>/gsplat_2k/sparse/0 with pycolmap, checks each image's camera matches its file's
size, that the test list is all camera 17, and reprojects the tie points into a few images (the
rescaled 2D observations must land on the rescaled projections: a wrong intrinsics rescale shows
up as pixel error here).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("litereality-eyeful-verify")
volume = modal.Volume.from_name("litereality-eyeful")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pycolmap==3.10.0", "numpy==1.26.4", "pillow==10.4.0")


@app.function(image=image, volumes={"/vol": volume}, cpu=2, memory=8192, timeout=900)
def verify(scene: str) -> dict:
    import numpy as np
    import pycolmap
    from PIL import Image

    root = Path("/vol") / scene / "gsplat_2k"
    rec = pycolmap.Reconstruction(str(root / "sparse/0"))
    bad_size = []
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        with Image.open(root / "images" / im.name) as f:
            if f.size != (cam.width, cam.height):
                bad_size.append(im.name)
    test = (root / "test_images.txt").read_text().split()
    train = (root / "train_images.txt").read_text().split()
    names = {im.name for im in rec.images.values()}
    errs = []
    for im in list(rec.images.values())[::150]:
        cam = rec.cameras[im.camera_id]
        for p2 in im.points2D:
            if p2.has_point3D():
                xyz = rec.points3D[p2.point3D_id].xyz
                uv = cam.img_from_cam(im.cam_from_world * xyz)
                errs.append(float(np.linalg.norm(uv - p2.xy)))
    errs = np.array(errs)
    return {
        "pycolmap_loaded": True,
        "cameras": len(rec.cameras), "images": len(rec.images), "points3D": len(rec.points3D),
        "camera_models": sorted({str(c.model) for c in rec.cameras.values()}),
        "image_size_mismatch": len(bad_size),
        "test": len(test), "train": len(train), "overlap": len(set(test) & set(train)),
        "test_all_cam17": all(n.startswith("17_") for n in test),
        "split_covers_model": set(test) | set(train) == names,
        "reproj_px": {"n": int(len(errs)), "median": float(np.median(errs)),
                      "p90": float(np.percentile(errs, 90))},
    }


@app.local_entrypoint()
def main(scene: str = "office1b"):
    print(json.dumps(verify.remote(scene), indent=2))
