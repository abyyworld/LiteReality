"""Stage what the browser scoring (scripts/eyeful/spark_score.py) needs on the Volume, small.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run modal run --env main scripts/eyeful/fetch_eval.py

One short CPU function (no GPU). It reads the dataset's COLMAP model (sparse/0, the frame the
trainer saw), keeps the test images (test_images.txt, camera 17), and writes, on the Volume
`litereality-eyeful` under /<scene>/spark_eval/:

    test_cams.json   per test image, in gsplat's val order (image names sorted): name, width,
                     height, fx, fy, cx, cy (PINHOLE), and world->camera R, t (COLMAP, OpenCV
                     axes: x right, y down, z forward)
    gt/<name>        the 71 test photos exactly as the trainer read them (byte copies)

Then fetch with `modal volume get` (about 90 MB, plus the runs' renders/ folders).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_eyeful import (  # noqa: E402
    CAMERA_PARAMS,
    MOUNT,
    VOLUME_NAME,
    image,
    qvec_to_R,
    read_cameras,
    read_images,
)

app = modal.App("litereality-eyeful-fetch-eval")
volume = modal.Volume.from_name(VOLUME_NAME)
image = image.add_local_python_source("prepare_eyeful")


@app.function(image=image, volumes={MOUNT: volume}, cpu=1, memory=2048, timeout=600,
              scaledown_window=2)
def stage(scene: str = "office1b") -> dict:
    root = MOUNT / scene / "gsplat_2k"
    out = MOUNT / scene / "spark_eval"
    if out.exists():
        shutil.rmtree(out)
    (out / "gt").mkdir(parents=True)
    test = {s.strip() for s in (root / "test_images.txt").read_text().splitlines() if s.strip()}
    cams = read_cameras(root / "sparse/0/cameras.bin")
    ims = sorted((im for im in read_images(root / "sparse/0/images.bin") if im["name"] in test),
                 key=lambda im: im["name"])
    rows = []
    for im in ims:
        c = cams[im["camera_id"]]
        if CAMERA_PARAMS[c["model"]][0] != "PINHOLE":
            raise ValueError(f"{im['name']}: camera model {c['model']}, not PINHOLE")
        fx, fy, cx, cy = c["params"]
        rows.append({"name": im["name"], "camera_id": im["camera_id"], "width": c["width"],
                     "height": c["height"], "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                     "R": qvec_to_R(im["q"]).tolist(), "t": list(im["t"])})
        shutil.copyfile(root / "images" / im["name"], out / "gt" / im["name"])
    doc = {"scene": scene, "source": str(root / "sparse/0"), "order": "image name, sorted",
           "convention": "COLMAP world->camera x_cam = R x_world + t, OpenCV axes",
           "views": rows}
    (out / "test_cams.json").write_text(json.dumps(doc, indent=1))
    volume.commit()
    return {"views": len(rows), "missing": sorted(test - {r["name"] for r in rows})}


@app.local_entrypoint()
def main(scene: str = "office1b"):
    print(json.dumps(stage.remote(scene)))
