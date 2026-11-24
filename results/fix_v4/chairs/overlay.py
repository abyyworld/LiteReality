"""Each body's splats of a bundle (opaque ones) drawn onto Stray photos, as the diagnosis's
overlay_owner_*.jpg: red Chair0, green Chair1, blue Chair2 (and yellow any other body).

    uv run --frozen python results/fix_v4/chairs/overlay.py run/atrium_stray_q/v2_chairs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from litereality_agent.models.splat import rgbd, spz

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SCAN = Path.home() / "LiteReality-scans/stray_atrium_q600"
COLOURS = {"Chair0": (255, 40, 40), "Chair1": (40, 230, 40), "Chair2": (40, 90, 255)}
FRAMES = (63, 12, 254, 612, 16)


def main(arg: str) -> None:
    d = Path(arg) if Path(arg).is_absolute() else REPO / arg
    bundle = json.loads((d / "splat.json").read_text())
    report = json.loads((d / "report.json").read_text())
    scene = Path(report.get("corrected_scene") or report["scene"])
    scene = scene if scene.is_absolute() else REPO / scene
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sets = []
    for name, file in bundle["bodies"].items():
        if not name.startswith("Chair"):
            continue
        b = model.body(name).id
        s = spz.decode((d / file).read_bytes())
        keep = s.opacities > 0.3
        with np.errstate(all="ignore"):
            world = s.positions[keep].astype(np.float64) @ np.asarray(data.xmat[b]).reshape(3, 3).T \
                + data.xpos[b]
        sets.append((world, COLOURS.get(name, (255, 220, 0))))
    frames = {f.index: f for f in rgbd.load_frames(SCAN)}
    rng = np.random.default_rng(0)
    for fi in FRAMES:
        f = frames[fi]
        im = Image.open(f.image).convert("RGB")
        dr = ImageDraw.Draw(im)
        w2c = np.linalg.inv(f.c2w)
        for pts, c in sets:
            if len(pts) > 6000:
                pts = pts[rng.choice(len(pts), 6000, replace=False)]
            cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
            z = -cam[:, 2]
            ok = z > 0.1
            u = f.K[0, 0] * cam[ok, 0] / z[ok] + f.K[0, 2]
            v = -f.K[1, 1] * cam[ok, 1] / z[ok] + f.K[1, 2]
            for a, b in zip(u[::2], v[::2]):
                dr.rectangle([a - 1, b - 1, a + 1, b + 1], fill=c)
        im = im.rotate(-90, expand=True)
        im.thumbnail((1000, 1000))
        im.save(OUT / f"overlay_{d.name}_f{fi:05d}.jpg", quality=85)
    print("overlays saved")


if __name__ == "__main__":
    main(sys.argv[1])
