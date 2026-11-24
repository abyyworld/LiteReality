"""Before/after renders of the served room with one chair moved 0.5 m (toward the chairs camera),
plus the room left behind with every body taken away: where a chair's splats stayed in the room,
a copy of it shows there. Spark 2.3.1 in headless Chromium (pipeline/evaluate SplatRenderer), as
the room call draws it.

    uv run --frozen --with playwright python results/fix_v4/chairs/render_move.py \
        run/atrium_stray_q/v2 run/atrium_stray_q/v2_chairs
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from litereality_agent.models.splat import spz
from litereality_agent.pipeline.evaluate import fidelity as fd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
K = np.array([[1335.5, 0.0, 960.0], [0.0, 1335.5, 720.0], [0.0, 0.0, 1.0]])
FULL, SIZE = (1920, 1440), (800, 600)
VIEWS = {"chairs": ((0.47, -0.15, 0.10), (-0.33, 0.87, -0.85)),
         "low": ((0.30, -0.35, -0.30), (-0.75, 0.55, -1.0))}
TOWARD = np.array([0.47, -0.15])            # the chairs camera, in plan
MOVE = 0.5
ONLY = ("Chair1", "Chair2")      # the dark office chair's body and the teal chair's

def look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = r, u, -f, eye
    return pose


def parts(out_dir: Path):
    bundle = json.loads((out_dir / "splat.json").read_text())
    report = json.loads((out_dir / "report.json").read_text())
    scene = Path(report.get("corrected_scene") or report["scene"])
    scene = scene if scene.is_absolute() else REPO / scene
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    room = spz.decode((out_dir / bundle["room"]).read_bytes())
    bodies = {}
    for name, file in bundle["bodies"].items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        bodies[name] = (spz.decode((out_dir / file).read_bytes()),
                        np.asarray(data.xmat[b]).reshape(3, 3).copy(), data.xpos[b].copy())
    return room, bodies


def scene_with(room, bodies, moved: str | None, keep_bodies: bool = True) -> bytes:
    out = [room]
    if keep_bodies:
        for name, (local, R, x) in bodies.items():
            x = x.copy()
            if name == moved:
                d = TOWARD - x[:2]
                x[:2] += MOVE * d / np.linalg.norm(d)
            with np.errstate(all="ignore"):
                out.append(local.transformed(R, x))
    return spz.encode(spz.Splats.concat(out))


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 9 * len(text) + 12, 22), fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return img


def main(dirs: list[str]) -> None:
    cams = [fd.three_camera(fd.View(k, Path(n), K, FULL, look_at(*v)), SIZE)
            for k, (n, v) in enumerate(VIEWS.items())]
    renderer = fd.SplatRenderer(timeout_s=900)
    sheet_rows, meta = [], {}
    budget = int(os.environ.get("MAX_RENDERS", "100"))     # new renders this run (resumable)
    left = False
    for arg in dirs:
        d = Path(arg) if Path(arg).is_absolute() else REPO / arg
        room, bodies = parts(d)
        chairs = [n for n in bodies if n.startswith("Chair") and n in ONLY]
        variants = [("as served", None, True)] + [(f"{c} moved {MOVE} m", c, True)
                                                  for c in chairs] \
            + [("room left behind (no bodies)", None, False)]
        for title, moved, keep in variants:
            t = time.perf_counter()
            slug = title.split(" (")[0].replace(" ", "_").replace(".", "")
            files = [OUT / f"{d.name}_{slug}_{view}.jpg" for view in VIEWS]
            if all(f.is_file() for f in files):          # resumable: one render is ~4 min
                sheet_rows.append([label(Image.open(f).convert("RGB"),
                                         f"{d.name}: {title} [{v}]")
                                   for f, v in zip(files, VIEWS)])
                continue
            if budget <= 0:
                print(f"{d.name}: {title} left for the next run", flush=True)
                left = True
                continue
            budget -= 1
            renders = renderer.render(scene_with(room, bodies, moved, keep), cams, SIZE)
            row = []
            for (view, _), rgba, file in zip(VIEWS.items(), renders, files, strict=True):
                rgb, _cover = fd.composite(rgba, SIZE)
                img = Image.fromarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8))
                img.save(file, quality=90)
                row.append(label(img, f"{d.name}: {title} [{view}]"))
            sheet_rows.append(row)
            meta[f"{d.name}: {title}"] = round(time.perf_counter() - t, 1)
            print(f"{d.name}: {title} rendered in {meta[f'{d.name}: {title}']} s", flush=True)
    if left:
        return
    w, h = SIZE
    sheet = Image.new("RGB", (w * len(VIEWS), h * len(sheet_rows)))
    for r, row in enumerate(sheet_rows):
        for c, img in enumerate(row):
            sheet.paste(img, (c * w, r * h))
    sheet.save(OUT / "move_sheet.jpg", quality=85)
    (OUT / "render_move.json").write_text(json.dumps({"views": VIEWS, "move_m": MOVE,
                                                      "toward": TOWARD.tolist(),
                                                      "seconds": meta}, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
