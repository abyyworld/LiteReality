"""Score one trained atrium splat on the FIXED held-out evaluation set, and render fixed 'wander' views.

Every variant is judged the same way, by the same renderer:

* renderer: `pipeline/evaluate`'s `SplatRenderer` - headless Chromium (Playwright), SwiftShader
  software WebGL, Spark 2.3.1 on three.js 0.180 (the room call's own splat renderer), so the
  numbers are what a viewer sees, at the export's SH degree (deploy/modal/splat
  "export_sh_degree", 1). Not gsplat: the trainer's own PSNR is in report.json beside it.
* eval set: the frames in eval_frames.txt (odometry rows of recording 901a305fdf, picked by
  pick_eval.py from the pool import ~/LiteReality-scans/stray_atrium_pool). Every training scan
  was imported with `--exclude eval_frames.txt --exclude-span 0.2`, so no variant trained on
  them or on any frame within 0.2 s of them. Each import fitted its own alignment to
  atrium_1059, so a pool pose is moved into the variant's world: T_variant @ inv(T_pool) @ pose.
* metrics at 480 x 360 (rendered at 2x, box-filtered): PSNR, PSNR after a per-view 3x4 colour
  fit (`psnr_cc`, as gsplat's cc_psnr: ARKit re-exposes every frame), SSIM (luma, 11x11
  Gaussian), LPIPS (AlexNet, lpips 0.1.4), and PSNR inside RoomPlan's door/window boxes
  (`psnr_openings`) and on the chairs/table (`psnr_objects`) from `fidelity.view_regions`.
* wander: 3 fixed cameras in the room frame (Z up, metres; WANDER below), the same for every
  variant, rendered at 960 x 720 (2x supersampled) from the SERVED room: the bundle's room splat
  plus each body's splat placed at its pose in the corrected physics (what the room call draws,
  chairs where the physics put them) - or `--raw` for the uncut trained snapshot.

    uv run --frozen --with playwright --with lpips==0.1.4 python results/stray_quality/score.py \\
        run v1 run/atrium_stray_q/v1            # eval + wander -> v1_scores.json, v1_wander.jpg
    uv run --frozen --with playwright --with lpips==0.1.4 python results/stray_quality/score.py \\
        wander v1 run/atrium_stray_q/v1 [--raw]
    uv run --frozen python results/stray_quality/score.py table v0 v1 v2 v3   # -> table.md
    uv run --frozen python results/stray_quality/score.py compare v0 v1 v2 v3 # -> compare.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POOL = Path("~/LiteReality-scans/stray_atrium_pool").expanduser()
EVAL_FRAMES = HERE / "eval_frames.txt"
SIZE = (480, 360)                 # metric resolution
SCALE = 2                         # rendered at 2x and box-filtered (as `fidelity`)
WANDER_SIZE = (960, 720)
# A phone camera (the recording's own: fx 1335.5 at 1920 x 1440, ~71 deg wide).
WANDER_K = np.array([[1335.5, 0.0, 960.0], [0.0, 1335.5, 720.0], [0.0, 0.0, 1.0]])
WANDER_FULL = (1920, 1440)
# Room frame (atrium_1059's, Z up; floor at z -1.27, eye height ~1.5 m = z 0.25). RoomPlan:
# desk Table0 centre (-0.58, 1.18) running along the window wall (Wall2, window centre
# (-0.85, 1.57, 0.62)); a second window on Wall3 (centre (-1.51, -0.48)); the door on the
# opposite wall (Wall0, centre (1.30, 0.28)); chairs Chair0..2 in a row on the room side of the
# desk around (-0.33, 0.87). All three eyes lie inside the walked area (capture x -1.05..0.80,
# y -0.50..1.14).
WANDER = {
    "desk_to_door": {"eye": (-1.0, -0.1, 0.25), "look": (1.2, 0.9, -0.5),
                     "what": "along the desk toward the door"},
    "windows": {"eye": (0.4, -0.2, 0.30), "look": (-2.0, 0.9, 0.4),
                "what": "at both windows (Wall2 right, Wall3 left)"},
    "chairs": {"eye": (0.47, -0.15, 0.10), "look": (-0.33, 0.87, -0.85),
               "what": "down at the three chairs"},
}
GPU_RATE = {"trainer": 3.95 / 3600, "other": 1.10 / 3600}   # $/s: H100; L4/A10G-class (est.)


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────


def _fd():
    from litereality_agent.pipeline.evaluate import fidelity
    return fidelity


def eval_rows(path: Path = EVAL_FRAMES) -> list[tuple[int, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        tok = line.split("#", 1)[0].split()
        if tok:
            out.append((int(tok[0]), tok[1]))
    return out


def stray_transform(scan: Path) -> np.ndarray:
    doc = json.loads((scan / "scene_alignment.json").read_text())
    return np.asarray(doc["strayImport"]["transform"], dtype=float)


def look_at(eye, target) -> np.ndarray:
    """Camera -> room pose (Z up) at `eye` looking at `target`; camera -Z forward, +Y up."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = r, u, -f, eye
    return pose


def eval_views(scan: Path, pool: Path = POOL) -> tuple[list, dict]:
    """The eval frames as `fidelity.View`s in `scan`'s world, and how far that world is from
    the pool's (the per-import alignment difference)."""
    fd = _fd()
    doc = json.loads((scan / "scene_alignment.json").read_text())
    if "strayImport" in doc:
        T = stray_transform(scan) @ np.linalg.inv(stray_transform(pool))
    else:
        # Another capture of the room (e.g. the phone video, `videoImport`): it was aligned to
        # the same LiteReality scan (atrium_1059) as the pool, so the pool's poses are already
        # in its world (to the ~1 cm of both alignments).
        T = np.eye(4)
    views = []
    for row, stem in eval_rows():
        meta = json.loads((pool / f"{stem}.json").read_text())
        if int(meta["stray"]["row"]) != row:
            raise ValueError(f"{pool}/{stem}.json is row {meta['stray']['row']}, not {row}")
        pose = fd.camera_pose((T @ np.asarray(meta["cameraPoseARFrame"]).reshape(4, 4)).reshape(-1))
        K = np.asarray(meta["intrinsics"], float).reshape(3, 3)
        image = pool / f"{stem}.jpg"
        with Image.open(image) as im:
            full = im.size
        n = stem[len("frame_"):]
        depth, conf = pool / f"depth_{n}.png", pool / f"conf_{n}.png"
        views.append(fd.View(row, image, K, full, pose, depth if depth.is_file() else None,
                             conf if conf.is_file() else None))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
    return views, {"translation_mm": round(float(np.linalg.norm(T[:3, 3])) * 1000, 2),
                   "rotation_deg": round(angle, 4)}


def colour_fit(render: np.ndarray, real: np.ndarray) -> np.ndarray:
    """The render under the 3x4 affine colour map that best fits the photo (least squares)."""
    x = render.reshape(-1, 3)
    A = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    M, *_ = np.linalg.lstsq(A, real.reshape(-1, 3), rcond=None)
    return np.clip(A @ M, 0.0, 1.0).reshape(render.shape)


class Lpips:
    def __init__(self) -> None:
        import lpips
        import torch

        self.torch = torch
        self.net = lpips.LPIPS(net="alex", verbose=False).eval()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        t = self.torch
        with t.no_grad():
            x = t.from_numpy(a.astype(np.float32)).permute(2, 0, 1)[None] * 2 - 1
            y = t.from_numpy(b.astype(np.float32)).permute(2, 0, 1)[None] * 2 - 1
            return float(self.net(x, y).item())


def served_splats(out_dir: Path, report: dict):
    """The room the call draws: the bundle's room splat plus each body's splat at its pose in
    the physics the run served (``corrected_scene``, else the scene it cut against)."""
    import mujoco

    from litereality_agent.models.splat import spz

    bundle = json.loads((out_dir / "splat.json").read_text())
    scene = report.get("corrected_scene") or report["scene"]
    scene = Path(scene) if Path(scene).is_absolute() else REPO / scene
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    parts = [spz.decode((out_dir / bundle["room"]).read_bytes())]
    for name, file in (bundle.get("bodies") or {}).items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise ValueError(f"{scene} has no body {name}")
        local = spz.decode((out_dir / file).read_bytes())
        parts.append(local.transformed(np.asarray(data.xmat[b]).reshape(3, 3),
                                       np.asarray(data.xpos[b])))
    return spz.encode(spz.Splats.concat(parts)), str(scene)


def _label(img: Image.Image, text: str, size: int = 18) -> Image.Image:
    from litereality_agent.fonts import font

    d = ImageDraw.Draw(img)
    f = font(size)
    box = d.textbbox((6, 4), text, font=f)
    d.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=f)
    return img


def _to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8))


# ── commands ────────────────────────────────────────────────────────────────────────────────────


def load_run(out_dir: Path) -> dict:
    report = json.loads((out_dir / "report.json").read_text())
    return report


def score(variant: str, out_dir: Path, spz_file: str = "raw.spz.bin") -> dict:
    fd = _fd()
    from litereality_agent.pipeline.quickroom.roomplan import parse_room

    report = load_run(out_dir)
    scan = Path(report["scan"]).expanduser()
    views, drift = eval_views(scan)
    plan = parse_room(scan)
    objects, openings = fd.scan_boxes(plan)
    spz_bytes = (out_dir / spz_file).read_bytes()
    renderer = fd.SplatRenderer(timeout_s=600)
    big = (SIZE[0] * SCALE, SIZE[1] * SCALE)
    t = time.perf_counter()
    renders = renderer.render(spz_bytes, [fd.three_camera(v, big) for v in views], big)
    render_s = time.perf_counter() - t
    lp = Lpips()
    rows, sheet = [], []
    for v, rgba in zip(views, renders, strict=True):
        real = fd.load_photo(v.image, SIZE)
        render, cover = fd.composite(rgba, SIZE)
        reg = fd.view_regions(v, plan, objects, openings, SIZE)
        smap = fd.ssim_map(real, render)
        row = {"row": v.index, "image": v.image.name,
               "psnr": fd.psnr(real, render), "psnr_cc": fd.psnr(real, colour_fit(render, real)),
               "ssim": float(smap.mean()), "lpips": lp(real, render),
               "coverage": float(cover.mean())}
        for name in ("openings", "objects"):
            m = reg[name]
            row[f"frac_{name}"] = float(m.mean())
            row[f"psnr_{name}"] = (fd.psnr(real, colour_fit(render, real), m)
                                   if m.mean() >= fd.MIN_MASK_FRACTION else None)
        rows.append({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()})
        sheet.append((real, render, row))

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    stats = report.get("stats") or {}
    last = (report.get("bundles") or [{}])[-1]
    depth = report.get("depth") or {}
    masks = report.get("masks") or {}
    trainer_s = float(report.get("trainer_s") or 0)
    other_s = float(depth.get("seconds") or 0) * (0 if depth.get("cached") else 1) \
        + float(masks.get("gpu_s") or 0)
    raw = out_dir / spz_file
    out = {
        "variant": variant, "out_dir": str(out_dir), "scan": str(scan), "spz": spz_file,
        "renderer": f"{renderer.name} {renderer.version} (SwiftShader, Spark 2.3.1)",
        "size": list(SIZE), "render_scale": SCALE, "views": len(rows),
        "alignment_vs_pool": drift, "render_s": round(render_s, 1),
        "summary": {k: mean(k) for k in ("psnr", "psnr_cc", "ssim", "lpips", "coverage",
                                         "psnr_openings", "psnr_objects")},
        "train": {"frames": report.get("frames"), "dataset": report.get("dataset"),
                  "options": report.get("options"),
                  "train_width": report.get("train_width") or 960, "trainer_s": trainer_s,
                  "total_s": report.get("total_s"), "gpu": report.get("gpu"),
                  "trainer_heldout": {k: stats.get(k) for k in ("psnr", "ssim", "lpips",
                                                                 "cc_psnr", "num_GS")},
                  "splats": last.get("splats"), "raw_mb": round(raw.stat().st_size / 1e6, 2),
                  "bundle_mb": last.get("mb"),
                  "gpu_s": {"trainer": trainer_s, "depth_and_masks": round(other_s, 1)},
                  "cost_est_usd": round(trainer_s * GPU_RATE["trainer"]
                                        + other_s * GPU_RATE["other"], 3),
                  "corrections": report.get("corrections"),
                  "corrected_scene": report.get("corrected_scene")},
        "per_view": rows,
    }
    (HERE / f"{variant}_scores.json").write_text(json.dumps(out, indent=1) + "\n")
    _eval_sheet(variant, sheet)
    s = out["summary"]
    print(f"{variant}: PSNR {s['psnr']} (cc {s['psnr_cc']}), SSIM {s['ssim']}, LPIPS {s['lpips']}, "
          f"openings {s['psnr_openings']}, objects {s['psnr_objects']}, coverage {s['coverage']} "
          f"over {len(rows)} views; render {render_s:.0f} s; alignment vs pool {drift}")
    return out


def _eval_sheet(variant: str, sheet: list, picks: int = 6) -> None:
    idx = np.linspace(0, len(sheet) - 1, min(picks, len(sheet))).round().astype(int)
    w, h = SIZE
    canvas = Image.new("RGB", (2 * w, h * len(idx)), (0, 0, 0))
    for k, i in enumerate(idx):
        real, render, row = sheet[i]
        canvas.paste(_label(_to_img(real), f"photo row {row['row']}"), (0, k * h))
        canvas.paste(_label(_to_img(render), f"{variant}  PSNR {row['psnr']:.1f} cc "
                            f"{row['psnr_cc']:.1f} LPIPS {row['lpips']:.3f}"), (w, k * h))
    canvas.save(HERE / f"{variant}_eval_sheet.jpg", quality=88)


def wander(variant: str, out_dir: Path, raw: bool = False) -> Path:
    fd = _fd()
    report = load_run(out_dir)
    if raw:
        spz_bytes, source = (out_dir / "raw.spz.bin").read_bytes(), "raw.spz.bin (uncut)"
    else:
        spz_bytes, scene = served_splats(out_dir, report)
        source = f"served bundle, bodies at {Path(scene).name}"
    big = (WANDER_SIZE[0] * SCALE, WANDER_SIZE[1] * SCALE)
    cams = []
    for k, (name, w) in enumerate(WANDER.items()):
        view = fd.View(k, Path(name), WANDER_K, WANDER_FULL, look_at(w["eye"], w["look"]))
        cams.append(fd.three_camera(view, big))
    renders = fd.SplatRenderer(timeout_s=600).render(spz_bytes, cams, big)
    folder = HERE / "wander"
    folder.mkdir(exist_ok=True)
    tiles = []
    for (name, w), rgba in zip(WANDER.items(), renders, strict=True):
        rgb, _ = fd.composite(rgba, WANDER_SIZE)
        img = _label(_to_img(rgb), f"{variant} - {w['what']}")
        img.save(folder / f"{variant}_{name}.jpg", quality=90)
        tiles.append(img)
    strip = Image.new("RGB", (WANDER_SIZE[0] * len(tiles), WANDER_SIZE[1]))
    for k, img in enumerate(tiles):
        strip.paste(img, (k * WANDER_SIZE[0], 0))
    path = HERE / f"{variant}_wander.jpg"
    strip.save(path, quality=88)
    meta = HERE / "wander" / f"{variant}.json"
    meta.write_text(json.dumps({"variant": variant, "out_dir": str(out_dir), "source": source,
                                "views": WANDER}, indent=1) + "\n")
    print(f"{variant}: wander -> {path} ({source})")
    return path


def table(variants: list[str]) -> str:
    head = ("| variant | frames | width | steps | PSNR | PSNR cc | SSIM | LPIPS | PSNR win/door "
            "| PSNR objects | trainer PSNR | train s | GPU $ est | splats | raw MB | bundle MB |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for v in variants:
        d = json.loads((HERE / f"{v}_scores.json").read_text())
        s, t = d["summary"], d["train"]
        opts = t.get("options") or {}
        lines.append(
            f"| {v} | {t['frames']} | {t.get('train_width') or 960} | {opts.get('steps', '?')} "
            f"| {s['psnr']:.2f} | {s['psnr_cc']:.2f} | {s['ssim']:.3f} | {s['lpips']:.3f} "
            f"| {s['psnr_openings'] or float('nan'):.2f} | {s['psnr_objects'] or float('nan'):.2f} "
            f"| {t['trainer_heldout'].get('psnr') or float('nan'):.2f} | {t['trainer_s']:.0f} "
            f"| {t['cost_est_usd']:.2f} | {t['splats'] or 0:,} | {t['raw_mb']} | {t['bundle_mb']} |")
    text = "\n".join(lines) + "\n"
    (HERE / "table.md").write_text(text)
    print(text)
    return text


def compare(variants: list[str], tile: tuple[int, int] = (640, 480)) -> Path:
    names = list(WANDER)
    canvas = Image.new("RGB", (tile[0] * len(variants), tile[1] * len(names)), (0, 0, 0))
    for c, v in enumerate(variants):
        for r, name in enumerate(names):
            with Image.open(HERE / "wander" / f"{v}_{name}.jpg") as im:
                canvas.paste(im.convert("RGB").resize(tile, Image.Resampling.LANCZOS),
                             (c * tile[0], r * tile[1]))
    path = HERE / "compare.jpg"
    canvas.save(path, quality=88)
    print(path)
    return path


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(REPO / "src"))
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "eval", "wander"):
        s = sub.add_parser(cmd)
        s.add_argument("variant")
        s.add_argument("out_dir")
        s.add_argument("--spz", default="raw.spz.bin", help="eval: the splat file in out_dir")
        s.add_argument("--raw", action="store_true", help="wander: the uncut raw snapshot")
    for cmd in ("table", "compare"):
        s = sub.add_parser(cmd)
        s.add_argument("variants", nargs="+")
    a = p.parse_args(argv)
    if a.cmd in ("run", "eval", "wander"):
        out = Path(a.out_dir)
        out = out if out.is_absolute() else REPO / out
        if a.cmd in ("run", "eval"):
            score(a.variant, out, a.spz)
        if a.cmd in ("run", "wander"):
            wander(a.variant, out, raw=a.raw)
    elif a.cmd == "table":
        table(a.variants)
    else:
        compare(a.variants)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
