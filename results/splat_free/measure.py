"""Score the splat_free variants of each room, locally (no GPU, no Modal).

For each room and variant (base / support3 / clean_support3, re-cut by `splatroom --recut
--no-masks --keep-physics` into run/<room>/splat_free_<variant>), the UNCUT trained snapshot
(raw.spz.bin for base, raw.clean.spz.bin for the cleaned ones) is drawn by the evaluation's own
headless Spark page (`SplatRenderer`, software WebGL) from:

* every photo the trainer held out (every 8th in name order) plus a few training photos, and
  scored against the photos with the evaluation's PSNR / SSIM (`fidelity.compare`);
* three "wander" cameras that are not capture views: ~0.8 m off the capture path, at the nearest
  photo's height, looking across the room. They are saved side by side per variant on the call's
  dark background, so holes read as holes, into results/splat_free/<room>_wander.jpg.

Run: uv run --frozen --with playwright python results/splat_free/measure.py [room ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import median_filter

from litereality_agent.models.splat import rgbd
from litereality_agent.pipeline.evaluate import fidelity
from litereality_agent.pipeline.evaluate import holdout as held
from litereality_agent.pipeline.splatroom import capture_positions, room_bounds

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "splat_free"
HOME = Path.home()
ROOMS = {
    "tea_room": HOME / "LiteReality-Agent/example-scans/tea_room",
    "office_room": HOME / "LiteReality-Agent/example-scans/office_room",
    "suite_3693": HOME / "LiteReality-scans/suite_3693",
    "atrium_1059": HOME / "LiteReality-scans/atrium_1059",
}
VARIANTS = ("base", "support3", "clean_support3")
SIZE = fidelity.SIZE                       # metrics at 320x240, as `litereality evaluate`
BIG = (SIZE[0] * fidelity.RENDER_SCALE, SIZE[1] * fidelity.RENDER_SCALE)
TEST_EVERY = 8
TRAIN_VIEWS = 2                            # a few training photos beside the held-out ones
WANDER = 3
OFF_PATH = (0.6, 1.0)                      # metres from the nearest capture spot
DARK = (0x1D, 0x22, 0x27)                  # the call's dark-theme scene background


def snapshot(room: str, variant: str) -> Path:
    folder = REPO / "run" / room / f"splat_free_{variant}"
    return folder / ("raw.spz.bin" if variant == "base" else "raw.clean.spz.bin")


def scored_views(scan: Path) -> tuple[list[fidelity.View], set[int]]:
    """Every held-out photo plus `TRAIN_VIEWS` training ones, and the training frame set."""
    training = fidelity.splat_training_frames(scan, TEST_EVERY)
    depth_frames = {f.index for f in rgbd.load_frames(scan)}
    views = [v for v in fidelity.capture_views(scan) if v.index in depth_frames]
    held_out = [v for v in views if v.index not in training]
    train = [v for v in views if v.index in training]
    picks = np.linspace(0, len(train) - 1, TRAIN_VIEWS).round().astype(int)
    chosen = held_out + [train[i] for i in sorted(set(picks.tolist()))]
    return sorted(chosen, key=lambda v: v.index), training


def _inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon for many points."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = ((y0 > y) != (y1 > y)) & (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0)
        inside ^= cross
    return inside


def _edge_distance(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    best = np.full(len(pts), np.inf)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(pts - (a + t[:, None] * ab), axis=1))
    return best


def wander_views(scan: Path) -> list[dict]:
    """Three cameras off the capture path, inside the room and outside every object box, spread
    apart, each looking at the far side of the room. Returns {"view", "pos", "off"} dicts."""
    frames = rgbd.load_frames(scan)
    spots = np.asarray(capture_positions(frames, most=10_000))
    room = room_bounds(scan, log=lambda _m: None)
    outlines = [np.asarray(p, dtype=float)[:, :2] for p in room.outlines]
    plan = fidelity.parse_room(scan)
    objects, _openings = fidelity.scan_boxes(plan)
    lo = np.min([p.min(axis=0) for p in outlines], axis=0)
    hi = np.max([p.max(axis=0) for p in outlines], axis=0)
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], 0.1), np.arange(lo[1], hi[1], 0.1))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    ok = np.zeros(len(grid), dtype=bool)
    for poly in outlines:
        ok |= _inside(poly, grid) & (_edge_distance(poly, grid) > 0.3)
    for box in objects:                    # not inside tall furniture (a head clears a table)
        if box.centre[2] + box.half[2] < room.floor_z + 1.0:
            continue
        c, s = np.cos(-box.yaw), np.sin(-box.yaw)
        rel = grid - box.centre[:2]
        local = np.column_stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]])
        ok &= ~np.all(np.abs(local) < box.half[:2] + 0.1, axis=1)
    d = np.linalg.norm(grid[:, None, :] - spots[None, :, :2], axis=2)
    near = d.argmin(axis=1)
    off = d.min(axis=1)
    ok &= (off >= OFF_PATH[0]) & (off <= OFF_PATH[1])
    cand = np.flatnonzero(ok)
    if len(cand) == 0:
        raise RuntimeError(f"{scan.name}: no wander spot {OFF_PATH} m off the capture path")
    # Farthest-point sampling from the candidate nearest the room's middle: spread apart.
    centre = grid[cand].mean(axis=0)
    chosen = [cand[np.argmin(np.linalg.norm(grid[cand] - centre, axis=1))]]
    while len(chosen) < min(WANDER, len(cand)):
        dist = np.min(np.linalg.norm(grid[cand][:, None] - grid[chosen][None], axis=2), axis=1)
        chosen.append(cand[int(dist.argmax())])
    corners = np.concatenate(outlines)
    ref = fidelity.capture_views(scan)[0]
    out = []
    for g in chosen:
        xy = grid[g]
        z = float(np.clip(spots[near[g], 2], room.floor_z + 1.1, room.floor_z + 1.7))
        far = corners[np.argmax(np.linalg.norm(corners - xy, axis=1))]
        look = far - xy
        look /= np.linalg.norm(look)
        pitch = np.radians(-8.0)
        fwd = np.array([look[0] * np.cos(pitch), look[1] * np.cos(pitch), np.sin(pitch)])
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        pose = np.eye(4)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, -fwd, [xy[0], xy[1], z]
        view = fidelity.View(-1, ref.image, ref.K, ref.full, pose)
        out.append({"view": view, "pos": [round(float(xy[0]), 2), round(float(xy[1]), 2),
                                          round(z, 2)],
                    "off": round(float(off[g]), 2)})
    return out


def speckle(rgb: np.ndarray, covered: np.ndarray) -> float:
    """Share of covered pixels whose luminance is >0.12 off its 3x3 median: isolated dots."""
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    dev = np.abs(lum - median_filter(lum, size=3))
    mask = covered > 0.5
    return float((dev[mask] > 0.12).mean()) if mask.any() else 0.0


def label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return img


def measure(room: str) -> dict:
    scan = ROOMS[room]
    views, training = scored_views(scan)
    wander = wander_views(scan)
    plan = fidelity.parse_room(scan)
    objects, openings = fidelity.scan_boxes(plan)
    regions = {v.index: fidelity.view_regions(v, plan, objects, openings, SIZE) for v in views}
    photos = {v.index: fidelity.load_photo(v.image, SIZE) for v in views}
    sources = held.Sources(True, {"training": sorted(training)})
    cams = ([fidelity.three_camera(v, BIG) for v in views]
            + [fidelity.three_camera(w["view"], BIG) for w in wander])
    result: dict = {"room": room, "scan": str(scan), "views": [v.index for v in views],
                    "held_out": [v.index for v in views if v.index not in training],
                    "wander": [{"pos": w["pos"], "off_path_m": w["off"]} for w in wander],
                    "variants": {}}
    tiles: dict[str, list[Image.Image]] = {}
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True)
    for variant in VARIANTS:
        path = snapshot(room, variant)
        done = cache / f"{room}_{variant}.json"
        evals = [cache / f"{room}_{variant}_eval_{v.index}.png" for v in views]
        if done.is_file() and all(e.is_file() for e in evals):
            # A render costs ~50 s a view in software WebGL: keep it.
            result["variants"][variant] = json.loads(done.read_text())
            tiles[variant] = [Image.open(cache / f"{room}_{variant}_{i}.png").convert("RGB")
                              for i in range(len(wander))]
            continue
        spz = path.read_bytes()
        renderer = fidelity.SplatRenderer()
        t = time.perf_counter()
        renders = renderer.render(spz, cams, BIG)
        seconds = time.perf_counter() - t
        rows = []
        for view, rgba, keep in zip(views, renders[:len(views)], evals, strict=True):
            Image.fromarray(rgba).save(keep)   # for the shared-coverage scores (README)
            render, coverage = fidelity.composite(rgba, SIZE)
            reg = regions[view.index]
            row = {"frame": view.index,
                   **fidelity.compare(photos[view.index], render,
                                      {"static": reg["static"], "objects": reg["objects"]}),
                   "coverage": round(float(coverage.mean()), 4),
                   "seen": sources.seen(view.index)}
            rows.append(row)
        wander_stats, tiles[variant] = [], []
        for w, rgba in zip(wander, renders[len(views):], strict=True):
            rgb, cov = fidelity.composite(rgba, BIG, background=DARK)
            wander_stats.append({"holes": round(float((cov < 0.5).mean()), 4),
                                 "speckle": round(speckle(rgb, cov), 4)})
            tiles[variant].append(fidelity._to_image(rgb))
        meta = json.loads((path.parent / "report.json").read_text())
        clean = next((b.get("clean") for b in reversed(meta.get("bundles", []))
                      if b.get("stage") == "trained"), None)
        result["variants"][variant] = {
            "snapshot": str(path.relative_to(REPO)), "splats": _count(spz),
            "clean": clean, "render_s": round(seconds, 1),
            "held_out": fidelity._summarise([r for r in rows if r["seen"] is False]),
            "training": fidelity._summarise([r for r in rows if r["seen"] is True]),
            "per_view": rows, "wander": wander_stats}
        done.write_text(json.dumps(result["variants"][variant], indent=1))
        for i, tile in enumerate(tiles[variant]):
            tile.save(cache / f"{room}_{variant}_{i}.png")
        print(f"  {room} {variant}: {seconds:.0f} s render", flush=True)
    # Rows = wander cameras, columns = variants.
    w, h = BIG
    sheet = Image.new("RGB", (w * len(VARIANTS), h * len(wander)), DARK)
    for c, variant in enumerate(VARIANTS):
        for r, tile in enumerate(tiles[variant]):
            s = result["variants"][variant]["wander"][r]
            sheet.paste(label(tile.copy(), f"{room} · {variant} · spot {r + 1} "
                              f"({wander[r]['off']} m off path) · holes {s['holes']:.1%} "
                              f"· speckle {s['speckle']:.2%}"), (c * w, r * h))
    sheet.save(OUT / f"{room}_wander.jpg", quality=88)
    return result


def _count(spz: bytes) -> int:
    from litereality_agent.models.splat import spz as spz_io

    return len(spz_io.decode(spz))


def shared(room: str) -> dict:
    """PSNR per variant over only the pixels EVERY variant covers (alpha >= 0.99) in the saved
    held-out renders: how good the splats that remain look, with the holes taken out."""
    scan = ROOMS[room]
    views, training = scored_views(scan)
    out: dict = {}
    for variant in VARIANTS:
        out[variant] = {"held_out": [], "training": []}
    for view in views:
        photo = fidelity.load_photo(view.image, SIZE)
        rgbs, covs = {}, []
        for variant in VARIANTS:
            rgba = np.asarray(Image.open(OUT / "cache" / f"{room}_{variant}_eval_{view.index}.png"))
            rgbs[variant], cov = fidelity.composite(rgba, SIZE)
            covs.append(cov)
        mask = np.min(covs, axis=0) >= 0.99
        split = "training" if view.index in training else "held_out"
        for variant in VARIANTS:
            out[variant][split].append((fidelity.psnr(photo, rgbs[variant], mask),
                                        float(mask.mean())))
    return {v: {k: {"psnr": round(float(np.mean([p for p, _ in rows if p is not None])), 3) if any(p is not None for p, _ in rows) else None,
                    "shared": round(float(np.mean([m for _, m in rows])), 4) if rows else None}
                for k, rows in d.items()} for v, d in out.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:2] == ["--shared"]:
        rooms = sys.argv[2:] or list(ROOMS)
        print(json.dumps({r: shared(r) for r in rooms}, indent=1))
        return
    # tea_room last: its first pass kept no per-view renders and is the one redone.
    rooms = sys.argv[1:] or ["office_room", "suite_3693", "atrium_1059", "tea_room"]
    for room in rooms:
        t = time.perf_counter()
        result = measure(room)
        result["seconds"] = round(time.perf_counter() - t, 1)
        (OUT / f"{room}.json").write_text(json.dumps(result, indent=1))
        print(f"{room}: done in {result['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
