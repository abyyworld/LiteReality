"""Photo | v2 | v4 at the photo's full 1920 x 1440 for chosen held-out eval views.

Both bundles' raw.spz.bin are drawn with the room call's Spark settings (SPARK_OPTIONS in
roomcall web/app.js) on Metal, each in its own scan's world (score.py's eval_views), exactly as
score.py does. Each panel is labelled with its full-resolution PSNR / PSNR_cc / LPIPS.

    uv run --frozen --with playwright --with lpips==0.1.4 python \\
        results/fix_v4/final/side_by_side.py SET:ROW[:TAG] ...
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results" / "stray_quality"))

import score as sc  # noqa: E402

BUNDLES = {"v2": REPO / "run/atrium_stray_q/v2", "v4": REPO / "run/atrium_stray_q/v4"}


def main(picks: list[str]) -> None:
    fd = sc._fd()
    spark = sc.spark_settings()
    renderer = fd.SplatRenderer(timeout_s=3600, spark=spark, gl="metal", settle_ms=1000)
    lp = sc.Lpips()
    wanted = [p.split(":") for p in picks]
    renders: dict[str, dict] = {}
    photos: dict[tuple, tuple] = {}
    for name, out_dir in BUNDLES.items():
        scan = Path(sc.load_run(out_dir)["scan"]).expanduser()
        spz = (out_dir / "raw.spz.bin").read_bytes()
        cams, keys = [], []
        for w in wanted:
            rows_file, pool = sc.EVAL_SETS[w[0]]
            views, _ = sc.eval_views(scan, pool, rows_file)
            v = next(v for v in views if v.index == int(w[1]))
            cams.append(fd.three_camera(v, tuple(v.full)))
            keys.append((w[0], int(w[1])))
            photos[keys[-1]] = (v.image, tuple(v.full))
        size = photos[keys[0]][1]
        for key, rgba in zip(keys, renderer.render(spz, cams, size), strict=True):
            renders.setdefault(key, {})[name] = fd.composite(rgba, size)[0]
    for w in wanted:
        key = (w[0], int(w[1]))
        image, size = photos[key]
        real = fd.load_photo(image, size)
        tiles = [sc._label(sc._to_img(real), f"photo  {w[0]} row {key[1]} (held out)", 40)]
        for name in BUNDLES:
            r = renders[key][name]
            tiles.append(sc._label(sc._to_img(r), f"{name}  PSNR {fd.psnr(real, r):.2f}  cc "
                                   f"{fd.psnr(real, sc.colour_fit(r, real)):.2f}  LPIPS "
                                   f"{lp(real, r):.3f}", 40))
        canvas = Image.new("RGB", (size[0] * 3, size[1]))
        for k, t in enumerate(tiles):
            canvas.paste(t, (k * size[0], 0))
        tag = f"_{w[2]}" if len(w) > 2 else ""
        path = HERE / f"side_{w[0]}_row{key[1]}{tag}.jpg"
        canvas.save(path, quality=88)
        print(path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
