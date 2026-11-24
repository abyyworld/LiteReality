"""Contact sheets of callcheck.py's raw shots: per body, a row per viewpoint (cap, far, low) and
a column per state, full frames scaled down or the centre crop (the body's rest spot).

    uv run python results/ghost_fix2/sheets.py RAWDIR SHEETDIR [TAG]
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
OUT.mkdir(parents=True, exist_ok=True)
COLS = {"chair": ["before", "roomonly", "com_{v}_carried", "com_{v}_carried_nohold",
                  "back_top_{v}_carried", "seat_corner_{v}_carried"],
        "door": ["before", "roomonly", "open"]}
W, H = 640, 360
for body in ("Chair0", "Chair1", "Chair2", "Wall0_Door_0"):
    cols = COLS["door" if "Door" in body else "chair"]
    for kind in ("full", "crop"):
        sheet = Image.new("RGB", (W * len(cols), H * 3))
        for r, v in enumerate(("cap", "far", "low")):
            for c, col in enumerate(cols):
                name = col.format(v=v) if "{v}" in col else f"{v}_{col}"
                f = RAW / f"{body}_{name}.png"
                if not f.is_file():
                    continue
                im = Image.open(f).convert("RGB")
                im = im.crop((480, 270, 1440, 810)) if kind == "crop" else im
                im = im.resize((W, H))
                ImageDraw.Draw(im).text((8, 8), f"{TAG} {body} {name}", fill=(255, 255, 0))
                sheet.paste(im, (c * W, r * H))
        sheet.save(OUT / f"sheet_{body}_{kind}.jpg", quality=85)
