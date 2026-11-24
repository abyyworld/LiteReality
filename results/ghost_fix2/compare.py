"""Before (v2_chairs_fix, a4a4ea4, shot by the round-1 skeptic) against after (v2_chairs_fix2):
per body and viewpoint, the room with every body faded out and the body carried off (grabbed at
its centre of mass; the door pulled open), centre crops of the rest spot.

    uv run python results/ghost_fix2/compare.py BEFORE_RAW AFTER_RAW OUTDIR
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

BEFORE, AFTER, OUT = (Path(a) for a in sys.argv[1:4])
W, H = 640, 360
for body in ("Chair0", "Chair1", "Chair2", "Wall0_Door_0"):
    door = "Door" in body
    cols = [("before", BEFORE, "{b}_{v}_roomonly", "room only"),
            ("before", BEFORE, "{b}_{v}_carried" if not door else "{b}_{v}_carried", "carried"),
            ("after", AFTER, "{b}_{v}_roomonly", "room only"),
            ("after", AFTER, "{b}_com_{v}_carried" if not door else "{b}_{v}_open", "carried")]
    sheet = Image.new("RGB", (W * len(cols), H * 3))
    for r, v in enumerate(("cap", "far", "low")):
        for c, (tag, folder, pattern, label) in enumerate(cols):
            f = folder / (pattern.format(b=body, v=v) + ".png")
            if not f.is_file():
                continue
            im = Image.open(f).convert("RGB").crop((320, 180, 1600, 900)).resize((W, H))
            ImageDraw.Draw(im).text((8, 8), f"{tag} {body} {v} {label}", fill=(255, 255, 0))
            sheet.paste(im, (c * W, r * H))
    sheet.save(OUT / f"compare_{body}.jpg", quality=85)
