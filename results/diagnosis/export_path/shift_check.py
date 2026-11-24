"""Is the Spark render registered to the photo? PSNR of each base render shifted by whole pixels
(960x720) against the photo; a best shift away from (0, 0) means a camera/projection offset."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from litereality_agent.pipeline.evaluate import fidelity as fd
H = Path(__file__).resolve().parent
scan = Path("~/LiteReality-scans/stray_atrium_q600").expanduser()
frames = json.loads((H / "variants_9views.json").read_text())["setup"]["frames"]
rows = []
for idx in frames:
    real = fd.load_photo(scan / f"frame_{idx:05d}.jpg", (960, 720))
    rend, _ = fd.composite(np.asarray(Image.open(H / f"renders/base_{idx:05d}.png").convert("RGBA")), (960, 720))
    m = 8
    best = None
    grid = {}
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            a = real[m:-m, m:-m]
            b = rend[m + dy:720 - m + dy, m + dx:960 - m + dx]
            p = float(10 * np.log10(1 / np.mean((a - b) ** 2)))
            grid[(dx, dy)] = p
            if best is None or p > best[0]:
                best = (p, dx, dy)
    rows.append({"frame": idx, "psnr_0": round(grid[(0, 0)], 3), "best_psnr": round(best[0], 3),
                 "best_dx": best[1], "best_dy": best[2]})
    print(rows[-1], flush=True)
(H / "shift_check.json").write_text(json.dumps(rows, indent=1))
