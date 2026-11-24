"""Screenshots of the local call at maxPixelRadius 512 vs 1e5 per view (how many pixels the cap changes).
Usage: ... cap_shot.py "http://127.0.0.1:8791/?debug=1" [views.json]; PNGs go beside this script."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from join import join

S = Path(__file__).parent


def psnr(x, y):
    return 10 * np.log10(255**2 / max(np.mean((x - y) ** 2), 1e-9))


ARGS = ["--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"]
src = (Path.home() / "LiteReality/src/litereality_agent/room_ops/roomcall/web/app.js").read_text()
body = src.replace("const sr = new mod.SparkRenderer(", "const sr = window.__sr = new mod.SparkRenderer(")
VIEWS = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else S / "fps_views_close.json").read_text())
with sync_playwright() as pw:
    b = pw.chromium.launch(args=ARGS, headless=True)
    page = b.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1).new_page()
    page.route("**/app.js", lambda r: r.fulfill(status=200, content_type="text/javascript", body=body))
    page.goto(sys.argv[1])
    for _ in range(240):
        ph = page.evaluate("() => window.__roomcall?.photo?.() ?? null")
        if ph and ph.get("shown") and ph.get("stage") == "trained" and page.evaluate("() => !!window.__sr"):
            break
        time.sleep(1)
    join(page)
    time.sleep(3)
    for i in range(len(VIEWS)):
        page.evaluate("([a, b]) => window.__roomcall.look(a, b)", VIEWS[i])
        imgs = {}
        for c in (512, 100000, 512):
            page.evaluate("(c) => { window.__sr.maxPixelRadius = c; }", c)
            time.sleep(2.5)
            p = S / f"shot_v{i}_{c}.png"
            page.screenshot(path=str(p), timeout=60000)
            imgs.setdefault(c, []).append(np.asarray(Image.open(p).convert("RGB")).astype(float))
        a, a2, bb = imgs[512][0], imgs[512][1], imgs[100000][0]
        print(f"view {i}: 512 vs 512 repeat PSNR {psnr(a, a2):.2f}; 512 vs 1e5 PSNR {psnr(a, bb):.2f}, "
              f"changed px >8: {np.mean(np.abs(a - bb).max(-1) > 8):.4f}", flush=True)
    b.close()
