"""Usage: uv run --frozen --with playwright python results/spark_cap/fps_ab.py \
    "http://127.0.0.1:8791/?debug=1" 512,2048,100000 out.json   [VIEWS=fps_views_close.json]
with the call served locally:  uv run --frozen litereality roomcall
    run/atrium_stray_q/v2/physics/scene_splat.xml --splat run/atrium_stray_q/v2 --no-open --port 8791 --avatars off

FPS of the local room call (photoreal atrium v2), headless Chromium on the Mac GPU (ANGLE Metal),
1920x1080 at device pixel ratio 1, the call's own page and render loop. One browser; the Spark
renderer's maxPixelRadius is switched at runtime and the caps are interleaved per view (A/B/A/B),
so thermal drift hits every cap alike."""
import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from join import join

URL = sys.argv[1]
CAPS = [float(c) for c in sys.argv[2].split(",")]
OUT = Path(sys.argv[3])
REPS = int(os.environ.get("REPS", "2"))
WARM_MS, MEASURE_MS = 1500, 4000
ARGS = ["--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"]
ARGS += os.environ.get("ARGS", "").split()
APP = Path.home() / "LiteReality/src/litereality_agent/room_ops/roomcall/web/app.js"
VIEWS = json.loads(Path(os.environ.get("VIEWS", Path(__file__).with_name("fps_views_main.json"))).read_text())
FPS_JS = """async (ms) => {
  const t0 = performance.now(); let n = 0;
  await new Promise((res) => { const f = () => { n++; if (performance.now() - t0 < ms) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
  return n * 1000 / (performance.now() - t0);
}"""


def main():
    src = APP.read_text()
    needle = "const sr = new mod.SparkRenderer("
    assert src.count(needle) == 1, "SparkRenderer construction not found"
    body = src.replace(needle, "const sr = window.__sr = new mod.SparkRenderer(")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=ARGS, headless=True)
        page = browser.new_context(viewport={"width": 1920, "height": 1080},
                                   device_scale_factor=1).new_page()
        page.route("**/app.js", lambda r: r.fulfill(status=200, content_type="text/javascript", body=body))
        page.goto(URL)
        gpu = page.evaluate("() => { const gl = document.createElement('canvas').getContext('webgl2'); const d = gl.getExtension('WEBGL_debug_renderer_info'); return gl.getParameter(d.UNMASKED_RENDERER_WEBGL); }")
        t = time.time()
        while time.time() - t < 240:
            ph = page.evaluate("() => window.__roomcall?.photo?.() ?? null")
            if ph and ph.get("shown") and ph.get("stage") == "trained" and page.evaluate("() => !!window.__sr"):
                break
            time.sleep(1)
        else:
            raise SystemExit("photoreal room never shown")
        join(page)
        default = page.evaluate("() => ({maxPixelRadius: __sr.maxPixelRadius, maxStdDev: __sr.maxStdDev, size: [__sr.renderer?.domElement?.width, __sr.renderer?.domElement?.height]})")
        size = page.evaluate("() => [document.querySelector('canvas').width, document.querySelector('canvas').height]")
        time.sleep(3)
        fps = {c: [[] for _ in VIEWS] for c in CAPS}
        for i, (frm, at) in enumerate(VIEWS):
            page.evaluate("([a, b]) => window.__roomcall.look(a, b)", [frm, at])
            page.evaluate(FPS_JS, 2500)
            for _ in range(REPS):
                for c in CAPS:
                    page.evaluate("(c) => { window.__sr.maxPixelRadius = c; }", c)
                    page.evaluate(FPS_JS, WARM_MS)
                    fps[c][i].append(round(page.evaluate(FPS_JS, MEASURE_MS), 2))
            print(i, {c: fps[c][i] for c in CAPS}, flush=True)
        browser.close()
    summary = {}
    for c in CAPS:
        per = [sum(v) / len(v) for v in fps[c]]
        summary[str(c)] = {"per_view": [round(x, 1) for x in per], "mean": round(sum(per) / len(per), 1),
                           "min": round(min(per), 1)}
    out = {"url": URL, "gpu": gpu, "canvas": size, "page_default": default, "views_mujoco": VIEWS,
           "reps": REPS, "warm_ms": WARM_MS, "measure_ms": MEASURE_MS, "args": ARGS,
           "raw": {str(c): fps[c] for c in CAPS}, "summary": summary}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"gpu": gpu, "canvas": size, "default": default, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
