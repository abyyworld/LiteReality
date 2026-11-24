"""Skeptic check of v4's grabbable chairs in the real room call (local server, headless Chromium on Metal).

Usage (call served first):
  uv run --frozen litereality roomcall run/atrium_stray_q/v4/physics/scene_splat.xml \
      --splat run/atrium_stray_q/v4 --no-open --port 8793 --avatars off
  uv run --frozen --with playwright python results/fix_v4/movecheck/movecheck.py "http://127.0.0.1:8793/?debug=1"

For each grabbable body: two viewpoints, a "before" shot, then the call's own grab/move protocol
(lift 0.5 m and carry 0.6 m sideways, held there), "held" shots from the same viewpoints, every
other body's position while held and after release, then a room reset. Also console errors,
the live Spark settings and fps."""
import json
import math
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path.home() / "LiteReality/results/spark_cap"))
from join import join  # noqa: E402

OUT = Path(__file__).parent
URL = sys.argv[1]
ARGS = ["--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"]
APP = Path.home() / "LiteReality/src/litereality_agent/room_ops/roomcall/web/app.js"
ROOM_C = (-0.15, 0.65)
FPS_JS = """async (ms) => {
  const t0 = performance.now(); let n = 0;
  await new Promise((res) => { const f = () => { n++; if (performance.now() - t0 < ms) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
  return n * 1000 / (performance.now() - t0);
}"""


def views_for(p):
    dx, dy = ROOM_C[0] - p[0], ROOM_C[1] - p[1]
    n = math.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    out = []
    for ang in (math.radians(35), math.radians(-35)):
        c, s = math.cos(ang), math.sin(ang)
        vx, vy = dx * c - dy * s, dx * s + dy * c
        out.append([[p[0] + 1.5 * vx, p[1] + 1.5 * vy, 0.0], [p[0], p[1], p[2] - 0.1]])
    return out, (dx, dy)


def main():
    src = APP.read_text()
    for needle in ("const sr = new mod.SparkRenderer(", "window.__roomcall = {"):
        assert src.count(needle) == 1, needle
    body = src.replace("const sr = new mod.SparkRenderer(", "const sr = window.__sr = new mod.SparkRenderer(")
    body = body.replace("window.__roomcall = {",
                        "window.__roomcall = { send: (m) => send(m), ver: () => shown?.v, "
                        "phot: () => photo, ")
    logs = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=ARGS, headless=True)
        page = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1).new_page()
        page.on("console", lambda m: logs.append({"type": m.type, "text": m.text[:400]}))
        page.on("pageerror", lambda e: logs.append({"type": "pageerror", "text": str(e)[:400]}))
        page.route("**/app.js", lambda r: r.fulfill(status=200, content_type="text/javascript", body=body))
        page.goto(URL)
        t = time.time()
        while time.time() - t < 300:
            ph = page.evaluate("() => window.__roomcall?.photo?.() ?? null")
            if ph and ph.get("shown") and ph.get("stage") == "trained" and page.evaluate("() => !!window.__sr"):
                break
            time.sleep(1)
        else:
            raise SystemExit("photoreal room never shown")
        join(page)
        time.sleep(4)
        info = {"photo": page.evaluate("() => window.__roomcall.photo()"),
                "spark": page.evaluate("() => ({maxPixelRadius: __sr.maxPixelRadius, maxStdDev: __sr.maxStdDev})"),
                "gpu": page.evaluate("() => { const gl = document.createElement('canvas').getContext('webgl2'); const d = gl.getExtension('WEBGL_debug_renderer_info'); return gl.getParameter(d.UNMASKED_RENDERER_WEBGL); }"),
                "photo_bodies_visible": page.evaluate("() => [...window.__roomcall.phot().bodies].map(([n, m]) => [n, m.visible, m.parent?.type ?? null, m.parent?.name ?? null])")}
        bodies = page.evaluate("() => window.__roomcall.bodies()")
        info["bodies"] = bodies
        print(json.dumps(info, indent=1), flush=True)
        page.screenshot(path=str(OUT / "default_view.png"))
        fps = {"default_view": round(page.evaluate(FPS_JS, 4000), 1)}

        def pos_all():
            return {b["name"]: [round(x, 3) for x in b["p"]] for b in page.evaluate("() => window.__roomcall.bodies()")}

        def shot(view, path):
            page.evaluate("([a, b]) => window.__roomcall.look(a, b)", view)
            time.sleep(2.5)
            page.screenshot(path=str(path))

        targets = [b for b in bodies if b["name"].startswith("Chair") or "Door" in b["name"]]
        results = {}
        for b in targets:
            name, bid, p = b["name"], b["id"], b["p"]
            views, (dx, dy) = views_for(p)
            if "Door" in name:   # a door swings: pull the handle side 0.4 m toward the room
                lift, side = 0.0, (dx * 0.4, dy * 0.4)
            else:
                lift = 0.5
                others = [o["p"] for o in bodies if o["name"].startswith("Chair") and o["id"] != bid]
                best = None
                for sgn in (1, -1):
                    sx, sy = -dy * 0.6 * sgn, dx * 0.6 * sgn
                    q = (p[0] + sx, p[1] + sy)
                    clear = min(math.hypot(q[0] - o[0], q[1] - o[1]) for o in others)
                    if best is None or clear > best[0]:
                        best = (clear, (sx, sy))
                side = best[1]
            target = [p[0] + side[0], p[1] + side[1], p[2] + lift]
            before_pos = pos_all()
            for k, v in enumerate(views):
                shot(v, OUT / f"{name}_v{k}_before.png")
            if "Door" not in name:
                fps[f"{name}_v0"] = round(page.evaluate(FPS_JS, 3000), 1)
            v_now = page.evaluate("() => window.__roomcall.ver()")
            page.evaluate("(m) => window.__roomcall.send(m)", {"t": "grab", "v": v_now, "body": bid, "point": p})
            steps = 20
            for i in range(1, steps + 1):   # carry it over 2 s like a hand would
                f = i / steps
                tgt = [p[0] + side[0] * f, p[1] + side[1] * f, p[2] + lift * min(1.0, 2 * f)]
                page.evaluate("(m) => window.__roomcall.send(m)", {"t": "move", "target": tgt})
                time.sleep(0.1)
            time.sleep(1.5)
            held_pos = pos_all()
            for k, v in enumerate(views):
                page.evaluate("(m) => window.__roomcall.send(m)", {"t": "move", "target": target})
                shot(v, OUT / f"{name}_v{k}_held.png")
            held_pos2 = pos_all()
            page.evaluate("(m) => window.__roomcall.send(m)", {"t": "release"})
            time.sleep(3)
            after_pos = pos_all()
            moved_self = math.dist(before_pos[name], held_pos[name])
            others_moved = {o: round(math.dist(before_pos[o], held_pos2[o]), 3)
                            for o in before_pos if o != name and math.dist(before_pos[o], held_pos2[o]) > 0.01}
            others_after = {o: round(math.dist(before_pos[o], after_pos[o]), 3)
                            for o in before_pos if o != name and math.dist(before_pos[o], after_pos[o]) > 0.01}
            results[name] = {"id": bid, "views": views, "target": target, "before": before_pos[name],
                             "held": held_pos[name], "held_end": held_pos2[name], "after_release": after_pos[name],
                             "self_moved_m": round(moved_self, 3), "others_moved_while_held": others_moved,
                             "others_moved_after_release": others_after}
            print(name, json.dumps(results[name]), flush=True)
            # side-by-side sheet: rows = views, cols = before | held
            sheet = Image.new("RGB", (1920, 1080))
            for k in range(2):
                for j, tag in enumerate(("before", "held")):
                    im = Image.open(OUT / f"{name}_v{k}_{tag}.png").convert("RGB").resize((960, 540))
                    ImageDraw.Draw(im).text((10, 10), f"{name} view{k} {tag}", fill=(255, 255, 0))
                    sheet.paste(im, (j * 960, k * 540))
            sheet.save(OUT / f"sheet_{name}.jpg", quality=90)
            page.evaluate("() => window.__roomcall.send({t: 'reset'})")
            time.sleep(4)
            back = pos_all()
            results[name]["reset_ok"] = all(math.dist(back[o], before_pos[o]) < 0.02 for o in back)
        fps["default_view_end"] = round(page.evaluate(FPS_JS, 4000), 1)
        browser.close()
    errs = [m for m in logs if m["type"] in ("error", "pageerror")]
    warns = [m for m in logs if m["type"] == "warning"]
    out = {"url": URL, "info": info, "fps": fps, "results": results, "console_errors": errs,
           "console_warnings": warns, "console_all": logs[-200:]}
    (OUT / "movecheck.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"fps": fps, "errors": errs, "warnings": warns[:20]}, indent=1))


if __name__ == "__main__":
    main()
