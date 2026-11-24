"""Second pass of the v4 chair skeptic check (see movecheck.py for how to serve the call).

  uv run --frozen --with playwright python results/fix_v4/movecheck/movecheck2.py "http://127.0.0.1:8793/?debug=1" OUTDIR

Per chair: two viewpoints taken from the phone's own capture path (sharpest renders), each aimed at
the chair so its original spot is the screen centre. Phase 1 lifts the chair straight up 0.9 m
(nothing lateral, so a neighbour that moves is bound to it, not knocked); phase 2 carries it 0.7 m
sideways at that height, away from the other chairs. Shots: before / lifted / carried, full frame
and a 960x540 centre crop (the chair's original spot) for ghosts. The door is pulled at its handle.
Every body's position and rotation is logged at each phase; the room is reset between bodies."""
import json
import math
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path.home() / "LiteReality/results/spark_cap"))
from join import join  # noqa: E402

URL, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
ARGS = ["--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"]
REPO = Path.home() / "LiteReality"
APP = REPO / "src/litereality_agent/room_ops/roomcall/web/app.js"
BUNDLE = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "run/atrium_stray_q/v4"
CAPTURE = json.loads((BUNDLE / "splat.json").read_text())["capture"]
FPS_JS = """async (ms) => {
  const t0 = performance.now(); let n = 0;
  await new Promise((res) => { const f = () => { n++; if (performance.now() - t0 < ms) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
  return n * 1000 / (performance.now() - t0);
}"""
STATE_JS = """() => [...window.__roomcall.sh().bodies.values()].map((b) => [b.name,
  b.node.position.toArray(), b.node.quaternion.toArray()])"""


def qangle(a, b):
    d = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def views_for(p):
    """Two capture-path points 0.9-2.4 m from the chair (horizontal), as far apart in bearing as possible."""
    cands = []
    for c in CAPTURE[::2]:
        d = math.hypot(c[0] - p[0], c[1] - p[1])
        if 0.9 <= d <= 2.4:
            cands.append((math.atan2(c[1] - p[1], c[0] - p[0]), c))
    if not cands:
        cands = [(math.atan2(c[1] - p[1], c[0] - p[0]), c) for c in CAPTURE[::2]]
    best = (-1, cands[0][1], cands[-1][1])
    for i, (a, c) in enumerate(cands):
        for b, e in cands[i + 1:]:
            sep = abs((a - b + math.pi) % (2 * math.pi) - math.pi)
            if sep > best[0]:
                best = (sep, c, e)
    aim = [p[0], p[1], p[2] - 0.1]
    return [[list(best[1]), aim], [list(best[2]), aim]], math.degrees(best[0])


def main():
    src = APP.read_text()
    for needle in ("const sr = new mod.SparkRenderer(", "window.__roomcall = {"):
        assert src.count(needle) == 1, needle
    body = src.replace("const sr = new mod.SparkRenderer(", "const sr = window.__sr = new mod.SparkRenderer(")
    body = body.replace("window.__roomcall = {",
                        "window.__roomcall = { send: (m) => send(m), ver: () => shown?.v, sh: () => shown, "
                        "phot: () => photo, ")
    logs, bad = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=ARGS, headless=True)
        page = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1).new_page()
        page.on("console", lambda m: logs.append({"type": m.type, "text": m.text[:400]}))
        page.on("pageerror", lambda e: logs.append({"type": "pageerror", "text": str(e)[:400]}))
        page.on("response", lambda r: bad.append([r.status, r.url]) if r.status >= 400 else None)
        page.route("**/app.js", lambda r: r.fulfill(status=200, content_type="text/javascript", body=body))
        t0 = time.time()
        page.goto(URL)
        while time.time() - t0 < 300:
            ph = page.evaluate("() => window.__roomcall?.photo?.() ?? null")
            if ph and ph.get("shown") and ph.get("stage") == "trained" and page.evaluate("() => !!window.__sr"):
                break
            time.sleep(1)
        else:
            raise SystemExit("photoreal room never shown")
        load_s = round(time.time() - t0, 1)
        join(page)
        page.add_style_tag(content="#help{display:none!important}")
        time.sleep(4)
        info = {"load_s": load_s, "photo": page.evaluate("() => window.__roomcall.photo()"),
                "spark": page.evaluate("() => ({maxPixelRadius: __sr.maxPixelRadius, maxStdDev: __sr.maxStdDev})")}
        page.screenshot(path=str(OUT / "default_view.png"))
        fps = {"default_view": round(page.evaluate(FPS_JS, 4000), 1)}

        def state():
            return {n: (p, q) for n, p, q in page.evaluate(STATE_JS)}

        def diff(a, b, skip):
            out = {}
            for n in a:
                if n == skip:
                    continue
                dp, dq = math.dist(a[n][0], b[n][0]), qangle(a[n][1], b[n][1])
                if dp > 0.01 or dq > 1.0:
                    out[n] = {"moved_m": round(dp, 3), "turned_deg": round(dq, 1)}
            return out

        def shoot(views, name, tag):
            for k, v in enumerate(views):
                page.evaluate("([a, b]) => window.__roomcall.look(a, b)", v)
                time.sleep(2.5)
                page.screenshot(path=str(OUT / f"{name}_v{k}_{tag}.png"))

        def hold(target, secs):
            n = int(secs * 10)
            for _ in range(n):
                page.evaluate("(m) => window.__roomcall.send(m)", {"t": "move", "target": target})
                time.sleep(0.1)

        bodies = page.evaluate("() => window.__roomcall.bodies()")
        chairs = [b for b in bodies if b["name"].startswith("Chair")]
        results = {}
        for b in bodies:
            name, bid, p = b["name"], b["id"], b["p"]
            door = "Door" in name
            s0 = state()
            v_now = page.evaluate("() => window.__roomcall.ver()")
            if door:
                q = s0[name][1]   # three quaternion of the node, MuJoCo frame: rotate local axes
                ang = 2 * math.atan2(q[2], q[3])
                ux, uy = math.cos(ang), math.sin(ang)
                handle = [p[0] + 0.85 * ux, p[1] + 0.85 * uy, p[2] + 1.0]
                aim = [p[0] + 0.45 * ux, p[1] + 0.45 * uy, p[2] + 1.0]
                views = [[[-0.2, 0.0, 0.1], aim], [[0.3, 1.0, 0.1], aim]]
                sep = None
            else:
                views, sep = views_for(p)
            shoot(views, name, "before")
            if not door:
                fps[f"{name}_v0"] = round(page.evaluate(FPS_JS, 3000), 1)
            rec = {"id": bid, "views": views, "view_separation_deg": sep, "start": s0[name]}
            if door:
                best = None
                for sgn in (1, -1):
                    page.evaluate("(m) => window.__roomcall.send(m)", {"t": "grab", "v": v_now, "body": bid, "point": handle})
                    tgt = [handle[0] - uy * 0.6 * sgn, handle[1] + ux * 0.6 * sgn, handle[2]]
                    hold(tgt, 2.5)
                    s1 = state()
                    turned = qangle(s0[name][1], s1[name][1])
                    if best is None or turned > best[0]:
                        best = (turned, sgn, tgt)
                    if turned > 10:
                        break
                    page.evaluate("(m) => window.__roomcall.send(m)", {"t": "release"})
                    time.sleep(1)
                rec.update(handle=handle, pull_target=best[2], door_turned_deg=round(best[0], 1))
                shoot(views, name, "lifted")
                for k in range(2):   # keep naming uniform: carried == lifted for the door
                    (OUT / f"{name}_v{k}_carried.png").write_bytes((OUT / f"{name}_v{k}_lifted.png").read_bytes())
                s2 = s3 = state()
                rec["others_while_held"] = diff(s0, s3, name)
            else:
                page.evaluate("(m) => window.__roomcall.send(m)", {"t": "grab", "v": v_now, "body": bid, "point": p})
                up = [p[0], p[1], p[2] + 0.9]
                for i in range(1, 11):
                    hold([p[0], p[1], p[2] + 0.9 * i / 10], 0.1)
                hold(up, 1.5)
                s1 = state()
                rec["lifted"] = s1[name]
                rec["others_on_straight_lift"] = diff(s0, s1, name)
                shoot(views, name, "lifted")
                others = [c["p"] for c in chairs if c["id"] != bid]
                best = None
                for k in range(16):
                    a = 2 * math.pi * k / 16
                    qx, qy = p[0] + 0.7 * math.cos(a), p[1] + 0.7 * math.sin(a)
                    clear = min(math.hypot(qx - o[0], qy - o[1]) for o in others)
                    if best is None or clear > best[0]:
                        best = (clear, [qx, qy, p[2] + 0.9])
                carry = best[1]
                for i in range(1, 11):
                    hold([up[0] + (carry[0] - up[0]) * i / 10, up[1] + (carry[1] - up[1]) * i / 10, up[2]], 0.1)
                hold(carry, 1.5)
                s2 = state()
                rec["carry_target"], rec["carried"] = carry, s2[name]
                for k, v in enumerate(views):
                    hold(carry, 0.3)
                    page.evaluate("([a, b]) => window.__roomcall.look(a, b)", v)
                    time.sleep(2.5)
                    page.screenshot(path=str(OUT / f"{name}_v{k}_carried.png"))
                s3 = state()
                rec["others_while_held"] = diff(s0, s3, name)
                rec["self_moved_m"] = round(math.dist(s0[name][0], s3[name][0]), 3)
            page.evaluate("(m) => window.__roomcall.send(m)", {"t": "release"})
            time.sleep(3)
            s4 = state()
            rec["others_after_release"] = diff(s0, s4, name)
            rec["self_after_release_m"] = round(math.dist(s0[name][0], s4[name][0]), 3)
            # sheets: full frames (before | lifted | carried) and centre crops (the original spot)
            for kind in ("full", "crop"):
                sheet = Image.new("RGB", (3 * 960, 2 * 540))
                for k in range(2):
                    for j, tag in enumerate(("before", "lifted", "carried")):
                        im = Image.open(OUT / f"{name}_v{k}_{tag}.png").convert("RGB")
                        im = im.resize((960, 540)) if kind == "full" else im.crop((480, 270, 1440, 810))
                        ImageDraw.Draw(im).text((10, 30), f"{name} view{k} {tag} ({kind})", fill=(255, 255, 0))
                        sheet.paste(im, (j * 960, k * 540))
                sheet.save(OUT / f"sheet_{name}_{kind}.jpg", quality=88)
            page.evaluate("() => window.__roomcall.send({t: 'reset'})")
            time.sleep(4)
            s5 = state()
            rec["reset_ok"] = not diff(s0, s5, None)
            results[name] = rec
            print(name, json.dumps({k: v for k, v in rec.items() if k not in ("views",)}), flush=True)
        page.evaluate("([a, b]) => window.__roomcall.look(a, b)", [[-0.9, -0.45, 0.05], [0.2, 1.2, -0.9]])
        time.sleep(2)
        page.screenshot(path=str(OUT / "room_wide_end.png"))
        fps["default_view_end"] = round(page.evaluate(FPS_JS, 4000), 1)
        browser.close()
    errs = [m for m in logs if m["type"] in ("error", "pageerror")]
    warns = [m for m in logs if m["type"] == "warning"]
    out = {"url": URL, "info": info, "fps": fps, "results": results, "console_errors": errs,
           "http_errors": bad, "console_warnings": warns, "console_all": logs[-200:]}
    (OUT / "movecheck.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"fps": fps, "errors": errs, "http_errors": bad, "warnings": warns[:20]}, indent=1))


if __name__ == "__main__":
    main()
