"""Door check in the real room call: the door shut and pulled open at its handle, from the
capture path, a grazing view along the wall beside it, and low.

  uv run --frozen --with playwright python results/door_fix/doorcheck.py URL OUT BUNDLE
"""
import math
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path.home() / "LiteReality/results/spark_cap"))
from join import join  # noqa: E402

URL, OUT, BUNDLE = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
ARGS = ["--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"]
APP = Path.home() / "LiteReality/src/litereality_agent/room_ops/roomcall/web/app.js"
FLOOR = -1.2662
STATE_JS = """() => [...window.__roomcall.sh().bodies.values()].map((b) => [b.name,
  b.node.position.toArray(), b.node.quaternion.toArray()])"""


def qangle(a, b):
    d = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def views(p, q):
    ang = 2 * math.atan2(q[2], q[3])
    ux, uy = math.cos(ang), math.sin(ang)
    nx, ny = -uy, ux
    mid = [p[0] + 0.45 * ux, p[1] + 0.45 * uy]
    if (-0.14 - mid[0]) * nx + (0.64 - mid[1]) * ny < 0:
        nx, ny = -nx, -ny                       # n points into the room
    aim = [mid[0], mid[1], p[2] + 1.0]
    return {
        "cap": [[-0.2, 0.0, 0.1], aim],
        "oblique": [[0.3, 1.0, 0.1], aim],
        # standing by the wall the door is in, looking along it (the user's screenshots)
        "grazing": [[mid[0] + 0.5 * nx - 1.3 * ux, mid[1] + 0.5 * ny - 1.3 * uy, 0.15], aim],
        "grazing2": [[mid[0] + 0.6 * nx + 1.5 * ux, mid[1] + 0.6 * ny + 1.5 * uy, 0.15], aim],
        "low": [[mid[0] + 1.4 * nx + 0.5 * ux, mid[1] + 1.4 * ny + 0.5 * uy, FLOOR + 0.35],
                [mid[0], mid[1], FLOOR + 0.5]],
    }, (ux, uy)


def main():
    src = APP.read_text()
    body = src.replace("const sr = new mod.SparkRenderer(", "const sr = window.__sr = new mod.SparkRenderer(")
    body = body.replace("window.__roomcall = {",
                        "window.__roomcall = { send: (m) => send(m), ver: () => shown?.v, sh: () => shown, ")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=ARGS, headless=True)
        page = browser.new_context(viewport={"width": 1600, "height": 900}).new_page()
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
        join(page)
        page.add_style_tag(content="#help{display:none!important}")
        time.sleep(5)

        def look(v, wait=2.5):
            page.evaluate("([a, b]) => window.__roomcall.look(a, b)", v)
            time.sleep(wait)

        def send(m):
            page.evaluate("(m) => window.__roomcall.send(m)", m)

        def hold(target, secs):
            for _ in range(int(secs * 10)):
                send({"t": "move", "target": target})
                time.sleep(0.1)

        state = lambda: {n: (p, q) for n, p, q in page.evaluate(STATE_JS)}  # noqa: E731
        door = next(b for b in page.evaluate("() => window.__roomcall.bodies()") if "Door" in b["name"])
        s0 = state()
        vs, (ux, uy) = views(door["p"], s0[door["name"]][1])
        for k, v in vs.items():
            look(v)
            page.screenshot(path=str(OUT / f"shut_{k}.png"))
        handle = [door["p"][0] + 0.85 * ux, door["p"][1] + 0.85 * uy, door["p"][2] + 1.0]
        best = None
        for sgn in (1, -1):
            send({"t": "grab", "v": page.evaluate("() => window.__roomcall.ver()"), "body": door["id"],
                  "point": handle})
            tgt = [handle[0] - uy * 0.7 * sgn, handle[1] + ux * 0.7 * sgn, handle[2]]
            for i in range(1, 11):
                hold([handle[j] + (tgt[j] - handle[j]) * i / 10 for j in range(3)], 0.1)
            hold(tgt, 2.0)
            turned = qangle(s0[door["name"]][1], state()[door["name"]][1])
            if best is None or turned > best[0]:
                best = (turned, tgt)
            if turned > 25:
                break
            send({"t": "release"})
            time.sleep(1.5)
        print("door turned", round(best[0], 1))
        for k, v in vs.items():
            hold(best[1], 0.3)
            look(v, 2.0)
            hold(best[1], 0.3)
            page.screenshot(path=str(OUT / f"open_{k}.png"))
        browser.close()


main()
