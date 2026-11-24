"""Ghost-fix-2 grab test in the real room call (adapted from the round-1 skeptic's skeptic.py).

  uv run --frozen --with playwright python results/ghost_fix2/callcheck.py URL OUT BUNDLE grab VIEWS_OUT.json
  uv run --frozen --with playwright python results/ghost_fix2/callcheck.py URL OUT BUNDLE rest VIEWS_IN.json

grab: per body, 3 viewpoints aimed at its rest spot (from the capture path, the far side, low).
Shots: before, roomonly (every body splat faded out, nothing moved). Then every chair is grabbed
at 3 points a user would click (heldcheck.points: centre of mass, top of the back, a front seat
corner), lifted 0.9 m (pulled out 0.6 m backwards first when something overhead stops a slow
lift) and carried 0.8 m: shots carried (and, for the centre-of-mass grab, carried_nohold: only the
held body faded) from the 3 views; tilt sampled all along. The door is pulled open at its handle.
rest: renders the given views with nothing grabbed.
"""
import json
import math
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path.home() / "LiteReality/results/spark_cap"))
from join import join  # noqa: E402

URL, OUT, BUNDLE, MODE, VIEWS = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
OUT.mkdir(parents=True, exist_ok=True)
ARGS = ["--use-gl=angle", "--use-angle=metal", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"]
APP = Path.home() / "LiteReality/src/litereality_agent/room_ops/roomcall/web/app.js"
SJ = json.loads((BUNDLE / "splat.json").read_text())
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco  # noqa: E402
from heldcheck import points  # noqa: E402

_M = mujoco.MjModel.from_xml_path(str(BUNDLE / "physics" / "scene_splat.xml"))
_D = mujoco.MjData(_M)
mujoco.mj_forward(_M, _D)
POINTS, BACK = {}, {}
for _b in range(1, _M.nbody):
    if _M.body_jntnum[_b] and _M.jnt_type[_M.body_jntadr[_b]] == mujoco.mjtJoint.mjJNT_FREE:
        POINTS[_M.body(_b).name] = {k: [float(x) for x in v] for k, v in points(_M, _D, _b).items()}
        _y = _D.xmat[_b].reshape(3, 3)[:2, 1]
        BACK[_M.body(_b).name] = [float(_y[0] / math.hypot(*_y)), float(_y[1] / math.hypot(*_y))]
CAPTURE = SJ["capture"]
FLOOR = -1.2662
ROOM = (-2.55 + 0.3, 2.27 - 0.3, -1.68 + 0.3, 2.95 - 0.3)   # floor box, 0.3 m in
STATE_JS = """() => [...window.__roomcall.sh().bodies.values()].map((b) => [b.name,
  b.node.position.toArray(), b.node.quaternion.toArray()])"""
INV_JS = """() => { const out = []; const sc = window.__scene;
  sc.traverse((o) => { let v = true; for (let p = o; p; p = p.parent) v = v && p.visible;
    const splat = /splat/i.test(o.constructor.name) || o.numSplats !== undefined;
    if (o.isMesh || splat) out.push({ctor: o.constructor.name, name: o.name, visible: v, splat,
      opacity: o.opacity, n: o.numSplats ?? o.packedSplats?.numSplats ?? null,
      url: o.url ?? o.userData?.url ?? null, bodyId: o.userData?.bodyId ?? null,
      parent: o.parent?.name || o.parent?.type}); });
  return out; }"""


def serve(data: bytes):
    return lambda route: route.fulfill(status=200, content_type="application/octet-stream",
                                       body=data)


def tilt(q):
    """Angle of a node's up axis from vertical (three quaternion x, y, z, w; heading ignored)."""
    x, y = q[0], q[1]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def qangle(a, b):
    d = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2 * math.acos(min(1.0, d)))


POLY = [(0.5232, 2.6491), (1.9705, 0.8054), (-0.8075, -1.3755), (-2.2549, 0.4681)]   # floor_photo_0


def inside(x, y, m=0.3):
    """In the floor quad and at least m from every wall."""
    sgn = None
    for (ax, ay), (bx, by) in zip(POLY, POLY[1:] + POLY[:1]):
        ex, ey = bx - ax, by - ay
        cr = (ex * (y - ay) - ey * (x - ax)) / (ex * ex + ey * ey) ** 0.5
        sgn = sgn if sgn is not None else (1 if cr > 0 else -1)
        if cr * sgn < m:
            return False
    return True


def at(p, bearing, dist, z):
    return [p[0] + dist * math.cos(bearing), p[1] + dist * math.sin(bearing), z]


def chair_views(p):
    aim = [p[0], p[1], p[2] - 0.1]
    caps = [c for c in CAPTURE if 0.9 <= math.hypot(c[0] - p[0], c[1] - p[1]) <= 2.4]
    cap = min(caps, key=lambda c: abs(math.hypot(c[0] - p[0], c[1] - p[1]) - 1.6))
    b0 = math.atan2(cap[1] - p[1], cap[0] - p[0])
    views = {"cap": [list(cap[:3]), aim]}
    # far side: opposite bearing, as far as the room allows (>= 0.7 m), else nearest bearing that fits
    best = None
    for dd in range(0, 100, 10):
        for s in (1, -1):
            b = b0 + math.pi + s * math.radians(dd)
            for dist in (1.5, 1.2, 1.0, 0.8):
                q = at(p, b, dist, 0.05)
                if inside(q[0], q[1]):
                    best = best or (q, dd)
                    break
        if best:
            break
    views["far"] = [best[0], aim]
    lowb = None
    for dd in (90, -90, 60, -60, 120, -120, 30, -30, 0):
        q = at(p, b0 + math.radians(dd), 1.3, FLOOR + 0.35)
        if inside(q[0], q[1]):
            lowb = q
            break
    views["low"] = [lowb, [p[0], p[1], FLOOR + 0.3]]
    views["high"] = [at(p, b0, 0.9, 0.55), [p[0], p[1], FLOOR + 0.2]]
    return views, best[1]


def door_views(p, q):
    ang = 2 * math.atan2(q[2], q[3])
    ux, uy = math.cos(ang), math.sin(ang)
    nx, ny = -uy, ux
    mid = [p[0] + 0.45 * ux, p[1] + 0.45 * uy]
    room_c = (-0.14, 0.64)
    if (room_c[0] - mid[0]) * nx + (room_c[1] - mid[1]) * ny < 0:
        nx, ny = -nx, -ny                       # n points into the room
    aim = [mid[0], mid[1], p[2] + 1.0]
    v = {"cap": [[-0.2, 0.0, 0.1], aim], "oblique": [[0.3, 1.0, 0.1], aim],
         "low": [[mid[0] + 1.4 * nx + 0.5 * ux, mid[1] + 1.4 * ny + 0.5 * uy, FLOOR + 0.35], [mid[0], mid[1], FLOOR + 0.5]],
         "far": [[mid[0] - 1.3 * nx, mid[1] - 1.3 * ny, 0.05], aim]}
    return v, (ux, uy)


def main():
    src = APP.read_text()
    for needle in ("const sr = new mod.SparkRenderer(", "window.__roomcall = {", "const scene = new THREE.Scene();"):
        assert src.count(needle) == 1, needle
    body = src.replace("const sr = new mod.SparkRenderer(", "const sr = window.__sr = new mod.SparkRenderer(")
    body = body.replace("const scene = new THREE.Scene();", "const scene = new THREE.Scene(); window.__scene = scene;")
    body = body.replace("window.__roomcall = {",
                        "window.__roomcall = { send: (m) => send(m), ver: () => shown?.v, sh: () => shown, "
                        "phot: () => photo, ")
    logs, reqs = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=ARGS, headless=True)
        page = browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1).new_page()
        page.on("console", lambda m: logs.append({"type": m.type, "text": m.text[:300]}))
        page.on("pageerror", lambda e: logs.append({"type": "pageerror", "text": str(e)[:300]}))
        page.on("response", lambda r: reqs.append([r.status, r.url]))
        if MODE == "tint" and sys.argv[6] != "none":
            tinted = Path(sys.argv[6]).read_bytes()
            page.route("**/room-*.spz", lambda r: r.fulfill(status=200, content_type="application/octet-stream", body=tinted))
            for extra in sys.argv[7:]:          # BODY=FILE: a body's splat swapped too (tinted)
                bname, bfile = extra.split("=", 1)
                page.route(f"**/body-{bname}-*.spz", serve(Path(bfile).read_bytes()))
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
        photo = page.evaluate("() => window.__roomcall.photo()")
        inv = page.evaluate(INV_JS)

        def look(v, wait=2.5):
            page.evaluate("([a, b]) => window.__roomcall.look(a, b)", v)
            time.sleep(wait)

        def shot(name):
            page.screenshot(path=str(OUT / f"{name}.png"))

        def fade(names, op):
            page.evaluate("([ns, op]) => { for (const [n, m] of window.__roomcall.phot().bodies) "
                          "if (ns === null || ns.includes(n)) m.opacity = op; }", [names, op])

        def state():
            return {n: (p, q) for n, p, q in page.evaluate(STATE_JS)}

        def diff(a, b, skip, m=0.01, deg=1.0):
            out = {}
            for n in a:
                if n == skip:
                    continue
                dp, dq = math.dist(a[n][0], b[n][0]), qangle(a[n][1], b[n][1])
                if dp > m or dq > deg:
                    out[n] = {"moved_m": round(dp, 3), "turned_deg": round(dq, 1)}
            return out

        def hold(target, secs):
            for _ in range(int(secs * 10)):
                page.evaluate("(m) => window.__roomcall.send(m)", {"t": "move", "target": target})
                time.sleep(0.1)

        def ramp(a, b, secs):
            for i in range(1, 11):
                hold([a[j] + (b[j] - a[j]) * i / 10 for j in range(3)], secs / 10)

        res = {"photo": photo, "inventory": inv}
        if MODE == "tint":
            views = json.loads(VIEWS.read_text())
            for key, v in views.items():
                look(v)
                shot(f"withbodies_{key}")
                fade(None, 0)
                time.sleep(1.5)
                shot(f"tint_{key}")
                fade(None, 1)
                time.sleep(1.0)
        elif MODE == "rest":
            views = json.loads(VIEWS.read_text())
            s0 = state()
            res["rest_state"] = s0
            for key, v in views.items():
                look(v)
                shot(f"rest_{key}")
            # noise floor: re-render the first three after looking elsewhere
            for key in list(views)[:3]:
                look([[0, 0, 0.1], [1, 1, 0]], 1.0)
                look(views[key])
                shot(f"rest_{key}_again")
        else:
            bodies = page.evaluate("() => window.__roomcall.bodies()")
            names = [b["name"] for b in bodies]
            allviews = {}
            for b in bodies:
                name, bid, p = b["name"], b["id"], b["p"]
                door = "Door" in name
                s0 = state()
                v_now = page.evaluate("() => window.__roomcall.ver()")
                if door:
                    views, (ux, uy) = door_views(p, s0[name][1])
                    views = {k: views[k] for k in ("cap", "far", "low")}
                    farrot = None
                else:
                    views, farrot = chair_views(p)
                    views = {k: views[k] for k in ("cap", "far", "low")}
                for k, v in views.items():
                    allviews[f"{name}_{k}"] = v
                rec = {"views": views, "far_rotated_deg": farrot, "start": s0[name], "grabs": {}}
                for k, v in views.items():
                    look(v)
                    shot(f"{name}_{k}_before")
                    fade(None, 0)
                    time.sleep(1.5)
                    shot(f"{name}_{k}_roomonly")
                    fade(None, 1)
                    time.sleep(1.0)

                def grab(point):
                    page.evaluate("(m) => window.__roomcall.send(m)",
                                  {"t": "grab", "v": v_now, "body": bid, "point": point})

                def drop_and_reset():
                    page.evaluate("(m) => window.__roomcall.send(m)", {"t": "release"})
                    time.sleep(1.0)
                    page.evaluate("() => window.__roomcall.send({t: 'reset'})")
                    time.sleep(4)

                if door:
                    handle = [p[0] + 0.85 * ux, p[1] + 0.85 * uy, p[2] + 1.0]
                    best = None
                    for sgn in (1, -1):
                        grab(handle)
                        tgt = [handle[0] - uy * 0.7 * sgn, handle[1] + ux * 0.7 * sgn, handle[2]]
                        ramp(handle, tgt, 1.0)
                        hold(tgt, 2.0)
                        turned = qangle(s0[name][1], state()[name][1])
                        if best is None or turned > best[0]:
                            best = (turned, sgn, tgt)
                        if turned > 25:
                            break
                        page.evaluate("(m) => window.__roomcall.send(m)", {"t": "release"})
                        time.sleep(1.5)
                    g = {"door_turned_deg": round(best[0], 1)}
                    for k, v in views.items():
                        hold(best[2], 0.3)
                        look(v, 2.0)
                        hold(best[2], 0.3)
                        shot(f"{name}_{k}_open")
                    g["others_while_held"] = diff(s0, state(), name)
                    rec["grabs"]["handle"] = g
                    drop_and_reset()
                else:
                    pts = POINTS[name]
                    for key, gp in pts.items():
                        g = {"point": [round(x, 3) for x in gp], "max_tilt_deg": 0.0}

                        def sample(g=g):
                            st = state()
                            g["max_tilt_deg"] = max(g["max_tilt_deg"], round(tilt(st[name][1]), 1))
                            return st

                        grab(gp)
                        blocked = False
                        for i in range(1, 13):
                            hold([gp[0], gp[1], gp[2] + 0.03 * i], 0.1)
                            st = sample()
                            if 0.03 * i - (st[name][0][2] - s0[name][0][2]) > 0.03:
                                blocked = True
                                break
                        start = [gp[0], gp[1], gp[2]]
                        if blocked:     # under the desk: down, out backwards 0.6 m, then up
                            drop_and_reset()
                            g["max_tilt_deg"] = 0.0
                            back = BACK[name]
                            grab(gp)
                            start = [gp[0] + 0.6 * back[0], gp[1] + 0.6 * back[1], gp[2] + 0.02]
                            ramp(gp, start, 1.5)
                            hold(start, 1.0)
                            sample()
                            g["pulled_out_m"] = round(math.dist(s0[name][0][:2], state()[name][0][:2]), 3)
                        up = [start[0], start[1], gp[2] + 0.9]
                        for i in range(1, 11):
                            hold([start[0], start[1], start[2] + (up[2] - start[2]) * i / 10], 0.12)
                            sample()
                        hold(up, 1.0)
                        s1 = sample()
                        g["rise_m"] = round(s1[name][0][2] - s0[name][0][2], 3)
                        g["others_on_lift"] = diff(s0, s1, name, 0.0, 0.0)
                        others = [o["p"] for o in bodies if o["id"] != bid and "Door" not in o["name"]]
                        best = None
                        for kk in range(24):
                            a = 2 * math.pi * kk / 24
                            qx, qy = up[0] + 0.8 * math.cos(a), up[1] + 0.8 * math.sin(a)
                            if not inside(qx, qy):
                                continue
                            clear = min([math.hypot(qx - o[0], qy - o[1]) for o in others]
                                        + [math.hypot(qx - p[0], qy - p[1])])
                            if best is None or clear > best[0]:
                                best = (clear, [qx, qy, up[2]])
                        carry = best[1]
                        for i in range(1, 11):
                            hold([up[j] + (carry[j] - up[j]) * i / 10 for j in range(3)], 0.15)
                            sample()
                        hold(carry, 1.0)
                        s2 = sample()
                        g["carried_plan_m"] = round(math.hypot(s2[name][0][0] - p[0], s2[name][0][1] - p[1]), 3)
                        g["height_m"] = round(s2[name][0][2] - s0[name][0][2], 3)
                        g["tilt_held_deg"] = round(tilt(s2[name][1]), 1)
                        for k, v in views.items():
                            hold(carry, 0.3)
                            look(v, 2.0)
                            hold(carry, 0.3)
                            sample()
                            shot(f"{name}_{key}_{k}_carried")
                            if key == "com":
                                fade([name], 0)
                                time.sleep(1.5)
                                shot(f"{name}_{key}_{k}_carried_nohold")
                                fade([name], 1)
                        g["others_while_held"] = diff(s0, state(), name, 0.0, 0.0)
                        page.evaluate("(m) => window.__roomcall.send(m)", {"t": "release"})
                        time.sleep(3)
                        g["others_after_release"] = diff(s0, state(), name, 0.0, 0.0)
                        rec["grabs"][key] = g
                        print(name, key, json.dumps(g), flush=True)
                        page.evaluate("() => window.__roomcall.send({t: 'reset'})")
                        time.sleep(4)
                rec["reset_ok"] = not diff(s0, state(), None)
                res[name] = rec
            VIEWS.write_text(json.dumps(allviews, indent=1))
            res["names"] = names
        browser.close()
    res["spz_requests"] = [r for r in reqs if ".spz" in r[1] or "raw" in r[1]]
    res["http_errors"] = [r for r in reqs if r[0] >= 400]
    res["console_errors"] = [m for m in logs if m["type"] in ("error", "pageerror")]
    (OUT / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("spz_requests", "http_errors", "console_errors")}, indent=1))


if __name__ == "__main__":
    main()
