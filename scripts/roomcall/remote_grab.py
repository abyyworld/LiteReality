"""Join a room call as a scripted participant and carry one object somewhere through the shared
physics: what everyone in the room then sees move. For demos and end-to-end checks.

    uv run python scripts/roomcall/remote_grab.py URL ROOM BODY_NAME --to X Y [--seconds 3]

URL is the room call's address (the deployed link or http://127.0.0.1:8790), ROOM its ?room= id,
BODY_NAME a movable body's name (scene.json lists them), X Y the room-frame floor position to
carry it to (metres, MuJoCo frame). Prints where the body was and where it came to rest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import urllib.request


async def run(url: str, room: str, name: str, to_xy: tuple[float, float], seconds: float) -> int:
    import websockets

    manifest = json.load(urllib.request.urlopen(url.rstrip("/") + "/scene.json", timeout=30))
    ids = {b.get("name"): b["id"] for b in manifest["bodies"]}
    if name not in ids:
        raise SystemExit(f"{name}: not a movable body here ({', '.join(sorted(map(str, ids)))})")
    body = ids[name]
    ws_url = url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    async with websockets.connect(f"{ws_url}/ws?room={room}", max_size=2**24) as ws:
        await ws.send(json.dumps({"t": "hello", "name": "script", "color": "#888888"}))
        version, pose = None, None
        deadline = time.monotonic() + 60
        while pose is None and time.monotonic() < deadline:
            msg = json.loads(await ws.recv())
            if msg.get("t") == "welcome":
                version = msg.get("v")
            if msg.get("t") == "state":
                version = msg.get("v", version)
                for row in msg.get("bodies") or []:
                    if row[0] == body:
                        pose = row[1:4]
        if pose is None:
            raise SystemExit("no state frame with that body")
        print(f"{name} (body {body}) at {[round(v, 3) for v in pose]} on version {version}")
        start = list(pose)
        await ws.send(json.dumps({"t": "grab", "v": version, "body": body, "point": start}))
        lift = start[2] + 0.15
        steps = max(1, int(seconds * 20))
        for i in range(1, steps + 1):
            a = i / steps
            target = [start[0] + (to_xy[0] - start[0]) * a, start[1] + (to_xy[1] - start[1]) * a,
                      lift if a < 1 else start[2]]
            await ws.send(json.dumps({"t": "move", "target": target}))
            await asyncio.sleep(seconds / steps)
        await ws.send(json.dumps({"t": "release"}))
        rest, until = None, time.monotonic() + 3
        while time.monotonic() < until:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
            if msg.get("t") == "state":
                for row in msg.get("bodies") or []:
                    if row[0] == body:
                        rest = row[1:4]
        moved = math.dist(start[:2], rest[:2]) if rest else float("nan")
        print(f"released; now at {[round(v, 3) for v in rest] if rest else '?'} "
              f"({moved:.2f} m from where it was)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url")
    parser.add_argument("room")
    parser.add_argument("body")
    parser.add_argument("--to", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    return asyncio.run(run(args.url, args.room, args.body, tuple(args.to), args.seconds))


if __name__ == "__main__":
    raise SystemExit(main())
