"""Send a scan into a room the way the phone's "Add room" does, as a scripted participant, and
follow its build to the end: the quick room, the photoreal splat, the realistic objects.

    uv run python scripts/roomcall/upload_scan.py URL ROOM SCAN_DIR [--timeout 900]

URL is the room call (the deployed link or http://127.0.0.1:8790), ROOM its ?room= id, SCAN_DIR a
LiteReality Scanner folder (frames, depth, roomplan/room.usdz). Prints each build message with
the time since the upload started, and the splat bundles the room gets.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path


def zip_scan(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                zf.write(path, Path(folder.name) / path.relative_to(folder))
    return buf.getvalue()


async def run(url: str, room: str, scan: Path, timeout: float) -> int:
    import websockets

    base = url.rstrip("/")
    ws_url = base.replace("https://", "wss://").replace("http://", "ws://")
    body = zip_scan(scan)
    print(f"scan {scan.name}: {len(body) / 1e6:.1f} MB zipped")
    async with websockets.connect(f"{ws_url}/ws?room={room}", max_size=2**24) as ws:
        await ws.send(json.dumps({"t": "hello", "name": "uploader", "color": "#888888"}))
        welcome = None
        while welcome is None:
            msg = json.loads(await ws.recv())
            if msg.get("t") == "welcome":
                welcome = msg
        req = urllib.request.Request(f"{base}/scan", data=body, method="POST", headers={
            "content-type": "application/zip", "x-roomcall-room": room,
            "x-roomcall-peer": str(welcome["id"]), "x-roomcall-key": str(welcome.get("key"))})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as r:
            job = json.load(r)["job"]
        print(f"  accepted as job {job}")
        seen_splat, version = None, None
        deadline = t0 + timeout
        while time.monotonic() < deadline:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 5))
            except asyncio.TimeoutError:
                msg = {}
            at = time.monotonic() - t0
            if msg.get("t") == "scan" and msg.get("job") == job:
                print(f"  {at:6.1f} s  {msg.get('status')}: {msg.get('detail')}")
                if msg.get("status") in ("done", "failed"):
                    break
            if msg.get("t") == "scene":
                version = msg.get("v")
                print(f"  {at:6.1f} s  room version {version} ({msg.get('stage')})")
            if msg.get("t") == "splat":
                print(f"  {at:6.1f} s  photoreal room updated")
            if version:
                try:
                    with urllib.request.urlopen(f"{base}/splat.json?v={version}", timeout=10) as r:
                        bundle = json.load(r)
                    key = (bundle.get("stage"), bundle.get("step"))
                    if key != seen_splat:
                        seen_splat = key
                        print(f"  {at:6.1f} s  splat bundle: {bundle.get('stage')} "
                              f"step {bundle.get('step')}, {len(bundle.get('bodies', {}))} objects")
                except Exception:  # noqa: BLE001 — none yet
                    pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url")
    parser.add_argument("room")
    parser.add_argument("scan", type=Path)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    return asyncio.run(run(args.url, args.room, args.scan.expanduser(), args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
