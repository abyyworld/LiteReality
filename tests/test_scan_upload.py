"""Scans sent into a call: POST /scan, the archive checks, the builds in child processes, and the
room that gets them — that room only.

Offline and fast: the builds run a stand-in CLI (`FAKE_CLI`, a few lines that write tiny scenes the
way `quickroom` and `realroom --interval` write theirs), so the server's side — limits, messages,
per-room versions, swaps, timeouts, kills, clean-up — is tested exactly, in a second or two each.
One test runs the real `quickroom` child on a synthetic scan; the real tea_room scan, through the
real `quickroom` and `realroom --fake`, is opt-in (`-m scan`, with `LR_TEST_TEA_ROOM` naming the
scan folder)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import io
import json
import os
import stat
import struct
import subprocess
import sys
import textwrap
import time
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import anyio
import mujoco
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_quickroom import _frame_looking_down, write_roomplan_usdz  # noqa: E402

from litereality_agent import cli  # noqa: E402
from litereality_agent.room_ops import roomcall  # noqa: E402
from litereality_agent.room_ops.roomcall import avatars, hosted, scans, server  # noqa: E402

DEMO = """
<mujoco model="lobby">
  <option timestep="0.004"/>
  <worldbody>
    <geom name="floor" type="plane" size="4 4 0.1"/>
    <body name="ball" pos="0 0 0.2"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
  </worldbody>
</mujoco>
"""
INDEX = """<!doctype html><html><head><meta charset="utf-8"><title>t</title>
<script type="importmap">{ "imports": {} }</script><script type="module" src="app.js"></script>
</head><body></body></html>
"""

# The stand-in CLI the builds run: `quickroom SCAN --out DIR --label=L` writes a scene with Chair0
# (free) and Table0 (static); `realroom SCAN --out DIR ...` writes the base there first (no
# objects.json, as the real one), then SCAN_FAKE_VERSIONS versions, each with objects.json and
# the lines `realroom` prints; `splatroom SCAN --out DIR ...` writes a one-file bundle there
# (and with SCAN_FAKE_SPLAT=hang then waits). SCAN_FAKE_QUICK / SCAN_FAKE_REAL: ok | fail | hang
# (and for the quick build, hog: it holds 200 MB and waits). Each run records its argv and pid
# under SCAN_FAKE_LOG.
FAKE_CLI = textwrap.dedent('''
    import json, os, sys, time
    from pathlib import Path

    cmd, scan = sys.argv[1], Path(sys.argv[2])
    args = sys.argv[3:]
    out = Path(args[args.index("--out") + 1])
    label = next(a.split("=", 1)[1] for a in args if a.startswith("--label="))
    log = Path(os.environ["SCAN_FAKE_LOG"])
    log.mkdir(parents=True, exist_ok=True)
    (log / f"{cmd}.{os.getpid()}.json").write_text(json.dumps({
        "argv": sys.argv[1:], "pid": os.getpid(), "cwd": os.getcwd(),
        "files": sorted(str(p.relative_to(scan)) for p in scan.rglob("*") if p.is_file())}))

    def scene(rgba):
        return f"""<mujoco model="{label}">
      <worldbody>
        <geom name="floor" type="plane" size="3 3 0.1"/>
        <body name="Table0" pos="1 0 0.4"><geom type="box" size="0.5 0.3 0.4" rgba="{rgba}"/></body>
        <body name="Chair0" pos="-1 0 0.3"><freejoint/>
          <geom type="box" size="0.2 0.2 0.3" mass="4" rgba="{rgba}"/></body>
      </worldbody>
    </mujoco>
    """

    def write(text):
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / "scene.xml.tmp"
        tmp.write_text(text)
        os.replace(tmp, out / "scene.xml")

    if cmd == "splatroom":
        out.mkdir(parents=True, exist_ok=True)
        spz = "room-0123456789abcdef.spz"
        (out / spz).write_bytes(b"splats of " + label.encode())
        (out / "splat.json").write_text(json.dumps({"room": spz, "bodies": {}, "stage": "instant",
                                                    "step": None, "created": time.time()}))
        if os.environ.get("SCAN_FAKE_SPLAT") == "hang":
            time.sleep(600)
        sys.exit(0)
    mode = os.environ.get("SCAN_FAKE_QUICK" if cmd == "quickroom" else "SCAN_FAKE_REAL", "ok")
    time.sleep(float(os.environ.get("SCAN_FAKE_SLEEP", "0")))
    if cmd == "quickroom":
        if mode == "hang":
            time.sleep(600)
        if mode == "hog":
            held = b"\\x01" * (200 * 1024 * 1024)
            time.sleep(600)
        if mode == "fail":
            print(f"{scan}/room.usdz is not a readable RoomPlan export: KeyError", file=sys.stderr)
            sys.exit(2)
        write(scene("0.5 0.5 0.5 1"))
        print("quick room  done")
        sys.exit(0)
    write(scene("0.5 0.5 0.5 1"))                   # the base, like RealRoom's
    print(f"realroom  {scan.name}: 2 objects, pass both - stand-in boxes", flush=True)
    print("  generating 2 object(s) with fake: Chair0 (preview+final), Table0 (preview+final)",
          flush=True)
    if mode == "hang":
        time.sleep(600)
    versions = int(os.environ.get("SCAN_FAKE_VERSIONS", "2"))
    for i in range(1, versions + 1):
        time.sleep(float(os.environ.get("SCAN_FAKE_STEP", "0.3")))
        p = "preview" if i == 1 else "final"
        print(f"  Chair0 {p} in 0.1 s", flush=True)
        print(f"  Table0 {p} in 0.1 s", flush=True)
        write(scene(f"0.{i} 0.3 0.3 1"))
        (out / "objects.json").write_text(json.dumps({
            "objects": {"Chair0": {"pass": p}, "Table0": {"pass": p}}, "replaced": {},
            "versions_written": i}))
        print(f"  version {i}: 2 generated object(s) (objects arrived)", flush=True)
    if mode == "fail":
        print("RuntimeError: TRELLIS went away", file=sys.stderr)
        sys.exit(1)
''')


# -- archives ------------------------------------------------------------------------------------

def jpeg(color=(120, 90, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, "JPEG")
    return buf.getvalue()


def usdz_bytes(tmp_path: Path) -> bytes:
    path = write_roomplan_usdz(tmp_path / "usdz-src" / "room.usdz")
    return path.read_bytes()


def make_zip(entries: dict[str, bytes | str], *, compression=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def scan_entries(tmp_path: Path, top: str = "My Room/", *, frames: int = 2) -> dict:
    entries: dict = {f"{top}room.usdz": usdz_bytes(tmp_path)}
    for k in range(frames):
        entries[f"{top}frame_{k:05d}.jpg"] = jpeg()
        entries[f"{top}frame_{k:05d}.json"] = json.dumps({"frame_index": k})
    return entries


def plan_of(tmp_path: Path, data: bytes, kind: str = "zip", limits=scans.Limits(), name="up.bin"):
    path = tmp_path / name
    path.write_bytes(data)
    return scans.inspect_upload(path, kind, limits, usdz_out=tmp_path / f"{name}.usdz")


def test_a_scan_zip_is_planned_and_unpacked_with_only_the_scans_files(tmp_path):
    entries = scan_entries(tmp_path, frames=3)
    entries.update({
        "My Room/": b"",
        "My Room/depth_00000.png": png_header(256, 192),
        "My Room/pointcloud.pcd": "# .PCD v0.7\n",
        "My Room/scan_label.json": json.dumps({"displayName": "  Tea\nroom  ", "scannerName": "x"}),
        "My Room/video.mov": b"\0" * 64,                          # not a scan's kind of file
        "My Room/scene.json": json.dumps({"capture": "/etc"}),    # never: the CLI would follow it
        "My Room/.DS_Store": b"\0" * 64,
        "__MACOSX/My Room/._room.usdz": b"\0" * 64,
        "beside/notes.txt": "not in the scan's folder",
    })
    plan = plan_of(tmp_path, make_zip(entries))
    assert plan.kind == "zip" and plan.label == "My Room" and plan.frames == 3
    assert str(plan.usdz) == "room.usdz"
    planned = {str(rel) for _name, rel in plan.members}
    assert "video.mov" not in planned and "scene.json" not in planned
    assert not any("MACOSX" in p or "DS_Store" in p or "notes" in p for p in planned)
    scan = scans.unpack_scan(plan, tmp_path / "scan", usdz=tmp_path / "up.bin.usdz")
    files = {str(p.relative_to(scan.root)) for p in scan.root.rglob("*") if p.is_file()}
    assert files == {"room.usdz", "frame_00000.jpg", "frame_00001.jpg", "frame_00002.jpg",
                     "frame_00000.json", "frame_00001.json", "frame_00002.json",
                     "depth_00000.png", "pointcloud.pcd", "scan_label.json"}
    assert scan.label == "Tea room" and scan.frames == 3          # scan_label.json's own name
    assert zipfile.is_zipfile(scan.root / "room.usdz")


@pytest.mark.parametrize("name", ["../evil.json", "/abs/evil.json", "My Room/../../evil.json",
                                  "C:/evil.json", "..\\evil.json", "My Room\\..\\..\\evil.json"])
def test_zip_slip_is_refused(tmp_path, name):
    entries = scan_entries(tmp_path)
    entries[name] = "{}"
    with pytest.raises(scans.ScanRequestError) as err:
        plan_of(tmp_path, make_zip(entries))
    assert err.value.status == 400 and "outside" in err.value.message
    assert not (tmp_path.parent / "evil.json").exists()


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFCHR | 0o600, stat.S_IFIFO | 0o600])
def test_links_and_special_files_are_refused(tmp_path, mode):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in scan_entries(tmp_path).items():
            zf.writestr(name, data)
        info = zipfile.ZipInfo("My Room/frame_00009.jpg")
        info.create_system = 3
        info.external_attr = mode << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(scans.ScanRequestError, match="links or special files"):
        plan_of(tmp_path, buf.getvalue())


def test_a_zip_bomb_is_refused_without_unpacking(tmp_path):
    entries = scan_entries(tmp_path)
    entries["My Room/pointcloud.pcd"] = b"\0" * (8 * 1024 * 1024)   # deflates ~1000x
    data = make_zip(entries)
    assert len(data) < 200_000
    with pytest.raises(scans.ScanRequestError) as err:
        plan_of(tmp_path, data)
    assert err.value.status == 413 and "zip bomb" in err.value.message


def test_too_much_unpacked_is_refused(tmp_path):
    entries = scan_entries(tmp_path)
    entries["My Room/pointcloud.pcd"] = os.urandom(300_000)
    limits = scans.Limits(unpacked=200_000)
    with pytest.raises(scans.ScanRequestError) as err:
        plan_of(tmp_path, make_zip(entries), limits=limits)
    assert err.value.status == 413 and "unpacks to more than" in err.value.message


def test_too_many_members_is_refused_before_zipfile_parses_it(tmp_path, monkeypatch):
    entries = scan_entries(tmp_path)
    for k in range(60):
        entries[f"My Room/frame_{k + 10:05d}.json"] = "{}"
    data = make_zip(entries)

    def no_parsing(*args, **kwargs):
        raise AssertionError("the end record alone should have refused it")

    monkeypatch.setattr(scans.zipfile, "ZipFile", no_parsing)
    with pytest.raises(scans.ScanRequestError) as err:
        plan_of(tmp_path, data, limits=scans.Limits(members=50))
    assert err.value.status == 413 and "more than 50 files" in err.value.message


def test_a_member_bigger_than_it_says_is_refused_and_not_left_behind(tmp_path):
    entries = scan_entries(tmp_path)
    payload = b"A" * 4000
    entries["My Room/frame_00007.json"] = payload
    data = bytearray(make_zip(entries, compression=zipfile.ZIP_STORED))
    # Rewrite that member's uncompressed size, in its local header and its directory entry, to
    # say 10 bytes: the bytes stored are still 4000.
    name = b"My Room/frame_00007.json"
    for sig, offset in ((b"PK\x03\x04", 22), (b"PK\x01\x02", 24)):
        at = 0
        while (at := data.find(sig, at)) >= 0:
            name_at = at + (30 if sig == b"PK\x03\x04" else 46)
            if data[name_at:name_at + len(name)] == name:
                data[at + offset:at + offset + 4] = (10).to_bytes(4, "little")
            at += 4
    plan = plan_of(tmp_path, bytes(data))
    with pytest.raises(scans.ScanRequestError) as err:
        scans.unpack_scan(plan, tmp_path / "scan", usdz=tmp_path / "up.bin.usdz")
    assert err.value.status in (400, 413)
    assert not (tmp_path / "scan" / "frame_00007.json").exists()


def test_encrypted_and_duplicate_members_are_refused(tmp_path):
    entries = scan_entries(tmp_path)
    entries["My Room/secret.json"] = "{}"
    data = bytearray(make_zip(entries))
    # zipfile cannot write an encrypted member: set the flag in its directory entry by hand.
    at = data.find(b"PK\x01\x02")
    while at >= 0:
        if data[at + 46:at + 46 + 19] == b"My Room/secret.json":
            data[at + 8] |= 0x1
        at = data.find(b"PK\x01\x02", at + 4)
    with pytest.raises(scans.ScanRequestError, match="encrypted"):
        plan_of(tmp_path, bytes(data))
    entries = scan_entries(tmp_path)
    entries["My Room/FRAME_00000.JPG"] = jpeg()           # the same file on a Mac's disk
    with pytest.raises(scans.ScanRequestError, match="twice"):
        plan_of(tmp_path, make_zip(entries), name="dup.bin")


def test_where_the_scan_is_in_the_zip(tmp_path):
    usdz = usdz_bytes(tmp_path)
    # Files at the root, no folder; the usdz under roomplan/ with another case: room.usdz there.
    plan = plan_of(tmp_path, make_zip({"roomplan/Room.usdz": usdz, "frame_00000.jpg": jpeg()}))
    assert str(plan.usdz) == "roomplan/room.usdz" and plan.label is None and plan.frames == 1
    scan = scans.unpack_scan(plan, tmp_path / "s1", usdz=tmp_path / "up.bin.usdz")
    assert (scan.root / "roomplan" / "room.usdz").is_file() and scan.label == "scanned room"
    # One .usdz by another name is the room; wrapped in two folders, the shallowest scan counts.
    plan = plan_of(tmp_path, make_zip({"a/b/Kitchen.usdz": usdz, "a/b/frame_00000.jpg": jpeg()}),
                   name="other.bin")
    assert str(plan.usdz) == "room.usdz" and plan.label == "b"
    # Two scans side by side: which one?
    with pytest.raises(scans.ScanRequestError, match="more than one scan"):
        plan_of(tmp_path, make_zip({"a/room.usdz": usdz, "b/room.usdz": usdz}), name="two.bin")
    with pytest.raises(scans.ScanRequestError, match="no room.usdz"):
        plan_of(tmp_path, make_zip({"x/frame_00000.jpg": jpeg()}), name="none.bin")
    with pytest.raises(scans.ScanRequestError, match="several .usdz"):
        plan_of(tmp_path, make_zip({"a.usdz": usdz, "b.usdz": usdz}), name="many.bin")


def test_bad_uploads_and_bad_usdz_are_refused(tmp_path):
    usdz = usdz_bytes(tmp_path)
    cases = {
        "not a zip": (b"hello, world" * 10, "zip"),
        "empty": (b"", "zip"),
        "cut short": (make_zip(scan_entries(tmp_path))[:3000], "zip"),
        "usdz not a zip": (b"#usda 1.0\n" * 10, "usdz"),
        "usdz without layers": (make_zip({"readme.txt": "hi"}), "usdz"),
        "usdz layers that are not": (make_zip({"room.usda": b"\0" * (4 * 1024 * 1024)}), "usdz"),
        "zip whose usdz is not": (make_zip({"s/room.usdz": b"nope" * 100}), "zip"),
    }
    for what, (data, kind) in cases.items():
        with pytest.raises(scans.ScanRequestError) as err:
            plan_of(tmp_path, data, kind=kind, name=what.replace(" ", "_"))
        assert err.value.status in (400, 413), what
    plan = plan_of(tmp_path, usdz, kind="usdz", name="bare.usdz")
    assert plan.kind == "usdz" and plan.frames == 0 and not plan.members


def test_clean_label():
    assert scans.clean_label("  tea\x00 \n room ") == "tea room"
    assert scans.clean_label("x" * 100) == "x" * 40
    assert scans.clean_label("\n\t") is None and scans.clean_label(3) is None


# -- what a scan says -----------------------------------------------------------------------------
# A scan is checked as an archive above; these are its contents, which the builds act on: the
# room.usdz's prim names become file names, and its layers, photos and depth maps become memory.

def usdz_layers(tmp_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(usdz_bytes(tmp_path))) as zf:
        return {name: zf.read(name).decode("utf-8") for name in zf.namelist()}


def pack_usdz(layers: dict[str, str | bytes]) -> bytes:
    return make_zip(layers, compression=zipfile.ZIP_STORED)      # as RoomPlan writes it


def png_header(width: int, height: int) -> bytes:
    """A PNG that says it is `width` x `height` (it holds no pixels): all the checks read."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IEND", b""))


def jpeg_header(width: int, height: int, *, apps: int = 0, app_bytes: int = 1000,
                noise: bool = False) -> bytes:
    """A JPEG that says it is `width` x `height`, after `apps` APP1 segments (EXIF-like; of
    random bytes with `noise`, which no zip compresses)."""
    app = b"".join(b"\xff\xe1" + struct.pack(">H", 2 + app_bytes)
                   + (os.urandom(app_bytes) if noise else b"\0" * app_bytes) for _ in range(apps))
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3) + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8" + app + sof + b"\xff\xda\x00\x02" + b"\0" * 16 + b"\xff\xd9"


@pytest.mark.parametrize("name", [
    "../../../../tmp/PWNED_Wall0",       # the review's write outside the job's folder
    "/etc/Wall0", "sub/Wall0", "..", ".hidden", "Wall0.png", "Wall 0", "Wall0\\", "Wall0\x00",
    "", "0Wall", "x" * 65])
def test_a_prim_name_that_is_not_an_identifier_is_refused(tmp_path, name):
    """The quick build writes a wall's texture to `textures/<wall name>.png` (and realroom names an
    object's files after it): a name with a path in it wrote anywhere the child could reach."""
    layers = usdz_layers(tmp_path)
    wall = next(n for n in layers if n.endswith("/Wall0.usda"))
    layers[wall] = layers[wall].replace('def Cube "Wall0"', f'def Cube "{name}"')
    assert f'Cube "{name}"' in layers[wall]
    with pytest.raises(scans.ScanRequestError, match="letters, digits and _") as err:
        plan_of(tmp_path, pack_usdz(layers), kind="usdz", name="names.usdz")
    assert err.value.status == 400


def test_prim_names_are_checked_wherever_the_reader_would_find_one(tmp_path):
    layers = usdz_layers(tmp_path)
    good = dict(layers)
    good["room.usda"] = good["room.usda"].rstrip()[:-1] + (
        '    def Xform "Sink_Storage0" {}\n    over "_x9" {}\n    def "Stuhl_\u00e40" {}\n}\n')
    plan_of(tmp_path, pack_usdz(good), kind="usdz", name="good.usdz")      # identifiers, UTF-8 too
    hidden = {
        "an over": ("room.usda", 'over "../x" {}'),
        "no type": ("room.usda", 'def "a/b" {}'),
        "inside a string": ("room.usda", 'custom string doc = "see def Xform "../x" {}"'),
        "a member that is not .usda": ("assets/notes.txt", 'def Cube "../../x" {}'),
        "an unclosed name": ("assets/tail.usda", 'def Cube "Wall0'),
    }
    for member, text in hidden.values():
        bad = dict(layers)
        bad[member] = bad.get(member, "") + "\n" + text + "\n"
        with pytest.raises(scans.ScanRequestError, match="letters, digits and _"):
            plan_of(tmp_path, pack_usdz(bad), kind="usdz", name="hidden.usdz")


def test_a_scan_whose_room_usdz_names_a_path_is_refused_before_any_build(web_dir, tmp_path,
                                                                        fake_cli, monkeypatch):
    layers = usdz_layers(tmp_path)
    for member in [n for n in layers if "/Walls/" in n]:
        layers[member] = layers[member].replace('def Cube "Wall', 'def Cube "../../../../x/Wall')
    entries = scan_entries(tmp_path)
    entries["My Room/room.usdz"] = pack_usdz(layers)
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        res = post(client, "r", wa, make_zip(entries))
        assert res.status_code == 400 and "letters, digits and _" in res.json()["detail"]
        assert "letters, digits and _" in until(a, scan_msg("failed"))["detail"]
        assert children(fake_cli, "quickroom") == [] and "r" not in hub.room_versions
        wait_for(lambda: not any((tmp_path / "scratch").iterdir()))


def test_images_are_sized_from_their_headers_as_pillow_sizes_them():
    cases = []
    for mode, size in [("L", (256, 192)), ("I;16", (256, 192)), ("RGB", (640, 480)),
                       ("RGBA", (33, 17))]:
        buf = io.BytesIO()
        Image.new(mode, size).save(buf, "PNG")
        cases.append(("png", buf.getvalue()))
    buf = io.BytesIO()
    Image.new("RGB", (1920, 1440), (90, 60, 30)).save(buf, "JPEG")
    cases.append(("jpeg", buf.getvalue()))
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    buf = io.BytesIO()                                # a camera's: EXIF, a big ICC, progressive
    Image.new("RGB", (4032, 3024)).save(buf, "JPEG", exif=exif.tobytes(),
                                        icc_profile=b"\0" * 200_000, progressive=True)
    cases.append(("jpeg", buf.getvalue()))
    cases += [("png", png_header(13000, 13000)), ("jpeg", jpeg_header(5000, 4000, apps=3))]
    for kind, data in cases:
        got = scans._image_size(io.BytesIO(data))
        with pytest.warns(Image.DecompressionBombWarning) if got[1] == 13000 else \
                contextlib.nullcontext(), Image.open(io.BytesIO(data)) as im:
            # Pillow itself only warns about the 13000 x 13000 one: 169 Mpx decodes to ~6 GB.
            assert got[:3] == (kind, *im.size), kind


def test_image_bombs_are_refused_by_their_headers_without_decoding_anything(tmp_path,
                                                                           monkeypatch):
    base = scan_entries(tmp_path)
    buf = io.BytesIO()
    Image.new("I;16", (256, 192)).save(buf, "PNG")
    base["My Room/depth_00000.png"] = buf.getvalue()           # the scanner's own sizes pass
    base["My Room/conf_00000.png"] = png_header(256, 192)
    base["My Room/frame_00001.jpg"] = jpeg_header(1920, 1440, apps=2)
    cases = {
        "the review's depth bomb": ("depth_00000.png", png_header(13000, 13000), 413, "depth maps"),
        "a map too wide": ("conf_00000.png", png_header(2049, 10), 413, "depth maps"),
        "a map of too many pixels": ("depth_00000.png", png_header(1100, 1000), 413, "depth maps"),
        "a photo too big": ("frame_00000.jpg", jpeg_header(5000, 3000), 413, "photos"),
        "any other image is a photo": ("thumb.png", png_header(5000, 5000), 413, "photos"),
        "a PNG named .jpg": ("frame_00000.jpg", png_header(64, 48), 400, "a PNG, not the JPEG"),
        "a JPEG named .png": ("depth_00000.png", jpeg_header(64, 48), 400, "a JPEG, not the PNG"),
        "some other format": ("depth_00000.png", b"GIF89a" + b"\0" * 64, 400, "not an image"),
        "a JPEG of endless segments": ("frame_00000.jpg", jpeg_header(64, 48, apps=600, app_bytes=0),
                                       400, "more segments"),
        "a JPEG's size far in": ("frame_00000.jpg",
                                 jpeg_header(64, 48, apps=20, app_bytes=65000, noise=True), 400,
                                 "header"),
        "a JPEG cut short": ("frame_00000.jpg", jpeg_header(64, 48, apps=1)[:600], 400,
                             "cut short"),
        "a big .json": ("frame_00000.json", os.urandom(1024 * 1024 + 1), 413, "a scan's are"),
    }
    entries = {name: base | {f"My Room/{member}": data}
               for name, (member, data, _status, _says) in cases.items()}
    plan_of(tmp_path, make_zip(base), name="base.zip")
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("an image was decoded"))
    for name, (_member, _data, status, says) in cases.items():
        with pytest.raises(scans.ScanRequestError) as err:
            plan_of(tmp_path, make_zip(entries[name]), name="bomb.zip")
        assert (err.value.status, says in err.value.message) == (status, True), \
            (name, err.value.message)
    many = base | {f"My Room/depth_{k:05d}.png": png_header(256, 192) for k in range(30)}
    with pytest.raises(scans.ScanRequestError, match="pixels in all") as err:
        plan_of(tmp_path, make_zip(many), limits=scans.Limits(maps_pixels=1_000_000),
                name="many.zip")
    assert err.value.status == 413


def test_a_room_usdz_that_takes_too_much_reading_is_refused(tmp_path):
    """The review's amplification: a small usdz whose root names one big layer over and over —
    the reader parses it again each time, and holds what it read."""
    layers = usdz_layers(tmp_path)
    chair = next(n for n in layers if n.endswith("/Chair0.usda"))
    layers["assets/big.usda"] = layers[chair] + "#" + "x" * 200_000 + "\n"

    def fanned(times: int) -> bytes:
        out = dict(layers)
        refs = "".join(f'    def Xform "Fan{i}" (\n        prepend references = '
                       f'@./assets/big.usda@\n    )\n    {{\n    }}\n' for i in range(times))
        out["room.usda"] = out["room.usda"].rstrip()[:-1] + refs + "}\n"
        return pack_usdz(out)

    plan_of(tmp_path, fanned(10), kind="usdz", name="ten.usdz")           # 2 MB read: fine
    with pytest.raises(scans.ScanRequestError, match="names its layers so many times") as err:
        plan_of(tmp_path, fanned(100), kind="usdz", name="hundred.usdz")   # 20 MB read
    assert err.value.status == 413


def test_room_usdz_layers_are_held_to_what_roomplan_writes(tmp_path):
    layers = usdz_layers(tmp_path)
    limits = scans.Limits(usd_prims=200, usd_references=200)
    plan_of(tmp_path, pack_usdz(layers), kind="usdz", limits=limits, name="ok.usdz")

    def root_plus(text: str) -> dict:
        return layers | {"room.usda": layers["room.usda"].rstrip()[:-1] + text + "}\n"}

    cases = {
        "a layer too big": (layers | {"assets/a.usda": "#" + "x" * (2 * 1024 * 1024)}, 413,
                            "bigger than 2 MB"),
        "too much in all": ({**layers, **{f"assets/a{k}.usda": "#" + "x" * 1_900_000
                                          for k in range(9)}}, 413, "bigger than 16 MB"),
        "too many prims": (root_plus("".join(f'def Xform "P{k}" {{}}\n' for k in range(200))), 413,
                           "parts"),
        "too many files named": (root_plus("".join(f"# @a{k}.usda@\n" for k in range(200))), 413,
                                 "names more than 200 files"),
        "nested too deep": (root_plus('def Xform "D" {\n' * 40 + "}\n" * 40), 400, "nests"),
        "a binary layer": (layers | {"assets/x.usdc": b"PXR-USDC" + b"\0" * 64}, 400, "binary"),
    }
    for what, (content, status, says) in cases.items():
        with pytest.raises(scans.ScanRequestError) as err:
            plan_of(tmp_path, pack_usdz(content), kind="usdz", limits=limits, name="bad.usdz")
        assert (err.value.status, says in err.value.message) == (status, True), \
            (what, err.value.message)
    with pytest.raises(scans.ScanRequestError, match="unpacks to more than") as err:
        plan_of(tmp_path, make_zip(layers | {f"assets/a{k}.usda": os.urandom(30_000).hex()
                                            for k in range(4)}),
                kind="usdz", limits=scans.Limits(usdz_unpacked=200_000), name="deflated.usdz")
    assert err.value.status == 413


# -- the server -----------------------------------------------------------------------------------

@pytest.fixture()
def web_dir(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text(INDEX)
    (web / "app.js").write_text("// app\n")
    (web / "app.css").write_text("")
    return web


@pytest.fixture()
def fake_cli(tmp_path, monkeypatch):
    path = tmp_path / "fake_cli.py"
    path.write_text(FAKE_CLI)
    log = tmp_path / "children"
    monkeypatch.setenv("SCAN_FAKE_LOG", str(log))
    for name in ("SCAN_FAKE_QUICK", "SCAN_FAKE_REAL", "SCAN_FAKE_SLEEP", "SCAN_FAKE_VERSIONS",
                 "SCAN_FAKE_STEP", "SCAN_FAKE_SPLAT"):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(command=[sys.executable, str(path)], log=log)


def children(fake_cli, cmd: str) -> list[dict]:
    if not fake_cli.log.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(fake_cli.log.glob(f"{cmd}.*.json"))]


async def _geometry_here(scene, label, **_kwargs):
    """`server.build_assets` without its child process: the stand-in CLI's scenes are tiny, and
    the child (a fresh interpreter importing MuJoCo and trimesh) is most of a test's time."""
    def build():
        return server.build_scene_assets(server.load_model(Path(scene)), label=label)
    return await asyncio.to_thread(build)


@contextlib.contextmanager
def make_client(web_dir, tmp_path, fake_cli=None, monkeypatch=None, **config):
    if monkeypatch is not None:
        monkeypatch.setattr(server, "OFFER_SLACK", 0.05)
        if fake_cli is not None:
            monkeypatch.setattr(server, "build_assets", _geometry_here)
    options = {"mode": "realistic", "fake": True, "scratch": tmp_path / "scratch",
               "watch_seconds": 0.05, "keep_seconds": 0, **config}
    if fake_cli is not None:
        options.setdefault("command", fake_cli.command)
    app = server.create_app(model=mujoco.MjModel.from_xml_string(DEMO), web_dir=web_dir,
                            settle_seconds=0.02, prefetch_seconds=0.1,
                            scans=scans.ScanConfig(**options))
    with TestClient(app) as client:
        yield client, app.state.roomcall


def recv(ws, timeout: float = 10.0) -> dict:
    async def get():
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()
    message = ws.portal.call(get)
    if message["type"] == "websocket.close":
        raise WebSocketDisconnect(message.get("code", 1000))
    return json.loads(message["text"])


def until(ws, predicate, seconds: float = 20.0, seen: list | None = None) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        message = recv(ws, max(0.05, deadline - time.monotonic()))
        if seen is not None and message["t"] != "state":
            seen.append(message)
        if predicate(message):
            return message
    raise AssertionError("no matching frame")


def join(client, stack, room: str, name: str):
    ws = stack.enter_context(client.websocket_connect(f"/ws?{urlencode({'room': room})}"))
    ws.send_json({"t": "hello", "name": name, "color": "#336699"})
    welcome = recv(ws)
    assert welcome["t"] == "welcome"
    return ws, welcome


def who(room: str, welcome: dict) -> dict:
    return {scans.ROOM_HEADER: room, scans.PEER_HEADER: welcome["id"],
            scans.KEY_HEADER: welcome["key"]}


def post(client, room: str, welcome: dict, body: bytes, content_type="application/zip", **headers):
    return client.post("/scan", content=body,
                       headers={"Content-Type": content_type, **who(room, welcome), **headers})


def scan_msg(status: str, job: str | None = None):
    return lambda m: m["t"] == "scan" and m["status"] == status and (job is None or m["job"] == job)


def wait_for(predicate, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out waiting")


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    if not os.path.isdir("/proc"):         # macOS: no /proc, so ask ps (Z = zombie)
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                               text=True).stdout.strip()
        return not state or state.startswith("Z")
    try:                                   # a zombie nobody has reaped yet is gone too
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] == "Z"
    except OSError:
        return True


def test_a_scan_becomes_that_rooms_room_only_quick_then_objects(web_dir, tmp_path, fake_cli,
                                                                monkeypatch):
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch) as (client, hub), \
            contextlib.ExitStack() as stack:
        live = hub.version.id
        a1, wa1 = join(client, stack, "roomA", "Ada")
        a2, _wa2 = join(client, stack, "roomA", "Bo")
        b1, wb1 = join(client, stack, "roomB", "Cy")
        assert wa1["scans"] == {"enabled": True, "mode": "realistic",
                                "max_bytes": scans.MAX_UPLOAD_BYTES, "cooldown": 60}
        assert "scan" not in wa1 and wa1["v"] == wb1["v"] == live
        res = post(client, "roomA", wa1, make_zip(scan_entries(tmp_path)))
        assert res.status_code == 202, res.text
        job = res.json()["job"]
        assert res.headers["cache-control"] == "no-store"
        seen: list = []
        quick_scene = until(a2, lambda m: m["t"] == "scene" and m["stage"] == "quick", seen=seen)
        quick = quick_scene["v"]
        assert quick != live and quick_scene["label"] == "My Room"
        swapped = until(a1, lambda m: m["t"] == "state" and m["v"] == quick)
        manifest = client.get("/scene.json", params={"v": quick}).json()
        chair = next(b["id"] for b in manifest["bodies"] if b["name"] == "Chair0")
        point = next(r for r in swapped["bodies"] if r[0] == chair)[1:4]
        a1.send_json({"t": "grab", "v": quick, "body": chair, "point": point})
        a1.send_json({"t": "move", "target": [point[0], point[1], point[2] + 0.2]})
        until(a1, lambda m: m["t"] == "state" and m["grabs"])
        objects = until(a2, lambda m: m["t"] == "scene" and m["stage"] == "objects", seen=seen)
        until(a2, scan_msg("done", job), seen=seen)
        last = until(a1, lambda m: m["t"] == "state" and m["v"] == hub.room_versions["roomA"].id)
        # Every status in order, each one telling who sent it.
        statuses = [m["status"] for m in seen if m["t"] == "scan"]
        order = ["uploading", "building", "quick", "objects", "done"]
        assert [s for s in order if s in statuses] == order
        assert statuses.index("building") < statuses.index("quick") < statuses.index("objects")
        assert all(m["by"] == wa1["id"] and m["job"] == job for m in seen if m["t"] == "scan")
        assert any(m["t"] == "scan" and "of 2" in m["detail"] for m in seen)
        assert objects["v"] not in (quick, live)
        # The grab came along by name, into each later version.
        manifest = client.get("/scene.json", params={"v": last["v"]}).json()
        chair2 = next(b["id"] for b in manifest["bodies"] if b["name"] == "Chair0")
        assert [g[:2] for g in last["grabs"]] == [[wa1["id"], chair2]]
        # Room B never heard a thing and still runs the live version.
        b_seen: list = []
        frame = until(b1, lambda m: m["t"] == "state", seen=b_seen)
        assert frame["v"] == live and not [m for m in b_seen if m["t"] in ("scan", "scene")]
        assert client.get("/scene.json").json()["v"] == live
        # A late joiner of room A gets the room's version and how it came about.
        _c, wc = join(client, stack, "roomA", "Di")
        assert wc["v"] == last["v"] and wc["scan"]["status"] == "done"
        _d, wd = join(client, stack, "roomC", "Ed")
        assert wd["v"] == live and "scan" not in wd
        status = client.get(f"/scan/{job}").json()
        assert status == {"job": job, "status": "done", "stage": "objects", "progress": 1.0,
                          "detail": status["detail"]} and "2 realistic objects" in status["detail"]
        health = client.get("/healthz").json()
        assert health["scans"]["rooms"] == 1 and health["version"] == live
        # The builds ran as the CLI, niced in their job's folder; which is gone now.
        quick_run, = children(fake_cli, "quickroom")
        real_run, = children(fake_cli, "realroom")
        assert quick_run["argv"][2:] == ["--out", quick_run["argv"][3], "--label=My Room"]
        assert "--pass" in real_run["argv"] and "--fake" in real_run["argv"]
        assert "--no-articulated" in real_run["argv"]
        assert {"room.usdz", "frame_00000.jpg"} <= set(real_run["files"])
        wait_for(lambda: not any((tmp_path / "scratch").iterdir()))


def test_uploads_off_is_503_and_the_welcome_says_so(web_dir, tmp_path):
    with make_client(web_dir, tmp_path, mode="off") as (client, _hub), \
            contextlib.ExitStack() as stack:
        _a, wa = join(client, stack, "r", "Ada")
        assert wa["scans"] == {"enabled": False, "mode": "off"}
        res = post(client, "r", wa, b"PK\x03\x04")
        assert res.status_code == 503 and "off" in res.json()["detail"]


def test_who_is_asking_and_what_comes_are_checked_before_the_body(web_dir, tmp_path, fake_cli):
    with make_client(web_dir, tmp_path, fake_cli, max_upload_bytes=1000) as (client, hub), \
            contextlib.ExitStack() as stack:
        _a, wa = join(client, stack, "r", "Ada")
        _b, wb = join(client, stack, "q", "Bo")
        cases = [
            ({}, 400),                                                  # no headers
            ({**who("r", wa), scans.KEY_HEADER: "0" * 32}, 403),        # someone else's key
            ({**who("r", wb), scans.KEY_HEADER: wb["key"]}, 403),       # not in that room
            (who("../r", wa), 403),
        ]
        for headers, status in cases:
            res = client.post("/scan", content=b"x", headers={"Content-Type": "application/zip",
                                                              **headers})
            assert res.status_code == status, (headers, res.text)
            assert "detail" in res.json()
        assert post(client, "r", wa, b"x", content_type="image/png").status_code == 415
        big = post(client, "r", wa, b"x" * 2000)
        assert big.status_code == 413 and "bigger than" in big.json()["detail"]
        assert not hub.scans.jobs                       # none of them got as far as a job
        assert client.get("/scan/nope").status_code == 404


def test_a_body_without_a_length_is_capped_as_it_streams(web_dir, tmp_path, fake_cli):
    with make_client(web_dir, tmp_path, fake_cli, max_upload_bytes=5000) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")

        def chunks():
            for _ in range(20):
                yield b"y" * 1000

        res = client.post("/scan", content=chunks(),
                          headers={"Content-Type": "application/zip", **who("r", wa)})
        assert res.status_code == 413
        failed = until(a, scan_msg("failed"))
        assert "bigger than" in failed["detail"]
        wait_for(lambda: not any((tmp_path / "scratch").iterdir()))


def test_one_upload_per_person_then_a_cooldown(web_dir, tmp_path, fake_cli, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(scans, "now", lambda: clock[0])
    monkeypatch.setenv("SCAN_FAKE_SLEEP", "1.0")
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch, mode="quick") as (client, _hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        body = make_zip(scan_entries(tmp_path))
        assert post(client, "r", wa, body).status_code == 202
        again = post(client, "r", wa, body)
        assert again.status_code == 409 and "on its way" in again.json()["detail"]
        until(a, scan_msg("done"))
        later = post(client, "r", wa, body)
        assert later.status_code == 429 and int(later.headers["retry-after"]) == 60
        assert later.json()["retry"] == 60
        clock[0] += 61
        assert post(client, "r", wa, body).status_code == 202


def test_one_quick_build_per_room_and_a_newer_scan_stops_the_realistic_one(
        web_dir, tmp_path, fake_cli, monkeypatch):
    monkeypatch.setenv("SCAN_FAKE_REAL", "hang")
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        _b, wb = join(client, stack, "r", "Bo")
        body = make_zip(scan_entries(tmp_path))
        monkeypatch.setenv("SCAN_FAKE_SLEEP", "1.5")
        first = post(client, "r", wa, body).json()["job"]
        busy = post(client, "r", wb, body)                 # the quick build is still running
        assert busy.status_code == 429 and "being built here" in busy.json()["detail"]
        until(a, scan_msg("objects", first))
        wait_for(lambda: children(fake_cli, "realroom"))
        pid = children(fake_cli, "realroom")[0]["pid"]
        assert not gone(pid)
        monkeypatch.setenv("SCAN_FAKE_SLEEP", "0")
        res = post(client, "r", wb, body)                 # its realistic build gives way
        assert res.status_code == 202, res.text
        second = res.json()["job"]
        stopped = until(a, scan_msg("failed", first))
        assert "newer scan" in stopped["detail"]
        wait_for(lambda: gone(pid))
        until(a, scan_msg("objects", second))
        assert client.get(f"/scan/{first}").json()["status"] == "failed"
        assert hub.scans.rooms["r"].job.id == second


def test_a_build_that_takes_too_long_is_killed_and_cleaned_up(web_dir, tmp_path, fake_cli,
                                                              monkeypatch):
    monkeypatch.setenv("SCAN_FAKE_QUICK", "hang")
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch, quick_timeout=1.0) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        job = post(client, "r", wa, make_zip(scan_entries(tmp_path))).json()["job"]
        failed = until(a, scan_msg("failed", job))
        assert "quickroom took longer than 1 s" in failed["detail"]
        pid = children(fake_cli, "quickroom")[0]["pid"]
        wait_for(lambda: gone(pid))
        wait_for(lambda: not any((tmp_path / "scratch").iterdir()))
        assert "r" not in hub.room_versions
        assert recv(a)["v"] == hub.version.id             # still on the room it had


@pytest.mark.skipif(scans.resident_bytes(os.getpid()) is None, reason="reads memory from /proc")
def test_a_build_that_takes_too_much_memory_is_killed_and_cleaned_up(web_dir, tmp_path, fake_cli,
                                                                     monkeypatch):
    """The backstop behind the checks on what a scan holds: whatever a build is made to hold, it
    is held to its share, not the container's (which every room shares)."""
    monkeypatch.setenv("SCAN_FAKE_QUICK", "hog")
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch, quick_memory=64 * 1024 ** 2,
                     memory_seconds=0.05) as (client, hub), contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        job = post(client, "r", wa, make_zip(scan_entries(tmp_path))).json()["job"]
        failed = until(a, scan_msg("failed", job))
        assert failed["detail"] == "Failed: quickroom needed more than 64 MB of memory"
        pid = children(fake_cli, "quickroom")[0]["pid"]
        wait_for(lambda: gone(pid))
        wait_for(lambda: not any((tmp_path / "scratch").iterdir()))
        assert "r" not in hub.room_versions


def test_when_the_builds_together_hold_too_much_the_biggest_goes(tmp_path, monkeypatch):
    mb = 1024 ** 2
    held = {101: 400 * mb, 102: 700 * mb}
    reads = {101: 0}

    def resident(pid):
        if pid == 101:
            reads[101] += 1
            if reads[101] > 3:
                return None                     # its child is gone: the watch ends
        return held[pid]

    monkeypatch.setattr(scans, "resident_bytes", resident)

    async def main():
        svc = scans.ScanService(scans.ScanConfig(mode="quick", children_memory=1000 * mb,
                                                 memory_seconds=0.01),
                                find_peer=lambda *a: None, broadcast=lambda *a: None,
                                install=None, release=lambda *a: None)
        small, big = (scans.ScanJob(j, "r", None, "c", tmp_path, "zip", created=0)
                      for j in ("small", "big"))
        small.procs = [SimpleNamespace(pid=101, returncode=None)]
        big.procs = [SimpleNamespace(pid=102, returncode=None)]
        small.rss = {101: 400 * mb}
        svc.jobs = {"small": small, "big": big}
        killed = []

        def kill(proc):
            killed.append(proc.pid)
            proc.returncode = -9

        monkeypatch.setattr(svc, "_kill_proc", kill)
        said_big: list[str] = []
        said_small: list[str] = []
        await asyncio.wait_for(svc._watch_memory(big, big.procs[0], "realroom", 0, said_big), 5)
        big.procs, big.rss = [], {}                      # as `_child` leaves it
        await asyncio.wait_for(svc._watch_memory(small, small.procs[0], "quickroom", 0,
                                                 said_small), 5)
        held[101] = 600 * mb                            # over its own share now
        reads[101] = 0
        said_own: list[str] = []
        await asyncio.wait_for(svc._watch_memory(small, small.procs[0], "quickroom", 500 * mb,
                                                 said_own), 5)
        return killed, said_big, said_small, said_own

    killed, said_big, said_small, said_own = asyncio.run(main())
    assert killed == [102, 101]
    assert said_big == ["the server ran short of memory for building rooms; try again later"]
    assert said_small == [] and said_own == ["quickroom needed more than 500 MB of memory"]


def test_a_stopped_build_leaves_nothing_for_asyncio_to_complain_about(tmp_path, caplog):
    """Stopping a build cancels its child's task (a newer scan, everyone gone, the server
    stopping); asyncio used to print a "_GatheringFuture exception was never retrieved"
    traceback for each one."""
    async def main():
        loop = asyncio.get_running_loop()
        complaints = []
        loop.set_exception_handler(lambda _loop, context: complaints.append(context["message"]))
        svc = scans.ScanService(
            scans.ScanConfig(mode="quick", command=[sys.executable, "-c",
                                                    "import time; time.sleep(30)"]),
            find_peer=lambda *a: None, broadcast=lambda *a: None, install=None,
            release=lambda *a: None)
        job = scans.ScanJob("s1", "r", None, "c", tmp_path, "zip", created=0)
        task = asyncio.ensure_future(svc._child(job, ["quickroom"], timeout=60))
        await asyncio.sleep(0.3)
        pid = job.procs[0].pid
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        async def broken():
            raise OSError("the store went away")

        await asyncio.gather(svc._spawn(broken()), return_exceptions=True)
        del task
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0.05)
        return complaints, pid

    complaints, pid = asyncio.run(main())
    assert complaints == []
    wait_for(lambda: gone(pid))
    assert "a background task failed: OSError: the store went away" in caplog.text


def test_a_failed_build_says_why_without_the_servers_paths(web_dir, tmp_path, fake_cli,
                                                           monkeypatch):
    monkeypatch.setenv("SCAN_FAKE_QUICK", "fail")
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch) as (client, _hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        job = post(client, "r", wa, make_zip(scan_entries(tmp_path))).json()["job"]
        failed = until(a, scan_msg("failed", job))
        assert "is not a readable RoomPlan export" in failed["detail"]
        assert str(tmp_path) not in failed["detail"] and failed["detail"].startswith("Failed: ")


def test_a_realistic_build_that_fails_keeps_the_versions_it_gave(web_dir, tmp_path, fake_cli,
                                                                 monkeypatch):
    monkeypatch.setenv("SCAN_FAKE_REAL", "fail")
    monkeypatch.setenv("SCAN_FAKE_VERSIONS", "1")
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        job = post(client, "r", wa, make_zip(scan_entries(tmp_path))).json()["job"]
        failed = until(a, scan_msg("failed", job), seconds=20)
        assert "TRELLIS went away" in failed["detail"] and "quick room stays" in failed["detail"]
        own = hub.room_versions["r"]
        assert own.stage == "objects"                    # the one version it wrote got in
        until(a, lambda m: m["t"] == "state" and m["v"] == own.id)


def test_everyone_leaving_stops_the_build_and_lets_the_room_go(web_dir, tmp_path, fake_cli,
                                                               monkeypatch):
    monkeypatch.setenv("SCAN_FAKE_REAL", "hang")
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch) as (client, hub):
        with contextlib.ExitStack() as stack:
            a, wa = join(client, stack, "r", "Ada")
            job = post(client, "r", wa, make_zip(scan_entries(tmp_path))).json()["job"]
            until(a, scan_msg("objects", job))
            wait_for(lambda: children(fake_cli, "realroom"))
            pid = children(fake_cli, "realroom")[0]["pid"]
            assert hub.room_versions["r"].stage == "quick"
        wait_for(lambda: gone(pid))
        wait_for(lambda: "r" not in hub.room_versions)
        wait_for(lambda: not any((tmp_path / "scratch").iterdir()))
        assert client.get(f"/scan/{job}").json()["status"] == "failed"
        with contextlib.ExitStack() as stack:
            _a, wa = join(client, stack, "r", "Ada")
            assert wa["v"] == hub.version.id and "scan" not in wa


def test_a_reload_within_the_keep_window_finds_the_scan_still_there(web_dir, tmp_path, fake_cli,
                                                                    monkeypatch):
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch, mode="quick",
                     keep_seconds=1.0) as (client, hub):
        with contextlib.ExitStack() as stack:
            a, wa = join(client, stack, "r", "Ada")
            job = post(client, "r", wa, make_zip(scan_entries(tmp_path))).json()["job"]
            done = until(a, scan_msg("done", job))
            assert "photos" not in done["detail"]
            scanned = hub.room_versions["r"].id
        with contextlib.ExitStack() as stack:            # the page reloaded
            _a, wa = join(client, stack, "r", "Ada")
            assert wa["v"] == scanned and wa["scan"]["status"] == "done"
        wait_for(lambda: "r" not in hub.room_versions, seconds=5)
        with contextlib.ExitStack() as stack:
            _a, wa = join(client, stack, "r", "Ada")
            assert wa["v"] == hub.version.id


def test_a_live_version_offered_to_every_room_passes_a_scanned_room_by(web_dir, tmp_path,
                                                                       fake_cli, monkeypatch):
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch, mode="quick") as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "scanned", "Ada")
        b, _wb = join(client, stack, "plain", "Bo")
        job = post(client, "scanned", wa, make_zip(scan_entries(tmp_path))).json()["job"]
        until(a, scan_msg("done", job))
        own = hub.room_versions["scanned"]
        published = server.SceneVersion.build(mujoco.MjModel.from_xml_string(
            DEMO.replace('size="0.1"', 'size="0.12"')), label="published", stage="full")
        client.portal.call(_offer, hub, published)
        scene = until(b, lambda m: m["t"] == "scene")
        assert scene["v"] == published.id
        until(b, lambda m: m["t"] == "state" and m["v"] == published.id)
        frame = until(a, lambda m: m["t"] == "state" and m["v"] == own.id)
        assert frame["v"] == own.id and hub.room_versions["scanned"] is own
        assert hub.rooms["scanned"].pending is None
        _c, wc = join(client, stack, "new", "Cy")
        assert wc["v"] == published.id


async def _offer(hub, version):
    hub.offer(version)


def test_a_usdz_alone_gets_a_quick_room_in_flat_colours(web_dir, tmp_path, fake_cli, monkeypatch):
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        res = post(client, "r", wa, usdz_bytes(tmp_path), content_type="model/vnd.usdz+zip")
        assert res.status_code == 202, res.text
        done = until(a, scan_msg("done", res.json()["job"]))
        assert "no photos" in done["detail"]
        run, = children(fake_cli, "quickroom")
        assert run["files"] == ["room.usdz"] and run["argv"][-1] == "--label=scanned room"
        assert not children(fake_cli, "realroom")
        assert hub.room_versions["r"].stage == "quick"


def test_over_the_daily_cap_a_room_still_gets_its_quick_room(web_dir, tmp_path, fake_cli,
                                                             monkeypatch):
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch, fake=False,
                     budget=avatars.MemoryBudget(0)) as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        job = post(client, "r", wa, make_zip(scan_entries(tmp_path))).json()["job"]
        quick = until(a, scan_msg("quick", job))
        done = until(a, scan_msg("done", job))
        assert "today's limit" in quick["detail"] and "today's limit" in done["detail"]
        assert not children(fake_cli, "realroom") and hub.room_versions["r"].stage == "quick"


def test_server_wide_and_per_address_limits(web_dir, tmp_path, fake_cli, monkeypatch):
    monkeypatch.setenv("SCAN_FAKE_SLEEP", "1.0")
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch, mode="quick", max_jobs=1,
                     ip_uploads_per_hour=2) as (client, _hub), contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r1", "Ada")
        _b, wb = join(client, stack, "r2", "Bo")
        _c, wc = join(client, stack, "r3", "Cy")
        body = make_zip(scan_entries(tmp_path))
        assert post(client, "r1", wa, body).status_code == 202
        busy = post(client, "r2", wb, body)
        assert busy.status_code == 429 and "other rooms" in busy.json()["detail"]
        until(a, scan_msg("done"))
        assert post(client, "r2", wb, body).status_code == 202
        until(_b, scan_msg("done"))
        hourly = post(client, "r3", wc, body)
        assert hourly.status_code == 429 and "your network" in hourly.json()["detail"]


def test_a_busy_server_says_so_at_once_without_reading(tmp_path):
    class Peer:
        def __init__(self, pid):
            self.id, self.key = pid, pid * 4

    async def run(tmp: Path):
        ada, bo = Peer("ada"), Peer("bo")
        peers = {("r", "ada"): ada, ("q", "bo"): bo}

        async def install(_room, _ref):
            return "v"

        service = scans.ScanService(
            scans.ScanConfig(mode="quick", upload_slots=1, scratch=tmp),
            find_peer=lambda room, pid: peers.get((room, pid)),
            broadcast=lambda room, message: None, install=install, release=lambda room: None)
        gate, reads = asyncio.Event(), []

        def slow():
            async def body():
                reads.append("ada")
                await gate.wait()
                yield b"not a zip"
            return body()

        def quick():
            async def body():
                reads.append("bo")
                yield b"not a zip"
            return body()

        first = asyncio.ensure_future(service.upload(slow, room="r", peer="ada", key=ada.key,
                                                     content_type="application/zip"))
        while not reads:
            await asyncio.sleep(0)
        with pytest.raises(scans.ScanRequestError) as err:
            await service.upload(quick, room="q", peer="bo", key=bo.key,
                                 content_type="application/zip")
        assert err.value.status == 503 and err.value.retry_after == scans.BUSY_RETRY
        assert reads == ["ada"]
        gate.set()
        with pytest.raises(scans.ScanRequestError) as err:
            await first
        assert err.value.status == 400               # read, then refused: not a zip
        service.close()

    asyncio.run(run(tmp_path / "scratch"))


def test_a_kept_scan_brings_its_room_back_after_a_restart(web_dir, tmp_path, fake_cli,
                                                          monkeypatch):
    store = scans.DirRoomStore(tmp_path / "volume" / hosted.ROOMS_DIR)
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch, mode="quick", store=store,
                     keep_seconds=30) as (client, hub), contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        job = post(client, "r", wa, make_zip(scan_entries(tmp_path))).json()["job"]
        until(a, scan_msg("done", job))
        scanned = hub.room_versions["r"].id
        wait_for(lambda: (store.root / "r" / "room.json").is_file())
    # The container restarted with people still in the call: they reconnect.
    with make_client(web_dir, tmp_path, fake_cli, monkeypatch, mode="quick", store=store,
                     keep_seconds=0) as (client, hub):
        with contextlib.ExitStack() as stack:
            _a, wa = join(client, stack, "r", "Ada")
            assert wa["v"] == scanned
            assert client.get("/scene.json", params={"v": scanned}).json()["label"] == "My Room"
            _b, wb = join(client, stack, "other", "Bo")
            assert wb["v"] == hub.version.id
        # Everyone left for good: the kept copy goes too.
        wait_for(lambda: not (store.root / "r").exists())


def test_the_room_store_keeps_only_what_the_scene_names_and_forgets_old_ones(tmp_path,
                                                                            monkeypatch):
    folder = tmp_path / "quick"
    (folder / "textures").mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(folder / "textures" / "wall.png")
    (folder / "scene.xml").write_text(
        '<mujoco><asset><texture name="w" type="2d" file="textures/wall.png"/></asset>'
        '<worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>')
    (folder / "quick.json").write_text("{}")             # not named by the scene: not kept
    commits = []
    store = scans.DirRoomStore(tmp_path / "rooms", commit=lambda: commits.append(1), ttl=100)
    store.save("r1", folder / "scene.xml", version="room-abc", label="Tea", stage="quick")
    kept = {str(p.relative_to(store.root)) for p in store.root.rglob("*") if p.is_file()}
    assert kept == {"r1/scene.xml", "r1/textures/wall.png", "r1/room.json"} and commits
    ref = store.load("r1")
    assert (ref.id, ref.label, ref.stage) == ("room-abc", "Tea", "quick")
    mujoco.MjModel.from_xml_path(str(ref.scene))
    store.save("../x", folder / "scene.xml", version="v", label="x", stage="quick")
    assert not (tmp_path / "x").exists() and store.load("../x") is None
    real = time.time
    monkeypatch.setattr(scans.time, "time", lambda: real() + 1000)
    assert store.load("r1") is None and not (store.root / "r1").exists()


def test_the_real_quickroom_child_builds_a_scan_sent_into_a_room(web_dir, tmp_path, monkeypatch):
    """The real `litereality quickroom` in a real child process, on a synthetic scan."""
    with make_client(web_dir, tmp_path, None, monkeypatch, mode="quick") as (client, hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r", "Ada")
        res = post(client, "r", wa, make_zip(scan_entries(tmp_path, top="test_room/", frames=0)))
        assert res.status_code == 202, res.text
        seen: list = []
        done = until(a, scan_msg("done"), seconds=60, seen=seen)
        assert [m["status"] for m in seen if m["t"] == "scan"][-3:] == ["building", "quick",
                                                                         "done"], done
        own = hub.room_versions["r"]
        names = {b["name"] for b in own.manifest["bodies"]}
        assert {"Chair0", "Chair1"} <= names and own.label == "test_room"
        until(a, lambda m: m["t"] == "state" and m["v"] == own.id)


@pytest.mark.scan
def test_tea_room_through_the_real_quickroom_and_realroom_children(web_dir, tmp_path,
                                                                   monkeypatch):
    """`LR_TEST_TEA_ROOM=<example-scans>/tea_room uv run pytest -m scan tests/test_scan_upload.py`"""
    folder = os.environ.get("LR_TEST_TEA_ROOM")
    if not folder or not (Path(folder) / "room.usdz").is_file():
        pytest.skip("set LR_TEST_TEA_ROOM to the tea_room scan folder")
    entries = {f"tea_room/{p.name}": p.read_bytes() for p in sorted(Path(folder).iterdir())
               if p.is_file()}
    with make_client(web_dir, tmp_path, None, monkeypatch, fake_delay=0.2,
                     interval=1.0) as (client, hub), contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "tea", "Ada")
        b, _wb = join(client, stack, "other", "Bo")
        started = time.monotonic()
        job = post(client, "tea", wa, make_zip(entries)).json()["job"]
        quick = until(a, lambda m: m["t"] == "scene" and m["stage"] == "quick", seconds=120)
        quick_at = time.monotonic() - started
        seen: list = []
        until(a, scan_msg("done", job), seconds=300, seen=seen)
        versions = [m["v"] for m in seen if m["t"] == "scene" and m["stage"] == "objects"]
        assert versions, seen
        own = hub.room_versions["tea"]
        assert own.stage == "objects" and quick["label"] == "fallside-kitchen"
        frame = until(b, lambda m: m["t"] == "state")
        assert frame["v"] == hub.version.id
        print(f"tea_room: quick room offered after {quick_at:.1f} s, {len(versions)} objects "
              f"version(s), done after {time.monotonic() - started:.1f} s")


# -- the command line -----------------------------------------------------------------------------

def _parse(argv):
    parser = argparse.ArgumentParser()
    roomcall.add_arguments(parser)
    return parser.parse_args(argv)


def test_the_uploads_flag_and_its_default(monkeypatch):
    configured = [False]
    monkeypatch.setattr(cli, "load_settings",
                        lambda: SimpleNamespace(modal_configured=lambda: configured[0]))
    for argv, modal, expect in [(["--demo"], False, "quick"), (["--demo"], True, "realistic"),
                                (["--demo", "--fake-objects"], False, "realistic"),
                                (["--demo", "--uploads", "off"], True, "off")]:
        configured[0] = modal
        args = _parse(argv)
        cli._default_uploads(args)
        assert args.uploads == expect, argv
    served = {}
    monkeypatch.setattr(server, "serve", lambda scene, **kw: served.update(kw) or 0)
    assert server.run(_parse(["--demo", "--uploads", "realistic", "--fake-objects"])) == 0
    assert served["uploads"] == "realistic" and served["fake_objects"] is True
    assert server.run(_parse(["--demo"])) == 0 and served["uploads"] == "quick"
    full = cli._parser().parse_args(["roomcall", "--demo", "--uploads", "quick", "--articulated"])
    assert full.uploads == "quick" and full.articulated is True
    # A scan names its room itself (scan_label.json): it reaches the child as one argument, even
    # when it looks like an option.
    for command in ("quickroom", "realroom"):
        child = cli._parser().parse_args([command, "scan", "--out", "o", "--label=--fake"])
        assert child.label == "--fake" and not getattr(child, "fake", False)


def test_realroom_streams_versions_for_a_watcher(tmp_path, monkeypatch):
    scan = tmp_path / "scans" / "test_room"
    write_roomplan_usdz(scan / "room.usdz")
    _frame_looking_down(scan)
    out = tmp_path / "objects"
    seen = []
    from litereality_agent.pipeline.realroom import build

    original = build.RealRoom.write

    def write(self, **kwargs):
        seen.append(json.loads((self.out_dir / "quick.json").read_text())["label"])
        return original(self, **kwargs)

    monkeypatch.setattr(build.RealRoom, "write", write)
    assert cli.main(["realroom", str(scan), "--out", str(out), "--fake", "--pass", "both",
                     "--interval", "0", "--label", "Room A", "--no-articulated"]) == 0
    report = json.loads((out / build.REPORT_NAME).read_text())
    # A version per arrival (preview, then final), and none before the first object.
    assert report["versions_written"] == 2 == len(seen)
    assert report["objects"]["Table0"]["pass"] == "final"


def test_inside_a_modal_container_trellis_needs_no_token(monkeypatch):
    function = SimpleNamespace(name="trellis")
    lookups = []
    fake_modal = SimpleNamespace(Function=SimpleNamespace(
        from_name=lambda app, name, **kw: lookups.append((app, name, kw)) or function))
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    settings = SimpleNamespace(modal_configured=lambda: False, modal_trellis_app="litereality",
                               modal_trellis_function="trellis")
    args = SimpleNamespace(fake=False)
    monkeypatch.delenv("MODAL_IS_REMOTE", raising=False)
    with pytest.raises(SystemExit, match="MODAL_TOKEN_ID"):
        cli._object_generator(args, settings)
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    generator = cli._object_generator(args, settings)
    assert generator.service.client.function is function
    assert lookups == [("litereality", "trellis", {})]        # the container's own environment


@pytest.mark.parametrize("pace, what", [((0.3, 1), "stalled"), ((0.05, 1), "too slowly")])
def test_an_upload_that_stalls_or_trickles_is_dropped(tmp_path, pace, what):
    class Peer:
        id, key = "ada", "ada-key"

    async def run():
        told = []

        async def install(_room, _ref):
            return "v"

        service = scans.ScanService(
            scans.ScanConfig(mode="quick", scratch=tmp_path / "scratch", chunk_timeout=0.2,
                             slow_grace=0.2, min_upload_rate=1000),
            find_peer=lambda room, pid: Peer() if (room, pid) == ("r", "ada") else None,
            broadcast=lambda room, message: told.append(message), install=install,
            release=lambda room: None)
        delay, size = pace

        def body():
            async def chunks():
                for _ in range(40):
                    await asyncio.sleep(delay)
                    yield b"z" * size
            return chunks()

        with pytest.raises(scans.ScanRequestError) as err:
            await service.upload(body, room="r", peer="ada", key="ada-key",
                                 content_type="application/zip")
        assert err.value.status == 408 and what in err.value.message
        assert told[-1]["status"] == "failed"
        await asyncio.sleep(0.1)                      # the job's folder goes in the background
        assert not any((tmp_path / "scratch").iterdir())
        service.close()

    asyncio.run(run())
