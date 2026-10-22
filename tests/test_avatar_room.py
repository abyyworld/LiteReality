"""Full-body people in the room call: the LRAV file, the stand-in body, the upload limits, and the
server's avatar store and protocol (POST /avatar, GET /avatars/<id>.bin, the "avatar" and "j"
messages). Offline: the GPU builder is a fake `modal` module."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
import re
import struct
import sys
import time
import types
import zlib
from pathlib import Path
from urllib.parse import urlencode

import anyio
import mujoco
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from litereality_agent import cli
from litereality_agent.room_ops import roomcall
from litereality_agent.room_ops.roomcall import avatar_format as af
from litereality_agent.room_ops.roomcall import avatars, hosted, server

SCENE = """
<mujoco model="avatar-test">
  <option timestep="0.004"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <body name="box" pos="0 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
</mujoco>
"""
IMPORT_MAP = ('{ "imports": {\n'
              '  "three": "https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js"\n'
              '}}')
INDEX = f"""<!doctype html><html><head><meta charset="utf-8"><title>t</title>
<script type="importmap">{IMPORT_MAP}</script><script type="module" src="app.js"></script>
</head><body></body></html>
"""
POSE_STANDIN = b"stand-in pose model"
WEB = server.WEB_DIR


# -- images --------------------------------------------------------------------------------------

def png(width: int = 200, height: int = 300, color=(255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, "PNG")
    return buf.getvalue()


def jpeg(width: int = 320, height: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (120, 90, 60)).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def shirt_and_trousers() -> bytes:
    """White background, a red "shirt" over a blue "trousers" rectangle."""
    image = np.full((300, 200, 3), 255, np.uint8)
    image[40:150, 70:130] = (200, 30, 30)
    image[150:280, 75:125] = (30, 30, 200)
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, "PNG")
    return buf.getvalue()


def png_claiming(width: int, height: int) -> bytes:
    """A PNG whose header claims width x height, with a token of image data."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\x00" * 64)) + chunk(b"IEND", b""))


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# -- the file --------------------------------------------------------------------------------------

def tiny_avatar(n: int = 400, seed: int = 1) -> af.Avatar:
    """A valid avatar: n splats in a 1.7 m column on the 22 body joints, each on one joint."""
    rng = np.random.default_rng(seed)
    positions = np.stack([rng.uniform(-0.2, 0.2, n), rng.uniform(0, 1.7, n),
                          rng.uniform(-0.1, 0.1, n)], axis=1).astype(np.float32)
    positions[0, 1], positions[1, 1] = 0.0, 1.7
    joints = rng.integers(0, af.BODY_JOINTS, (n, 1))
    rest = avatars.STANDIN_REST.copy()
    colors = rng.integers(0, 256, (n, 4)).astype(np.uint8)
    rotations = np.zeros((n, 4), np.float32)
    rotations[:, 3] = 1
    return af.Avatar(positions=positions, scales=np.full((n, 3), 0.01, np.float32),
                     rotations=rotations, colors=colors,
                     skin=af.quantize_skin(joints, np.ones((n, 1))),
                     joint_names=list(af.SMPLX_JOINTS[:af.BODY_JOINTS]),
                     parents=list(af.SMPLX_PARENTS[:af.BODY_JOINTS]), rest=rest, floor_y=0.0,
                     kind="standin", meta={"note": "test"})


def rewrite(data: bytes, edit) -> bytes:
    """The file with its header passed through `edit` (re-padded), data block unchanged."""
    hlen = struct.unpack_from("<I", data, 8)[0]
    header = json.loads(data[16:16 + hlen])
    edit(header)
    text = json.dumps(header).encode()
    padded = (len(text) + 15) // 16 * 16
    return (data[:8] + struct.pack("<II", padded, 0) + text.ljust(padded, b" ")
            + data[16 + hlen:])


def array_at(data: bytes, name: str) -> int:
    hlen = struct.unpack_from("<I", data, 8)[0]
    header = json.loads(data[16:16 + hlen])
    return 16 + hlen + header["arrays"][name]["offset"]


def test_the_file_round_trips():
    a = tiny_avatar()
    b = af.unpack(af.pack(a, shuffle=False))
    assert b.count == a.count and b.kind == "standin" and b.meta == {"note": "test"}
    np.testing.assert_array_equal(b.positions, a.positions)
    np.testing.assert_allclose(b.scales, a.scales, rtol=1e-3)
    np.testing.assert_allclose(b.rotations, a.rotations, atol=1e-3)
    np.testing.assert_array_equal(b.colors, a.colors)
    np.testing.assert_array_equal(b.skin, a.skin)
    np.testing.assert_allclose(b.rest, a.rest, atol=1e-6)
    assert b.joint_names == a.joint_names and b.parents == a.parents and b.floor_y == 0.0


def test_the_shuffle_is_a_seeded_permutation_and_a_prefix_is_a_fair_sample():
    a = tiny_avatar(n=4000)
    order = np.argsort(a.positions[:, 1])                  # sorted by height: a prefix is biased
    a = af.Avatar(**{**a.__dict__, **{k: getattr(a, k)[order] for k in
                                      ("positions", "scales", "rotations", "colors", "skin")}})
    data = af.pack(a, seed=3)
    assert data == af.pack(a, seed=3) and data != af.pack(a, seed=4)
    b = af.unpack(data)
    rows = lambda x: sorted(map(tuple, x.positions.tolist()))   # noqa: E731
    assert rows(a) == rows(b)                               # the same splats, reordered
    skin_of = {tuple(p): s for p, s in zip(a.positions.tolist(), a.skin.tolist(), strict=True)}
    assert all(skin_of[tuple(p)] == s for p, s in zip(b.positions.tolist(), b.skin.tolist(),
                                                       strict=True))
    prefix = b.positions[:400, 1]
    assert abs(prefix.mean() - a.positions[:, 1].mean()) < 0.1
    assert a.positions[:400, 1].mean() < 0.3                # the unshuffled prefix is not


def _bad_files() -> dict[str, bytes]:
    good = af.pack(tiny_avatar())
    out = {"magic": b"XXXX" + good[4:],
           "version": good[:4] + struct.pack("<I", 2) + good[8:],
           "truncated": good[:-64],
           "trailing": good + b"\0" * 32,
           "count": rewrite(good, lambda h: h.update(count=af.MAX_SPLATS + 1)),
           "parents": rewrite(good, lambda h: h["joints"]["parents"].__setitem__(5, 9)),
           "names": rewrite(good, lambda h: h["joints"]["names"].__setitem__(3, "Spine1")),
           "too big": good + b"\0" * (af.MAX_BYTES - len(good) + 1)}
    nan = bytearray(good)
    nan[array_at(good, "positions"):array_at(good, "positions") + 4] = struct.pack("<f", np.nan)
    out["nan"] = bytes(nan)
    joint = bytearray(good)
    joint[array_at(good, "skin"):array_at(good, "skin") + 2] = struct.pack("<H", 255 | 40 << 8)
    out["joint past J"] = bytes(joint)
    weights = bytearray(good)
    weights[array_at(good, "skin"):array_at(good, "skin") + 2] = struct.pack("<H", 200)
    out["weights"] = bytes(weights)
    return out


@pytest.mark.parametrize("case", sorted(_bad_files()))
def test_every_bad_file_is_refused(case):
    with pytest.raises(af.AvatarFormatError):
        af.unpack(_bad_files()[case])


def test_pack_refuses_what_unpack_would():
    small = tiny_avatar()
    small.positions = small.positions * 0.1                  # 17 cm tall: not a person
    with pytest.raises(af.AvatarFormatError, match="tall"):
        af.pack(small)
    many = tiny_avatar(n=af.MAX_SPLATS + 1)
    with pytest.raises(af.AvatarFormatError, match="splats"):
        af.pack(many)


@pytest.mark.parametrize("joints, weights", [
    (np.array([[3], [7]]), np.array([[1.0], [0.2]])),
    (np.arange(12).reshape(2, 6), np.array([[1, 5, 2, 0.5, 9, 3], [0.1, 0.1, 0.1, 0.1, 0.1, 9]])),
    (np.array([[4, 5, 6], [1, 2, 3]]), np.zeros((2, 3))),
])
def test_quantize_skin_keeps_four_heaviest_summing_to_255(joints, weights):
    skin = af.quantize_skin(joints, weights)
    assert skin.shape == (2, 4) and skin.dtype == np.uint16
    w = (skin & 0xFF).astype(int)
    assert (w.sum(axis=1) == 255).all()
    assert (np.diff(w, axis=1) <= 0).all()                  # heaviest first
    if weights.shape[1] == 6:
        assert (skin[0] >> 8).tolist()[0] == 4 and (skin[1] >> 8).tolist()[0] == 11
    if not weights.any():
        assert ((skin >> 8)[:, 0] == joints[:, 0]).all() and (w[:, 0] == 255).all()


def test_head_mask_is_the_head_joints_majority():
    skin = af.quantize_skin(np.array([[15, 12], [12, 15], [22, 9], [16, 17]]),
                            np.array([[0.6, 0.4], [0.6, 0.4], [1, 0], [1, 0]]))
    a = tiny_avatar(n=4)
    a.skin = skin
    assert a.head_mask().tolist() == [True, False, True, False]


# -- the stand-in ----------------------------------------------------------------------------------

def test_the_standin_is_a_valid_deterministic_person():
    images = [shirt_and_trousers(), jpeg()]
    started = time.perf_counter()
    first = avatars.make_standin(images)
    assert time.perf_counter() - started < 1.0
    data = af.pack(first)
    assert data == af.pack(avatars.make_standin(images))
    a = af.unpack(data)
    assert a.count == avatars.STANDIN_SPLATS and a.kind == "standin" and a.floor_y == 0.0
    assert abs(float(a.positions[:, 1].min())) < 0.01        # feet on the floor
    assert 1.70 < float(np.percentile(a.positions[:, 1], 99.8)) < 1.80
    weights, joints = a.skin & 0xFF, a.skin >> 8
    assert set(joints[weights > 0].ravel().tolist()) == set(range(af.BODY_JOINTS))
    assert a.head_mask().sum() > 500
    assert a.meta["note"] and set(a.meta["tints"]) == {"skin", "hair", "top", "bottom"}


def test_the_standin_wears_the_colours_in_the_photo():
    a = avatars.make_standin([shirt_and_trousers()])
    heaviest = (a.skin >> 8)[:, 0]
    torso = a.colors[np.isin(heaviest, [6, 9]), :3].astype(float).mean(axis=0)
    legs = a.colors[np.isin(heaviest, [1, 2, 4, 5]), :3].astype(float).mean(axis=0)
    assert torso[0] > 150 and torso[2] < 80                  # red shirt
    assert legs[2] > 150 and legs[0] < 80                    # blue trousers


def photo(kind: str) -> bytes:
    """Synthetic 640x480 photos of the framings person_tints has to read."""
    yy, xx = np.mgrid[:480, :640]
    red, blue, skin = (200, 30, 30), (40, 40, 160), (205, 160, 130)
    if kind == "waist-up":                     # a grey wall; the red shirt runs off both sides
        image = np.full((480, 640, 3), 200, np.uint8)
        image[((xx - 320) / 70) ** 2 + ((yy - 120) / 90) ** 2 < 1] = skin
        image[yy > 230] = red
    elif kind == "too close":                  # a big head; blue shoulders fill the frame below
        image = np.full((480, 640, 3), (120, 130, 125), np.uint8)
        image[((xx - 320) / 90) ** 2 + ((yy - 160) / 120) ** 2 < 1] = (180, 140, 110)
        image[yy > 300] = blue
    elif kind == "sky":                        # a sky shading from dark to pale, sand below
        t = (yy / 480)[..., None]
        image = ((1 - t) * np.array([40, 90, 200]) + t * np.array([200, 220, 240])).astype(np.uint8)
        image[yy > 430] = (190, 170, 140)
        image[((xx - 320) / 22) ** 2 + ((yy - 75) / 28) ** 2 < 1] = skin
        image[(yy > 105) & (yy < 250) & (abs(xx - 320) < 50)] = (30, 160, 60)
        image[(yy >= 250) & (yy < 425) & (abs(xx - 320) < 40)] = (60, 50, 40)
    else:                                      # "floor": full length, a floor from the knees down
        image = np.full((480, 640, 3), (215, 210, 200), np.uint8)
        image[yy > 260] = (120, 90, 60)
        image[((xx - 320) / 22) ** 2 + ((yy - 60) / 28) ** 2 < 1] = skin
        image[(yy > 90) & (yy < 220) & (abs(xx - 320) < 50)] = red
        legs = (yy >= 220) & (yy < 340) & (abs(xx - 320) < 40) & (abs(xx - 320) > 6)
        image[legs] = (30, 30, 200)
        image[(yy >= 340) & (yy < 350) & (abs(xx - 320) < 45)] = (20, 20, 20)
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def near(rgb, want, tol: int = 12) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(rgb, want, strict=True))


def test_tints_when_clothes_run_off_the_frame():
    """Rows the person fills edge to edge are the person: their ends are no background."""
    waist_up = avatars.person_tints([photo("waist-up")])
    assert near(waist_up["skin"], (205, 160, 130)) and near(waist_up["top"], (200, 30, 30))
    assert waist_up["bottom"] == avatars.DEFAULT_TINTS["bottom"]      # no trousers in the photo
    close = avatars.person_tints([photo("too close")])
    assert near(close["skin"], (180, 140, 110)) and near(close["top"], (40, 40, 160))


def test_tints_with_a_shaded_sky_or_a_floor_in_front():
    """A background shading from top to bottom is not the person (the row-end test), and a
    floor seen behind and in front of someone's legs is not their clothes."""
    sky = avatars.person_tints([photo("sky")])
    assert near(sky["skin"], (205, 160, 130)) and near(sky["top"], (30, 160, 60))
    assert near(sky["bottom"], (60, 50, 40))
    floor = avatars.person_tints([photo("floor")])
    assert near(floor["skin"], (205, 160, 130)) and near(floor["top"], (200, 30, 30))
    assert near(floor["bottom"], (30, 30, 200))


def test_tints_without_a_clear_person():
    # Nothing stands out from the background: the central box is taken as the person.
    grey = avatars.person_tints([png(color=(90, 90, 90))])
    assert set(grey.values()) == {(90, 90, 90)}
    assert avatars.person_tints([b"not an image"]) == avatars.DEFAULT_TINTS
    assert avatars.person_tints([]) == avatars.DEFAULT_TINTS


# -- the upload ------------------------------------------------------------------------------------

def body(images=None, **extra) -> bytes:
    message = {"room": "r", "peer": "p", "key": "k", "mode": "photo",
               "images_b64": [b64(jpeg())] if images is None else images, **extra}
    return json.dumps(message).encode()


def test_a_good_upload_parses():
    up = avatars.parse_upload(body([b64(jpeg()), b64(png())], extra="ignored"))
    assert (up.room, up.peer, up.key, up.mode) == ("r", "p", "k", "photo")
    assert len(up.images) == 2 and up.images[0][:3] == b"\xff\xd8\xff"


@pytest.mark.parametrize("raw, message", [
    (b"not json", "JSON"),
    (b"[1, 2]", "object"),
    (body(room=5), "required"),
    (body(key=""), "required"),
    (body(mode="selfie"), "mode"),
    (body([]), "1 to 8"),
    (body([b64(jpeg())] * 9), "1 to 8"),
    (body("abc"), "1 to 8"),
    (body([12]), "base64"),
    (body(["!!!not base64!!!"]), "base64"),
    (body(["data:image/png;base64," + b64(png())]), "base64"),
    (body([b64(b"GIF89a" + b"\0" * 100)]), "JPEG or PNG"),
    (body([b64(b"\xff\xd8\xff" + b"\0" * 500)]), "could not be read"),
    (body([b64(png(100, 100))]), "each side"),
    (body([b64(png(5000, 200))]), "each side"),
    (body([b64(png(4000, 3500))]), "too many pixels"),
    (body([b64(png_claiming(30000, 30000))]), "too many pixels"),
])
def test_bad_uploads_are_400(raw, message):
    with pytest.raises(avatars.AvatarRequestError) as err:
        avatars.parse_upload(raw)
    assert err.value.status == 400 and message in err.value.message


def test_an_oversized_image_or_body_is_refused(monkeypatch):
    big = b"\xff\xd8\xff" + bytes(avatars.MAX_IMAGE_BYTES)
    with pytest.raises(avatars.AvatarRequestError) as err:
        avatars.parse_upload(body([b64(big)]))
    assert err.value.status == 400 and "MiB" in err.value.message
    monkeypatch.setattr(avatars, "MAX_UPLOAD_BYTES", 1000)
    with pytest.raises(avatars.AvatarRequestError) as err:
        avatars.parse_upload(body([b64(jpeg())]))
    assert err.value.status == 413


@pytest.mark.parametrize("raw", [
    b'{"pad": [' + b"[]," * 5000 + b"[]]}",           # millions of objects at full size
    b'{"pad": [' + b"0," * 5000 + b"0]}",
    b'{"a": {"b": {"c": {"d": {}}}}}',
])
def test_upload_json_of_another_shape_is_refused_unparsed(monkeypatch, raw):
    def loads(_text):
        raise AssertionError("parsed")

    monkeypatch.setattr(avatars, "json", types.SimpleNamespace(loads=loads))
    with pytest.raises(avatars.AvatarRequestError) as err:
        avatars.parse_upload(raw)
    assert err.value.status == 400 and "not a body upload" in err.value.message


def test_images_are_checked_without_decoding_a_pixel(monkeypatch):
    from PIL import ImageFile

    def load(_self):
        raise AssertionError("decoded")

    monkeypatch.setattr(ImageFile.ImageFile, "load", load)
    up = avatars.parse_upload(body([b64(jpeg(2000, 1500)), b64(png(1000, 800))]))
    assert len(up.images) == 2
    damaged = bytearray(png())
    damaged[45] ^= 0xFF                                      # inside IDAT: its checksum fails
    with pytest.raises(avatars.AvatarRequestError, match="could not be read"):
        avatars.parse_upload(body([b64(bytes(damaged))]))


def test_who_is_asking_may_come_from_the_request_and_must_agree():
    raw = json.dumps({"mode": "photo", "images_b64": [b64(jpeg())]}).encode()
    up = avatars.parse_upload(raw, room="r", peer="p", key="k")
    assert (up.room, up.peer, up.key) == ("r", "p", "k")
    with pytest.raises(avatars.AvatarRequestError, match="another room"):
        avatars.parse_upload(body(), room="r", peer="someone else", key="k")
    with pytest.raises(avatars.AvatarRequestError, match="required"):
        avatars.parse_upload(raw)


def test_parsing_in_threads_leaves_the_warning_filters_alone():
    import warnings
    from concurrent.futures import ThreadPoolExecutor

    before = list(warnings.filters)
    raw = body([b64(jpeg())] * 4)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: avatars.parse_upload(raw), range(32)))
    assert warnings.filters == before


def test_client_keys():
    assert avatars.client_key("203.0.113.9") == "203.0.113.9"
    assert avatars.client_key("::ffff:203.0.113.9") == "203.0.113.9"
    # One network holds a whole IPv6 /64: every address in it is the same client.
    assert avatars.client_key("2001:db8:1:2::1") == avatars.client_key("2001:db8:1:2:ffff::9")
    assert avatars.client_key("2001:db8:1:2::1") != avatars.client_key("2001:db8:1:3::1")
    assert avatars.client_key(None) == "" and avatars.client_key("testclient") == "testclient"


# -- the service, before and after the body is read --------------------------------------------------

class FakePeer:
    def __init__(self, pid: str) -> None:
        self.id, self.key, self.avatar_id, self.avatar_info = pid, pid * 4, None, None


def fake_service(peers: dict, **config):
    sent: list = []
    service = avatars.AvatarService(
        avatars.AvatarConfig(**{"mode": "standin", **config}),
        find_peer=lambda room, pid: peers.get((room, pid)),
        broadcast=lambda room, message: sent.append((room, message)),
        tell=lambda peer, message: sent.append((peer.id, message)))
    return service, sent


async def _hang(_images, _mode, _aid):
    await asyncio.Event().wait()


def test_strangers_and_limits_are_refused_before_the_body_is_read(monkeypatch):
    parsed, reads = [], []
    parse = avatars.parse_upload
    monkeypatch.setattr(avatars, "parse_upload", lambda *a, **k: parsed.append(1) or parse(*a, **k))
    raw = json.dumps({"mode": "photo", "images_b64": [b64(jpeg())]}).encode()

    async def read():
        reads.append(1)
        return raw

    async def run():
        ada = FakePeer("ada")
        service, _sent = fake_service({("r", "ada"): ada}, builder=_hang)

        async def attempt(room="r", peer="ada", key=ada.key):
            try:
                await service.upload(read, room=room, peer=peer, key=key, client="203.0.113.9")
            except avatars.AvatarRequestError as exc:
                return exc.status
            return 202

        assert [await attempt(peer="nobody"), await attempt(key="0" * 16),
                await attempt(room="../r"), await attempt(key=None),
                await attempt(key="kéy")] == [403, 403, 403, 400, 403]
        assert not reads and not parsed
        assert await attempt() == 202 and len(reads) == len(parsed) == 1
        assert await attempt() == 409                        # still building: not read again
        assert len(reads) == len(parsed) == 1
        service.close()

    asyncio.run(run())


def test_a_busy_server_says_so_at_once_without_reading():
    raw = json.dumps({"mode": "photo", "images_b64": [b64(jpeg())]}).encode()

    async def run():
        ada, bo = FakePeer("ada"), FakePeer("bo")
        service, _sent = fake_service({("r", "ada"): ada, ("q", "bo"): bo}, builder=_hang,
                                      upload_slots=1)
        gate, reads = asyncio.Event(), []

        async def slow():
            reads.append("ada")
            await gate.wait()
            return raw

        async def quick():
            reads.append("bo")
            return raw

        first = asyncio.create_task(service.upload(slow, room="r", peer="ada", key=ada.key))
        while not reads:
            await asyncio.sleep(0)
        with pytest.raises(avatars.AvatarRequestError) as err:
            await service.upload(quick, room="q", peer="bo", key=bo.key)
        assert err.value.status == 503 and err.value.retry_after == avatars.BUSY_RETRY
        assert reads == ["ada"]
        gate.set()
        assert (await first)["status"] == "building"
        assert (await service.upload(quick, room="q", peer="bo", key=bo.key))["status"] == "building"
        service.close()

    asyncio.run(run())


# -- the server ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def web_dir(tmp_path_factory):
    web = tmp_path_factory.mktemp("avatar-web")
    (web / "index.html").write_text(INDEX)
    (web / "app.js").write_text("// app\n")
    (web / "app.css").write_text("")
    (web / "body.js").write_text("export const body = 1;\n")
    (web / "pose_worker.js").write_text("// worker\n")
    (web / server.POSE_MODEL).write_bytes(POSE_STANDIN)
    return web


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(SCENE)


@contextlib.contextmanager
def make_client(model, web_dir, **config):
    app = server.create_app(model=model, web_dir=web_dir, settle_seconds=0.02,
                            avatars=avatars.AvatarConfig(**{"mode": "standin", **config}))
    with TestClient(app) as client:
        yield client, app.state.roomcall


def recv(ws, timeout: float = 5.0) -> dict:
    async def get():
        with anyio.fail_after(timeout):
            return await ws._send_rx.receive()
    message = ws.portal.call(get)
    if message["type"] == "websocket.close":
        raise WebSocketDisconnect(message.get("code", 1000))
    return json.loads(message["text"])


def until(ws, predicate, limit: int = 400) -> dict:
    for _ in range(limit):
        message = recv(ws)
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
    """The headers that say who is uploading."""
    return {avatars.ROOM_HEADER: room, avatars.PEER_HEADER: welcome["id"],
            avatars.KEY_HEADER: welcome["key"]}


def post(client, room: str, welcome: dict, images=None, **extra):
    data = {"room": room, "peer": welcome["id"], "key": welcome["key"], "mode": "turnaround",
            "images_b64": images or [b64(shirt_and_trousers())], **extra}
    return client.post("/avatar", content=json.dumps(data),
                       headers={"Content-Type": "application/json", **who(room, welcome)})


def avatar_msg(status: str, peer: str | None = None):
    return lambda m: (m["t"] == "avatar" and m["status"] == status
                      and (peer is None or m["id"] == peer))


def test_a_body_is_built_announced_and_served(model, web_dir):
    with make_client(model, web_dir) as (client, hub), contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "r1", "Ada")
        b, wb = join(client, stack, "r1", "Bo")
        assert re.fullmatch(r"[0-9a-f]{32}", wa["key"]) and wa["key"] != wb["key"]
        assert wa["avatars"] == {"enabled": True, "mode": "standin", "cooldown": 120,
                                 "max_images": 8}
        res = post(client, "r1", wa)
        assert res.status_code == 202, res.text
        made = res.json()
        assert made["status"] == "building" and avatars.AVATAR_ID.fullmatch(made["avatar"])
        assert re.fullmatch(r"[0-9a-f]{32}", made["key"])
        for ws in (a, b):
            until(ws, avatar_msg("building", wa["id"]))
            ready = until(ws, avatar_msg("ready", wa["id"]))
            assert ready["url"] == f"/avatars/{made['avatar']}.bin" and ready["kind"] == "standin"
        got = client.get(ready["url"])
        assert got.status_code == 200 and got.headers["content-type"] == "application/octet-stream"
        assert got.headers["cache-control"] == "private, max-age=86400, immutable"
        assert got.headers["etag"] == f'"{made["avatar"]}"'
        assert "content-security-policy" in got.headers
        assert af.unpack(got.content).kind == "standin"
        # A late joiner hears of it in its welcome; the room's info for everyone else is as before.
        c, wc = join(client, stack, "r1", "Cy")
        info = {p["name"]: p for p in wc["peers"]}
        assert info["Ada"]["avatar"] == {"status": "ready", "url": ready["url"], "kind": "standin"}
        assert "avatar" not in info["Bo"]
        assert client.get("/healthz").json()["avatars"] == {
            "mode": "standin", "stored": 1, "building": 0, "gpu_today": 0}
        # The key is only this browser's: nobody else's works, and no stranger's.
        assert post(client, "r1", {**wb, "key": wa["key"]}).status_code == 403
        assert post(client, "r1", {**wa, "id": "nobody"}).status_code == 403
        assert post(client, "elsewhere", wa).status_code == 403


@pytest.mark.parametrize("name", ["../x", "a123.bin", "A0123456789ABCDEF.bin",
                                  "a0123456789abcdef.bin", "a0123456789abcdef", "x.bin"])
def test_unknown_or_malformed_avatar_names_are_404(model, web_dir, name):
    with make_client(model, web_dir) as (client, _hub):
        res = client.get("/avatars/" + name)
        assert res.status_code == 404


def test_avatars_off_means_503_and_a_disabled_welcome(model, web_dir):
    with make_client(model, web_dir, mode="off") as (client, _hub), \
            contextlib.ExitStack() as stack:
        _a, wa = join(client, stack, "off", "Ada")
        assert wa["avatars"] == {"enabled": False}
        assert post(client, "off", wa).status_code == 503


def test_oversized_bodies_are_413_before_they_are_read(model, web_dir, monkeypatch):
    monkeypatch.setattr(avatars, "MAX_UPLOAD_BYTES", 500)
    with make_client(model, web_dir) as (client, _hub), contextlib.ExitStack() as stack:
        _a, wa = join(client, stack, "big", "Ada")
        assert post(client, "big", wa).status_code == 413
        res = client.post("/avatar", content=b"x" * 600)
        assert res.status_code == 413


def test_the_cooldown_is_per_connection_and_time_lets_it_through(model, web_dir, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(avatars, "now", lambda: clock[0])
    with make_client(model, web_dir) as (client, _hub), contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "cool", "Ada")
        assert post(client, "cool", wa).status_code == 202
        until(a, avatar_msg("ready"))
        again = post(client, "cool", wa)
        assert again.status_code == 429
        assert int(again.headers["retry-after"]) == 120 and again.json()["retry"] == 120
        clock[0] += 121
        assert post(client, "cool", wa).status_code == 202
        until(a, avatar_msg("ready"))
        # The new body replaced the old one: one stored.
        assert client.get("/healthz").json()["avatars"]["stored"] == 1


def test_one_build_per_person_and_the_room_cap(model, web_dir):
    release = None

    async def slow(images, mode, aid):
        await release.wait()
        return await avatars.build_standin(images, mode, aid, splats=500)

    with make_client(model, web_dir, builder=slow, room_max_building=1) as (client, _hub), \
            contextlib.ExitStack() as stack:
        release = client.portal.call(_event)
        a, wa = join(client, stack, "cap", "Ada")
        _b, wb = join(client, stack, "cap", "Bo")
        assert post(client, "cap", wa).status_code == 202
        assert post(client, "cap", wa).status_code == 409        # mine is still building
        capped = post(client, "cap", wb)
        assert capped.status_code == 429 and "retry" in capped.json()
        client.portal.call(release.set)
        until(a, avatar_msg("ready"))
        assert post(client, "cap", wb).status_code == 202


async def _event() -> asyncio.Event:
    return asyncio.Event()


def test_the_room_hourly_cap_survives_the_room(model, web_dir):
    with make_client(model, web_dir, room_builds_per_hour=1, cooldown=0) as (client, _hub):
        with contextlib.ExitStack() as stack:
            a, wa = join(client, stack, "hour", "Ada")
            assert post(client, "hour", wa).status_code == 202
            until(a, avatar_msg("ready"))
        with contextlib.ExitStack() as stack:                   # the room was dropped and remade
            _b, wb = join(client, stack, "hour", "Bo")
            assert post(client, "hour", wb).status_code == 429


def test_the_store_is_bounded(model, web_dir):
    with make_client(model, web_dir, max_stored=1, cooldown=0) as (client, _hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "full", "Ada")
        _b, wb = join(client, stack, "full", "Bo")
        assert post(client, "full", wa).status_code == 202
        until(a, avatar_msg("ready"))
        assert post(client, "full", wb).status_code == 503       # Ada's is live: not evicted


def test_who_is_asking_comes_in_headers(model, web_dir):
    with make_client(model, web_dir) as (client, _hub), contextlib.ExitStack() as stack:
        _a, wa = join(client, stack, "hdr", "Ada")
        whole = {"room": "hdr", "peer": wa["id"], "key": wa["key"], "mode": "photo",
                 "images_b64": [b64(jpeg())]}
        res = client.post("/avatar", content=json.dumps(whole))           # no headers
        assert res.status_code == 400 and "required" in res.json()["error"]
        res = client.post("/avatar", content=json.dumps({**whole, "room": "elsewhere"}),
                          headers=who("hdr", wa))
        assert res.status_code == 400
        res = client.post("/avatar", content=json.dumps({"mode": "photo",
                                                         "images_b64": [b64(jpeg())]}),
                          headers=who("hdr", wa))
        assert res.status_code == 202


def test_limits_per_client_address(model, web_dir):
    """Connections and room ids are free to make: uploads per hour, bodies held and bodies read
    per minute are also counted per client address (TestClient's is "testclient")."""
    with make_client(model, web_dir, cooldown=0, ip_uploads_per_hour=2) as (client, _hub), \
            contextlib.ExitStack() as stack:
        for room in ("ip1", "ip2"):
            ws, w = join(client, stack, room, "Ada")
            assert post(client, room, w).status_code == 202
            until(ws, avatar_msg("ready"))
        _ws, w = join(client, stack, "ip3", "Bo")
        res = post(client, "ip3", w)
        assert res.status_code == 429 and "network" in res.json()["error"]
        assert int(res.headers["retry-after"]) > 3000
    with make_client(model, web_dir, cooldown=0, ip_max_stored=1) as (client, _hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "held", "Ada")
        _b, wb = join(client, stack, "held2", "Bo")
        assert post(client, "held", wa).status_code == 202
        until(a, avatar_msg("ready"))
        assert post(client, "held2", wb).status_code == 429     # the address holds one already
        assert post(client, "held", wa).status_code == 202      # replacing your own is fine
    with make_client(model, web_dir, ip_attempts_per_minute=2) as (client, _hub), \
            contextlib.ExitStack() as stack:
        _a, wa = join(client, stack, "tries", "Ada")
        assert post(client, "tries", wa, images=["!!!"]).status_code == 400
        assert post(client, "tries", wa, images=["!!!"]).status_code == 400
        res = post(client, "tries", wa)
        assert res.status_code == 429 and "retry" in res.json()


def test_builds_in_flight_are_capped_server_wide(model, web_dir):
    with make_client(model, web_dir, builder=_hang, max_building=1) as (client, _hub), \
            contextlib.ExitStack() as stack:
        _a, wa = join(client, stack, "fl1", "Ada")
        _b, wb = join(client, stack, "fl2", "Bo")
        assert post(client, "fl1", wa).status_code == 202
        res = post(client, "fl2", wb)                       # another room, the same server
        assert res.status_code == 429 and "server" in res.json()["error"]


def test_a_build_that_never_ends_becomes_a_standin(model, web_dir):
    with make_client(model, web_dir, builder=_hang, build_timeout=0.2) as (client, _hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "slow", "Ada")
        assert post(client, "slow", wa).status_code == 202
        ready = until(a, avatar_msg("ready"))
        assert ready["kind"] == "standin" and "took too long" in ready["note"]


def test_each_client_gets_a_share_of_the_days_gpu_builds(model, web_dir):
    calls = []

    async def gpu(images, mode, aid):
        calls.append(aid)
        data, _kind, _note = await avatars.build_standin(images, mode, aid, splats=500)
        return data, "lhm", ""

    with make_client(model, web_dir, mode="modal", builder=gpu, ip_gpu_per_day=1,
                     cooldown=0) as (client, _hub), contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "share", "Ada")
        assert post(client, "share", wa).status_code == 202
        assert until(a, avatar_msg("ready"))["kind"] == "lhm"
        assert post(client, "share", wa).status_code == 202
        ready = until(a, avatar_msg("ready"))
        assert ready["kind"] == "standin" and "share" in ready["note"] and len(calls) == 1


def test_a_full_store_drops_the_body_of_an_idle_connection(model, web_dir, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(avatars, "now", lambda: clock[0])
    with make_client(model, web_dir, max_stored=1, cooldown=0,
                     idle_evict_seconds=60) as (client, _hub), contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "idle", "Ada")
        b, wb = join(client, stack, "idle2", "Bo")
        assert post(client, "idle", wa).status_code == 202
        until(a, avatar_msg("ready"))
        clock[0] += 30
        a.send_json({"t": "pose", "p": [0, 0, 1.6], "yaw": 0})       # Ada is here
        until(a, lambda m: m["t"] == "state" and wa["id"] in m["avatars"])
        clock[0] += 30
        assert post(client, "idle2", wb).status_code == 503         # not idle long enough
        clock[0] += 31
        assert post(client, "idle2", wb).status_code == 202
        assert until(a, avatar_msg("none"))["id"] == wa["id"]
        until(b, avatar_msg("ready", wb["id"]))


class SpentBudget:
    used = 40

    async def take(self) -> bool:
        return False


def test_the_daily_gpu_cap_gives_a_standin_with_a_note(model, web_dir):
    with make_client(model, web_dir, mode="modal", budget=SpentBudget()) as (client, _hub), \
            contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "gpu", "Ada")
        assert wa["avatars"]["mode"] == "modal"
        assert post(client, "gpu", wa).status_code == 202
        ready = until(a, avatar_msg("ready"))
        assert ready["kind"] == "standin" and "daily GPU budget used up" in ready["note"]
        assert client.get("/healthz").json()["avatars"]["gpu_today"] == 40


def frames_until(ws, predicate, limit: int = 400) -> list[dict]:
    seen = []
    for _ in range(limit):
        seen.append(recv(ws))
        if predicate(seen[-1]):
            return seen
    raise AssertionError("no matching frame")


def test_leave_then_claim_from_a_new_connection_and_expiry(model, web_dir, monkeypatch):
    clock = [5000.0]
    monkeypatch.setattr(avatars, "now", lambda: clock[0])
    with make_client(model, web_dir) as (client, hub), contextlib.ExitStack() as stack:
        watcher, _ww = join(client, stack, "claim", "Watcher")
        with client.websocket_connect("/ws?room=claim") as a:
            a.send_json({"t": "hello", "name": "Ada"})
            wa = recv(a)
            made = post(client, "claim", wa).json()
            ready = until(a, avatar_msg("ready"))
        until(watcher, lambda m: m["t"] == "leave")
        # A forged key and a malformed id are ignored; the right key moves it to the new connection.
        b, wb = join(client, stack, "claim", "Ada")
        b.send_json({"t": "avatar", "claim": made["avatar"], "key": "0" * 32})
        b.send_json({"t": "avatar", "claim": "../etc", "key": made["key"]})
        b.send_json({"t": "avatar", "claim": made["avatar"], "key": made["key"]})
        back = until(watcher, avatar_msg("ready"))
        assert back["id"] == wb["id"] and back["url"] == ready["url"]
        # A second connection claiming it takes it over; the first is told it has none.
        with client.websocket_connect("/ws?room=claim") as c:
            c.send_json({"t": "hello", "name": "Ada again"})
            wc = recv(c)
            c.send_json({"t": "avatar", "claim": made["avatar"], "key": made["key"]})
            until(watcher, avatar_msg("none", wb["id"]))
            until(watcher, avatar_msg("ready", wc["id"]))
        until(watcher, lambda m: m["t"] == "leave" and m["id"] == wc["id"])
        assert client.get(ready["url"]).status_code == 200      # released, still claimable
        clock[0] += avatars.KEEP_SECONDS + 1
        assert client.get(ready["url"]).status_code == 404      # forgotten
        b.send_json({"t": "avatar", "claim": made["avatar"], "key": made["key"]})
        b.send_json({"t": "pose", "p": [0, 0, 1.6], "yaw": 0})
        seen = frames_until(watcher, lambda m: m["t"] == "state" and wb["id"] in m["avatars"])
        assert not [m for m in seen if m["t"] == "avatar"]
        assert hub.avatars.stats()["stored"] == 0


def test_a_tab_that_takes_over_a_building_body_leaves_the_first_one_told(model, web_dir):
    """A duplicated tab carries sessionStorage, so it claims the first tab's body while it is
    still being built: the first tab and its room hear what it has left."""
    release, hold = None, [False]

    async def builder(images, mode, aid):
        if hold[0]:
            await release.wait()
        return await avatars.build_standin(images, mode, aid, splats=500)

    with make_client(model, web_dir, builder=builder, cooldown=0) as (client, _hub), \
            contextlib.ExitStack() as stack:
        release = client.portal.call(_event)
        watcher, _ww = join(client, stack, "dup", "Watcher")
        _a, wa = join(client, stack, "dup", "Tab1")
        b, wb = join(client, stack, "dup", "Tab2")
        hold[0] = True
        made = post(client, "dup", wa).json()
        until(watcher, avatar_msg("building", wa["id"]))
        b.send_json({"t": "avatar", "claim": made["avatar"], "key": made["key"]})
        assert until(watcher, lambda m: m["t"] == "avatar" and m["id"] == wa["id"]) == {
            "t": "avatar", "id": wa["id"], "status": "none"}
        until(watcher, avatar_msg("building", wb["id"]))
        client.portal.call(release.set)
        until(watcher, avatar_msg("ready", wb["id"]))
        _c, wc = join(client, stack, "dup", "Late")
        assert "avatar" not in {p["name"]: p for p in wc["peers"]}["Tab1"]
        # Tab1 with a body of its own, building a new one that Tab2 takes: Tab1 keeps the old.
        hold[0] = False
        assert post(client, "dup", wa).status_code == 202
        old = until(watcher, avatar_msg("ready", wa["id"]))
        hold[0] = True
        release = client.portal.call(_event)
        made = post(client, "dup", wa).json()
        until(watcher, avatar_msg("building", wa["id"]))
        b.send_json({"t": "avatar", "claim": made["avatar"], "key": made["key"]})
        told = until(watcher, lambda m: m["t"] == "avatar" and m["id"] == wa["id"])
        assert told["status"] == "ready" and told["url"] == old["url"]
        client.portal.call(release.set)
        until(watcher, avatar_msg("ready", wb["id"]))


def test_a_claim_that_does_not_check_out_gets_none_back_to_the_claimer(model, web_dir):
    with make_client(model, web_dir) as (client, _hub), contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "gone", "Ada")
        a.send_json({"t": "avatar", "claim": "a0123456789abcdef", "key": "0" * 32})
        assert until(a, lambda m: m["t"] == "avatar") == {"t": "avatar", "id": wa["id"],
                                                         "status": "none"}


JOINTS = list(range(-19, 20))                          # 39 values


@pytest.mark.parametrize("j, length", [
    (JOINTS, 44),
    ([1.0] * 39, 44),                                  # whole floats: JSON has one number type
    (JOINTS[:-1], 5),
    ([1.5] * 39, 5),
    ([401] + [0] * 38, 5),
    (["NaN"] + [0] * 38, 5),
    ([True] + [0] * 38, 5),
    ("1,2,3", 5),
])
def test_tracked_joints_ride_on_the_state_rows(model, web_dir, j, length):
    with make_client(model, web_dir) as (client, _hub), contextlib.ExitStack() as stack:
        a, wa = join(client, stack, "pose", "Ada")
        a.send_json({"t": "pose", "p": [0.5, 0.2, 1.6], "yaw": 0.25, "pitch": -0.1, "j": j})
        row = until(a, lambda m: m["t"] == "state" and wa["id"] in m["avatars"])["avatars"][wa["id"]]
        assert len(row) == length and row[:5] == [0.5, 0.2, 1.6, 0.25, -0.1]
        if length == 44:
            assert row[5:] == [int(v) for v in j]
            a.send_json({"t": "pose", "p": [0.5, 0.2, 1.6], "yaw": 0.25, "pitch": -0.1})
            until(a, lambda m: m["t"] == "state" and len(m["avatars"][wa["id"]]) == 5)


def test_clean_joints():
    assert server.clean_joints([0] * 39) == [0] * 39
    assert server.clean_joints([400.0] * 39) == [400] * 39
    for bad in ([0] * 38, [float("nan")] + [0] * 38, [float("inf")] + [0] * 38, None, {}):
        assert server.clean_joints(bad) is None


# -- the GPU builder (a fake `modal`) ---------------------------------------------------------------

class FakeFunction:
    calls: list = []

    def __init__(self, reply):
        self._reply = reply
        self.remote = types.SimpleNamespace(aio=self._call)

    async def _call(self, request):
        FakeFunction.calls.append(request)
        reply = self._reply
        return await reply(request) if callable(reply) else reply


def fake_modal(monkeypatch, reply, *, missing: bool = False):
    module = types.ModuleType("modal")

    class NotFoundError(Exception):
        pass

    names = []

    def from_name(app, function, environment_name=None):
        names.append((app, function, environment_name))
        if missing:
            raise NotFoundError("no app")
        return FakeFunction(reply)

    module.Function = types.SimpleNamespace(from_name=from_name)
    module.exception = types.SimpleNamespace(NotFoundError=NotFoundError)
    monkeypatch.setitem(sys.modules, "modal", module)
    FakeFunction.calls = []
    return names


def run_builder(images, *, timeout=5.0, budget=None, aid="a0123456789abcdef"):
    builder = avatars.ModalBuilder(environment="main", budget=budget or avatars.MemoryBudget(5),
                                   timeout=timeout, splats=500)
    return asyncio.run(builder(images, "turnaround", aid))


def test_the_gpu_builder_success_path(monkeypatch):
    lrav = af.pack(tiny_avatar())
    names = fake_modal(monkeypatch, lambda r: _reply({"avatar_b64": b64(lrav), "id": r["id"],
                                                      "seconds": 1.0, "stats": {}}))
    data, kind, note = run_builder([jpeg()])
    assert (data, kind, note) == (lrav, "lhm", "")
    assert names == [("litereality-avatar", "avatar", "main")]
    request = FakeFunction.calls[0]
    assert request["mode"] == "turnaround" and request["id"] == "a0123456789abcdef"
    assert base64.b64decode(request["images_b64"][0]) == jpeg()


async def _reply(value):
    return value


async def _never(_request):
    await asyncio.sleep(10)


@pytest.mark.parametrize("reply, missing, why", [
    ({"error": "no person found in the images", "id": "a0123456789abcdef"}, False, "no person"),
    ({"avatar_b64": "QUJD", "id": "a0000000000000000"}, False, "another request"),
    ({"avatar_b64": b64(b"LRAV garbage"), "id": "a0123456789abcdef"}, False, "GPU builder failed"),
    ({"avatar_b64": "not base64!", "id": "a0123456789abcdef"}, False, "not base64"),
    (None, True, "isn't deployed"),
    (_never, False, "too long"),
])
def test_the_gpu_builder_falls_back_to_the_standin(monkeypatch, reply, missing, why):
    fake_modal(monkeypatch, reply, missing=missing)
    data, kind, note = run_builder([jpeg()], timeout=0.2)
    assert kind == "standin" and why in note
    assert af.unpack(data).kind == "standin"


def test_no_modal_client_or_no_budget_is_a_standin(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)           # `import modal` raises ImportError
    _data, kind, note = run_builder([jpeg()])
    assert kind == "standin" and "Modal client" in note
    fake_modal(monkeypatch, {"error": "unused"})
    _data, kind, note = run_builder([jpeg()], budget=SpentBudget())
    assert kind == "standin" and "budget" in note and not FakeFunction.calls


def test_budgets_count_per_utc_day():
    day = ["2026-10-22"]
    budget = avatars.MemoryBudget(2, today=lambda: day[0])
    assert [asyncio.run(budget.take()) for _ in range(3)] == [True, True, False]
    day[0] = "2026-10-23"
    assert asyncio.run(budget.take()) and budget.used == 1


def test_the_modal_dict_budget_fails_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", types.ModuleType("modal"))   # no Dict at all
    assert asyncio.run(avatars.ModalDictBudget(5).take()) is False


def test_a_hung_budget_store_is_a_spent_budget(monkeypatch):
    async def hang(*_args, **_kwargs):
        await asyncio.sleep(30)

    module = types.ModuleType("modal")
    call = types.SimpleNamespace(aio=hang)
    store = types.SimpleNamespace(get=call, put=call)
    module.Dict = types.SimpleNamespace(from_name=lambda name, create_if_missing: store)
    monkeypatch.setitem(sys.modules, "modal", module)
    started = time.perf_counter()
    assert asyncio.run(avatars.ModalDictBudget(5, timeout=0.1).take()) is False
    assert time.perf_counter() - started < 2


def test_gpu_calls_are_capped_and_a_long_wait_is_a_standin(monkeypatch):
    lrav = af.pack(tiny_avatar())
    gate = None

    async def reply(request):
        await gate.wait()
        return {"avatar_b64": b64(lrav), "id": request["id"], "seconds": 1.0, "stats": {}}

    fake_modal(monkeypatch, reply)

    async def run():
        nonlocal gate
        gate = asyncio.Event()
        budget = avatars.MemoryBudget(5)
        builder = avatars.ModalBuilder(environment="main", budget=budget, timeout=5, splats=500,
                                       gpu_calls=1, queue_seconds=0.2)
        first = asyncio.create_task(builder([jpeg()], "photo", "a0123456789abcdef"))
        while not FakeFunction.calls:
            await asyncio.sleep(0.01)
        _data, kind, note = await builder([jpeg()], "photo", "a1111111111111111")
        assert kind == "standin" and "busy" in note
        assert budget.used == 1 and len(FakeFunction.calls) == 1   # the wait spent no budget
        gate.set()
        assert (await first)[1] == "lhm"

    asyncio.run(run())


# -- the real client, the hosted app, the CLI -------------------------------------------------------

def test_the_real_client_files_are_served(model):
    app = server.create_app(model=model, settle_seconds=0.02)
    with TestClient(app) as client:
        for name in ("body.js", "pose_worker.js"):
            res = client.get("/" + name)
            assert res.status_code == 200 and res.headers["content-type"].startswith(
                "text/javascript") and res.content == (WEB / name).read_bytes()
        task = client.get("/" + server.POSE_MODEL)
        assert task.status_code == 200 and len(task.content) == 5_777_746
        assert hashlib.sha256(task.content).hexdigest() == (
            "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a")
        again = client.get("/" + server.POSE_MODEL, headers={"If-None-Match": task.headers["etag"]})
        assert again.status_code == 304
        csp = client.get("/").headers["content-security-policy"]
    assert "'wasm-unsafe-eval'" in csp and "'unsafe-eval'" not in csp
    page = (WEB / "index.html").read_text("utf-8")
    pins = dict(re.findall(r'"(three|three-mesh-bvh|@sparkjsdev/spark)": "[^"]*@([\d.]+)/', page))
    assert pins == {"three": "0.180.0", "three-mesh-bvh": "0.9.15", "@sparkjsdev/spark": "2.3.1"}
    app_js = (WEB / "app.js").read_text("utf-8")
    assert "import('@sparkjsdev/spark')" in app_js
    assert not re.search(r"^\s*import\b[^;(]*['\"]@sparkjsdev/spark['\"]", app_js, re.M)
    assert re.search(r"^import \{[^}]*\} from './body\.js';", app_js, re.M)
    for header in (avatars.ROOM_HEADER, avatars.PEER_HEADER, avatars.KEY_HEADER):
        assert f"'{header}'" in app_js.lower()               # who uploads, as the server reads it
    body_js = (WEB / "body.js").read_text("utf-8")
    assert not re.search(r"^\s*import\b", body_js, re.M) and "import(" not in body_js


def test_the_avatar_file_reader_in_body_js_is_the_normative_one():
    """§1.5 of the contract is copied verbatim into body.js; it must keep mirroring the Python."""
    text = (WEB / "body.js").read_text("utf-8")
    assert "export function parseAvatar(buf)" in text and "export function poseSkeleton(" in text
    assert "0x5641524c" in text and f"LRAV_MAX_SPLATS = {af.MAX_SPLATS}" in text
    assert f"HEAD_JOINTS = [{', '.join(map(str, af.HEAD_JOINTS))}]" in text
    names = re.search(r"BODY_JOINT_NAMES = \[([^\]]*)\]", text).group(1)
    assert re.findall(r"'(\w+)'", names) == list(af.SMPLX_JOINTS[:af.BODY_JOINTS])


def test_hosted_builds_with_gpu_bodies_without_touching_modal(tmp_path, monkeypatch):
    class Untouchable(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError(f"modal.{name} touched while building the app")

    monkeypatch.setitem(sys.modules, "modal", Untouchable("modal"))
    app = hosted.build_app(tmp_path / "volume", scratch=tmp_path / "demo")
    config = app.state.roomcall.avatars.config
    assert config.mode == "modal" and config.environment is None
    assert isinstance(config.budget, avatars.ModalDictBudget)
    assert config.budget.cap == avatars.GPU_DAILY_CAP


def test_the_avatars_flag():
    args = cli._parser().parse_args(["roomcall", "--demo"])
    assert args.avatars == "standin"
    args = cli._parser().parse_args(["roomcall", "--demo", "--avatars", "modal"])
    assert args.avatars == "modal"
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["roomcall", "--avatars", "gpu"])
    assert roomcall.AVATAR_MODES == avatars.MODES


def test_modal_bodies_without_the_client_exit_with_a_hint(monkeypatch):
    args = cli._parser().parse_args(["roomcall", "--demo", "--avatars", "modal"])
    monkeypatch.setitem(sys.modules, "modal", None)
    monkeypatch.setenv("MODAL_TOKEN_ID", "before")         # restored after the test
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "before")
    with pytest.raises(SystemExit, match="uv sync --extra modal"):
        roomcall.run(args, credentials=("id", "secret"))
    assert os.environ["MODAL_TOKEN_ID"] == "id"            # exported for this process first


def test_serve_passes_the_avatar_mode(monkeypatch, web_dir, tmp_path):
    import uvicorn

    seen = {}

    class Server:
        started = True

        def __init__(self, config):
            seen["app"] = config.app

        def run(self, sockets):
            pass

    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(server, "WEB_DIR", web_dir)
    scene = tmp_path / "scene.xml"
    scene.write_text(SCENE)
    assert server.serve(scene, port=0, open_browser=False, avatars="modal",
                        modal_env="dev") == 0
    config = seen["app"].state.roomcall.avatars.config
    assert (config.mode, config.environment) == ("modal", "dev")


def test_the_web_dir_has_what_the_server_names():
    for name in ("index.html", "app.js", "app.css", "body.js", "pose_worker.js",
                 server.SEGMENTER_MODEL, server.POSE_MODEL):
        assert (Path(server.WEB_DIR) / name).is_file(), name
