"""The photoreal room: SPZ/PLY files, the instant splat, the training set, the cut to the physics,
and the room call serving it (models/splat, pipeline/splatroom, room_ops/roomcall/splats)."""

from __future__ import annotations

import gzip
import json
import struct
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_quickroom import FLOOR_Y, _frame_looking_down  # noqa: E402

from litereality_agent.models.splat import dataset, ply, rgbd, spz  # noqa: E402
from litereality_agent.models.splat.spz import Splats, quat_to_matrices  # noqa: E402
from litereality_agent.pipeline import splatroom  # noqa: E402
from litereality_agent.pipeline.splatroom.split import corrected_scene, split  # noqa: E402
from litereality_agent.room_ops.roomcall import server  # noqa: E402
from litereality_agent.room_ops.roomcall.splats import SplatSource, room_name  # noqa: E402


def random_splats(n: int = 400, degree: int = 1, seed: int = 0) -> Splats:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sh = rng.uniform(-0.4, 0.4, (n, spz.sh_dim(degree), 3)).astype(np.float32) if degree else None
    return Splats(rng.uniform(-3, 3, (n, 3)).astype(np.float32),
                  np.exp(rng.uniform(-6, -1, (n, 3))).astype(np.float32), q.astype(np.float32),
                  rng.uniform(0.05, 0.95, (n, 3)).astype(np.float32),
                  rng.uniform(0.05, 0.95, n).astype(np.float32), sh)


def test_spz_round_trip_keeps_every_splat_within_its_quantisation():
    s = random_splats()
    data = spz.encode(s)
    magic, version, n, degree, bits, flags, _ = struct.unpack_from("<IIIBBBB", gzip.decompress(data))
    assert (magic, version, n, degree, bits, flags) == (0x5053474E, 3, len(s), 1, 12, 1)
    d = spz.decode(data)
    assert np.abs(d.positions - s.positions).max() <= 0.5 / 4096 + 1e-6
    assert np.abs(d.colors - s.colors).max() < 0.005
    assert np.abs(d.opacities - s.opacities).max() <= 0.5 / 255 + 1e-6
    assert np.abs(d.scales / s.scales - 1).max() < 0.035
    assert np.abs(quat_to_matrices(d.rotations) - quat_to_matrices(s.rotations)).max() < 0.01
    assert np.abs(d.sh - s.sh).max() < 0.07


def test_ply_round_trip_and_sh_trimmed_to_what_is_asked():
    s = random_splats(degree=3)
    back = ply.read(ply.write(s))
    for a, b in ((back.positions, s.positions), (back.scales, s.scales), (back.colors, s.colors),
                 (back.opacities, s.opacities), (back.sh, s.sh)):
        assert np.allclose(a, b, atol=1e-5)
    assert np.allclose(quat_to_matrices(back.rotations), quat_to_matrices(s.rotations), atol=1e-5)
    assert ply.read(ply.write(s), max_sh_degree=1).sh.shape == (len(s), 3, 3)


@pytest.fixture
def scan(tmp_path):
    folder = tmp_path / "scan"
    folder.mkdir()
    _frame_looking_down(folder)
    return folder


def test_instant_splats_lie_on_what_the_lidar_saw_in_its_colours(scan):
    s = rgbd.instant_splats(rgbd.load_frames(scan), voxel=0.02)
    assert len(s) > 500
    z = s.positions[:, 2]
    top, floor = np.abs(z - (FLOOR_Y + 0.75)) < 0.01, np.abs(z - FLOOR_Y) < 0.01
    assert (top | floor).mean() > 0.97                     # nothing floats between the two
    red = s.colors[top]
    assert red[:, 0].mean() > 0.6 and red[:, 1].mean() < 0.3
    normals = quat_to_matrices(s.rotations)[:, :, 2]       # a disc's thin axis is its normal
    assert np.abs(normals[:, 2]).mean() > 0.98
    assert np.all(s.scales[:, 2] < s.scales[:, 0])


def _read_colmap(root: Path):
    sparse = root / "sparse" / "0"
    cams = {}
    raw = (sparse / "cameras.bin").read_bytes()
    (n,), at = struct.unpack_from("<Q", raw), 8
    for _ in range(n):
        cid, model, w, h, fx, fy, cx, cy = struct.unpack_from("<iiQQdddd", raw, at)
        at += struct.calcsize("<iiQQdddd")
        cams[cid] = (model, w, h, fx, fy, cx, cy)
    raw = (sparse / "images.bin").read_bytes()
    (n,), at = struct.unpack_from("<Q", raw), 8
    images = []
    for _ in range(n):
        iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack_from("<i4d3di", raw, at)
        at += struct.calcsize("<i4d3di")
        end = raw.index(b"\0", at)
        name = raw[at:end].decode()
        at = end + 1
        (m,) = struct.unpack_from("<Q", raw, at)
        at += 8
        pts = np.frombuffer(raw, dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")], count=m,
                            offset=at)
        at += pts.nbytes
        images.append((iid, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz]), cid, name, pts))
    raw = (sparse / "points3D.bin").read_bytes()
    (n,) = struct.unpack_from("<Q", raw)
    rec = np.frombuffer(raw, dtype=[("id", "<u8"), ("xyz", "<f8", 3), ("rgb", "u1", 3),
                                    ("err", "<f8"), ("len", "<u8"), ("img", "<i4"),
                                    ("idx", "<i4")], count=n, offset=8)
    return cams, images, rec


def _qvec_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_the_training_set_is_a_colmap_model_whose_lidar_points_reproject(scan, tmp_path):
    ds = dataset.build(rgbd.load_frames(scan), width=160, points_per_frame=300)
    root = ds.write(tmp_path / "ds")
    cams, images, points = _read_colmap(root)
    assert ds.size == (160, 120) and len(images) == 1 and len(points) == ds.points == 300
    (iid, q, t, cid, name, pts), = images
    assert name == "frame_00000.jpg" and (root / "images" / name).is_file()
    model, w, h, fx, fy, cx, cy = cams[cid]
    assert (model, w, h) == (1, 160, 120)
    cam = points["xyz"] @ _qvec_matrix(q).T + t
    assert np.all(cam[:, 2] > 0)                         # OpenCV: in front is +z
    uv = np.stack([fx * cam[:, 0] / cam[:, 2] + cx, fy * cam[:, 1] / cam[:, 2] + cy], 1)
    by_id = {int(p): k for k, p in enumerate(points["id"])}
    order = [by_id[int(i)] for i in pts["id"]]
    assert np.abs(uv[order] - np.stack([pts["x"], pts["y"]], 1)).max() < 1e-3
    assert np.all(points["img"] == iid) and np.all(points["len"] == 1)
    z = points["xyz"][:, 2]
    assert np.all((np.abs(z - FLOOR_Y) < 0.01) | (np.abs(z - (FLOOR_Y + 0.75)) < 0.01))


SCENE = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <geom name="wall" type="box" pos="0 2 1" size="3 0.05 1"/>
    <body name="Box0" pos="{x} 0 0.25">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.25" group="3"/>
    </body>
  </worldbody>
</mujoco>"""


def _grid(lo, hi, step=0.04):
    axes = [np.arange(a, b + 1e-9, step) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def _splats_at(points: np.ndarray) -> Splats:
    n = len(points)
    q = np.tile([0, 0, 0, 1.0], (n, 1)).astype(np.float32)
    return Splats(points.astype(np.float32), np.full((n, 3), 0.01, np.float32), q,
                  np.full((n, 3), 0.5, np.float32), np.full(n, 0.9, np.float32))


def _room_splats(box_x: float) -> Splats:
    box = _grid((box_x - 0.2, -0.2, 0.02), (box_x + 0.2, 0.2, 0.5))
    floor = _grid((-2, -2, 0), (2, 1.9, 0))
    wall = _grid((-2, 1.95, 0), (2, 1.95, 2))
    return _splats_at(np.concatenate([box, floor, wall]))


def test_a_body_is_cut_out_in_its_own_frame_and_the_floor_under_it_stays(tmp_path):
    model = mujoco.MjModel.from_xml_string(SCENE.format(x=1.0))
    s = _room_splats(1.0)
    parts = split(s, model, snap_bodies=False)
    body = parts.bodies["Box0"]
    assert len(body) > 300 and not parts.corrections
    assert np.abs(body.positions.mean(0) - (0, 0, 0)).max() < 0.06   # body frame: centred
    assert body.positions[:, 2].min() > -0.25 + 0.01                  # no floor taken
    assert len(parts.room) + len(body) == len(s)
    under = (np.abs(parts.room.positions[:, 0] - 1.0) < 0.15) & (parts.room.positions[:, 2] == 0)
    assert under.sum() > 20


def test_a_body_scanned_off_its_object_is_snapped_onto_it_and_the_scene_follows(tmp_path):
    xml = tmp_path / "scene.xml"
    xml.write_text(SCENE.format(x=0.8))                  # RoomPlan put it 0.2 m off
    model = mujoco.MjModel.from_xml_path(str(xml))
    parts = split(_room_splats(1.0), model)
    fix = parts.corrections["Box0"]
    assert abs(fix["dx"] - 0.2) <= 0.021 and abs(fix["dy"]) <= 0.021
    assert fix["after"] > fix["before"]
    assert np.abs(parts.bodies["Box0"].positions.mean(0)[:2]).max() < 0.06
    fixed = corrected_scene(xml, parts.corrections)
    moved = mujoco.MjModel.from_xml_path(str(fixed))
    data = mujoco.MjData(moved)
    mujoco.mj_forward(moved, data)
    assert abs(data.xpos[moved.body("Box0").id][0] - 1.0) <= 0.021
    assert corrected_scene(xml, {}) is None


def test_bundles_are_written_content_named_and_served_to_the_room(tmp_path):
    model = mujoco.MjModel.from_xml_string(SCENE.format(x=1.0))
    out = tmp_path / "splat"
    first = splatroom.write_bundle(out, split(_room_splats(1.0), model, snap_bodies=False),
                                   stage="instant")
    assert set(first["bodies"]) == {"Box0"} and (out / first["room"]).is_file()
    floor_only = _splats_at(_grid((-1, -1, 0), (1, 1, 0)))
    again = splatroom.write_bundle(out, split(floor_only, model, snap_bodies=False),
                                   stage="trained", step=300)
    assert sorted(p.name for p in out.glob("*.spz")) == [again["room"]]   # old files dropped
    assert splatroom.read_bundle(out)["stage"] == "trained"

    app = server.create_app(model=model, label="tea room", web_dir=None,
                            splats=SplatSource.fixed(out))
    with TestClient(app) as client:
        found = client.get("/splat.json")
        assert found.status_code == 200
        body = found.json()
        assert body["stage"] == "trained" and body["step"] == 300 and body["bodies"] == {}
        name = room_name(app.state.roomcall.version.id)
        assert body["room"] == f"splat/{name}/{again['room']}"
        got = client.get("/" + body["room"])
        assert got.status_code == 200 and got.content == (out / again["room"]).read_bytes()
        assert "immutable" in got.headers["cache-control"]
        assert client.get(f"/splat/{name}/../splat.json").status_code == 404
        assert client.get(f"/splat/{name}/room-0123.spz").status_code == 404
        assert client.get("/splat.json?v=nope").status_code == 404

    plain = server.create_app(model=model, label="tea room", web_dir=None)
    with TestClient(plain) as client:
        assert client.get("/splat.json").status_code == 404


def test_a_bundle_naming_anything_but_its_own_files_is_ignored(tmp_path):
    (tmp_path / "splat.json").write_text(json.dumps({"room": "../../etc/passwd", "bodies": {}}))
    assert SplatSource.fixed(tmp_path).bundle(None) is None
    (tmp_path / "splat.json").write_text(json.dumps(
        {"room": "room-0123456789abcdef.spz", "bodies": {"Chair0": "/abs.spz"}}))
    assert SplatSource.fixed(tmp_path).bundle(None) is None
    assert SplatSource.per_room(tmp_path).bundle("../x") is None
