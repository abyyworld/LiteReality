"""The objects stage: each quick-room object's photo -> a generated mesh, fitted into its body and
swapped into the scene as it arrives, with the physics left exactly as the quick room had it.

Everything runs offline: the generator is `FakeGenerator` (a textured box in TRELLIS's own GLB
conventions) or a fake Modal function, the scan is the quick room's synthetic RoomPlan export
(`test_quickroom.write_roomplan_usdz`) with one synthetic capture frame looking down at the table.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_quickroom import _frame_looking_down, write_roomplan_usdz  # noqa: E402

from litereality_agent import cli  # noqa: E402
from litereality_agent.models.trellis import modal as trellis_modal  # noqa: E402
from litereality_agent.pipeline.realroom import (  # noqa: E402
    FakeGenerator,
    ModalTrellisGenerator,
    RealRoom,
    build,
    crops,
    fake_glb,
    fit,
    generate,
    generate_room,
    write_realistic_scene,
)
from litereality_agent.room_ops.roomcall.model_info import (  # noqa: E402
    COLLISION_GROUP,
    VISUAL_GROUP,
    movable_body_ids,
)


@pytest.fixture()
def scan(tmp_path) -> Path:
    folder = tmp_path / "scans" / "test_room"
    write_roomplan_usdz(folder / "room.usdz")
    return folder


@pytest.fixture()
def seen_scan(scan) -> Path:
    _frame_looking_down(scan)
    return scan


def _glb(mesh: trimesh.Trimesh) -> bytes:
    return mesh.export(file_type="glb")


def _body(model, name: str) -> int:
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert b >= 0, name
    return b


def _geoms_of(model, body: int) -> list[int]:
    start, count = int(model.body_geomadr[body]), int(model.body_geomnum[body])
    return list(range(start, start + count))


# ── the stand-in generator ───────────────────────────────────────────────────────────────────────


def test_fake_glb_is_shaped_like_a_trellis_glb():
    photo = Image.new("RGBA", (64, 48), (200, 30, 30, 255))
    buffer = io.BytesIO()
    photo.save(buffer, format="PNG")
    data = fake_glb(buffer.getvalue())
    assert data[:4] == b"glTF"
    mesh = fit.load_glb(data)
    lo, hi = mesh.bounds
    assert np.all(lo >= -0.5) and np.all(hi <= 0.5)            # inside the unit cube
    material = mesh.visual.material
    assert material.baseColorTexture is not None              # a PBR base colour, like TRELLIS
    assert len(mesh.visual.uv) == len(mesh.vertices)
    # The +Z face exists and faces +Z: that is the front.
    front = mesh.face_normals[:, 2] > 0.99
    assert front.sum() == 2
    small = fit.load_glb(fake_glb(buffer.getvalue(), pass_name="preview"))
    assert np.all(small.extents < mesh.extents)


def test_fake_generator_streams_every_pass_and_fails_on_request():
    gen = FakeGenerator(fail=["B"])
    got = list(gen.stream({"A": b"", "B": b""}, ["preview", "final"]))
    assert [(r.handle, r.pass_name, r.ok) for r in got] == [
        ("A", "preview", True), ("B", "preview", False), ("A", "final", True),
        ("B", "final", False)]
    with pytest.raises(ValueError, match="unknown pass"):
        list(gen.stream({"A": b""}, ["best"]))


# ── fitting ──────────────────────────────────────────────────────────────────────────────────────


def _marked_box() -> trimesh.Trimesh:
    """A box with a marker cube on its FRONT (+Z in glTF) and near its TOP (+Y)."""
    body = trimesh.creation.box((0.6, 0.8, 0.4))
    marker = trimesh.creation.box((0.1, 0.1, 0.1),
                                  transform=trimesh.transformations.translation_matrix(
                                      (0.0, 0.3, 0.25)))
    return trimesh.util.concatenate([body, marker])


def test_the_mesh_fills_the_box_front_to_front_and_up_to_up():
    placement = fit.Placement(half=np.array([0.5, 0.25, 0.4]))
    fitted = fit.fit(_glb(_marked_box()), placement)
    v = fitted.vertices
    assert np.allclose(v.min(axis=0), -placement.half, atol=1e-6)
    assert np.allclose(v.max(axis=0), placement.half, atol=1e-6)
    # The marker (glTF front +Z, up +Y) ends at the box's front (-Y) and top (+Z).
    front = v[v[:, 1] < v[:, 1].min() + 1e-6]
    assert front[:, 2].mean() > 0.2
    assert fitted.yaw == 0


def test_a_mesh_long_the_wrong_way_is_turned_a_quarter():
    # 0.3 wide, 1.2 deep in glTF; the box is 2.0 wide, 0.6 deep: only a quarter turn fits.
    mesh = trimesh.creation.box((0.3, 0.5, 1.2))
    placement = fit.Placement(half=np.array([1.0, 0.3, 0.4]))
    fitted = fit.fit(_glb(mesh), placement)
    assert fitted.yaw in (90, 270)
    assert fitted.scores[0]["aspect_err"] > 1.0 > fitted.scores[90]["aspect_err"]


def _side_camera(distance: float = 3.0) -> fit.View:
    """A camera on the world's +X axis at 0.4 m, looking back along -X (ARKit camera axes)."""
    c2w = np.eye(4)
    c2w[:3, 0] = (0.0, 1.0, 0.0)          # camera +X (image right) = world +Y
    c2w[:3, 1] = (0.0, 0.0, 1.0)          # camera +Y (image up) = world +Z
    c2w[:3, 2] = (1.0, 0.0, 0.0)          # camera looks down its -Z = world -X
    c2w[:3, 3] = (distance, 0.0, 0.4)
    K = np.array([[200.0, 0, 128.0], [0, 200.0, 96.0], [0, 0, 1.0]])
    return fit.View(0, K, np.linalg.inv(c2w), np.zeros((192, 256), bool))


def _chair() -> trimesh.Trimesh:
    """A seat with a tall back at the REAR (glTF -Z), in the unit cube."""
    seat = trimesh.creation.box((0.8, 0.1, 0.8),
                                transform=trimesh.transformations.translation_matrix((0, -0.1, 0)))
    back = trimesh.creation.box((0.8, 0.9, 0.1),
                                transform=trimesh.transformations.translation_matrix(
                                    (0, 0.0, -0.35)))
    legs = [trimesh.creation.box((0.06, 0.45, 0.06),
                                 transform=trimesh.transformations.translation_matrix(
                                     (sx * 0.35, -0.32, sz * 0.35)))
            for sx in (-1, 1) for sz in (-1, 1)]
    return trimesh.util.concatenate([seat, back, *legs])


def test_the_cameras_tell_front_from_back():
    placement = fit.Placement(half=np.array([0.25, 0.25, 0.4]), world_centre=np.array([0, 0, 0.4]))
    chair = _chair()
    vertices = np.asarray(chair.vertices) @ fit.YUP_TO_BODY.T
    faces = np.asarray(chair.faces)
    view = _side_camera()
    # What the capture saw: the chair the OTHER way round from what the generator returned.
    seen = placement.to_world(fit.place(vertices, 180, placement))
    view = fit.View(0, view.K, view.w2c, fit.silhouette(seen, faces, view))
    assert view.mask.sum() > 200
    yaw, scores = fit.choose_yaw(vertices, faces, placement, [view])
    assert yaw == 180
    assert scores[180]["iou"] > 0.95 > scores[0]["iou"]
    # ...and without cameras RoomPlan's own front stands.
    assert fit.choose_yaw(vertices, faces, placement)[0] == 0


def test_heavy_meshes_are_decimated_with_their_texture():
    sphere = trimesh.creation.icosphere(subdivisions=4)
    uv = (sphere.vertices[:, :2] + 1) / 2
    image = Image.new("RGB", (8, 8), (10, 200, 10))
    sphere.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    placement = fit.Placement(half=np.array([0.3, 0.3, 0.3]))
    fitted = fit.fit(_glb(sphere), placement, max_faces=1000)
    assert fitted.decimated and fitted.faces_in > 1000
    assert len(fitted.faces) <= 1000
    assert fitted.uv is not None and len(fitted.uv) == len(fitted.vertices)
    assert fitted.texture is not None


def test_floor_standing_meshes_are_put_on_the_floor():
    placement = fit.Placement(half=np.array([0.2, 0.2, 0.4]), floor_local=-0.43)
    fitted = fit.fit(_glb(trimesh.creation.box((0.4, 0.8, 0.4))), placement)
    assert fitted.vertices[:, 2].min() == pytest.approx(-0.43)


# ── the scene ────────────────────────────────────────────────────────────────────────────────────


def _two_tone_glb() -> bytes:
    """A fake TRELLIS box whose photo is red on top and blue below."""
    photo = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
    photo.paste((0, 0, 255, 255), (0, 16, 32, 32))
    buffer = io.BytesIO()
    photo.save(buffer, format="PNG")
    return fake_glb(buffer.getvalue())


@pytest.fixture()
def realistic(scan, tmp_path):
    out = tmp_path / "objects"
    scene = write_realistic_scene(scan, out, {"Chair0": _two_tone_glb(), "Table0": fake_glb()},
                                  views=False)
    return scene, mujoco.MjModel.from_xml_path(str(out / build.BASE_NAME)), \
        mujoco.MjModel.from_xml_path(str(scene))


def test_the_look_changes_and_the_physics_does_not(realistic):
    scene, quick, real = realistic
    names = lambda m: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)  # noqa: E731
                       for b in range(m.nbody)]
    assert names(real) == names(quick)
    assert np.allclose(real.body_mass, quick.body_mass)
    assert np.allclose(real.body_inertia, quick.body_inertia)
    assert real.nq == quick.nq and real.njnt == quick.njnt
    assert movable_body_ids(real) == movable_body_ids(quick)
    for handle in ("Chair0", "Table0"):
        b = _body(real, handle)
        groups = {int(real.geom_group[g]) for g in _geoms_of(real, b)}
        assert groups == {VISUAL_GROUP, COLLISION_GROUP}
        visual = [g for g in _geoms_of(real, b) if real.geom_group[g] == VISUAL_GROUP]
        assert len(visual) == 1
        g = visual[0]
        assert real.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        assert real.geom_contype[g] == 0 and real.geom_conaffinity[g] == 0
        # the colliders are the quick room's parts, unchanged
        assert sum(1 for g in _geoms_of(real, b) if real.geom_group[g] == COLLISION_GROUP) == \
            len(_geoms_of(quick, _body(quick, handle)))
    sofa = _body(real, "Sofa0")
    assert {int(real.geom_group[g]) for g in _geoms_of(real, sofa)} == {0}    # still a template
    report = json.loads((scene.parent / build.REPORT_NAME).read_text())
    assert set(report["objects"]) == {"Chair0", "Table0"}
    assert "Sofa0" in report["template"]


def test_the_visual_mesh_fills_its_body_box(realistic):
    _scene, quick, real = realistic
    data = mujoco.MjData(real)
    mujoco.mj_forward(real, data)
    b = _body(real, "Chair0")
    g = next(g for g in _geoms_of(real, b) if real.geom_group[g] == VISUAL_GROUP)
    mid = int(real.geom_dataid[g])
    verts = real.mesh_vert[real.mesh_vertadr[mid]:real.mesh_vertadr[mid] + real.mesh_vertnum[mid]]
    world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    local = (world - data.xpos[b]) @ data.xmat[b].reshape(3, 3)
    colliders = [c for c in _geoms_of(real, b) if real.geom_group[c] == COLLISION_GROUP]
    lo = np.min([real.geom_pos[c] - real.geom_size[c] for c in colliders], axis=0)
    hi = np.max([real.geom_pos[c] + real.geom_size[c] for c in colliders], axis=0)
    assert np.allclose(local.min(axis=0), lo, atol=2e-3)
    assert np.allclose(local.max(axis=0), hi, atol=2e-3)


def test_texture_rows_reach_mujoco_the_right_way_up(realistic):
    """The photo's top row must land on the top of the mesh: trimesh holds OBJ-convention UVs,
    the OBJ carries them as they are, and MuJoCo's loader flips them to count from the top row."""
    _scene, _quick, real = realistic
    data = mujoco.MjData(real)
    mujoco.mj_forward(real, data)
    b = _body(real, "Chair0")
    g = next(g for g in _geoms_of(real, b) if real.geom_group[g] == VISUAL_GROUP)
    mid = int(real.geom_dataid[g])
    texid = int(real.mat_texid[real.geom_matid[g], int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
    assert texid >= 0
    h, w, n = (int(real.tex_height[texid]), int(real.tex_width[texid]),
               int(real.tex_nchannel[texid]))
    tex = real.tex_data[real.tex_adr[texid]:real.tex_adr[texid] + h * w * n].reshape(h, w, n)
    va, fa = int(real.mesh_vertadr[mid]), int(real.mesh_faceadr[mid])
    ta = int(real.mesh_texcoordadr[mid])
    verts = real.mesh_vert[va:va + real.mesh_vertnum[mid]]
    faces = real.mesh_face[fa:fa + real.mesh_facenum[mid]]
    ftc = real.mesh_facetexcoord[fa:fa + real.mesh_facenum[mid]]
    uv = real.mesh_texcoord[ta:ta + real.mesh_texcoordnum[mid]]
    body_frame = verts @ np.asarray(
        _quat_mat(real.geom_quat[g])).T + real.geom_pos[g]
    top = body_frame[:, 2].max()
    front = body_frame[:, 1].min()
    checked = 0
    for face, tcs in zip(faces, ftc):
        corners = body_frame[face]
        if not np.allclose(corners[:, 1], front, atol=1e-4):
            continue                              # only the front face: its photo is upright
        for corner, tc in zip(corners, tcs):
            row = min(h - 1, int(uv[tc][1] * h))
            colour = tex[row, min(w - 1, int(uv[tc][0] * w))]
            if corner[2] > top - 1e-4:
                assert colour[0] > 200 and colour[2] < 60, colour      # red at the top
            elif corner[2] < body_frame[:, 2].min() + 1e-4:
                assert colour[2] > 200 and colour[0] < 60, colour      # blue at the bottom
            checked += 1
    assert checked >= 4


def _quat_mat(q) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=float))
    return out.reshape(3, 3)


def test_the_room_call_can_draw_it(realistic):
    from litereality_agent.room_ops.roomcall.geometry import build_scene_assets

    _scene, quick, real = realistic
    glb, manifest = build_scene_assets(real, label="test")
    _glb_q, manifest_q = build_scene_assets(quick, label="test")
    assert glb[:4] == b"glTF"
    assert [b["name"] for b in manifest["bodies"]] == [b["name"] for b in manifest_q["bodies"]]
    data = mujoco.MjData(real)
    for _ in range(300):
        mujoco.mj_step(real, data)
    assert np.all(np.isfinite(data.qpos))


def test_versions_are_written_whole_and_never_overwrite_a_served_file(scan, tmp_path):
    room = RealRoom(scan, tmp_path / "objects")
    room.add("Chair0", fake_glb(pass_name="preview"), pass_name="preview")
    first = room.write()
    preview_mesh = room.installed["Chair0"].files.mesh
    room.add("Chair0", fake_glb(), pass_name="final")
    room.write()
    assert room.installed["Chair0"].files.mesh != preview_mesh
    assert (room.mesh_dir / preview_mesh).is_file()          # the last version may still be served
    room.add("Chair0", fake_glb(pass_name="preview"), pass_name="preview")
    assert room.installed["Chair0"].pass_name == "final"     # a late preview never wins
    for name in ("Sofa0", "Table1", "Dishwasher0"):
        room.add(name, fake_glb())
        room.write()
    assert not (room.mesh_dir / preview_mesh).exists()       # ...but not forever
    assert not list(first.parent.glob("*.tmp"))
    mujoco.MjModel.from_xml_path(str(first))


def test_a_base_with_its_own_asset_folders_is_respected(scan, tmp_path):
    out = tmp_path / "objects"
    room = RealRoom(scan, out)                               # builds the quick room
    root = ET.parse(room.base_path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", "assets/m")
    compiler.set("texturedir", "assets/t")
    for texture in root.iter("texture"):                     # keep its own textures resolving
        src = out / texture.get("file")
        dst = out / "assets" / "t" / texture.get("file")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    ET.ElementTree(root).write(room.base_path)
    room = RealRoom(scan, out)                               # reused: built from this scan
    room.add("Chair0", fake_glb())
    scene = room.write()
    assert list((out / "assets" / "m" / build.ASSET_DIR).glob("Chair0-*.obj"))
    assert list((out / "assets" / "t" / build.ASSET_DIR).glob("Chair0-*.png"))
    mujoco.MjModel.from_xml_path(str(scene))


# ── crops ────────────────────────────────────────────────────────────────────────────────────────


def test_crops_cut_out_what_the_capture_saw(seen_scan, tmp_path):
    room = RealRoom(seen_scan, tmp_path / "objects")
    got = room.make_crops()
    assert set(got) == set(room.generated_handles())
    table = got["Table0"]
    assert table.ok and table.image.mode == "RGBA"
    assert max(table.image.size) >= crops.MIN_SIDE
    arr = np.asarray(table.image)
    opaque = arr[..., 3] > 127
    assert opaque.mean() > 0.3
    red = arr[..., 0][opaque].mean(), arr[..., 1][opaque].mean()
    assert red[0] > 150 and red[1] < 90                      # the table's red top
    assert table.views and table.views[0].mask.any()
    unseen = got["Sofa0"]
    assert not unseen.ok and unseen.note
    saved = json.loads((tmp_path / "objects" / build.CROP_DIR / "crops.json").read_text())
    assert saved["Table0"]["frame"] == 0


def test_crops_degrade_without_a_capture(scan, tmp_path):
    room = RealRoom(scan, tmp_path / "objects")
    got = room.make_crops()
    assert got and all(not c.ok and "no frames" in c.note for c in got.values())


def test_a_sideways_phone_still_gives_an_upright_crop():
    upright = np.eye(4)
    upright[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]     # looking along +Y, room up = image up
    assert crops.upright_turns(upright) == 0
    rolled = upright.copy()
    rolled[:3, 0], rolled[:3, 1] = upright[:3, 1].copy(), -upright[:3, 0]
    assert crops.upright_turns(rolled) in (1, 3)
    flipped = upright.copy()
    flipped[:3, 0], flipped[:3, 1] = -upright[:3, 0], -upright[:3, 1]
    assert crops.upright_turns(flipped) == 2


# ── generating, version by version ───────────────────────────────────────────────────────────────


def test_versions_stream_as_objects_arrive_and_the_cache_is_reused(seen_scan, tmp_path):
    room = RealRoom(seen_scan, tmp_path / "objects")
    published = []
    gen = FakeGenerator()
    progress = generate_room(room, gen, min_interval=0.0, log=lambda _m: None,
                             on_version=lambda path, info: published.append(info["installed"]))
    assert progress.installed == {"Table0": "final"}
    assert gen.calls == [("Table0", "preview"), ("Table0", "final")]
    assert [p.get("Table0") for p in published] == ["preview", "final"]
    # Again, same photo: nothing is generated, the cached final is installed and published.
    room = RealRoom(seen_scan, tmp_path / "objects")
    again = FakeGenerator()
    progress = generate_room(room, again, log=lambda _m: None)
    assert again.calls == [] and progress.cached == ["Table0:final"]
    assert progress.versions == 1


def test_a_failed_object_keeps_its_template(seen_scan, tmp_path):
    room = RealRoom(seen_scan, tmp_path / "objects")
    progress = generate_room(room, FakeGenerator(fail=["Table0"]), log=lambda _m: None)
    assert progress.installed == {}
    assert set(progress.failed) == {"Table0:preview", "Table0:final"}
    report = json.loads((room.out_dir / build.REPORT_NAME).read_text())
    assert "Table0" in report["template"] and "Table0" in report["errors"]


def test_the_cost_estimate_grows_with_the_room():
    small = generate.estimate(2, ["preview", "final"])
    big = generate.estimate(20, ["preview", "final"])
    assert 0 < small[0] < big[0] and 0 < small[1] < big[1]
    assert generate.estimate(0, ["final"]) == (0.0, 0.0)


# ── TRELLIS on Modal, faked ──────────────────────────────────────────────────────────────────────


class _FakeFunction:
    """Stands in for `modal.Function`: `remote.aio` answers each request with a GLB."""

    def __init__(self, *, fail: str | None = None, wrong_id: str | None = None,
                 delay: float = 0.01) -> None:
        self.requests: list[dict] = []
        self.fail, self.wrong_id, self.delay = fail, wrong_id, delay

        async def aio(request: dict) -> dict:
            self.requests.append(request)
            await asyncio.sleep(self.delay)
            handle = request["id"].split(":")[0]
            if handle == self.fail:
                return {"error": "CUDA out of memory", "id": request["id"]}
            rid = "someone-else:final" if handle == self.wrong_id else request["id"]
            return {"glb_b64": base64.b64encode(fake_glb()).decode(), "id": rid}

        self.remote = SimpleNamespace(aio=aio)


def _png(alpha: bool = True) -> bytes:
    image = Image.new("RGBA", (16, 16), (90, 60, 30, 255))
    if alpha:
        image.putpixel((0, 0), (0, 0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _modal_generator(function: _FakeFunction) -> ModalTrellisGenerator:
    service = trellis_modal.ModalTrellisService(app_name="unused",
                                                client=SimpleNamespace(function=function))
    return ModalTrellisGenerator(service=service, max_in_flight=2)


def test_modal_generator_asks_for_each_pass_as_its_own_request():
    function = _FakeFunction()
    gen = _modal_generator(function)
    got = list(gen.stream({"Chair0": _png(), "Table0": _png()}, ["preview", "final"]))
    assert sorted((r.handle, r.pass_name) for r in got) == [
        ("Chair0", "final"), ("Chair0", "preview"), ("Table0", "final"), ("Table0", "preview")]
    assert all(r.ok and r.glb[:4] == b"glTF" for r in got)
    by_id = {r["id"]: r for r in function.requests}
    assert by_id["Chair0:preview"]["pipeline_type"] == "512"
    assert by_id["Chair0:preview"]["texture_size"] == 512
    assert by_id["Chair0:final"]["pipeline_type"] == "1024_cascade"
    assert by_id["Chair0:final"]["texture_size"] == generate.PASSES["final"].texture_size
    sent = Image.open(io.BytesIO(base64.b64decode(by_id["Table0:final"]["image_b64"])))
    assert sent.mode == "RGBA" and np.asarray(sent)[..., 3].min() < 250   # its own mask
    # both previews went out before either final
    order = [r["id"] for r in function.requests]
    assert max(order.index("Chair0:preview"), order.index("Table0:preview")) < \
        min(order.index("Chair0:final"), order.index("Table0:final"))


def test_modal_generator_reports_one_objects_failure_and_carries_on():
    function = _FakeFunction(fail="Chair0", wrong_id="Sofa0")
    gen = _modal_generator(function)
    got = {(r.handle, r.pass_name): r
           for r in gen.stream({"Chair0": _png(), "Table0": _png(), "Sofa0": _png()},
                               ["preview", "final"])}
    assert not got["Chair0", "preview"].ok and "out of memory" in got["Chair0", "preview"].error
    assert not got["Chair0", "final"].ok                      # its final was still asked for
    assert got["Table0", "final"].ok
    assert not got["Sofa0", "final"].ok and "expected" in got["Sofa0", "final"].error


def test_a_real_room_with_the_modal_generator(seen_scan, tmp_path):
    room = RealRoom(seen_scan, tmp_path / "objects")
    progress = generate_room(room, _modal_generator(_FakeFunction()), log=lambda _m: None)
    assert progress.installed == {"Table0": "final"}
    mujoco.MjModel.from_xml_path(str(room.scene_path))


# ── the command line ─────────────────────────────────────────────────────────────────────────────


def _main(argv: list[str]):
    args = cli._parser().parse_args(argv)
    return args, args.handler(args)


def test_cli_realroom_builds_offline_with_the_stand_in(seen_scan, tmp_path, capsys):
    out = tmp_path / "o"
    _args, rc = _main(["realroom", str(seen_scan), "--fake", "--out", str(out)])
    assert rc == 0
    report = json.loads((out / build.REPORT_NAME).read_text())
    assert report["objects"]["Table0"]["pass"] == "final"
    text = capsys.readouterr().out
    assert "1 final" in text and "no GPU, no cost" in text
    mujoco.MjModel.from_xml_path(str(out / "scene.xml"))


def test_cli_realistic_roomcall_publishes_the_quick_room_then_each_version(seen_scan, tmp_path,
                                                                          monkeypatch):
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "run"))
    monkeypatch.setattr(build, "MIN_INTERVAL", 0.0)
    calls = []
    monkeypatch.setattr(cli.roomcall, "run",
                        lambda args, **kw: calls.append((args.scene, args.deploy, args.stage,
                                                         kw)) or 0)
    _args, rc = _main(["roomcall", str(seen_scan), "--realistic", "--fake", "--fake-delay", "0",
                       "--publish"])
    assert rc == 0
    scene = str(tmp_path / "run" / "test_room" / "objects" / "scene.xml")
    assert calls[0][:3] == (scene, "publish", "quick")
    assert [c[2] for c in calls[1:]] == ["objects", "objects"]         # preview, then final
    assert all(c[0] == scene and c[1] == "publish" for c in calls)
    assert set(calls[0][3]) == {"credentials", "profile", "environment"}


def test_cli_realistic_roomcall_serves_and_watches_here(seen_scan, tmp_path, monkeypatch):
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "run"))
    served = []
    monkeypatch.setattr(cli.roomcall, "run", lambda args, **kw: served.append(args) or 0)
    args, rc = _main(["roomcall", str(seen_scan), "--realistic", "--fake", "--fake-delay", "0",
                      "--no-open"])
    assert rc == 0
    args._realroom_worker.join(timeout=60)
    served = served[0]
    assert served.watch is True and served.stage == "objects"
    assert served.scene.endswith(str(Path("test_room") / "objects" / "scene.xml"))
    report = json.loads((Path(served.scene).parent / build.REPORT_NAME).read_text())
    assert report["objects"]["Table0"]["pass"] == "final"


def test_cli_realistic_needs_a_scan_and_modal(tmp_path, scan, monkeypatch):
    with pytest.raises(SystemExit, match="needs a scan"):
        _main(["roomcall", "--demo", "--realistic", "--fake"])
    from litereality_agent.settings import LiteRealitySettings

    monkeypatch.setattr(cli, "load_settings", lambda: LiteRealitySettings(_env_file=None))
    monkeypatch.delenv("MODAL_PROFILE", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    with pytest.raises(SystemExit, match="TRELLIS on Modal"):
        _main(["realroom", str(scan), "--out", str(tmp_path / "o")])


def test_math_of_placement_round_trips():
    placement = fit.Placement(half=np.ones(3), centre=np.array([0.1, 0.0, 0.0]),
                              world_centre=np.array([1.0, 2.0, 0.5]), world_yaw=math.pi / 2)
    assert np.allclose(placement.to_world(np.array([[1.1, 0.0, 0.0]])), [[1.0, 3.0, 0.5]])


def test_a_mesh_mujoco_refuses_is_that_objects_failure_not_the_rooms(scan, tmp_path, monkeypatch):
    room = RealRoom(scan, tmp_path / "objects")
    room.add("Table0", fake_glb())
    import mujoco as mj

    real = mj.MjModel.from_xml_string

    def picky(xml, *a, **kw):
        if "Chair0-" in xml:
            raise ValueError("mesh is degenerate")
        return real(xml, *a, **kw)

    monkeypatch.setattr(mj.MjModel, "from_xml_string", picky)
    with pytest.raises(ValueError, match="refuses Chair0"):
        room.add("Chair0", fake_glb())
    monkeypatch.undo()
    assert "Chair0" not in room.installed
    assert not list(room.mesh_dir.glob("Chair0-*"))
    mj.MjModel.from_xml_path(str(room.write()))


def test_a_partly_seen_object_borrows_its_twins_mesh(scan, tmp_path):
    room = RealRoom(scan, tmp_path / "objects")
    photo = Image.new("RGBA", (64, 64), (40, 40, 40, 255))
    room.crops = {h: crops.ObjectCrop(h, room.boxes[h].category, None, note="unseen")
                  for h in room.generated_handles()}
    room.crops["Chair0"] = crops.ObjectCrop("Chair0", "chair", photo, score=0.9)
    room.crops["Chair1"] = crops.ObjectCrop("Chair1", "chair", photo, score=0.2)   # a sliver
    room.crops["Sofa0"] = crops.ObjectCrop("Sofa0", "sofa", photo, score=0.1)      # no twin
    gen = FakeGenerator()
    progress = generate_room(room, gen, passes=["final"], log=lambda _m: None)
    assert gen.calls == [("Chair0", "final")]                 # one generation for both chairs
    assert progress.borrowed == {"Chair1": "Chair0"}
    assert progress.installed == {"Chair0": "final", "Chair1": "final"}
    assert "Sofa0" in progress.skipped and "too little" in progress.skipped["Sofa0"]
    report = json.loads((room.out_dir / build.REPORT_NAME).read_text())
    assert report["borrowed"] == {"Chair1": "Chair0"} and "Sofa0" in report["template"]
    assert build.twin_of(room, "Sofa0", ["Chair0"]) is None   # another category is no twin


def test_cli_no_textures_reaches_the_quick_room(scan, tmp_path, monkeypatch):
    from litereality_agent.pipeline import quickroom

    seen = []
    real = quickroom.write_quick_scene

    def spy(*args, **kwargs):
        seen.append(kwargs.get("textures", True))
        return real(*args, **kwargs)

    monkeypatch.setattr(quickroom, "write_quick_scene", spy)
    monkeypatch.setattr(build, "write_quick_scene", spy)
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "run"))
    monkeypatch.setattr(cli.roomcall, "run", lambda args, **kw: 0)
    _main(["quickroom", str(scan), "--no-textures", "--out", str(tmp_path / "q")])
    _main(["roomcall", str(scan), "--no-open", "--no-textures"])
    _main(["realroom", str(scan), "--fake", "--no-textures", "--out", str(tmp_path / "r")])
    _main(["quickroom", str(scan), "--out", str(tmp_path / "q2")])
    assert seen == [False, False, False, True]
