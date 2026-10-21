"""The quick room's photo walls and floor: the capture's frames projected onto RoomPlan's surfaces.

Frames here are synthetic and exact: a pinhole camera in the fixture room (`test_quickroom`'s
5 x 4 m RoomPlan export) looking straight at Wall0 from 2 m, its image painted from the WORLD
position each pixel sees — red where x < 2.5, blue beyond, a green band high up — so any slip in
the projection, the image layout or the uv convention shows up as the wrong colour at a known
spot on the wall. The same spot is then looked up through MuJoCo's compiled mesh and texture
arrays and through the room call's glb, which is the whole path to the browser.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image
from test_quickroom import FLOOR_Y, write_roomplan_usdz

from litereality_agent.pipeline.quickroom import build, roomplan, textures, write_quick_scene
from litereality_agent.room_ops.roomcall.model_info import VISUAL_GROUP, visual_geom_ids
from litereality_agent.room_ops.roomcall.sim import RoomSim, load_model

RED, BLUE, GREEN = (200, 40, 40), (40, 40, 200), (40, 190, 40)
CAM_Y, CAM_H = 2.0, 1.25          # m from Wall0, m above the floor
GREEN_ABOVE = 1.6                 # m above the floor: the green band


def _pattern(x: np.ndarray, z_above_floor: np.ndarray) -> np.ndarray:
    rgb = np.where((x < 2.5)[..., None], np.array(RED), np.array(BLUE))
    return np.where((z_above_floor > GREEN_ABOVE)[..., None], np.array(GREEN), rgb)


def write_wall_frame(scan: Path, index: int, cam_x: float, *, gain: float = 1.0,
                     grey: int | None = None) -> None:
    """A frame looking along -Y at Wall0 (the plane y = 0) from (cam_x, CAM_Y), image up = +Z:
    RGB painted from the world point each pixel sees (or a flat ``grey``, times ``gain``), depth
    the distance to the wall (z-depth: the same for every pixel), confidence high."""
    W, H, w, h = 320, 240, 64, 48
    fx = fy = 200.0
    cx, cy = W / 2, H / 2
    cam_z = FLOOR_Y + CAM_H
    # camera -> ARKit: camera +X = world -X, camera +Y (image up) = Z-up +Z = ARKit +Y, and it
    # looks down its -Z = world -Y, i.e. camera +Z = Z-up +Y = ARKit -Z.
    pose = np.eye(4)
    pose[:3, 0] = (-1, 0, 0)
    pose[:3, 1] = (0, 1, 0)
    pose[:3, 2] = (0, 0, -1)
    pose[:3, 3] = (cam_x, cam_z, -CAM_Y)
    us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x = cam_x - CAM_Y * (us - cx) / fx
    z = cam_z - CAM_Y * (vs - cy) / fy
    rgb = (np.full((H, W, 3), grey, float) if grey is not None
           else _pattern(x, z - FLOOR_Y).astype(float))
    Image.fromarray(np.clip(rgb * gain, 0, 255).astype(np.uint8)).save(
        scan / f"frame_{index:05d}.jpg", quality=95)
    Image.fromarray(np.full((h, w), int(CAM_Y * 1000), np.uint16)).save(
        scan / f"depth_{index:05d}.png")
    Image.fromarray(np.full((h, w), 2, np.uint8)).save(scan / f"conf_{index:05d}.png")
    (scan / f"frame_{index:05d}.json").write_text(json.dumps({
        "frame_index": index, "intrinsics": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
        "cameraPoseARFrame": pose.reshape(-1).tolist()}))


@pytest.fixture()
def scan(tmp_path) -> Path:
    folder = tmp_path / "scans" / "test_room"
    write_roomplan_usdz(folder / "room.usdz")
    return folder


def _texel(surface: textures.Surface, point) -> np.ndarray:
    u, v = surface.uv(np.asarray(point, float))[0]
    rows, cols = surface.shape
    return surface.image[min(rows - 1, int(v * rows)), min(cols - 1, int(u * cols))].astype(int)


def _close(rgb, want, tol: int = 30) -> bool:
    return bool(np.all(np.abs(np.asarray(rgb, int) - np.asarray(want)) <= tol))


# ── the estimate ─────────────────────────────────────────────────────────────────────────────────


def test_a_wall_is_painted_with_what_the_camera_saw_where_it_saw_it(scan):
    write_wall_frame(scan, 0, 2.5)
    plan = roomplan.parse_room(scan)
    result = textures.estimate(scan, plan)
    assert result.source == "frames" and result.frames_used == 1
    wall = result.surfaces["Wall0"]
    assert wall.kind == "wall" and wall.image.shape == (*wall.shape, 3)
    assert wall.coverage > 0.4
    fz = plan.floor_z
    assert _close(_texel(wall, (2.0, 0.0, fz + 1.0)), RED)
    assert _close(_texel(wall, (3.0, 0.0, fz + 1.0)), BLUE)
    assert _close(_texel(wall, (3.0, 0.0, fz + 2.0)), GREEN)
    # what no frame saw has no photo, and says so
    assert "floor" not in result.surfaces and result.rejected["floor"] == 0.0
    assert "Wall2" not in result.surfaces
    summary = result.summary()
    assert summary["surfaces"]["Wall0"]["coverage"] == pytest.approx(wall.coverage, abs=1e-3)


def test_the_image_layout_is_along_the_wall_and_down_from_the_ceiling(scan):
    plan = roomplan.parse_room(scan)
    surfaces = {s.name: s for s in textures.layout_surfaces(plan)}
    wall = surfaces["Wall0"]                        # (0, 0) -> (5, 0), the room on +Y
    assert np.allclose(wall.normal, (0, 1, 0))
    top_left = (0.0, 0.0, plan.ceiling_z)
    bottom_right = (5.0, 0.0, plan.floor_z)
    assert np.allclose(wall.uv(np.array([top_left, bottom_right])), [[0, 0], [1, 1]])
    floor = surfaces["floor"]                       # the plan view, +Y up the image
    assert np.allclose(floor.uv(np.array([(0, 4, plan.floor_z), (5, 0, plan.floor_z)])),
                       [[0, 0], [1, 1]])
    # an opening is not drawn; furniture against a wall hides the strip behind it
    door = plan.openings["Wall0_Door_0"]
    rows, cols = wall.shape
    t_mid, z_mid = door.offset, door.height / 2
    assert not wall.drawn[int((1 - z_mid / plan.height) * rows), int(t_mid / 5 * cols)]
    wall3 = surfaces["Wall3"]                       # the counter run stands against it
    assert wall3.hidden.any() and wall3.drawn.all()


def test_exposure_is_balanced_so_two_frames_leave_no_seam(scan):
    # The same plain grey wall through two frames, one exposed 30 % darker. Without balancing,
    # the half only the dark frame saw would be 45 levels darker than the other.
    write_wall_frame(scan, 0, 1.9, grey=150)
    write_wall_frame(scan, 1, 3.3, grey=150, gain=0.7)
    plan = roomplan.parse_room(scan)
    result = textures.estimate(scan, plan)
    wall = result.surfaces["Wall0"]
    z = plan.floor_z + 1.2
    only_bright, shared, only_dark = (_texel(wall, (x, 0.0, z)) for x in (0.4, 2.6, 4.6))
    assert abs(int(only_bright.mean()) - int(only_dark.mean())) <= 8
    assert abs(int(shared.mean()) - int(only_dark.mean())) <= 8
    g = result.gains
    assert g[1][0] / g[0][0] == pytest.approx(1 / 0.7, rel=0.06)


def test_unseen_texels_are_filled_never_black():
    rgb = np.zeros((32, 48, 3), np.float32)
    known = np.zeros((32, 48), bool)
    rgb[:, :10] = (100, 120, 140)
    known[:, :10] = True
    out = textures.push_pull(rgb, known)
    assert np.allclose(out[:, :10], rgb[:, :10])
    assert out[:, 10:].min() > 90 and np.allclose(out[:, -1], (100, 120, 140), atol=2)


def test_without_frames_or_time_there_are_no_photos_and_no_error(scan):
    plan = roomplan.parse_room(scan)
    empty = textures.estimate(scan, plan)
    assert empty.source == "none" and not empty.surfaces and "no frames" in empty.note
    write_wall_frame(scan, 0, 2.5)
    late = textures.estimate(scan, plan, budget_s=0.0)
    assert not late.surfaces and "time budget" in late.note
    (scan / "frame_00001.json").write_text("{broken")
    (scan / "frame_00001.jpg").write_bytes(b"not a jpeg")
    (scan / "depth_00001.png").write_bytes(b"")
    some = textures.estimate(scan, plan)                       # one good frame, one bad
    assert "Wall0" in some.surfaces and "unreadable" in some.note


# ── the meshes they go on ────────────────────────────────────────────────────────────────────────


def _area(poly, tris) -> float:
    p = np.asarray(poly, float)
    total = 0.0
    for a, b, c in tris:
        cross = (p[b] - p[a])[0] * (p[c] - p[a])[1] - (p[b] - p[a])[1] * (p[c] - p[a])[0]
        assert cross > 0                                       # counter-clockwise, not flat
        total += cross / 2
    return total


def test_floor_outlines_triangulate_even_when_not_convex():
    ell = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (0, 3)]
    assert _area(ell, build.triangulate(ell)) == pytest.approx(6.0)
    assert _area(ell[::-1], build.triangulate(ell[::-1])) == pytest.approx(6.0)
    with_flat_corner = [(0, 0), (2, 0), (4, 0), (4, 2), (0, 2)]
    assert _area(with_flat_corner, build.triangulate(with_flat_corner)) == pytest.approx(8.0)
    assert build.triangulate([(0, 0), (1, 0)]) == []


def test_a_wall_photo_is_one_welded_grid_with_its_openings_cut_out():
    cuts = [(0.55, 1.45, 0.0, 2.05), (1.9, 3.1, 1.0, 2.0)]
    pts, quads = build._wall_grid(5.0, 2.5, cuts)
    assert len(np.unique(pts.round(6), axis=0)) == len(pts)          # welded: no duplicates
    area = sum((pts[q[1], 0] - pts[q[0], 0]) * (pts[q[2], 1] - pts[q[1], 1]) for q in quads)
    assert area == pytest.approx(12.5 - 0.9 * 2.05 - 1.2 * 1.0)
    for q in quads:                                                   # nothing in an opening
        t, z = pts[q].mean(axis=0)
        assert not any(c[0] < t < c[1] and c[2] < z < c[3] for c in cuts)
        # no T-junction: no other corner lies inside any of its edges
        for a, b in zip(q, np.roll(q, -1)):
            lo, hi = np.minimum(pts[a], pts[b]), np.maximum(pts[a], pts[b])
            inside = np.all((pts >= lo - 1e-9) & (pts <= hi + 1e-9), axis=1)
            inside[[a, b]] = False
            assert not inside.any()


# ── the room ─────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def textured(scan, tmp_path):
    write_wall_frame(scan, 0, 2.5)
    path = write_quick_scene(scan, tmp_path / "out")
    report = json.loads((tmp_path / "out" / build.REPORT_NAME).read_text())
    return path, report, load_model(path)


def _geom(model, name: str) -> int:
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert g >= 0, name
    return g


def _affine_uv(points: np.ndarray, uv: np.ndarray):
    """uv as an affine function of position, fitted to a planar mesh's corners."""
    A = np.c_[points, np.ones(len(points))]
    coef, *_ = np.linalg.lstsq(A, uv, rcond=None)
    return lambda p: np.r_[np.asarray(p, float), 1.0] @ coef


def test_the_room_wears_its_photos_on_drawn_only_meshes(textured):
    path, report, model = textured
    assert (path.parent / build.TEXTURE_DIR / "Wall0.png").is_file()
    assert report["textures"]["source"] == "frames" and "Wall0" in report["textures"]["surfaces"]
    assert "textures" in report["seconds"]
    g = _geom(model, "Wall0_photo")
    assert int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)
    assert model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
    assert model.geom_group[g] == VISUAL_GROUP and g in visual_geom_ids(model)
    # the boxes are still the walls: same colliders, drawn too (jambs, the outside)
    box = _geom(model, "Wall0_0")
    assert model.geom_contype[box] == 1 and model.geom_group[box] == 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "Wall2_photo") < 0   # never seen
    # a published version carries them: the room call sends what the scene names
    from litereality_agent.room_ops.roomcall.hosted import scene_files

    sent = {str(relative) for _local, relative in scene_files(path)}
    assert {"scene.xml", f"{build.TEXTURE_DIR}/Wall0.png"} <= sent


def test_the_photo_lands_where_it_was_taken_through_mujoco(textured, scan):
    """World point -> the compiled mesh's own texcoords -> MuJoCo's texture rows: the V
    convention end to end (an XML texcoord's v counts from the PNG's top row)."""
    _path, _report, model = textured
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    g = _geom(model, "Wall0_photo")
    mesh = int(model.geom_dataid[g])
    va, fa, fn = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    ta = model.mesh_texcoordadr[mesh]
    faces = model.mesh_face[fa:fa + fn].reshape(-1)
    ftc = model.mesh_facetexcoord[fa:fa + fn].reshape(-1)
    local = model.mesh_vert[va + faces]
    world = local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    uv_of = _affine_uv(world, model.mesh_texcoord[ta + ftc])
    mat = int(model.geom_matid[g])
    tex = int(np.asarray(model.mat_texid[mat]).reshape(-1)[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
    w, h, c = int(model.tex_width[tex]), int(model.tex_height[tex]), int(model.tex_nchannel[tex])
    image = model.tex_data[model.tex_adr[tex]:model.tex_adr[tex] + w * h * c].reshape(h, w, c)
    fz = roomplan.parse_room(scan).floor_z
    inset = build.PHOTO_INSET

    def colour(x: float, z: float) -> np.ndarray:
        u, v = uv_of((x, inset, fz + z))
        return image[int(v * h), int(u * w), :3]

    assert _close(colour(2.0, 1.0), RED)
    assert _close(colour(3.0, 1.0), BLUE)
    assert _close(colour(3.0, 2.0), GREEN)


def _glb_textured_meshes(glb: bytes):
    """(positions, uv, image) of every textured primitive in a glb, read straight from its
    buffers (glTF: v counts from the image's top row)."""
    json_len = struct.unpack_from("<I", glb, 12)[0]
    tree = json.loads(glb[20:20 + json_len])
    start = 20 + json_len
    blob = glb[start + 8:start + 8 + struct.unpack_from("<I", glb, start)[0]]

    def view(index: int) -> bytes:
        bv = tree["bufferViews"][index]
        return blob[bv.get("byteOffset", 0):bv.get("byteOffset", 0) + bv["byteLength"]]

    def accessor(index: int) -> np.ndarray:
        a = tree["accessors"][index]
        width = {"VEC2": 2, "VEC3": 3}[a["type"]]
        raw = view(a["bufferView"])[a.get("byteOffset", 0):]
        return np.frombuffer(raw, np.float32, a["count"] * width).reshape(-1, width)

    out = []
    for mesh in tree["meshes"]:
        for prim in mesh["primitives"]:
            pbr = tree["materials"][prim["material"]].get("pbrMetallicRoughness", {})
            if "baseColorTexture" not in pbr:
                continue
            source = tree["textures"][pbr["baseColorTexture"]["index"]]["source"]
            image = Image.open(io.BytesIO(view(tree["images"][source]["bufferView"])))
            out.append((accessor(prim["attributes"]["POSITION"]),
                        accessor(prim["attributes"]["TEXCOORD_0"]),
                        np.asarray(image.convert("RGB"))))
    return out


def test_the_photo_lands_where_it_was_taken_in_the_browsers_glb(textured, scan):
    from litereality_agent.room_ops.roomcall.geometry import build_scene_assets

    _path, _report, model = textured
    glb, manifest = build_scene_assets(model, label="test")
    fz = roomplan.parse_room(scan).floor_z
    assert manifest["floor_z"] == pytest.approx(fz, abs=1e-3)       # the slab, not the photo
    inset = build.PHOTO_INSET
    walls = [m for m in _glb_textured_meshes(glb) if np.allclose(m[0][:, 1], inset, atol=1e-3)]
    assert len(walls) == 1
    positions, uv, image = walls[0]
    uv_of = _affine_uv(positions.astype(float), uv.astype(float))
    h, w = image.shape[:2]

    def colour(x: float, z: float) -> np.ndarray:
        u, v = uv_of((x, inset, fz + z))
        return image[int(v * h), int(u * w)]

    assert _close(colour(2.0, 1.0), RED, tol=40)
    assert _close(colour(3.0, 1.0), BLUE, tol=40)
    assert _close(colour(3.0, 2.0), GREEN, tol=40)


def test_photos_change_nothing_about_the_physics(textured, scan, tmp_path):
    path, report, model = textured
    flat_path = write_quick_scene(scan, tmp_path / "flat", textures=False)
    flat = load_model(flat_path)
    assert "<texture" not in flat_path.read_text()
    assert not (flat_path.parent / build.TEXTURE_DIR).exists()
    assert model.nbody == flat.nbody and model.njnt == flat.njnt
    assert np.allclose(model.body_mass, flat.body_mass)
    contacts = []
    for m in (model, flat):
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        contacts.append(d.ncon)
    assert contacts[0] == contacts[1]
    assert report["initial_overlaps"] == [] and report["pinned"] == []
    sim = RoomSim(model)
    for _ in range(10):
        sim.step_realtime(0.1)
    assert np.all(np.isfinite(sim.data.qpos)) and sim.unstable_resets == 0


def test_textures_can_be_turned_off_and_old_photos_are_cleared(scan, tmp_path, monkeypatch):
    write_wall_frame(scan, 0, 2.5)
    out = tmp_path / "out"
    write_quick_scene(scan, out)
    assert (out / build.TEXTURE_DIR / "Wall0.png").is_file()
    (out / build.TEXTURE_DIR / "Wall9.png").write_bytes(b"stale")   # a wall a re-scan lost
    write_quick_scene(scan, out)
    assert not (out / build.TEXTURE_DIR / "Wall9.png").exists()
    monkeypatch.setenv("LR_QUICK_TEXTURES", "0")
    path = write_quick_scene(scan, out)
    assert "<texture" not in path.read_text() and not (out / build.TEXTURE_DIR).exists()
    report = json.loads((out / build.REPORT_NAME).read_text())
    assert report["textures"]["source"] == "none" and report["textures"]["note"] == "textures off"
    monkeypatch.delenv("LR_QUICK_TEXTURES")
    write_quick_scene(scan, out, colours=False)                    # no frames read at all
    assert not (out / build.TEXTURE_DIR).exists()


def test_a_scene_whose_photos_do_not_compile_goes_out_flat(scan, tmp_path, monkeypatch):
    write_wall_frame(scan, 0, 2.5)
    real_save = textures.save

    def corrupt(surface, folder):
        path = real_save(surface, folder)
        path.write_bytes(b"not a png")
        return path

    monkeypatch.setattr(textures, "save", corrupt)
    path = write_quick_scene(scan, tmp_path / "out")
    assert "<texture" not in path.read_text()
    load_model(path)
    report = json.loads((tmp_path / "out" / build.REPORT_NAME).read_text())
    assert "did not compile" in report["textures"]["note"] and not report["textures"]["surfaces"]


def test_a_textured_build_stays_fast(scan, tmp_path):
    import time

    for i, x in enumerate((1.5, 2.5, 3.5)):
        write_wall_frame(scan, i, x)
    started = time.perf_counter()
    write_quick_scene(scan, tmp_path / "out")
    assert time.perf_counter() - started < 6.0
