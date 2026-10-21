"""The room call's glb: the layout the client animates, the texture orientation, and the speed.

The texture test is the one that matters most. MuJoCo, OBJ, trimesh and glTF each pick their own
direction for v, and a wrong flip does not fail anything — it just paints the room upside down. So
it checks a KNOWN texel at a KNOWN vertex, read straight out of the glb the browser will get.
"""

from __future__ import annotations

import io
import json
import re
import struct
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from PIL import Image

from litereality_agent.room_ops.roomcall import geometry
from litereality_agent.room_ops.roomcall.geometry import build_scene_assets
from litereality_agent.room_ops.roomcall.model_info import movable_body_ids

# Image quadrants, in IMAGE coordinates (row 0 = top of the PNG).
TOP_LEFT, TOP_RIGHT = (220, 30, 30), (30, 200, 40)
BOTTOM_LEFT, BOTTOM_RIGHT = (30, 40, 220), (230, 220, 30)

# A unit quad facing +Z with a shallow apex under it (MuJoCo refuses a flat mesh). Each corner's vt
# sits in the middle of one quadrant; under OBJ's convention (v up) vertex (0,0,0) at vt
# (0.25, 0.25) is in the BOTTOM-left quadrant of the image.
QUAD_OBJ = """\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0.5 0.5 -0.3
vt 0.25 0.25
vt 0.75 0.25
vt 0.75 0.75
vt 0.25 0.75
vt 0.5 0.5
f 1/1 2/2 3/3
f 1/1 3/3 4/4
f 5/5 2/2 1/1
f 5/5 3/3 2/2
f 5/5 4/4 3/3
f 5/5 1/1 4/4
"""
EXPECTED_TEXEL = {(0, 0, 0): BOTTOM_LEFT, (1, 0, 0): BOTTOM_RIGHT,
                  (1, 1, 0): TOP_RIGHT, (0, 1, 0): TOP_LEFT}

SCENE = """\
<mujoco model="geometry_test">
  <compiler meshdir="." texturedir="."/>
  <asset>
    <texture name="quad_tex" type="2d" file="quad.png"/>
    <material name="quad_mat" texture="quad_tex"/>
    <material name="red" rgba="0.8 0.1 0.1 1"/>
    <mesh name="quad" file="quad.obj"/>
    <mesh name="tetra" vertex="0 0 0  0.2 0 0  0 0.2 0  0 0 0.2"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 1" rgba="0.6 0.6 0.6 1"/>
    <geom name="wall" type="box" pos="0 2 1" size="2 0.05 1" rgba="0.9 0.9 0.9 1"/>
    <geom name="post" type="cylinder" pos="1.5 1.5 0.5" size="0.05 0.5" rgba="0.3 0.3 0.3 1"/>
    <geom name="lamp" type="sphere" pos="-1.5 1.5 1" size="0.1" material="red"/>
    <geom name="deco" type="mesh" mesh="tetra" pos="-1.5 -1.5 0" rgba="0 1 0 1"/>
    <geom name="ghost" type="box" pos="0 -1.8 0.5" size="0.1 0.1 0.1" rgba="1 0 1 0"/>
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" pos="0 0 0.7" size="0.6 0.4 0.02" rgba="0.5 0.3 0.1 1"/>
    </body>
    <body name="crate" pos="1 -1 0.5">
      <freejoint/>
      <geom name="crate_vis" type="mesh" mesh="quad" material="quad_mat" group="2"
            contype="0" conaffinity="0" density="0"/>
      <geom name="crate_col" type="box" pos="0.5 0.5 -0.1" size="0.5 0.5 0.1" group="3"
            rgba="0 0 1 1"/>
      <body name="lid" pos="0 0 0.2">
        <joint name="lid_hinge" type="hinge" axis="1 0 0" range="0 1.5"/>
        <geom name="lid_bar" type="capsule" fromto="0 0 0 1 0 0" size="0.03"/>
        <geom name="lid_knob" type="ellipsoid" pos="0.5 0 0.05" size="0.05 0.03 0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _quadrant_png(path: Path, side: int = 64) -> None:
    img = np.zeros((side, side, 3), np.uint8)
    h = side // 2
    img[:h, :h], img[:h, h:] = TOP_LEFT, TOP_RIGHT
    img[h:, :h], img[h:, h:] = BOTTOM_LEFT, BOTTOM_RIGHT
    Image.fromarray(img).save(path)


@pytest.fixture
def model(tmp_path: Path) -> mujoco.MjModel:
    _quadrant_png(tmp_path / "quad.png")
    (tmp_path / "quad.obj").write_text(QUAD_OBJ)
    (tmp_path / "scene.xml").write_text(SCENE)
    return mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))


# ── raw glb reading: what three.js's GLTFLoader sees, with no library's conventions in between ──


def _parse(glb: bytes) -> tuple[dict, bytes]:
    assert glb[:4] == b"glTF"
    json_len = struct.unpack_from("<I", glb, 12)[0]
    tree = json.loads(glb[20:20 + json_len])
    if len(glb) <= 20 + json_len:
        return tree, b""                       # nothing to draw: no BIN chunk at all
    bin_len = struct.unpack_from("<I", glb, 20 + json_len)[0]
    return tree, glb[28 + json_len:28 + json_len + bin_len]


def _accessor(tree: dict, binary: bytes, index: int) -> np.ndarray:
    acc = tree["accessors"][index]
    view = tree["bufferViews"][acc["bufferView"]]
    assert "byteStride" not in view
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[
        acc["componentType"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    arr = np.frombuffer(binary, dtype=dtype, count=acc["count"] * width, offset=offset)
    return arr.reshape(acc["count"], width) if width > 1 else arr


def _image(tree: dict, binary: bytes, texture: int) -> Image.Image:
    view = tree["bufferViews"][tree["images"][tree["textures"][texture]["source"]]["bufferView"]]
    start = view.get("byteOffset", 0)
    return Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert("RGB")


def _node(tree: dict, name: str) -> dict:
    hits = [n for n in tree["nodes"] if n.get("name") == name]
    assert len(hits) == 1, name
    return hits[0]


def _child_meshes(tree: dict, node: dict) -> list[dict]:
    return [tree["meshes"][tree["nodes"][c]["mesh"]] for c in node.get("children", [])
            if "mesh" in tree["nodes"][c]]


# ── tests ────────────────────────────────────────────────────────────────────────────────────────


def test_node_layout_names_only_movable_bodies(model):
    glb, manifest = build_scene_assets(model, label="test room")
    tree, _ = _parse(glb)
    movable = movable_body_ids(model)
    crate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "crate")
    lid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lid")
    assert movable == {crate, lid}

    top = {tree["nodes"][i]["name"] for i in tree["scenes"][tree.get("scene", 0)]["nodes"]}
    assert top == {"static"} | {f"body_{b}" for b in movable}
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", n["name"]) for n in tree["nodes"])
    assert manifest["bodies"] == [{"id": b, "name": n, "node": f"body_{b}"}
                                  for b, n in sorted({crate: "crate", lid: "lid"}.items())]

    # Body nodes carry the CURRENT world pose of the body frame; the lid sits 0.2 above the crate.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for b in movable:
        matrix = np.asarray(_node(tree, f"body_{b}").get("matrix", np.eye(4).ravel()))
        np.testing.assert_allclose(matrix.reshape(4, 4).T[:3, 3], data.xpos[b], atol=1e-6)
    # The welded table is NOT a node of its own: it is baked into static.
    assert not any(n.get("name", "").startswith("body_") and n["name"] not in top
                   and "_m" not in n["name"] for n in tree["nodes"])

    # And it round-trips through a real loader.
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    assert {"static", f"body_{crate}", f"body_{lid}"} <= set(scene.graph.nodes)
    assert len(scene.geometry) >= 3
    json.dumps(manifest)


def test_texture_v_orientation_matches_mujoco(model):
    """A known texel at a known vertex, read from the glb with glTF's convention (v from the top
    row). If any of the flips is wrong the corners swap top for bottom."""
    glb, _ = build_scene_assets(model, label="t")
    tree, binary = _parse(glb)
    crate = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "crate")
    meshes = _child_meshes(tree, _node(tree, f"body_{crate}"))
    # Only the visual mesh: the group-3 collider box must not be drawn next to it.
    assert len(meshes) == 1 and len(meshes[0]["primitives"]) == 1
    prim = meshes[0]["primitives"][0]
    material = tree["materials"][prim["material"]]
    tex = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    image = _image(tree, binary, tex)
    w, h = image.size
    pos = _accessor(tree, binary, prim["attributes"]["POSITION"])
    uv = _accessor(tree, binary, prim["attributes"]["TEXCOORD_0"])
    assert len(pos) <= 12              # a 5-vertex OBJ, unwelded only at its UV corners
    for corner, colour in EXPECTED_TEXEL.items():
        at = np.where(np.linalg.norm(pos - np.asarray(corner, float), axis=1) < 1e-5)[0]
        assert len(at), corner
        for i in at:
            u, v = uv[i]
            px = image.getpixel((min(int(u * w), w - 1), min(int(v * h), h - 1)))
            assert np.abs(np.subtract(px, colour)).max() < 40, (corner, (u, v), px, colour)


def test_colours_follow_mujoco_and_invisible_geoms_are_skipped(model):
    glb, _ = build_scene_assets(model, label="t")
    tree, _ = _parse(glb)
    colours = [np.asarray(m["pbrMetallicRoughness"]["baseColorFactor"])
               for m in tree["materials"]]

    def linear(c):
        c = np.asarray(c, float)
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    def has(rgb):
        return any(np.abs(c[:3] - linear(rgb)).max() < 0.01 for c in colours)

    assert has((0.8, 0.1, 0.1))          # material colour: the geom left rgba at its default
    assert has((0.0, 1.0, 0.0))          # geom rgba
    assert not has((0.0, 0.0, 1.0))      # crate collider (group 3, a visual exists): not drawn
    assert not has((1.0, 0.0, 1.0))      # alpha-0 geom: MuJoCo draws nothing, nor do we


def test_manifest_bounds_floor_and_spawn(model):
    _, manifest = build_scene_assets(model, label="lounge")
    assert manifest["label"] == "lounge" and manifest["up"] == "z"
    lo, hi = np.asarray(manifest["bounds"]["min"]), np.asarray(manifest["bounds"]["max"])
    # The infinite floor is sized to the room (the wall reaches y = 2.05), not to 10 m.
    assert 2.05 <= hi[1] <= 2.5 and -2.2 <= lo[1] <= -1.5
    assert hi[2] == pytest.approx(2.0, abs=1e-3)          # wall top
    assert lo[2] == pytest.approx(-0.05, abs=1e-3)        # the floor slab's underside
    assert manifest["floor_z"] == pytest.approx(0.0, abs=1e-6)
    x, y, z = manifest["spawn"]
    assert z == pytest.approx(1.6)
    assert lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]
    # Not standing on the table (0.6 x 0.4 half extents at the origin).
    assert abs(x) > 0.6 or abs(y) > 0.4


def test_empty_and_plane_only_scenes():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="0 0 1"/></worldbody></mujoco>')
    glb, manifest = build_scene_assets(model, label="p")
    assert manifest["bounds"]["max"][:2] == [5.0, 5.0] and manifest["bodies"] == []
    assert manifest["spawn"] == [0.0, 0.0, 1.6]
    # Nothing visible at all (an alpha-0 free body): still a valid glb with the node layout.
    empty = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom size="0.1" rgba="1 1 1 0"/></body></worldbody></mujoco>')
    glb, manifest = build_scene_assets(empty, label="e")
    tree, _ = _parse(glb)
    assert [tree["nodes"][i]["name"] for i in tree["scenes"][0]["nodes"]] == ["static", "body_1"]
    assert tree["nodes"][1]["translation"] == [0.0, 0.0, 1.0]
    assert manifest["bodies"] == [{"id": 1, "name": "body1", "node": "body_1"}]


def test_floor_from_exporter_shaped_room():
    """The exporter's structure: a "room" body on slide joints (the shake rig) carrying an invisible
    group-3 floor box, with a floor lamp that must not be mistaken for the floor."""
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <body name="room"><joint name="room_x" type="slide" axis="1 0 0" damping="200"/>
          <geom name="floor" type="box" pos="0.5 0.2 0.25" size="2 1.5 0.05" group="3"
                rgba="0.5 0.5 0.5 0"/>
          <geom name="floor_surface" type="box" pos="0.5 0.2 0.29" size="2 1.5 0.01" group="2"/>
        </body>
        <body name="floor_lamp" pos="1 1 1.5"><freejoint/>
          <geom type="box" size="0.1 0.1 0.6"/></body>
      </worldbody></mujoco>""")
    _, manifest = build_scene_assets(model, label="r")
    assert manifest["floor_z"] == pytest.approx(0.3, abs=1e-6)
    assert manifest["spawn"][2] == pytest.approx(1.9, abs=1e-6)


# An exporter-shaped L: an invisible collider box over the whole bounding box (rays skip alpha-0
# geoms) and the drawn floor only on the two arms, so the bounding box's centre (3, 3) is outside.
L_ROOM = """
<mujoco><worldbody>
  <geom name="floor" type="box" pos="3 3 -1.25" size="3 3 0.05" group="3" rgba="0.5 0.5 0.5 0"/>
  <geom name="carpet_a" type="box" pos="3 0.8 -1.205" size="3 0.8 0.005" group="2"
        contype="0" conaffinity="0" rgba="0.6 0.5 0.4 ALPHA"/>
  <geom name="carpet_b" type="box" pos="0.8 3.8 -1.205" size="0.8 2.2 0.005" group="2"
        contype="0" conaffinity="0" rgba="0.6 0.5 0.4 ALPHA"/>
</worldbody></mujoco>
"""


def test_spawn_stands_on_the_floor_of_an_l_shaped_room():
    model = mujoco.MjModel.from_xml_string(L_ROOM.replace("ALPHA", "1"))
    _, manifest = build_scene_assets(model, label="l")
    x, y, z = manifest["spawn"]
    assert z == pytest.approx(-1.2 + 1.6)
    # On an arm, with the whole 0.3 m footprint on the carpet — not floating at (3, 3).
    assert (0.3 <= x <= 1.3 and 0.3 <= y <= 5.7) or (0.3 <= x <= 5.7 and 0.3 <= y <= 1.3), (x, y)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    d = mujoco.mj_ray(model, data, np.array([x, y, z]), np.array([0.0, 0.0, -1.0]),
                      np.ones(6, np.uint8), 1, -1, np.zeros(1, np.int32))
    assert z - d == pytest.approx(-1.2, abs=1e-4)

    # With no floor drawn at all, a spot over nothing is still better than no spawn.
    bare = mujoco.MjModel.from_xml_string(L_ROOM.replace("ALPHA", "0"))
    _, manifest = build_scene_assets(bare, label="bare")
    assert manifest["spawn"] == [3.0, 3.0, pytest.approx(0.4)]


def test_spawn_keeps_a_body_width_from_a_table_edge():
    """A 0.74 m table at the room centre: one ray down slips past its edge, and the side rays at
    knee, chest and eye height pass over its top, so the old search stood people 3 cm from it."""
    walls = "".join(f'<geom type="box" pos="{p}" size="{s}"/>' for p, s in [
        ("2.5 -0.05 1.25", "2.6 0.05 1.25"), ("2.5 4.05 1.25", "2.6 0.05 1.25"),
        ("-0.05 2 1.25", "0.05 2 1.25"), ("5.05 2 1.25", "0.05 2 1.25")])
    legs = "".join(f'<geom type="box" pos="{dx} {dy} 0.35" size="0.025 0.025 0.35"/>'
                   for dx in (-0.6, 0.6) for dy in (-0.3, 0.3))
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco><worldbody><geom name="floor" type="plane" size="0 0 1"/>{walls}
        <body name="table" pos="2.5 2 0">
          <geom name="top" type="box" pos="0 0 0.72" size="0.7 0.4 0.02"/>{legs}</body>
      </worldbody></mujoco>""")
    _, manifest = build_scene_assets(model, label="t")
    x, y, _z = manifest["spawn"]
    gap = np.maximum(np.abs([x - 2.5, y - 2.0]) - [0.7, 0.4], 0.0)
    assert np.linalg.norm(gap) > 0.25, (x, y)
    assert 0.35 <= x <= 4.65 and 0.35 <= y <= 3.65


def _static_primitive(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    glb, _ = build_scene_assets(model, label="uv")
    tree, binary = _parse(glb)
    meshes = _child_meshes(tree, _node(tree, "static"))
    assert len(meshes) == 1 and len(meshes[0]["primitives"]) == 1
    prim = meshes[0]["primitives"][0]
    return (_accessor(tree, binary, prim["attributes"]["POSITION"]).astype(float),
            _accessor(tree, binary, prim["attributes"]["TEXCOORD_0"]).astype(float))


@pytest.mark.parametrize(("geom", "uniform", "scl"), [
    # A finite plane, stretched: texrepeat across the plane, so scl = texrepeat / half size.
    ('type="plane" size="2 1 0.1"', "false", (2 / 2, 3 / 1)),
    # A box at an angle, uniform: scl = texrepeat in the geom's own metres.
    ('type="box" pos="1 -2 0.5" euler="0 0 30" size="0.4 0.3 0.2"', "true", (2.0, 3.0)),
    # An infinite plane has no size to stretch across: scl = texrepeat even when not uniform.
    ('type="plane" pos="0.5 0.5 -1" size="0 0 1"', "false", (2.0, 3.0)),
], ids=["stretched_plane", "uniform_box", "infinite_plane"])
def test_primitive_texture_mapping_matches_mujoco(tmp_path: Path, geom, uniform, scl):
    """MuJoCo maps a 2d texture onto a primitive by object-plane texgen (render_gl3.c settexture):
    s = 0.5*scl_x*x - 0.5, t = -0.5*scl_y*y - 0.5 in the geom's frame, t counted from the
    texture's first row like glTF's v. So t runs AGAINST y, and a uniform texture repeats
    texrepeat times per two metres; the glb's uv must be exactly (s, t)."""
    _quadrant_png(tmp_path / "quad.png")
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco><compiler texturedir="{tmp_path}"/>
        <asset><texture name="t" type="2d" file="quad.png"/>
          <material name="m" texture="t" texrepeat="2 3" texuniform="{uniform}"/></asset>
        <worldbody><geom {geom} material="m"/></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pos, uv = _static_primitive(model)
    local = (pos - data.geom_xpos[0]) @ data.geom_xmat[0].reshape(3, 3)
    expected = np.column_stack([0.5 * scl[0] * local[:, 0] - 0.5,
                                -0.5 * scl[1] * local[:, 1] - 0.5])
    np.testing.assert_allclose(uv, expected, atol=1e-4)


def test_texture_shared_by_mesh_and_primitive_is_one_image(tmp_path: Path):
    """The mesh is double-sided and the box is not: two materials, but ONE image and texture,
    or the browser decodes and uploads the same texture twice."""
    _quadrant_png(tmp_path / "quad.png")
    (tmp_path / "quad.obj").write_text(QUAD_OBJ)
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco><compiler meshdir="{tmp_path}" texturedir="{tmp_path}"/>
        <asset><texture name="t" type="2d" file="quad.png"/><material name="m" texture="t"/>
          <mesh name="quad" file="quad.obj"/></asset>
        <worldbody><geom type="box" size="1 1 0.1" material="m"/>
          <geom type="mesh" mesh="quad" pos="0 0 1" material="m"/></worldbody></mujoco>""")
    glb, _ = build_scene_assets(model, label="shared")
    tree, binary = _parse(glb)
    assert len(tree["images"]) == 1 and len(tree["textures"]) == 1
    assert sorted(m["doubleSided"] for m in tree["materials"]) == [False, True]
    assert all(m["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
               for m in tree["materials"])
    assert _image(tree, binary, 0).size == (64, 64)


def test_texture_mean_is_computed_once_per_texture(tmp_path: Path, monkeypatch):
    """Meshes without texcoords are tinted with their texture's mean colour. On a 4096² scan
    texture that mean is a quarter of a second, so it must not be paid once per geom."""
    _quadrant_png(tmp_path / "quad.png")
    calls = []
    real = geometry._texture_mean

    def counting(model, texid):
        calls.append(texid)
        return real(model, texid)

    monkeypatch.setattr(geometry, "_texture_mean", counting)
    geoms = "".join(f'<geom type="mesh" mesh="tetra" material="m" pos="{i * 0.3} 0 0"/>'
                    for i in range(30))
    model = mujoco.MjModel.from_xml_string(f"""
      <mujoco><compiler texturedir="{tmp_path}"/>
        <asset><texture name="t" type="2d" file="quad.png"/><material name="m" texture="t"/>
          <mesh name="tetra" vertex="0 0 0  0.2 0 0  0 0.2 0  0 0 0.2"/></asset>
        <worldbody>{geoms}</worldbody></mujoco>""")
    glb, _ = build_scene_assets(model, label="mean")
    assert calls == [0]
    tree, _ = _parse(glb)
    assert "images" not in tree or not tree["images"]          # tinted, not textured
    mean = np.mean([TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT], axis=0) / 255.0
    lin = np.where(mean <= 0.04045, mean / 12.92, ((mean + 0.055) / 1.055) ** 2.4)
    colour = tree["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"]
    np.testing.assert_allclose(colour[:3], lin, atol=0.01)


def _big_scene(tmp_path: Path, geoms: int = 200) -> mujoco.MjModel:
    """~200 geoms the way an export looks: textured visual meshes over collider boxes, a few dozen
    free bodies, primitives for the structure."""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.15)      # 2562 verts, 5120 faces
    uv = sphere.vertices[:, :2] / 0.3 + 0.5
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in sphere.vertices]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
    lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in sphere.faces + 1]
    (tmp_path / "blob.obj").write_text("\n".join(lines) + "\n")
    _quadrant_png(tmp_path / "tex.png", side=1024)
    rng = np.random.default_rng(0)
    parts = []
    n_bodies = geoms // 5
    for i in range(n_bodies):
        x, y = rng.uniform(-2.0, 2.0, 2)
        parts.append(
            f'<body name="obj{i}" pos="{x:.3f} {y:.3f} {0.2 + 0.4 * (i % 4):.3f}">'
            '<freejoint/>'
            '<geom type="mesh" mesh="blob" material="tex" group="2" contype="0" '
            'conaffinity="0" density="0"/>'
            '<geom type="box" size="0.15 0.15 0.15" group="3"/>'
            f'<geom type="sphere" size="0.05" pos="0 0 0.2" rgba="{i % 2} 0.5 0.5 1"/>'
            '</body>')
    for i in range(geoms - 3 * n_bodies):
        x, y = rng.uniform(-2.5, 2.5, 2)
        kind = ("box", "cylinder", "capsule", "ellipsoid")[i % 4]
        parts.append(f'<geom type="{kind}" pos="{x:.3f} {y:.3f} 1.5" size="0.05 0.04 0.03"/>')
    xml = f"""<mujoco><compiler meshdir="." texturedir="."/>
      <asset><texture name="tex" type="2d" file="tex.png"/><material name="tex" texture="tex"/>
        <mesh name="blob" file="blob.obj"/></asset>
      <worldbody><geom name="floor" type="plane" size="0 0 1"/>{''.join(parts)}</worldbody>
    </mujoco>"""
    (tmp_path / "big.xml").write_text(xml)
    return mujoco.MjModel.from_xml_path(str(tmp_path / "big.xml"))


def test_200_geom_room_is_fast(tmp_path: Path):
    model = _big_scene(tmp_path)
    assert model.ngeom >= 200
    t0 = time.perf_counter()
    glb, manifest = build_scene_assets(model, label="big")
    elapsed = time.perf_counter() - t0
    tree, _ = _parse(glb)
    assert len(manifest["bodies"]) == 40
    assert len(tree.get("images", [])) == 1          # one texture shared by every body
    # Per material merging: static is a handful of meshes, not 80 geoms' worth.
    assert len(_child_meshes(tree, _node(tree, "static"))) <= 4
    assert elapsed < 10.0, f"build_scene_assets took {elapsed:.1f}s"
