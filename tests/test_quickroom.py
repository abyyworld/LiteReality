"""The quick room: a RoomPlan room.usdz -> a stable MuJoCo room in seconds, the room call's first
version of a fresh scan.

The fixture is a SYNTHETIC RoomPlan export written the way the LiteReality Scanner app writes one
(an uncompressed zip of ASCII .usda layers, components referenced from the root, ARKit Y-up,
row-major matrices) so the parser is exercised on the real layout rather than on its own output.
It carries every case the builder has to get right: a chair tucked under a table (only a table
built as a top on legs lets it fit), a chair poking into a wall, a sink overlapping its base
cabinet (a counter run to merge), a cabinet hung on a wall, a small side table, an open door and a
window.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

from litereality_agent import cli
from litereality_agent.models.object_generation.sim.properties import OCCUPANCY_DENSITY
from litereality_agent.pipeline.quickroom import build, colours, roomplan, write_quick_scene
from litereality_agent.room_ops.roomcall.model_info import movable_body_ids
from litereality_agent.room_ops.roomcall.sim import RoomSim, load_model

WALL_PIECE = re.compile(r"Wall\d+_\d+$")    # a wall's own boxes, not what hangs in its openings
FLOOR_Y = -1.2          # ARKit height of the floor: the room is not at the origin, like a real scan
HEIGHT = 2.5
# Z-up floor plan of a 5 x 4 m room, walls counter-clockwise.
WALLS = {"Wall0": ((0.0, 0.0), (5.0, 0.0)), "Wall1": ((5.0, 0.0), (5.0, 4.0)),
         "Wall2": ((5.0, 4.0), (0.0, 4.0)), "Wall3": ((0.0, 4.0), (0.0, 0.0))}
FLOOR = ((0, 0), (5, 0), (5, 4), (0, 4))
# (wall, name, category, offset along the wall to the centre, width, height, sill)
OPENINGS = [("Wall0", "Door0", "Door(Isopen: True)", 1.0, 0.9, 2.05, 0.0),
            ("Wall2", "Window0", "Window", 2.5, 1.2, 1.0, 1.0)]
# name, category, Z-up centre (x, y, height of the box's bottom above the floor), (w, d, h), yaw
OBJECTS = [
    ("Table0", "Table", (2.5, 2.0, 0.0), (1.2, 0.8, 0.75), 0.0),
    # faces +Y (front = local -Y at yaw 180), its front 25 cm in under the table top
    ("Chair0", "Chair", (2.5, 1.6, 0.0), (0.5, 0.5, 0.85), 180.0),
    # 6 cm into Wall1
    ("Chair1", "Chair", (4.81, 1.0, 0.0), (0.5, 0.5, 0.85), 90.0),
    # a counter run against Wall3, facing +X into the room: the sink sits in its base cabinet
    ("Storage0", "Storage", (0.3, 2.0, 0.0), (1.4, 0.6, 0.9), 90.0),
    ("Sink0", "Sink", (0.3, 2.2, 0.7), (0.5, 0.45, 0.2), 90.0),
    # hung on Wall1, 1.45 m up, nothing under it
    ("Storage1", "Storage", (4.8, 3.0, 1.45), (1.0, 0.35, 0.7), -90.0),
    ("Sofa0", "Sofa", (2.5, 3.5, 0.0), (1.8, 0.85, 0.8), 0.0),
    ("Dishwasher0", "Dishwasher", (0.3, 3.2, 0.0), (0.6, 0.6, 0.9), 90.0),
    ("Table1", "Table", (4.0, 3.0, 0.0), (0.5, 0.5, 0.5), 30.0),
]


# ── a RoomPlan export, written by hand ───────────────────────────────────────────────────────────


def _matrix(yaw_deg: float, centre_zup) -> str:
    """USD row-major (row-vector) transform for a box with Z-up yaw at a Z-up centre. Rows are the
    images of the local axes in ARKit: local x -> (cos, 0, -sin), up -> +Y, local z -> x * y."""
    a = math.radians(yaw_deg)
    x, y, z = centre_zup
    rows = [(math.cos(a), 0.0, -math.sin(a), 0.0), (0.0, 1.0, 0.0, 0.0),
            (math.sin(a), 0.0, math.cos(a), 0.0), (x, z, -y, 1.0)]
    return "( " + ", ".join("(" + ", ".join(f"{v:.9g}" for v in r) + ")" for r in rows) + " )"


def _component(name: str, category: str, body: str) -> str:
    return f'''#usda 1.0
(
    defaultPrim = "{name}"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "{name}" (
    assetInfo = {{
        asset identifier = @./{name}.usda@
        string name = "{name}"
    }}
    customData = {{
        string Category = "{category}"
        string UUID = "00000000-0000-4000-8000-{abs(hash(name)) % 10**12:012d}"
    }}
    kind = "component"
)
{{
{body}
    def Material "{name}_color"
    {{
        token outputs:surface.connect = </{name}/{name}_color/surfaceShader.outputs:surface>

        def Shader "surfaceShader"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1, 1, 1)
            token outputs:surface
        }}
    }}
}}
'''


def _cube(name: str, scale, yaw: float, centre) -> str:
    return f'''    def Cube "{name}"
    {{
        rel material:binding = </{name}/{name}_color>
        double size = 1
        double3 xformOp:scale = ({scale[0]:.9g}, {scale[1]:.9g}, {scale[2]:.9g})
        matrix4d xformOp:transform = {_matrix(yaw, centre)}
        uniform token[] xformOpOrder = ["xformOp:transform", "xformOp:scale"]
    }}
'''


def _floor_mesh(name: str = "Floor0", corners=FLOOR) -> str:
    # Local XY at z = 0 (top) and -0.0001 (bottom), fanned from the first corner (every floor here
    # is star-shaped from it); the transform turns local z into ARKit +Y.
    n = len(corners)
    pts = [(x, y, 0.0) for x, y in corners] + [(x, y, -0.0001) for x, y in corners]
    top = [(0, i, i + 1) for i in range(1, n - 1)]
    index = [v for t in top for v in t] + [v + n for a, b, c in top for v in (a, c, b)]
    points = ", ".join(f"({x}, {y}, {z})" for x, y, z in pts)
    rows = "( (1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, %s, 0, 1) )" % FLOOR_Y
    return f'''    def Mesh "{name}"
    {{
        int[] faceVertexCounts = [{", ".join(["3"] * (len(index) // 3))}]
        int[] faceVertexIndices = [{", ".join(map(str, index))}]
        rel material:binding = </{name}/{name}_color>
        point3f[] points = [{points}]
        uniform token subdivisionScheme = "none"
        matrix4d xformOp:transform = {rows}
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
'''


def _group(name: str, refs: list[str], indent: str) -> str:
    if len(refs) == 1:
        ref = f"prepend references = @{refs[0]}@"
    else:
        inner = ",\n".join(f"{indent}        @{r}@" for r in refs)
        ref = f"prepend references = [\n{inner}\n{indent}    ]"
    return (f'{indent}def Xform "{name}" (\n{indent}    kind = "group"\n{indent}    {ref}\n'
            f'{indent})\n{indent}{{\n{indent}}}\n')


def write_roomplan_usdz(path: Path, objects=OBJECTS, openings=OPENINGS, *, walls=None,
                        floors=(FLOOR,), mesh_walls=()) -> Path:
    """``floors``: one polygon per floor slab (none: an iOS 16 export); ``mesh_walls``: walls
    written as a Mesh rather than a box, as a mesh-mode export does."""
    files: dict[str, str] = {}
    arch = []
    for wall, ((x0, y0), (x1, y1)) in (WALLS if walls is None else walls).items():
        folder = f"assets/Parametric/Walls/{wall}"
        length = math.hypot(x1 - x0, y1 - y0)
        yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
        mid = ((x0 + x1) / 2, (y0 + y1) / 2, FLOOR_Y + HEIGHT / 2)
        files[f"{folder}/{wall}.usda"] = _component(
            wall, "Wall", _floor_mesh(wall, ((x0, y0), (x1, y1), (x1, y1 + 0.01)))
            if wall in mesh_walls else _cube(wall, (length, HEIGHT, 0.0001), yaw, mid))
        refs = [f"./{folder}/{wall}.usda"]
        for w, name, category, offset, width, height, sill in openings:
            if w != wall:
                continue
            u = ((x1 - x0) / length, (y1 - y0) / length)
            centre = (x0 + u[0] * offset, y0 + u[1] * offset, FLOOR_Y + sill + height / 2)
            files[f"{folder}/{name}.usda"] = _component(
                name, category, _cube(name, (width, height, 0.0005), yaw, centre))
            refs.append(f"./{folder}/{name}.usda")
        index = wall[len("Wall"):]
        arch.append(_group(f"Wall_{index}_grp", refs, " " * 12))
    for k, corners in enumerate(floors):
        files[f"assets/Parametric/Floors/Floor{k}.usda"] = _component(
            f"Floor{k}", "Floor", _floor_mesh(f"Floor{k}", corners))
    by_category: dict[str, list[str]] = {}
    for name, category, (x, y, bottom), (w, d, h), yaw in objects:
        rel = f"assets/Parametric/{category}/{name}.usda"
        centre = (x, y, FLOOR_Y + bottom + h / 2)
        files[rel] = _component(name, category, _cube(name, (w, h, d), yaw, centre))
        by_category.setdefault(category, []).append(f"./{rel}")
    groups = "".join(_group(f"{c}_grp", refs, " " * 12) for c, refs in by_category.items())
    root = f'''#usda 1.0
(
    defaultPrim = "room"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "room" (
    kind = "assembly"
)
{{
    def Xform "Parametric_grp" (
        kind = "group"
    )
    {{
        def Xform "Arch_grp" (
            kind = "group"
        )
        {{
{"".join(arch)}        }}

{"".join(_group(f"Floor_{k}_grp", [f"./assets/Parametric/Floors/Floor{k}.usda"], " " * 8)
         for k in range(len(floors)))}
        def Xform "Object_grp" (
            kind = "group"
        )
        {{
{groups}        }}
    }}
}}
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("room.usda", root)
        for name, text in files.items():
            zf.writestr(name, text)
    return path


@pytest.fixture()
def scan(tmp_path) -> Path:
    folder = tmp_path / "scans" / "test_room"
    write_roomplan_usdz(folder / "room.usdz")
    return folder


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """The quick room of the fixture scan, built once: (scene path, report, compiled model)."""
    root = tmp_path_factory.mktemp("quick")
    folder = root / "scans" / "test_room"
    write_roomplan_usdz(folder / "room.usdz")
    path = write_quick_scene(folder, root / "out")
    report = json.loads((root / "out" / build.REPORT_NAME).read_text())
    return path, report, load_model(path)


def _body(model, name: str) -> int:
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert b >= 0, name
    return b


def _geoms(model, prefix: str) -> list[int]:
    return [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(prefix)]


def _aabb(model, data, g: int) -> tuple[np.ndarray, np.ndarray]:
    rot = data.geom_xmat[g].reshape(3, 3)
    half = np.abs(rot) @ model.geom_size[g]
    return data.geom_xpos[g] - half, data.geom_xpos[g] + half


# ── parsing ──────────────────────────────────────────────────────────────────────────────────────


def test_parse_reads_walls_openings_floor_and_boxes_in_the_pipelines_frame(scan):
    plan = roomplan.parse_room(scan)
    assert plan.source == scan / "room.usdz"
    assert plan.floor_z == pytest.approx(FLOOR_Y - 0.0001, abs=1e-4)
    assert plan.ceiling_z == pytest.approx(plan.floor_z + HEIGHT, abs=1e-3)
    assert list(plan.walls) == ["Wall0", "Wall1", "Wall2", "Wall3"]
    for name, (start, end) in WALLS.items():
        wall = plan.walls[name]
        assert wall.start == pytest.approx(start, abs=1e-4)
        assert wall.end == pytest.approx(end, abs=1e-4)
        assert wall.thickness < 0.001              # RoomPlan's sheet, as measured
    door, window = plan.openings["Wall0_Door_0"], plan.openings["Wall2_Window_0"]
    assert (door.kind, door.wall, door.source, door.is_open) == ("door", "Wall0", "Door0", True)
    assert (door.offset, door.width, door.height) == pytest.approx((1.0, 0.9, 2.05), abs=1e-3)
    assert door.span == pytest.approx((0.55, 1.45), abs=1e-3)
    assert (window.kind, window.is_open) == ("window", None)
    assert window.sill == pytest.approx(1.0, abs=1e-3)
    assert [len(loop) for loop in plan.floor.outlines] == [4]
    assert plan.warnings == [] and plan.skipped == []
    for name, category, (x, y, bottom), size, yaw in OBJECTS:
        obj = plan.objects[name]
        assert obj.category == category.lower() and obj.roomplan_category == category
        assert obj.center == pytest.approx((x, y, plan.floor_z + 0.0001 + bottom + size[2] / 2),
                                           abs=1e-3)
        assert obj.size == pytest.approx(size, abs=1e-4)
        assert math.cos(math.radians(obj.yaw - yaw)) == pytest.approx(1.0, abs=1e-6)


def test_parse_finds_roomplan_subfolder_and_refuses_what_it_cannot_read(tmp_path):
    write_roomplan_usdz(tmp_path / "s" / "roomplan" / "room.usdz")
    assert roomplan.find_room_usdz(tmp_path / "s") == tmp_path / "s" / "roomplan" / "room.usdz"
    assert roomplan.parse_room(tmp_path / "s").walls
    assert roomplan.find_room_usdz(tmp_path) is None
    with pytest.raises(FileNotFoundError):
        roomplan.parse_room(tmp_path)
    crate = tmp_path / "crate.usdz"
    with zipfile.ZipFile(crate, "w") as zf:
        zf.writestr("room.usdc", b"PXR-USDC")
    with pytest.raises(ValueError, match="binary USD"):
        roomplan.parse_room(crate)


def _main(monkeypatch, argv: list[str]) -> int:
    """`cli.main` without it copying the checkout's .env into this test process's environment."""
    monkeypatch.setattr(cli, "load_settings",
                        lambda: type("S", (), {"apply_environment": lambda self: None})())
    return cli.main(argv)


def test_a_damaged_export_is_one_clear_error(scan, monkeypatch):
    """A layer the root references but the zip lacks, a file that is not a zip: a ValueError that
    names the file (which `cli.main` prints as one line), never a KeyError from inside the
    reader."""
    usdz = scan / "room.usdz"
    with zipfile.ZipFile(usdz) as zf:
        layers = {n: zf.read(n) for n in zf.namelist()}
    with zipfile.ZipFile(usdz, "w") as zf:
        for name, data in layers.items():
            if "Chair0" not in name:
                zf.writestr(name, data)
    with pytest.raises(ValueError, match=r"room\.usdz is not a readable RoomPlan export: .*Chair0"):
        roomplan.parse_room(scan)
    usdz.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="not a readable RoomPlan export"):
        roomplan.parse_room(scan)
    with pytest.raises(SystemExit, match="not a readable RoomPlan export"):
        _main(monkeypatch, ["quickroom", str(scan), "--out", str(scan / "out")])


# The L of an L-shaped room, its walls named out of perimeter order and two of them reversed.
L_ROOM = ((0, 0), (6, 0), (6, 3), (3, 3), (3, 6), (0, 6))
_L = [(L_ROOM[i], L_ROOM[(i + 1) % 6]) for i in range(6)]
L_WALLS = {"Wall0": _L[0], "Wall1": _L[3][::-1], "Wall2": _L[1], "Wall3": _L[5],
           "Wall4": _L[2][::-1], "Wall5": _L[4]}


def test_without_a_floor_mesh_the_walls_are_chained_into_the_outline(tmp_path):
    """iOS 16 writes no floor mesh: the outline is the walls chained end to end — not their
    start points in name order, which here would be a bow-tie and grow walls into the room."""
    write_roomplan_usdz(tmp_path / "room.usdz", objects=[], openings=[], walls=L_WALLS, floors=())
    plan = roomplan.parse_room(tmp_path)
    assert plan.floor is None
    (loop,) = plan.outlines()
    assert {tuple(p) for p in loop.round(3).tolist()} == {tuple(map(float, c)) for c in L_ROOM}
    assert abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(loop, np.roll(loop, -1, 0)))) / 2 \
        == pytest.approx(27.0)                                   # a simple polygon: the L's area
    assert plan.on_floor(np.array([(1, 1), (5, 1), (1, 5), (5, 5)])).tolist() == \
        [True, True, True, False]
    model = load_model(write_quick_scene(tmp_path, tmp_path / "out", colours=False))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for g in range(model.ngeom):
        if WALL_PIECE.match(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""):
            rot, half = data.geom_xmat[g].reshape(3, 3), model.geom_size[g] - 0.01
            probe = np.array([data.geom_xpos[g] + rot @ np.array([sx * half[0], sy * half[1], 0])
                              for sx in np.linspace(-1, 1, 9) for sy in (-1, 0, 1)])
            assert not plan.on_floor(probe[:, :2]).any(), "a wall grown into the room"


def test_every_floor_slab_counts_and_an_implausible_one_is_dropped(tmp_path):
    """Two rooms, a slab each: the floor is their union (a chair in the first room is not "off
    the floor plate"), at the lowest slab's height. A collapsed slab — 4 cm across under a 5 x 4
    m room — fails the ingest adapter's own MIN_PLATE_FRACTION gate and is dropped here as there,
    so the repair solves the room the full pipeline solves."""
    from litereality_agent.pipeline.scene_init.layout.adjust import check

    two = {**WALLS, "Wall4": ((5.0, 0.0), (9.0, 0.0)), "Wall5": ((9.0, 0.0), (9.0, 4.0)),
           "Wall6": ((9.0, 4.0), (5.0, 4.0))}
    chair = [("Chair0", "Chair", (2.5, 2.0, 0.0), (0.5, 0.5, 0.85), 0.0)]
    write_roomplan_usdz(tmp_path / "a" / "room.usdz", objects=chair, openings=[], walls=two,
                        floors=(FLOOR, ((5, 0), (9, 0), (9, 4), (5, 4))))
    plan = roomplan.parse_room(tmp_path / "a")
    assert [len(loop) for loop in plan.floor.outlines] == [4, 4]
    assert plan.on_floor(np.array([(2.5, 2.0), (7.0, 2.0), (10.0, 2.0)])).tolist() == \
        [True, True, False]
    assert len(plan.shell()["floor"]["faces"]) == 8
    assert not [v for v in check(plan.shell()) if v.severity == "error"]

    write_roomplan_usdz(tmp_path / "b" / "room.usdz", objects=chair, openings=[],
                        floors=(((0.1, 0.1), (0.4, 0.1), (0.4, 0.14), (0.1, 0.14)),))
    plan = roomplan.parse_room(tmp_path / "b")
    assert plan.floor is None and "not believable" in plan.warnings[0]
    assert plan.floor_z == pytest.approx(FLOOR_Y, abs=1e-4)          # from the walls, as ingest
    assert plan.shell()["floor"] == {"verts": [], "faces": []}
    assert not [v for v in check(plan.shell()) if v.severity == "error"]


def test_a_mesh_wall_is_reported_not_silently_dropped(tmp_path, capsys, monkeypatch):
    openings = [("Wall1", "Door0", "Door(Isopen: True)", 1.0, 0.9, 2.05, 0.0)]
    write_roomplan_usdz(tmp_path / "s" / "room.usdz", objects=[], openings=openings,
                        mesh_walls=("Wall1",))
    plan = roomplan.parse_room(tmp_path / "s")
    assert "Wall1" not in plan.walls and plan.skipped == ["Wall1", "Door0"]
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "run"))
    assert _main(monkeypatch, ["quickroom", str(tmp_path / "s"), "--no-colours"]) == 0
    out = capsys.readouterr().out
    assert "warning" in out and "Wall1 is a mesh" in out and "Door0 has no wall" in out
    report = json.loads((tmp_path / "run" / "s" / "quick" / build.REPORT_NAME).read_text())
    assert len(report["warnings"]) == 2


# ── templates, physics, free/static ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["table", "chair", "stool", "sofa", "bed", "counter", "stairs",
                                  "bathtub", "toilet", "box"])
def test_every_template_fills_its_box_and_stays_inside_it(kind):
    size = (0.9, 0.7, 0.8 if kind != "bed" else 1.0)
    parts = build.template(kind, size)
    lo = np.min([np.subtract(p.centre, p.half) for p in parts], axis=0)
    hi = np.max([np.add(p.centre, p.half) for p in parts], axis=0)
    half = np.asarray(size) / 2
    assert np.all(lo >= -half - 1e-9) and np.all(hi <= half + 1e-9)
    assert lo[2] == pytest.approx(-half[2]) and hi[2] == pytest.approx(half[2])   # floor to top
    assert all(min(p.half) > 0 for p in parts)


@pytest.mark.parametrize("kind", ["table", "chair", "stool", "sofa", "bed", "counter", "stairs",
                                  "bathtub", "toilet", "box"])
@pytest.mark.parametrize("size", [(1.8, 0.12, 0.8), (0.6, 0.6, 0.015), (0.01, 0.01, 0.01),
                                  (2.0, 0.03, 0.8)])
def test_no_template_part_is_thinner_than_min_extent(kind, size):
    """A 12 cm deep sofa or a 2 cm tall table used to leave a part of size 0, and MuJoCo then
    refused the whole room ("size must be positive")."""
    assert all(min(p.half) >= build.MIN_EXTENT / 2 for p in build.template(kind, size))


def test_a_table_is_a_top_on_legs_and_a_chair_has_its_back_behind_it():
    table = {p.name: p for p in build.template("table", (1.2, 0.8, 0.75))}
    assert set(table) == {"top", "leg0", "leg1", "leg2", "leg3"}
    top = table["top"]
    assert top.centre[2] + top.half[2] == pytest.approx(0.375)
    assert top.half[2] * 2 <= 0.04
    assert all(table[f"leg{i}"].half[0] <= 0.03 for i in range(4))
    chair = {p.name: p for p in build.template("chair", (0.5, 0.5, 0.85))}
    assert {"seat", "back"} <= set(chair)
    # RoomPlan's front is local -Y (`shell.facing_yaw`), so the back is at +Y
    assert chair["back"].centre[1] > 0.2
    assert chair["seat"].centre[2] + chair["seat"].half[2] == pytest.approx(-0.425 + 0.4675)


def test_bodies_keep_the_scans_handles_and_the_exporters_free_static_rule(built):
    _path, report, model = built
    bodies = report["bodies"]
    joints = {name: b["joint"] for name, b in bodies.items()}
    assert joints["Chair0"] == joints["Chair1"] == "free"
    assert joints["Table1"] == "free"                               # a small side table moves
    assert joints["Table0"] == joints["Sofa0"] == joints["Dishwasher0"] == "static"
    assert joints["Storage1"] == "static" and bodies["Storage1"]["why"].startswith("hung")
    assert joints["Wall0_Door_0"] == "hinge"
    # the counter run merged under the pipeline's own name; its members are gone
    assert bodies["Sink_Storage0"]["merged_from"] == ["Sink0", "Storage0"]
    assert "Sink0" not in bodies and "Storage0" not in bodies
    assert report["layout"]["merged"] == {"Sink_Storage0": ["Sink0", "Storage0"]}
    # free joints named as the exporter names them, so a live swap finds them
    for name in ("Chair0", "Chair1", "Table1"):
        j = int(model.body_jntadr[_body(model, name)])
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) == f"{name}_free"
    movable = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
               for b in movable_body_ids(model)}
    assert movable == {"Chair0", "Chair1", "Table1", "Wall0_Door_0"}


def test_mass_is_box_volume_times_occupancy_density(built):
    _path, report, model = built
    for name, category in (("Chair0", "chair"), ("Dishwasher0", "dishwasher"),
                           ("Sofa0", "sofa"), ("Table1", "table")):
        b = report["bodies"][name]
        expected = float(np.prod(b["size"])) * OCCUPANCY_DENSITY[category]
        assert b["mass_kg"] == pytest.approx(expected, rel=1e-3)
        # ...and that is what MuJoCo integrates, spread over the template's parts
        assert float(model.body_mass[_body(model, name)]) == pytest.approx(expected, rel=1e-3)
    door = report["bodies"]["Wall0_Door_0"]
    assert door["mass_kg"] == pytest.approx(0.9 * 2.05 * build.DOOR_UNIT_DEPTH
                                            * OCCUPANCY_DENSITY["door"], rel=1e-3)
    assert report["bodies"]["Sofa0"]["friction"] == pytest.approx(0.75)    # fabric
    assert report["bodies"]["Dishwasher0"]["material_guess"] == "metal"


def test_layout_wall_thickness_is_the_one_the_ingest_layout_uses():
    import inspect

    from litereality_agent.pipeline.scene_init.layout import adapter

    default = inspect.signature(adapter.shell_from_scene_data).parameters["thickness"].default
    assert build.LAYOUT_WALL_THICKNESS == default


# ── structure ────────────────────────────────────────────────────────────────────────────────────


def test_walls_are_cut_around_openings_and_grown_outward(built):
    _path, _report, model = built
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fz = FLOOR_Y - 0.0001
    door_lo = np.array([0.56, -0.2, fz + 0.01])
    door_hi = np.array([1.44, 0.2, fz + 2.04])
    window_lo = np.array([1.91, 3.8, fz + 1.01])
    window_hi = np.array([3.09, 4.2, fz + 1.99])
    for prefix, lo_cut, hi_cut in (("Wall0_", door_lo, door_hi), ("Wall2_", window_lo, window_hi)):
        pieces = [g for g in _geoms(model, prefix) if WALL_PIECE.match(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        assert len(pieces) >= 3
        for g in pieces:
            lo, hi = _aabb(model, data, g)
            assert np.any(hi <= lo_cut + 1e-6) or np.any(lo >= hi_cut - 1e-6), "wall over opening"
    for g in _geoms(model, "Wall"):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if not WALL_PIECE.match(name):
            continue                                   # the door leaf, glass and window frame
        lo, hi = _aabb(model, data, g)
        # nothing of a wall inside the 5 x 4 m room: the scanned plane is its room-side face
        assert hi[0] <= 0.0 + 1e-6 or lo[0] >= 5.0 - 1e-6 or hi[1] <= 0.0 + 1e-6 \
            or lo[1] >= 4.0 - 1e-6, name
        assert model.geom_size[g][1] == pytest.approx(build.WALL_THICKNESS / 2)
    assert _geoms(model, "Wall2_Window_0_glass")
    glass = _geoms(model, "Wall2_Window_0_glass")[0]
    assert model.mat_rgba[model.geom_matid[glass]][3] < 0.5
    assert _geoms(model, "floor") and _geoms(model, "ceiling")


def test_wall_pieces_column_decomposition():
    assert build._wall_pieces(4.0, 2.5, []) == [(0.0, 4.0, 0.0, 2.5)]
    door = build._wall_pieces(4.0, 2.5, [(1.0, 2.0, 0.0, 2.0)])
    assert door == [(0.0, 1.0, 0.0, 2.5), (1.0, 2.0, 2.0, 2.5), (2.0, 4.0, 0.0, 2.5)]
    both = build._wall_pieces(4.0, 2.5, [(1.0, 2.0, 0.0, 2.0), (2.5, 3.5, 1.0, 2.0)])
    assert (2.5, 3.5, 0.0, 1.0) in both and (2.5, 3.5, 2.0, 2.5) in both
    area = sum((b - a) * (z1 - z0) for a, b, z0, z1 in both)
    assert area == pytest.approx(4.0 * 2.5 - 1.0 * 2.0 - 1.0 * 1.0)


def test_the_door_hangs_on_a_hinge_at_its_jamb_and_starts_open(built):
    _path, report, model = built
    leaf = _body(model, "Wall0_Door_0")
    j = int(model.body_jntadr[leaf])
    assert int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) == "Wall0_Door_0_hinge"
    assert model.jnt_range[j][0] == pytest.approx(0.0)
    assert model.jnt_range[j][1] >= math.radians(90)
    # RoomPlan saw it open: it starts open, and qpos 0 is closed
    assert float(model.qpos0[model.jnt_qposadr[j]]) == pytest.approx(math.radians(80), abs=1e-3)
    assert report["door_angles_deg"]["Wall0_Door_0"] == pytest.approx(80.0)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    hinge = data.xpos[leaf]
    jambs = [np.array([0.55, 0.0]), np.array([1.45, 0.0])]
    assert min(np.linalg.norm(hinge[:2] - jamb) for jamb in jambs) < 0.02
    # open INTO the room (+Y here)
    tip = data.geom_xpos[_geoms(model, "Wall0_Door_0_leaf")[0]]
    assert tip[1] > 0.3
    # closed, the leaf sits in its cutout without touching the jambs
    data.qpos[model.jnt_qposadr[j]] = 0.0
    mujoco.mj_forward(model, data)
    for i in range(data.ncon):
        c = data.contact[i]
        assert leaf not in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]) \
            or c.dist > -0.001


def test_a_hand_can_swing_the_door(built):
    _path, _report, model = built
    sim = RoomSim(model, settle_seconds=0.3)
    leaf = _body(model, "Wall0_Door_0")
    adr = int(model.jnt_qposadr[model.body_jntadr[leaf]])
    start = float(sim.data.qpos[adr])
    g = _geoms(model, "Wall0_Door_0_leaf")[0]
    rot = sim.data.geom_xmat[g].reshape(3, 3)
    edge = sim.data.geom_xpos[g] + rot @ np.array([model.geom_size[g][0] * 0.9, 0.0, 0.0])
    hinge = sim.data.xpos[leaf]
    assert sim.grab("hand", leaf, edge)
    # pull the free edge round towards closed
    r = edge[:2] - hinge[:2]
    a = -math.radians(40)
    target = np.r_[hinge[:2] + np.array([[math.cos(a), -math.sin(a)],
                                          [math.sin(a), math.cos(a)]]) @ r, edge[2]]
    sign = 1.0 if float(np.cross(np.r_[r, 0], np.r_[target[:2] - hinge[:2], 0])[2]) > 0 else -1.0
    for _ in range(30):
        sim.move("hand", target)
        sim.step_realtime(0.05)
    moved = float(sim.data.qpos[adr]) - start
    assert abs(moved) > math.radians(15)
    axis = model.jnt_axis[model.body_jntadr[leaf]][2]
    assert np.sign(moved * axis) == sign


def test_awkward_openings_still_get_working_doors_and_compile(tmp_path):
    """RoomPlan draws doors past the end of their wall, sills under the floor, doors taller than
    the wall and windows a few centimetres tall. Each door still hangs on a hinge inside its wall
    with its leaf clear of floor and ceiling (none pinned shut), and nothing fails to compile."""
    openings = [("Wall1", "Door0", "Door(Isopen: True)", 3.8, 0.9, 2.0, 0.0),     # past the end
                ("Wall0", "Door1", "Door(Isopen: True)", 1.5, 0.9, 2.05, -0.05),  # sill under
                ("Wall2", "Door2", "Door(Isopen: False)", 1.5, 0.9, 2.6, 0.0),    # above the wall
                ("Wall3", "Window0", "Window", 2.0, 0.3, 0.07, 1.5)]              # 7 cm tall
    write_roomplan_usdz(tmp_path / "room.usdz", objects=[], openings=openings)
    path = write_quick_scene(tmp_path, tmp_path / "out", colours=False)
    report = json.loads((tmp_path / "out" / build.REPORT_NAME).read_text())
    assert report["pinned"] == [] and report["initial_overlaps"] == []
    doors = {"Wall1_Door_0": 4.0, "Wall0_Door_1": 5.0, "Wall2_Door_2": 5.0}
    assert {n: report["bodies"][n]["joint"] for n in doors} == dict.fromkeys(doors, "hinge")
    model = load_model(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    plan = roomplan.parse_room(tmp_path)
    for name, length in doors.items():
        wall = plan.walls[plan.openings[name].wall]
        hinge = data.xpos[_body(model, name)][:2]
        t = float(np.dot(hinge - np.asarray(wall.start), wall.along))
        assert 0.0 <= t <= length
        lo, hi = _aabb(model, data, _geoms(model, f"{name}_leaf")[0])
        assert lo[2] > plan.floor_z and hi[2] < plan.ceiling_z
    assert _geoms(model, "Wall3_Window_0_glass") and not _geoms(model, "Wall3_Window_0_frame_left")


# ── the whole room ───────────────────────────────────────────────────────────────────────────────


def _deep_contacts(model, tol: float) -> list[tuple[str, str, float]]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < -tol:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
                     for g in (c.geom1, c.geom2)]
            out.append((*names, c.dist))
    return out


def test_the_room_loads_with_no_overlaps_settles_and_stays_finite(built):
    _path, report, model = built
    assert report["initial_overlaps"] == []
    assert _deep_contacts(model, 0.001) == []
    sim = RoomSim(model)
    settled = {b: sim.data.xpos[b].copy() for b in sim.movable}
    for _ in range(100):
        sim.step_realtime(0.1)
    assert np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel))
    assert sim.unstable_resets == 0 and sim.respawns == 0
    for b in sim.movable:
        assert np.linalg.norm(sim.data.xpos[b] - settled[b]) < 0.01


def test_a_chair_tucked_under_the_table_fits_without_being_moved(built):
    _path, report, model = built
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    chair, table = _body(model, "Chair0"), _body(model, "Table0")
    # it really is under the table: the chair's front edge is inside the table's footprint
    assert data.xpos[chair][1] + 0.25 > 1.6 + 0.2
    assert "Chair0" not in report["separated_m"]
    for i in range(data.ncon):
        c = data.contact[i]
        pair = {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
        if pair == {chair, table}:
            assert c.dist > -0.0005


def test_a_box_table_would_not_have_fitted(scan, tmp_path, monkeypatch):
    """The regression the templates exist for: as a solid box the table swallows the chair."""
    monkeypatch.setattr(build, "template",
                        lambda kind, size: [build.Part("box", (0, 0, 0), tuple(v / 2 for v in size))])
    report_dir = tmp_path / "boxes"
    write_quick_scene(scan, report_dir, colours=False)
    report = json.loads((report_dir / build.REPORT_NAME).read_text())
    assert report["separated_m"].get("Chair0", 0.0) > 0.2 or "Chair0" in report["pinned"]


def test_an_overlap_the_layout_leaves_is_separated(scan, tmp_path, monkeypatch):
    """With the layout repair off, Chair1 starts 6 cm into Wall1: the separation pass pushes it
    back into the room — not up, not through the wall — and nothing starts overlapping."""
    monkeypatch.setenv("LR_LAYOUT", "0")
    path = write_quick_scene(scan, tmp_path / "out", colours=False)
    report = json.loads((tmp_path / "out" / build.REPORT_NAME).read_text())
    assert report["layout"] == {"skipped": "$LR_LAYOUT=0", "merged": {"Sink_Storage0":
                                                                       ["Sink0", "Storage0"]}}
    assert report["separated_m"]["Chair1"] == pytest.approx(0.06, abs=0.01)
    assert report["pinned"] == [] and report["initial_overlaps"] == []
    model = load_model(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    chair = data.xpos[_body(model, "Chair1")]
    assert chair[0] == pytest.approx(5.0 - 0.25, abs=0.01)
    assert chair[2] == pytest.approx(FLOOR_Y - 0.0001 + 0.425 + build.FLOOR_GAP, abs=1e-3)
    assert _deep_contacts(model, 0.001) == []


def test_build_is_fast(scan, tmp_path):
    import time

    started = time.perf_counter()
    write_quick_scene(scan, tmp_path / "out")
    assert time.perf_counter() - started < 5.0


def _office(tables: int, pitch: float = 0.4) -> list:
    """Rows of desks, each with four chairs pushed in too close together and a cabinet: an open-
    plan office as RoomPlan sees one, with a violation or two at every desk."""
    objects = []
    for i in range(tables):
        x, y = 0.6 + (i % 2) * 2.3, 0.4 + (i // 2) * 1.9
        objects.append((f"Table{i}", "Table", (x + 0.7, y + 0.8, 0.0), (1.4, 0.8, 0.75), 0.0))
        objects += [(f"Chair{4 * i + k}", "Chair", (x + 0.15 + pitch * k, y + 0.35, 0.0),
                     (0.5, 0.5, 0.85), 180.0) for k in range(4)]
        objects.append((f"Storage{i}", "Storage", (x + 1.7, y + 1.5, 0.0), (0.4, 0.4, 0.9), 0.0))
    return objects


OFFICE = ((0, 0), (5, 0), (5, 10), (0, 10))
OFFICE_WALLS = {f"Wall{i}": (OFFICE[i], OFFICE[(i + 1) % 4]) for i in range(4)}


def test_a_room_too_big_to_repair_in_seconds_keeps_its_boxes_and_says_so(tmp_path):
    """60 objects: the repair alone took 32 s here (150 objects: ten minutes) while the room call
    waited. Past the size gate it is not tried; the room still starts with nothing overlapping."""
    import time

    write_roomplan_usdz(tmp_path / "room.usdz", objects=_office(10), openings=[],
                        walls=OFFICE_WALLS, floors=(OFFICE,))
    started = time.perf_counter()
    write_quick_scene(tmp_path, tmp_path / "out", colours=False)
    assert time.perf_counter() - started < 5.0
    report = json.loads((tmp_path / "out" / build.REPORT_NAME).read_text())
    assert len(report["bodies"]) == 60
    assert report["layout"]["skipped"].startswith("budget: 60 objects")
    assert report["layout"]["after"] == report["layout"]["before"] != []
    assert report["initial_overlaps"] == [] and report["pinned"] == []


def test_the_repair_has_a_hard_deadline(scan, monkeypatch):
    import threading

    if threading.current_thread() is not threading.main_thread():
        pytest.skip("the deadline is a timer signal, which only the main thread receives")
    plan = roomplan.parse_room(scan)
    objects, _merged = build.merge_counter_runs(plan)
    monkeypatch.setattr(build, "LAYOUT_SECONDS", 1e-3)
    out, layout = build.repair_layout(plan, objects)
    assert out == objects and layout["skipped"] == "budget: the repair ran past 0.001 s"
    assert layout["before"]
    import signal

    assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    monkeypatch.setattr(build, "LAYOUT_MAX_VIOLATIONS", len(layout["before"]) - 1)
    monkeypatch.setattr(build, "LAYOUT_SECONDS", 30.0)
    _out, layout = build.repair_layout(plan, objects)
    assert layout["skipped"].startswith("budget: ") and "violations" in layout["skipped"]


def test_chairs_side_by_side_are_pushed_apart_not_stacked(tmp_path, monkeypatch):
    """Two equal chairs meet seat to seat, where the shallowest way out is straight up. Lifting
    one stacked it on the other and it toppled off seconds into the call; free bodies beside each
    other now move apart sideways, and everything starts on the floor."""
    monkeypatch.setenv("LR_LAYOUT", "0")          # leave the overlaps to the separation pass
    chairs = [(f"Chair{i}", "Chair", (1.5 + 0.3 * i, 2.0 + 0.05 * i, 0.0), (0.5, 0.5, 0.85),
               15.0 * i) for i in range(4)]
    write_roomplan_usdz(tmp_path / "room.usdz", objects=chairs, openings=[])
    path = write_quick_scene(tmp_path, tmp_path / "out", colours=False)
    report = json.loads((tmp_path / "out" / build.REPORT_NAME).read_text())
    assert report["pinned"] == [] and report["initial_overlaps"] == []
    assert set(report["separated_m"]) >= {"Chair0", "Chair3"}
    model = load_model(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fz = FLOOR_Y - 0.0001
    for i in range(4):
        assert data.xpos[_body(model, f"Chair{i}")][2] == pytest.approx(
            fz + 0.425 + build.FLOOR_GAP, abs=1e-3)
    sim = RoomSim(model)
    settled = {b: sim.data.xpos[b].copy() for b in sim.movable}
    for _ in range(40):
        sim.step_realtime(0.1)
    assert all(np.linalg.norm(sim.data.xpos[b] - settled[b]) < 0.01 for b in sim.movable)


def test_a_chair_seen_through_the_doorway_has_floor_under_it(tmp_path):
    """A chair outside the walls (seen through an opening) used to stand over nothing: it fell
    for ever and was respawned again and again. The floor reaches under every object."""
    openings = [("Wall1", "Opening0", "Opening", 2.0, 1.2, 2.1, 0.0)]
    chair = [("Chair0", "Chair", (6.2, 2.0, 0.0), (0.5, 0.5, 0.85), 0.0)]
    write_roomplan_usdz(tmp_path / "room.usdz", objects=chair, openings=openings)
    model = load_model(write_quick_scene(tmp_path, tmp_path / "out", colours=False))
    floor = _geoms(model, "floor")[0]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lo, hi = _aabb(model, data, floor)
    assert hi[0] > 6.2 + 0.25
    sim = RoomSim(model)
    chair_id = _body(model, "Chair0")
    start = sim.data.xpos[chair_id].copy()
    for _ in range(30):
        sim.step_realtime(0.1)
    assert sim.respawns == 0
    assert np.linalg.norm(sim.data.xpos[chair_id] - start) < 0.01


# ── colours ──────────────────────────────────────────────────────────────────────────────────────


def _frame_looking_down(scan: Path, index: int = 0) -> None:
    """One capture frame from 2.2 m above the floor, straight down at Table0: its top red, the
    floor grey — RGB, depth (mm) and confidence the way the scanner writes them."""
    W, H, w, h = 320, 240, 64, 48
    fx = fy = 200.0
    cx, cy = W / 2, H / 2
    cam = np.array([2.5, 2.0, FLOOR_Y + 2.2])                 # Z-up
    # camera -> ARKit world: looking down -Y_arkit, image up = ARKit -Z (Z-up +Y)
    pose = np.eye(4)
    pose[:3, 0] = (1, 0, 0)
    pose[:3, 1] = (0, 0, -1)
    pose[:3, 2] = (0, 1, 0)
    pose[:3, 3] = (cam[0], cam[2], -cam[1])
    sx, sy = w / W, h / H
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dx, dy = (us - cx * sx) / (fx * sx), -(vs - cy * sy) / (fy * sy)
    floor_z, top_z = FLOOR_Y, FLOOR_Y + 0.75
    x, y = cam[0] + dx * (cam[2] - top_z), cam[1] + dy * (cam[2] - top_z)
    on_top = (np.abs(x - 2.5) < 0.6) & (np.abs(y - 2.0) < 0.4)
    depth = np.where(on_top, cam[2] - top_z, cam[2] - floor_z)
    Image.fromarray((depth * 1000).astype(np.uint16)).save(scan / f"depth_{index:05d}.png")
    Image.fromarray(np.full((h, w), 2, np.uint8)).save(scan / f"conf_{index:05d}.png")
    small = np.where(on_top[..., None], np.array([200, 30, 30], np.uint8),
                     np.array([120, 120, 120], np.uint8)).astype(np.uint8)
    Image.fromarray(small).resize((W, H), Image.Resampling.NEAREST).save(
        scan / f"frame_{index:05d}.jpg", quality=95)
    (scan / f"frame_{index:05d}.json").write_text(json.dumps({
        "frame_index": index, "intrinsics": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
        "cameraPoseARFrame": pose.reshape(-1).tolist()}))


def test_colours_come_from_the_capture_where_it_has_frames(scan):
    _frame_looking_down(scan)
    plan = roomplan.parse_room(scan)
    kind = build.template_kind(plan.objects["Table0"], plan.floor_z)
    regions = {"Table0": build.sample_regions(kind, build.template(kind, (1.2, 0.8, 0.75)))}
    palette = colours.estimate(scan, plan, regions=regions)
    assert palette.source == "frames" and palette.frames_used == 1
    top = palette.objects["Table0"].top
    assert top is not None and top[0] > 0.6 and top[1] < 0.3 and top[2] < 0.3
    assert palette.floor == pytest.approx((120 / 255,) * 3, abs=0.06)
    assert palette.views["Table0"] == [0]
    # an object no frame saw keeps its category colour
    assert palette.objects["Sofa0"].source == "category"
    assert palette.objects["Sofa0"].rgb == colours.category_rgb("sofa")


def test_colours_degrade_to_categories_without_frames(scan):
    plan = roomplan.parse_room(scan)
    palette = colours.estimate(scan, plan)
    assert palette.source == "category" and palette.frames_used == 0
    assert "no frames" in palette.note
    assert palette.objects["Chair0"].rgb == colours.category_rgb("chair")
    (scan / "frame_00000.json").write_text("{not json")
    (scan / "frame_00000.jpg").write_bytes(b"")
    (scan / "depth_00000.png").write_bytes(b"")
    broken = colours.estimate(scan, plan)                     # never raises
    assert broken.source == "category" and "unreadable" in broken.note


def test_the_quick_room_is_painted_with_the_capture(scan, tmp_path):
    _frame_looking_down(scan)
    path = write_quick_scene(scan, tmp_path / "out")
    report = json.loads((tmp_path / "out" / build.REPORT_NAME).read_text())
    assert report["colours"]["source"] == "frames"
    model = load_model(path)
    top = _geoms(model, "Table0_top")[0]
    rgba = model.mat_rgba[model.geom_matid[top]]
    assert rgba[0] > 0.6 and rgba[1] < 0.3


# ── the command line ─────────────────────────────────────────────────────────────────────────────


def test_cli_quickroom_writes_the_room_and_says_what_it_built(scan, tmp_path, capsys,
                                                              monkeypatch):
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "run"))
    args = cli._parser().parse_args(["quickroom", str(scan)])
    assert args.handler(args) == 0
    out = capsys.readouterr().out
    scene = tmp_path / "run" / "test_room" / "quick" / "scene.xml"
    assert scene.is_file() and (scene.parent / build.REPORT_NAME).is_file()
    assert "quick room" in out and "3 free, 1 hinged" in out and "Sink_Storage0" in out
    # by name, through $LR_SCANS_DIR, into --out
    monkeypatch.setenv("LR_SCANS_DIR", str(scan.parent))
    args = cli._parser().parse_args(["quickroom", "test_room", "--out", str(tmp_path / "o"),
                                     "--no-colours"])
    assert args.handler(args) == 0
    assert (tmp_path / "o" / "scene.xml").is_file()
    args = cli._parser().parse_args(["quickroom", str(tmp_path)])
    with pytest.raises(SystemExit, match="no RoomPlan room.usdz"):
        args.handler(args)


def test_cli_roomcall_builds_the_quick_room_for_a_scan_folder(scan, tmp_path, monkeypatch):
    from litereality_agent.room_ops.roomcall import server

    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "run"))
    served = []
    monkeypatch.setattr(cli.roomcall, "run", lambda args, **kw: served.append(args) or 0)
    args = cli._parser().parse_args(["roomcall", str(scan), "--no-open"])
    assert args.handler(args) == 0
    quick = tmp_path / "run" / "test_room" / "quick" / "scene.xml"
    assert quick.is_file()
    assert server.resolve_scene(served[-1].scene) == quick
    assert served[-1].label == "test_room"
    load_model(quick)
    # a bare scan name resolves through $LR_SCANS_DIR the same way
    monkeypatch.setenv("LR_SCANS_DIR", str(scan.parent))
    args = cli._parser().parse_args(["roomcall", "test_room", "--no-open", "--label", "Tea"])
    assert args.handler(args) == 0
    assert server.resolve_scene(served[-1].scene) == quick and served[-1].label == "Tea"


def test_cli_roomcall_prefers_the_pipelines_room_and_leaves_scenes_alone(scan, tmp_path,
                                                                         monkeypatch):
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "run"))
    served = []
    monkeypatch.setattr(cli.roomcall, "run", lambda args, **kw: served.append(args) or 0)
    full = tmp_path / "run" / "test_room" / "mujoco_seed" / "scene.xml"
    full.parent.mkdir(parents=True)
    full.write_text("<mujoco/>")
    args = cli._parser().parse_args(["roomcall", str(scan), "--no-open"])
    assert args.handler(args) == 0
    assert Path(served[-1].scene) == tmp_path / "run" / "test_room"
    assert not (tmp_path / "run" / "test_room" / "quick").exists()
    args = cli._parser().parse_args(["roomcall", str(full), "--no-open"])
    assert args.handler(args) == 0
    assert served[-1].scene == str(full) and served[-1].label is None
    args = cli._parser().parse_args(["roomcall", "--demo", "--no-open"])
    assert args.handler(args) == 0
    assert served[-1].scene is None


def test_cli_a_rescan_is_served_fresh_not_the_old_full_export(scan, tmp_path, monkeypatch,
                                                              capsys):
    """A re-scan under the same name is a different room: the run's older full export would put
    everyone in the room as it was. The quick room (the FILE, not the run folder, which would
    resolve back to the old export) is served, and the reason printed."""
    import os

    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "run"))
    served = []
    monkeypatch.setattr(cli.roomcall, "run", lambda args, **kw: served.append(args) or 0)
    full = tmp_path / "run" / "test_room" / "mujoco_seed" / "scene.xml"
    full.parent.mkdir(parents=True)
    full.write_text("<mujoco/>")
    usdz = scan / "room.usdz"
    os.utime(full, (usdz.stat().st_mtime - 3600,) * 2)            # exported an hour before
    args = cli._parser().parse_args(["roomcall", str(scan), "--no-open"])
    assert args.handler(args) == 0
    quick = tmp_path / "run" / "test_room" / "quick" / "scene.xml"
    assert served[-1].scene == str(quick) and quick.is_file()
    assert "the scan is newer than the pipeline's room" in capsys.readouterr().out
    os.utime(full, (usdz.stat().st_mtime + 3600,) * 2)            # ...and re-exported since
    args = cli._parser().parse_args(["roomcall", str(scan), "--no-open"])
    assert args.handler(args) == 0
    assert served[-1].scene == str(tmp_path / "run" / "test_room")
    assert "serving the pipeline's room" in capsys.readouterr().out


def test_cli_a_roomplan_subfolder_is_named_after_its_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("LITEREALITY_OUTPUT", str(tmp_path / "run"))
    write_roomplan_usdz(tmp_path / "scans" / "sub_room" / "roomplan" / "room.usdz")
    for target in (tmp_path / "scans" / "sub_room" / "roomplan",
                   tmp_path / "scans" / "sub_room" / "roomplan" / "room.usdz"):
        scan_dir, context = cli._quick_target(str(target))
        assert context.scan == "sub_room" and scan_dir == tmp_path / "scans" / "sub_room"


# ── the real scans (opt-in: `pytest -m scan`) ────────────────────────────────────────────────────


@pytest.mark.scan
@pytest.mark.parametrize("name", ["tea_room", "office_room"])
def test_real_scan_builds_in_seconds_and_settles(name, tmp_path):
    import time

    from litereality_agent.settings import load_settings

    folder = load_settings().resolved_scans_dir() / name
    if roomplan.find_room_usdz(folder) is None:
        pytest.skip(f"no {name} scan under $LR_SCANS_DIR")
    started = time.perf_counter()
    path = write_quick_scene(folder, tmp_path / name)
    assert time.perf_counter() - started < 10.0
    report = json.loads((tmp_path / name / build.REPORT_NAME).read_text())
    assert report["initial_overlaps"] == [] and report["pinned"] == []
    if any(folder.glob("frame_*.jpg")):          # its walls and floor wear the capture's photos
        photos = report["textures"]
        assert photos["source"] == "frames" and "floor" in photos["surfaces"]
        assert sum(k.startswith("Wall") for k in photos["surfaces"]) >= report["walls"] - 2
    if name == "tea_room":     # results/tea_room/physics_table.md
        expected = {"Chair0": 5.106, "Chair1": 5.617, "Chair2": 5.348, "Chair3": 5.617,
                    "Dishwasher0": 30.881}
        for body, kg in expected.items():
            assert report["bodies"][body]["mass_kg"] == pytest.approx(kg, abs=0.002)
    sim = RoomSim(load_model(path))
    for _ in range(100):
        sim.step_realtime(0.1)
    assert np.all(np.isfinite(sim.data.qpos)) and sim.unstable_resets == 0
