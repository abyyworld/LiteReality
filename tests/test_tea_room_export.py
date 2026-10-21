"""A tea_room-shaped room through the REAL MuJoCo export: door in a 31-degree wall, a table, and a
chair tucked under it — the three objects behind tea_room's 128.7 kg door leaf and its chairs
starting 7 cm inside the table.

Built the way the pipeline builds it — each object a GLB compiled by the object compiler into its
`sim/` sidecar, placed into a `Room.glb` the way `build_room` fits assets into RoomPlan boxes,
with node names that do not survive (as on tea_room, where all three were `placed_by_geometry`) —
and exported by `mujoco_scene.export`. Blender-free.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
mujoco = pytest.importorskip("mujoco")

from sim_fixtures import (  # noqa: E402
    ZUP_TO_YUP,
    box,
    door_parts,
    placed_like_build_room,
    write_glb,
)

from litereality_agent.models.object_generation.sim import properties  # noqa: E402
from litereality_agent.room_ops.export import mujoco_scene  # noqa: E402

WALL_YAW = math.radians(31.0)


def _table() -> tuple[dict, dict]:
    parts = {"Top": box((0.9, 0.8, 0.04), (0, 0, 0.705))}
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        parts[f"Leg{i}"] = box((0.05, 0.05, 0.685), (sx * 0.4, sy * 0.35, 0.3425))
    return parts, {}


def _chair() -> tuple[dict, dict]:
    """Front faces -Y. The seat and backrest are ONE convex piece, the way CoACD hands back a
    generated chair when it merges them: it fills the wedge above the seat."""
    pts = []
    for x in (-0.23, 0.23):
        pts += [(x, -0.25, 0.41), (x, -0.25, 0.45), (x, 0.20, 0.807), (x, 0.25, 0.807),
                (x, 0.25, 0.41)]
    parts = {"Shell": trimesh.convex.convex_hull(np.array(pts))}
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        parts[f"Leg{i}"] = box((0.04, 0.04, 0.41), (sx * 0.2, sy * 0.21, 0.205))
    return parts, {}


# handle -> (object, category, parts builder, kind, yaw, box centre, fitted scale)
TABLE_Y = 0.0
OBJECTS = {
    "Door0": ("Wall10_Door_0", "door", door_parts, "Procedural", WALL_YAW, (-1.0, -2.0, None),
              None),
    "Table0": ("Table0", "table", _table, "Procedural", 0.0, (1.0, TABLE_Y, None), (1, 1, 1)),
    # tucked 36 cm: its front edge is 0.36 m inside the table's far edge, its backrest outside
    "Chair1": ("ChairCluster1", "chair", _chair, "Static", 0.0, (1.0, TABLE_Y + 0.4 + 0.25 - 0.36,
                                                                  None), (1, 1, 1)),
}


def _room(tmp_path: Path) -> Path:
    scene_dir = tmp_path / "tea_room_like"
    room, preview = scene_dir / "room", scene_dir / "room_preview"
    (preview / "Object").mkdir(parents=True)
    room_scene = trimesh.Scene()
    records = []
    for handle, (name, category, build, kind, yaw, centre, scale) in OBJECTS.items():
        parts, extras = build()
        glb = write_glb(preview / "Object" / f"{name}.glb", parts, extras)
        sim_dir = room / "Objects" / kind / name / "sim"
        properties.write_model(properties.build_model(glb, sim_dir, category=category), sim_dir)

        whole = trimesh.util.concatenate(list(parts.values()))
        if scale is None:                     # an opening: 1.10x its hole, depth through the wall
            aw, ad, ah = whole.extents
            sx = 0.767 * 1.10 / aw
            scale = (sx, max(sx, 0.15 / ad), 1.98 * 1.10 / ah)
        height = whole.extents[2] * scale[2]
        transform = placed_like_build_room(whole, yaw, scale, (centre[0], centre[1], height / 2))
        room_scene.graph.update(frame_to=handle, frame_from=room_scene.graph.base_frame)
        placed = []
        for node, mesh in parts.items():
            world = mesh.copy()
            world.apply_transform(transform)
            placed.append(world)
            yup = world.copy()
            yup.vertices = np.asarray(yup.vertices) @ ZUP_TO_YUP.T
            # A renamed node: nothing to match on, so the export has to place it by geometry.
            room_scene.add_geometry(yup, node_name=f"{node}_{abs(hash(handle)) % 16**6:06x}",
                                    geom_name=f"{handle}_{node}", parent_node_name=handle)
        lo, hi = trimesh.util.concatenate(placed).bounds
        records.append({"id": handle, "handle": handle, "category": category,
                        "center": ((lo + hi) / 2).round(4).tolist(), "bbox_min": lo.round(4).tolist(),
                        "bbox_max": hi.round(4).tolist(), "size": (hi - lo).round(4).tolist(),
                        "source_glb": f"{name}.glb", "rests_on": None, "attached_to": None})
    room_scene.export(preview / "Room.glb")

    wall_start = np.array([-1.0, -2.0]) - 1.5 * np.array([math.cos(WALL_YAW), math.sin(WALL_YAW)])
    wall_end = wall_start + 3.0 * np.array([math.cos(WALL_YAW), math.sin(WALL_YAW)])
    shell = {
        "floor_z": 0.0, "ceiling_z": 2.7,
        "walls": {"Wall10": {"start": wall_start.round(4).tolist(),
                             "end": wall_end.round(4).tolist(), "thickness": 0.1}},
        "openings": {"Door0": {"wall": "Wall10", "type": "door", "offset": 1.5, "width": 0.767,
                               "height": 1.98, "sill": 0.0}},
        "objects": {"Table0": {"category": "table", "center": records[1]["center"],
                               "size": records[1]["size"], "yaw": 0.0},
                    "Chair1": {"category": "chair", "center": records[2]["center"],
                               "size": records[2]["size"], "yaw": 0.0}},
        "floor": {"verts": [[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]],
                  "faces": [[0, 1, 2, 3]]},
    }
    room.mkdir(parents=True, exist_ok=True)
    (room / "Room.py").write_text(f"SHELL = {json.dumps(shell)}\n", encoding="utf-8")
    (preview / "room_layout.json").write_text(json.dumps({
        "scene": "tea_room_like", "bounds": {"min": [-3, -3, 0], "max": [3, 3, 2.7]},
        "objects": records}), encoding="utf-8")
    return room


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    room = _room(tmp_path_factory.mktemp("tea"))
    xml = mujoco_scene.export(room)
    return xml, json.loads((xml.parent / "export_report.json").read_text())


def test_every_object_reads_its_own_physics_and_is_placed_by_geometry(exported):
    _xml, report = exported
    assert sorted(report["from_sidecar"]) == ["ChairCluster1", "Table0", "Wall10_Door_0"]
    assert sorted(report["placed_by_geometry"]) == ["ChairCluster1", "Table0", "Wall10_Door_0"]
    assert not report.get("placement_rejected")


def test_the_door_leaf_weighs_what_a_door_leaf_weighs(exported):
    xml, _report = exported
    model = mujoco.MjModel.from_xml_path(str(xml))
    leaf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Door0_Door_leaf")
    assert leaf >= 0
    assert 20.0 <= float(model.body_mass[leaf]) <= 40.0          # tea_room: 128.7


def test_the_tucked_chair_starts_clear_of_the_table_and_stays_put(exported):
    xml, report = exported
    assert report["loads_clean"] is True, report.get("initial_overlaps")
    moved = report["separated"]["Chair1"]
    assert 1.0 < moved["moved_cm"] < 20.0
    assert abs(moved["by"][2]) < 1e-6                             # along the floor
    assert moved["by"][1] > 0                                     # out from under the table
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    chair = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Chair1")
    start = data.xpos[chair].copy()
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert float(np.linalg.norm(data.xpos[chair] - start)) < 0.02  # tea_room's Chair3: 1 m


def test_the_chair_body_in_the_xml_was_moved_by_the_reported_amount(exported):
    xml, report = exported
    body = ET.parse(xml).getroot().find(".//worldbody/body[@name='Chair1']")
    centre = next(o["center"] for o in json.loads(
        (Path(xml).parents[1] / "room_preview" / "room_layout.json").read_text())["objects"]
        if o["handle"] == "Chair1")
    pos = np.asarray([float(v) for v in body.get("pos").split()])
    assert np.allclose(pos - np.asarray(centre), report["separated"]["Chair1"]["by"], atol=2e-4)
