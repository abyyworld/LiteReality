"""tea_room's door leaf simulated at 128.7 kg — and the cabinet doors at a third of a kilo.

The leaf's own sidecar was right (~27 kg: an authored density times its own volume). The room put
it wrong. A door is fitted into its wall opening, and when no node name survives the room build
the placement is recovered from bounding boxes AND A YAW. Openings live in the SHELL's `openings`
— a wall plus an offset — not its `objects`, so the exporter handed the door yaw 0 while tea_room's
Wall10 runs at 31 degrees. Rotating by the wrong yaw makes the door's own bounding box the
axis-aligned box AROUND the angled door: its 15 cm depth became 56 cm, the volume ratio 4.7, the
leaf's mass was scaled by the same, and its collider became a slab laid across the doorway. The old
proof compared bounding boxes, which a bounding-box fit matches by construction, so nothing noticed.

All synthetic and Blender-free: a door built from boxes, written as a real GLB with the
articulation extras Blender's exporter writes, compiled by the real compiler.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from sim_fixtures import (  # noqa: E402
    LEAF_DENSITY,
    LEAF_H,
    LEAF_T,
    LEAF_W,
    box,
    door_parts,
    placed_like_build_room,
    write_door,
    write_glb,
)

from litereality_agent.models.object_generation.sim import properties  # noqa: E402
from litereality_agent.room_ops.export import mujoco_scene, sim_assets  # noqa: E402

WALL_YAW = math.radians(31.0)          # tea_room's Wall10, from the RoomPlan box points
NAME = "Wall10_Door_0"


def _fit_scale(whole) -> tuple[float, float, float]:
    """`build_room._place_in_box` for an opening: 1.10x the hole, depth stretched through a wall."""
    aw, ad, ah = whole.extents
    sx = 0.767 * 1.10 / aw                     # tea_room's Door0 is 0.767 m wide, 1.98 m tall
    return sx, max(sx, (0.10 + 0.05) / ad), 1.98 * 1.10 / ah


@pytest.fixture
def door(tmp_path):
    """The door's GLB, its compiled sidecar, and the door as the room placed it in Wall10."""
    room = tmp_path / "room"
    sim_dir = room / "Objects" / "Procedural" / NAME / "sim"
    glb = write_door(tmp_path / "room_preview" / "Object" / f"{NAME}.glb")
    model = properties.build_model(glb, sim_dir)
    properties.write_model(model, sim_dir)
    whole = trimesh.util.concatenate(list(door_parts()[0].values()))
    truth = placed_like_build_room(whole, WALL_YAW, _fit_scale(whole), (1.55, -0.33, 1.09))
    placed = whole.copy()
    placed.apply_transform(truth)
    return {"room": room, "glb": glb, "model": model, "truth": truth, "placed": placed,
            "sidecar": sim_assets.load(sim_dir / f"{NAME}.physics.json")}


def test_the_leaf_is_compiled_at_its_authored_density_times_its_own_volume(door):
    leaf = next(link for link in door["model"].links if link.name == "Door_leaf")
    handle_volume = 0.12 * 0.06 * 0.03
    assert leaf.mass_source == "authored_density"
    assert leaf.mass == pytest.approx(LEAF_DENSITY * (LEAF_W * LEAF_H * LEAF_T + handle_volume),
                                      rel=0.02)
    assert 20.0 <= leaf.mass <= 40.0


def test_a_door_in_an_angled_wall_placed_with_yaw_zero_gets_four_times_too_heavy(door):
    """The bug itself, pinned so it cannot quietly come back through `placement_from_bbox`."""
    model = door["sidecar"]
    wrong = sim_assets.place(model, sim_assets.placement_from_bbox(model, door["placed"], 0.0))
    sidecar_leaf = model.link("Door_leaf").mass
    assert wrong.links["Door_leaf"].mass > 3.0 * sidecar_leaf
    # ...and the geometric proof is what catches it, where the bounds comparison could not.
    residual = sim_assets.check_placement(model, sim_assets.placement_from_bbox(
        model, door["placed"], 0.0), door["placed"])
    assert residual > 0.15


@pytest.mark.parametrize("stated", [0.0, None, WALL_YAW, WALL_YAW + math.pi / 2])
def test_the_fit_recovers_the_true_placement_whatever_yaw_it_is_told(door, stated):
    model = door["sidecar"]
    fit = sim_assets.fit_placement(model, door["placed"], [stated])
    assert fit is not None
    assert math.cos(fit.yaw - WALL_YAW) == pytest.approx(1.0, abs=1e-6)   # not the half turn
    as_zup = sim_assets._A4 @ fit.transform @ sim_assets._A4_INV
    assert np.allclose(as_zup, door["truth"], atol=1e-6)

    placed = sim_assets.place(model, fit.transform)
    leaf = placed.links["Door_leaf"].mass
    assert leaf == pytest.approx(model.link("Door_leaf").mass
                                 * abs(np.linalg.det(door["truth"][:3, :3])), rel=1e-6)
    assert 20.0 <= leaf <= 40.0
    # The leaf's collider lies ALONG the wall — as wide as the door, as thin as a leaf plus its
    # handle — and not as a slab across the doorway.
    collider = trimesh.util.concatenate(placed.links["Door_leaf"].colliders)
    _, extents = trimesh.bounds.oriented_bounds_2D(np.asarray(collider.vertices)[:, :2])
    assert max(extents) == pytest.approx(0.767, rel=0.05)
    assert min(extents) < 0.15


def test_an_openings_shell_yaw_is_the_direction_of_its_wall():
    shell = {
        "walls": {"Wall10": {"start": [0.0, 0.0],
                             "end": [math.cos(WALL_YAW) * 3, math.sin(WALL_YAW) * 3]}},
        "openings": {"Door0": {"wall": "Wall10", "type": "door", "offset": 1.2, "width": 0.77,
                               "height": 1.98, "sill": 0.0}},
        "objects": {"Table0": {"yaw": 148.965}},
    }
    assert mujoco_scene._shell_yaw(shell, "Door0") == pytest.approx(WALL_YAW)
    assert mujoco_scene._shell_yaw(shell, "Table0") == pytest.approx(math.radians(148.965))
    assert mujoco_scene._shell_yaw(shell, "Nowhere0") is None


@pytest.mark.parametrize("yaw", [0.0, WALL_YAW, None])
def test_the_room_export_gives_the_door_leaf_a_plausible_mass(door, yaw):
    """Through `_object_physics` itself, with no node name to match — the tea_room path."""
    room_scene = trimesh.Scene()
    room_scene.add_geometry(door["placed"], node_name="Door_leaf_3f9a2c", geom_name="g")
    report: dict = {}
    placed = mujoco_scene._object_physics(
        door["room"], {"source_glb": f"{NAME}.glb", "handle": "Door0"}, room_scene,
        ["Door_leaf_3f9a2c", "Door_leaf_77aa01"], door["placed"], yaw, report)
    assert placed is not None, report
    assert report["placed_by_geometry"] == [NAME]
    assert 20.0 <= placed.links["Door_leaf"].mass <= 40.0
    assert ("yaw_corrected" in report) == (yaw == 0.0)


# ------------------------------------------------- the same bug in the other parts


def _cabinet(path):
    """A 0.6 x 0.5 x 0.8 m carcass of 18 mm boards with one hinged door, no authored physics."""
    t = 0.018
    parts = {
        "Carcass_L": box((t, 0.5, 0.8), (-0.3 + t / 2, 0, 0.4)),
        "Carcass_R": box((t, 0.5, 0.8), (0.3 - t / 2, 0, 0.4)),
        "Carcass_Top": box((0.6 - 2 * t, 0.5, t), (0, 0, 0.8 - t / 2)),
        "Carcass_Bottom": box((0.6 - 2 * t, 0.5, t), (0, 0, t / 2)),
        "Carcass_Back": box((0.6 - 2 * t, t, 0.8 - 2 * t), (0, 0.25 - t / 2, 0.4)),
        "door_l": box((0.596, t, 0.796), (0, -0.25 - t / 2, 0.4)),
    }
    extras = {"door_l": {"articulation_type": "revolute", "articulation_axis": [0, 0, 1],
                         "limit_min": 0.0, "limit_max": 1.6, "joint_origin": [-0.298, -0.25, 0]}}
    return write_glb(path, parts, extras), parts


def test_a_cabinet_door_gets_its_share_of_the_mass_by_material_not_by_hull(tmp_path):
    """tea_room's sink cabinet doors were 0.32 and 0.53 kg: the carcass's HULL — the whole empty
    cabinet — took the mass share its boards never had."""
    glb, parts = _cabinet(tmp_path / "Storage9.glb")
    model = properties.build_model(glb, tmp_path / "sim")
    by_name = {link.name: link for link in model.links}
    door_volume = float(parts["door_l"].volume)
    carcass_volume = sum(float(m.volume) for n, m in parts.items() if n != "door_l")
    total = sum(link.mass for link in model.links)

    assert by_name["door_l"].mass == pytest.approx(
        total * door_volume / (door_volume + carcass_volume), rel=0.05)
    hull_share = total * door_volume / (door_volume + 0.6 * 0.5 * 0.8)
    assert by_name["door_l"].mass > 3.0 * hull_share
    assert by_name["door_l"].mass > 1.5            # a real 0.6 x 0.8 m door is 3-6 kg


def test_material_volume_is_bounded_and_survives_uv_seams():
    slab = box((0.8, 0.04, 2.0), (0, 0, 1.0))
    # A glTF primitive splits vertices at every UV seam: the same box as a soup of loose faces.
    soup = trimesh.Trimesh(np.asarray(slab.triangles).reshape(-1, 3),
                           np.arange(len(slab.faces) * 3).reshape(-1, 3), process=False)
    assert properties._material_volume(soup) == pytest.approx(slab.volume, rel=1e-6)
    # Raised panels laid over the slab: the overlap must not be counted twice beyond the hull.
    panels = trimesh.util.concatenate([slab, box((0.6, 0.05, 0.8), (0, 0, 0.6)),
                                       box((0.6, 0.05, 0.8), (0, 0, 1.5))])
    assert properties._material_volume(panels) <= panels.convex_hull.volume + 1e-9
    # A pane modelled as a single quad has no material to speak of.
    quad = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 1, 2], [0, 2, 3]])
    assert properties._material_volume(quad) < 1e-5
