"""tea_room's tucked chairs: Chair1 and Chair3 started 7.4 cm inside Table0; Chair3 was thrown 1 m.

RoomPlan records a chair pushed under a table as a box overlapping the table's box, and the room
fits each asset into its own box, so the two start interlocked wherever their colliders do not
leave each other room: a chair's seat-to-backrest hull fills the wedge above its seat, which is
under the table top. Three things made that a launch rather than an untidy start:

  * the object compiler collided a recipe-built link as ONE mesh; when CoACD died on it (it
    segfaults on exactly these carcasses) the whole table became one hull, filling the space under
    its top where tucked chairs sit;
  * the exporter wrote the overlap into the scene as it was, so `initial_overlaps` failed the gate;
  * the shake's relax lifted the chair out of the "fixture" — up, deeper into the table top — 19 cm,
    until the solver threw it.

All synthetic and Blender-free; MuJoCo does the measuring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
mujoco = pytest.importorskip("mujoco")

from sim_fixtures import box, write_glb  # noqa: E402

from litereality_agent.models.object_generation.sim import properties  # noqa: E402
from litereality_agent.room_ops.export import mujoco_scene, mujoco_shake  # noqa: E402

TOP_UNDERSIDE = 0.685


def _table_parts() -> dict:
    parts = {"Top": box((0.9, 0.8, 0.04), (0, 0, 0.705))}
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        parts[f"Leg{i}"] = box((0.05, 0.05, 0.685), (sx * 0.4, sy * 0.35, 0.3425))
    return parts


def _inside_any(colliders, out_dir: Path, points) -> np.ndarray:
    from scipy.spatial import Delaunay

    hit = np.zeros(len(points), dtype=bool)
    for c in colliders:
        hull = trimesh.load(out_dir / c.file, force="mesh")
        hit |= Delaunay(np.asarray(hull.vertices)).find_simplex(points) >= 0
    return hit


UNDER_THE_TOP = np.array([[0.0, 0.0, 0.45], [0.2, -0.3, 0.6], [0.0, -0.39, 0.66]])


def test_a_tables_colliders_are_its_parts_and_leave_the_space_under_its_top_free(
        tmp_path, monkeypatch):
    pytest.importorskip("scipy")

    def no_coacd(*_a, **_k):
        raise AssertionError("a table built from convex parts needs no decomposition")

    monkeypatch.setattr(properties, "_decompose", no_coacd)
    glb = write_glb(tmp_path / "Table0.glb", _table_parts())
    model = properties.build_model(glb, tmp_path / "sim")
    (base,) = model.links
    assert len(base.colliders) == 5
    assert not _inside_any(base.colliders, tmp_path / "sim", UNDER_THE_TOP).any()


def test_when_coacd_gives_nothing_a_concave_link_keeps_its_parts_not_one_solid_hull(
        tmp_path, monkeypatch):
    """The fallback used to be one hull of the whole link: a table as a solid block."""
    pytest.importorskip("scipy")
    monkeypatch.setattr(properties, "_decompose", lambda *_a, **_k: [])
    # A ring foot round one leg is a concave part, so the part-by-part shortcut does not apply and
    # the link really goes to (a dead) CoACD.
    parts = _table_parts()
    ring = trimesh.creation.annulus(r_min=0.05, r_max=0.09, height=0.04)
    ring.apply_translation((-0.4, -0.35, 0.02))
    parts["Foot"] = ring
    mesh = trimesh.util.concatenate(list(parts.values()))
    notes: list[str] = []
    colliders = properties._colliders(mesh, "Table0_base_link", tmp_path, 19.0, True, notes)
    assert len(colliders) >= 5
    assert not _inside_any(colliders, tmp_path, UNDER_THE_TOP).any()
    assert notes and "decomposition produced nothing" in notes[0]


def test_a_concave_single_mesh_still_goes_to_coacd(tmp_path, monkeypatch):
    """A generated chair is one mesh: the part-by-part shortcut must not swallow it."""
    calls = []

    def fake(mesh, budget):
        calls.append(budget)
        return [(np.asarray(box((0.2, 0.2, 0.2), (0, 0, 0)).vertices),
                 np.asarray(box((0.2, 0.2, 0.2), (0, 0, 0)).faces))]

    monkeypatch.setattr(properties, "_decompose", fake)
    ring = trimesh.creation.annulus(r_min=0.25, r_max=0.35, height=0.4)
    properties._colliders(ring, "Ring0_base_link", tmp_path, 9.0, True)
    assert calls == [8]


# ------------------------------------------------------------------ the scene


def _wedge(depth=0.50, width=0.46, seat=0.45, top=0.807, back=0.05):
    """The hull CoACD makes when it merges a seat with its backrest: it fills the space above the
    seat, which is exactly the space under a table top when the chair is tucked in."""
    pts = []
    for y in (-width / 2, width / 2):
        pts += [(0, y, seat - 0.04), (depth, y, seat - 0.04), (depth, y, seat), (back, y, top),
                (0, y, top)]
    return trimesh.convex.convex_hull(np.array(pts))


def _scene(root: Path, tuck: float) -> Path:
    """A room body with a floor and a far wall, a static Table0, and a free Chair1 whose front
    `tuck` metres are under the table (the chair's back is at x=0, its front at x=0.5)."""
    meshes = root / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    _wedge().export(meshes / "Chair1_c0.obj")
    edge = 0.50 - tuck
    legs = "".join(
        f'<geom type="box" size="0.025 0.025 0.3425" pos="{x} {y} 0.3425"/>'
        for x in (-0.4, 0.4) for y in (-0.35, 0.35))
    chair_legs = "".join(
        f'<geom type="box" size="0.02 0.02 0.205" pos="{x} {y} 0.205"/>'
        for x in (0.03, 0.47) for y in (-0.2, 0.2))
    xml = root / "scene.xml"
    xml.write_text(f"""<mujoco><compiler meshdir="meshes" autolimits="true"/>
<option timestep="0.002" integrator="implicitfast"/>
<default><geom condim="4" friction="0.55 0.03 0.005" solref="0.01 1" solimp="0.9 0.95 0.001"/>
</default>
<asset><mesh name="Chair1_c0" file="Chair1_c0.obj"/></asset>
<worldbody>
  <body name="room">
    <geom name="floor" type="box" size="3 3 0.05" pos="0 0 -0.05"/>
    <geom name="wall_far" type="box" size="0.05 2 1.3" pos="2.9 0 1.3"/>
    <body name="Table0" pos="{edge + 0.45:.4f} 0 0">
      <geom type="box" size="0.45 0.4 0.02" pos="0 0 0.705"/>{legs}
    </body>
  </body>
  <body name="Chair1" pos="0 0 0"><freejoint name="Chair1_free"/>
    <inertial pos="0.25 0 0.45" mass="5.6" diaginertia="0.3 0.3 0.2"/>
    <geom type="mesh" mesh="Chair1_c0"/>{chair_legs}
  </body>
</worldbody></mujoco>""", encoding="utf-8")
    return xml


def _deepest(model, data) -> float:
    mujoco.mj_forward(model, data)
    return min([float(data.contact[i].dist) for i in range(data.ncon)] + [0.0])


def _settle(xml: Path, shift=None, seconds=1.0, relax_like_the_shake=False):
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    if shift is not None:
        data.qpos[0:3] += shift
    start_depth = _deepest(model, data)
    start = data.qpos[0:3].copy()
    lifted = {}
    if relax_like_the_shake:
        lifted = mujoco_shake._lift_out_of_walls(model, data)
        gravity = model.opt.gravity.copy()
        model.opt.gravity[:] = 0.0
        for _ in range(int(0.6 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            data.qvel[:] *= 0.55
        model.opt.gravity[:] = gravity
        data.qvel[:] = 0.0
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return start_depth, float(np.linalg.norm(data.qpos[0:3] - start)), lifted


def test_a_tucked_chair_starts_inside_the_table(tmp_path):
    depth, _moved, _lifted = _settle(_scene(tmp_path, 0.36), seconds=0.0)
    assert depth < -0.03


def test_the_export_slides_a_tucked_chair_out_along_the_floor(tmp_path):
    xml = _scene(tmp_path, 0.36)
    moved = mujoco_scene._separate_initial_overlaps(xml, {"Chair1"})
    shift = moved["Chair1"]
    assert abs(shift[2]) < 1e-9                       # along the floor: not lifted, not sunk
    assert shift[0] < 0                               # out from under the table, not further in
    assert 0.01 < float(np.linalg.norm(shift)) < 0.15
    depth, travelled, _ = _settle(xml, shift)
    assert depth >= -mujoco_scene.SEPARATION_TOL      # the scene now loads clean
    assert travelled < 0.01                           # and the chair stays where it was put


def test_a_chair_that_merely_touches_is_not_moved(tmp_path):
    assert mujoco_scene._separate_initial_overlaps(_scene(tmp_path, 0.20), {"Chair1"}) == {}


def test_only_the_bodies_asked_for_are_moved(tmp_path):
    assert mujoco_scene._separate_initial_overlaps(_scene(tmp_path, 0.36), {"Table0"}) == {}


def test_a_body_no_short_move_can_free_is_left_where_it_is(tmp_path):
    xml = _scene(tmp_path, 0.36)
    text = xml.read_text().replace(
        '<geom name="floor"', '<geom name="vault" type="box" size="1.5 1.5 1.5" pos="0.25 0 1"/>'
        '<geom name="floor"')
    xml.write_text(text)
    assert mujoco_scene._separate_initial_overlaps(xml, {"Chair1"}) == {}


def test_the_moves_land_on_the_bodies_in_the_scene_xml():
    import xml.etree.ElementTree as ET

    world = ET.fromstring('<worldbody><body name="Chair1" pos="1 2 0.4"/>'
                          '<body name="Chair2" pos="0 0 0"/></worldbody>')
    mujoco_scene._apply_moves(world, {"Chair1": np.array([-0.07, 0.01, 0.0])})
    assert world.find("body[@name='Chair1']").get("pos") == "0.9300 2.0100 0.4000"
    assert world.find("body[@name='Chair2']").get("pos") == "0 0 0"


def test_the_shake_no_longer_lifts_a_chair_deeper_into_a_table_top(tmp_path):
    """The tea_room launch: lifting is right for a mug sunk into a shelf from above, and wrong for a
    chair under a table, where each lift drove it further into the top until it was thrown."""
    xml = _scene(tmp_path, 0.36)
    _depth, travelled, lifted = _settle(xml, relax_like_the_shake=True)
    assert "Chair1" not in lifted
    assert travelled < 0.05


def test_the_shake_still_lifts_a_body_sunk_into_a_fixture_from_above(tmp_path):
    xml = _scene(tmp_path, 0.0)
    # A small box authored 3 cm into the table top, the way a mug sits inside a shelf's hull.
    text = xml.read_text().replace(
        "</worldbody>",
        '<body name="Mug0" pos="1.2 0 0.735"><freejoint/>'
        '<geom type="box" size="0.04 0.04 0.05" mass="0.3"/></body></worldbody>')
    xml.write_text(text)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    lifted = mujoco_shake._lift_out_of_walls(model, data)
    assert lifted.get("Mug0", 0.0) > 0.02
