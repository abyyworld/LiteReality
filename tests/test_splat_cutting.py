"""The cut of a room splat into the room and its bodies: corrections fixed once, the room shell's
walls kept with the room, mask votes per body, and the moved scene kept in step with the cut
(pipeline/splatroom/split)."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_splat import (  # noqa: E402
    SCENE,
    _grid,
    _red_masks,
    _room_splats,
    _scan_above_box,
    _splats_at,
)

from litereality_agent.models.splat import rgbd  # noqa: E402
from litereality_agent.pipeline.splatroom.split import (  # noqa: E402
    corrected_scene,
    left_behind,
    split,
)


def test_an_empty_set_of_corrections_is_kept_and_never_snaps_again():
    model = mujoco.MjModel.from_xml_string(SCENE.format(x=1.0))
    first = split(_room_splats(1.0), model)
    assert first.corrections == {}                       # the box was where the splat is
    fixed: dict = {}
    later = split(_room_splats(1.2), model, corrections=fixed)
    assert later.corrections == {} and fixed == {}
    # Cut where the physics is (x 1.0), not where a fresh snap would have put it.
    centre = later.bodies["Box0"].positions.mean(0)
    assert centre[0] > 0.06


WALLED = """<mujoco>
  <worldbody>
    <geom name="floor" type="box" pos="0 0 -0.05" size="5 5 0.05"/>
    <geom name="wall" type="box" pos="0 2 1" size="3 0.05 1"/>
    <geom name="photo" type="box" pos="0 0 1" size="3 3 1" contype="0" conaffinity="0"
          group="2"/>
    <body name="Box0" pos="1 1.74 0.25">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.25" group="3"/>
    </body>
  </worldbody>
</mujoco>"""


def test_a_body_against_a_wall_leaves_the_wall_but_keeps_its_feet_on_the_floor():
    model = mujoco.MjModel.from_xml_string(WALLED)
    box = _grid((0.8, 1.54, 0.02), (1.2, 1.94, 0.5))
    wall = _grid((0.0, 1.95, 0.0), (2.0, 1.95, 1.0))
    s = _splats_at(np.concatenate([box, wall]))
    parts = split(s, model, snap_bodies=False, fill=False)
    world_y = parts.bodies["Box0"].positions[:, 1] + 1.74
    assert world_y.max() < 1.945                            # no wall in the chair
    assert len(parts.bodies["Box0"]) >= 0.85 * len(box)
    # The floor slab is no wall: the splats just above it stay with the body.
    assert parts.bodies["Box0"].positions[:, 2].min() < -0.25 + 0.03


TWO = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="A" pos="1 0 0.25">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.25" group="3"/>
    </body>
    <body name="B" pos="{bx} 0 0.25">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.25" group="3"/>
    </body>
  </worldbody>
</mujoco>"""


def _masks_for_a(tmp_path, model, *, b_box=None):
    """Two photos from above A (A red), planned for A only; with `b_box`, B gets that prompt box
    in them too (and so an empty mask: nothing red there)."""
    from litereality_agent.pipeline.splatroom import masks as object_masks

    scan = tmp_path / "scan"
    scan.mkdir()
    _scan_above_box(scan, box_x=1.0, index=0, cam_dx=-0.1)
    _scan_above_box(scan, box_x=1.0, index=1, cam_dx=0.1)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    views = object_masks.plan_views(rgbd.load_frames(scan), model, data,
                                    {"A": model.body("A").id})
    assert len(views) == 2
    for view in views.values():
        if b_box is not None:
            W, H = view.size
            view.boxes["A"] = [0.0, 0.0, float(W), float(H)]    # A's grown prompt box spans B
            view.boxes["B"] = b_box
    return object_masks.request(views, _red_masks)


def test_a_splat_outside_a_neighbours_mask_stays_with_its_own_body(tmp_path):
    model = mujoco.MjModel.from_xml_string(TWO.format(bx=1.6))
    got = _masks_for_a(tmp_path, model, b_box=[0.0, 0.0, 5.0, 5.0])
    assert {n for per in got.masks.values() for n in per} == {"A", "B"}
    top_a = _grid((0.82, -0.18, 0.5), (1.18, 0.18, 0.5), step=0.03)
    top_b = _grid((1.42, -0.18, 0.5), (1.78, 0.18, 0.5), step=0.03)
    s = _splats_at(np.concatenate([top_a, top_b]))
    parts = split(s, model, masks=got, snap_bodies=False, fill=False)
    assert parts.counts["A"] >= 0.9 * len(top_a)
    # Only A's photo saw B's top, outside A's mask: B's own masks never said it is not B.
    assert parts.counts.get("B", 0) == len(top_b)


def test_a_free_body_without_a_mask_is_still_snapped_when_others_have_masks(tmp_path):
    model = mujoco.MjModel.from_xml_string(TWO.format(bx=-0.8))     # B really is at -1.0
    got = _masks_for_a(tmp_path, model)
    assert {n for per in got.masks.values() for n in per} == {"A"}
    a = _grid((0.82, -0.18, 0.5), (1.18, 0.18, 0.5), step=0.015)
    b = _grid((-1.2, -0.2, 0.02), (-0.8, 0.2, 0.5))
    floor = _grid((-2.0, -1.0, 0.0), (2.0, 1.0, 0.0), step=0.05)
    parts = split(_splats_at(np.concatenate([a, b, floor])), model, masks=got, fill=False)
    fix = parts.corrections["B"]
    assert abs(fix["dx"] + 0.2) <= 0.021 and abs(fix["dy"]) <= 0.021
    assert "from" not in fix                                 # snapped, not voted
    assert np.abs(parts.bodies["B"].positions.mean(0)[:2]).max() < 0.06


def test_a_cut_that_moves_nothing_removes_the_scene_an_earlier_one_moved(tmp_path):
    xml = tmp_path / "scene.xml"
    xml.write_text(SCENE.format(x=0.8))
    stale = corrected_scene(xml, {"Box0": {"dx": 0.2, "dy": 0.0, "dyaw_deg": 0.0}})
    assert stale is not None and stale.is_file()
    assert corrected_scene(xml, {}) is None
    assert not stale.exists() and xml.is_file()
    stale = corrected_scene(xml, {"Box0": {"dx": 0.2, "dy": 0.0, "dyaw_deg": 0.0}})
    assert corrected_scene(xml, {"NotABody": {"dx": 0.1, "dy": 0.0, "dyaw_deg": 0.0}}) is None
    assert not stale.exists()


def test_a_move_into_another_movable_body_is_scaled_back():
    import mujoco

    from litereality_agent.pipeline.splatroom.split import _physical

    xml = """<mujoco><worldbody><geom type="plane" size="5 5 0.1"/>
      <body name="A" pos="0 0 0.25"><freejoint/><geom type="box" size="0.2 0.2 0.25" group="3"/></body>
      <body name="B" pos="0.6 0 0.25"><freejoint/><geom type="box" size="0.2 0.2 0.25" group="3"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    into = {"A": {"dx": 0.4, "dy": 0.0, "dyaw_deg": 0.0}}        # 0.2 m into B
    kept = _physical(model, into)
    assert "A" not in kept or kept["A"]["dx"] <= 0.2 + 1e-9         # never deeper into B
    free = _physical(model, {"A": {"dx": -0.3, "dy": 0.0, "dyaw_deg": 0.0}})
    assert free["A"]["dx"] == -0.3 and "scaled" not in free["A"]
    both = _physical(model, {"A": {"dx": 0.15, "dy": 0.0, "dyaw_deg": 0.0},
                             "B": {"dx": -0.15, "dy": 0.0, "dyaw_deg": 0.0}})
    gap = 0.6 + both.get("B", {}).get("dx", 0) - both.get("A", {}).get("dx", 0)
    assert gap >= 0.4 - 0.005                                         # the two moves together too


DOOR = """<mujoco>
  <worldbody>
    <geom name="floor" type="box" pos="0 0 -0.05" size="4 4 0.05"/>
    <geom name="wall_l" type="box" pos="-1 0.022 1.1" size="1 0.05 1.1"/>
    <geom name="wall_r" type="box" pos="1.9 0.022 1.1" size="1 0.05 1.1"/>
    <geom name="wall_top" type="box" pos="0.45 0.022 2.1" size="0.45 0.05 0.1"/>
    <body name="Door0" pos="0 0 0">
      <joint name="Door0_hinge" type="hinge" axis="0 0 1" range="0 1.7"/>
      <geom name="Door0_leaf" type="box" pos="0.45 0.022 1.0" size="0.45 0.02 1.0"/>
    </body>
  </worldbody>
</mujoco>"""


def test_a_doors_face_goes_with_the_leaf_not_left_in_the_frame():
    """A trained splat spreads a door's face 10 cm off its 4 cm leaf (the atrium: the room-facing
    face 4-10 cm toward the cameras). Cut to the leaf's box, the room kept a whole closed door,
    window and handle, that stayed in the frame while the leaf swung."""
    model = mujoco.MjModel.from_xml_string(DOOR)
    face = np.concatenate([_grid((0.03, y, 0.05), (0.87, y, 1.95), step=0.03)
                           for y in (-0.06, -0.04, -0.02, 0.0, 0.02)])
    wall = np.concatenate([_grid((-1.5, 0.022, 0.05), (-0.01, 0.022, 2.0), step=0.05),
                           _grid((0.91, 0.022, 0.05), (2.5, 0.022, 2.0), step=0.05)])
    floor = _grid((-1.5, -1.5, 0.0), (2.5, 1.5, 0.0), step=0.1)
    s = _splats_at(np.concatenate([face, wall, floor]))
    parts = split(s, model, snap_bodies=False, fill=False)
    assert parts.counts["Door0"] >= 0.98 * len(face)
    assert len(parts.room) >= len(wall) + len(floor) - 2           # the wall stays the room's
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    b = model.body("Door0").id
    owner = np.concatenate([np.full(len(face), b), np.full(len(wall) + len(floor), -1)])
    assert left_behind(s.positions, s.opacities, owner, model, data)["Door0"]["pass"]
    # the cut the atrium had (only the leaf's own +-2 cm and its margin) fails the check
    kept = np.abs(face[:, 1] - 0.022) > 0.04
    owner[:len(face)][kept] = -1
    assert not left_behind(s.positions, s.opacities, owner, model, data)["Door0"]["pass"]


def test_the_wall_never_seen_behind_a_body_is_patched():
    """The cameras never saw the wall behind a chair: carried off, it left a chair-shaped window
    onto the empty scene background (the atrium's "white shell" at Chair0's spot)."""
    model = mujoco.MjModel.from_xml_string(WALLED)
    box = _grid((0.8, 1.54, 0.02), (1.2, 1.94, 0.5))
    wall = _grid((0.0, 1.95, 0.0), (2.0, 1.95, 1.0), step=0.02)
    hidden = (wall[:, 0] > 0.78) & (wall[:, 0] < 1.22) & (wall[:, 2] < 0.62)
    floor = _grid((-1, 0, 0), (3, 1.9, 0), step=0.05)
    s = _splats_at(np.concatenate([box, wall[~hidden], floor]))
    parts = split(s, model, snap_bodies=False)
    assert parts.counts.get("wall_fill", 0) > 0
    room = parts.room.positions
    on_wall = np.abs(room[:, 1] - 1.95) < 0.03
    for x in (0.85, 1.0, 1.15):
        for z in (0.1, 0.3, 0.5):
            near = on_wall & (np.abs(room[:, 0] - x) < 0.03) & (np.abs(room[:, 2] - z) < 0.03)
            assert near.any(), (x, z)
    # every patch lies on the wall's face, none stands in the room
    fill = room[len(room) - parts.counts["wall_fill"]:]
    assert np.all(np.abs(fill[:, 1] - 1.95) < 0.03)


def test_an_opened_door_shows_a_dim_space_not_the_empty_background():
    """F5: the cameras never saw past the closed door, so the trained room has nothing in the
    doorway and a door swung open showed the bare scene background. A dim space in the wall's and
    floor's colours now stands behind the leaf, hidden while it is shut."""
    model = mujoco.MjModel.from_xml_string(DOOR)
    face = np.concatenate([_grid((0.03, y, 0.05), (0.87, y, 1.95), step=0.03)
                           for y in (-0.06, -0.02, 0.02)])
    wall = np.concatenate([_grid((-1.5, 0.022, 0.05), (-0.01, 0.022, 2.0), step=0.05),
                           _grid((0.91, 0.022, 0.05), (2.5, 0.022, 2.0), step=0.05)])
    floor = _grid((-1.5, -1.5, 0.0), (2.5, -0.05, 0.0), step=0.05)     # the room: y < 0
    s = _splats_at(np.concatenate([face, wall, floor]))
    s.colors[len(face):len(face) + len(wall)] = 0.8                     # pale walls
    parts = split(s, model, snap_bodies=False)
    added = parts.counts.get("door_fill", 0)
    assert added > 500
    fill = parts.room.subset(np.arange(len(parts.room) - added, len(parts.room)))
    assert fill.positions[:, 1].min() > 0.04                          # all behind the leaf
    assert np.all(fill.colors < 0.8) and np.all(fill.colors > 0.2)   # dim, not black
    # every line of sight from the room through the opening ends on the space behind it
    from scipy.spatial import cKDTree

    tree = cKDTree(fill.positions)
    eye = np.array([0.45, -1.5, 1.2])
    for x in (0.1, 0.45, 0.8):
        for z in (0.2, 1.0, 1.9):
            ray = np.array([x, 0.03, z]) - eye
            ts = np.linspace(1.0, 4.0, 600)
            pts = eye + ts[:, None] * ray
            assert (tree.query(pts)[0] < 0.04).any(), (x, z)


def test_a_wall_patch_never_hides_what_was_seen_behind_the_wall_face():
    """A wall RoomPlan put a little in front of the real one: the trained wall stands behind
    the box's face, and the patch must not cover it."""
    model = mujoco.MjModel.from_xml_string(WALLED)
    box = _grid((0.8, 1.54, 0.02), (1.2, 1.94, 0.5))
    wall = _grid((0.0, 1.95, 0.0), (2.0, 1.95, 1.0), step=0.02)
    hidden = (wall[:, 0] > 0.78) & (wall[:, 0] < 1.22) & (wall[:, 2] < 0.62)
    recess = wall[hidden & (wall[:, 0] < 1.0)] + [0.0, 0.15, 0.0]   # seen, 15 cm behind the face
    floor = _grid((-1, 0, 0), (3, 1.9, 0), step=0.05)
    s = _splats_at(np.concatenate([box, wall[~hidden], recess, floor]))
    parts = split(s, model, snap_bodies=False)
    room = parts.room.positions
    fill = room[len(room) - parts.counts["wall_fill"]:]
    assert (fill[:, 0] > 1.02).any()                 # the unseen half is patched
    assert not ((fill[:, 0] < 0.98) & (fill[:, 2] < 0.6)).any()   # the seen half is not


def _door_face(extra_back: bool = True):
    """The atrium's door as training left it: a face toward the room (y < 0) in layers, gold
    from the room and, by its higher SH, blue from where no camera stood; behind it, deeper in
    the leaf, a layer of junk only the far side shows."""
    from litereality_agent.models.splat.spz import SH_C1

    face = np.concatenate([_grid((0.03, y, 0.05), (0.87, y, 1.95), step=0.03)
                           for y in (-0.03, -0.02, -0.01)])
    back = _grid((0.03, 0.03, 0.05), (0.87, 0.03, 1.95), step=0.03) if extra_back \
        else np.zeros((0, 3))
    wall = np.concatenate([_grid((-1.5, 0.022, 0.05), (-0.01, 0.022, 2.0), step=0.05),
                           _grid((0.91, 0.022, 0.05), (2.5, 0.022, 2.0), step=0.05)])
    floor = _grid((-1.5, -1.5, 0.0), (2.5, -0.05, 0.0), step=0.05)
    s = _splats_at(np.concatenate([face, back, wall, floor]))
    s.sh = np.zeros((len(s), 3, 3), np.float32)
    s.sh[:len(face), 0] = np.array([-0.2, 0.0, 0.2]) / SH_C1    # from +y (the room looking in)
    s.colors[len(face):len(face) + len(back)] = (0.9, 0.2, 0.9)   # the junk: magenta
    s.colors[len(face) + len(back):len(face) + len(back) + len(wall)] = 0.8
    return s, len(face), len(back)


def test_a_door_shows_every_side_the_colour_the_room_saw_it_in():
    """The door was only filmed from the room. Its splats' higher SH, from anywhere else, turned
    the leaf pink and green the moment it swung open or was seen edge on; it now keeps the
    colour the room saw, from every side."""
    model = mujoco.MjModel.from_xml_string(DOOR)
    s, n_face, _ = _door_face(extra_back=False)
    parts = split(s, model, snap_bodies=False)
    door = parts.bodies["Door0"]
    assert door.sh is None or not np.any(door.sh)
    front = door.colors[:n_face]
    assert np.allclose(np.median(front, axis=0), (0.7, 0.5, 0.3), atol=0.05)


def test_an_opened_doors_far_side_is_a_door_not_what_lay_behind_its_face():
    """What training left deeper in the leaf than its face was hidden from every camera and was
    all the far side showed (the gold and magenta shards of the opened atrium door). It goes,
    and an opaque backing in the face's colour closes the leaf; a window in the face (no
    splats) stays see-through."""
    from litereality_agent.pipeline.splatroom.split import DOOR_BACKING

    model = mujoco.MjModel.from_xml_string(DOOR)
    s, n_face, n_back = _door_face()
    window = (np.abs(s.positions[:, 0] - 0.6) < 0.08) & (np.abs(s.positions[:, 2] - 1.3) < 0.3) \
        & (s.positions[:, 1] < 0.035)
    s.opacities[window] = 0.0                                       # the glass: nothing there
    parts = split(s, model, snap_bodies=False)
    door = parts.bodies["Door0"]
    p = door.positions
    magenta = (door.colors[:, 0] > 0.8) & (door.colors[:, 1] < 0.3)
    assert not magenta.any()                                        # the junk is gone
    backing = (np.abs(p[:, 1] - (0.002 + DOOR_BACKING)) < 1e-3) & (door.scales[:, 2] < 0.001)
    assert backing.sum() > 3000                                     # a sheet over the leaf
    solid = backing & ((np.abs(p[:, 0] - 0.2) < 0.05) & (np.abs(p[:, 2] - 0.6) < 0.05))
    glass = backing & ((np.abs(p[:, 0] - 0.6) < 0.03) & (np.abs(p[:, 2] - 1.3) < 0.1))
    assert solid.any() and glass.any()
    assert door.opacities[solid].min() > 0.9 and door.opacities[glass].max() < 0.3
    assert np.allclose(np.median(door.colors[solid], axis=0), (0.7, 0.5, 0.3), atol=0.06)


def test_floaters_glimpsed_past_a_shut_door_are_cleared_from_its_doorway():
    """Past the atrium's door edges the cameras glimpsed the corridor; training smeared that into
    floaters behind the leaf (a white sheet hanging in the doorway once it opened). The space
    behind a door is cleared, then the dim space fills it; the wall it stands in stays."""
    model = mujoco.MjModel.from_xml_string(DOOR)
    s, n_face, n_back = _door_face(extra_back=False)
    sheet = _grid((0.2, 0.6, 0.5), (0.6, 0.6, 1.5), step=0.03)          # behind the door
    s = type(s).concat([s, _splats_at(sheet)])
    s.colors[-len(sheet):] = 1.0
    parts = split(s, model, snap_bodies=False)
    assert parts.counts.get("door_clear", 0) >= len(sheet)
    room = parts.room
    added = parts.counts["door_fill"]
    seen = room.subset(np.arange(len(room) - added))
    behind = (seen.positions[:, 1] > 0.1) & (np.abs(seen.positions[:, 0] - 0.45) < 0.5)
    assert not behind.any()
    on_wall = np.abs(seen.positions[:, 1] - 0.022) < 0.01
    assert on_wall.sum() >= 0.98 * (len(_grid((-1.5, 0.022, 0.05), (-0.01, 0.022, 2.0), step=0.05))
                                    + len(_grid((0.91, 0.022, 0.05), (2.5, 0.022, 2.0), step=0.05)))
