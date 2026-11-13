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
from litereality_agent.pipeline.splatroom.split import corrected_scene, split  # noqa: E402


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
