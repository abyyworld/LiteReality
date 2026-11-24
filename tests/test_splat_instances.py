"""Chairs found in the splat itself and the physics put on them (pipeline/splatroom/instances):
RoomPlan's regularised row of boxes matched to the real objects over the whole room, the bodies
re-posed with a free yaw and resized, a clone for an object no body matched, a phantom removed,
two touching chairs told apart, each cut by its cluster, and the move test that gates it all."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from litereality_agent.models.splat.spz import Splats
from litereality_agent.pipeline.splatroom import instances
from litereality_agent.pipeline.splatroom.split import _penetration, plan_overlap, split

CHAIR = """
    <body name="{name}" pos="{x} {y} 0.45" quat="{qw} 0 0 {qz}">
      <freejoint name="{name}_free"/>
      <geom name="{name}_seat" type="box" pos="0 0 0" size="0.225 0.225 0.02"/>
      <geom name="{name}_back" type="box" pos="0 0.21 0.22" size="0.225 0.015 0.22"/>
      <geom name="{name}_leg0" type="box" pos="-0.21 -0.21 -0.225" size="0.015 0.015 0.225"/>
      <geom name="{name}_leg1" type="box" pos="0.21 -0.21 -0.225" size="0.015 0.015 0.225"/>
      <geom name="{name}_leg2" type="box" pos="0.21 0.21 -0.225" size="0.015 0.015 0.225"/>
      <geom name="{name}_leg3" type="box" pos="-0.21 0.21 -0.225" size="0.015 0.015 0.225"/>
    </body>"""

ROOM = """<mujoco>
  <worldbody>
    <geom name="floor" type="box" pos="0 0 -0.05" size="4 4 0.05"/>
    <geom name="wall_n" type="box" pos="0 3.5 1" size="4 0.05 1"/>
    <geom name="wall_s" type="box" pos="0 -3.5 1" size="4 0.05 1"/>
    <geom name="wall_e" type="box" pos="3.5 0 1" size="0.05 4 1"/>
    <geom name="wall_w" type="box" pos="-3.5 0 1" size="0.05 4 1"/>
    {bodies}
  </worldbody>
</mujoco>"""


def _scene(tmp_path: Path, chairs: list[tuple[str, float, float, float]]) -> Path:
    bodies = "".join(CHAIR.format(name=n, x=x, y=y, qw=math.cos(math.radians(a) / 2),
                                  qz=math.sin(math.radians(a) / 2)) for n, x, y, a in chairs)
    path = tmp_path / "scene.xml"
    path.write_text(ROOM.format(bodies=bodies))
    return path


def _slab(lo, hi, step=0.012):
    axes = [np.arange(a, b + 1e-9, step) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def _chair_points(x: float, y: float, yaw_deg: float, *, back_height: float = 0.95,
                  width: float = 0.5) -> np.ndarray:
    """A real chair's opaque splats: a seat at 0.45 m, a back on its +y side, four legs."""
    w = width / 2
    parts = [_slab((-w, -w, 0.43), (w, w, 0.47)),
             _slab((-w, w - 0.03, 0.47), (w, w, back_height))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = (sx * (w - 0.03), sy * (w - 0.03))
            parts.append(_slab((c[0] - 0.015, c[1] - 0.015, 0.0), (c[0] + 0.015, c[1] + 0.015,
                                                                    0.43)))
    p = np.concatenate(parts)
    a = math.radians(yaw_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return p @ R.T + (x, y, 0.0)


def _haze(points: np.ndarray, rng, n: int) -> np.ndarray:
    """Faint splats hugging a surface (most of a trained chair's splats are these)."""
    pick = points[rng.integers(0, len(points), n)]
    return pick + rng.normal(0, 0.012, (n, 3))


def _splats(opaque: list[np.ndarray], faint: list[np.ndarray] = ()) -> Splats:
    pts = [*opaque, *faint]
    ops = [np.full(len(p), 0.9, np.float32) for p in opaque] \
        + [np.full(len(p), 0.05, np.float32) for p in faint]
    pos = np.concatenate(pts).astype(np.float32)
    n = len(pos)
    return Splats(pos, np.full((n, 3), 0.008, np.float32),
                  np.tile([0, 0, 0, 1.0], (n, 1)).astype(np.float32),
                  np.full((n, 3), 0.5, np.float32), np.concatenate(ops))


def _floor() -> np.ndarray:
    return _slab((-3, -3, 0.0), (3, 3, 0.0), step=0.05)


def _model(path: Path):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _owners_of(points: np.ndarray, parts, model) -> dict[str, float]:
    """Share of `points` (world) that ended up in each body of `parts` (matched by position)."""
    from scipy.spatial import cKDTree

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = {}
    tree = cKDTree(points)
    for name, local in parts.bodies.items():
        b = model.body(name).id
        world = local.positions @ np.asarray(data.xmat[b]).reshape(3, 3).T + data.xpos[b]
        d, _ = tree.query(world, k=1)
        out[name] = float((d < 1e-4).sum() / len(points))
    return out


# ---------------------------------------------------------------------------------- the row


@pytest.fixture
def row(tmp_path):
    """RoomPlan's row: three chairs 1.2 m apart, all at yaw 0. The real chairs: one a little off
    Chair0 (turned 30 deg), one 0.5 m off Chair1 (turned 90 deg), one beside Chair2 (turned 180)."""
    scene = _scene(tmp_path, [("Chair0", -1.2, 0, 0), ("Chair1", 0, 0, 0), ("Chair2", 1.2, 0, 0)])
    real = {"Chair0": (-1.3, 0.1, 30.0), "Chair1": (0.25, 0.45, 90.0),
            "Chair2": (1.35, -0.15, 180.0)}
    pts = {k: _chair_points(*v) for k, v in real.items()}
    rng = np.random.default_rng(0)
    faint = [_haze(p, rng, len(p) // 2) for p in pts.values()]
    splats = _splats([*pts.values(), _floor()], faint)
    return scene, real, pts, splats


def test_the_row_is_matched_to_the_real_chairs_and_turned_to_face_them(row):
    scene, real, _pts, splats = row
    fixed, report = instances.fit_scene(splats.positions, splats.opacities, scene,
                                        log=lambda _m: None)
    assert fixed is not None and fixed.name == "scene_splat.xml"
    model, data = _model(fixed)
    assert set(report["bodies"]) == {"Chair0", "Chair1", "Chair2"}
    assert not report["clones"] and not report["removed"]
    for name, (x, y, yaw) in real.items():
        b = model.body(name).id
        # the body's footprint centre is on the real chair's
        c, _half = instances.body_frame_box(model, data, b)
        centre = data.xpos[b] + np.asarray(data.xmat[b]).reshape(3, 3) @ c
        assert np.hypot(centre[0] - x, centre[1] - y) < 0.06, name
        R = np.asarray(data.xmat[b]).reshape(3, 3)
        got = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        assert abs((got - yaw + 180) % 360 - 180) < 8, (name, got, yaw)
        # standing on the floor, not in it
        assert _penetration(model, data, b) < 0.01


def test_each_chair_is_cut_by_its_cluster_and_passes_the_move_test(row):
    scene, _real, pts, splats = row
    fixed, _ = instances.fit_scene(splats.positions, splats.opacities, scene,
                                   log=lambda _m: None)
    model = mujoco.MjModel.from_xml_path(str(fixed))
    parts = split(splats, model, corrections={}, instances=True, fill=False)
    for name, p in pts.items():
        share = _owners_of(p, parts, model)
        assert share[name] > 0.97, (name, share)
        assert all(v < 0.01 for k, v in share.items() if k != name), (name, share)
    # the faint haze went with its chair too; the floor stayed
    assert sum(len(b) for b in parts.bodies.values()) > 1.3 * sum(len(p) for p in pts.values())
    assert np.isclose(parts.room.positions[:, 2], 0.0).sum() > 0.95 * len(_floor())
    assert parts.move_test["pass"] and parts.move_test["cut"] == "instances"
    assert len(parts.move_test["objects"]) == 3
    assert parts.grabbable == {"Chair0": True, "Chair1": True, "Chair2": True}


def test_boxes_left_where_roomplan_put_them_fail_the_move_test(row):
    scene, _real, _pts, splats = row
    model = mujoco.MjModel.from_xml_path(str(scene))
    parts = split(splats, model, snap_bodies=False, fill=False)
    assert parts.move_test["cut"] == "boxes"
    assert not parts.move_test["pass"]
    # Chair1's box stands 0.5 m off its chair: its splat holds too little of any chair to offer
    assert parts.grabbable.get("Chair1") is False


def test_the_fitted_physics_matches_the_bundle(row, tmp_path):
    """Every body the cut wrote a splat for is a body of the fitted scene, at the pose the cut
    used (the splats are in that body's frame at qpos0)."""
    scene, _real, pts, splats = row
    fixed, _ = instances.fit_scene(splats.positions, splats.opacities, scene,
                                   log=lambda _m: None)
    model, data = _model(fixed)
    parts = split(splats, model, corrections={}, instances=True, fill=False)
    for name, local in parts.bodies.items():
        b = model.body(name).id
        world = local.positions @ np.asarray(data.xmat[b]).reshape(3, 3).T + data.xpos[b]
        assert np.abs(world.mean(0)[:2] - pts[name].mean(0)[:2]).max() < 0.05


def _ghost_chair(x: float, y: float, yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """A real chair as a trained splat has it: an opaque seat and back (its cluster's core) and
    legs and a seat rim too faint and thin to make a voxel count (the atrium's blue chair)."""
    w = 0.25
    core = [_slab((-w, -w, 0.43), (w, w, 0.47)), _slab((-w, w - 0.03, 0.47), (w, w, 0.95))]
    faint = [_slab((-w - 0.02, -w - 0.02, 0.40), (w + 0.02, w + 0.02, 0.42), step=0.02)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = (sx * (w - 0.02), sy * (w - 0.02))
            faint.append(_slab((c[0] - 0.01, c[1] - 0.01, 0.05), (c[0] + 0.01, c[1] + 0.01, 0.40),
                               step=0.02))
    a = math.radians(yaw_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return (np.concatenate(core) @ R.T + (x, y, 0.0),
            np.concatenate(faint) @ R.T + (x, y, 0.0))


def test_a_chairs_faint_legs_and_rim_go_with_it_and_leave_no_ghost(tmp_path):
    """The doubled chair: its cluster is only the opaque seat and back, so a cut by cluster alone
    left the legs and the seat's rim in the room, a chair-shaped ghost at the spot the chair was
    carried from, while the move test (scored on the cluster's own core) said 100% in body."""
    scene = _scene(tmp_path, [("Chair0", -0.8, 0, 0), ("Chair1", 0.8, 0, 0)])
    core0, legs0 = _ghost_chair(-0.85, 0.05, 20.0)
    core1, legs1 = _ghost_chair(0.8, -0.05, -10.0)
    splats = _splats([core0, core1, _floor()], [legs0, legs1])
    fixed, _ = instances.fit_scene(splats.positions, splats.opacities, scene,
                                   log=lambda _m: None)
    model, data = _model(fixed)
    found = instances.find(splats.positions, splats.opacities, model, data)
    low = legs0[:, 2] < 0.35                       # the legs below the seat's grown voxels
    assert low.sum() > 200 and not found.labels(legs0[low].astype(np.float64)).any()
    parts = split(splats, model, corrections={}, instances=True, fill=False)
    for name, core, legs in (("Chair0", core0, legs0), ("Chair1", core1, legs1)):
        assert _owners_of(core, parts, model)[name] > 0.97
        share = _owners_of(legs, parts, model)
        assert share[name] > 0.97, (name, share)
    # nothing of either chair is left standing in the room
    room = parts.room.positions
    assert not (room[:, 2] > instances.CUT_FLOOR).any()
    left = parts.move_test["left_behind"]
    assert left["Chair0"]["left"] < 0.01 and left["Chair1"]["left"] < 0.01
    assert parts.move_test["pass"]


def test_the_left_behind_check_fails_a_cut_that_leaves_the_legs(tmp_path):
    """The gate that would have caught the ghost: the room keeping the legs and rim at the rest
    pose fails `left_behind` even though every opaque core splat went with its body."""
    from litereality_agent.pipeline.splatroom.split import LEFT_MAX, left_behind

    scene = _scene(tmp_path, [("Chair0", -0.8, 0, 0)])
    core, legs = _ghost_chair(-0.85, 0.05, 20.0)
    splats = _splats([core, legs, _floor()])
    fixed, _ = instances.fit_scene(splats.positions, splats.opacities, scene,
                                   log=lambda _m: None)
    model, data = _model(fixed)
    b = model.body("Chair0").id
    owner = np.concatenate([np.full(len(core), b), np.full(len(legs) + len(_floor()), -1)])
    rows = left_behind(splats.positions, splats.opacities, owner, model, data, floor=0.0)
    assert rows["Chair0"]["left"] > LEFT_MAX and not rows["Chair0"]["pass"]
    owner[:len(core) + len(legs)] = b
    assert left_behind(splats.positions, splats.opacities, owner, model, data,
                       floor=0.0)["Chair0"]["pass"]


# ---------------------------------------------------------------------------------- phantoms


def test_a_phantom_body_is_removed_and_a_chair_with_no_body_gets_a_clone(tmp_path):
    scene = _scene(tmp_path, [("Chair0", -1.5, 0, 0), ("Chair1", 1.5, 0, 0)])
    there = _chair_points(-1.45, 0.05, 0.0)
    lonely = _chair_points(0.0, 2.0, -90.0)               # 2 m from either box
    splats = _splats([there, lonely, _floor()])
    fixed, report = instances.fit_scene(splats.positions, splats.opacities, scene,
                                        log=lambda _m: None)
    assert [r["body"] for r in report["removed"]] == ["Chair1"]   # nothing stands in its box
    assert list(report["clones"]) == ["Chair2"] and report["clones"]["Chair2"]["of"] == "Chair0"
    model, data = _model(fixed)
    names = {model.body(b).name for b in range(model.nbody)}
    assert {"Chair0", "Chair2"} <= names and "Chair1" not in names
    clone = ET.parse(fixed).getroot().find("worldbody/body[@name='Chair2']")
    assert {e.get("name") for e in clone.iter() if e.get("name")} >= {"Chair2", "Chair2_free",
                                                                      "Chair2_seat"}
    b = model.body("Chair2").id
    assert np.hypot(*(data.xpos[b][:2] - (0.0, 2.0))) < 0.08
    parts = split(splats, model, corrections={}, instances=True, fill=False)
    assert set(parts.bodies) == {"Chair0", "Chair2"} and parts.move_test["pass"]


def test_a_body_whose_box_holds_something_is_kept_not_removed(tmp_path):
    """No chair found near it, but its box is not empty (a bag, a chair seen too thinly): kept."""
    scene = _scene(tmp_path, [("Chair0", -1.5, 0, 0), ("Chair1", 1.5, 0, 0)])
    there = _chair_points(-1.45, 0.05, 0.0)
    clutter = _slab((1.3, -0.1, 0.2), (1.6, 0.1, 0.4))        # too small to be a chair
    splats = _splats([there, clutter, _floor()])
    _fixed, report = instances.fit_scene(splats.positions, splats.opacities, scene,
                                         log=lambda _m: None)
    assert not report["removed"]
    assert [r["body"] for r in report["kept_unmatched"]] == ["Chair1"]


# ---------------------------------------------------------------------------------- touching


def test_two_touching_chairs_bridged_by_haze_are_two_objects(tmp_path):
    scene = _scene(tmp_path, [("Chair0", -0.3, 0, 0), ("Chair1", 0.3, 0, 0)])
    a, b = _chair_points(-0.27, 0.0, 0.0), _chair_points(0.27, 0.05, 0.0)
    # sparse opaque splats between the two seats: enough to join them at 2 per voxel
    bridge = _slab((-0.03, -0.1, 0.44), (0.03, 0.1, 0.46), step=0.02)
    splats = _splats([a, b, bridge, _floor()])
    model, data = _model(scene)
    found = instances.find(splats.positions, splats.opacities, model, data)
    chairs = sorted(found.objects(), key=lambda i: i.centre[0])
    assert len(chairs) == 2
    assert chairs[0].centre[0] < -0.15 and chairs[1].centre[0] > 0.15
    fixed, _ = instances.fit_scene(splats.positions, splats.opacities, scene,
                                   log=lambda _m: None)
    model, data = _model(fixed)
    for name in ("Chair0", "Chair1"):              # fitted to touching chairs, yet apart
        assert _penetration(model, data, model.body(name).id) <= instances.SETTLE_SLACK + 1e-3
        # ... in plan too: lifting one straight up must not catch the other
        assert plan_overlap(model, data, model.body(name).id) <= -instances.SETTLE_GAP + 1e-3
    parts = split(splats, model, corrections={}, instances=True, fill=False)
    assert _owners_of(a, parts, model)["Chair0"] > 0.95
    assert _owners_of(b, parts, model)["Chair1"] > 0.95


def test_a_sill_along_the_wall_is_not_a_chair(tmp_path):
    scene = _scene(tmp_path, [("Chair0", -1.5, 0, 0)])
    sill = _slab((-2.0, 3.2, 0.3), (1.0, 3.4, 0.9), step=0.02)      # long, thin, chair-high
    model, data = _model(scene)
    splats = _splats([_chair_points(-1.5, 0.0, 0.0), sill, _floor()])
    found = instances.find(splats.positions, splats.opacities, model, data)
    assert len(found.objects()) == 1


def test_categories_come_from_the_quick_rooms_report(tmp_path):
    scene = _scene(tmp_path, [("Seat0", -1.5, 0, 0), ("Seat1", 1.5, 0, 0)])
    (tmp_path / "quick.json").write_text('{"bodies": {"Seat0": {"category": "chair"}, '
                                         '"Seat1": {"category": "chair"}}}')
    model, data = _model(scene)
    cats = instances.categories(model, data, scene)
    assert list(cats) == ["chair"] and len(cats["chair"].bodies) == 2
    assert instances.category_of("Chair12") == "chair"


# ---------------------------------------------------------------------------------- ghost fix 2


def _splayed_chair(x: float, y: float, yaw_deg: float, *, reach: float = 0.15):
    """A real chair the way a trained splat has it: an opaque seat and back (the cluster's core)
    and four faint legs that splay `reach` past the seat's edge to the floor (thin metal legs, an
    office chair's star base), too faint to join the cluster. Returns (core, legs), world."""
    w = 0.25
    core = [_slab((-w, -w, 0.43), (w, w, 0.47)), _slab((-w, w - 0.03, 0.47), (w, w, 0.95))]
    legs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            top = np.array([sx * (w - 0.02), sy * (w - 0.02), 0.42])
            foot = np.array([sx * (w + reach), sy * (w + reach), 0.035])
            t = np.linspace(0.0, 1.0, 60)[:, None]
            line = top + t * (foot - top)
            legs.append(np.concatenate([line + (dx, dy, 0) for dx in (-0.006, 0.006)
                                        for dy in (-0.006, 0.006)]))
    a = math.radians(yaw_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return (np.concatenate(core) @ R.T + (x, y, 0.0),
            np.concatenate(legs) @ R.T + (x, y, 0.0))


def _carpet(lo=-2.0, hi=2.0) -> np.ndarray:
    """Carpet pile: faint splats 3.5-7 cm over the floor everywhere."""
    return _slab((lo, lo, 0.035), (hi, hi, 0.07), step=0.035)


def _with_opacity(groups: list[tuple[np.ndarray, float]]) -> Splats:
    pos = np.concatenate([g for g, _o in groups]).astype(np.float32)
    ops = np.concatenate([np.full(len(g), o, np.float32) for g, o in groups])
    n = len(pos)
    return Splats(pos, np.full((n, 3), 0.008, np.float32),
                  np.tile([0, 0, 0, 1.0], (n, 1)).astype(np.float32),
                  np.full((n, 3), 0.5, np.float32), ops)


def test_legs_that_splay_past_the_fitted_box_go_with_the_chair_and_the_carpet_stays(tmp_path):
    """F1: the fitted box is the chair's opaque core; its legs stood 5-20 cm outside it, so the
    cut (by that box) left them standing at the rest spot when the chair was carried off. The
    leg zone takes what is connected to the chair, not the carpet around it, and the left-behind
    check looks there too."""
    scene = _scene(tmp_path, [("Chair0", 0.0, 0.0, 0.0)])
    core, legs = _splayed_chair(0.05, -0.05, 15.0)
    carpet = _carpet()
    splats = _with_opacity([(core, 0.9), (legs, 0.2), (carpet, 0.2), (_floor(), 0.9)])
    fixed, _ = instances.fit_scene(splats.positions, splats.opacities, scene,
                                   log=lambda _m: None)
    model, data = _model(fixed)
    parts = split(splats, model, corrections={}, instances=True, fill=False)
    assert _owners_of(core, parts, model)["Chair0"] > 0.97
    share = _owners_of(legs, parts, model)["Chair0"]
    assert share > 0.9, share
    # the carpet away from the chair's footprint and its feet stays the room's
    from scipy.spatial import cKDTree

    far = cKDTree(legs[:, :2]).query(carpet[:, :2])[0] > 0.08
    far &= np.abs(carpet[:, 0] - 0.05) + np.abs(carpet[:, 1] + 0.05) > 0.45
    assert _owners_of(carpet[far], parts, model).get("Chair0", 0.0) < 0.01
    assert parts.move_test["left_behind"]["Chair0"]["pass"]

    # the cut that left the legs fails the check, though by opacity they are little
    from litereality_agent.pipeline.splatroom.split import left_behind

    b = model.body("Chair0").id
    owner = np.full(len(splats), -1)
    owner[:len(core)] = b
    row = left_behind(splats.positions, splats.opacities, owner, model, data, floor=0.0)["Chair0"]
    assert row["left"] <= 0.08 and not row["legs_pass"] and not row["pass"], row


def test_two_chairs_whose_legs_reach_into_each_others_zone_each_keep_their_own(tmp_path):
    scene = _scene(tmp_path, [("Chair0", -0.4, 0.0, 0.0), ("Chair1", 0.4, 0.0, 0.0)])
    core0, legs0 = _splayed_chair(-0.4, 0.0, 0.0, reach=0.12)
    core1, legs1 = _splayed_chair(0.4, 0.0, 0.0, reach=0.12)
    splats = _with_opacity([(core0, 0.9), (core1, 0.9), (legs0, 0.2), (legs1, 0.2),
                            (_floor(), 0.9)])
    fixed, _ = instances.fit_scene(splats.positions, splats.opacities, scene,
                                   log=lambda _m: None)
    model, _data = _model(fixed)
    parts = split(splats, model, corrections={}, instances=True, fill=False)
    for name, other, legs in (("Chair0", "Chair1", legs0), ("Chair1", "Chair0", legs1)):
        share = _owners_of(legs, parts, model)
        assert share[name] > 0.9 and share.get(other, 0.0) < 0.05, (name, share)


DESK = """
    <body name="Table0" pos="0 0.45 0">
      <geom name="Table0_top" type="box" pos="0 0 0.72" size="0.8 0.4 0.02"/>
      <geom name="Table0_leg0" type="box" pos="-0.75 0.35 0.35" size="0.025 0.025 0.35"/>
      <geom name="Table0_leg1" type="box" pos="0.75 0.35 0.35" size="0.025 0.025 0.35"/>
    </body>"""


def test_the_desk_edge_over_a_tucked_chair_stays_with_the_desk(tmp_path):
    """F2: the trained desk edge spreads a few centimetres past the desk's box, into the box of the
    chair tucked under it. A splat inside a body's box was locked to the room only if strictly
    inside a static box, so the chair carried a strip of the desk's front edge."""
    scene = _scene(tmp_path, [("Chair0", 0.0, -0.1, 180.0)])
    scene.write_text(scene.read_text().replace("</worldbody>", DESK + "\n  </worldbody>"))
    core = _chair_points(0.0, -0.1, 180.0, back_height=0.9)
    # the desk's front edge as trained: its top, front face and lip, up to 2.5 cm off its box
    edge = np.concatenate([_slab((-0.8, 0.05, 0.742), (0.8, 0.2, 0.76), step=0.012),
                           _slab((-0.8, 0.026, 0.69), (0.8, 0.05, 0.75), step=0.012)])
    desk = _slab((-0.8, 0.2, 0.742), (0.8, 0.85, 0.745), step=0.03)
    splats = _with_opacity([(core, 0.9), (edge, 0.9), (desk, 0.9), (_floor(), 0.9)])
    fixed, report = instances.fit_scene(splats.positions, splats.opacities, scene,
                                        log=lambda _m: None)
    assert "Chair0" in report["bodies"]
    model, _data = _model(fixed)
    from litereality_agent.pipeline.splatroom.split import _cut_boxes, _inside_any

    b = model.body("Chair0").id
    in_box = _inside_any(edge.astype(np.float64), _cut_boxes(model, _data, [b]), 0.0)
    assert in_box.sum() > 200                       # the chair's box does reach the desk edge
    parts = split(splats, model, corrections={}, instances=True, fill=False)
    assert _owners_of(edge, parts, model).get("Chair0", 0.0) < 0.01
    assert _owners_of(core, parts, model)["Chair0"] > 0.9


def test_chairs_too_close_are_slid_apart_not_shrunk_and_cut_by_what_they_hold(tmp_path):
    """F3: two fitted chairs overlapping in plan were narrowed to 0.73 of their footprint to keep
    apart, so a carried chair passed 27 % into the desk, the walls and the other chair, and its
    cut lost its legs. Now they keep their size and are slid a few centimetres apart; the cut
    takes each object where the splat has it (its extent), not where the collision box went."""
    scene = _scene(tmp_path, [("Chair0", -0.3, 0, 0), ("Chair1", 0.3, 0, 0)])
    a, b = _chair_points(-0.27, 0.0, 0.0), _chair_points(0.27, 0.05, 0.0)
    bridge = _slab((-0.03, -0.1, 0.44), (0.03, 0.1, 0.46), step=0.02)
    splats = _splats([a, b, bridge, _floor()])
    fixed, report = instances.fit_scene(splats.positions, splats.opacities, scene,
                                        log=lambda _m: None)
    model, data = _model(fixed)
    for name in ("Chair0", "Chair1"):
        fit = report["bodies"][name]
        assert "shrunk_to_clear" not in fit, fit
        body = model.body(name).id
        assert plan_overlap(model, data, body) <= -instances.SETTLE_GAP + 1e-3
        shift = np.hypot(*fit.get("shifted_to_clear_m", (0.0, 0.0)))
        assert shift <= instances.SETTLE_SHIFT + 1e-6
        # the collision boxes are as big as the object the cut takes
        site = model.site(name + instances.EXTENT).id
        geoms = range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
        width = max(model.geom_pos[g][0] + model.geom_size[g][0] for g in geoms) \
            - min(model.geom_pos[g][0] - model.geom_size[g][0] for g in geoms)
        assert width >= 0.98 * 2 * model.site_size[site][0]
    assert report["bodies"]["Chair0"].get("shifted_to_clear_m")
    parts = split(splats, model, corrections={}, instances=True, fill=False)
    assert _owners_of(a, parts, model)["Chair0"] > 0.95
    assert _owners_of(b, parts, model)["Chair1"] > 0.95
