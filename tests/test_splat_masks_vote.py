"""Photo masks voted onto the splats (pipeline/splatroom/masks.vote), the rules that keep one
object's mask from deciding another's splats: a body votes only near its own box, a photo's
masks are exclusive, and "seen but in no mask" never takes a splat from the body holding it."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_splat import _grid, _scan_above_box, _splats_at  # noqa: E402
from test_splat_cutting import TWO  # noqa: E402

from litereality_agent.models.splat import rgbd  # noqa: E402
from litereality_agent.pipeline.splatroom import masks as object_masks  # noqa: E402
from litereality_agent.pipeline.splatroom.split import split  # noqa: E402


def _views(tmp_path, model, bodies):
    scan = tmp_path / "scan"
    scan.mkdir()
    _scan_above_box(scan, box_x=1.0, index=0, cam_dx=-0.1)
    _scan_above_box(scan, box_x=1.0, index=1, cam_dx=0.1)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    views = object_masks.plan_views(rgbd.load_frames(scan), model, data,
                                    {n: model.body(n).id for n in bodies})
    return views, data


def test_a_body_never_votes_on_splats_far_from_its_own_box(tmp_path):
    model = mujoco.MjModel.from_xml_string(TWO.format(bx=2.2))
    views, data = _views(tmp_path, model, ["A"])
    masks = {}
    for key, view in views.items():
        W, H = view.size
        view.boxes["A"] = [0.0, 0.0, float(W), float(H)]         # a prompt box over everything
        masks[key] = {"A": np.ones((H, W), bool)}                # and a mask over everything
    got = object_masks.ObjectMasks(views, masks)
    top_b = _grid((2.02, -0.18, 0.5), (2.38, 0.18, 0.5), step=0.03)
    top_a = _grid((0.82, -0.18, 0.5), (1.18, 0.18, 0.5), step=0.03)
    voted = object_masks.vote(np.concatenate([top_a, top_b]), got, model, data,
                              {"A": model.body("A").id, "B": model.body("B").id})
    assert (voted[:len(top_a)] == model.body("A").id).all()
    assert not (voted[len(top_a):] == model.body("A").id).any()     # B's top is not A's


def test_a_photos_masks_are_made_exclusive_front_object_first(tmp_path):
    model = mujoco.MjModel.from_xml_string(TWO.format(bx=1.6))
    views, data = _views(tmp_path, model, ["A"])
    view = next(iter(views.values()))
    W, H = view.size
    both = np.zeros((H, W), bool)
    both[:, : W // 2] = True
    out = object_masks._exclusive({"A": both, "B": both.copy()}, view, model, data,
                                  {"A": model.body("A").id, "B": model.body("B").id})
    assert not (out["A"] & out["B"]).any()
    assert out["A"].sum() + out["B"].sum() == both.sum()


def test_seen_in_no_mask_never_takes_a_splat_from_the_body_whose_box_holds_it(tmp_path):
    model = mujoco.MjModel.from_xml_string(TWO.format(bx=-1.0))
    views, _data = _views(tmp_path, model, ["A"])
    masks = {}
    for key, view in views.items():
        W, H = view.size
        empty = np.zeros((H, W), bool)                    # SAM missed A in this photo
        masks[key] = {"A": empty}
    got = object_masks.ObjectMasks(views, masks)
    top_a = _grid((0.82, -0.18, 0.5), (1.18, 0.18, 0.5), step=0.03)
    parts = split(_splats_at(top_a), model, masks=got, snap_bodies=False, fill=False)
    assert parts.counts["A"] == len(top_a)


def test_fitted_bodies_are_prompted_tight_with_their_centre(tmp_path):
    model = mujoco.MjModel.from_xml_string(TWO.format(bx=-1.0))
    scan = tmp_path / "scan"
    scan.mkdir()
    _scan_above_box(scan, box_x=1.0, index=0)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frames = rgbd.load_frames(scan)
    loose = object_masks.plan_views(frames, model, data, {"A": model.body("A").id})
    tight = object_masks.plan_views(frames, model, data, {"A": model.body("A").id},
                                    grow=object_masks.FITTED_GROW, centre_point=True)
    (lv,), (tv,) = loose.values(), tight.values()
    width = lambda b: b[2] - b[0]  # noqa: E731
    assert width(tv.boxes["A"]) < width(lv.boxes["A"])
    (u, v), = tv.positives["A"]
    box = tv.boxes["A"]
    assert box[0] < u < box[2] and box[1] < v < box[3]
    assert not lv.positives
