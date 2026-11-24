"""Move test against the diagnosis's own ground truth (results/diagnosis/chairs/s6_fractions.py),
independent of the instance finder used by the cut: per real chair (opaque 3 cm components in
the uncut room, grown one voxel, the touching teal/dark pair split by 2-means seeded at the two
seats), the share of its splats (by count and by opacity) in the room and in each body of the
served bundle (room splat + each body splat at its pose in the bundle's physics).

    uv run --frozen python results/fix_v4/chairs/truth_table.py run/atrium_stray_q/v2 \
        run/atrium_stray_q/v2_chairs        # -> results/fix_v4/chairs/truth_table.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage
from scipy.cluster.vq import kmeans2

from litereality_agent.models.splat import spz
from litereality_agent.pipeline.splatroom.split import _inside_any, _static_boxes
from litereality_agent.room_ops.roomcall.model_info import movable_body_ids

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FLOOR = -1.2677
SCANNED = REPO / "run/atrium_stray_q/v2/physics/scene.xml"      # static boxes: the same for all
SEEDS = {"blue chair": (0.57, 1.37), "teal chair": (-1.1, 0.15), "dark chair": (-0.65, 0.45)}


def served(out_dir: Path):
    bundle = json.loads((out_dir / "splat.json").read_text())
    report = json.loads((out_dir / "report.json").read_text())
    scene = Path(report.get("corrected_scene") or report["scene"])
    scene = scene if scene.is_absolute() else REPO / scene
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    room = spz.decode((out_dir / bundle["room"]).read_bytes())
    fill = int(bundle["counts"].get("floor_fill", 0))
    P, OP, L, names = [room.positions[:len(room) - fill]], [room.opacities[:len(room) - fill]], \
        [np.full(len(room) - fill, -1)], {-1: "room"}
    for name, file in bundle["bodies"].items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        s = spz.decode((out_dir / file).read_bytes())
        with np.errstate(all="ignore"):
            P.append((s.positions.astype(np.float64) @ np.asarray(data.xmat[b]).reshape(3, 3).T
                      + data.xpos[b]).astype(np.float32))
        OP.append(s.opacities)
        L.append(np.full(len(s), b))
        names[b] = name
    return np.concatenate(P), np.concatenate(OP), np.concatenate(L), names, str(scene)


def truth(P, OP):
    m = mujoco.MjModel.from_xml_path(str(SCANNED))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    static = _static_boxes(m, d, sorted(movable_body_ids(m)))
    region = (P[:, 2] > FLOOR + 0.04) & (P[:, 2] < FLOOR + 1.10) & (P[:, 0] > -1.9) \
        & (P[:, 0] < 1.3) & (P[:, 1] > -0.4) & (P[:, 1] < 2.1)
    ridx = np.flatnonzero(region)
    with np.errstate(all="ignore"):
        ridx = ridx[~_inside_any(P[ridx].astype(np.float64), static, 0.03)]
    v = 0.03
    lo = P[ridx].min(0)
    g = ((P[ridx] - lo) / v).astype(int)
    op = OP[ridx] > 0.3
    occ = np.zeros(g.max(0) + 1, np.int32)
    np.add.at(occ, tuple(g[op].T), 1)
    lab, _n = ndimage.label(occ >= 2, structure=np.ones((3, 3, 3)))
    lab_all = ndimage.grey_dilation(lab, size=(3, 3, 3))[tuple(g.T)]

    def comp_at(xy):
        dd = np.hypot(P[ridx, 0] - xy[0], P[ridx, 1] - xy[1])
        vals, cnt = np.unique(lab_all[(dd < 0.12) & (lab_all > 0) & op], return_counts=True)
        return int(vals[np.argmax(cnt)])

    c = {k: comp_at(xy) for k, xy in SEEDS.items()}
    sets = {"blue chair": ridx[lab_all == c["blue chair"]]}
    pair = ridx[lab_all == c["teal chair"]]
    if c["dark chair"] == c["teal chair"]:
        _cent, lbl = kmeans2(P[pair, :2].astype(np.float64),
                             np.array([SEEDS["teal chair"], SEEDS["dark chair"]]),
                             minit="matrix", seed=0)
        sets["teal chair"], sets["dark chair"] = pair[lbl == 0], pair[lbl == 1]
    else:
        sets["teal chair"], sets["dark chair"] = pair, ridx[lab_all == c["dark chair"]]
    return sets


def table(out_dir: Path) -> dict:
    P, OP, L, names, scene = served(out_dir)
    rows = {}
    for k, ix in truth(P, OP).items():
        ids = sorted(set(L[ix].tolist()))
        n, w = len(ix), float(OP[ix].sum())
        by_count = {names[i]: round(float((L[ix] == i).sum() / n), 4) for i in ids}
        by_op = {names[i]: round(float(OP[ix][L[ix] == i].sum() / w), 4) for i in ids}
        bodies = {a: b for a, b in by_count.items() if a != "room"}
        top = max(bodies.items(), key=lambda kv: kv[1]) if bodies else (None, 0.0)
        leak = max((b for a, b in bodies.items() if a != top[0]), default=0.0)
        rows[k] = {"splats": n, "centre": P[ix].mean(0).round(3).tolist(), "by_count": by_count,
                   "by_opacity": by_op, "body": top[0], "in_body": top[1],
                   "in_room": by_count.get("room", 0.0), "in_other_bodies": round(leak, 4),
                   "pass": top[1] >= 0.9 and leak <= 0.05}
    return {"bundle": str(out_dir), "scene": scene, "chairs": rows}


if __name__ == "__main__":
    out = {}
    for arg in sys.argv[1:]:
        d = Path(arg)
        d = d if d.is_absolute() else REPO / d
        out[d.name] = table(d)
        for k, r in out[d.name]["chairs"].items():
            print(f"{d.name:10s} {k:11s} {r['splats']:6d} splats: body {r['body']} "
                  f"{r['in_body']:.1%}, room {r['in_room']:.1%}, other bodies "
                  f"{r['in_other_bodies']:.1%}  {'PASS' if r['pass'] else 'FAIL'}")
    (OUT / "truth_table.json").write_text(json.dumps(out, indent=1) + "\n")
