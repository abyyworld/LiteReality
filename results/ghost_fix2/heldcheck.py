"""Headless grab check of a bundle's physics (RoomSim, no browser): every free body grabbed at
three points a user would click (centre of mass, top of the back, a front seat corner), lifted
0.9 m and carried 0.8 m. A body something overhangs (it lags a slow lift by 2.5 cm) is first
pulled out 0.6 m backwards (its +y), as a user would. Then each is yanked straight up at its origin, as F3's teal chair was. Reports the tilt while held (angle of its up axis from vertical, so turning in
heading does not count), the rise, the deepest contact it makes with anything, and how far every
other body moved.

    uv run python results/ghost_fix2/heldcheck.py run/atrium_stray_q/v2_chairs_fix2 [OUT.json]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from litereality_agent.room_ops.roomcall.sim import RoomSim, load_model

DT = 1 / 60


def tilt(d, b) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(d.xmat[b].reshape(3, 3)[2, 2])))))


def turned(q0, q1) -> float:
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q0, q1))))))


def clearance(m, d, b, xy, free) -> float:
    """Plan distance from `xy` to the nearest wall (minus the body's reach) or other chair."""
    from litereality_agent.pipeline.splatroom.split import _shell_geoms

    best = min((float(np.hypot(*(d.xpos[o][:2] - xy))) for o in free if o != b), default=9.0)
    for g in _shell_geoms(m, d):
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g][:3]
        loc = (np.array([xy[0], xy[1], c[2]]) - c) @ R
        out = np.maximum(np.abs(loc) - m.geom_aabb[g][3:], 0.0)
        best = min(best, float(np.linalg.norm(out)) - 0.35)
    return best


def points(m, d, b) -> dict:
    """World points on body `b` a user would grab, from its geoms' boxes."""
    R, x = d.xmat[b].reshape(3, 3), d.xpos[b]
    corners = []
    for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
        Rg = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + Rg @ m.geom_aabb[g][:3]
        for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
            corners.append(R.T @ (c + Rg @ (m.geom_aabb[g][3:] * s) - x))
    loc = np.asarray(corners)
    lo, hi = loc.min(0), loc.max(0)
    seat = [g for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
            if "seat" in (m.geom(g).name or "")]
    seat_top = (m.geom_pos[seat[0]][2] + m.geom_size[seat[0]][2]) if seat else 0.0
    out = {"com": d.xipos[b].copy(),
           "back_top": x + R @ np.array([0.0, hi[1] - 0.03, hi[2] - 0.03]),
           "seat_corner": x + R @ np.array([lo[0] + 0.03, lo[1] + 0.03, seat_top])}
    return out


def run(bundle: Path) -> dict:
    m = load_model(bundle / "physics" / "scene_splat.xml")
    base = RoomSim(m)
    settled = base.settled_state()
    free = [b for b in base.movable if m.body_jntnum[b]
            and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE]
    res = {}
    for b in free:
        name = m.body(b).name
        for key, p in points(m, base.data, b).items():
            s = RoomSim(m, settled=settled)
            d = s.data
            start = {o: (d.xpos[o].copy(), d.xquat[o].copy()) for o in s.movable}
            rest = d.xpos[b].copy()
            assert s.grab("u", b, p)
            hand = np.array(p, dtype=float)
            row = {"point": key, "grab": np.round(p, 3).tolist(), "max_tilt_deg": 0.0,
                   "deepest_contact_m": 0.0, "pulled_out": False}

            def go(to, seconds, record=True, row=row, s=s, d=d):
                nonlocal hand
                frm = hand.copy()
                n = max(1, int(seconds / DT))
                for i in range(1, n + 1):
                    hand = frm + (to - frm) * i / n
                    s.move("u", hand)
                    s.step_realtime(DT)
                    if record:
                        row["max_tilt_deg"] = max(row["max_tilt_deg"], tilt(d, b))
                        for k in range(d.ncon):
                            c = d.contact[k]
                            if b in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                                row["deepest_contact_m"] = max(row["deepest_contact_m"],
                                                               -float(c.dist))

            blocked = False
            for k in range(1, 13):                      # up 36 cm, 3 cm at a time
                go(p + [0, 0, 0.03 * k], 0.1)
                if 0.03 * k - (d.xpos[b][2] - rest[2]) > 0.025:
                    blocked = True
                    break
            if blocked:                                 # something overhead: pull it out first
                s.reset()
                assert s.grab("u", b, p)
                hand = np.array(p, dtype=float)
                back = d.xmat[b].reshape(3, 3)[:, 1].copy()
                back[2] = 0
                back /= np.linalg.norm(back)
                go(p + 0.6 * back, 1.2)
                row["pulled_out"] = True
            lift_from = hand.copy()
            go(lift_from + [0, 0, 0.9 - (hand[2] - p[2])], 1.5)
            go(hand, 1.0)                               # hold still
            row["rise_m"] = round(float(d.xpos[b][2] - rest[2]), 3)
            row["others_after_lift"] = {
                m.body(o).name: [round(float(np.linalg.norm(d.xpos[o] - start[o][0])), 3),
                                 round(turned(start[o][1], d.xquat[o]), 1)]
                for o in s.movable if o != b}
            away = max((np.array([math.cos(a), math.sin(a)]) for a in np.linspace(0, 2 * math.pi, 24,
                                                                                endpoint=False)),
                       key=lambda u: clearance(m, d, b, d.xpos[b][:2] + 0.8 * u, free))
            go(hand + np.r_[0.8 * away, 0.0], 1.5)
            go(hand, 1.0)
            row["carried_m"] = round(float(np.linalg.norm(d.xpos[b][:2] - rest[:2])), 3)
            row["tilt_held_end_deg"] = round(tilt(d, b), 1)
            row["touching_at_end"] = sorted({m.geom(int(g)).name for k in range(d.ncon)
                                             for g in (d.contact[k].geom1, d.contact[k].geom2)
                                             if b in (m.geom_bodyid[d.contact[k].geom1],
                                                      m.geom_bodyid[d.contact[k].geom2])
                                             and m.geom_bodyid[g] != b})
            row["max_tilt_deg"] = round(row["max_tilt_deg"], 1)
            row["deepest_contact_m"] = round(row["deepest_contact_m"], 4)
            row["others_after_carry"] = {
                m.body(o).name: [round(float(np.linalg.norm(d.xpos[o] - start[o][0])), 3),
                                 round(turned(start[o][1], d.xquat[o]), 1)]
                for o in s.movable if o != b}
            row["resets"] = s.unstable_resets
            res[f"{name}/{key}"] = row
            print(name, key, {k: v for k, v in row.items() if k != "grab"}, flush=True)
        # A user who yanks it straight up from where it stands, at its origin (under the desk:
        # the desk stops it; it must not tip over and wedge itself into the desk).
        s = RoomSim(m, settled=settled)
        d = s.data
        p = d.xpos[b].copy()
        assert s.grab("u", b, p)
        yank = {"max_tilt_deg": 0.0, "deepest_contact_m": 0.0}
        for i in range(1, 61):
            s.move("u", p + [0, 0, 0.9 * min(1.0, i / 30)])
            s.step_realtime(DT)
            yank["max_tilt_deg"] = max(yank["max_tilt_deg"], round(tilt(d, b), 1))
            for k in range(d.ncon):
                c = d.contact[k]
                if b in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                    yank["deepest_contact_m"] = max(yank["deepest_contact_m"],
                                                    round(-float(c.dist), 4))
        yank["rise_m"] = round(float(d.xpos[b][2] - p[2]), 3)
        res[f"{name}/yank_origin"] = yank
        print(name, "yank_origin", yank, flush=True)
    return res


if __name__ == "__main__":
    out = run(Path(sys.argv[1]))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
