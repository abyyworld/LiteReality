"""Bug 2 from the physics side: are chairs one body, welded, overlapping, or pushed along by contact?
Uses the server's own RoomSim on the served (corrected) scene."""
import numpy as np, json
from common import *
from litereality_agent.room_ops.roomcall.sim import RoomSim
from litereality_agent.room_ops.roomcall.model_info import movable_body_ids, body_name
m = mujoco.MjModel.from_xml_path(str(B / "physics/scene_splat.xml"))
print("equality constraints (welds):", m.neq, " bodies:", [(body_name(m, b), int(m.body_parentid[b])) for b in sorted(movable_body_ids(m))])
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
print("contacts at rest pose between movable bodies:")
for i in range(d.ncon):
    c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
    print("  ", body_name(m, b1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), "<->", body_name(m, b2), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2), f"dist {c.dist*1000:.1f} mm")
# gaps between chair union boxes (top-down)
from litereality_agent.pipeline.splatroom.split import _union_box
ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("Chair0", "Chair1", "Chair2")}
for a in ids:
    for b in ids:
        if a < b:
            print(f"  centre distance {a}-{b}: {np.linalg.norm(d.xpos[ids[a]][:2]-d.xpos[ids[b]][:2]):.3f} m")
sim = RoomSim(m)
pos0 = {n: sim.data.xpos[i].copy() for n, i in ids.items()}
print("settle moved:", {n: round(float(np.linalg.norm(pos0[n] - d.xpos[ids[n]])), 4) for n in ids})
res = {}
for held, toward in (("Chair2", "Chair1"), ("Chair1", "Chair2"), ("Chair1", "Chair0")):
    sim.reset() if hasattr(sim, "reset") else None
    s = RoomSim(m)
    b = ids[held]
    start = s.data.xpos[b].copy()
    grip = start + np.array([0, 0, 0.3])
    s.grab("me", b, grip)
    dirn = s.data.xpos[ids[toward]] - start; dirn[2] = 0; dirn /= np.linalg.norm(dirn)
    for k in range(60):                              # drag 0.6 m toward the other chair over 2 s
        s.move("me", grip + dirn * min(0.6, 0.01 * (k + 1)) )
        s.step_realtime(1 / 30) if hasattr(s, "step_realtime") else None
    moved = {n: round(float(np.linalg.norm(s.data.xpos[i][:2] - pos0[n][:2])), 3) for n, i in ids.items()}
    res[f"drag {held} 0.6 m toward {toward}"] = moved
    print(f"drag {held} 0.6 m toward {toward}: displacement of each chair (m):", moved)
json.dump(res, open(OUT / "physics_drag.json", "w"), indent=1)
