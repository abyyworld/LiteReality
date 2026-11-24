"""Recolour a bundle's room splat by why each splat near a body stays in the room, to see what a
ghost at a rest spot is made of:
  yellow   patches (floor, wall and door fill),
  red      within STATIC_KEEP of something static (locked to the room),
  green    in a free body's leg zone (split._leg_zones) or within 5 cm of its cut boxes,
  cyan     5-20 cm past its cut boxes above the leg zone,
  magenta  within 5 cm of static furniture lower than 0.6 m (a desk's legs and frame).
    uv run python results/ghost_fix2/tint.py BUNDLE OUT.spz
"""
import importlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from litereality_agent.models.splat.spz import decode, encode
from litereality_agent.room_ops.roomcall.model_info import movable_body_ids

S = importlib.import_module("litereality_agent.pipeline.splatroom.split")
B, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sj = json.loads((B / "splat.json").read_text())
room = decode((B / sj["room"]).read_bytes())
m = mujoco.MjModel.from_xml_path(str(B / "physics/scene_splat.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
P = np.nan_to_num(room.positions.astype(np.float64), nan=1e3, posinf=1e3, neginf=-1e3)
fill = sum(int(sj["counts"].get(k, 0)) for k in ("floor_fill", "wall_fill", "door_fill"))
n = len(room)
floor = float(json.loads((B / "report.json").read_text())["instances"]["floor"])
free = S._free_top(m)
cut = S._cut_boxes(m, d, free)
zones = S._leg_zones(m, d, free, floor)
static = S._static_boxes(m, d, sorted(movable_body_ids(m)))
above = P[:, 2] > floor + 0.03
near5 = S._inside_any(P, cut, 0.05) | S._inside_any(P, zones, 0.0)
near20 = S._inside_any(P, cut, 0.20) & ~near5
locked = S._inside_any(P, static, S.STATIC_KEEP) & (near5 | near20)
furniture = [bx for bx in static if bx[0] != 0]                 # not the room shell's walls
legs = S._inside_any(P, furniture, 0.05) & (P[:, 2] < floor + 0.6)
col = room.colors.copy()
sh = None if room.sh is None else room.sh.copy()
for mask, rgb in ((legs & above, (1, 0, 1)), (near20 & above, (0, 1, 1)), (near5 & above, (0, 1, 0)),
                  (locked & above, (1, 0, 0))):
    col[mask] = rgb
    if sh is not None:
        sh[mask] = 0
col[n - fill:] = (1, 1, 0)
if sh is not None:
    sh[n - fill:] = 0
room.colors, room.sh = col, sh
OUT.write_bytes(encode(room))
print({"green": int((near5 & above & ~locked).sum()), "cyan": int((near20 & above & ~locked).sum()),
       "red": int((locked & above).sum()), "yellow": fill, "magenta": int((legs & above).sum())})
