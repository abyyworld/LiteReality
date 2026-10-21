"""Synthetic objects for the physics tests: built with trimesh, written as real GLBs.

Everything here is Blender-free. A recipe-built object reaches the physics compiler as a Y-up GLB
whose nodes carry the articulation as glTF `extras`, so that is what these helpers write: trimesh
exports the geometry and the extras are then patched into the GLB's JSON chunk, which is exactly
where Blender's exporter puts them.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import trimesh

# Z-up (the build frame) -> glTF Y-up. The inverse of properties.YUP_TO_ZUP.
ZUP_TO_YUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def box(extents, centre) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=extents,
                                transform=trimesh.transformations.translation_matrix(centre))


def write_glb(path: Path, parts: dict[str, trimesh.Trimesh], extras: dict[str, dict] | None = None
              ) -> Path:
    """`parts` are Z-up meshes keyed by node name; `extras` are attached to those nodes."""
    scene = trimesh.Scene()
    for name, mesh in parts.items():
        yup = mesh.copy()
        yup.vertices = np.asarray(yup.vertices) @ ZUP_TO_YUP.T
        scene.add_geometry(yup, node_name=name, geom_name=f"{name}_geom")
    raw = scene.export(file_type="glb")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_with_extras(raw, extras or {}))
    return path


def _with_extras(raw: bytes, extras: dict[str, dict]) -> bytes:
    length = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20:20 + length].decode("utf-8"))
    for node in doc.get("nodes", []):
        if node.get("name") in extras:
            node["extras"] = extras[node["name"]]
    body = json.dumps(doc).encode("utf-8")
    body += b" " * (-len(body) % 4)
    rest = raw[20 + length:]
    out = struct.pack("<II", len(body), 0x4E4F534A) + body + rest
    return b"glTF" + struct.pack("<II", 2, 12 + len(out)) + out


# ------------------------------------------------------------------ a door

LEAF_W, LEAF_H, LEAF_T = 0.80, 2.00, 0.04          # a real interior leaf
LEAF_DENSITY = 450.0                               # kg/m^3: a laminated core door
FRAME_DEPTH = 0.12


def door_parts() -> tuple[dict, dict]:
    """A frame (two jambs + a head, welded) and a hinged leaf, Z-up, front facing -Y."""
    jamb = 0.04
    parts = {
        "Frame_L": box((jamb, FRAME_DEPTH, LEAF_H + jamb), (-LEAF_W / 2 - jamb / 2, 0, (LEAF_H + jamb) / 2)),
        "Frame_R": box((jamb, FRAME_DEPTH, LEAF_H + jamb), (LEAF_W / 2 + jamb / 2, 0, (LEAF_H + jamb) / 2)),
        "Frame_Head": box((LEAF_W, FRAME_DEPTH, jamb), (0, 0, LEAF_H + jamb / 2)),
        "Door_leaf": box((LEAF_W, LEAF_T, LEAF_H), (0, 0, LEAF_H / 2)),
        # the handle is what makes the door asymmetric, as a real one is
        "Door_leaf_handle": box((0.12, 0.06, 0.03), (LEAF_W / 2 - 0.1, -0.05, 1.0)),
    }
    extras = {"Door_leaf": {"articulation_type": "revolute", "articulation_axis": [0, 0, 1],
                            "limit_min": 0.0, "limit_max": 1.6,
                            "joint_origin": [-LEAF_W / 2, 0.0, 0.0], "density": LEAF_DENSITY}}
    return parts, extras


def write_door(path: Path) -> Path:
    parts, extras = door_parts()
    return write_glb(path, parts, extras)


def placed_like_build_room(mesh: trimesh.Trimesh, yaw: float, scale, centre) -> np.ndarray:
    """The Z-up 4x4 `build_room` applies: T(centre) . Rz(yaw) . diag(scale) . (p - asset centre)."""
    lo, hi = mesh.bounds
    c, s = math.cos(yaw), math.sin(yaw)
    linear = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.diag(scale)
    transform = np.eye(4)
    transform[:3, :3] = linear
    transform[:3, 3] = np.asarray(centre, float) - linear @ ((lo + hi) / 2.0)
    return transform
