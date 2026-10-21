"""geometry.py — the room as ONE glb the browser draws, and the manifest that names its parts.

The room call never ships MuJoCo to the browser. The server owns the only simulation and streams
body poses; the browser only has to DRAW the room and move a handful of nodes. So the model is
turned into a glb once, at startup, laid out for exactly that job:

* every movable body is a top-level node ``body_<id>`` holding its drawn geoms in the body's own
  frame. A state frame is then one position + quaternion per node — the client never re-derives a
  geom pose, so it can never disagree with the server about where a drawer's front is.
* everything that cannot move (the world body and whatever is welded to it) is baked in world
  frame under one ``static`` node and merged per material: a scanned room is hundreds of geoms,
  and hundreds of draw calls is what makes a phone stutter, not the triangle count.

The glb stays in MuJoCo's coordinates (Z up, metres). The client rotates the whole scene once;
baking a Y-up conversion in here would put two frames on the wire and every bug between them.

What gets drawn is `model_info.visual_geom_ids` — the same rule the physics side uses — so a
collider (group 3) is never drawn over the visual mesh it approximates.
"""

from __future__ import annotations

import json
import math
import struct
import time
from collections import defaultdict
from dataclasses import dataclass, field

import mujoco
import numpy as np
import trimesh
from PIL import Image

from litereality_agent.room_ops.roomcall.model_info import (
    body_name,
    movable_body_ids,
    visual_geom_ids,
)

MAX_TEXTURE_SIDE = 2048
# Total texels across every texture in the glb. A scanned room carries a stitched texture per wall
# and one per object; at full size that is a 100+ MB download before anyone sees the room. Above
# the budget every texture is scaled by the same factor, so relative sharpness is preserved.
TEXTURE_TEXEL_BUDGET = 40_000_000
EYE_HEIGHT = 1.6
# Half a person's width: the spawn spot's ring of downward rays, so nobody starts at a table edge.
SPAWN_FOOTPRINT = 0.3
# An infinite plane (size 0) with nothing else in the scene to size it to.
EMPTY_PLANE_HALF = 5.0
PLANE_MARGIN = 0.25
SLAB_THICKNESS = 0.05
# MuJoCo's internal geom rgba default. A geom whose rgba is still exactly this takes its
# material's colour; any other value overrides the material (engine_vis_visualize.c setMaterial).
_DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)

_G = mujoco.mjtGeom
_TEX_RGB = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
_TEX_2D = int(mujoco.mjtTexture.mjTEXTURE_2D)


@dataclass
class _Piece:
    """One geom's triangles in the frame of the node it will hang under."""

    verts: np.ndarray            # (n, 3) float64
    faces: np.ndarray            # (m, 3) int64, local to this piece
    normals: np.ndarray | None   # (n, 3)
    uv: np.ndarray | None        # (n, 2) in MuJoCo's texcoord convention (v down, like glTF)


@dataclass(frozen=True)
class _Look:
    """Everything that decides a glTF material. Pieces with equal looks merge into one mesh."""

    rgba: tuple[float, float, float, float]   # display-space (sRGB) colour, as MuJoCo shows it
    texid: int                                # -1: untextured
    double_sided: bool
    emission: float
    roughness: float
    metallic: float


@dataclass
class _Container:
    pieces: dict[_Look, list[_Piece]] = field(default_factory=lambda: defaultdict(list))


# ── geometry helpers ─────────────────────────────────────────────────────────────────────────────


def _quat_mat(quat: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=np.float64))
    return out.reshape(3, 3)


def _box(half: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """24 vertices, not 8: a box shares no normal between faces, and a welded box shades like a
    blob."""
    verts, normals, faces = [], [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            n = np.zeros(3)
            n[axis] = sign
            u, v = (axis + 1) % 3, (axis + 2) % 3
            base = len(verts)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis] = sign
                p[u], p[v] = a, b
                verts.append(p * half)
                normals.append(n)
            quad = [(0, 1, 2), (0, 2, 3)] if sign > 0 else [(0, 2, 1), (0, 3, 2)]
            faces.extend([(base + i, base + j, base + k) for i, j, k in quad])
    return np.asarray(verts), np.asarray(faces, dtype=np.int64), np.asarray(normals)


def _cylinder(radius: float, half: float, sections: int = 32):
    """Side ring and caps as separate vertices so the rim stays sharp."""
    ang = np.linspace(0.0, 2.0 * math.pi, sections, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros(sections)], axis=1)
    side = np.concatenate([ring * [radius, radius, 0] + [0, 0, -half],
                           ring * [radius, radius, 0] + [0, 0, half]])
    side_n = np.concatenate([ring, ring])
    i = np.arange(sections)
    j = (i + 1) % sections
    side_f = np.concatenate([np.stack([i, j, j + sections], 1),
                             np.stack([i, j + sections, i + sections], 1)])
    verts, normals, faces = [side], [side_n], [side_f]
    base = 2 * sections
    for z, sign in ((half, 1.0), (-half, -1.0)):
        cap = np.concatenate([[[0.0, 0.0, z]], ring * [radius, radius, 0] + [0, 0, z]])
        cap_f = np.stack([np.zeros(sections, dtype=np.int64), 1 + i, 1 + j], 1)
        if sign < 0:
            cap_f = cap_f[:, [0, 2, 1]]
        verts.append(cap)
        normals.append(np.tile([0.0, 0.0, sign], (sections + 1, 1)))
        faces.append(cap_f + base)
        base += sections + 1
    return (np.concatenate(verts), np.concatenate(faces).astype(np.int64),
            np.concatenate(normals))


_UNIT_SPHERE = trimesh.creation.icosphere(subdivisions=3)


def _primitive(gtype: int, size: np.ndarray):
    """(verts, faces, normals) of a primitive in its geom frame, or None for a type not drawn."""
    if gtype == _G.mjGEOM_BOX:
        return _box(np.asarray(size, dtype=float))
    if gtype == _G.mjGEOM_SPHERE:
        v = np.asarray(_UNIT_SPHERE.vertices)
        return v * size[0], np.asarray(_UNIT_SPHERE.faces, dtype=np.int64), v.copy()
    if gtype == _G.mjGEOM_ELLIPSOID:
        v = np.asarray(_UNIT_SPHERE.vertices)
        radii = np.maximum(np.asarray(size, dtype=float), 1e-9)
        n = v / radii
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        return v * radii, np.asarray(_UNIT_SPHERE.faces, dtype=np.int64), n
    if gtype == _G.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(height=2.0 * float(size[1]), radius=float(size[0]),
                                        count=[24, 12])
        return (np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int64),
                np.asarray(mesh.vertex_normals))
    if gtype == _G.mjGEOM_CYLINDER:
        return _cylinder(float(size[0]), float(size[1]))
    return None


def _plane_slab(half_x: float, half_y: float, centre_xy=(0.0, 0.0)):
    """A plane drawn as a thin slab whose TOP is the plane: a zero-thickness quad vanishes when seen
    from below or edge-on, and the client is free to look at the floor from anywhere."""
    v, f, n = _box(np.array([half_x, half_y, SLAB_THICKNESS / 2.0]))
    v = v + [centre_xy[0], centre_xy[1], -SLAB_THICKNESS / 2.0]
    return v, f, n


def _mesh_arrays(model: mujoco.MjModel, mesh_id: int):
    """A mesh asset unwelded to one vertex per distinct (position, normal, texcoord) corner.

    MuJoCo indexes positions, normals and texcoords separately per face corner; glTF has a single
    index. Keying the corners on all three keeps every hard edge and UV seam while still sharing
    vertices inside a smooth patch, so a scanned mesh does not triple in size on the way out.
    """
    va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    verts = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float64)
    faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.int64)
    if fn == 0 or vn == 0:
        return None
    na, nn = int(model.mesh_normaladr[mesh_id]), int(model.mesh_normalnum[mesh_id])
    ta, tn = int(model.mesh_texcoordadr[mesh_id]), int(model.mesh_texcoordnum[mesh_id])
    key = faces.reshape(-1).copy()
    fnorm = ftc = None
    if nn > 0:
        fnorm = np.asarray(model.mesh_facenormal[fa:fa + fn], dtype=np.int64).reshape(-1)
        key = key * nn + fnorm
    if tn > 0 and ta >= 0:
        ftc = np.asarray(model.mesh_facetexcoord[fa:fa + fn], dtype=np.int64).reshape(-1)
        key = key * tn + ftc
    _uniq, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    out_v = verts[faces.reshape(-1)[first]]
    out_n = None
    if fnorm is not None:
        out_n = np.asarray(model.mesh_normal[na:na + nn], dtype=np.float64)[fnorm[first]]
    out_uv = None
    if ftc is not None:
        out_uv = np.asarray(model.mesh_texcoord[ta:ta + tn], dtype=np.float64)[ftc[first]]
    return out_v, inverse.reshape(-1, 3).astype(np.int64), out_n, out_uv


def _projected_uv(local: np.ndarray, half_xy: tuple[float, float], repeat: np.ndarray,
                  uniform: bool) -> np.ndarray:
    """MuJoCo's texture mapping for a primitive: the geom's local XY, projected along Z (a 2d
    texture is only meaningful on a plane's top).

    This is render_gl3.c `settexture`'s object-plane texgen, rewritten in the geom's local metres:
    s = 0.5*scl_x*x - 0.5 and t = -0.5*scl_y*y - 0.5, with scl = `texrepeat`, divided by the half
    size unless `texuniform` (and never along an infinite plane's unsized axis). So a uniform
    texture repeats `texrepeat` times per TWO metres, and t runs against y. GL's t and glTF's v
    both count from the texture's first row, so (s, t) is the glTF uv as it stands.
    """
    scl = np.asarray(repeat, dtype=float)
    if not uniform:
        half = np.asarray(half_xy, dtype=float)
        scl = np.where(half > 0, scl / np.maximum(half, 1e-9), scl)
    return np.column_stack([0.5 * scl[0] * local[:, 0] - 0.5, -0.5 * scl[1] * local[:, 1] - 0.5])


# ── appearance ───────────────────────────────────────────────────────────────────────────────────


def _geom_rgba(model: mujoco.MjModel, g: int) -> np.ndarray:
    matid = int(model.geom_matid[g])
    rgba = np.asarray(model.geom_rgba[g], dtype=float)
    if matid >= 0 and tuple(np.float32(rgba)) == tuple(np.float32(_DEFAULT_RGBA)):
        return np.asarray(model.mat_rgba[matid], dtype=float)
    return rgba


def _texture_array(model: mujoco.MjModel, texid: int) -> np.ndarray:
    """The texture as (h, w, c) uint8, rows top-first exactly as MuJoCo stores them. Measured, not
    assumed: a PNG's top row is tex_data's first row, and MuJoCo's mesh texcoord v already counts
    from that top row (its OBJ loader flips v), which is glTF's convention too."""
    w, h = int(model.tex_width[texid]), int(model.tex_height[texid])
    c = int(model.tex_nchannel[texid])
    adr = int(model.tex_adr[texid])
    data = np.asarray(model.tex_data[adr:adr + w * h * c], dtype=np.uint8)
    return data.reshape(h, w, c)


def _texture_mean(model: mujoco.MjModel, texid: int) -> np.ndarray:
    arr = _texture_array(model, texid)
    rgb = arr[..., :3] if arr.shape[2] >= 3 else np.repeat(arr[..., :1], 3, axis=2)
    return rgb.reshape(-1, 3).mean(axis=0) / 255.0


def _look(model: mujoco.MjModel, g: int, *, textured: bool,
          means: dict[int, np.ndarray]) -> tuple[_Look, bool]:
    """The geom's look, and whether its pieces need UVs. A texture that cannot be mapped (a cube map,
    or a mesh without texcoords) still tints the geom with its mean colour, so a wooden table
    shows up brown rather than as MuJoCo's default grey. `means` caches that colour per texture:
    a 4096² scan texture takes a quarter of a second to average, and a hundred geoms share it."""
    rgba = _geom_rgba(model, g)
    matid = int(model.geom_matid[g])
    texid, emission, roughness, metallic = -1, 0.0, 0.7, 0.0
    if matid >= 0:
        emission = float(model.mat_emission[matid])
        shininess = float(model.mat_shininess[matid])
        roughness = float(np.clip(0.9 - 0.5 * shininess, 0.2, 1.0))
        # MuJoCo 3.x's PBR fields default to -1 ("not set"); an authored value wins.
        if float(model.mat_roughness[matid]) >= 0:
            roughness = float(np.clip(model.mat_roughness[matid], 0.0, 1.0))
        if float(model.mat_metallic[matid]) >= 0:
            metallic = float(np.clip(model.mat_metallic[matid], 0.0, 1.0))
        tid = int(np.asarray(model.mat_texid[matid]).reshape(-1)[_TEX_RGB])
        if tid >= 0:
            if textured and int(model.tex_type[tid]) == _TEX_2D:
                texid = tid
            else:
                if tid not in means:
                    means[tid] = _texture_mean(model, tid)
                rgba = rgba.copy()
                rgba[:3] *= means[tid]
    gtype = int(model.geom_type[g])
    look = _Look(rgba=tuple(round(float(x), 4) for x in rgba), texid=texid,
                 double_sided=gtype == _G.mjGEOM_MESH, emission=round(emission, 4),
                 roughness=round(roughness, 3), metallic=round(metallic, 3))
    return look, texid >= 0


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """glTF colour factors are LINEAR; MuJoCo's rgba is what it puts on screen. Without this every
    flat-coloured object comes out washed-out in three.js (0.5 grey displays as 0.74)."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _texture_images(model: mujoco.MjModel, texids: set[int]) -> dict[int, Image.Image]:
    sizes = {t: (int(model.tex_width[t]), int(model.tex_height[t])) for t in texids}
    scale = {t: min(1.0, MAX_TEXTURE_SIDE / max(w, h, 1)) for t, (w, h) in sizes.items()}
    total = sum(w * h * scale[t] ** 2 for t, (w, h) in sizes.items())
    budget = 1.0 if total <= TEXTURE_TEXEL_BUDGET else math.sqrt(TEXTURE_TEXEL_BUDGET / total)
    images: dict[int, Image.Image] = {}
    for t in sorted(texids):
        arr = _texture_array(model, t)
        c = arr.shape[2]
        if c == 1:
            img = Image.fromarray(arr[..., 0], "L").convert("RGB")
        elif c == 2:
            img = Image.fromarray(np.repeat(arr[..., :1], 3, axis=2), "RGB")
        elif c == 4 and int(arr[..., 3].min()) < 255:
            img = Image.fromarray(arr, "RGBA")
        else:
            img = Image.fromarray(np.ascontiguousarray(arr[..., :3]), "RGB")
        s = scale[t] * budget
        if s < 0.999:
            w, h = sizes[t]
            img = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                             Image.Resampling.LANCZOS)
        if img.mode == "RGB":
            # trimesh writes a PIL image back in its own `format` and PNG otherwise. Photographic
            # scan textures are 5-10x smaller as JPEG; only a texture with real alpha stays PNG.
            img.format = "JPEG"
        images[t] = img
    return images


def _material(look: _Look, images: dict[int, Image.Image], name: str):
    rgba = np.asarray(look.rgba, dtype=float)
    lin = np.concatenate([_srgb_to_linear(rgba[:3]), [float(np.clip(rgba[3], 0.0, 1.0))]])
    image = images.get(look.texid) if look.texid >= 0 else None
    blend = rgba[3] < 0.999 or (image is not None and image.mode == "RGBA")
    kwargs = {}
    if look.emission > 0:
        kwargs["emissiveFactor"] = [float(x) for x in np.clip(lin[:3] * look.emission, 0, 1)]
    return trimesh.visual.material.PBRMaterial(
        name=name, baseColorFactor=[float(x) for x in lin], baseColorTexture=image,
        metallicFactor=look.metallic, roughnessFactor=look.roughness,
        alphaMode="BLEND" if blend else "OPAQUE", doubleSided=look.double_sided, **kwargs)


# ── layout ───────────────────────────────────────────────────────────────────────────────────────


def _place(piece: _Piece, rot: np.ndarray, pos: np.ndarray) -> _Piece:
    return _Piece(piece.verts @ rot.T + pos, piece.faces,
                  None if piece.normals is None else piece.normals @ rot.T, piece.uv)


def _geom_piece(model: mujoco.MjModel, g: int, want_uv: bool,
                mesh_cache: dict[int, tuple]) -> _Piece | None:
    """The geom's triangles in its OWN frame, or None for a type that is not drawn (hfield, sdf)."""
    gtype = int(model.geom_type[g])
    size = np.asarray(model.geom_size[g], dtype=float)
    if gtype == _G.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        if mid < 0:
            return None
        if mid not in mesh_cache:
            mesh_cache[mid] = _mesh_arrays(model, mid)
        arrays = mesh_cache[mid]
        if arrays is None:
            return None
        v, f, n, uv = arrays
        return _Piece(v, f, n, uv if want_uv else None)
    if gtype == _G.mjGEOM_PLANE:
        v, f, n = _plane_slab(size[0], size[1])
        half = (size[0], size[1])
    else:
        prim = _primitive(gtype, size)
        if prim is None:
            return None
        v, f, n = prim
        half = (size[0], size[1] if gtype in (_G.mjGEOM_BOX, _G.mjGEOM_ELLIPSOID) else size[0])
    uv = None
    if want_uv:
        matid = int(model.geom_matid[g])
        uv = _projected_uv(v, half, np.asarray(model.mat_texrepeat[matid], dtype=float),
                           bool(model.mat_texuniform[matid]))
    return _Piece(v, f, n, uv)


def _infinite_plane_piece(model: mujoco.MjModel, g: int, want_uv: bool, rot: np.ndarray,
                          pos: np.ndarray, room: tuple[np.ndarray, np.ndarray] | None) -> _Piece:
    """An infinite plane, sized to cover the rest of the room. Only a horizontal one can be fitted
    to the room's footprint; a tilted one gets a fixed slab around its own origin."""
    half, centre = (EMPTY_PLANE_HALF, EMPTY_PLANE_HALF), (0.0, 0.0)
    if room is not None and abs(rot[2, 2]) > 0.99:
        lo, hi = room
        # The room's footprint, expressed in the plane's own XY.
        corners = np.array([[x, y, pos[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        local = (corners - pos) @ rot
        llo, lhi = local.min(axis=0), local.max(axis=0)
        half = tuple((lhi[:2] - llo[:2]) / 2.0 + PLANE_MARGIN)
        centre = tuple((lhi[:2] + llo[:2]) / 2.0)
    v, f, n = _plane_slab(half[0], half[1], centre)
    uv = None
    if want_uv:
        matid = int(model.geom_matid[g])
        repeat = np.asarray(model.mat_texrepeat[matid], dtype=float)
        # The plane's OWN size, not the slab's: an unsized axis has nothing to repeat across, so
        # MuJoCo repeats along it per (two) metres whatever `texuniform` says.
        size = (float(model.geom_size[g][0]), float(model.geom_size[g][1]))
        uv = _projected_uv(v, size, repeat, bool(model.mat_texuniform[matid]))
    return _Piece(v, f, n, uv)


def _merge(pieces: list[_Piece]) -> _Piece:
    if len(pieces) == 1:
        return pieces[0]
    offsets = np.cumsum([0] + [len(p.verts) for p in pieces[:-1]])
    verts = np.concatenate([p.verts for p in pieces])
    faces = np.concatenate([p.faces + o for p, o in zip(pieces, offsets, strict=True)])
    normals = None
    if all(p.normals is not None for p in pieces):
        normals = np.concatenate([p.normals for p in pieces])
    uv = None
    if all(p.uv is not None for p in pieces):
        uv = np.concatenate([p.uv for p in pieces])
    return _Piece(verts, faces, normals, uv)


def _aabb(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
    if len(points):
        np.minimum(lo, points.min(axis=0), out=lo)
        np.maximum(hi, points.max(axis=0), out=hi)


def _hoist_world_root(tree: dict) -> None:
    """trimesh wraps the scene in an identity node called "world". Drop it so ``static`` and every
    ``body_<id>`` are the scene's top-level nodes, as the client expects."""
    nodes = tree.get("nodes") or []
    scene = (tree.get("scenes") or [{}])[tree.get("scene", 0)]
    root_ix = scene.get("nodes") or []
    if len(root_ix) != 1 or root_ix[0] != 0 or not nodes:
        return
    root = nodes[0]
    if set(root) - {"name", "children", "extras"} or tree.get("skins") or tree.get("animations"):
        return                                       # the root carries a transform: keep it
    scene["nodes"] = [c - 1 for c in root.get("children", [])]
    tree["nodes"] = nodes[1:]
    for node in tree["nodes"]:
        if "children" in node:
            node["children"] = [c - 1 for c in node["children"]]


_TEXTURE_SLOTS = ("baseColorTexture", "metallicRoughnessTexture", "normalTexture",
                  "occlusionTexture", "emissiveTexture")


def _share_images(tree: dict) -> None:
    """One glTF image and texture per MuJoCo texture. trimesh writes an image entry for every
    material that uses one — the bytes are shared (it dedupes bufferViews by hash), the entries
    are not — and three.js decodes and uploads each entry on its own. A texture on a double-sided
    mesh and on a single-sided primitive is two materials, so it would cost twice the GPU memory."""
    images, textures = tree.get("images") or [], tree.get("textures") or []
    if len(images) < 2 and len(textures) < 2:
        return

    def dedupe(items: list[dict]) -> tuple[list[dict], list[int]]:
        kept: dict[str, int] = {}
        remap = []
        for item in items:
            remap.append(kept.setdefault(json.dumps(item, sort_keys=True), len(kept)))
        return [json.loads(k) for k in kept], remap

    images, image_ix = dedupe(images)
    for tex in textures:
        if "source" in tex:
            tex["source"] = image_ix[tex["source"]]
    textures, tex_ix = dedupe(textures)
    tree["images"], tree["textures"] = images, textures
    for mat in tree.get("materials") or []:
        for holder in (mat, mat.get("pbrMetallicRoughness") or {}):
            for slot in _TEXTURE_SLOTS:
                ref = holder.get(slot)
                if isinstance(ref, dict) and "index" in ref:
                    ref["index"] = tex_ix[ref["index"]]


def _edit_glb_json(glb: bytes) -> bytes:
    """Tidy trimesh's glb in its JSON chunk; the binary chunk passes through untouched."""
    magic, version, _length = struct.unpack_from("<4sII", glb, 0)
    json_len, json_type = struct.unpack_from("<I4s", glb, 12)
    if magic != b"glTF" or json_type != b"JSON":
        return glb
    tree = json.loads(glb[20:20 + json_len])
    rest = glb[20 + json_len:]
    _hoist_world_root(tree)
    _share_images(tree)
    blob = json.dumps(tree, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 4)
    out = struct.pack("<I4s", len(blob), b"JSON") + blob + rest
    return struct.pack("<4sII", magic, version, 12 + len(out)) + out


def _nodes_only_glb(names: list[str], data: mujoco.MjData) -> bytes:
    """A glb with the node layout and nothing to draw. trimesh refuses to export an empty scene, and
    a model whose every geom is invisible is still a room people can join."""
    nodes: list[dict] = []
    for name in names:
        node: dict = {"name": name}
        if name.startswith("body_"):
            b = int(name[5:])
            node["translation"] = [float(x) for x in data.xpos[b]]
            w, x, y, z = (float(v) for v in data.xquat[b])
            node["rotation"] = [x, y, z, w]
        nodes.append(node)
    tree = {"asset": {"version": "2.0", "generator": "litereality roomcall"}, "scene": 0,
            "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes}
    blob = json.dumps(tree, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 4)
    chunk = struct.pack("<I4s", len(blob), b"JSON") + blob
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunk)) + chunk


# ── floor and spawn ──────────────────────────────────────────────────────────────────────────────


def _geom_world_aabb(model: mujoco.MjModel, data: mujoco.MjData, g: int):
    centre, half = np.asarray(model.geom_aabb[g, :3]), np.asarray(model.geom_aabb[g, 3:])
    corners = centre + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1)
                                        for z in (-1, 1)])
    world = corners @ np.asarray(data.geom_xmat[g]).reshape(3, 3).T + data.geom_xpos[g]
    return world.min(axis=0), world.max(axis=0)


def _floor(model: mujoco.MjModel, data: mujoco.MjData, drawn_lo: np.ndarray | None):
    """(floor_z, footprint centre or None). Every geom counts here, colliders included: the
    exporter's floor is an invisible group-3 box named "floor" whose top is exactly the floor.

    Movable bodies count too, because the exporter hangs the whole structure on a "room" body with
    slide joints (the shake rig), so its floor is, formally, on a body that moves. A name match
    alone is not enough — "floor_lamp" is a body name — so a floor must also be flat and at least a
    square metre.
    """
    named, planes = [], []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        gtype = int(model.geom_type[g])
        rot = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        if gtype == _G.mjGEOM_PLANE:
            if rot[2, 2] > 0.9:
                planes.append(float(data.geom_xpos[g][2]))
            continue
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") + " " + body_name(
            model, b)
        if "floor" not in name.lower() or gtype not in (_G.mjGEOM_BOX, _G.mjGEOM_MESH):
            continue
        lo, hi = _geom_world_aabb(model, data, g)
        area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        if area >= 1.0 and hi[2] - lo[2] <= 0.5:
            named.append((gtype != _G.mjGEOM_BOX, -area, float(hi[2]), (lo + hi) / 2.0))
    if named:
        _mesh, _area, top, centre = min(named, key=lambda t: (t[0], t[1]))
        return top, centre[:2]
    if planes:
        return min(planes), None
    return (float(drawn_lo[2]) if drawn_lo is not None else 0.0), None


def _spawn(model: mujoco.MjModel, data: mujoco.MjData, floor_z: float, centre: np.ndarray,
           lo: np.ndarray, hi: np.ndarray, *, budget_s: float = 2.0) -> list[float]:
    """The free standing spot nearest the room centre: rays down from eye height, at the spot and
    on a ring a body's width around it, must land on the floor, and rays out at knee, chest and
    eye height must clear 35 cm.

    One ray down is not enough twice over. It slips between a table's edge and a chair, and the
    side rays pass over a 0.7 m table top, so people started with the table through their hips.
    And a ray that finds NOTHING is not floor: the room's centre is its bounding box's, which in an
    L-shaped room is outside the room, where nothing at all is under it. So a spot over nothing
    is only the fallback, for an export that draws no floor (its floor is an invisible collider,
    which rays skip).
    """
    eye = floor_z + EYE_HEIGHT
    groups = np.ones(6, dtype=np.uint8)
    hit = np.zeros(1, dtype=np.int32)
    down = np.array([0.0, 0.0, -1.0])
    dirs = [np.array([math.cos(a), math.sin(a), 0.0]) for a in np.arange(8) * math.pi / 4]
    feet = [(0.0, 0.0)] + [(SPAWN_FOOTPRINT * v[0], SPAWN_FOOTPRINT * v[1]) for v in dirs]
    # Out to the bounding box's corners: in an L the nearest floor can be far from the centre.
    reach = float(math.hypot(hi[0] - lo[0], hi[1] - lo[1]) / 2.0)
    deadline = time.perf_counter() + budget_s

    def footing(x: float, y: float, need_floor: bool) -> int:
        """2: every ray down lands on the floor. 1: none lands on anything else, but some find
        nothing. 0: something else is under the footprint (or, with `need_floor`, nothing)."""
        missed = False
        for dx, dy in feet:
            d = mujoco.mj_ray(model, data, np.array([x + dx, y + dy, eye]), down, groups, 1, -1,
                              hit)
            if d < 0:
                if need_floor:
                    return 0
                missed = True
            elif abs((eye - d) - floor_z) > 0.15:
                return 0
        return 1 if missed else 2

    def clear(x: float, y: float) -> bool:
        for z in (floor_z + 0.4, floor_z + 1.0, eye):
            p = np.array([x, y, z])
            for v in dirs:
                d = mujoco.mj_ray(model, data, p, v, groups, 1, -1, hit)
                if 0 <= d < 0.35:
                    return False
        return True

    fallback: list[float] | None = None
    step = 0.25
    r = 0.0
    while r <= reach and time.perf_counter() < deadline:
        count = 1 if r == 0 else max(6, int(2 * math.pi * r / step))
        for k in range(count):
            a = 2 * math.pi * k / count
            x, y = centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)
            if not (lo[0] + 0.3 <= x <= hi[0] - 0.3 and lo[1] + 0.3 <= y <= hi[1] - 0.3):
                continue
            # Once there is a fallback, only a spot WITH floor under it can do better.
            ground = footing(x, y, need_floor=fallback is not None)
            if ground and clear(x, y):
                spot = [round(float(x), 4), round(float(y), 4), round(eye, 4)]
                if ground == 2:
                    return spot
                fallback = spot
        r += step
    if fallback is not None:
        return fallback
    return [round(float(centre[0]), 4), round(float(centre[1]), 4), round(eye, 4)]


# ── entry point ──────────────────────────────────────────────────────────────────────────────────


def build_scene_assets(model: mujoco.MjModel, *, label: str) -> tuple[bytes, dict]:
    """The room as glb bytes plus the manifest the client needs to animate and walk it.

    Poses come from `mj_forward` on a fresh MjData (qpos0). Body nodes take MuJoCo's BODY frame
    (`xpos`/`xquat`), not the inertial frame, which is what the state stream sends too.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    movable = movable_body_ids(model)

    static = _Container()
    bodies: dict[int, _Container] = {b: _Container() for b in sorted(movable)}
    mesh_cache: dict[int, tuple] = {}
    tex_means: dict[int, np.ndarray] = {}
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    deferred: list[tuple[int, _Look, bool]] = []

    for g in visual_geom_ids(model):
        if float(_geom_rgba(model, g)[3]) <= 0.0:
            continue                       # MuJoCo draws an alpha-0 geom as nothing; so do we
        gtype = int(model.geom_type[g])
        b = int(model.geom_bodyid[g])
        look, want_uv = _look(model, g, textured=True, means=tex_means)
        if gtype == _G.mjGEOM_MESH and want_uv:
            mid = int(model.geom_dataid[g])
            if mid < 0 or int(model.mesh_texcoordnum[mid]) == 0:
                look, want_uv = _look(model, g, textured=False, means=tex_means)
        if (gtype == _G.mjGEOM_PLANE and b not in movable
                and (model.geom_size[g][0] <= 0 or model.geom_size[g][1] <= 0)):
            deferred.append((g, look, want_uv))
            continue
        piece = _geom_piece(model, g, want_uv, mesh_cache)
        if piece is None:
            continue
        if b in movable:
            piece = _place(piece, _quat_mat(model.geom_quat[g]), np.asarray(model.geom_pos[g]))
            bodies[b].pieces[look].append(piece)
            world_rot = np.asarray(data.xmat[b]).reshape(3, 3)
            _aabb(piece.verts @ world_rot.T + data.xpos[b], lo, hi)
        else:
            piece = _place(piece, np.asarray(data.geom_xmat[g]).reshape(3, 3),
                           np.asarray(data.geom_xpos[g]))
            static.pieces[look].append(piece)
            _aabb(piece.verts, lo, hi)

    room = (lo.copy(), hi.copy()) if np.all(np.isfinite(lo)) else None
    for g, look, want_uv in deferred:
        rot = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        pos = np.asarray(data.geom_xpos[g])
        piece = _place(_infinite_plane_piece(model, g, want_uv, rot, pos, room), rot, pos)
        static.pieces[look].append(piece)
        _aabb(piece.verts, lo, hi)
    if not np.all(np.isfinite(lo)):
        lo, hi = np.array([-5.0, -5.0, 0.0]), np.array([5.0, 5.0, 3.0])

    # ── glb ──
    looks = {look for c in [static, *bodies.values()] for look in c.pieces}
    images = _texture_images(model, {look.texid for look in looks if look.texid >= 0})
    materials = {look: _material(look, images, f"mat_{i}")
                 for i, look in enumerate(sorted(looks, key=repr))}
    scene = trimesh.Scene(base_frame="world")

    def add(container: _Container, node: str) -> None:
        for i, (look, pieces) in enumerate(sorted(container.pieces.items(),
                                                  key=lambda kv: repr(kv[0]))):
            merged = _merge(pieces)
            uv = None
            if merged.uv is not None:
                # trimesh holds OBJ-convention UVs (v up) and flips them on glb export.
                uv = np.column_stack([merged.uv[:, 0], 1.0 - merged.uv[:, 1]])
            mesh = trimesh.Trimesh(
                vertices=merged.verts.astype(np.float32), faces=merged.faces,
                vertex_normals=None if merged.normals is None
                else merged.normals.astype(np.float32),
                visual=trimesh.visual.TextureVisuals(uv=uv, material=materials[look]),
                process=False)
            name = f"{node}_m{i}"
            scene.add_geometry(mesh, geom_name=name, node_name=name, parent_node_name=node)

    scene.graph.update(frame_from="world", frame_to="static", matrix=np.eye(4))
    add(static, "static")
    for b, container in bodies.items():
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(data.xmat[b]).reshape(3, 3)
        pose[:3, 3] = data.xpos[b]
        scene.graph.update(frame_from="world", frame_to=f"body_{b}", matrix=pose)
        add(container, f"body_{b}")
    if scene.geometry:
        glb = _edit_glb_json(scene.export(file_type="glb"))
    else:
        glb = _nodes_only_glb(["static", *(f"body_{b}" for b in bodies)], data)

    # ── manifest ──
    floor_z, centre = _floor(model, data, lo)
    if centre is None:
        centre = (lo[:2] + hi[:2]) / 2.0
    spawn = _spawn(model, data, floor_z, np.asarray(centre, dtype=float), lo, hi)
    manifest = {
        "label": label,
        "up": "z",
        "bodies": [{"id": b, "name": body_name(model, b), "node": f"body_{b}"}
                   for b in sorted(movable)],
        "bounds": {"min": [round(float(x), 4) for x in lo],
                   "max": [round(float(x), 4) for x in hi]},
        "floor_z": round(float(floor_z), 4),
        "spawn": spawn,
    }
    return glb, manifest


__all__ = ["build_scene_assets"]
