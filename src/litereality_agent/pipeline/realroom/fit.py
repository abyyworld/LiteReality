"""fit.py — a generated GLB turned into the visual mesh of one quick-room body.

TRELLIS returns an object in its OWN frame: glTF +Y up, front +Z, somewhere inside the unit cube,
with no metric scale. The quick room already knows where the object is and how big — its RoomPlan
box, the body frame of the template that stands in for it (origin at the box centre, +X along the
width, front towards -Y, +Z up). So fitting is the pipeline's `build_room._place_in_box` rule:

1. turn the mesh into the body frame: glTF (x, y, z) -> (x, -z, y), which takes up +Y to +Z and
   the front +Z to -Y, the box's front;
2. pick which way round it goes (`choose_yaw`): RoomPlan's front is 180 deg wrong for about one
   object in seven, and TRELLIS has no fixed front for a symmetric thing at all. Each quarter turn
   is scored by how well its footprint's aspect matches the box's width : depth and — where the
   capture saw the object — how well its silhouette, seen from those very cameras, overlaps the
   object's depth mask. The quarter turns keep the scaled mesh inside the box; RoomPlan's own
   front wins a tie;
3. scale it, non-uniformly, to fill the box exactly, so the visual stands where the colliders
   stand; floor-standing furniture is then snapped down onto the floor (`floor_local`);
4. decimate anything heavier than `MAX_FACES` (TRELLIS already decimates to its face target; this
   is the safety net), keeping the texture.

The result is a `Fitted` mesh in the body frame, written as an OBJ (positions, OBJ-convention UVs)
and a PNG for MuJoCo (`write_files`) — MuJoCo reads OBJ and PNG, not GLB or JPEG.
"""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# glTF Y-up (front +Z) -> body Z-up (front -Y): (x, y, z) -> (x, -z, y). A proper rotation.
YUP_TO_BODY = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
MAX_FACES = 150_000
MAX_TEXTURE_SIDE = 2048
YAWS = (0, 90, 180, 270)
# Score = mean silhouette IoU - ASPECT_WEIGHT * |ln(mesh aspect / box aspect)| (+ FRONT_PRIOR for
# RoomPlan's own front). A sofa twice as wide as deep turned a quarter loses ~0.35; a chair, whose
# footprint is near square, is decided by its silhouette.
ASPECT_WEIGHT = 0.25
FRONT_PRIOR = 0.02
SILHOUETTE_FACES = 20_000     # faces rasterised per view: plenty at depth-map resolution
NEAR = 0.05


@dataclass(frozen=True)
class View:
    """One camera that saw the object: intrinsics at the mask's resolution, the world (Z-up) ->
    camera (ARKit: +X right, +Y up, looking down -Z) transform, and the object's mask from depth."""

    frame: int
    K: np.ndarray
    w2c: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class Placement:
    """Where the object's box is: in its body frame (``centre``, ``half``) and in the world
    (``world_centre``, ``world_yaw``, radians) — the world one only matters for the cameras."""

    half: np.ndarray
    centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    world_centre: np.ndarray = field(default_factory=lambda: np.zeros(3))
    world_yaw: float = 0.0
    floor_local: float | None = None     # the floor's height in the body frame, to snap onto

    def to_world(self, local: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.world_yaw), math.sin(self.world_yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return (local - self.centre) @ rot.T + self.world_centre


@dataclass
class Fitted:
    vertices: np.ndarray            # (n, 3) body frame, metres
    faces: np.ndarray               # (m, 3)
    uv: np.ndarray | None           # (n, 2) OBJ convention (v up), one per vertex
    texture: object | None          # PIL RGB image, or None (then ``rgba``)
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)
    yaw: int = 0                    # quarter turn chosen, degrees
    scores: dict[int, dict] = field(default_factory=dict)
    faces_in: int = 0
    decimated: bool = False

    def summary(self) -> dict:
        return {"yaw_deg": self.yaw, "faces": int(len(self.faces)), "faces_in": self.faces_in,
                "decimated": self.decimated, "textured": self.texture is not None,
                "scores": {str(k): v for k, v in self.scores.items()}}


# ── reading ──────────────────────────────────────────────────────────────────────────────────────


def load_glb(source: bytes | str | Path):
    """One triangle mesh from a GLB, every node's transform applied, UVs and texture kept."""
    import trimesh

    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else bytes(source)
    scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene", process=False)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        geom = scene.geometry.get(name)
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces):
            geom = geom.copy()
            geom.apply_transform(transform)
            meshes.append(geom)
    if not meshes:
        raise ValueError("the GLB holds no triangles")
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def _look(mesh) -> tuple[np.ndarray | None, object | None, tuple[float, float, float, float]]:
    """(per-vertex UVs, RGB texture, fallback colour): the base-colour texture of a PBR (TRELLIS)
    or simple material, tinted by its factor; a mesh without one gets its material's colour."""
    from PIL import Image

    visual = mesh.visual
    material = getattr(visual, "material", None)
    rgba = (0.7, 0.7, 0.7, 1.0)                      # no colour given at all: a neutral grey
    tint = (1.0, 1.0, 1.0)                           # what multiplies the texture
    factor = getattr(material, "baseColorFactor", None)
    if factor is None:
        factor = getattr(material, "diffuse", None)
    if factor is not None:
        f = np.asarray(factor, dtype=float).reshape(-1)[:4]
        f = f / 255.0 if f.max() > 1.0 else f
        rgba = tuple(float(v) for v in np.pad(f, (0, 4 - len(f)), constant_values=1.0))
        tint = rgba[:3]
    uv = getattr(visual, "uv", None)
    image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if uv is None or image is None or len(uv) != len(mesh.vertices):
        colours = getattr(visual, "vertex_colors", None)
        if colours is not None and len(colours) == len(mesh.vertices) and image is None:
            mean = np.asarray(colours, dtype=float)[:, :3].mean(axis=0) / 255.0
            rgba = (*[float(v) for v in mean], 1.0)
        return None, None, rgba
    texture = image.convert("RGB")
    if any(abs(c - 1.0) > 1e-3 for c in tint):
        arr = np.asarray(texture, dtype=float) * np.asarray(tint)
        texture = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if max(texture.size) > MAX_TEXTURE_SIDE:
        texture.thumbnail((MAX_TEXTURE_SIDE, MAX_TEXTURE_SIDE), Image.Resampling.LANCZOS)
    return np.asarray(uv, dtype=float).copy(), texture, rgba


# ── placing ──────────────────────────────────────────────────────────────────────────────────────


def _rot_z(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def place(vertices: np.ndarray, yaw: float, placement: Placement) -> np.ndarray:
    """``vertices`` (body orientation, any scale) turned by ``yaw`` degrees about +Z and scaled
    per axis to fill the box exactly."""
    v = vertices @ _rot_z(yaw).T
    lo, hi = v.min(axis=0), v.max(axis=0)
    scale = 2.0 * np.asarray(placement.half, dtype=float) / np.maximum(hi - lo, 1e-9)
    return (v - (lo + hi) / 2.0) * scale + np.asarray(placement.centre, dtype=float)


def silhouette(world: np.ndarray, faces: np.ndarray, view: View, *,
               max_faces: int = SILHOUETTE_FACES) -> np.ndarray:
    """The mesh's silhouette in ``view`` at its mask's resolution (bool). Faces with a corner
    behind the camera are skipped; above ``max_faces`` an even sample is drawn and every vertex
    splatted, which at depth-map resolution closes the gaps."""
    import cv2

    h, w = view.mask.shape
    cam = world @ view.w2c[:3, :3].T + view.w2c[:3, 3]
    depth = -cam[:, 2]
    front = depth > NEAR
    safe = np.where(front, depth, 1.0)
    u = view.K[0, 0] * cam[:, 0] / safe + view.K[0, 2]
    v = -view.K[1, 1] * cam[:, 1] / safe + view.K[1, 2]
    pix = np.stack([u, v], axis=1)
    keep = front[faces].all(axis=1)
    tris = faces[keep]
    if len(tris) > max_faces:
        tris = tris[np.linspace(0, len(tris) - 1, max_faces).astype(int)]
    out = np.zeros((h, w), np.uint8)
    pts = np.round(pix[tris] * 16).astype(np.int64)
    # Far outside the image: clamp, so a huge triangle near the camera cannot overflow cv2's ints.
    pts = np.clip(pts, -16 * 4 * max(w, h), 16 * 5 * max(w, h)).astype(np.int32)
    for tri in pts:
        cv2.fillConvexPoly(out, tri, 1, lineType=cv2.LINE_8, shift=4)
    if len(faces) > max_faces:
        ui, vi = np.floor(u[front]).astype(int), np.floor(v[front]).astype(int)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        out[vi[inside], ui[inside]] = 1
    return out.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)) / union if union else 0.0


def choose_yaw(vertices: np.ndarray, faces: np.ndarray, placement: Placement,
               views: Sequence[View] = (), yaws: Sequence[int] = YAWS
               ) -> tuple[int, dict[int, dict]]:
    """The quarter turn (degrees) that best fits the box and what the cameras saw, and the
    score of each: {yaw: {"aspect_err", "iou" (None without views), "score"}}."""
    half = np.asarray(placement.half, dtype=float)
    box_aspect = max(half[0], 1e-6) / max(half[1], 1e-6)
    scores: dict[int, dict] = {}
    for yaw in yaws:
        turned = vertices @ _rot_z(yaw).T
        ext = np.maximum(turned.max(axis=0) - turned.min(axis=0), 1e-6)
        aspect_err = abs(math.log((ext[0] / ext[1]) / box_aspect))
        iou = None
        if views:
            world = placement.to_world(place(vertices, yaw, placement))
            iou = float(np.mean([_iou(silhouette(world, faces, view), view.mask)
                                 for view in views]))
        score = (iou or 0.0) - ASPECT_WEIGHT * aspect_err + (FRONT_PRIOR if yaw == 0 else 0.0)
        scores[int(yaw)] = {"aspect_err": round(aspect_err, 4),
                            "iou": None if iou is None else round(iou, 4),
                            "score": round(score, 4)}
    best = max(scores, key=lambda y: (scores[y]["score"], y == 0))
    return best, scores


# ── decimating ───────────────────────────────────────────────────────────────────────────────────


def decimate(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray | None, target: int
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Quadric decimation (open3d) to ``target`` faces. The UVs follow each kept vertex from its
    nearest original one: exact away from the texture's seams, a smear on the few triangles that
    straddle one — acceptable for a safety net TRELLIS's own decimation rarely leaves work for."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices.astype(np.float64)),
                                     o3d.utility.Vector3iVector(faces.astype(np.int32)))
    small = mesh.simplify_quadric_decimation(int(target))
    v = np.asarray(small.vertices, dtype=float)
    f = np.asarray(small.triangles, dtype=np.int64)
    if uv is None or not len(v):
        return v, f, None
    _dist, nearest = cKDTree(vertices).query(v)
    return v, f, uv[nearest]


# ── the whole fit ────────────────────────────────────────────────────────────────────────────────


def fit(glb: bytes | str | Path, placement: Placement, *, views: Sequence[View] = (),
        max_faces: int = MAX_FACES) -> Fitted:
    """A GLB as the visual mesh of the body whose box is ``placement`` (see the module doc)."""
    mesh = load_glb(glb)
    uv, texture, rgba = _look(mesh)
    vertices = np.asarray(mesh.vertices, dtype=float) @ YUP_TO_BODY.T
    faces = np.asarray(mesh.faces, dtype=np.int64)
    faces_in = len(faces)
    decimated = False
    if faces_in > max_faces:
        vertices, faces, uv = decimate(vertices, faces, uv, max_faces)
        decimated = True
        if uv is None:
            texture = None
    yaw, scores = choose_yaw(vertices, faces, placement, views)
    placed = place(vertices, yaw, placement)
    if placement.floor_local is not None:
        placed[:, 2] += float(placement.floor_local) - float(placed[:, 2].min())
    return Fitted(vertices=placed, faces=faces, uv=uv, texture=texture, rgba=rgba, yaw=yaw,
                  scores=scores, faces_in=faces_in, decimated=decimated)


def obj_text(fitted: Fitted) -> str:
    """The OBJ MuJoCo reads: positions, one UV per vertex in OBJ's convention (v up — MuJoCo's
    loader flips it to count from the texture's top row, as glTF does), faces ``a/a b/b c/c``."""
    v = np.asarray(fitted.vertices, dtype=float)
    f = np.asarray(fitted.faces, dtype=np.int64) + 1
    lines = ["# litereality realroom: visual mesh in its body's frame, metres"]
    lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in v]     # 0.1 mm
    if fitted.uv is not None and fitted.texture is not None:
        lines += [f"vt {s:.5f} {t:.5f}" for s, t in np.asarray(fitted.uv, dtype=float)]
        lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in f]
    else:
        lines += [f"f {a} {b} {c}" for a, b, c in f]
    return "\n".join(lines) + "\n"


def png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


@dataclass(frozen=True)
class FittedFiles:
    """A fitted mesh as written: file names relative to the mesh / texture folders, content-named
    so a version never overwrites a file an earlier version (still being served) names."""

    mesh: str
    texture: str | None
    rgba: tuple[float, float, float, float]


def write_files(fitted: Fitted, stem: str, mesh_dir: Path, texture_dir: Path) -> FittedFiles:
    """Write the OBJ (and PNG) under ``mesh_dir`` / ``texture_dir`` as ``<stem>-<digest>.*``."""
    obj = obj_text(fitted).encode("ascii")
    png = png_bytes(fitted.texture) if fitted.texture is not None and fitted.uv is not None \
        else None
    digest = hashlib.sha256(obj + (png or b"")).hexdigest()[:10]
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = f"{stem}-{digest}.obj"
    _write_atomic(mesh_dir / mesh_name, obj)
    texture_name = None
    if png is not None:
        texture_dir.mkdir(parents=True, exist_ok=True)
        texture_name = f"{stem}-{digest}.png"
        _write_atomic(texture_dir / texture_name, png)
    return FittedFiles(mesh_name, texture_name, fitted.rgba)


def _write_atomic(path: Path, data: bytes) -> None:
    if path.is_file() and path.stat().st_size == len(data):
        return                      # content-named: the same name is the same bytes
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
