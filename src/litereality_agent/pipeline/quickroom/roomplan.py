"""roomplan.py — a RoomPlan ``room.usdz`` read straight out of its zip, in the pipeline's frame.

The full pipeline reads the same file through Blender (``extract_shell``) or the vendored
preprocessing (which pulls in OpenCV and open3d at import). Both cost seconds before a single wall
is known, and the quick room's whole budget is a few seconds. Underneath, a RoomPlan "parametric"
export is an UNCOMPRESSED zip of ASCII ``.usda`` layers, one tiny file per component::

    room.usda                                  root: groups whose metadata `references` the parts
    assets/Parametric/Walls/WallN/WallN.usda   one Cube per wall (scale = length, height, ~0)
    assets/Parametric/Walls/WallN/DoorM.usda   an opening lives in its wall's folder
    assets/Parametric/Floors/Floor0.usda       a Mesh (the floor slab), not a Cube
    assets/Parametric/<Cat>/<Cat>N.usda        one Cube per object (scale = width, height, depth)

so a few hundred lines of bracket matching and ``ast.literal_eval`` (values only — nothing in the
file is ever executed) read it in ~20 ms.

THE FRAME IS THE PIPELINE'S, deliberately. Every stage after this one — TRELLIS objects, the full
export — is swapped into the same live call, and a swap carries each body's state by NAME and
POSE. So walls, objects and openings come out exactly as ``layout.adapter.shell_from_scene_data``
states them: Z up via ``ARKIT_TO_Z_UP`` (``(x, y, z) -> (x, -z, y)``, the matrix Blender's USD
importer puts on the root prim), an object's ``yaw`` the heading of its own local +X (never the
longer axis — that is the chair-orientation bug ``shell.py`` documents), and the scan's own handles
(``Chair0``, ``Wall10_Door_0``). Checked on tea_room: every object box matches the pipeline's
layout baseline to the 4th decimal.
"""

from __future__ import annotations

import ast
import math
import re
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from litereality_agent.pipeline.scene_init.layout.adapter import MIN_PLATE_FRACTION, _polygon_area
from litereality_agent.pipeline.scene_init.layout.shell import ARKIT_TO_Z_UP

USDZ_CANDIDATES = ("room.usdz", "roomplan/room.usdz")
OPENING_KINDS = {"Door": "door", "Window": "window", "Opening": "opening"}
_ROUND = 4
CORNER_TOL = 0.05          # m: two wall ends this close are the same corner


def find_room_usdz(scan_dir: Path | str) -> Path | None:
    """The scan's RoomPlan export: ``room.usdz`` or ``roomplan/room.usdz``, the two places the
    LiteReality Scanner app has written it (the CLI's `SCAN_MARKERS`)."""
    scan_dir = Path(scan_dir)
    for name in USDZ_CANDIDATES:
        if (scan_dir / name).is_file():
            return scan_dir / name
    return None


# ── the room ─────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Wall:
    """A wall's centreline on the floor plan. ``thickness`` is as MEASURED: RoomPlan's walls are
    0.1 mm sheets, and inventing a real thickness is the builder's job, on the outside."""

    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    height: float

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    @property
    def along(self) -> np.ndarray:
        delta = np.subtract(self.end, self.start)
        n = float(np.linalg.norm(delta))
        return delta / n if n > 1e-9 else np.array([1.0, 0.0])


@dataclass(frozen=True)
class Opening:
    """A door, window or open doorway, in the SHELL's terms. ``offset`` is the distance from the
    wall's start to the opening's CENTRE (`shell.opening_span` explains why that matters);
    ``sill`` is the bottom edge above ``floor_z``."""

    handle: str                 # Wall10_Door_0 — what the pipeline calls it
    source: str                 # Door0 — what RoomPlan calls it
    wall: str
    kind: str                   # door | window | opening
    offset: float
    width: float
    height: float
    sill: float
    is_open: bool | None = None  # RoomPlan's "Door(Isopen: True)"; None when it did not say

    @property
    def span(self) -> tuple[float, float]:
        return self.offset - self.width / 2.0, self.offset + self.width / 2.0


@dataclass(frozen=True)
class RoomObject:
    """One RoomPlan box. ``size`` is (width along local +X, depth along local +Y, height); the
    object's front faces local -Y (`shell.facing_yaw`: yaw - 90 deg). ``matrix_arkit`` maps the
    unit cube onto the box in ARKit space, which is where the capture's camera poses live."""

    handle: str
    category: str               # "chair", "sink_storage" — the handle minus its digits, lowered
    roomplan_category: str      # RoomPlan's own label ("Chair", "Storage")
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float                  # degrees
    matrix_arkit: tuple[tuple[float, ...], ...] | None = None
    merged_from: tuple[str, ...] = ()

    @property
    def bottom(self) -> float:
        return self.center[2] - self.size[2] / 2.0

    @property
    def top(self) -> float:
        return self.center[2] + self.size[2] / 2.0

    @property
    def volume(self) -> float:
        return float(np.prod(self.size))

    def footprint(self, grow: float = 0.0) -> np.ndarray:
        """The 4 XY corners of the oriented footprint, ``grow`` metres larger on every side."""
        hw, hd = self.size[0] / 2.0 + grow, self.size[1] / 2.0 + grow
        a = math.radians(self.yaw)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        local = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])
        return local @ rot.T + np.asarray(self.center[:2])

    def moved(self, center: tuple[float, float, float], size: tuple[float, float, float]
              ) -> RoomObject:
        return replace(self, center=tuple(float(v) for v in center),
                       size=tuple(float(v) for v in size))


@dataclass(frozen=True)
class Floor:
    """Every floor slab of the export as one mesh. A multi-room capture has a slab per room, so
    ``outlines`` holds one loop per slab and the floor is their union (`RoomPlan.on_floor`)."""

    verts: np.ndarray                       # (n, 3) Z-up
    faces: tuple[tuple[int, ...], ...]
    outlines: tuple[tuple[tuple[float, float], ...], ...]  # each slab's top boundary loop


@dataclass
class RoomPlan:
    source: Path
    floor_z: float
    ceiling_z: float
    walls: dict[str, Wall]
    openings: dict[str, Opening]
    objects: dict[str, RoomObject]
    floor: Floor | None
    skipped: list[str] = field(default_factory=list)   # components that are not boxes
    warnings: list[str] = field(default_factory=list)  # what was left out or distrusted, and why

    @property
    def height(self) -> float:
        return self.ceiling_z - self.floor_z

    def outlines(self) -> list[np.ndarray]:
        """The floor polygons: each slab's top face, or — for an export without a (believable)
        floor mesh — the walls chained end to end (`wall_loops`)."""
        loops = self.floor.outlines if self.floor is not None else ()
        if not any(len(loop) >= 3 for loop in loops):
            loops = wall_loops(self.walls.values())
        return [np.asarray(loop, dtype=float) for loop in loops if len(loop) >= 3]

    def on_floor(self, xy: np.ndarray) -> np.ndarray:
        """Per XY point, whether it is inside ANY floor polygon: the union of the room's floors."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        inside = np.zeros(len(xy), dtype=bool)
        for poly in self.outlines():
            inside |= inside_polygon(xy, poly)
        return inside

    def shell(self, *, wall_thickness: float | None = None,
              objects: dict[str, RoomObject] | None = None) -> dict[str, Any]:
        """The room as the SHELL dict the layout pass works on (`layout/shell.py` schema).

        Openings are left out on purpose: the pipeline's layout stage builds its shell from
        ``scene_data`` without them, and the repair reseats openings it is given — handing it ours
        would make the quick room's furniture land somewhere the full pipeline's does not. A floor
        plate `_assemble` found implausible is left out the same way the ingest adapter drops it.
        """
        objects = self.objects if objects is None else objects
        floor = {"verts": [], "faces": []}
        if self.floor is not None:
            floor = {"verts": self.floor.verts.round(_ROUND).tolist(),
                     "faces": [list(f) for f in self.floor.faces]}
        shell_objects: dict[str, Any] = {}
        for handle, obj in objects.items():
            record = {"category": obj.category, "center": list(obj.center),
                      "size": list(obj.size), "yaw": obj.yaw}
            if obj.merged_from:
                record["merged_from"] = list(obj.merged_from)
            shell_objects[handle] = record
        return {
            "walls": {name: {"start": list(w.start), "end": list(w.end),
                             "thickness": w.thickness if wall_thickness is None else wall_thickness,
                             "height": w.height}
                      for name, w in self.walls.items()},
            "openings": {},
            "objects": shell_objects,
            "floor": floor,
            "floor_z": self.floor_z,
            "ceiling_z": self.ceiling_z,
        }


# ── a tiny USDA reader ───────────────────────────────────────────────────────────────────────────


def _balanced(text: str, i: int, open_ch: str, close_ch: str) -> int:
    """Index just past the bracket that closes ``text[i]``; skips "strings" and @asset paths@,
    which may contain brackets of their own."""
    depth, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i = text.index('"', i + 1)
        elif c == "@":
            i = text.index("@", i + 1)
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced brackets in a USDA layer")


_DEF = re.compile(r'\b(def|over)\s+(?:(\w+)\s+)?"([^"]+)"')


def _prims(text: str) -> list[dict]:
    """The top-level prims of a USDA body: ``[{type, name, meta, attrs, children}]``. ``attrs``
    is the prim's own text with its children cut out, so an attribute of a child can never be
    read as one of its parent."""
    prims, i = [], 0
    while True:
        m = _DEF.search(text, i)
        if not m:
            return prims
        j = m.end()
        while text[j].isspace():
            j += 1
        meta = ""
        if text[j] == "(":
            k = _balanced(text, j, "(", ")")
            meta, j = text[j + 1:k - 1], k
            while text[j].isspace():
                j += 1
        if text[j] != "{":
            raise ValueError(f"expected a body for prim {m.group(3)!r}")
        k = _balanced(text, j, "{", "}")
        body = text[j + 1:k - 1]
        children = _prims(body)
        own = body
        for child in children:
            own = own.replace(child["_src"], "")
        prims.append({"type": m.group(2) or "", "name": m.group(3), "meta": meta,
                      "attrs": own, "children": children, "_src": text[m.start():k]})
        i = k


def _attr(attrs: str, decl: str) -> Any:
    """The literal value of ``<decl> = <value>`` (which may span lines), or None."""
    m = re.search(re.escape(decl) + r"\s*=\s*", attrs)
    if not m:
        return None
    j = m.end()
    if attrs[j] in "([":
        raw = attrs[j:_balanced(attrs, j, attrs[j], ")" if attrs[j] == "(" else "]")]
    else:
        raw = re.match(r"[^\n]+", attrs[j:]).group(0)
    return ast.literal_eval(raw.strip())


def _local_matrix(attrs: str) -> np.ndarray:
    """Column-vector 4x4 local -> parent, composed in ``xformOpOrder`` (USD: the first op is the
    outermost). USD writes ``matrix4d`` row-major for ROW vectors, hence the transpose."""
    matrix = np.eye(4)
    for op in _attr(attrs, "uniform token[] xformOpOrder") or []:
        if op == "xformOp:transform":
            step = np.array(_attr(attrs, "matrix4d xformOp:transform"), dtype=float).T
        elif op == "xformOp:scale":
            step = np.diag([*_attr(attrs, "double3 xformOp:scale"), 1.0])
        elif op == "xformOp:translate":
            step = np.eye(4)
            step[:3, 3] = _attr(attrs, "double3 xformOp:translate")
        else:
            raise ValueError(f"unsupported xformOp {op!r} (RoomPlan writes transform and scale)")
        matrix = matrix @ step
    return matrix


def _meta_str(meta: str, key: str) -> str | None:
    m = re.search(r"string\s+" + key + r'\s*=\s*"([^"]*)"', meta)
    return m.group(1) if m else None


def _references(meta: str) -> list[str]:
    m = re.search(r"references\s*=\s*", meta)
    if not m:
        return []
    j = m.end()
    if meta[j] == "[":
        return re.findall(r"@([^@]+)@", meta[j:_balanced(meta, j, "[", "]")])
    one = re.match(r"@([^@]+)@", meta[j:])
    return [one.group(1)] if one else []


def _shapes(zf: zipfile.ZipFile, layer: str, prims: list[dict], parent: np.ndarray,
            cache: dict[str, str], out: list[dict]) -> None:
    """Every Cube/Mesh leaf under ``prims`` with its world matrix, following `references` into
    the component layers (the only composition arc RoomPlan uses)."""
    for prim in prims:
        world = parent @ _local_matrix(prim["attrs"])
        for ref in _references(prim["meta"]):
            path = str(PurePosixPath(PurePosixPath(layer).parent / ref)) if ref.startswith(".") \
                else ref
            if path not in cache:
                cache[path] = zf.read(path).decode("utf-8")
            text = cache[path]
            default = re.search(r'defaultPrim\s*=\s*"([^"]+)"', text)
            roots = _prims(text)
            if not roots:
                continue
            root = next((r for r in roots if default and r["name"] == default.group(1)), roots[0])
            root_world = world @ _local_matrix(root["attrs"])
            info = {"file": path, "category": _meta_str(root["meta"], "Category")}
            for shape in root["children"]:
                if shape["type"] in ("Cube", "Mesh"):
                    out.append({**info, "name": shape["name"], "type": shape["type"],
                                "world": root_world @ _local_matrix(shape["attrs"]),
                                "attrs": shape["attrs"]})
        if prim["type"] in ("Cube", "Mesh"):
            out.append({"file": layer, "category": None, "name": prim["name"],
                        "type": prim["type"], "world": world, "attrs": prim["attrs"]})
        _shapes(zf, layer, prim["children"], world, cache, out)


# ── geometry ─────────────────────────────────────────────────────────────────────────────────────


def _r(value: float) -> float:
    return round(float(value), _ROUND)


def _box_frame(matrix: np.ndarray, prefer_long: bool):
    """(centre, along, width, depth, height) of a Z-up box matrix with its extents baked into the
    columns — `layout/shell._box_frame`, so this reads every box the way the pipeline does."""
    cols = [matrix[:3, k] for k in range(3)]
    lengths = [float(np.linalg.norm(c)) for c in cols]
    units = [c / (n if n > 1e-12 else 1.0) for c, n in zip(cols, lengths)]
    vertical = max(range(3), key=lambda k: abs(units[k][2]))
    horizontal = [k for k in range(3) if k != vertical]
    if prefer_long and lengths[horizontal[1]] > lengths[horizontal[0]]:
        horizontal = horizontal[::-1]
    wi, di = horizontal
    along = np.array([cols[wi][0], cols[wi][1], 0.0])
    if np.linalg.norm(along) < 1e-9:
        along = np.array([1.0, 0.0, 0.0])
    along = along / np.linalg.norm(along)
    return matrix[:3, 3], along, lengths[wi], lengths[di], lengths[vertical]


def _outline(verts: np.ndarray, faces: list[list[int]]) -> list[tuple[float, float]]:
    """The boundary loop of the floor slab's TOP face as an ordered XY polygon. The slab's mesh
    also has sides and a bottom; an edge used by exactly one top triangle is on the boundary."""
    ztop = float(verts[:, 2].max())
    top = [f for f in faces if all(abs(verts[i, 2] - ztop) < 1e-6 for i in f)]
    edges: dict[tuple, int] = {}
    for f in top:
        for a, b in zip(f, [*f[1:], f[0]]):
            key = tuple(sorted((tuple(np.round(verts[a, :2], 4)), tuple(np.round(verts[b, :2], 4)))))
            edges[key] = edges.get(key, 0) + 1
    boundary = [e for e, n in edges.items() if n == 1]
    if not boundary:
        return []
    links: dict[tuple, list[tuple]] = {}
    for a, b in boundary:
        links.setdefault(a, []).append(b)
        links.setdefault(b, []).append(a)
    start = boundary[0][0]
    loop, prev, cur = [start], None, start
    for _ in range(len(boundary) + 1):
        nxt = [p for p in links[cur] if p != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        loop.append(cur)
    pts = np.array(loop)
    keep = []
    for i in range(len(pts)):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
        if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-5:
            keep.append((_r(b[0]), _r(b[1])))
    return keep


def inside_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd ray casting for many points against one polygon."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        crosses = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < at)
    return inside


def wall_loops(walls) -> list[list[tuple[float, float]]]:
    """The walls chained end to end into closed loops: the floor outline of an export with no
    floor mesh (iOS 16 writes none). Walls are neither named in perimeter order nor all running
    the same way round, so each step goes on to whichever unused wall has an END nearest the
    corner reached, and a loop closes when its own start is nearer than any wall left."""
    left = [w for w in walls if w.length > 1e-3]
    loops = []
    while left:
        wall = left.pop(0)
        loop, at = [wall.start, wall.end], wall.end
        while left:
            d, k, flip = min(min((math.dist(at, w.start), k, False), (math.dist(at, w.end), k, True))
                             for k, w in enumerate(left))
            if math.dist(at, loop[0]) <= d:
                break
            wall = left.pop(k)
            near, far = (wall.end, wall.start) if flip else (wall.start, wall.end)
            if d > CORNER_TOL:
                loop.append(near)               # a gap in the walls: the outline follows them
            loop.append(far)
            at = far
        if math.dist(loop[-1], loop[0]) <= CORNER_TOL:
            loop.pop()
        loops.append(loop)
    return loops


def _category(handle: str) -> str:
    """``Sink_Storage0`` -> ``sink_storage``: the rule `layout.adapter._category` applies."""
    return "".join(c for c in handle if not c.isdigit()).strip("_").lower()


# ── entry point ──────────────────────────────────────────────────────────────────────────────────


def parse_room(path: Path | str) -> RoomPlan:
    """Read a RoomPlan ``room.usdz`` (or the scan folder holding one)."""
    path = Path(path)
    if path.is_dir():
        found = find_room_usdz(path)
        if found is None:
            raise FileNotFoundError(f"no room.usdz in {path} (looked for {', '.join(USDZ_CANDIDATES)})")
        path = found
    # Whatever a damaged export trips over — a layer the root references but the zip lacks
    # (KeyError), a truncated value, a file that is not a zip at all — is one clear ValueError
    # naming the file, not a traceback from the middle of the reader.
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if any(n.endswith((".usdc", ".usd")) for n in names):
                raise ValueError(f"{path} holds binary USD layers; only RoomPlan's ASCII export "
                                 "is read")
            root_name = next((n for n in names if n.endswith(".usda") and "/" not in n), None)
            if root_name is None:
                raise ValueError(f"{path} has no root .usda layer")
            root_text = zf.read(root_name).decode("utf-8")
            header = root_text[:max(0, root_text.find("def "))]
            up = re.search(r'upAxis\s*=\s*"(\w)"', header)
            mpu = re.search(r"metersPerUnit\s*=\s*([0-9.eE+-]+)", header)
            if (up and up.group(1) != "Y") or (mpu and float(mpu.group(1)) != 1.0):
                raise ValueError(f"{path}: expected RoomPlan's upAxis = Y and metersPerUnit = 1")
            shapes: list[dict] = []
            _shapes(zf, root_name, _prims(root_text), np.eye(4), {}, shapes)
        return _assemble(path, shapes)
    except ValueError as exc:
        if str(path) in str(exc):
            raise
        raise ValueError(f"{path} is not a readable RoomPlan export: {exc}") from exc
    except (KeyError, IndexError, TypeError, AttributeError, SyntaxError,
            zipfile.BadZipFile) as exc:
        what = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, KeyError) and str(exc.args[0] if exc.args else "").startswith("There "):
            what = f"a layer it references is missing ({exc.args[0]})"   # zipfile's own words
        raise ValueError(f"{path} is not a readable RoomPlan export: {what}") from exc


def _assemble(path: Path, shapes: list[dict]) -> RoomPlan:
    walls: dict[str, Wall] = {}
    raw_openings: list[dict] = []
    objects: dict[str, RoomObject] = {}
    floors: list[tuple[str, np.ndarray, list[list[int]]]] = []
    skipped: list[str] = []
    warnings: list[str] = []
    wall_bottoms: list[float] = []

    for s in shapes:
        zup = ARKIT_TO_Z_UP @ s["world"]
        category = s["category"] or re.sub(r"\d+$", "", s["name"])
        if s["type"] == "Mesh":
            if not category.startswith("Floor"):
                skipped.append(s["name"])          # a mesh-mode export: no box to build from
                warnings.append(f"{s['name']} is a mesh, not a box: nothing is built for it"
                                + (" — the room has a gap there" if category == "Wall" else ""))
                continue
            pts = np.array(_attr(s["attrs"], "point3f[] points"), dtype=float).reshape(-1, 3)
            index = list(_attr(s["attrs"], "int[] faceVertexIndices") or [])
            counts = list(_attr(s["attrs"], "int[] faceVertexCounts") or [])
            faces, k = [], 0
            for c in counts:
                faces.append(index[k:k + c])
                k += c
            floors.append((s["name"], (zup @ np.c_[pts, np.ones(len(pts))].T).T[:, :3], faces))
            continue
        size = float(_attr(s["attrs"], "double size") or 2.0)
        baked = zup @ np.diag([size, size, size, 1.0])
        centre, along, width, depth, height = _box_frame(baked, prefer_long=category == "Wall")
        if category == "Wall":
            half = along[:2] * width / 2.0
            walls[s["name"]] = Wall(
                name=s["name"],
                start=(_r(centre[0] - half[0]), _r(centre[1] - half[1])),
                end=(_r(centre[0] + half[0]), _r(centre[1] + half[1])),
                thickness=_r(depth), height=_r(height))
            wall_bottoms.append(float(centre[2]) - height / 2.0)
        elif re.match(r"(Door|Window|Opening)", category):
            m = re.search(r"Walls/(Wall\d+)/", s["file"])
            raw_openings.append({"name": s["name"], "category": category, "centre": centre,
                                 "width": width, "height": height,
                                 "wall": m.group(1) if m else None})
        else:
            box = s["world"] @ np.diag([size, size, size, 1.0])
            objects[s["name"]] = RoomObject(
                handle=s["name"], category=_category(s["name"]), roomplan_category=category,
                center=(_r(centre[0]), _r(centre[1]), _r(centre[2])),
                size=(_r(width), _r(depth), _r(height)),
                yaw=round(math.degrees(math.atan2(along[1], along[0])), 2),
                matrix_arkit=tuple(tuple(round(float(v), 6) for v in row) for row in box))

    # The floor's own mesh says where the floor is, as the layout adapter reads it; walls are the
    # fallback for an export without one. The ceiling is the tallest wall above that floor.
    floor = _floor(floors, walls, warnings)
    if floor is not None:
        floor_z = float(floor.verts[:, 2].min())
    else:
        floor_z = min(wall_bottoms, default=0.0)
    ceiling_z = floor_z + max((w.height for w in walls.values()), default=2.5)

    openings: dict[str, Opening] = {}
    for raw in raw_openings:
        wall = walls.get(raw["wall"]) if raw["wall"] else None
        if wall is None:
            wall = _nearest_wall(walls, raw["centre"][:2])
        if wall is None:
            skipped.append(raw["name"])
            warnings.append(f"{raw['name']} has no wall to be cut into: left out")
            continue
        kind_word = re.match(r"(Door|Window|Opening)", raw["category"]).group(1)
        offset = float(np.dot(raw["centre"][:2] - np.asarray(wall.start), wall.along))
        sill = float(raw["centre"][2] - raw["height"] / 2.0 - floor_z)
        state = re.search(r"Isopen:\s*(\w+)", raw["category"], re.IGNORECASE)
        index = re.sub(r"^\D+", "", raw["name"]) or "0"
        handle = f"{wall.name}_{kind_word}_{index}"
        openings[handle] = Opening(
            handle=handle, source=raw["name"], wall=wall.name, kind=OPENING_KINDS[kind_word],
            offset=_r(offset), width=_r(raw["width"]), height=_r(raw["height"]),
            sill=_r(max(0.0, sill) if sill > -0.02 else sill),
            is_open=(state.group(1).lower() == "true") if state else None)

    return RoomPlan(source=path, floor_z=round(floor_z, _ROUND), ceiling_z=round(ceiling_z, _ROUND),
                    walls=dict(sorted(walls.items(), key=lambda kv: _numeric(kv[0]))),
                    openings=openings,
                    objects=dict(sorted(objects.items())),
                    floor=floor, skipped=skipped, warnings=warnings)


def _floor(floors: list[tuple[str, np.ndarray, list[list[int]]]], walls: dict[str, Wall],
           warnings: list[str]) -> Floor | None:
    """Every floor slab as one `Floor` — or None when there is none worth believing.

    A multi-room capture has a slab per room: all of them count, the lowest sets the floor height
    and the floor is their union. The plate then has to pass the ingest adapter's own plausibility
    gate (`adapter.MIN_PLATE_FRACTION`, measured as the adapter measures it): a slab covering
    almost none of what its walls enclose is a collapsed transform, not a floor, and the pipeline
    drops it — so the quick room does too, or its layout repair would be solving a different room.
    """
    verts, faces, outlines = [], [], []
    for _name, world, mesh_faces in floors:
        base = sum(len(v) for v in verts)
        verts.append(world)
        faces += [[base + i for i in f] for f in mesh_faces]
        loop = _outline(world, mesh_faces)
        if len(loop) >= 3:
            outlines.append(tuple(loop))
    if not verts or not len(np.concatenate(verts)):
        return None
    all_verts = np.concatenate(verts)
    tris = [[f[0], f[i], f[i + 1]] for f in faces for i in range(1, len(f) - 1)]
    plate = _polygon_area(all_verts.tolist(), tris)
    ends = np.array([p for w in walls.values() for p in (w.start, w.end)], dtype=float)
    if len(ends):
        span = ends.max(axis=0) - ends.min(axis=0)
        enclosed = float(span[0] * span[1])
        if enclosed > 0 and plate < MIN_PLATE_FRACTION * enclosed:
            warnings.append(f"{'+'.join(n for n, _w, _f in floors)}: a {plate:.2f} m2 floor plate "
                            f"under {enclosed:.1f} m2 of walls is not believable — ignored, as "
                            "ingest ignores it; the walls give the floor instead")
            return None
    return Floor(verts=all_verts, faces=tuple(tuple(f) for f in faces), outlines=tuple(outlines))


def _numeric(name: str) -> tuple[str, int]:
    """Wall2 before Wall10."""
    m = re.match(r"(\D*)(\d*)", name)
    return m.group(1), int(m.group(2) or 0)


def _nearest_wall(walls: dict[str, Wall], point: np.ndarray) -> Wall | None:
    """The wall an opening outside any wall folder belongs to: the one whose segment it is
    closest to (an opening's centre sits ON its wall's plane)."""
    best, best_d = None, math.inf
    for wall in walls.values():
        s = np.asarray(wall.start)
        t = float(np.clip(np.dot(point - s, wall.along), 0.0, wall.length))
        d = float(np.linalg.norm(point - (s + wall.along * t)))
        if d < best_d:
            best, best_d = wall, d
    return best if best_d < 0.5 else None
