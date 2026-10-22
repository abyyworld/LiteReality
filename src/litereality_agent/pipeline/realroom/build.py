"""build.py — the quick room with each generated object's mesh in place of its template's look.

The objects stage changes what the room LOOKS like and nothing about how it behaves. Every body
keeps the quick room's name, joint, mass, friction and template parts — the parts become the
body's COLLIDERS (group 3: never drawn), so a table keeps its legs and the chairs tucked under it
still fit — and gains one more geom: the generated mesh, visual only (group 2, ``contype=0
conaffinity=0 density=0``). Same names, same masses, same contacts: a live swap from the quick
version to this one (and from one object's preview to its final) carries every body's state by
name and nobody feels a thing, except that the room now looks like their room.

    out/                        run/<scan>/objects by default
      scene.xml                 the room as it stands now: the quick room, then one version per
                                batch of objects as they arrive (written atomically, meshes first)
      quick_scene.xml           the quick room it is built on (`BASE_NAME`), kept for rebuilds
      quick.json, ...           the quick room's own report and assets
      generated/                fitted meshes (OBJ) and textures (PNG), content-named
      glb/                      what the generator returned, per object and pass (a cache)
      crops/                    the photos it was given (+ crops.json)
      objects.json              this stage's report (`REPORT_NAME`)

Only what ``scene.xml`` names is part of a version (`room_ops.roomcall.hosted.scene_files`), so
the cache and the crops never travel with a publish.

Other sources can take an object over entirely (`Source`, `Arrival`): Infinigen's openable
furniture (`articulated`) replaces a cabinet's template — colliders, links, joints and all —
rather than dressing it, so such a handle is ``taken`` (TRELLIS never generates it) and its
arrival is a scene EDIT applied to every version after it lands. If that source fails an object
(or stops altogether), the object is given back and TRELLIS generates its look after all, so
asking for more never leaves a room worse than `--realistic` alone. Producers fail on their
own: a generator call that raises as a whole stops only its own objects.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import queue
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from litereality_agent.pipeline.quickroom import REPORT_NAME as QUICK_REPORT
from litereality_agent.pipeline.quickroom import SCENE_NAME, find_room_usdz, write_quick_scene
from litereality_agent.pipeline.realroom import crops as crop_views
from litereality_agent.pipeline.realroom import fit as fitting
from litereality_agent.pipeline.realroom.generate import PASSES, Generator, check_passes

BASE_NAME = "quick_scene.xml"
REPORT_NAME = "objects.json"
ASSET_DIR = "generated"
GLB_DIR = "glb"
CROP_DIR = "crops"
DEFAULT_DIR = "objects"         # under the scan's run folder
STAGE = "objects"
# Versions whose files are kept on disk: the one just written and the two before it, which a
# server may still be loading or serving while the next is announced.
KEEP_VERSIONS = 3
# Seconds between versions while objects stream in: each is a compile, a geometry build and (with
# `--publish`) an upload, and the people in the room see a toast for each.
MIN_INTERVAL = 10.0
# A floor-standing object's box ends at most this far above the floor; its mesh is put ON it.
GROUND_GAP = 0.08
# Twins: the same category, every side within this (m, or share of the side, the larger) and a
# colour this close (sRGB 0-1, Euclidean) — the four identical chairs of a meeting room. An object
# the capture saw too little of takes its twin's mesh instead of keeping its template.
TWIN_SIZE = (0.04, 0.08)
TWIN_COLOUR = 0.3
VISUAL_ATTRS = {"group": "2", "contype": "0", "conaffinity": "0", "density": "0"}
COLLIDER_GROUP = "3"


# ── the composition ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Installed:
    """One object's current look in the scene: which pass it is from and its files."""

    handle: str
    pass_name: str
    files: fitting.FittedFiles
    summary: dict = field(default_factory=dict)


def asset_dirs(root: ET.Element) -> tuple[str, str]:
    """(mesh folder, texture folder) MuJoCo resolves files against, relative to the scene's
    folder: the compiler's meshdir / texturedir, else its assetdir, else the scene's own."""
    compiler = root.find("compiler")
    get = (lambda k: compiler.get(k)) if compiler is not None else (lambda _k: None)
    asset = get("assetdir") or ""
    return get("meshdir") or asset, get("texturedir") or asset


def _unique(taken: set[str], name: str) -> str:
    out, k = name, 1
    while out in taken:
        k += 1
        out = f"{name}_{k}"
    taken.add(out)
    return out


def compose(base: ET.Element, installed: dict[str, Installed],
            edits: Sequence[Callable[[ET.Element], Any]] = ()) -> tuple[ET.Element, list[str]]:
    """The base quick room with each installed object's template parts made colliders and its
    generated mesh added as the body's look, then each of ``edits`` (another source's objects,
    `Arrival`) applied. Returns (scene root, handles not found)."""
    root = copy.deepcopy(base)
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(next((i + 1 for i, el in enumerate(root)
                          if el.tag in ("compiler", "option", "size", "visual", "default")), 0),
                    asset)
    names: dict[str, set[str]] = {}
    for el in root.iter():
        if el.get("name"):
            names.setdefault(el.tag, set()).add(el.get("name"))
    bodies = {b.get("name"): b for b in root.iter("body") if b.get("name")}
    missing = []
    for handle in sorted(installed):
        item = installed[handle]
        body = bodies.get(handle)
        if body is None:
            missing.append(handle)
            continue
        for geom in body.findall("geom"):
            if geom.get("group") != VISUAL_ATTRS["group"]:
                geom.set("group", COLLIDER_GROUP)
        mesh = _unique(names.setdefault("mesh", set()), f"rr_{handle}")
        ET.SubElement(asset, "mesh", name=mesh, file=f"{ASSET_DIR}/{item.files.mesh}",
                      inertia="shell")
        attrs = {"name": _unique(names.setdefault("geom", set()), f"{handle}_visual"),
                 "type": "mesh", "mesh": mesh, **VISUAL_ATTRS}
        material = _unique(names.setdefault("material", set()), f"rr_{handle}")
        if item.files.texture:
            texture = _unique(names.setdefault("texture", set()), f"rr_{handle}")
            ET.SubElement(asset, "texture", name=texture, type="2d",
                          file=f"{ASSET_DIR}/{item.files.texture}")
            ET.SubElement(asset, "material", name=material, texture=texture,
                          specular="0.15", shininess="0.25")
        else:
            rgba = " ".join(f"{v:.4g}" for v in item.files.rgba)
            ET.SubElement(asset, "material", name=material, rgba=rgba, specular="0.15",
                          shininess="0.25")
        attrs["material"] = material
        ET.SubElement(body, "geom", attrs)
    for edit in edits:
        edit(root)
    ET.indent(root, space="  ")
    return root, missing


@dataclass(frozen=True)
class _Done:
    """A producer of `generate_room` has finished (``source``: an extra `Source`'s name)."""

    source: str | None = None


@dataclass(frozen=True)
class _Stopped:
    """A generator call raised as a whole: its ``handles`` get nothing more from it."""

    handles: tuple[str, ...]
    passes: tuple[str, ...]
    error: str


@dataclass
class Arrival:
    """An object from a `Source` other than the generator, ready (``edit``) or failed
    (``error``). ``edit`` changes a scene root in place — every version from now on is composed
    with it — and must only name files it has already written."""

    handle: str
    source: str
    edit: Callable[[ET.Element], Any] | None
    seconds: float = 0.0
    error: str = ""
    summary: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.edit is not None and not self.error


class Source(Protocol):
    """Objects from somewhere else than TRELLIS (`articulated.ArticulatedSource`)."""

    name: str

    def plan(self, room: RealRoom) -> Sequence[str]:
        """Decide what it will build, add the handles TRELLIS must leave alone to
        ``room.taken`` and return every handle it will build. Called again, the same answer."""
        ...

    def stream(self, room: RealRoom) -> Iterator[Arrival]:
        """Each planned object as it is ready, a failed one as an `Arrival` with its ``error``
        (it then goes back to the generator). Never raises for one object."""
        ...


# ── the room ─────────────────────────────────────────────────────────────────────────────────────


class RealRoom:
    """The objects stage of one scan, built up object by object.

    Opening it builds the quick room into ``out_dir`` (unless one built from this scan is there
    already) and keeps it as the base; `add` fits one generated GLB into its body and writes its
    files; `write` writes the scene as it stands — meshes first, then ``scene.xml`` atomically,
    after checking it compiles — so whatever watches the folder only ever sees whole versions.
    """

    def __init__(self, scan_dir: Path | str, out_dir: Path | str, *, label: str | None = None,
                 colours: bool = True, textures: bool = True, rebuild: bool = False) -> None:
        self.scan_dir = Path(scan_dir)
        if self.scan_dir.is_file():
            self.scan_dir = self.scan_dir.parent
        self.out_dir = Path(out_dir)
        self.label = label
        self.base_path = self.out_dir / BASE_NAME
        self.scene_path = self.out_dir / SCENE_NAME
        if rebuild or not self._base_is_current():
            self._build_base(colours=colours, textures=textures)
        self.base = ET.parse(self.base_path).getroot()
        quick_report = self.out_dir / QUICK_REPORT
        self.quick = json.loads(quick_report.read_text()) if quick_report.is_file() else {}
        self.floor_z = float(self.quick.get("floor_z", 0.0))
        self.boxes = crop_views.boxes_from_scene(self.base_path, self.quick)
        mesh_dir, tex_dir = asset_dirs(self.base)
        self.mesh_dir = self.out_dir / mesh_dir / ASSET_DIR
        self.texture_dir = self.out_dir / tex_dir / ASSET_DIR
        self.installed: dict[str, Installed] = {}
        self.history: list[set[str]] = []
        self.crops: dict[str, crop_views.ObjectCrop] = {}
        self.versions = 0
        self.errors: dict[str, str] = {}
        self.borrowed: dict[str, str] = {}
        # Handles another `Source` builds (TRELLIS leaves them alone), and its scene edits.
        self.taken: set[str] = set()
        self.edits: dict[str, Callable[[ET.Element], Any]] = {}
        self.replaced: dict[str, dict] = {}

    # -- the base -----------------------------------------------------------------------------

    def _base_is_current(self) -> bool:
        usdz = find_room_usdz(self.scan_dir)
        if not self.base_path.is_file() or usdz is None:
            return False
        if not (self.out_dir / QUICK_REPORT).is_file():
            return False
        return self.base_path.stat().st_mtime >= usdz.stat().st_mtime

    def _build_base(self, *, colours: bool, textures: bool) -> None:
        # Built in place: the quick room's photos (walls, floor, ceiling) are written next to its
        # scene.xml and named relative to it, so this stage's scene lives in the same folder.
        scene = write_quick_scene(self.scan_dir, self.out_dir, label=self.label, colours=colours,
                                  textures=textures)
        tmp = self.base_path.with_name(self.base_path.name + ".tmp")
        tmp.write_bytes(scene.read_bytes())
        os.replace(tmp, self.base_path)

    # -- the objects --------------------------------------------------------------------------

    def generated_handles(self) -> list[str]:
        return [h for h, box in self.boxes.items() if box.generated() and h not in self.taken]

    def make_crops(self, *, grabcut: bool = True) -> dict[str, crop_views.ObjectCrop]:
        """Every generated object's photo and views (saved under crops/)."""
        walls = []
        try:
            from litereality_agent.pipeline.quickroom import parse_room

            usdz = find_room_usdz(self.scan_dir)
            if usdz is not None:
                walls = [(w.start, w.end) for w in parse_room(usdz).walls.values()]
        except Exception:  # noqa: BLE001 — walls only sharpen the masks
            walls = []
        self.crops = crop_views.object_crops(self.scan_dir, self.boxes, self.floor_z,
                                             walls=walls, grabcut=grabcut,
                                             handles=self.generated_handles())
        crop_views.save_crops(self.crops, self.out_dir / CROP_DIR)
        return self.crops

    def add_crops(self, handles: Sequence[str], *, grabcut: bool = True
                  ) -> dict[str, crop_views.ObjectCrop]:
        """Photos and views for ``handles`` too, kept with the others: an object another
        `Source` claimed and then could not build goes back to the generator mid-run, and it was
        left out of `make_crops` while it was taken."""
        walls = []
        try:
            from litereality_agent.pipeline.quickroom import parse_room

            usdz = find_room_usdz(self.scan_dir)
            if usdz is not None:
                walls = [(w.start, w.end) for w in parse_room(usdz).walls.values()]
        except Exception:  # noqa: BLE001 — walls only sharpen the masks
            walls = []
        more = crop_views.object_crops(self.scan_dir, self.boxes, self.floor_z, walls=walls,
                                       grabcut=grabcut,
                                       handles=[h for h in handles if h in self.boxes])
        self.crops.update(more)
        crop_views.save_crops(self.crops, self.out_dir / CROP_DIR)
        return more

    def placement(self, handle: str) -> fitting.Placement:
        box = self.boxes[handle]
        floor_local = None
        if box.bottom - self.floor_z < GROUND_GAP:
            # On the floor, in the body frame (whose origin is the body's position).
            floor_local = self.floor_z - float(box.body_pos[2])
        return fitting.Placement(half=box.half, centre=box.local_centre,
                                 world_centre=box.centre, world_yaw=box.yaw,
                                 floor_local=floor_local)

    def add(self, handle: str, glb: bytes | str | Path, *, pass_name: str = "final") -> Installed:
        """Fit one generated GLB into ``handle``'s body and write its files (not yet the scene).
        A pass older than the one installed is ignored: a late preview never replaces a final."""
        if handle not in self.boxes:
            raise KeyError(f"{handle} is not a body of the quick room")
        have = self.installed.get(handle)
        if have is not None and PASSES[have.pass_name].rank > PASSES[pass_name].rank:
            return have
        started = time.perf_counter()
        crop = self.crops.get(handle)
        fitted = fitting.fit(glb, self.placement(handle), views=crop.views if crop else ())
        files = fitting.write_files(fitted, handle, self.mesh_dir, self.texture_dir)
        self._check_compiles(handle, files)
        summary = {"pass": pass_name, **fitted.summary(),
                   "fit_seconds": round(time.perf_counter() - started, 3)}
        self.installed[handle] = Installed(handle, pass_name, files, summary)
        self.errors.pop(handle, None)
        return self.installed[handle]

    def _check_compiles(self, handle: str, files: fitting.FittedFiles) -> None:
        """MuJoCo must take the mesh on its own (a degenerate one is refused here, as this
        object's failure, rather than failing every version of the room from now on)."""
        import mujoco

        asset = ET.Element("asset")
        ET.SubElement(asset, "mesh", name="m", inertia="shell",
                      file=str(self.mesh_dir / files.mesh))
        geom = {"type": "mesh", "mesh": "m", **VISUAL_ATTRS}
        if files.texture:
            ET.SubElement(asset, "texture", name="t", type="2d",
                          file=str(self.texture_dir / files.texture))
            ET.SubElement(asset, "material", name="mt", texture="t")
            geom["material"] = "mt"
        root = ET.Element("mujoco")
        root.append(asset)
        body = ET.SubElement(ET.SubElement(root, "worldbody"), "body")
        ET.SubElement(body, "geom", geom)
        ET.SubElement(body, "geom", type="box", size="0.1 0.1 0.1")
        try:
            mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        except Exception as exc:
            for name, folder in ((files.mesh, self.mesh_dir), (files.texture, self.texture_dir)):
                if name and not any(name in (i.files.mesh, i.files.texture)
                                    for i in self.installed.values()):
                    (folder / name).unlink(missing_ok=True)
            raise ValueError(f"MuJoCo refuses {handle}'s mesh: {exc}") from None

    def write(self, *, check: bool = True) -> Path:
        """Write the scene as it stands now and return its path."""
        root, missing = compose(self.base, self.installed, list(self.edits.values()))
        xml = ET.tostring(root, encoding="unicode") + "\n"
        tmp = self.scene_path.with_name(self.scene_path.name + ".tmp")
        tmp.write_text(xml, encoding="utf-8")
        if check:
            try:
                import mujoco

                mujoco.MjModel.from_xml_path(str(tmp))
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
        os.replace(tmp, self.scene_path)    # a watcher never reads half a scene
        self.versions += 1
        files = {i.files.mesh for i in self.installed.values()}
        files |= {i.files.texture for i in self.installed.values() if i.files.texture}
        self.history.append(files)
        self._prune()
        self.write_report(missing)
        return self.scene_path

    def _prune(self) -> None:
        keep = set().union(*self.history[-KEEP_VERSIONS:]) if self.history else set()
        for folder in {self.mesh_dir, self.texture_dir}:
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if path.is_file() and path.name not in keep and path.suffix in (".obj", ".png"):
                    path.unlink(missing_ok=True)

    def write_report(self, missing: Sequence[str] = ()) -> None:
        """objects.json: what each object is now, what failed, what is still a template."""
        handles = self.generated_handles()
        report = {
            "stage": STAGE,
            "label": self.label or self.quick.get("label"),
            "scene": SCENE_NAME,
            "base": BASE_NAME,
            "versions_written": self.versions,
            "objects": {h: i.summary for h, i in sorted(self.installed.items())},
            "template": sorted(h for h in handles if h not in self.installed),
            "borrowed": dict(self.borrowed),
            "replaced": dict(self.replaced),
            "errors": self.errors,
            "missing": list(missing),
            "crops": {h: c.summary() for h, c in self.crops.items()},
            "note": "generated meshes are the LOOK only: every body keeps the quick room's "
                    "template parts as its colliders, and its mass, joint and friction",
        }
        path = self.out_dir / REPORT_NAME
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, path)


def write_realistic_scene(scan_dir: Path | str, out_dir: Path | str,
                          generated: dict[str, bytes | str | Path], *, pass_name: str = "final",
                          label: str | None = None, views: bool = True) -> Path:
    """The quick room of ``scan_dir`` with each of ``generated`` (handle -> GLB bytes or path)
    as its body's look, written into ``out_dir``; returns the scene.xml. ``views`` uses the
    capture's cameras to choose each mesh's quarter turn (`fit.choose_yaw`)."""
    room = RealRoom(scan_dir, out_dir, label=label)
    if views and generated:
        room.make_crops()
    for handle, glb in generated.items():
        room.add(handle, glb, pass_name=pass_name)
    return room.write()


# ── generating, version by version ───────────────────────────────────────────────────────────────


def _glb_cache(room: RealRoom, handle: str, pass_name: str, image: bytes) -> Path:
    digest = hashlib.sha256(image).hexdigest()[:12]
    return room.out_dir / GLB_DIR / f"{handle}.{pass_name}.{digest}.glb"


@dataclass
class Progress:
    """What `generate_room` did: per object the passes installed and what failed."""

    scene: Path
    versions: int = 0
    installed: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    cached: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    borrowed: dict[str, str] = field(default_factory=dict)    # handle -> the twin it looks like
    seconds: float = 0.0


def _hex_rgb(value: str | None) -> np.ndarray | None:
    try:
        text = str(value).lstrip("#")
        return np.array([int(text[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0
    except (TypeError, ValueError):
        return None


def twin_of(room: RealRoom, handle: str, donors: list[str]) -> str | None:
    """The donor most like ``handle``: same category, near-identical size, a similar colour."""
    box = room.boxes[handle]
    colours = ((room.quick.get("colours") or {}).get("objects") or {})
    mine = _hex_rgb((colours.get(handle) or {}).get("rgb"))
    best, best_d = None, float("inf")
    for donor in donors:
        other = room.boxes.get(donor)
        if other is None or donor == handle or other.category != box.category:
            continue
        a, b = 2.0 * box.half, 2.0 * other.half
        if np.any(np.abs(a - b) > np.maximum(TWIN_SIZE[0], TWIN_SIZE[1] * np.maximum(a, b))):
            continue
        theirs = _hex_rgb((colours.get(donor) or {}).get("rgb"))
        d = float(np.linalg.norm(mine - theirs)) if mine is not None and theirs is not None else 0.0
        if d <= TWIN_COLOUR and d < best_d:
            best, best_d = donor, d
    return best


def generate_room(room: RealRoom, generator: Generator, *, passes: Sequence[str] = ("preview",
                                                                                     "final"),
                  on_version: Callable[[Path, dict], Any] | None = None,
                  min_interval: float | None = None, log: Callable[[str], Any] = print,
                  reuse: bool = True, extras: Sequence[Source] = ()) -> Progress:
    """Generate every object of ``room`` and write a new version as they arrive.

    Crops first (cached GLBs for the same photo are installed without asking again), then the
    generator streams results; a version is written whenever something new is installed and the
    last one is ``min_interval`` seconds old, and once more at the end. ``on_version(scene,
    info)`` is called after each (`--publish` uploads it there). ``extras`` are other sources
    (`Source`): what they plan is theirs, and their arrivals stream into the same versions.
    Never raises for one object."""
    started = time.monotonic()
    passes = check_passes(passes)
    min_interval = MIN_INTERVAL if min_interval is None else float(min_interval)
    progress = Progress(room.scene_path)
    # What each other source took from the generator: handed back if that source fails it.
    claimed: dict[str, set[str]] = {}
    for extra in extras:
        try:
            claimed[extra.name] = set(extra.plan(room) or ()) & room.taken
        except Exception as exc:  # noqa: BLE001 — that source builds nothing; TRELLIS does
            log(f"  {extra.name}: could not plan ({type(exc).__name__}: {exc})")
            claimed[extra.name] = set()
    crops = room.crops or room.make_crops()
    images: dict[str, bytes] = {}
    poor: list[str] = []
    for handle in room.generated_handles():
        crop = crops.get(handle)
        if crop is None or not crop.ok:
            progress.skipped[handle] = crop.note if crop else "no crop"
        elif not crop.good:
            poor.append(handle)
        else:
            images[handle] = crop.png()
    borrowers: dict[str, list[str]] = {}
    for handle in poor:
        donor = twin_of(room, handle, list(images))
        if donor is None:
            progress.skipped[handle] = (f"the capture saw too little of it (best view "
                                        f"{crops[handle].score:.2f}) and nothing like it")
            continue
        progress.borrowed[handle] = room.borrowed[handle] = donor
        borrowers.setdefault(donor, []).append(handle)

    def install(handle: str, glb, pass_name: str) -> None:
        """``handle``'s mesh, and its twins' (each fitted into its own box)."""
        room.add(handle, glb, pass_name=pass_name)
        for twin in borrowers.get(handle, ()):
            try:
                room.add(twin, glb, pass_name=pass_name)
                progress.installed[twin] = pass_name
            except Exception as exc:  # noqa: BLE001 — the twin keeps its template
                room.errors[twin] = f"{pass_name} (from {handle}): {exc}"

    def from_cache(handle: str, png: bytes) -> list[str]:
        """The best pass already generated for this very photo, installed as it is; returns the
        passes after it, which are still to ask for (a cached final makes its preview moot)."""
        best = None
        for p in sorted(passes, key=lambda name: -PASSES[name].rank):
            cached = _glb_cache(room, handle, p, png)
            if not (reuse and cached.is_file()):
                continue
            try:
                install(handle, cached, p)
            except Exception as exc:  # noqa: BLE001 — a bad cache entry is generated again
                log(f"  cache {cached.name} unusable ({exc}); generating it again")
                continue
            best = p
            progress.installed[handle] = p
            progress.cached.append(f"{handle}:{p}")
            break
        return [p for p in passes if best is None or PASSES[p].rank > PASSES[best].rank]

    todo: dict[str, list[str]] = {}
    for handle, png in images.items():
        need = from_cache(handle, png)
        if need:
            todo[handle] = need

    last = time.monotonic()
    dirty = bool(progress.cached)

    def publish(why: str) -> None:
        nonlocal last, dirty
        try:
            scene = room.write()
        except Exception as exc:  # noqa: BLE001 — keep the last good version, keep going
            log(f"  could not write a version ({type(exc).__name__}: {exc})")
            dirty = False
            last = time.monotonic()
            return
        progress.versions += 1
        dirty = False
        info = {"why": why, "installed": dict(progress.installed), "version": progress.versions}
        log(f"  version {progress.versions}: {len(progress.installed)} generated object(s) "
            f"({why})")
        if on_version is not None:
            try:
                on_version(scene, info)
            except Exception as exc:  # noqa: BLE001 — a failed upload must not stop generating
                log(f"  publishing version {progress.versions} failed: {exc}")
        last = time.monotonic()

    if dirty:
        publish("cached objects")

    if todo or extras:
        if todo:
            log(f"  generating {len(todo)} object(s) with {generator.name}: "
                + ", ".join(f"{h} ({'+'.join(p)})" for h, p in sorted(todo.items())))
        # Group by pass list so the generator sees each object's passes in order.
        results: queue.Queue = queue.Queue()
        groups: dict[tuple[str, ...], dict[str, bytes]] = {}
        for handle, need in todo.items():
            groups.setdefault(tuple(need), {})[handle] = images[handle]
        running = 0
        arrived: dict[str, set[str]] = {name: set() for name in claimed}

        def one_group(need: tuple[str, ...], group: dict[str, bytes]) -> None:
            try:
                for result in generator.stream(group, list(need)):
                    results.put(result)
            except BaseException as exc:  # noqa: BLE001 — handed to the loop below
                results.put(_Stopped(tuple(group), need, f"{type(exc).__name__}: {exc}"))
            finally:
                results.put(_Done())

        def one_extra(extra: Source) -> None:
            try:
                for arrival in extra.stream(room):
                    results.put(arrival)
            except BaseException as exc:  # noqa: BLE001 — that source stops; the rest go on
                results.put(Arrival("", extra.name, None, 0.0, f"{type(exc).__name__}: {exc}"))
            finally:
                results.put(_Done(extra.name))

        def start(target: Callable[..., None], *args: Any) -> None:
            nonlocal running
            running += 1                    # counted here, so a producer started late is waited for
            threading.Thread(target=target, args=args, daemon=True,
                             name="realroom-generate").start()

        def give_back(source: str, handles: Sequence[str]) -> None:
            """Objects ``source`` claimed but did not build go to the generator after all, so
            they end up as `--realistic` alone would make them rather than as bare templates."""
            nonlocal dirty
            mine = claimed.get(source, set())
            back = [h for h in handles if h in mine and h not in arrived.get(source, ())]
            if not back:
                return
            for handle in back:
                mine.discard(handle)
                room.taken.discard(handle)
            back = [h for h in back if h in room.boxes and room.boxes[h].generated()]
            if not back:
                return
            try:
                more = room.add_crops(back)
            except Exception as exc:  # noqa: BLE001 — they keep their templates
                for handle in back:
                    progress.skipped[handle] = f"given back by {source}; no crop ({exc})"
                return
            asked: dict[tuple[str, ...], dict[str, bytes]] = {}
            for handle in back:
                crop = more.get(handle)
                if crop is None or not crop.ok or not crop.good:
                    why = (crop.note if crop is not None else "") or "the capture saw too little"
                    progress.skipped[handle] = f"given back by {source}; {why}"
                    continue
                images[handle] = crop.png()
                need = from_cache(handle, images[handle])
                dirty = dirty or handle in progress.installed
                if need:
                    asked.setdefault(tuple(need), {})[handle] = images[handle]
            for need, group in asked.items():
                log(f"  {source} gave back {', '.join(sorted(group))}: generating with "
                    f"{generator.name} ({'+'.join(need)})")
                start(one_group, need, group)

        for item in groups.items():
            start(one_group, *item)
        for extra in extras:
            start(one_extra, extra)
        while running:
            wait = None
            if dirty and math.isfinite(min_interval):
                wait = max(0.0, last + min_interval - time.monotonic())
            try:
                item = results.get(timeout=wait)
            except queue.Empty:
                publish("objects arrived")
                continue
            if isinstance(item, _Done):
                running -= 1
                if item.source is not None:          # what it never answered for goes back
                    give_back(item.source, sorted(claimed.get(item.source, ())))
                continue
            if isinstance(item, _Stopped):
                # One generator call failed as a whole (the app is not deployed, credentials
                # expired): its objects keep what they have; every other producer goes on.
                log(f"  {generator.name} stopped for {', '.join(item.handles)}: {item.error}")
                for handle in item.handles:
                    if progress.installed.get(handle) != item.passes[-1]:
                        progress.failed[f"{handle}:{item.passes[-1]}"] = item.error
                        room.errors.setdefault(handle, f"{generator.name}: {item.error}")
                continue
            if isinstance(item, Arrival):
                if not item.ok:
                    key = f"{item.handle or item.source}:{item.source}"
                    progress.failed[key] = item.error
                    if item.handle:
                        room.errors[item.handle] = f"{item.source}: {item.error}"
                    log(f"  {item.handle or item.source} ({item.source}) failed: {item.error}")
                    give_back(item.source, [item.handle] if item.handle else
                              sorted(claimed.get(item.source, ())))
                else:
                    arrived.setdefault(item.source, set()).add(item.handle)
                    room.edits[item.handle] = item.edit
                    room.replaced[item.handle] = {"source": item.source, **item.summary}
                    room.errors.pop(item.handle, None)
                    progress.installed[item.handle] = item.source
                    dirty = True
                    log(f"  {item.handle} {item.source} in {item.seconds:.1f} s")
                if dirty and time.monotonic() - last >= min_interval:
                    publish("objects arrived")
                continue
            if not item.ok:
                progress.failed[f"{item.handle}:{item.pass_name}"] = item.error
                room.errors[item.handle] = f"{item.pass_name}: {item.error}"
                log(f"  {item.handle} {item.pass_name} failed: {item.error}")
                continue
            path = _glb_cache(room, item.handle, item.pass_name, images[item.handle])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item.glb)
            try:
                install(item.handle, item.glb, item.pass_name)
            except Exception as exc:  # noqa: BLE001 — a mesh that cannot be fitted keeps the template
                progress.failed[f"{item.handle}:{item.pass_name}"] = f"fit: {exc}"
                room.errors[item.handle] = f"{item.pass_name} fit: {exc}"
                log(f"  {item.handle} {item.pass_name}: could not fit ({exc})")
                continue
            if room.installed[item.handle].pass_name == item.pass_name:
                progress.installed[item.handle] = item.pass_name
                dirty = True
                log(f"  {item.handle} {item.pass_name} in {item.seconds:.1f} s")
            if dirty and time.monotonic() - last >= min_interval:
                publish("objects arrived")
    if dirty:
        publish("all done")
    else:
        room.write_report()          # failures and skips are worth recording without a version
    progress.seconds = time.monotonic() - started
    return progress


def counts(room: RealRoom) -> dict[str, int]:
    """How many of the room's objects are at each pass (and still a template)."""
    out = {"template": 0, **{p: 0 for p in PASSES}}
    for handle in room.generated_handles():
        item = room.installed.get(handle)
        out[item.pass_name if item else "template"] += 1
    return out

