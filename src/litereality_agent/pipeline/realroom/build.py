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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def compose(base: ET.Element, installed: dict[str, Installed]) -> tuple[ET.Element, list[str]]:
    """The base quick room with each installed object's template parts made colliders and its
    generated mesh added as the body's look. Returns (scene root, handles not found)."""
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
    ET.indent(root, space="  ")
    return root, missing


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
        return [h for h, box in self.boxes.items() if box.generated()]

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
        root, missing = compose(self.base, self.installed)
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
                  reuse: bool = True) -> Progress:
    """Generate every object of ``room`` and write a new version as they arrive.

    Crops first (cached GLBs for the same photo are installed without asking again), then the
    generator streams results; a version is written whenever something new is installed and the
    last one is ``min_interval`` seconds old, and once more at the end. ``on_version(scene,
    info)`` is called after each (`--publish` uploads it there). Never raises for one object."""
    started = time.monotonic()
    passes = check_passes(passes)
    min_interval = MIN_INTERVAL if min_interval is None else float(min_interval)
    progress = Progress(room.scene_path)
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

    todo: dict[str, list[str]] = {}
    for handle, png in images.items():
        # The best pass already generated for this very photo is installed as it is; only the
        # passes after it are asked for (a cached final makes its preview moot).
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
        need = [p for p in passes if best is None or PASSES[p].rank > PASSES[best].rank]
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

    if todo:
        log(f"  generating {len(todo)} object(s) with {generator.name}: "
            + ", ".join(f"{h} ({'+'.join(p)})" for h, p in sorted(todo.items())))
        # Group by pass list so the generator sees each object's passes in order.
        results: queue.Queue = queue.Queue()
        groups: dict[tuple[str, ...], dict[str, bytes]] = {}
        for handle, need in todo.items():
            groups.setdefault(tuple(need), {})[handle] = images[handle]

        def one_group(need: tuple[str, ...], group: dict[str, bytes]) -> None:
            try:
                for result in generator.stream(group, list(need)):
                    results.put(result)
            except BaseException as exc:  # noqa: BLE001 — handed to the loop below
                results.put(exc)

        def produce() -> None:
            threads = [threading.Thread(target=one_group, args=item, daemon=True)
                       for item in groups.items()]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            results.put(None)

        threading.Thread(target=produce, daemon=True, name="realroom-generate").start()
        while True:
            wait = None
            if dirty and math.isfinite(min_interval):
                wait = max(0.0, last + min_interval - time.monotonic())
            try:
                item = results.get(timeout=wait)
            except queue.Empty:
                publish("objects arrived")
                continue
            if item is None:
                break
            if isinstance(item, BaseException):
                log(f"  the generator stopped: {type(item).__name__}: {item}")
                break
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

