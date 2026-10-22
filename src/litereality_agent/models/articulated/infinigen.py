"""infinigen.py — openable furniture from Infinigen-Sim, sized to a scanned box, as our bodies.

The quick room's cabinets are boxes and its door is a slab on a hinge; TRELLIS gives an object
its looks but not its parts. Infinigen-Sim (princeton-vl/infinigen, BSD-3) builds articulated
assets procedurally — a cabinet with hinged doors and shelves, a drawer chest, a dishwasher with
a drop-down door and racks, a fridge, an oven, a door in its frame, a sliding window — in 10-90 s
on a CPU, sized to whatever box it is given. This module turns one scanned object into one of
those and the result into MuJoCo bodies of OUR scene:

    spec_for_object / door_spec / window_spec    category + RoomPlan box -> `Spec` (or None)
    Runner (LocalRunner | ModalRunner | FakeRunner) + AssetCache -> Infinigen's MJCF export
    load_asset -> clean -> `Asset`                its links, joints, visual and collider meshes
    build(asset, handle, Placement, Overrides)    -> `Built`: MJCF elements + files for the scene

What is taken from Infinigen and what is not (feasibility study, 2026-10-21, 30 assets):

* **Taken:** the shape, the parts and how they move — each link becomes a child body with its
  hinge or slide where Infinigen put it, over Infinigen's range; its CoACD hulls become the
  colliders, its textured meshes the look.
* **Mass, overridden.** Exported masses are 3-30x too heavy (metal densities on solid hulls that
  fill a cabinet's hollow inside: a door leaf at 50-431 kg, a dishwasher at 614-757). The total
  is the scan's box volume x ``OCCUPANCY_DENSITY`` — the number the quick room and the full
  pipeline use — shared over the links by MATERIAL volume (`properties._material_volume`: boards
  and slabs, not the air between them), so a cabinet door weighs a few kilos.
* **Joints, overridden.** Hinges come with damping ~1000 and a hard-coded frictionloss of 1000
  N·m, drawers with 1000 N: a robot-learning setting that no hand opens. Every joint gets
  ``JOINT_DEFAULTS``; a flap on a horizontal hinge (a dishwasher or oven door) gets just enough
  friction to stay where it is left, as its counterbalance springs do — with the defaults alone
  it falls open under its own weight.
* **Fixed up.** Handles, knobs, buttons and lock levers are welded to what they are on (a 2 cm
  button on its own joint is a body MuJoCo can throw); parts a sizing bug left outside the
  object (the dishwasher's side buttons) are dropped; a rack poking through the top is moved
  down; a slide longer than its tub is clamped (the dishwasher's racks slid out a full metre);
  a door frame's bottom bar loses its collider (`drop_sill`: no curb across the doorway).
* **Contacts.** A part never collides with what it hangs on, nor with a part it touches at rest
  (a cabinet's two doors); every other pair of parts does (`rest_contacts`), so a dishwasher's
  rack is held in by its closed door.
* **Fitted.** Infinigen's front is +X with +Y across it and Z up; ours is the box frame of
  `quickroom` — +X across the front, front towards -Y. The asset's main bounding box (handles
  excluded) is scaled into the scanned box per axis; the 1-8 cm residuals of its own sizing go
  into that scale. An opening (door, window) keeps its own thickness and stands in the wall.
* **Tinted.** Infinigen picks random procedural materials. The baked diffuse texture of every
  body part is recoloured to the photo colour quickroom measured for that object, keeping the
  texture's own light and dark (wood grain, panel edges); handles and glass keep theirs.

Where Infinigen runs: its own Python 3.11 env with exactly ``bpy==4.2.0`` — never inside ours.
``LR_INFINIGEN_PYTHON`` pointing at that env's python runs `infinigen_worker.py` there as a
subprocess (`LocalRunner`); otherwise the ``litereality-infinigen`` CPU function on Modal does
(`ModalRunner`, deploy/modal/infinigen). Either way a finished export is kept in `AssetCache` by
(factory, size to the centimetre, seed): the same cabinet in the same room is never built twice.
The offline stand-ins (`FakeRunner`) are cached apart and can never be read back as real assets.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from litereality_agent.models.articulated import infinigen_worker as worker
from litereality_agent.models.object_generation.sim import properties

WORKER = Path(worker.__file__).resolve()
WORKER_VERSION = worker.WORKER_VERSION
FACTORIES = worker.FACTORIES
INFINIGEN_COMMIT = "224598d4a2ff95b4fc928e72c222598b7122c698"   # BSD-3, the study's checkout
MODAL_APP = "litereality-infinigen"
MODAL_FUNCTION = "generate"
DEFAULT_SEED = 1001
IMAGE_RES = 512            # px per baked texture; the room call scales textures to a budget anyway
PYTHON_ENV = "LR_INFINIGEN_PYTHON"
CACHE_ENV = "LR_INFINIGEN_CACHE"
WORKERS_ENV = "LR_INFINIGEN_WORKERS"
JOB_TIMEOUT = 20 * 60      # s one asset may take (the slowest healthy one was 90 s)
# Rough seconds per asset on one core (the study's warm runs), to give the slowest work to the
# workers first.
EXPECTED_SECONDS = {"door": 15, "window": 80, "cabinet": 30, "drawer": 40, "dishwasher": 35,
                    "refrigerator": 40, "oven": 20}

# ── what becomes what ────────────────────────────────────────────────────────────────────────────
STORAGE_WORDS = ("storage", "cabinet", "sink", "cupboard", "dresser", "drawer", "chest")
DRAWER_WORDS = ("drawer", "dresser", "chest")
MAX_CABINET_HEIGHT = 1.2   # m: taller floor-standing storage is a wardrobe or shelves, not this
MIN_CABINET_HEIGHT = 0.25
HUNG_GAP = 0.30            # m clear of the floor: wall-hung (quickroom's rule)
STANDING_GAP = 0.08        # m: at most this far off the floor is standing on it
ONE_DOOR_WIDTH = 0.55      # m: narrower cabinets get one door (Infinigen's own threshold)
SHELF_PITCH = 0.35         # m between a cabinet's shelves
TOE_KICK = 0.08            # m: a floor-standing cabinet's plinth
# A dishwasher's door is hinged at its bottom edge: standing on the floor it cannot open past
# the floor, so it stands on a plinth as a fitted one does (8 cm let the study's door open).
PLINTH = {"dishwasher": 0.08, "oven": 0.08}
PLINTH_SETBACK = 0.05      # m the plinth stands back from the front
PLINTH_RGB = (0.12, 0.12, 0.13)

# ── cleaning ─────────────────────────────────────────────────────────────────────────────────────
MINOR_WORDS = ("handle", "knob", "button", "lock")
JUNK_WORDS = ("wire",)
OUTLIER = 0.03             # m a part may stand outside the main box at the sides, top, bottom
OUTLIER_FRONT = 0.12       # ... and in front of it or behind it (handles do)
SLIDE_FULL = 0.95          # a slide longer than this share of its parent's depth is a bug ...
SLIDE_CLAMP = 0.75         # ... and is clamped to this share of it
POKE = 0.005               # m: an inner part reaching past its parent's top or bottom is moved in
AXIS_SNAP = math.radians(2.0)

# ── physics ──────────────────────────────────────────────────────────────────────────────────────
ARMATURE = 0.002           # as the quick room's door hinge: keeps a light link's step stable
FLAP_HOLD = 1.25           # a flap's friction, in multiples of the most its weight can twist it
# MuJoCo's dry friction is a soft constraint: under a steady load below it, a joint still creeps
# (a dishwasher door at 40 deg sank 10 deg a minute at the defaults). Stiffer for a flap: 0.2 deg
# in 10 s, and a 40 N pull still opens it all the way.
FLAP_SOLREF = "0.004 1"
FLAP_SOLIMP = "0.99 0.999 0.001"
GRAVITY = 9.81
# m: two parts of one asset whose colliders are this close at rest do not collide; all others do
REST_GAP = 0.002
VISUAL_ATTRS = {"group": "2", "contype": "0", "conaffinity": "0", "density": "0"}
COLLIDER_GROUP = "3"
GLASS_RGBA = (0.78, 0.87, 0.95, 0.22)    # quickroom's window glass
GLASS_TRANSMISSION = 80    # mean 8-bit transmission above which a baked part is glass
UNUSED_TEXEL = 2.0 / 255.0  # luminance at or below which a texel is unused atlas space
DETAIL_LUMINANCE = 0.25    # a bake at least this bright keeps all of its shading when tinted
ARTICULATED_DIR = "articulated"           # beside the scene's meshes / textures
FAKE_DIR = "fake"          # the cache's folder for stand-ins (`FakeRunner`), apart from real ones
# Infinigen's front is +X with +Y across it; the box frame's front is -Y with +X across it.
R_INFINIGEN = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


# ── specs ────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Spec:
    """One asset to build: an Infinigen factory, a size (W across the front, D front to back, H
    up; metres, to the centimetre), a seed and the factory options. Equal specs are equal assets
    — the cache key is made of nothing else."""

    factory: str
    size: tuple[float, float, float]
    seed: int = DEFAULT_SEED
    options: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def make(cls, factory: str, size: Sequence[float], *, seed: int = DEFAULT_SEED,
             **options: Any) -> Spec:
        if factory not in FACTORIES:
            raise ValueError(f"unknown factory {factory!r}: expected one of {', '.join(FACTORIES)}")
        rounded = tuple(round(max(0.01, float(v)), 2) for v in size)
        if len(rounded) != 3:
            raise ValueError(f"size needs W, D, H: {size!r}")
        opts = tuple(sorted((k, v) for k, v in options.items() if v is not None))
        return cls(factory, rounded, int(seed), opts)

    @property
    def opts(self) -> dict[str, Any]:
        return dict(self.options)

    @property
    def key(self) -> str:
        w, d, h = (int(round(v * 100)) for v in self.size)
        key = f"{self.factory}-{w}x{d}x{h}cm-s{self.seed}"
        if self.options:
            text = json.dumps(self.options, sort_keys=True, default=str)
            key += "-" + hashlib.sha256(text.encode()).hexdigest()[:6]
        return key + f"-w{WORKER_VERSION}"

    def job(self) -> dict:
        return {"key": self.key, "factory": self.factory, "size": list(self.size),
                "seed": self.seed, "options": self.opts}


def _tokens(category: str) -> str:
    return (category or "").lower()


def _luminance(rgb: Sequence[float]) -> float:
    r, g, b = (float(v) for v in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def frame_material(rgb: Sequence[float] | None) -> str | None:
    """Infinigen's cabinet frame material nearest the photo: tinting then only adjusts it."""
    if rgb is None:
        return None
    y = _luminance(rgb)
    return "white" if y > 0.55 else "black_wood" if y < 0.2 else "wood"


def spec_for_object(category: str, size: Sequence[float], *, bottom_gap: float,
                    hung: bool | None = None, drawers: bool = False, worktop: float = 0.0,
                    rgb: Sequence[float] | None = None, seed: int = DEFAULT_SEED) -> Spec | None:
    """The asset for one scanned object, or None when Infinigen has nothing that fits.

    ``size`` is its box (W, D, H); ``bottom_gap`` how far the box's bottom is above the floor;
    ``worktop`` the height of a counter top kept on it (the asset goes under it); ``drawers``
    makes a floor-standing storage unit a drawer chest instead of a cabinet. What maps:

    * dishwasher, refrigerator, oven (a stove's oven) -> those factories; a floor-standing
      dishwasher or oven stands on a `PLINTH`, so its drop-down door clears the floor;
    * storage (cabinets, sink units, counters) hung on a wall -> a wall cabinet (no toe kick);
      standing and at most `MAX_CABINET_HEIGHT` tall -> a cabinet with one door if it is narrower
      than `ONE_DOOR_WIDTH`, else two, or a drawer chest when ``drawers`` says so;
    * anything else (taller storage, sofas, beds, tables, a sink with no storage under it) ->
      None: TRELLIS or the template keeps it.
    """
    c = _tokens(category)
    w, d, h = (float(v) for v in size)
    h -= max(0.0, worktop)
    standing = bottom_gap < STANDING_GAP
    hung = (bottom_gap > HUNG_GAP) if hung is None else hung
    for appliance in ("dishwasher", "refrigerator", "oven"):
        words = {"refrigerator": ("refrigerator", "fridge"), "oven": ("oven", "stove")}.get(
            appliance, (appliance,))
        if any(t in c for t in words):
            plinth = PLINTH.get(appliance, 0.0) if standing else 0.0
            if h - plinth < 0.3 or w < 0.2 or d < 0.2:
                return None
            return Spec.make(appliance, (w, d, h - plinth), seed=seed)
    if not any(t in c for t in STORAGE_WORDS):
        return None
    if c == "sink" or h < MIN_CABINET_HEIGHT or w < 0.2 or d < 0.15:
        return None               # a basin on its own, or a sliver: nothing to open
    if hung:
        shelves = max(1, round(h / SHELF_PITCH))
        return Spec.make("cabinet", (w, d, h), seed=seed, bottom=0.0, shelves=shelves,
                         frame_material=frame_material(rgb))
    if not standing or h > MAX_CABINET_HEIGHT:
        return None
    if drawers or any(t in c for t in DRAWER_WORDS):
        rows = int(np.clip(round((h - 0.1) / 0.22), 1, 4))
        cols = 1 if w < 0.7 else 2
        return Spec.make("drawer", (w, d, h), seed=seed, rows=rows, cols=cols, bottom=0.1)
    shelves = max(1, round((h - TOE_KICK) / SHELF_PITCH))
    return Spec.make("cabinet", (w, d, h), seed=seed, bottom=TOE_KICK, shelves=shelves,
                     frame_material=frame_material(rgb))


def door_spec(width: float, height: float, *, hinge: str = "left", wall_thickness: float = 0.1,
              seed: int = DEFAULT_SEED) -> Spec:
    """A door leaf in its frame filling a ``width`` x ``height`` opening. ``hinge`` is the side
    it hangs on seen from the side it opens to ("left" / "right")."""
    if hinge not in ("left", "right"):
        raise ValueError(f"hinge {hinge!r}: expected left or right")
    return Spec.make("door", (width, wall_thickness, height), seed=seed, orientation=hinge,
                     wall_thickness=round(float(wall_thickness), 3))


def window_spec(width: float, height: float, *, seed: int = DEFAULT_SEED) -> Spec:
    """Two sliding sashes in a frame filling a ``width`` x ``height`` opening."""
    return Spec.make("window", (width, 0.1, height), seed=seed, panels=2)


# ── running Infinigen ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RunResult:
    """One spec's export: its folder (``result.json`` + Infinigen's MJCF) or why there is none."""

    spec: Spec
    folder: Path | None
    seconds: float = 0.0
    error: str = ""
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.folder is not None and not self.error


class Runner(Protocol):
    name: str
    fake: bool                 # writes stand-ins, cached apart (`AssetCache`)

    def run(self, specs: Sequence[Spec], out_dir: Path) -> Iterator[RunResult]:
        """Build every spec into ``out_dir/<key>/``, yielding each as it finishes. Never raises
        for one asset's failure."""
        ...


def _record(folder: Path) -> dict:
    try:
        rec = json.loads((Path(folder) / "result.json").read_text())
    except (OSError, ValueError):
        return {}
    return rec if isinstance(rec, dict) else {}


def _mjcf_of(folder: Path) -> Path | None:
    rec = _record(folder)
    if not rec.get("ok") or not rec.get("mjcf"):
        return None
    path = folder / rec["mjcf"]
    return path if path.is_file() else None


def is_fake(runner: Any) -> bool:
    """Whether ``runner`` writes stand-ins (`FakeRunner`) rather than Infinigen's own exports."""
    return bool(getattr(runner, "fake", False))


class AssetCache:
    """Finished exports, one folder per `Spec.key`, under ``$LR_INFINIGEN_CACHE`` (default
    ``~/.cache/litereality/infinigen``). A folder only ever appears whole (moved into place).

    Stand-ins (`FakeRunner`) live apart, under ``fake/``, and every hit is checked against the
    ``fake`` flag its ``result.json`` carries: the key is only what the asset is MADE of
    (factory, size, seed), so without that a ``--fake`` run — or a test with
    ``$LR_ARTICULATED`` exported — would leave boxes that a later real run installs as
    "cached" Infinigen furniture (a 90 x 205 cm door is any room's door)."""

    def __init__(self, root: Path | str | None = None) -> None:
        root = root or os.environ.get(CACHE_ENV) or Path.home() / ".cache" / "litereality" / \
            "infinigen"
        self.root = Path(root).expanduser()

    def path(self, spec: Spec, *, fake: bool = False) -> Path:
        return (self.root / FAKE_DIR if fake else self.root) / spec.key

    def get(self, spec: Spec, *, fake: bool = False) -> Path | None:
        """The finished export for ``spec`` from the runner kind ``fake`` says, or None (a folder
        of the other kind under this key is never returned: it is built again over)."""
        folder = self.path(spec, fake=fake)
        if _mjcf_of(folder) is None or bool(_record(folder).get("fake")) != bool(fake):
            return None
        return folder

    def put(self, spec: Spec, folder: Path, *, fake: bool = False) -> Path:
        """Move a finished export folder into the cache (replacing an older one), without the
        baked maps nothing reads (`KEEP_TEXTURES`: a cabinet is 3 MB instead of 7)."""
        for png in Path(folder).rglob("*.png"):
            if not png.name.endswith(KEEP_TEXTURES):
                png.unlink(missing_ok=True)
        dest = self.path(spec, fake=fake)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".tmp{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.move(str(folder), str(tmp))
        old = dest.with_name(dest.name + f".old{os.getpid()}")
        if dest.exists():
            os.replace(dest, old)
        os.replace(tmp, dest)
        shutil.rmtree(old, ignore_errors=True)
        return dest


def generate(specs: Iterable[Spec], runner: Runner, cache: AssetCache, *,
             log: Callable[[str], Any] | None = None) -> Iterator[RunResult]:
    """Every spec's export, cached ones first, the rest built by ``runner`` and cached as each
    finishes. Equal specs are built once."""
    unique: dict[str, Spec] = {}
    for spec in specs:
        unique.setdefault(spec.key, spec)
    fake = is_fake(runner)
    todo = []
    for spec in unique.values():
        hit = cache.get(spec, fake=fake)
        if hit is not None:
            yield RunResult(spec, hit, cached=True)
        else:
            todo.append(spec)
    if not todo:
        return
    if log:
        log(f"  infinigen: building {len(todo)} asset(s) with {runner.name}: "
            + ", ".join(s.key for s in todo))
    scratch = Path(tempfile.mkdtemp(prefix="lr-infinigen-", dir=_scratch_root(cache)))
    try:
        for result in runner.run(todo, scratch):
            if result.ok:
                try:
                    result = replace(result, folder=cache.put(result.spec, result.folder,
                                                              fake=fake))
                except OSError as exc:
                    result = replace(result, folder=None, error=f"cache: {exc}")
            yield result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _scratch_root(cache: AssetCache) -> str:
    cache.root.mkdir(parents=True, exist_ok=True)
    return str(cache.root)


def _balanced(specs: Sequence[Spec], workers: int) -> list[list[Spec]]:
    """Longest first onto the least-loaded worker: the window (80 s) is never last in a queue."""
    loads = [0.0] * workers
    out: list[list[Spec]] = [[] for _ in range(workers)]
    for spec in sorted(specs, key=lambda s: -EXPECTED_SECONDS.get(s.factory, 30)):
        k = int(np.argmin(loads))
        out[k].append(spec)
        loads[k] += EXPECTED_SECONDS.get(spec.factory, 30)
    return [chunk for chunk in out if chunk]


def default_workers() -> int:
    try:
        return max(1, int(os.environ.get(WORKERS_ENV, "")))
    except ValueError:
        return max(1, min(4, (os.cpu_count() or 2) // 2))


class LocalRunner:
    """`infinigen_worker.py` in a local Infinigen env (``python`` is that env's interpreter),
    ``workers`` processes at once, each building its share one after another (the second asset
    in a process skips ~3 s of imports). Each process gets its own Blender user folder."""

    name = "infinigen-local"
    fake = False

    def __init__(self, python: Path | str, *, workers: int | None = None,
                 image_res: int = IMAGE_RES, timeout: float = JOB_TIMEOUT) -> None:
        self.python = Path(python).expanduser()
        self.workers = workers or default_workers()
        self.image_res = int(image_res)
        self.timeout = float(timeout)

    def check(self) -> None:
        if not self.python.is_file():
            raise RuntimeError(f"${PYTHON_ENV}={self.python} is not a python interpreter")

    def run(self, specs: Sequence[Spec], out_dir: Path) -> Iterator[RunResult]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        by_key = {s.key: s for s in specs}
        results: queue.Queue = queue.Queue()
        chunks = _balanced(list(specs), min(self.workers, len(specs)) or 1)

        def one(index: int, chunk: list[Spec]) -> None:
            jobs = out_dir / f"jobs{index}.json"
            jobs.write_text(json.dumps({"image_res": self.image_res,
                                        "jobs": [s.job() for s in chunk]}))
            log = (out_dir / f"worker{index}.log").open("w")
            seen: set[str] = set()
            started = time.monotonic()
            env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
            try:
                proc = subprocess.Popen(
                    [str(self.python), "-I", str(WORKER), str(jobs), str(out_dir)],
                    stdout=subprocess.PIPE, stderr=log, text=True, env=env,
                    cwd=str(out_dir))
            except OSError as exc:
                for spec in chunk:
                    results.put(RunResult(spec, None, 0.0, f"could not start: {exc}"))
                results.put(None)
                log.close()
                return
            timer = threading.Timer(self.timeout * len(chunk), proc.kill)
            timer.start()
            try:
                for line in proc.stdout:
                    if not line.startswith("RESULT "):
                        continue
                    try:
                        rec = json.loads(line[len("RESULT "):])
                    except ValueError:
                        continue
                    spec = by_key.get(rec.get("key"))
                    if spec is None:
                        continue
                    seen.add(spec.key)
                    folder = out_dir / spec.key
                    if rec.get("ok") and _mjcf_of(folder) is not None:
                        results.put(RunResult(spec, folder, float(rec.get("seconds", 0.0))))
                    else:
                        results.put(RunResult(spec, None, float(rec.get("seconds", 0.0)),
                                              str(rec.get("error") or "no MJCF written")))
                proc.wait()
            finally:
                timer.cancel()
                log.close()
            for spec in chunk:
                if spec.key not in seen:
                    tail = _tail(out_dir / f"worker{index}.log")
                    results.put(RunResult(spec, None, time.monotonic() - started,
                                          f"the worker exited ({proc.returncode}) before "
                                          f"finishing it: {tail}"))
            results.put(None)

        threads = [threading.Thread(target=one, args=(i, c), daemon=True,
                                    name=f"infinigen-{i}") for i, c in enumerate(chunks)]
        for t in threads:
            t.start()
        running = len(threads)
        while running:
            item = results.get()
            if item is None:
                running -= 1
                continue
            yield item


def _tail(path: Path, lines: int = 3) -> str:
    try:
        text = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return ""
    return " | ".join(text[-lines:])[-400:]


class ModalRunner:
    """The ``generate`` function of the ``litereality-infinigen`` app on Modal: one CPU container
    per asset (up to the app's ``max_containers``), each result unpacked as it lands."""

    name = "infinigen-modal"
    fake = False

    def __init__(self, *, app_name: str = MODAL_APP, function_name: str = MODAL_FUNCTION,
                 environment_name: str | None = None, profile: str | None = None,
                 credentials: tuple[str, str] | None = None, function: Any = None,
                 image_res: int = IMAGE_RES) -> None:
        self.app_name, self.function_name = app_name, function_name
        self.environment_name = environment_name
        self.profile, self.credentials = profile, credentials
        self._function = function
        self.image_res = int(image_res)

    @property
    def function(self) -> Any:
        if self._function is None:
            from litereality_agent.runtimes.modal import ModalClient

            self._function = ModalClient(self.app_name, self.function_name,
                                         environment_name=self.environment_name or "main",
                                         profile=self.profile,
                                         credentials=self.credentials).function
        return self._function

    def check(self) -> None:
        """Fail now, once, if the app is not deployed (rather than once per asset later)."""
        hydrate = getattr(self.function, "hydrate", None)
        if callable(hydrate):
            hydrate()

    def request(self, spec: Spec) -> dict:
        return {"job": spec.job(), "image_res": self.image_res}

    def run(self, specs: Sequence[Spec], out_dir: Path) -> Iterator[RunResult]:
        """``Function.map`` over the specs, unordered: each asset as its container finishes.
        The function answers every job with its own key (an error included); what does not come
        back with a key — a container that died — is reported once the map is done."""
        out_dir = Path(out_dir)
        by_key = {s.key: s for s in specs}
        started = time.monotonic()
        seen: set[str] = set()
        lost: list[str] = []
        try:
            outputs = self.function.map([self.request(s) for s in specs], order_outputs=False,
                                        return_exceptions=True)
            for output in outputs:
                key = output.get("key") if isinstance(output, dict) else None
                if key not in by_key or key in seen:
                    lost.append(f"{type(output).__name__}: {output}"[:300])
                    continue
                seen.add(key)
                yield self._unpack(output, by_key[key], out_dir, time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001 — the whole batch failed: say so per asset
            lost.append(f"{type(exc).__name__}: {exc}"[:300])
        for spec in specs:
            if spec.key not in seen:
                yield RunResult(spec, None, time.monotonic() - started,
                                "; ".join(lost) or "no answer from Modal")

    def _unpack(self, output: dict, spec: Spec, out_dir: Path, elapsed: float) -> RunResult:
        if not output.get("ok") or not output.get("zip_b64"):
            return RunResult(spec, None, float(output.get("seconds", elapsed)),
                             str(output.get("error") or "no asset returned"))
        folder = out_dir / spec.key
        try:
            unpack_zip(base64.b64decode(output["zip_b64"]), folder)
        except (ValueError, zipfile.BadZipFile, OSError) as exc:
            return RunResult(spec, None, elapsed, f"bad archive: {exc}")
        if _mjcf_of(folder) is None:
            return RunResult(spec, None, elapsed, "the archive holds no MJCF")
        return RunResult(spec, folder, float(output.get("seconds", elapsed)))


# Infinigen bakes five maps per part; the MJCF uses the diffuse one and the conversion reads the
# transmission one (glass). The rest stay behind: a cabinet is 3-6 MB instead of 15.
KEEP_TEXTURES = ("_DIFFUSE.png", "_TRANSMISSION.png")


def zip_folder(folder: Path) -> bytes:
    """An export folder as zip bytes, without the maps nothing here reads."""
    folder = Path(folder)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".png" and not path.name.endswith(KEEP_TEXTURES):
                continue
            zf.write(path, path.relative_to(folder).as_posix())
    return buffer.getvalue()


def unpack_zip(data: bytes, folder: Path) -> Path:
    """`zip_folder`'s bytes back into ``folder`` (refusing paths that would leave it)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    root = folder.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            target = (folder / info.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive path {info.filename!r} leaves its folder")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
    return folder


class FakeRunner:
    """Infinigen without Infinigen: `fake_export` writes a small MJCF shaped like the real
    exporter's (an ``object`` body, ``link_N`` children, ``<label>_N`` visual meshes with baked
    diffuse PNGs, ``_colN`` hull colliders, robot-grade densities and joint dynamics) for each
    spec, so the conversion and the room are exercised offline exactly as real assets go through
    them. ``fail`` names factories that error; ``delay`` is seconds per asset."""

    name = "infinigen-fake"
    fake = True

    def __init__(self, *, fail: Sequence[str] = (), delay: float = 0.0) -> None:
        self.fail = set(fail)
        self.delay = float(delay)
        self.calls: list[str] = []

    def check(self) -> None:
        return None

    def run(self, specs: Sequence[Spec], out_dir: Path) -> Iterator[RunResult]:
        for spec in specs:
            started = time.perf_counter()
            if self.delay:
                time.sleep(self.delay)
            self.calls.append(spec.key)
            if spec.factory in self.fail:
                yield RunResult(spec, None, self.delay, "fake failure")
                continue
            folder = fake_export(spec, Path(out_dir) / spec.key)
            yield RunResult(spec, folder, time.perf_counter() - started)


def runner_from_env(*, fake: bool = False, modal_kwargs: Mapping[str, Any] | None = None,
                    workers: int | None = None) -> Runner:
    """``$LR_INFINIGEN_PYTHON`` -> `LocalRunner`; else ``fake`` -> `FakeRunner`; else Modal."""
    python = os.environ.get(PYTHON_ENV, "").strip()
    if python:
        return LocalRunner(python, workers=workers)
    if fake:
        return FakeRunner()
    kwargs = dict(modal_kwargs or {})
    return ModalRunner(environment_name=kwargs.get("environment"),
                       profile=kwargs.get("profile"), credentials=kwargs.get("credentials"))


# ── the stand-in export ──────────────────────────────────────────────────────────────────────────


def _box_obj(lo: Sequence[float], hi: Sequence[float]) -> str:
    """An axis-aligned box as OBJ text with UVs (each face the whole texture), outward faces."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in corners]
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    for a, b, c, d in quads:
        lines.append(f"f {a + 1}/1 {b + 1}/2 {c + 1}/3")
        lines.append(f"f {a + 1}/1 {c + 1}/3 {d + 1}/4")
    return "\n".join(lines) + "\n"


def _fake_parts(spec: Spec) -> list[dict]:
    """The links of a stand-in asset in Infinigen's frame (front +X, centred in Y, floor at 0):
    each {label, box (lo, hi), joint: (name, type, pos, axis, range) | None, parent, glass}."""
    W, D, H = spec.size
    opts = spec.opts
    hx, hy = D / 2, W / 2
    t = 0.02                                       # door / front thickness
    parts: list[dict] = []
    if spec.factory == "door":
        fw = 0.05
        depth = 0.04
        # hulls as CoACD cuts the real square frame: two jambs, the top and the bottom bar
        parts.append({"label": "door_frame", "box": ((-depth / 2, -hy, 0), (depth / 2, hy, H)),
                      "hulls": [((-depth / 2, -hy, 0), (depth / 2, -hy + fw, H)),
                                ((-depth / 2, hy - fw, 0), (depth / 2, hy, H)),
                                ((-depth / 2, -hy + fw, H - fw), (depth / 2, hy - fw, H)),
                                ((-depth / 2, -hy + fw, 0), (depth / 2, hy - fw, fw))]})
        left = opts.get("orientation", "left") == "left"
        y_h = -hy + fw if left else hy - fw
        parts.append({"label": "door", "box": ((-0.018, -hy + fw, fw), (0.018, hy - fw, H - fw)),
                      "joint": ("door_hinge_left", "hinge", (0.018, y_h, H / 2),
                                (0, 0, -1 if left else 1), (0.0, 2.0)), "parent": 1})
        y_k = hy - fw - 0.08 if left else -hy + fw + 0.08
        parts.append({"label": "handle", "box": ((-0.09, y_k - 0.06, 0.95), (0.09, y_k + 0.01, 1.0)),
                      "joint": ("door_handle_right", "hinge", (0, y_k, 0.975), (1, 0, 0),
                                (0.0, 1.0)), "parent": 2})
        return parts
    if spec.factory == "window":
        fd = 0.06
        parts.append({"label": "Frame", "box": ((-fd / 2, -hy, 0), (fd / 2, hy, H))})
        for k, (sign, name) in enumerate(((-1, "SlidingPanelL"), (1, "SlidingPanelR"))):
            y0, y1 = (-hy + 0.04, 0.0) if sign < 0 else (0.0, hy - 0.04)
            x = -0.015 if sign < 0 else 0.015
            parts.append({"label": "Pane", "glass": True,
                          "box": ((x - 0.01, y0, 0.05), (x + 0.01, y1, H - 0.05)),
                          "joint": (name, "slide", (x, (y0 + y1) / 2, H / 2), (0, -sign, 0),
                                    (0.0, 0.6 * hy)), "parent": 1})
        parts.append({"label": "LockLever", "box": ((fd / 2, 0.2, 0.01), (fd / 2 + 0.01, 0.3, 0.06)),
                      "joint": ("Lock", "hinge", (fd / 2, 0.25, 0.03), (1, 0, 0), (0.0, 1.0)),
                      "parent": 1})
        return parts
    body_hx = hx - t
    parts.append({"label": {"dishwasher": "dishwasher_body", "refrigerator": "body",
                            "oven": "oven base", "drawer": "drawer_base"}.get(spec.factory,
                                                                               "cabinet_base"),
                  "box": ((-hx, -hy, 0), (body_hx, hy, H))})
    if spec.factory == "cabinet":
        doors = 1 if W < ONE_DOOR_WIDTH else 2
        z0 = float(opts.get("bottom", 0.0))
        spans = [(-hy, hy)] if doors == 1 else [(-hy, 0.0), (0.0, hy)]
        side = 0.018                       # Infinigen's side board: the hinge is set in by it
        for k, (y0, y1) in enumerate(spans):
            right = k == 0
            # On the door's back face, a side board in from its outer edge — so, as in the real
            # asset, the door's outer front corner sweeps ~1 cm past the cabinet's side.
            y_h = y0 + side if right else y1 - side
            parts.append({"label": "right_door" if right else "left_door",
                          "box": ((body_hx, y0, z0), (hx, y1, H)),
                          "joint": ("cabinet_hinge" if right else "cabinet_hinge_multiple",
                                    "hinge", (body_hx, y_h, (z0 + H) / 2),
                                    (0, 0, -1 if right else 1), (0.0, 2.3)), "parent": 1})
            yk = y1 - 0.05 if right else y0 + 0.05
            parts.append({"label": "handle", "box": ((hx, yk - 0.01, H - 0.2), (hx + 0.03, yk + 0.01,
                                                                                 H - 0.1)),
                          "parent": len(parts)})
        return parts
    if spec.factory == "drawer":
        rows, cols = int(opts.get("rows", 3)), int(opts.get("cols", 2))
        z0 = float(opts.get("bottom", 0.1))
        dz, dy = (H - z0) / rows, W / cols
        for r in range(rows):
            for c in range(cols):
                y0 = -hy + c * dy
                zlo = z0 + r * dz
                parts.append({"label": "drawer_door",
                              "box": ((-hx + 0.05, y0 + 0.01, zlo + 0.01),
                                      (hx, y0 + dy - 0.01, zlo + dz - 0.01)),
                              "joint": ("drawer_slider", "slide", (hx, y0 + dy / 2, zlo + dz / 2),
                                        (1, 0, 0), (0.0, 0.8 * D)), "parent": 1})
        return parts
    # appliances: a front door, an inner rack (sliding a metre, as the real dishwasher's did),
    # and for the dishwasher, buttons a sizing bug put outside the body
    if spec.factory in ("dishwasher", "oven"):
        parts.append({"label": "door", "box": ((body_hx, -hy, 0.0), (hx, hy, H)),
                      "joint": ("door_joint", "hinge", (body_hx, 0.0, 0.0), (0, 1, 0),
                                (0.0, 1.5)), "parent": 1})
        parts.append({"label": "racks", "box": ((-hx + 0.03, -hy + 0.02, H - 0.15),
                                                 (body_hx - 0.03, hy - 0.02, H + 0.05)),
                      "joint": ("rack_joint", "slide", (0.0, 0.0, H - 0.05), (1, 0, 0),
                                (0.0, 1.0)), "parent": 1})
        if spec.factory == "dishwasher":
            for k, y in enumerate((-hy - 0.15, 0.0)):
                parts.append({"label": "buttons", "box": ((hx, y - 0.02, H - 0.06),
                                                           (hx + 0.02, y + 0.02, H - 0.03)),
                              "joint": ("button_joint", "slide", (hx, y, H - 0.045),
                                        (1, 0, 0), (-0.02, 0.0)), "parent": 2})
        return parts
    if spec.factory == "refrigerator":
        parts.append({"label": "door1", "box": ((body_hx, -hy, 0.0), (hx, hy, H)),
                      "joint": ("door_hinge_3", "hinge", (hx, -hy, H / 2), (0, 0, -1),
                                (0.0, math.pi)), "parent": 1})
        parts.append({"label": "drawer2", "box": ((-hx + 0.05, -hy + 0.05, 0.06),
                                                   (body_hx - 0.05, hy - 0.05, 0.25)),
                      "joint": ("drawer_joint_2", "slide", (0.0, 0.0, 0.15), (1, 0, 0),
                                (0.0, 0.4 * D)), "parent": 1})
        return parts
    raise KeyError(spec.factory)


def fake_export(spec: Spec, folder: Path | str) -> Path:
    """Write a stand-in Infinigen export for ``spec`` into ``folder`` (see `FakeRunner`)."""
    from PIL import Image

    folder = Path(folder)
    export = folder / "export" / spec.factory / str(spec.seed)
    for sub in ("visual", "collision", "textures"):
        (export / "assets" / sub).mkdir(parents=True, exist_ok=True)
    parts = _fake_parts(spec)
    rng = np.random.default_rng(spec.seed)
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets", inertiagrouprange="0 0")
    ET.SubElement(ET.SubElement(root, "default"), "geom", rgba="1 1 1 1")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    obj = ET.SubElement(world, "body", name="object")
    bodies: list[ET.Element] = []
    origins: list[np.ndarray] = []
    counts: dict[str, int] = {}
    labels = {}
    for i, part in enumerate(parts):
        lo, hi = (np.asarray(v, float) for v in part["box"])
        centre = (lo + hi) / 2
        parent_index = part.get("parent")
        parent = obj if parent_index is None else bodies[parent_index - 1]
        parent_origin = np.zeros(3) if parent_index is None else origins[parent_index - 1]
        body = ET.SubElement(parent, "body", name=f"link_{i}",
                             pos=" ".join(f"{v:.6g}" for v in centre - parent_origin))
        bodies.append(body)
        origins.append(centre)
        n = counts.get(part["label"], 0)
        counts[part["label"]] = n + 1
        name = f"{part['label']}_{n}"
        labels[i] = part["label"]
        (export / "assets" / "visual" / f"{name}.obj").write_text(_box_obj(lo - centre,
                                                                           hi - centre))
        colour = tuple(int(v) for v in rng.integers(40, 220, 3))
        tex = Image.new("RGB", (16, 16), colour)
        tex.putpixel((3, 3), (0, 0, 0))                   # some texture to keep
        tex.save(export / "assets" / "textures" / f"{name}_DIFFUSE.png")
        trans = 200 if part.get("glass") else 0
        Image.new("L", (16, 16), trans).save(export / "assets" / "textures" /
                                             f"{name}_TRANSMISSION.png")
        ET.SubElement(asset, "mesh", name=name, file=f"visual/{name}.obj", inertia="legacy")
        ET.SubElement(asset, "material", name=f"{name}_mat", texture=f"{name}_tex")
        ET.SubElement(asset, "texture", name=f"{name}_tex", type="2d",
                      file=f"assets/textures/{name}_DIFFUSE.png")
        density = f"{rng.uniform(2500, 5000):.6g}"
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, group="1", contype="0",
                      conaffinity="0", friction="1.1 0.005 0.0001", density=density,
                      material=f"{name}_mat", pos="0 0 0")
        # two hulls per part, splitting it along its longest side (CoACD's pieces), unless the
        # part says how it is cut
        axis = int(np.argmax(hi - lo))
        mid = (lo[axis] + hi[axis]) / 2
        pieces = []
        for a, b in ((lo[axis], mid), (mid, hi[axis])):
            plo, phi = lo.copy(), hi.copy()
            plo[axis], phi[axis] = a, b
            pieces.append((plo, phi))
        if part.get("hulls"):
            pieces = [tuple(np.asarray(v, float) for v in box) for box in part["hulls"]]
        for k, (plo, phi) in enumerate(pieces):
            col = f"{name}_col{k}"
            (export / "assets" / "collision" / f"{col}.obj").write_text(
                _box_obj(plo - centre, phi - centre))
            ET.SubElement(asset, "mesh", name=col, file=f"collision/{col}.obj", inertia="convex")
            ET.SubElement(body, "geom", name=col, type="mesh", mesh=col, group="0", contype="1",
                          conaffinity="1", friction="1.1 0.005 0.0001", density=density,
                          pos="0 0 0")
        if part.get("joint"):
            jname, jtype, jpos, jaxis, jrange = part["joint"]
            ET.SubElement(body, "joint", name=f"{jname}_{i}", type=jtype,
                          pos=" ".join(f"{v:.6g}" for v in np.asarray(jpos, float) - centre),
                          axis=" ".join(f"{float(v):.6g}" for v in jaxis), limited="true",
                          range=f"{jrange[0]:.6g} {jrange[1]:.6g}", ref="0.0", stiffness="500",
                          damping="1000", frictionloss="1000")
    contact = ET.SubElement(root, "contact")
    for i, part in enumerate(parts):
        if part.get("parent"):
            ET.SubElement(contact, "exclude", body1=f"link_{part['parent'] - 1}",
                          body2=f"link_{i}")
    ET.indent(root, space="  ")
    xml = export / f"{spec.factory}.xml"
    xml.write_text(ET.tostring(root, encoding="unicode") + "\n")
    lo = np.min([p["box"][0] for p in parts], axis=0)
    hi = np.max([p["box"][1] for p in parts], axis=0)
    (export / "metadata.json").write_text(json.dumps(
        {"part_labels": sorted(set(labels.values())),
         "bounding_box": {"min": lo.tolist(), "max": hi.tolist()}}, indent=1))
    rec = {"key": spec.key, "factory": spec.factory, "size": list(spec.size), "seed": spec.seed,
           "options": spec.opts, "worker_version": WORKER_VERSION, "ok": True,
           "mjcf": xml.relative_to(folder).as_posix(), "seconds": 0.0, "fake": True}
    (folder / "result.json").write_text(json.dumps(rec, indent=1))
    return folder


# ── reading an export ────────────────────────────────────────────────────────────────────────────


@dataclass
class MeshPart:
    """One mesh geom of the export, posed in Infinigen's object frame (closed): its OBJ, where
    the OBJ's origin is (``pos``), how it is turned (``rot``, a signed permutation), its label
    (the geom name without its index) and, for a visual, its diffuse texture."""

    name: str
    label: str
    file: Path
    pos: np.ndarray
    rot: np.ndarray
    texture: Path | None = None
    glass: bool = False
    friction: float | None = None
    _vertices: np.ndarray | None = field(default=None, repr=False)

    def vertices(self) -> np.ndarray:
        """The OBJ's vertices in the object frame."""
        if self._vertices is None:
            self._vertices = _obj_vertices(self.file)
        return self._vertices @ self.rot.T + self.pos

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        v = self.vertices()
        if not len(v):
            return self.pos.copy(), self.pos.copy()
        return v.min(axis=0), v.max(axis=0)


@dataclass
class JointSpec:
    name: str
    type: str                  # hinge | slide
    pos: np.ndarray            # object frame
    axis: np.ndarray           # object frame, unit
    range: tuple[float, float]


@dataclass
class Link:
    """One rigid part after cleaning. ``parent`` is None for the base; ``origin`` is where its
    frame is in the object frame (closed), ``joint`` how it moves against its parent."""

    name: str
    parent: str | None
    origin: np.ndarray
    joint: JointSpec | None = None
    visuals: list[MeshPart] = field(default_factory=list)
    colliders: list[MeshPart] = field(default_factory=list)
    part: str = ""             # what it is: base, door, drawer, rack, sash, flap ...

    def labels(self) -> set[str]:
        return {v.label for v in self.visuals}


@dataclass
class Asset:
    spec: Spec | None
    folder: Path
    links: dict[str, Link]
    root: str
    notes: list[str] = field(default_factory=list)

    def children(self, name: str) -> list[Link]:
        return [lk for lk in self.links.values() if lk.parent == name]

    def moving(self) -> list[Link]:
        return [lk for lk in self.links.values() if lk.joint is not None]


def _obj_vertices(path: Path) -> np.ndarray:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def _vec(text: str | None, default: Sequence[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(v) for v in text.split()], dtype=float)


def _quat_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / (np.linalg.norm(q) or 1.0)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mat_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, float)
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def _signed_permutation(m: np.ndarray, tol: float = 1e-3) -> np.ndarray | None:
    """``m`` rounded to the axis permutation (with signs) it is, or None if it is not one."""
    out = np.round(m)
    if np.abs(m - out).max() > tol or not np.allclose(np.abs(out).sum(axis=0), 1):
        return None
    return out


def _label(name: str) -> str:
    return re.sub(r"_\d+$", "", re.sub(r"_col\d+$", "", name))


def _mean_8bit(path: Path) -> float:
    from PIL import Image

    try:
        with Image.open(path) as im:
            return float(np.asarray(im.convert("L"), dtype=float).mean())
    except OSError:
        return 0.0


def load_asset(folder: Path | str, spec: Spec | None = None) -> Asset:
    """Read an export folder (``result.json`` + Infinigen's MJCF) into links in the object frame.
    Bodies and geoms may be rotated only by quarter turns (Infinigen's never are)."""
    folder = Path(folder)
    xml = _mjcf_of(folder)
    if xml is None:
        raise FileNotFoundError(f"{folder}: no finished Infinigen export (result.json + MJCF)")
    root = ET.parse(xml).getroot()
    compiler = root.find("compiler")
    meshdir = (compiler.get("meshdir") if compiler is not None else None) or ""
    texdir = (compiler.get("texturedir") if compiler is not None else None) or ""
    meshes = {m.get("name"): xml.parent / meshdir / m.get("file") for m in root.iter("mesh")}
    textures = {t.get("name"): xml.parent / texdir / t.get("file") for t in root.iter("texture")}
    materials = {m.get("name"): m.get("texture") for m in root.iter("material")}
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{xml}: no worldbody")
    links: dict[str, Link] = {}
    notes: list[str] = []

    def walk(body: ET.Element, parent: str | None, R: np.ndarray, p: np.ndarray) -> None:
        R = R @ _quat_mat(_vec(body.get("quat"), (1, 0, 0, 0)))
        name = body.get("name") or f"body{len(links)}"
        link = Link(name, parent, p.copy())
        for joint in body.findall("joint"):
            jtype = joint.get("type", "hinge")
            if jtype not in ("hinge", "slide"):
                notes.append(f"{name}: {jtype} joint ignored")
                continue
            axis = R @ _vec(joint.get("axis"), (0, 0, 1))
            axis = axis / (np.linalg.norm(axis) or 1.0)
            limited = joint.get("limited", "true") != "false"
            lo, hi = _vec(joint.get("range"), (0, 0)) if limited else (0.0, 0.0)
            link.joint = JointSpec(joint.get("name") or f"{name}_joint", jtype,
                                   p + R @ _vec(joint.get("pos"), (0, 0, 0)), axis,
                                   (float(lo), float(hi)))
            break                                     # one joint per link, as Infinigen exports
        for geom in body.findall("geom"):
            if geom.get("type") != "mesh" or geom.get("mesh") not in meshes:
                continue
            gR = R @ _quat_mat(_vec(geom.get("quat"), (1, 0, 0, 0)))
            perm = _signed_permutation(gR)
            if perm is None:
                raise ValueError(f"{xml.name}: geom {geom.get('name')} is turned by other than "
                                 "a quarter turn; it cannot be scaled per axis")
            gname = geom.get("name") or geom.get("mesh")
            friction = geom.get("friction")
            part = MeshPart(gname, _label(gname), meshes[geom.get("mesh")],
                            p + R @ _vec(geom.get("pos"), (0, 0, 0)), perm,
                            friction=float(friction.split()[0]) if friction else None)
            if geom.get("group", "0") == "1":
                tex = textures.get(materials.get(geom.get("material")))
                part.texture = tex if tex is not None and tex.is_file() else None
                if tex is not None:
                    trans = tex.with_name(tex.name.replace("_DIFFUSE", "_TRANSMISSION"))
                    part.glass = trans.is_file() and _mean_8bit(trans) > GLASS_TRANSMISSION
                link.visuals.append(part)
            else:
                link.colliders.append(part)
        links[name] = link
        for child in body.findall("body"):
            walk(child, name, R, p + R @ _vec(child.get("pos"), (0, 0, 0)))

    tops = world.findall("body")
    if not tops:
        raise ValueError(f"{xml}: no bodies")
    for top in tops:
        walk(top, None, np.eye(3), _vec(top.get("pos"), (0, 0, 0)))
    # The exporter's ``object`` wrapper holds nothing: its first link is the base.
    roots = [lk for lk in links.values() if lk.parent is None]
    root_name = roots[0].name
    if len(roots) == 1 and not roots[0].visuals and not roots[0].colliders \
            and roots[0].joint is None:
        kids = [lk for lk in links.values() if lk.parent == root_name]
        if len(kids) == 1:
            del links[root_name]
            kids[0].parent = None
            root_name = kids[0].name
    if spec is None:
        try:
            rec = json.loads((folder / "result.json").read_text())
            spec = Spec.make(rec["factory"], rec["size"], seed=int(rec.get("seed", 0)),
                             **(rec.get("options") or {}))
        except (OSError, ValueError, KeyError):
            spec = None
    return Asset(spec, folder, links, root_name, notes)


# ── cleaning ─────────────────────────────────────────────────────────────────────────────────────


def _has(text: str, words: Iterable[str]) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def _minor_label(label: str) -> bool:
    """A handle's, knob's, button's or lock's mesh — but not a part that merely carries them
    (the dishwasher's whole front panel is ``_with_buttons``)."""
    return _has(label, MINOR_WORDS) and "with" not in label.lower()


def _is_minor(link: Link) -> bool:
    """A handle, knob, button or lock lever: what a part is held by, not a part."""
    if link.joint is not None and _has(link.joint.name, MINOR_WORDS):
        return True
    labels = link.labels() or {c.label for c in link.colliders}
    return bool(labels) and all(_minor_label(lb) for lb in labels)


def _part_kind(link: Link) -> str:
    if link.joint is None:
        return "base"
    words = (link.joint.name + " " + " ".join(sorted(link.labels()))).lower()
    for token, kind in (("drawer", "drawer"), ("rack", "rack"), ("shel", "shelf"),
                        ("sliding", "sash"), ("pane", "sash"), ("hingepanel", "sash"),
                        ("door", "door"), ("hinge", "door")):
        if token in words:
            return kind
    return "flap" if link.joint.type == "hinge" else "slide"


def _bounds(parts: Iterable[MeshPart]) -> tuple[np.ndarray, np.ndarray] | None:
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for part in parts:
        a, b = part.bounds()
        lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    return (lo, hi) if np.all(np.isfinite(lo)) else None


def main_parts(asset: Asset) -> list[MeshPart]:
    """The visual meshes the object's size is measured on: not its handles, knobs or buttons."""
    return [v for lk in asset.links.values() for v in lk.visuals
            if not _minor_label(v.label) and not _has(v.label, JUNK_WORDS)]


def main_bounds(asset: Asset) -> tuple[np.ndarray, np.ndarray]:
    box = _bounds(main_parts(asset)) or _bounds(v for lk in asset.links.values()
                                                for v in lk.visuals + lk.colliders)
    if box is None:
        raise ValueError(f"{asset.folder}: the asset has no meshes")
    return box


def clean(asset: Asset) -> Asset:
    """Weld the minor links on, drop parts outside the object, keep inner parts inside, clamp
    over-long slides and name every part (see the module docstring). Returns a new Asset."""
    asset = copy.deepcopy(asset)
    links, notes = asset.links, asset.notes
    # 1. handles, knobs, buttons, locks: welded to what they hang on
    for name in [n for n, lk in links.items() if lk.parent is not None and _is_minor(lk)]:
        link = links.pop(name)
        parent = links[link.parent]
        parent.visuals += link.visuals
        parent.colliders += link.colliders
        for child in links.values():
            if child.parent == name:
                child.parent = link.parent
        notes.append(f"{name} ({link.joint.name if link.joint else 'fixed'}) welded to "
                     f"{link.parent}")
    # 2. junk, and minor parts a sizing bug left outside the object
    lo, hi = main_bounds(asset)
    for link in links.values():
        keep_v, keep_c = [], []
        dropped = set()
        for v in link.visuals:
            a, b = v.bounds()
            # A handle may stand proud of the front — and of the back too: a door has one on
            # each face. Nothing may stand off the sides, the top or the bottom.
            outside = np.maximum(lo - a, b - hi)
            if _has(v.label, JUNK_WORDS) or (_minor_label(v.label) and np.any(
                    outside > np.array([OUTLIER_FRONT, OUTLIER, OUTLIER]))):
                dropped.add(v.name)
            else:
                keep_v.append(v)
        for c in link.colliders:
            owner = re.sub(r"_col\d+$", "", c.name)
            if owner not in dropped:
                keep_c.append(c)
        if dropped:
            notes.append(f"{link.name}: dropped {len(dropped)} part(s) outside the object "
                         f"({', '.join(sorted(dropped)[:4])})")
        link.visuals, link.colliders = keep_v, keep_c
    # 3. inner moving parts kept inside their parent; 4. slides no longer than their parent
    for link in links.values():
        if link.joint is None or link.parent is None:
            continue
        parent_box = _bounds(links[link.parent].visuals)
        own_box = _bounds(link.visuals)
        if parent_box is None or own_box is None:
            continue
        if link.joint.type == "slide":
            plo, phi = parent_box
            olo, ohi = own_box
            shift = 0.0
            if ohi[2] > phi[2] + POKE and abs(link.joint.axis[2]) < 0.5:
                shift = phi[2] - POKE - ohi[2]
            elif olo[2] < plo[2] - POKE and abs(link.joint.axis[2]) < 0.5:
                shift = plo[2] + POKE - olo[2]
            if shift:
                _translate(link, np.array([0.0, 0.0, shift]))
                notes.append(f"{link.name}: moved {shift * 100:+.1f} cm to stay inside "
                             f"{link.parent}")
            extent = float(np.abs((phi - plo) @ np.abs(link.joint.axis)))
            lo_r, hi_r = link.joint.range
            if extent > 0 and max(abs(lo_r), abs(hi_r)) > SLIDE_FULL * extent:
                limit = SLIDE_CLAMP * extent
                link.joint.range = (max(lo_r, -limit), min(hi_r, limit))
                notes.append(f"{link.name}: slide clamped from {max(abs(lo_r), abs(hi_r)):.2f} "
                             f"to {limit:.2f} m (its parent is {extent:.2f} m deep)")
    # 5. names: what each part is, numbered top to bottom then left to right
    for link in links.values():
        link.part = _part_kind(link)
    return asset


def _translate(link: Link, delta: np.ndarray) -> None:
    link.origin = link.origin + delta
    if link.joint is not None:
        link.joint.pos = link.joint.pos + delta
    for part in link.visuals + link.colliders:
        part.pos = part.pos + delta


def _face_area(link: Link) -> float:
    """The area of a part's largest face: a door's leaf is its biggest moving part."""
    box = _bounds(link.visuals)
    return float(np.prod(np.sort(box[1] - box[0])[1:])) if box else 0.0


def drop_sill(asset: Asset) -> tuple[Asset, float]:
    """A door without the colliders of its frame's bottom bar, and the share of the asset's
    height that lies below the leaf.

    Infinigen's square frame runs all the way round, so its bottom bar is a curb across the
    doorway (6 cm in both scans) that the quick room's flush doorway does not have: a chair
    dragged through snags on it. The caller sinks that share under the floor (the leaf then
    hangs as the quick room's does, a centimetre clear of it); only the bar's look stays, as a
    threshold. Returns the asset unchanged and 0 when there is no leaf above a bar."""
    moving = asset.moving()
    if not moving:
        return asset, 0.0
    lo, hi = main_bounds(asset)
    leaf = max(moving, key=_face_area)
    box = _bounds(leaf.visuals)
    if box is None:
        return asset, 0.0
    gap = float(box[0][2] - lo[2])
    if gap <= 1e-3:
        return asset, 0.0
    asset = copy.deepcopy(asset)
    base = asset.links[asset.root]
    keep, dropped = [], []
    for c in base.colliders:
        (dropped if float(c.bounds()[1][2]) <= lo[2] + gap + 1e-3 else keep).append(c)
    base.colliders = keep
    if dropped:
        asset.notes.append(f"{base.name}: the frame's bottom bar has no collider "
                           f"({', '.join(c.name for c in dropped)}; {gap * 100:.1f} cm tall)")
    return asset, gap / max(float(hi[2] - lo[2]), 1e-6)


def part_names(asset: Asset, prefix: str, *, primary: str | None = None) -> dict[str, str]:
    """Body names for the asset's links: the base is ``prefix``; each moving part
    ``<prefix>_<part>`` (``_<part>0``, ``_<part>1``, ... top to bottom, then from the viewer's
    left, when there are several). ``primary`` names the largest door instead (a room door's leaf
    keeps the door's own handle, so a live swap carries its angle)."""
    names = {asset.root: prefix}
    moving = asset.moving()
    biggest = None
    if primary is not None and moving:
        biggest = max(moving, key=_face_area).name
        names[biggest] = primary
    groups: dict[str, list[Link]] = {}
    for link in moving:
        if link.name != biggest:
            groups.setdefault(link.part or "part", []).append(link)
    for kind, group in sorted(groups.items()):
        def order(lk: Link) -> tuple[float, float]:
            box = _bounds(lk.visuals) or (lk.origin, lk.origin)
            centre = (box[0] + box[1]) / 2
            return (-round(float(centre[2]), 2), round(float(centre[1]), 2))

        group.sort(key=order)
        for k, link in enumerate(group):
            names[link.name] = f"{prefix}_{kind}" if len(group) == 1 else f"{prefix}_{kind}{k}"
    return names


# ── into the scene ───────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Placement:
    """Where the asset goes, in the frame of the body that will hold it (quickroom's box frame:
    +X across the front, front towards -Y, +Z up). The asset's main box is scaled onto
    ``size`` (W, D, H) centred on ``centre`` — except that an opening (``keep_depth``) keeps its
    own thickness and puts its front face at ``front_y`` instead. ``turned`` faces the asset's
    front to +Y: RoomPlan had the box's front backwards (it is against a wall)."""

    centre: tuple[float, float, float]
    size: tuple[float, float, float]
    keep_depth: bool = False
    front_y: float | None = None
    turned: bool = False

    @property
    def rotation(self) -> np.ndarray:
        """Infinigen's frame into the box frame."""
        return (np.diag([-1.0, -1.0, 1.0]) @ R_INFINIGEN) if self.turned else R_INFINIGEN


@dataclass(frozen=True)
class Overrides:
    """Our physics and look for one object: its estimated ``mass`` (kg, box volume x
    ``OCCUPANCY_DENSITY``), shared by all its links — or, ``moving_mass``, only by its moving
    ones (a door leaf: its frame is part of the building) —, its collider ``friction``, the photo
    colour to ``tint`` it (sRGB 0-1), and where a door starts (``open_angle``, radians)."""

    mass: float
    friction: float = properties.DEFAULT_FRICTION
    tint: tuple[float, float, float] | None = None
    moving_mass: bool = False
    open_angle: float = 0.0


@dataclass
class Built:
    """One asset as MJCF for the scene: ``base_geoms`` go into the body named ``names[root]``,
    ``children`` are its child bodies, ``assets`` the meshes/textures/materials, ``excludes``
    the contact pairs to switch off, ``files`` what to write (relative path -> bytes or a file
    to copy) under the scene's mesh / texture folders."""

    handle: str
    names: dict[str, str]
    base_geoms: list[ET.Element]
    children: list[ET.Element]
    assets: list[ET.Element]
    excludes: list[tuple[str, str]]
    mesh_files: dict[str, Path]
    texture_files: dict[str, bytes | Path]
    report: dict
    root: str = ""


def _fmt(values: Iterable[float]) -> str:
    return " ".join(f"{float(v):.6g}" if abs(float(v)) > 1e-9 else "0" for v in values)


def _snap(axis: np.ndarray) -> np.ndarray:
    k = int(np.argmax(np.abs(axis)))
    if math.acos(min(1.0, abs(float(axis[k])))) < AXIS_SNAP:
        out = np.zeros(3)
        out[k] = math.copysign(1.0, float(axis[k]))
        return out
    return axis / (np.linalg.norm(axis) or 1.0)


def tint_texture(path: Path, rgb: Sequence[float]) -> bytes:
    """The texture recoloured to ``rgb`` (sRGB 0-1) with its own shading kept: every texel
    becomes ``rgb`` times its luminance over the texture's mean luminance — grain and panel
    edges stay, the colour becomes the photo's. The mean is taken over the BAKED texels only:
    Infinigen's atlases leave a third to a half of the image black and unused, and counting it
    pushed every real texel to white. PNG bytes."""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=float) / 255.0
    lum = arr @ np.array([0.2126, 0.7152, 0.0722])
    baked = lum > UNUSED_TEXEL
    mean = float(lum[baked].mean()) if baked.any() else 0.0
    ratio = np.where(baked, lum / mean, 1.0) if mean > 1e-3 else np.ones_like(lum)
    # Shading is kept, but only so much of it: a near-black bake artifact stays dark, not black;
    # and lifting a dark bake to a light colour (a black window frame made white, 8x brighter)
    # lifts its speckle with it — the darker the bake and the brighter the tint, the less of its
    # shading is kept. Unused texels take the plain colour, so a UV island's edge never shows them.
    target = _luminance(rgb)
    detail = min(float(np.clip(mean / DETAIL_LUMINANCE, 0.15, 1.0)),
                 float(np.clip(2.0 * mean / max(target, 1e-3), 0.1, 1.0)))
    ratio = 1.0 + detail * (np.clip(ratio, 0.35, 1.8) - 1.0)
    out = np.clip(ratio[..., None] * np.asarray(rgb, float)[None, None, :3], 0.0, 1.0)
    buffer = io.BytesIO()
    Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _material_volume(part: MeshPart, scale: np.ndarray) -> float:
    import trimesh

    v = _obj_vertices(part.file)
    faces = _obj_faces(part.file)
    if not len(v) or not len(faces):
        return 0.0
    mesh = trimesh.Trimesh(v * (np.abs(part.rot.T) @ scale), faces, process=False)
    return float(properties._material_volume(mesh))


def _hull_volume(part: MeshPart, scale: np.ndarray) -> float:
    import trimesh

    v = _obj_vertices(part.file) * (np.abs(part.rot.T) @ scale)
    if len(v) < 4:
        return 0.0
    try:
        return float(trimesh.Trimesh(v, process=False).convex_hull.volume)
    except Exception:  # noqa: BLE001 — a flat or degenerate piece
        return 0.0


def _obj_faces(path: Path) -> np.ndarray:
    faces = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    arr = np.asarray(faces, dtype=int).reshape(-1, 3)
    return arr - 1 if len(arr) and arr.min() > 0 else arr


def fit_transform(asset: Asset, placement: Placement) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(scale S per Infinigen axis, Infinigen point the box centre maps from, box-frame point it
    maps to): a point ``p`` of the asset lands at ``to + R (S * (p - frm))``."""
    lo, hi = main_bounds(asset)
    ext = np.maximum(hi - lo, 1e-3)
    W, D, H = (float(v) for v in placement.size)
    S = np.array([1.0 if placement.keep_depth else D / ext[0], W / ext[1], H / ext[2]])
    frm = (lo + hi) / 2
    to = np.asarray(placement.centre, dtype=float).copy()
    if placement.keep_depth and placement.front_y is not None:
        # the asset's front (+X max) on the plane y = front_y; R maps +X to -Y
        to[1] = float(placement.front_y) + ext[0] / 2
    return S, frm, to


def build(asset: Asset, handle: str, placement: Placement, overrides: Overrides, *,
          prefix: str | None = None, primary: str | None = None, asset_key: str | None = None
          ) -> Built:
    """MJCF for one cleaned asset fitted into ``placement``, with ``overrides`` (see `Built`).
    ``prefix`` names the bodies (default ``handle``); ``primary`` the largest moving part."""
    S, frm, to = fit_transform(asset, placement)
    R = placement.rotation

    def T(p: np.ndarray) -> np.ndarray:
        return to + R @ (S * (np.asarray(p, float) - frm))

    key = asset_key or (asset.spec.key if asset.spec else asset.folder.name)
    names = part_names(asset, prefix or handle, primary=primary)
    tag = re.sub(r"[^A-Za-z0-9_]", "_", handle)
    folder = f"{ARTICULATED_DIR}/{key}"
    report: dict[str, Any] = {"key": key, "scale": [round(float(v), 4) for v in S],
                              "turned": placement.turned, "notes": list(asset.notes),
                              "parts": {}}

    # masses by material volume
    vol = {name: sum(_material_volume(v, S) for v in lk.visuals if not v.glass)
           or sum(_hull_volume(c, S) for c in lk.colliders) or 1e-6
           for name, lk in asset.links.items()}
    sharing = [n for n, lk in asset.links.items() if lk.joint is not None] \
        if overrides.moving_mass else list(asset.links)
    if not sharing:
        sharing = list(asset.links)
    total_vol = sum(vol[n] for n in sharing) or 1e-6
    rate = overrides.mass / total_vol
    masses = {n: max(properties.MIN_LINK_MASS,
                     overrides.mass * vol[n] / total_vol if n in sharing else rate * vol[n])
              for n in asset.links}

    assets: list[ET.Element] = []
    mesh_files: dict[str, Path] = {}
    texture_files: dict[str, bytes | Path] = {}
    taken: set[str] = set()

    def unique(name: str) -> str:
        out, k = name, 1
        while out in taken:
            k += 1
            out = f"{name}_{k}"
        taken.add(out)
        return out

    def mesh_asset(part: MeshPart, kind: str) -> str:
        rel = f"{folder}/{kind}/{part.file.name}"
        mesh_files[rel] = part.file
        name = unique(f"ig_{tag}_{part.name}")
        scale = np.abs(part.rot.T) @ S
        ET_mesh = ET.Element("mesh", name=name, file=rel, scale=_fmt(scale),
                             inertia="convex" if kind == "collision" else "shell")
        assets.append(ET_mesh)
        return name

    def material(part: MeshPart, minor: bool) -> str:
        name = unique(f"ig_{tag}_{part.name}")
        if part.glass:
            assets.append(ET.Element("material", name=name, rgba=_fmt(GLASS_RGBA),
                                     specular="0.5", shininess="0.8"))
            return name
        if part.texture is None:
            assets.append(ET.Element("material", name=name, rgba="0.7 0.7 0.7 1"))
            return name
        if overrides.tint is not None and not minor:
            hexrgb = "".join(f"{int(round(c * 255)):02x}" for c in overrides.tint)
            rel = f"{folder}/textures/{part.texture.stem}.{hexrgb}.png"
            texture_files[rel] = tint_texture(part.texture, overrides.tint)
        else:
            rel = f"{folder}/textures/{part.texture.name}"
            texture_files[rel] = part.texture
        tex = unique(f"ig_{tag}_{part.name}_tex")
        assets.append(ET.Element("texture", name=tex, type="2d", file=rel))
        assets.append(ET.Element("material", name=name, texture=tex, specular="0.2",
                                 shininess="0.3"))
        return name

    def geoms(link: Link, body_origin: np.ndarray, body_name: str) -> list[ET.Element]:
        out = []
        cvols = [_hull_volume(c, S) for c in link.colliders]
        csum = sum(cvols)
        for v in link.visuals:
            minor = _minor_label(v.label)
            el = ET.Element("geom", name=unique(f"{body_name}_{v.name}"), type="mesh",
                            mesh=mesh_asset(v, "visual"), pos=_fmt(T(v.pos) - body_origin),
                            quat=_fmt(_mat_quat(R @ v.rot)), material=material(v, minor),
                            **VISUAL_ATTRS)
            out.append(el)
        friction = f"{overrides.friction:.3g} 0.03 0.005"
        for c, cv in zip(link.colliders, cvols, strict=True):
            mass = masses[link.name] * (cv / csum if csum > 0 else 1.0 / len(link.colliders))
            out.append(ET.Element("geom", name=unique(f"{body_name}_{c.name}"), type="mesh",
                                  mesh=mesh_asset(c, "collision"),
                                  pos=_fmt(T(c.pos) - body_origin),
                                  quat=_fmt(_mat_quat(R @ c.rot)), group=COLLIDER_GROUP,
                                  mass=f"{max(mass, 1e-4):.6g}", friction=friction))
        return out

    def joint_dynamics(link: Link) -> dict[str, str]:
        kind = "revolute" if link.joint.type == "hinge" else "prismatic"
        d = properties.JOINT_DEFAULTS[kind]
        friction = float(d["friction"])
        why = "JOINT_DEFAULTS"
        axis = _snap(R @ (S * link.joint.axis))
        if link.joint.type == "hinge" and abs(axis[2]) < 0.5:
            # a flap: friction just over what its own weight can twist it by
            box = _bounds(link.visuals) or (link.origin, link.origin)
            com = T((box[0] + box[1]) / 2)
            arm = com - T(link.joint.pos)
            lever = float(np.linalg.norm(arm - axis * float(arm @ axis)))
            friction = max(friction, FLAP_HOLD * masses[link.name] * GRAVITY * lever)
            why = f"flap: holds its own weight ({FLAP_HOLD:g} x m g r)"
            return {"damping": f"{float(d['damping']):.4g}", "frictionloss": f"{friction:.4g}",
                    "armature": f"{ARMATURE:g}", "stiffness": "0",
                    "solreffriction": FLAP_SOLREF, "solimpfriction": FLAP_SOLIMP, "_why": why}
        return {"damping": f"{float(d['damping']):.4g}", "frictionloss": f"{friction:.4g}",
                "armature": f"{ARMATURE:g}", "stiffness": "0", "_why": why}

    base = asset.links[asset.root]
    base_geoms = geoms(base, np.zeros(3), names[asset.root])
    report["parts"][names[asset.root]] = {"kind": "base", "mass_kg": round(masses[asset.root], 3),
                                          "material_volume_m3": round(vol[asset.root], 6)}

    def child_bodies(parent: Link, parent_origin: np.ndarray) -> list:
        out = []
        for link in sorted(asset.children(parent.name), key=lambda lk: names[lk.name]):
            name = names[link.name]
            origin = T(link.joint.pos) if link.joint is not None else T(link.origin)
            attrs = {"name": name, "pos": _fmt(origin - parent_origin)}
            entry: dict[str, Any] = {"kind": link.part, "mass_kg": round(masses[link.name], 3),
                                     "material_volume_m3": round(vol[link.name], 6)}
            body = ET.Element("body", attrs)
            if link.joint is not None:
                jt = link.joint
                axis = _snap(R @ (S * jt.axis))
                lo_r, hi_r = jt.range
                if jt.type == "slide":
                    along = float(np.linalg.norm(S * jt.axis))
                    lo_r, hi_r = lo_r * along, hi_r * along
                dyn = joint_dynamics(link)
                why = dyn.pop("_why")
                angle = 0.0
                if name == primary and overrides.open_angle and jt.type == "hinge":
                    angle = float(np.clip(overrides.open_angle, lo_r, hi_r))
                    if angle:
                        body.set("quat", " ".join(f"{v:.10g}" for v in _axis_quat(axis, angle)))
                jname = f"{name}_{jt.type}"
                ET.SubElement(body, "joint", name=jname, type=jt.type, pos="0 0 0",
                              axis=_fmt(axis), range=_fmt((lo_r, hi_r)), ref=f"{angle:.10g}",
                              **dyn)
                entry.update({"joint": jname, "type": jt.type, "axis": [round(float(a), 4)
                                                                         for a in axis],
                              "range": [round(float(lo_r), 4), round(float(hi_r), 4)],
                              "damping": float(dyn["damping"]),
                              "frictionloss": float(dyn["frictionloss"]), "dynamics": why,
                              "from_infinigen": jt.name, "start": round(angle, 4)})
            for el in geoms(link, origin, name):
                body.append(el)
            if not link.colliders:
                body.append(_inertial(link, T, origin, masses[link.name]))
            for el in child_bodies(link, origin):
                body.append(el)
            report["parts"][name] = entry
            out.append(body)
        return out

    children = child_bodies(base, np.zeros(3))
    # A part never collides with what it hangs on — as Infinigen's own export has it: MuJoCo
    # skips a parent and its child only when the parent moves, and a cabinet is welded to the
    # world — nor with a part it touches at rest; every other pair does (`rest_contacts`).
    hung = {tuple(sorted((names[lk.parent], names[lk.name]))) for lk in asset.links.values()
            if lk.parent is not None}
    touching = set(rest_contacts(names[asset.root], base_geoms, children, assets, mesh_files))
    excludes = sorted(hung | touching)
    report["contacts_off"] = [f"{a} <-> {b}" for a, b in sorted(touching - hung)]
    lo, hi = main_bounds(asset)
    report.update({
        "asset_size_m": [round(float(v), 4) for v in (hi - lo)[[1, 0, 2]]],
        "target_size_m": [round(float(v), 4) for v in placement.size],
        "mass_kg": round(sum(masses[n] for n in sharing), 3),
        # what compiles into the scene beyond the estimate: a door's or a window's frame is the
        # building's (welded, at the same density as its moving parts)
        "static_mass_kg": round(sum(masses[n] for n in asset.links if n not in sharing), 3),
        "mass_rule": ("moving parts share the estimate (the frame is the building's)"
                      if overrides.moving_mass else
                      "box volume x OCCUPANCY_DENSITY, shared by material volume"),
        "tint": None if overrides.tint is None else
        "#" + "".join(f"{int(round(c * 255)):02x}" for c in overrides.tint),
    })
    return Built(handle, names, base_geoms, children, assets, excludes, mesh_files,
                 texture_files, report, root=names[asset.root])


def rest_contacts(root: str, base_geoms: Sequence[ET.Element], children: Sequence[ET.Element],
                  assets: Sequence[ET.Element], mesh_files: Mapping[str, Path], *,
                  gap: float = REST_GAP) -> list[tuple[str, str]]:
    """The pairs of the asset's own bodies whose colliders are within ``gap`` of each other at
    rest — as built (a door started open) or with every joint closed. Of the parts that do not
    hang on one another, only these have their contact switched off (`build`).

    Infinigen's hulls of neighbouring parts overlap by a millimetre or so where they meet (the
    two doors of a cabinet), and an overlap at t=0 is an explosion, so those pairs must not
    collide. Every other pair must: switching all of them off let a dishwasher's rack slide out
    through its closed door and the door close through a pulled-out rack."""
    import mujoco

    root_el = ET.Element("mujoco", model="rest_contacts")
    ET.SubElement(root_el, "compiler", angle="radian", autolimits="true")
    asset = ET.SubElement(root_el, "asset")
    body = ET.SubElement(ET.SubElement(root_el, "worldbody"), "body", name=root)
    for el in list(base_geoms) + list(children):
        body.append(copy.deepcopy(el))
    used: set[str] = set()
    for parent in list(root_el.iter("body")):          # only the colliders take part
        for geom in list(parent.findall("geom")):
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                parent.remove(geom)
                continue
            geom.attrib.pop("material", None)
            if geom.get("mesh"):
                used.add(geom.get("mesh"))
    for el in assets:
        if el.tag == "mesh" and el.get("name") in used and el.get("file") in mesh_files:
            mesh = copy.deepcopy(el)
            mesh.set("file", str(Path(mesh_files[el.get("file")]).resolve()))
            asset.append(mesh)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root_el, encoding="unicode"))
    model.geom_margin[:] = gap
    model.geom_gap[:] = 0.0
    data = mujoco.MjData(model)
    pairs: set[tuple[str, str]] = set()
    closed = model.qpos0.copy()
    for j in range(model.njnt):
        if int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                      int(mujoco.mjtJoint.mjJNT_SLIDE)):
            closed[int(model.jnt_qposadr[j])] = 0.0
    for qpos in (model.qpos0, closed):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                        int(model.geom_bodyid[g])) for g in (c.geom1, c.geom2))
            if b1 and b2 and b1 != b2 and c.dist < gap:
                pairs.add(tuple(sorted((b1, b2))))
    return sorted(pairs)


def _inertial(link: Link, T: Callable[[np.ndarray], np.ndarray], origin: np.ndarray,
              mass: float) -> ET.Element:
    """A part with no collider still needs a mass to move: its drawn box, uniformly solid."""
    box = _bounds(link.visuals) or (link.origin - 0.01, link.origin + 0.01)
    a, b = T(box[0]), T(box[1])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    e = np.maximum(hi - lo, 0.01)
    diag = mass / 12.0 * np.array([e[1] ** 2 + e[2] ** 2, e[0] ** 2 + e[2] ** 2,
                                   e[0] ** 2 + e[1] ** 2])
    return ET.Element("inertial", pos=_fmt((lo + hi) / 2 - origin), mass=f"{mass:.6g}",
                      diaginertia=_fmt(diag))


def _axis_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float) / (np.linalg.norm(axis) or 1.0)
    return np.array([math.cos(angle / 2), *(axis * math.sin(angle / 2))])


def write_files(built: Built, mesh_root: Path, texture_root: Path) -> list[Path]:
    """Copy / write a `Built`'s files under the scene's mesh and texture folders (skipping what
    is already there unchanged); returns the paths written."""
    written = []
    for rel, src in built.mesh_files.items():
        dest = Path(mesh_root) / rel
        if dest.is_file() and dest.stat().st_size == Path(src).stat().st_size:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        written.append(dest)
    for rel, data in built.texture_files.items():
        dest = Path(texture_root) / rel
        payload = data if isinstance(data, bytes) else Path(data).read_bytes()
        if dest.is_file() and dest.read_bytes() == payload:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, dest)
        written.append(dest)
    return written


def standalone_xml(built: Built, mesh_root: Path, texture_root: Path) -> str:
    """The asset on its own (base fixed to the world), for checking that MuJoCo takes it."""
    root = ET.Element("mujoco", model=built.handle)
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    asset = ET.SubElement(root, "asset")
    for el in built.assets:
        el = copy.deepcopy(el)
        if el.get("file"):
            base = mesh_root if el.tag == "mesh" else texture_root
            el.set("file", str(Path(base) / el.get("file")))
        asset.append(el)
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body", name=built.root)
    for el in built.base_geoms + built.children:
        body.append(copy.deepcopy(el))
    if built.excludes:
        contact = ET.SubElement(root, "contact")
        for a, b in built.excludes:
            ET.SubElement(contact, "exclude", body1=a, body2=b)
    return ET.tostring(root, encoding="unicode")


__all__ = ["ARTICULATED_DIR", "DEFAULT_SEED", "FACTORIES", "INFINIGEN_COMMIT", "MODAL_APP",
           "MODAL_FUNCTION", "PLINTH", "Asset", "AssetCache", "Built", "FakeRunner", "Link",
           "LocalRunner", "MeshPart", "ModalRunner", "Overrides", "Placement", "RunResult",
           "Runner", "Spec", "build", "clean", "door_spec", "fake_export", "fit_transform",
           "drop_sill", "frame_material", "generate", "is_fake", "load_asset", "main_bounds", "part_names",
           "rest_contacts", "runner_from_env", "spec_for_object", "standalone_xml", "tint_texture",
           "unpack_zip", "window_spec", "write_files", "zip_folder"]


if __name__ == "__main__":  # pragma: no cover — `python -m ... <export folder>` prints a summary
    a = clean(load_asset(sys.argv[1]))
    print(json.dumps({n: {"parent": lk.parent, "part": lk.part,
                          "joint": lk.joint.name if lk.joint else None}
                      for n, lk in a.links.items()}, indent=1))
