"""infinigen_worker.py — runs INSIDE an Infinigen environment and turns sized specs into MJCF.

This file is executed by the Infinigen env's own Python (3.11 with exactly ``bpy==4.2.0`` and
``infinigen[sim]`` at the pinned commit), never imported by LiteReality: it must not import
``litereality_agent``, and LiteReality must not import ``bpy``. `models.articulated.infinigen`
starts it as a subprocess (``$LR_INFINIGEN_PYTHON``) or ships it to the Modal function
(``deploy/modal/infinigen/app.py``), and reads back what it writes.

    <infinigen python> infinigen_worker.py JOBS.json OUT_DIR

``JOBS.json`` is ``{"jobs": [{"key", "factory", "size": [W, D, H], "seed", "options": {}}],
"image_res": 512}`` with sizes in metres in the object's own terms — W across its front, D front
to back, H up. For each job it writes ``OUT_DIR/<key>/`` holding Infinigen's own MJCF export (the
``<name>.xml``, ``metadata.json`` and ``assets/``) and ``result.json``; it also prints one
``RESULT {json}`` line per job as it finishes, so a caller can stream progress.

Why the factories are driven the way they are (feasibility study, 2026-10-21):

* **Sizing.** The stock ``create_asset`` ignores its arguments, so each factory is subclassed and
  its ``sample_parameters`` (or ``sample_params``) patched to the scanned size: the door through
  its constants object, the window and dishwasher through ``Width/Height/Depth``, the drawer chest
  through ``Size`` and its grid, the fridge through ``Main Fridge Dimensions``, the oven through
  ``Dimensions``, the cabinet through its shelf cells. The outer size still comes out 1-8 cm off
  (frames, knobs); the caller scales the residual away.
* **Colliders.** ``-exp mjcf -c`` runs CoACD on every part, and its preprocess voxel-remeshes even
  a 2 cm button into ~100k triangles and then takes minutes on it (one dishwasher: 12 min, another
  killed at 20). A part under ``SMALL_PART`` across, or with more than ``MAX_TRIANGLES``, gets
  one convex hull instead — total CoACD time 7 s, and the asset still works.
* **Offline.** Importing the asset library asks Blender for two online add-ons (rocks and
  fluids, which no sim factory uses); Infinigen's helper then syncs extensions.blender.org even
  with online installs off — 20 s per process, and parallel workers fight over one lock in
  ``~/.config/blender``. The worker answers those requests itself (`offline_addons`) and gives
  Blender a throw-away user folder.
* **MJCF, not URDF.** The stock URDF exporter crashes on the current trimesh
  (``'MassProperties' object does not support item assignment``).

Masses and joint dynamics are exported as Infinigen sets them (3-30x too heavy, joints tuned for
robots); the caller overrides both, so nothing here tries to fix them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

T_START = time.time()

WORKER_VERSION = 1
SMALL_PART = 0.15          # m across: a part smaller than this gets one convex hull, not CoACD
MAX_TRIANGLES = 60_000     # a part with more triangles than this, too
FACTORIES = ("door", "window", "cabinet", "drawer", "dishwasher", "refrigerator", "oven")


def _wrap_sample(cls, fn_name, update):
    """Subclass ``cls`` so that ``fn_name()`` returns the sampled dict updated by ``update``."""

    def patched(self):
        d = getattr(cls, fn_name)(self)
        update(self, d)
        return d

    return type(f"Sized{cls.__name__}", (cls,), {fn_name: patched})


def make_factory(job: dict):
    """(base class, sized factory instance, the parameters chosen) for one job."""
    from infinigen.assets.sim_objects.mapping import OBJECT_CLASS_MAP

    factory = job["factory"]
    W, D, H = (float(v) for v in job["size"])
    seed = int(job["seed"])
    opts = dict(job.get("options") or {})
    note: dict = {}
    base = OBJECT_CLASS_MAP[factory]

    if factory == "door":
        consts = types.SimpleNamespace(door_width=W, door_size=H,
                                       wall_thickness=float(opts.get("wall_thickness", 0.2)))
        fac = base(seed, constants=consts)
        bf = fac.base_factory
        # One leaf in a square frame: the random choice can be a double door (2 W wide) or a
        # dome, and RoomPlan's opening is a rectangle with one leaf in it.
        bf.door_frame_style = "full_frame_square"
        # The frame adds its width on every side: the LEAF is what is made smaller, so frame
        # and leaf together fill the opening the scan measured.
        bf.width = W - 2 * bf.door_frame_width
        bf.height = H - 2 * bf.door_frame_width
        if opts.get("orientation") in ("left", "right"):
            bf.door_orientation = opts["orientation"]
        if bf.handle_type in ("bar", "none"):
            # A push bar is a fire door's; "none" leaves nothing to take hold of.
            bf.handle_type, bf.handle_joint = "lever", "hinge"
            bf.handle_info_dict = {"handle_type": "lever"}
        note = {"door_class": type(bf).__name__, "handle_type": bf.handle_type,
                "door_frame_width": bf.door_frame_width, "leaf_depth": bf.depth,
                "orientation": bf.door_orientation}
        return base, fac, note

    if factory == "window":

        def upd(self, d):
            d["Width"], d["Height"] = W, H
            d["JointType"] = 1                  # sliding sashes
            d["PanelHAmount"] = int(opts.get("panels", 2))
            d["Shutter"] = False
            d["WithHandle"] = False
            note.update({k: d[k] for k in ("Width", "Height", "FrameWidth", "FrameThickness",
                                           "JointType", "PanelHAmount") if k in d})

        return base, _wrap_sample(base, "sample_parameters", upd)(seed), note

    if factory == "dishwasher":

        def upd(self, d):
            d["Depth"], d["Width"], d["Height"] = D, W, H
            d["DoorThickness"] = 0.06 * D
            note.update({k: d[k] for k in ("Depth", "Width", "Height")})

        return base, _wrap_sample(base, "sample_parameters", upd)(seed), note

    if factory == "refrigerator":

        def upd(self, d):
            d["Main Fridge Dimensions"] = (D, W, H)
            # One door and no freezer: the only configuration whose sub-parts follow an
            # overridden size (more doors put the handle on the hinge side).
            d["Has Top Freezer"] = False
            d["Has Bottom Freezer"] = False
            d["Top Freezer Height"] = 0
            d["Bottom Freezer 1 Height"] = 0
            d["Bottom Freezer 2 Height"] = 0
            d["Main Fridge Num Doors"] = 1
            d["Bottom Freezer Num Freezers"] = 0
            d["Handle Length"] = min(d["Handle Length"], 0.6 * H)
            d["Handle Z Offset"] = 0.0
            note.update({k: d[k] for k in ("Main Fridge Dimensions", "Main Fridge Num Doors")})

        return base, _wrap_sample(base, "sample_parameters", upd)(seed), note

    if factory == "oven":

        def upd(self, d):
            d["Dimensions"] = (D, W, H)
            # Both add to the height outside "Dimensions": a control box on top, a drawer below.
            d["Include Panel Box"] = False
            d["Include Lower Drawer"] = False
            d["Include Wire"] = False
            # Everything sampled FROM the dimensions has to follow them, or the door's window is
            # wider than the oven and the racks float above it.
            border = d["Major Border Thickness"]
            inner = (H - 2 * border) * 0.95
            racks = 1 if inner < 0.4 else 2
            d["Rack Count"] = racks
            d["Rack Offset"] = inner / (racks + 1)
            d["Rack Z Offset"] = -d["Rack Offset"] * ((racks - 1) / 2)
            d["Handle Length"] = min(d["Handle Length"], 0.8 * W)
            d["Handle Pos Offset"] = (0, 0, d["Handle Pos Offset"][2])
            if d["Window Width"] > 0:
                d["Window Width"] = 0.65 * W
            d["Window Height"] = 0.6 * H
            d["Drawer Handle Length"] = d["Handle Length"]
            note.update({"Dimensions": d["Dimensions"], "Rack Count": racks})

        return base, _wrap_sample(base, "sample_parameters", upd)(seed), note

    if factory == "drawer":
        rows = int(opts.get("rows", 3))
        cols = int(opts.get("cols", 2))

        def upd(self, d):
            d["Size"] = (D, W, H)
            d["Num Rows"] = rows
            d["Num Columns"] = cols
            d["Bottom Offset"] = float(opts.get("bottom", 0.1))
            note.update({k: d[k] for k in ("Size", "Num Rows", "Num Columns", "Thickness")
                         if k in d})

        return base, _wrap_sample(base, "sample_parameters", upd)(seed), note

    if factory == "cabinet":
        side = 0.018
        div = 0.018
        bottom = float(opts.get("bottom", 0.08))
        door_t = 0.02          # the doors stand in front of the shelf depth
        cells = max(1, int(opts.get("shelves", 2)))
        inner_h = H - bottom - (cells + 1) * div
        shelf_params = {
            "shelf_depth": D - door_t,
            "side_board_thickness": side,
            "division_board_thickness": div,
            "bottom_board_height": bottom,
            "shelf_cell_width": [W - 2 * side],
            "shelf_cell_height": [inner_h / cells] * cells,
        }

        def sample_params(self):
            self.shelf_params = dict(shelf_params)

        cls = type(f"Sized{base.__name__}", (base,), {"sample_params": sample_params})
        fac = cls(seed)
        if opts.get("frame_material") in ("white", "black_wood", "wood"):
            fac.mat_params = {"frame_material": opts["frame_material"]}
        note.update(shelf_params)
        note["frame_material"] = opts.get("frame_material")
        return base, fac, note

    raise KeyError(f"unknown factory {factory!r}: expected one of {', '.join(FACTORIES)}")


def offline_addons() -> None:
    """Never install a Blender add-on: report one that is missing, as Infinigen's own helper does
    for ``fail="warn"``, without its online sync. Must run before the asset library is imported
    (its modules bind the helper by name at import)."""
    import bpy
    import infinigen.core.init as ig_init

    def require_blender_addon(addon: str, fail: str = "fatal", allow_online=False):
        if f"bl_ext.blender_org.{addon}" in bpy.context.preferences.addons.keys():
            return True
        if fail == "warn":
            return False
        raise ValueError(f"Blender add-on {addon!r} is not installed (the worker installs none)")

    ig_init.require_blender_addon = require_blender_addon


def patch_collision() -> None:
    """One convex hull instead of CoACD for small or huge parts (see the module docstring)."""
    import numpy as np
    import trimesh
    from infinigen.tools import export as ig_export

    original = ig_export.coacd.run_coacd

    def run_coacd(mesh, **kw):
        v = np.asarray(mesh.vertices)
        small = float(np.linalg.norm(v.max(0) - v.min(0))) < SMALL_PART if len(v) else True
        if small or len(mesh.indices) > MAX_TRIANGLES:
            hull = trimesh.Trimesh(mesh.vertices, mesh.indices).convex_hull
            return [(hull.vertices, hull.faces)]
        return original(mesh, **kw)

    ig_export.coacd.run_coacd = run_coacd


def spawn(base, fac, name: str, export_dir: Path, image_res: int) -> tuple[Path, list, dict]:
    """`infinigen.core.sim.sim_factory.spawn_simready`, taking an instance (so a sized subclass
    can be used) and always exporting colliders (``visual_only=False``)."""
    from infinigen.core.sim import kinematic_compiler
    from infinigen.core.sim.exporters.factory import sim_exporter_factory
    from infinigen.core.util import blender as butil

    t: dict = {}
    t0 = time.time()
    obj = fac.spawn_asset(i=0)
    t["create"] = time.time() - t0
    t0 = time.time()
    blueprint = kinematic_compiler.compile(obj)
    butil.apply_modifiers(obj)
    t["compile"] = time.time() - t0
    blueprint["name"] = name
    dims = [round(float(x), 4) for x in obj.dimensions]
    t0 = time.time()
    export = sim_exporter_factory(exporter="mjcf", legacy=False)
    path, _meta = export(
        blend_obj=obj, sim_blueprint=blueprint, seed=fac.factory_seed,
        sample_joint_params_fn=base.sample_joint_parameters, export_dir=export_dir,
        image_res=image_res, options=base.__dict__.get("options"),
        extra_exclude=base.__dict__.get("extra_exclude"), visual_only=False)
    t["export"] = time.time() - t0
    return Path(path), dims, t


def run_job(job: dict, out_dir: Path, image_res: int) -> dict:
    import bpy

    started = time.time()
    folder = out_dir / job["key"]
    folder.mkdir(parents=True, exist_ok=True)
    rec = {"key": job["key"], "factory": job["factory"], "size": job["size"],
           "seed": job["seed"], "options": job.get("options") or {},
           "worker_version": WORKER_VERSION}
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)     # a clean scene per asset
        base, fac, note = make_factory(job)
        path, dims, timings = spawn(base, fac, job["factory"], folder / "export", image_res)
        rec.update(ok=True, mjcf=str(path.relative_to(folder)), blender_dims=dims,
                   timings={k: round(v, 3) for k, v in timings.items()}, params=note)
    except Exception as exc:  # noqa: BLE001 — reported per job; the next one still runs
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-3000:])
    rec["seconds"] = round(time.time() - started, 3)
    (folder / "result.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", type=Path, help="JSON: {jobs: [...], image_res}")
    ap.add_argument("out", type=Path, help="one folder per job is written here")
    args = ap.parse_args(argv)
    spec = json.loads(args.jobs.read_text())
    image_res = int(spec.get("image_res", 512))
    args.out.mkdir(parents=True, exist_ok=True)
    scratch = None
    if "BLENDER_USER_RESOURCES" not in os.environ:
        scratch = tempfile.mkdtemp(prefix="lr-blender-")
        os.environ["BLENDER_USER_RESOURCES"] = scratch
    try:
        import bpy  # noqa: F401 — imported once, before the first job is timed

        offline_addons()
        patch_collision()
        from infinigen.assets.sim_objects.mapping import OBJECT_CLASS_MAP  # noqa: F401

        ready = {"import_seconds": round(time.time() - T_START, 3)}
        print(f"READY {json.dumps(ready)}", flush=True)
        failed = 0
        for job in spec.get("jobs", []):
            rec = run_job(job, args.out, image_res)
            failed += not rec.get("ok")
            short = {k: v for k, v in rec.items() if k != "traceback"}
            print(f"RESULT {json.dumps(short, default=str)}", flush=True)
        return 1 if failed else 0
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
