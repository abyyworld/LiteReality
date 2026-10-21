"""Compile physics for every object a scan produced.

Called once each builder has FINISHED its own objects, and once more after every builder has
finished, where the recipe that built them is still the thing anyone would go and fix. A failure
here never fails the stage — an object without a physics sidecar is exactly as usable as it was
before this existed — but it is recorded per object so the gap is visible rather than silent.

ORDERING IS THE CALLER'S JOB, and it used to be wrong: the compile ran at the end of the
procedural branch over every glb in `reconstruct/`, while the TRELLIS branch was still writing
into the same folder. On tea_room `Dishwasher0.glb` landed after the sweep, so the dishwasher
shipped with no `physics.json`, and a TRELLIS-only run never compiled anything at all. So this
module now (a) compiles a NAMED set of objects, so a branch compiles only what it has finished,
(b) refuses a glb that is missing or still being written rather than compiling half a file, and
(c) can compile only what is missing or older than its glb, which makes a final sweep after all
branches cheap.
"""

from __future__ import annotations

import json
import struct
import threading
import traceback
from pathlib import Path

from .checks import check, write_report
from .properties import build_model, write_model
from .urdf import to_urdf

# Two branches finish at different moments and both record their rows in one report.
_REPORT_LOCK = threading.Lock()


def sidecar_path(glb: Path) -> Path:
    """Where `compile_object` writes this glb's physics."""
    glb = Path(glb)
    return glb.parent / "sim" / f"{glb.stem}.physics.json"


def sidecar_is_current(glb: Path) -> bool:
    """A physics sidecar exists and was compiled from this glb or a later one."""
    glb, physics = Path(glb), sidecar_path(glb)
    try:
        return physics.stat().st_mtime >= glb.stat().st_mtime
    except OSError:
        return False


def glb_problem(glb: Path) -> str | None:
    """Why this glb cannot be compiled YET, or None.

    The GLB header states the file's total length, so a file another process is still writing is
    recognisable as such instead of failing somewhere inside trimesh with an error about buffers.
    """
    glb = Path(glb)
    try:
        size = glb.stat().st_size
        with glb.open("rb") as stream:
            header = stream.read(12)
    except OSError:
        return "no glb"
    if len(header) < 12 or header[:4] != b"glTF":
        return f"not a complete GLB ({size} bytes, no glTF header)"
    declared = struct.unpack_from("<I", header, 8)[0]
    if size < declared:
        return f"incomplete GLB ({size} of {declared} bytes) — still being written?"
    return None


def compile_object(glb: Path, *, run_checks: bool = True, decompose: bool = True) -> dict:
    """One GLB -> physics json + URDF + MJCF (+ the gate). Never raises."""
    glb = Path(glb)
    problem = glb_problem(glb)
    if problem:
        return {"object": glb.stem, "status": "error", "error": problem}
    out = glb.parent / "sim"
    try:
        model = build_model(glb, out, decompose=decompose)
        write_model(model, out)
        to_urdf(model, out)
        row = {"object": model.name, "status": "ok", "links": len(model.links),
               "joints": len(model.joints), "mass_kg": model.total_mass,
               "colliders": sum(len(link.colliders) for link in model.links)}
        if run_checks:
            report = check(model, out)
            write_report(report, out)
            row["check"] = "pass" if report.passed else "fail"
            row["findings"] = report.findings
        return row
    except Exception as exc:                                 # noqa: BLE001 — never fail the stage
        return {"object": glb.stem, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3)}


def object_glbs(recon_dir: Path, names=None) -> list[Path]:
    """The glb each object lives in, in both shapes it comes in.

    A recipe-built object lives in its own directory next to the `object.py` that made it; a
    generative one (TRELLIS chairs, sofas) is a bare glb at the top level. Only the first kind can
    carry articulation, which is why the articulation work only ever looked there — but a chair is
    still a rigid body that a scene has to give a mass and a collider, and skipping it leaves the
    room export inventing both for exactly the objects most likely to be knocked over.

    With `names`, only those objects — each at `<name>/<name>.glb`, else `<name>.glb` — and a name
    with neither is simply not returned (its builder did not produce it).
    """
    recon_dir = Path(recon_dir)
    if names is None:
        return sorted(recon_dir.glob("*/*.glb")) + sorted(recon_dir.glob("*.glb"))
    found = []
    for name in sorted(set(names)):
        for candidate in (recon_dir / name / f"{name}.glb", recon_dir / f"{name}.glb"):
            if candidate.is_file():
                found.append(candidate)
                break
    return found


def compile_directory(recon_dir: Path, *, run_checks: bool = True, names=None,
                      only_stale: bool = False) -> dict:
    """Compile the objects under `reconstruct/` and record them in `sim_report.json`.

    `names` limits it to those objects (see `object_glbs`); `only_stale` skips every object whose
    sidecar is already newer than its glb. The report is MERGED by object, so a branch that
    compiles its own objects does not erase the rows another branch wrote.
    """
    recon_dir = Path(recon_dir)
    glbs = object_glbs(recon_dir, names)
    todo = [g for g in glbs if not (only_stale and sidecar_is_current(g))]
    rows = [compile_object(glb, run_checks=run_checks) for glb in todo]
    summary = {"objects": len(rows),
               "current": len(glbs) - len(todo),
               "ok": sum(r["status"] == "ok" for r in rows),
               "passed": sum(r.get("check") == "pass" for r in rows),
               "failed": sum(r.get("check") == "fail" for r in rows),
               "errors": sum(r["status"] == "error" for r in rows),
               "rows": rows}
    _merge_report(recon_dir / "sim_report.json", rows,
                  {g.stem for g in object_glbs(recon_dir, None)})
    return summary


def missing_sidecars(recon_dir: Path, names) -> list[str]:
    """Objects that have a glb but no physics compiled from it (or only from an older glb)."""
    return [g.stem for g in object_glbs(recon_dir, names) if not sidecar_is_current(g)]


def _merge_report(path: Path, rows: list[dict], present: set[str]) -> None:
    """Fold `rows` into the report, dropping rows for objects whose glb is gone."""
    with _REPORT_LOCK:
        previous: list[dict] = []
        try:
            previous = list(json.loads(path.read_text(encoding="utf-8")).get("rows") or [])
        except (OSError, ValueError, AttributeError):
            previous = []
        by_object = {r.get("object"): r for r in previous if r.get("object") in present}
        by_object.update({r.get("object"): r for r in rows})
        merged = [by_object[k] for k in sorted(by_object, key=str)]
        report = {"objects": len(merged),
                  "ok": sum(r.get("status") == "ok" for r in merged),
                  "passed": sum(r.get("check") == "pass" for r in merged),
                  "failed": sum(r.get("check") == "fail" for r in merged),
                  "errors": sum(r.get("status") == "error" for r in merged),
                  "rows": merged}
        if path.parent.is_dir():
            path.write_text(json.dumps(report, indent=2) + "\n")
