"""report.py — one report per room and stage, and the table that compares the stages.

A room reaches the call in stages — ``quick`` (RoomPlan boxes as templates, seconds), ``objects``
(TRELLIS meshes over the same colliders, minutes), ``full`` (the pipeline's own export, hours) —
and the paper's question is what each stage buys for what it costs. So every stage of a room gets
the same audit (`physics.audit_physics`, and with ``render`` `fidelity.evaluate_fidelity` against
the scan's own photos), plus whatever its build recorded about its own timing:

* quick — ``quick.json``'s ``seconds`` (parse, layout, colours, textures, total);
* objects — ``objects.json``: objects generated vs left as templates and the fitting time (the
  generation wall time is not recorded by the stage);
* full — ``export_report.json`` beside the scene (loads clean, colliders, overlaps) and, for a run
  folder, the pipeline's own per-stage durations in ``.litereality/pipeline.json``.

Every target is compared with ITS OWN capture: the scan given with it, else the one its build
recorded (`infer_scan`). A scan given for a scene whose build recorded another is used — the
caller said so — but the report says so too (``scan_mismatch``).

Output, under ``out_dir``::

    <name>/report.json        everything, machine-readable (strict JSON: no NaN or Infinity)
    <name>/report.md          the same for people
    <name>/contact_sheet.jpg  photo | render | regions, with --render (and _holdout.jpg)
    comparison.json / .md / .tex   the paper's tables, in markdown and LaTeX: physics and cost
                              one row per stage; visual fidelity one row per stage AND split
                              (held-out k-fold, as built), so the two are never mixed unlabelled;
                              and (json, md) the paired per-view differences between the stages
                              of a room with bootstrap intervals

A stage whose fidelity cannot be measured (no frames, no RoomPlan export, no browser) still gets
its physics report and its row; the report says why fidelity is missing.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from litereality_agent.pipeline.evaluate import fidelity, physics
from litereality_agent.pipeline.evaluate import holdout as held

SCENE_NAME = "scene.xml"
# Export folder -> stage, in the order a room goes through them.
STAGE_FOLDERS = {"quick": "quick", "objects": "objects", "mujoco_seed": "full", "mujoco": "full"}
STAGE_ORDER = ("demo", "quick", "objects", "full")
REPORT_JSON, REPORT_MD = "report.json", "report.md"
COMPARISON = "comparison"


# ── finding what to evaluate ─────────────────────────────────────────────────────────────────────


def stage_of(scene: Path) -> str:
    """The stage a scene file is, by the folder it was exported into; "demo" for the room call's
    built-in room; otherwise "scene"."""
    scene = Path(scene)
    folder = scene.parent.name
    if folder in STAGE_FOLDERS:
        return STAGE_FOLDERS[folder]
    try:
        head = scene.read_text(encoding="utf-8", errors="replace")[:400]
    except OSError:
        head = ""
    return "demo" if 'model="roomcall_demo"' in head else "scene"


def find_scenes(target: Path | str) -> list[tuple[str, Path]]:
    """(name, scene.xml) for ``target``: a scene file, a folder holding ``scene.xml``, or a run
    folder (``run/<scan>``), which gives every stage it has — quick/, objects/, mujoco_seed/ (or
    mujoco/) — in stage order."""
    target = Path(target).expanduser()
    if target.is_file():
        return [(_name(target), target)]
    if not target.is_dir():
        raise ValueError(f"no such scene or run folder: {target}")
    if (target / SCENE_NAME).is_file():
        return [(_name(target / SCENE_NAME), target / SCENE_NAME)]
    found, seen = [], set()
    for folder, stage in STAGE_FOLDERS.items():
        scene = target / folder / SCENE_NAME
        if scene.is_file() and stage not in seen:
            seen.add(stage)
            found.append((stage, scene))
    if not found:
        raise ValueError(f"{target}: no scene.xml, and none of "
                         f"{', '.join(f + '/scene.xml' for f in STAGE_FOLDERS)}")
    return found


def _name(scene: Path) -> str:
    stage = stage_of(scene)
    return stage if stage != "scene" else (scene.parent.name or "scene")


def infer_scan(scene: Path | str) -> Path | None:
    """The capture a scene was built from, when its build recorded it: the quick room's
    ``quick.json`` (also copied into the objects stage's folder) names the room.usdz; a full
    export in ``run/<scan>/mujoco_seed`` finds it through its sibling ``quick/``, or through the
    run's own ``scene.json`` (the pipeline's scene package, which records its capture)."""
    scene = Path(scene)
    for path in (scene.parent / "quick.json", scene.parent.parent / "quick" / "quick.json"):
        scan = held.recorded_scan(_read_json(path))
        if scan is not None:
            return scan
    for package in (scene.parent / "scene.json", scene.parent.parent / "scene.json"):
        if not package.is_file():
            continue
        try:
            from litereality_agent.room_ops.manifest import read

            capture = read(package).path("capture")
        except Exception:  # noqa: BLE001 — a malformed package just records nothing usable
            continue
        if capture is not None and capture.is_dir():
            return capture
    return None


def room_label(scene: Path | str, scan: Path | None) -> str:
    """What a table calls the room: its capture's name, else its run folder's (``run/<scan>/
    quick/scene.xml`` -> ``<scan>``), else the scene's folder."""
    if scan is not None:
        return scan.name
    scene = Path(scene)
    folder = scene.parent
    if folder.name in STAGE_FOLDERS and folder.parent.name:
        return folder.parent.name
    quick = _read_json(folder / "quick.json") or {}
    return str(quick.get("label") or folder.name or "room")


# ── what each stage recorded about itself ────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def stage_timings(scene: Path, stage: str) -> dict[str, Any]:
    """Build time and what the stage says about itself (see the module docstring)."""
    folder = Path(scene).parent
    out: dict[str, Any] = {}
    quick = _read_json(folder / "quick.json")
    if quick and stage != "full":
        # Also a quick room written somewhere else (an ablation built with other options).
        seconds = quick.get("seconds") or {}
        out["quick_build_s"] = seconds.get("total")
        out["quick_parts_s"] = {k: v for k, v in seconds.items() if k != "total"}
        if stage != "objects":
            out["build_s"] = seconds.get("total")
    objects = _read_json(folder / "objects.json") if stage == "objects" else None
    if objects:
        installed = objects.get("objects") or {}
        out["objects_generated"] = len(installed)
        out["objects_template"] = len(objects.get("template") or [])
        out["objects_passes"] = sorted({str(v.get("pass")) for v in installed.values()})
        fit = [float(v.get("fit_seconds") or 0.0) for v in installed.values()]
        out["objects_fit_s"] = round(sum(fit), 3)
        out["versions_written"] = objects.get("versions_written")
        out["note"] = "generation wall time is not recorded by the objects stage"
    export = _read_json(folder / "export_report.json") if stage == "full" else None
    if export:
        out["export"] = {"loads_clean": export.get("loads_clean"),
                         "colliders": export.get("colliders"),
                         "initial_overlaps": len(export.get("initial_overlaps") or []),
                         "free": export.get("free"), "articulated": export.get("articulated")}
    if stage == "full":
        state = _read_json(folder.parent / ".litereality" / "pipeline.json")
        stages = (state or {}).get("stages") or {}
        durations = {k: round(float(v.get("duration_seconds") or 0.0), 1)
                     for k, v in stages.items() if isinstance(v, dict)}
        if durations:
            out["pipeline_stage_s"] = durations
            out["build_s"] = round(sum(durations.values()), 1)
    return out


# ── evaluating ───────────────────────────────────────────────────────────────────────────────────


# What a failed fidelity measurement may raise without that costing the stage its physics report:
# no frames / no RoomPlan export (ValueError, FileNotFoundError), no browser or a page that would
# not start (RendererUnavailable, RuntimeError), and whatever the browser driver throws.
FIDELITY_ERRORS: tuple[type[BaseException], ...] = (ValueError, OSError, RuntimeError)


def evaluate_scene(scene: Path | str, out_dir: Path | str, *, name: str | None = None,
                   scan: Path | str | None = None, render: bool = False,
                   renderer: fidelity.Renderer | None = None, size: tuple[int, int] = fidelity.SIZE,
                   max_views: int | None = None, physics_options: dict | None = None,
                   holdout: int = 0, holdout_split: str = held.DEFAULT_SPLIT) -> dict:
    """Audit one scene and write ``<out_dir>/<name>/report.{json,md}`` (+ the contact sheets with
    ``render``). Returns the report. Fidelity needs the scan (given, or `infer_scan`); without one,
    or when it cannot be measured, the report says why rather than losing the physics it did."""
    scene = Path(scene).expanduser().resolve()
    stage = stage_of(scene)
    name = name or _name(scene)
    folder = Path(out_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    recorded = infer_scan(scene)
    scan_dir = Path(scan).expanduser().resolve() if scan else recorded
    report: dict[str, Any] = {
        "name": name, "stage": stage, "scene": str(scene), "room": room_label(scene, scan_dir),
        "scan": str(scan_dir) if scan_dir else None,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "timings": stage_timings(scene, stage),
        "physics": physics.audit_physics(scene, **(physics_options or {})),
        "fidelity": None,
    }
    if scan_dir is not None and recorded is not None and recorded.resolve() != scan_dir:
        report["scan_mismatch"] = (f"compared with {scan_dir}, but the scene's build recorded "
                                   f"{recorded}")
    if render:
        if scan_dir is None:
            report["fidelity_skipped"] = ("no scan to compare with: pass --scan (the scene's "
                                          "build did not record one)")
        else:
            try:
                report["fidelity"] = fidelity.evaluate_fidelity(
                    scene, scan_dir, renderer=renderer, size=size, max_views=max_views,
                    out_dir=folder, title=f"{scan_dir.name} · {name.split('/')[-1]}",
                    holdout=holdout,
                    holdout_split=holdout_split)
            except FIDELITY_ERRORS as exc:
                report["fidelity_skipped"] = f"fidelity failed: {type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 — Playwright's errors are plain Exceptions
                if type(exc).__module__.split(".")[0] != "playwright":
                    raise
                report["fidelity_skipped"] = f"rendering failed: {type(exc).__name__}: {exc}"
    else:
        report["fidelity_skipped"] = "not rendered (pass --render)"
    write_json(folder / REPORT_JSON, report)
    (folder / REPORT_MD).write_text(stage_markdown(report), encoding="utf-8")
    return report


Target = Path | str | tuple[Path | str, Path | str | None]


def evaluate(targets: Sequence[Target] | Path | str, out_dir: Path | str, *,
             scan: Path | str | None = None, render: bool = False,
             renderer: fidelity.Renderer | None = None, size: tuple[int, int] = fidelity.SIZE,
             max_views: int | None = None, physics_options: dict | None = None,
             holdout: int = 0, holdout_split: str = held.DEFAULT_SPLIT, log=print) -> dict:
    """Every scene ``targets`` name (see `find_scenes`), each with `evaluate_scene`, then the
    comparison across them in ``out_dir/comparison.{json,md,tex}``.

    A target is a path, or a (path, scan) pair that names that target's own capture. ``scan``
    applies to every plain path (a caller that names one capture for all); left out, each scene is
    compared with the capture its own build recorded — never with another target's."""
    if isinstance(targets, (str, Path)):
        targets = [targets]
    scenes: list[tuple[str, Path, Path | str | None]] = []
    for target in targets:
        path, own = target if isinstance(target, tuple) else (target, scan)
        scenes += [(name, found, own) for name, found in find_scenes(path)]
    rooms = {room_label(found, Path(own) if own else infer_scan(found))
             for _n, found, own in scenes}
    names: dict[str, int] = {}
    reports = []
    renderer = renderer if renderer is not None or not render else fidelity.default_renderer()
    for name, scene, own in scenes:
        if len(rooms) > 1:
            # Several rooms: "tea_room/quick", not two folders both called "quick".
            name = f"{room_label(scene, Path(own) if own else infer_scan(scene))}/{name}"
        names[name] = names.get(name, 0) + 1
        unique = name if names[name] == 1 else f"{name}_{names[name]}"
        if log:
            log(f"evaluate  {unique}: {scene}")
        reports.append(evaluate_scene(scene, out_dir, name=unique, scan=own, render=render,
                                      renderer=renderer, size=size, max_views=max_views,
                                      physics_options=physics_options, holdout=holdout,
                                      holdout_split=holdout_split))
    return {"reports": reports, **write_comparison(reports, out_dir)}


def write_comparison(reports: list[dict], out_dir: Path | str) -> dict:
    """``out_dir/comparison.{json,md,tex}`` from stage reports (as `evaluate_scene` returns them,
    or as read back from their report.json): the physics rows, the fidelity rows and the paired
    differences between stages of a room."""
    ordered = sorted(reports, key=_stage_key)
    rows = [headline(r) for r in ordered]
    fid_rows = [row for r in ordered for row in fidelity_rows(r)]
    pairs = paired_differences(ordered)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{COMPARISON}.json", {"physics": rows, "fidelity": fid_rows,
                                            "paired": pairs})
    (out / f"{COMPARISON}.md").write_text(comparison_markdown(rows, fid_rows, pairs),
                                          encoding="utf-8")
    (out / f"{COMPARISON}.tex").write_text(comparison_latex(rows, fid_rows), encoding="utf-8")
    return {"comparison": rows, "fidelity": fid_rows, "paired": pairs}


def _stage_key(report: dict) -> tuple[str, int, str]:
    stage = report["stage"]
    order = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
    return (report.get("room") or "", order, report["name"])


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def finite(value: Any) -> Any:
    """``value`` with every non-finite float (NaN, +-inf) replaced by None, recursively: what a
    strict JSON parser (JSON.parse, jq) can read. The audit itself should never produce one; this
    is the guard that keeps a future one from making report.json unreadable."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if hasattr(value, "tolist"):
        return finite(value.tolist())
    return value


def write_json(path: Path, data: Any) -> None:
    """Strict JSON (``allow_nan=False``, after `finite`)."""
    text = json.dumps(finite(data), indent=1, default=_json_default, allow_nan=False)
    Path(path).write_text(text, encoding="utf-8")


# ── the numbers that go in the paper ─────────────────────────────────────────────────────────────


def _mean(summary: dict | None, key: str):
    entry = (summary or {}).get(key)
    return entry["mean"] if isinstance(entry, dict) else None


FIDELITY_KEYS = ("psnr", "ssim", "psnr_static", "ssim_static", "psnr_objects", "ssim_objects",
                 "psnr_gain", "psnr_baseline", "coverage")


def headline(report: dict) -> dict:
    """One flat row of a report: what the physics and cost table shows (plus, for a quick look,
    the fidelity of its primary split — held out when there is one — with that split named)."""
    p, t = report["physics"], report.get("timings") or {}
    i, c, m = p["interactability"], p["cost"], p["masses"]
    conc, cap = c.get("concurrency") or {}, c.get("capacity") or {}
    step, rtf = c.get("ms_per_step_spread") or {}, c.get("realtime_factor_spread") or {}
    primary = (fidelity_rows(report) or [None])[0]
    row = {
        "room": report.get("room") or (Path(report["scan"]).name if report.get("scan") else None),
        "name": report["name"], "stage": report["stage"],
        "scan": Path(report["scan"]).name if report.get("scan") else None,
        "build_s": t.get("build_s"),
        "movable": i["movable"], "interactable": i["interactable"], "tested": i["tested"],
        "interactable_rate": i["interactable_rate"],
        "mass_outliers": len(m["outliers"]), "mass_checked": m["checked"],
        "overlaps_authored": len(p["overlaps"]["authored"]),
        "overlaps_settled": len(p["overlaps"]["settled"]),
        "settle_shift_mm": p["settle_shift"]["max_mm"],
        "drift_mm": p["stability"]["max_drift_mm"],
        "resets": p["stability"]["resets"], "respawns": p["stability"]["respawns"],
        "ms_per_step": c["ms_per_step"], "ms_per_step_min": step.get("min"),
        "ms_per_step_max": step.get("max"),
        "realtime_factor": c["realtime_factor"], "realtime_factor_min": rtf.get("min"),
        "realtime_factor_max": rtf.get("max"),
        "rooms_realtime": c.get("rooms_realtime"), "rooms_at_limit": cap.get("at_limit"),
        "rooms_busy": cap.get("machine_busy"),
        "rooms_flat_out": conc.get("rooms_flat_out"),
        "split": primary["split"] if primary else None,
        "views": primary["views"] if primary else None,
    }
    for key in FIDELITY_KEYS:
        row[key] = primary[key] if primary else None
    return row


def fidelity_rows(report: dict) -> list[dict]:
    """The visual-fidelity rows of one report, one per split, the held-out one first: ``held-out
    (k-fold ...)`` when the k-fold ran, then ``as built`` over every view — labelled with how many
    of those views the build itself read, since those are training views."""
    fid = report.get("fidelity")
    if not fid:
        return []
    base = {"room": report.get("room"), "name": report["name"], "stage": report["stage"]}
    rows = []
    hold = fid.get("holdout") or {}
    if hold.get("summary"):
        rows.append({**base, "split": f"held-out ({hold['k']}-fold {hold['split']})",
                     "views": hold["views"], "views_seen": 0,
                     **{k: _mean(hold["summary"], k) for k in FIDELITY_KEYS}})
    seen = fid.get("views_seen") or {}
    if seen.get("unknown"):
        label = "as built (build's frames not recorded)"
    elif seen.get("seen") == fid["views"]:
        label = "as built (all views seen by build)"
    elif seen.get("seen"):
        label = f"as built ({seen['seen']} of {fid['views']} seen by build)"
    else:
        label = "as built (no view seen by build)"
    rows.append({**base, "split": label, "views": fid["views"],
                 "views_seen": seen.get("seen") if not seen.get("unknown") else None,
                 **{k: _mean(fid["summary"], k) for k in FIDELITY_KEYS}})
    return rows


def _f(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}{suffix}" if isinstance(value, float) else f"{value}{suffix}"
    return str(value)


def _range(value, lo, hi, digits: int) -> str:
    """``median (min-max)``, or the median alone when there was one sample."""
    if value is None:
        return "—"
    if lo is None or hi is None or lo == hi:
        return _f(value, digits)
    return f"{_f(value, digits)} ({_f(lo, digits)}-{_f(hi, digits)})"


# (header, row -> cell) — shared by the markdown and LaTeX tables so they cannot disagree.
_PHYSICS_COLUMNS = (
    ("Room", lambda r: r["room"] or r["scan"] or "—"),
    ("Stage", lambda r: r["name"].split("/")[-1]),
    ("Build (s)", lambda r: _f(r["build_s"], 1)),
    ("Interactable", lambda r: f"{r['interactable']}/{r['tested']}"),
    ("Mass outliers", lambda r: f"{r['mass_outliers']}/{r['mass_checked']}"),
    ("Overlaps >5 mm (auth./settled)",
     lambda r: f"{r['overlaps_authored']}/{r['overlaps_settled']}"),
    ("Drift 10 s (mm)", lambda r: _f(r["drift_mm"], 2)),
    ("ms/step", lambda r: _range(r["ms_per_step"], r["ms_per_step_min"], r["ms_per_step_max"], 3)),
    ("RTF", lambda r: _range(r["realtime_factor"], r["realtime_factor_min"],
                             r["realtime_factor_max"], 0)),
    ("Rooms RT", lambda r: _f(r["rooms_realtime"], 0) + ("+" if r.get("rooms_at_limit") else "")
     + ("*" if r.get("rooms_busy") else "")),
)
_FIDELITY_COLUMNS = (
    ("Room", lambda r: r["room"] or "—"),
    ("Stage", lambda r: r["name"].split("/")[-1]),
    ("Views", lambda r: r["split"]),
    ("n", lambda r: _f(r["views"], 0)),
    ("PSNR", lambda r: _f(r["psnr"], 2)),
    ("SSIM", lambda r: _f(r["ssim"], 3)),
    ("PSNR walls/floor", lambda r: _f(r["psnr_static"], 2)),
    ("SSIM walls/floor", lambda r: _f(r["ssim_static"], 3)),
    ("PSNR objects", lambda r: _f(r["psnr_objects"], 2)),
    ("SSIM objects", lambda r: _f(r["ssim_objects"], 3)),
    ("PSNR exp.-matched", lambda r: _f(r["psnr_gain"], 2)),
    ("PSNR flat baseline", lambda r: _f(r["psnr_baseline"], 2)),
)

PHYSICS_CAPTION = (
    "Interactable: grabbable, moved by the standard pull, and at rest after release. Masses "
    "against per-category plausible ranges. ms/step and RTF (real-time factor of one room on one "
    "thread): median (min-max) of 5 runs after a warm-up. Rooms RT: the most rooms one room-call "
    "server process kept in real time, measured through the server's own loop ('+': the "
    "server's 16-room cap; '*': measured while other processes kept the machine busy, not "
    "representative). Timing columns are machine-dependent (CPU and load in each report.md).")
FIDELITY_CAPTION = (
    "PSNR in dB, mean over the capture's views at 320x240; SSIM on luma. Held-out: each photo is "
    "scored against a quick room rebuilt without it (k-fold over the capture's frames). As "
    "built: every view against the room as built, including the views the build read (its "
    "walls are made of those photos): training views, not a generalisation result. Objects: "
    "pixels whose LiDAR point lies in a RoomPlan box; walls/floor: the rest, less doors and "
    "windows. Definitions in each report.md.")


def _table(columns, rows: list[dict]) -> list[str]:
    head = [c[0] for c in columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return lines + ["| " + " | ".join(fn(r) for _h, fn in columns) + " |" for r in rows]


PAIRED_KEYS = ("psnr", "ssim", "psnr_static", "ssim_static", "psnr_objects", "ssim_objects")
BOOTSTRAP = 10_000            # resamples of the views for a paired difference's 95 % interval


def _views_of(report: dict) -> tuple[str, dict[int, dict]] | None:
    """(split label, frame -> row) of a report's primary fidelity split: held out if it ran."""
    fid = report.get("fidelity") or {}
    hold = fid.get("holdout") or {}
    if hold.get("per_view"):
        return f"held-out ({hold['k']}-fold {hold['split']})", {r["frame"]: r
                                                                for r in hold["per_view"]}
    if fid.get("per_view"):
        return "as built", {r["frame"]: r for r in fid["per_view"]}
    return None


def paired_difference(a: list[float], b: list[float], *, resamples: int = BOOTSTRAP,
                      seed: int = 0) -> dict:
    """Mean of b - a over paired views with a percentile-bootstrap 95 % interval (views
    resampled with replacement, a fixed seed so the table is reproducible), and in how many
    views b is better."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "better": int((d > 0).sum()), "n": int(len(d))}


def paired_differences(reports: list[dict]) -> list[dict]:
    """Every pair of stages of the same room, on the views both scored, per metric: the paired
    difference (later stage minus earlier) with its bootstrap interval. Two stages compared on
    the same photos differ far less from photo to photo than either does from the photos, so this
    is what can say whether a 0.5 dB step is real. A pair is compared on the split both have
    (held out when both ran it), and the row says which."""
    out = []
    by_room: dict[str, list[dict]] = {}
    for rep in reports:
        if _views_of(rep) is not None:
            by_room.setdefault(rep.get("room") or "", []).append(rep)
    for room, group in by_room.items():
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                (split_a, a), (split_b, b) = _views_of(first), _views_of(second)
                if split_a != split_b:
                    split_a = split_b = "as built"
                    a = {r["frame"]: r for r in first["fidelity"]["per_view"]}
                    b = {r["frame"]: r for r in second["fidelity"]["per_view"]}
                row: dict[str, Any] = {"room": room, "a": first["name"], "b": second["name"],
                                       "split": split_a}
                for key in PAIRED_KEYS:
                    frames = [f for f in sorted(set(a) & set(b))
                              if a[f].get(key) is not None and b[f].get(key) is not None]
                    row[key] = (paired_difference([a[f][key] for f in frames],
                                                  [b[f][key] for f in frames])
                                if len(frames) >= 2 else None)
                out.append(row)
    return out


def _delta(entry: dict | None, digits: int) -> str:
    if not entry:
        return "—"
    lo, hi = entry["ci95"]
    return (f"{entry['mean']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] "
            f"{entry['better']}/{entry['n']}")


_PAIRED_COLUMNS = (
    ("Room", lambda r: r["room"] or "—"),
    ("B - A", lambda r: f"{r['b'].split('/')[-1]} - {r['a'].split('/')[-1]}"),
    ("Views", lambda r: r["split"]),
    ("PSNR", lambda r: _delta(r["psnr"], 2)),
    ("SSIM", lambda r: _delta(r["ssim"], 3)),
    ("PSNR walls/floor", lambda r: _delta(r["psnr_static"], 2)),
    ("SSIM walls/floor", lambda r: _delta(r["ssim_static"], 3)),
    ("PSNR objects", lambda r: _delta(r["psnr_objects"], 2)),
    ("SSIM objects", lambda r: _delta(r["ssim_objects"], 3)),
)
PAIRED_CAPTION = (
    "Paired differences over the views both stages scored: mean of B - A, its 95 % percentile-"
    "bootstrap interval over views (10,000 resamples, seed 0), and in how many views B is better. "
    "An interval that excludes 0 is a difference the capture's views agree on.")


def comparison_markdown(rows: list[dict], fid_rows: list[dict] | None = None,
                        pairs: list[dict] | None = None) -> str:
    """The comparison as markdown: the physics and cost table, the fidelity table (one row per
    stage and split) and the paired differences between stages, each with what it means."""
    lines = ["### Physics and cost", ""] + _table(_PHYSICS_COLUMNS, rows)
    lines += ["", PHYSICS_CAPTION]
    if fid_rows:
        lines += ["", "### Visual fidelity", ""] + _table(_FIDELITY_COLUMNS, fid_rows)
        lines += ["", FIDELITY_CAPTION]
    if pairs:
        lines += ["", "### Paired differences between stages", ""]
        lines += _table(_PAIRED_COLUMNS, pairs) + ["", PAIRED_CAPTION]
    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    return (text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#").replace("—", "--").replace(">", "$>$")
            .replace("+", "$^{+}$"))


def _latex_table(columns, rows: list[dict], caption: str) -> list[str]:
    cols = "ll" + "r" * (len(columns) - 2)
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(_latex_escape(h) for h, _fn in columns) + r" \\", r"\midrule"]
    lines += [" & ".join(_latex_escape(fn(r)) for _h, fn in columns) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{_latex_escape(caption)}}}",
              r"\end{table}"]
    return lines


def comparison_latex(rows: list[dict], fid_rows: list[dict] | None = None) -> str:
    """The same tables as booktabs ``table`` floats with their captions (needs
    ``\\usepackage{booktabs}``)."""
    lines = _latex_table(_PHYSICS_COLUMNS, rows, PHYSICS_CAPTION)
    if fid_rows:
        lines += [""] + _latex_table(_FIDELITY_COLUMNS, fid_rows, FIDELITY_CAPTION)
    return "\n".join(lines) + "\n"


def _summary_table(summaries: list[tuple[str, dict | None]]) -> list[str]:
    """Metric rows, one column per (label, summary): mean (n views)."""
    present = [(label, s) for label, s in summaries if s]
    if not present:
        return []
    keys = [k for k in present[0][1] if isinstance(present[0][1].get(k), dict)]
    lines = ["| metric | " + " | ".join(label for label, _s in present) + " |",
             "|---|" + "---|" * len(present)]
    for key in keys:
        cells = []
        for _label, s in present:
            entry = s.get(key)
            cells.append(f"{entry['mean']:.3f} ({entry['n']})" if isinstance(entry, dict) else "—")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return lines


def stage_markdown(report: dict) -> str:
    """One stage of one room, for people."""
    lines = [f"# Evaluation: {report.get('room') or 'room'} · {report['name'].split('/')[-1]}",
             "",
             f"- scene: `{report['scene']}`",
             f"- scan: `{report['scan'] or '—'}`",
             f"- stage: {report['stage']} · evaluated {report['created']}"]
    if report.get("scan_mismatch"):
        lines.append(f"- **scan mismatch**: {report['scan_mismatch']}")
    t = report.get("timings") or {}
    if t.get("build_s") is not None:
        lines.append(f"- build time: {t['build_s']} s")
    if t.get("quick_parts_s"):
        lines.append("- quick room parts: " + ", ".join(f"{k} {v} s"
                                                       for k, v in t["quick_parts_s"].items()))
    if t.get("objects_generated") is not None:
        lines.append(f"- objects: {t['objects_generated']} generated "
                     f"({', '.join(t.get('objects_passes') or [])}), {t['objects_template']} "
                     f"still templates, fitting {t['objects_fit_s']} s")
    if t.get("pipeline_stage_s"):
        lines.append("- pipeline stages: " + ", ".join(f"{k} {v} s"
                                                       for k, v in t["pipeline_stage_s"].items()))
    p = report["physics"]
    mach = p["machine"]
    load = mach.get("load_average")
    lines += [f"- machine: {mach['cpu']}, {mach['cores']} threads "
              f"({mach.get('usable_cpus', mach['cores'])} usable), MuJoCo {mach['mujoco']}"
              + (f", load average {load[0]:.1f} while measured" if load else ""), ""]
    lines.append(physics.markdown(p))
    fid = report.get("fidelity")
    lines += ["### Visual fidelity", ""]
    if fid:
        seen = fid.get("views_seen") or {}
        masks = fid.get("object_mask") or {}
        lines += [f"{fid['views']} views at {fid['size'][0]}x{fid['size'][1]} "
                  f"(rendered at {fid['render_scale']}x, {fid['renderer']} "
                  f"{fid.get('renderer_version') or ''}); object pixels from LiDAR depth in "
                  f"{masks.get('depth', 0)} views, box silhouettes in {masks.get('box', 0)}.", ""]
        if seen.get("unknown"):
            lines.append("The build recorded no frames it read: every view may be a training "
                         "view.")
        else:
            lines.append(f"The build read {seen.get('seen', 0)} of these {fid['views']} views "
                         "(its walls, floor, colours or crops come from them): those are "
                         "training views.")
        for note in (fid.get("sources") or {}).get("notes") or ():
            lines.append(f"- {note}")
        hold = fid.get("holdout") or {}
        columns: list[tuple[str, dict | None]] = []
        if hold.get("summary"):
            columns.append((f"held out ({hold['k']}-fold {hold['split']})", hold["summary"]))
            columns.append(("as built, same views", hold.get("as_built_same_views")))
        columns.append(("as built, all views", fid["summary"]))
        splits = fid.get("splits") or {}
        columns.append(("as built, seen views", splits.get("seen")))
        columns.append(("as built, unseen views", splits.get("unseen")))
        lines += [""] + _summary_table(columns)
        if hold.get("summary"):
            gap = hold.get("gap") or {}
            lines += ["", f"Held out: the quick room rebuilt {hold['k']} times "
                      f"({hold['split']} folds, options {hold['options']}), each fold's views "
                      f"scored against the room built without them. As built minus held out: "
                      + ", ".join(f"{k} {v:+.3f}" for k, v in gap.items()
                                  if k in ("psnr", "ssim", "psnr_static", "psnr_objects")) + "."]
            if hold.get("budget_hit_folds"):
                lines.append(f"Folds {hold['budget_hit_folds']} hit the texture time budget "
                             "(fewer frames used): their rooms are what the quick room makes on "
                             "this machine under its load.")
        elif hold.get("skipped"):
            lines += ["", f"Held out: not run — {hold['skipped']}."]
        for key in ("contact_sheet", "contact_sheet_holdout"):
            sheet = fid.get(key) if key == "contact_sheet" else hold.get("contact_sheet")
            if sheet:
                lines += ["", f"![{key.replace('_', ' ')}]({sheet})"]
        lines += ["", "Definitions:", ""]
        lines += [f"- **{k}**: {v}" for k, v in fid["definitions"].items()]
    else:
        lines.append(report.get("fidelity_skipped", "not evaluated"))
    lines += ["", "### Physics definitions", ""]
    lines += [f"- **{k}**: {v}" for k, v in p["definitions"].items()]
    return "\n".join(lines) + "\n"


__all__ = ["STAGE_FOLDERS", "comparison_latex", "comparison_markdown", "evaluate",
           "evaluate_scene", "fidelity_rows", "find_scenes", "finite", "headline", "infer_scan",
           "paired_difference", "paired_differences", "room_label", "stage_markdown", "stage_of",
           "stage_timings", "write_comparison", "write_json"]
