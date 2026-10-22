"""holdout.py — scoring a room only on photos its build never saw.

The quick room's walls, floor and ceiling ARE photos: `quickroom.textures` resamples up to 40 of
the capture's frames onto them and `quickroom.colours` reads 24 for the furniture's colours; the
objects stage cuts each object's crops from three more. Rendering the room from those cameras and
comparing with those photos measures how well a build reproduces its own input, not how well it
reproduces the room, and a "photo textures help" result made that way would rightly be rejected.
Two things here keep the evaluation honest:

* `build_sources` says, for any stage, which frames its build read — from what the build
  recorded: ``quick.json``'s texture and colour sources through the very selectors the build
  used (they are deterministic: frames with image, depth and pose, evenly spread), and
  ``objects.json``'s crop views. Every scored view is tagged with it, so a report can always say
  how many of its views the build saw and split the numbers by that.
* `kfold` + `scan_without` + `quick_rebuilder`: a held-out protocol for the stage that is cheap to
  rebuild (the quick room, seconds). The capture's photo frames are split into k folds; for each
  fold the quick room is rebuilt — same options, by this checkout — from a copy of the capture
  with that fold's frames REMOVED (image, depth, confidence and pose: nothing of them is left to
  read), and only that fold's views are scored against it. Every view is scored once, by a room
  that never saw it. ``interleaved`` folds take every k-th frame, the usual novel-view protocol
  (the neighbours along the trajectory stay in the build, so a held-out view is near seen ones);
  ``block`` folds take contiguous stretches of the trajectory, the stricter test (a wall only that
  stretch saw has no photo to come from).

The stages that are not rebuilt here (objects: a GPU job per object; full: hours) are scored on
the views as built and tagged, so a table can never mix the two without saying so.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = ("interleaved", "block")
DEFAULT_SPLIT = "interleaved"
# A capture's per-frame files: frame_00012.json (pose, intrinsics), .jpg, depth_00012.png,
# conf_00012.png. Leaving a frame out of a build means leaving out every one of them.
FRAME_FILE = re.compile(r"^(?:frame|depth|conf)_(\d+)\.[A-Za-z0-9]+$")
QUICK_REPORT, OBJECTS_REPORT = "quick.json", "objects.json"
ROLES = ("textures", "colours", "crops")


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def recorded_scan(quick: dict | None) -> Path | None:
    """The capture folder a quick room's ``quick.json`` says it was built from (its ``source``
    names the room.usdz, possibly in the scanner's ``roomplan/`` subfolder), if it still exists."""
    try:
        source = Path(quick["source"])  # type: ignore[index]
    except (KeyError, TypeError):
        return None
    scan = source.parent if source.suffix == ".usdz" else source
    if scan.name == "roomplan":
        scan = scan.parent
    return scan if scan.is_dir() else None


# ── which frames a build read ────────────────────────────────────────────────────────────────────


@dataclass
class Sources:
    """The capture frames one stage's build read, by role (textures, colours, crops).
    ``recorded`` is False when the build kept no such record (the full pipeline): then no view
    can be called unseen, and nothing is."""

    recorded: bool
    frames: dict[str, list[int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def read(self) -> set[int]:
        return {i for frames in self.frames.values() for i in frames}

    def roles(self, index: int) -> list[str]:
        """Which roles read frame ``index`` (empty: the build never saw it)."""
        return [role for role in ROLES if index in set(self.frames.get(role, ()))]

    def seen(self, index: int) -> bool | None:
        """Whether the build read frame ``index``; None when the build kept no record."""
        return bool(self.roles(index)) if self.recorded else None

    def summary(self) -> dict:
        return {"recorded": self.recorded,
                "frames": {role: sorted(v) for role, v in self.frames.items()},
                "notes": list(self.notes)}


def build_sources(scene_xml: Path | str, scan_dir: Path | str) -> Sources:
    """Which of ``scan_dir``'s frames the build of ``scene_xml`` read (see the module docstring).
    The frame selections are recomputed on the capture the build recorded — the same folder, the
    same deterministic selectors — so they are exactly the frames it chose from; a texture pass
    that hit its time budget used a subset of them, so counting all of them as seen is the
    conservative side."""
    from litereality_agent.pipeline.quickroom import colours as quick_colours
    from litereality_agent.pipeline.quickroom import textures as quick_textures

    folder = Path(scene_xml).parent
    quick = _read_json(folder / QUICK_REPORT)
    objects = _read_json(folder / OBJECTS_REPORT)
    if quick is not None and not ({"textures", "colours"} & set(quick)):
        quick = None                 # a report from before the quick room recorded its photos
    if quick is None and objects is None:
        return Sources(False, notes=["the build recorded no frames it read (no quick.json with "
                                     "its texture and colour sources, no objects.json)"])
    out = Sources(True)
    scan = recorded_scan(quick) or Path(scan_dir)
    if quick is not None and recorded_scan(quick) is None:
        out.notes.append(f"the capture quick.json names is gone; frames chosen from {scan_dir}")
    elif quick is not None and scan.resolve() != Path(scan_dir).resolve():
        out.notes.append(f"the scene was built from {scan}, not the capture it is compared with "
                         f"({scan_dir})")
    if quick is not None:
        tex, col = quick.get("textures") or {}, quick.get("colours") or {}
        if tex.get("source") == "frames":
            picked = [f.index for f in quick_colours.capture_frames(scan, quick_textures.MAX_FRAMES)]
            out.frames["textures"] = picked
            total = tex.get("frames_total")
            if isinstance(total, int) and total != len(picked):
                out.notes.append(f"quick.json says the textures chose from {total} frames, the "
                                 f"capture now gives {len(picked)}: was it changed?")
            if tex.get("frames_used") is not None and tex.get("frames_used") != total:
                out.notes.append(f"the textures used {tex.get('frames_used')} of {total} frames "
                                 "(time budget); all of them count as seen")
        if col.get("source") == "frames":
            out.frames["colours"] = [f.index for f in
                                     quick_colours.capture_frames(scan, quick_colours.MAX_FRAMES)]
    if objects is not None:
        crops: set[int] = set()
        for crop in (objects.get("crops") or {}).values():
            if isinstance(crop, dict):
                crops.update(int(v) for v in crop.get("views") or () if isinstance(v, int))
                if isinstance(crop.get("frame"), int):
                    crops.add(int(crop["frame"]))
        out.frames["crops"] = sorted(crops)
    return out


# ── folds ────────────────────────────────────────────────────────────────────────────────────────


def kfold(indices: list[int], k: int, split: str = DEFAULT_SPLIT) -> list[list[int]]:
    """``indices`` (frame numbers, any order) in ``k`` folds: every k-th along the trajectory
    (``interleaved``) or k contiguous stretches of it (``block``). Each index is in exactly one
    fold; ``k`` is at most the number of indices."""
    if split not in SPLITS:
        raise ValueError(f"split {split!r}: expected one of {', '.join(SPLITS)}")
    ordered = sorted(set(int(i) for i in indices))
    k = max(1, min(int(k), len(ordered)))
    if not ordered:
        return []
    if split == "interleaved":
        return [ordered[i::k] for i in range(k)]
    return [chunk.tolist() for chunk in np.array_split(np.asarray(ordered), k)]


def scan_without(scan_dir: Path | str, exclude: set[int], dest: Path | str) -> Path:
    """A copy of the capture at ``dest`` with every file of the ``exclude``d frames left out:
    links to everything else (the RoomPlan export, the other frames), so it costs nothing and the
    original is never touched. A build pointed at it cannot read what is not there."""
    scan_dir, dest = Path(scan_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    for entry in sorted(scan_dir.iterdir()):
        match = FRAME_FILE.match(entry.name)
        if match and int(match.group(1)) in exclude:
            continue
        target = dest / entry.name
        try:
            target.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
        except OSError:  # a filesystem without symlinks: copy instead
            if entry.is_dir():
                shutil.copytree(entry, target)
            else:
                shutil.copy2(entry, target)
    return dest


def frames_present(scan_dir: Path | str) -> set[int]:
    """Frame numbers with any file in ``scan_dir`` (for checking an excluded frame is gone)."""
    out = set()
    for entry in Path(scan_dir).iterdir():
        match = FRAME_FILE.match(entry.name)
        if match:
            out.add(int(match.group(1)))
    return out


# ── rebuilding a stage without some frames ───────────────────────────────────────────────────────


Rebuild = Callable[[Path, Path], Path]


def quick_rebuilder(scene_xml: Path | str) -> tuple[Rebuild | None, dict[str, Any]]:
    """(rebuild(scan, out) -> scene.xml, options) for a quick room, from its own ``quick.json``:
    the same label, colours and textures settings. (None, {"reason": ...}) for anything else —
    the objects stage (a GPU job per object) and the full pipeline (hours) are not rebuilt."""
    folder = Path(scene_xml).parent
    quick = _read_json(folder / QUICK_REPORT)
    if quick is None:
        return None, {"reason": "not a quick room (no quick.json beside the scene): only the "
                                "quick room is cheap enough to rebuild per fold"}
    if (folder / OBJECTS_REPORT).is_file() or quick.get("stage") not in (None, "quick"):
        return None, {"reason": "the objects stage is not rebuilt per fold (a GPU job per "
                                "object); its views are scored as built and tagged"}
    colours_on = (quick.get("colours") or {}).get("note") != "colours off"
    textures_on = colours_on and (quick.get("textures") or {}).get("note") != "textures off"
    options = {"label": quick.get("label"), "colours": colours_on, "textures": textures_on}

    def rebuild(scan: Path, out: Path) -> Path:
        from litereality_agent.pipeline.quickroom import write_quick_scene

        return write_quick_scene(scan, out, label=options["label"], colours=colours_on,
                                 textures=textures_on)

    return rebuild, options


__all__ = ["DEFAULT_SPLIT", "SPLITS", "Sources", "build_sources", "frames_present", "kfold",
           "quick_rebuilder", "recorded_scan", "scan_without"]
