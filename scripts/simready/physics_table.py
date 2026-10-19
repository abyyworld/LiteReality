#!/usr/bin/env python3
"""physics_table.py — the physics a room was GIVEN, one row per thing that can move.

    uv run python scripts/simready/physics_table.py run/<scan>/mujoco_seed/scene.xml [--csv out.csv]

Every number here is an estimate made before anything was watched moving: a mass from a category
density over the object's bounding box (or a figure read off a label), a friction looked up from the
material name the recipe chose, a restitution from the same table. That is the point of printing
them — they are the "initial physics" a later dynamic observation would correct, so they have to be
legible on their own rather than buried in an MJCF.

Two tables, because they answer different questions:

``simulated``  read from the compiled scene, so it is exactly what MuJoCo steps: each free body
               (with everything mounted on it) and each articulated part, its mass, the sliding
               friction of its collision geoms, and how it is jointed. The room's own shaker
               joints (``room_*``) are not furniture and are left out.
``sources``    read from each object's ``physics.json`` sidecar in the Room package beside the
               scene: where the mass came from, and the restitution — which MuJoCo has no
               parameter for, so it is carried for other engines and never simulated here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import mujoco

JOINT_KIND = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


def _name(model, kind, index: int, fallback: str) -> str:
    return mujoco.mj_id2name(model, kind, index) or fallback


def _frictions(model, body: int) -> str:
    """Sliding friction of the body's COLLISION geoms; visual-only geoms never touch anything."""
    values = sorted({
        round(float(model.geom_friction[g][0]), 3)
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) == body
        and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
    })
    return ", ".join(f"{v:g}" for v in values) or "-"


def simulated(scene: Path) -> list[dict]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    rows = []
    for body in range(1, model.nbody):
        joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) == body
                  and not _name(model, mujoco.mjtObj.mjOBJ_JOINT, j, "").startswith("room_")]
        if not joints:
            continue                        # static geometry, or the shaker carrying the room
        kinds = [JOINT_KIND.get(int(model.jnt_type[j]), "?") for j in joints]
        free = "free" in kinds
        rows.append({
            "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body, f"body{body}"),
            "joint": ", ".join(kinds),
            # A free body's mass is its subtree: a cabinet weighs its carcass AND its drawers.
            # An articulated part is reported on its own, which is the mass its joint moves.
            "mass_kg": round(float(model.body_subtreemass[body] if free else model.body_mass[body]), 3),
            "friction": _frictions(model, body),
        })
    return rows


def default_room(scene: Path) -> Path | None:
    """The Room package a scene was exported from, for the two places the pipeline writes them."""
    scene = scene.resolve()
    if scene.parent.name == "mujoco_seed":
        return scene.parent.parent / "scene_init" / "scene_stage" / "room_init" / "room"
    if scene.parent.name == "mujoco" and scene.parent.parent.name == "realism_authoring":
        return scene.parent.parent / "room"
    return None


def sources(room: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(room).glob("Objects/*/*/sim/*.physics.json")):
        try:
            model = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        links = model.get("links") or []

        def distinct(key):
            values = sorted({round(float(link[key]), 3) for link in links if link.get(key) is not None})
            return ", ".join(f"{v:g}" for v in values) or "-"

        notes = model.get("notes") or []
        rows.append({
            "object": model.get("name") or path.stem.removesuffix(".physics"),
            "category": model.get("category") or "-",
            "mass_kg": round(float(model.get("total_mass") or 0.0), 3),
            "mass_source": ", ".join(sorted({link.get("mass_source") or "?" for link in links})),
            "friction": distinct("friction"),
            "restitution": distinct("restitution"),
            "materials": ", ".join(sorted({link.get("material") or "-" for link in links})),
            "note": notes[0] if notes else "",
        })
    return rows


def markdown(rows: list[dict]) -> str:
    if not rows:
        return "_none_"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(row[k]) for k in keys) + " |" for row in rows]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, help="an exported mujoco scene.xml")
    ap.add_argument("--room", type=Path, default=None,
                    help="Room package holding the physics sidecars (default: inferred from the scene path)")
    ap.add_argument("--csv", type=Path, default=None, help="also write the simulated table as CSV")
    args = ap.parse_args(argv)
    if not args.scene.is_file():
        print(f"no scene at {args.scene}", file=sys.stderr)
        return 1

    sim = simulated(args.scene)
    print(f"## Simulated ({args.scene})\n\n{markdown(sim)}\n")
    room = args.room or default_room(args.scene)
    if room is not None and room.is_dir():
        print(f"## Where the numbers came from ({room})\n\n{markdown(sources(room))}\n")
        print("MuJoCo has no restitution parameter: the restitution column is carried for other "
              "engines and is not simulated.")
    else:
        print("(no Room package found beside the scene; pass --room to list the physics sidecars)")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["body", "joint", "mass_kg", "friction"])
            writer.writeheader()
            writer.writerows(sim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
