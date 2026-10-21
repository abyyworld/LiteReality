"""quickroom — a fresh scan as a simulated room in seconds: the room call's first version of it.

People hop on a video call right after scanning. The full pipeline takes hours; this takes a few
seconds on the laptop, with no Blender, no model calls and no GPU, so the call can start in the
scanned room straight away and better versions are swapped in live as they finish:

    stage "quick"    RoomPlan boxes -> furniture templates, real colours, estimated physics  (this)
    stage "objects"  TRELLIS meshes per object
    stage "full"     the pipeline's own export, run/<scan>/mujoco_seed/scene.xml

Each stage is just a ``scene.xml`` published as a new version of the same link, so the quick room
uses the pipeline's frame, handles, layout repair and physics tables throughout (see `build`).

    litereality quickroom <scan> [--out DIR]          # -> run/<scan>/quick/scene.xml + quick.json
    litereality roomcall <scan folder or name>        # builds it first when there is no scene yet

Parts: `roomplan` (room.usdz -> walls, openings, boxes), `colours` (what colour each thing is,
from the capture's frames and depth), `textures` (the walls, floor and ceiling as photos from the
same frames), `build` (templates, physics, MJCF, the separation pass).
"""

from __future__ import annotations

from litereality_agent.pipeline.quickroom.build import (
    REPORT_NAME,
    SCENE_NAME,
    TEXTURE_DIR,
    build_scene,
    write_quick_scene,
    write_textures,
)
from litereality_agent.pipeline.quickroom.roomplan import (
    USDZ_CANDIDATES,
    RoomPlan,
    find_room_usdz,
    parse_room,
)

__all__ = ["REPORT_NAME", "SCENE_NAME", "TEXTURE_DIR", "USDZ_CANDIDATES", "RoomPlan",
           "build_scene", "find_room_usdz", "parse_room", "write_quick_scene", "write_textures"]
