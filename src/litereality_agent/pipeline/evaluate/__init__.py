"""evaluate — the numbers a paper about a shared scanned room needs, per room and per stage.

People on a video call share one scanned room and one MuJoCo simulation of it. Two questions
decide whether that works, and both are answered here automatically and reproducibly, for every
stage a room goes through (quick -> objects -> full):

* **Can people do things in it, and does it hold together?** (`physics`) Every movable body is
  grabbed and pulled through the room call's own `RoomSim`, masses are checked against plausible
  real-world ranges, overlaps, drift and resets are counted, and the step cost says how many rooms
  a CPU can run in real time.
* **Does it look like the room that was scanned?** (`fidelity`) The room is drawn as the call
  draws it, from the scan's own camera poses, and compared with the real photos (PSNR / SSIM,
  overall, on walls and floor, and on the objects as the LiDAR saw them), with a contact sheet to
  see why. The quick room's walls are made of those photos, so every view is tagged with whether
  the build read it, and the quick room is also scored k-fold on views its build never saw
  (`holdout`).

`report` writes one report per stage (JSON + markdown) and the comparison table across stages
(markdown + LaTeX), with the build timings each stage recorded.

    litereality evaluate run/<scan> [--render]          # every stage the run folder has
    litereality evaluate tea_room office_room --render  # several rooms, each with its own scan
    litereality evaluate path/to/scene.xml --scan <scan> --render [--holdout 4]
"""

from __future__ import annotations

from litereality_agent.pipeline.evaluate.fidelity import (
    PlaywrightRenderer,
    Renderer,
    RendererUnavailable,
    evaluate_fidelity,
)
from litereality_agent.pipeline.evaluate.holdout import build_sources, kfold, quick_rebuilder
from litereality_agent.pipeline.evaluate.physics import MASS_RANGES, audit_physics
from litereality_agent.pipeline.evaluate.report import (
    evaluate,
    evaluate_scene,
    find_scenes,
    infer_scan,
)

__all__ = ["MASS_RANGES", "PlaywrightRenderer", "Renderer", "RendererUnavailable",
           "audit_physics", "build_sources", "evaluate", "evaluate_fidelity", "evaluate_scene",
           "find_scenes", "infer_scan", "kfold", "quick_rebuilder"]
