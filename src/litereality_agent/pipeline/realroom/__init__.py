"""realroom — the room call's second version of a scan: the quick room, but every object looks
like itself.

    stage "quick"    RoomPlan boxes -> furniture templates, real colours, estimated physics
    stage "objects"  each object's photo -> TRELLIS mesh, swapped in as it arrives    (this)
    stage "full"     the pipeline's own export

The quick room is up in seconds; this stage then cuts each object out of the capture's best photo
of it (`crops`), sends every one to TRELLIS on Modal at once — a fast preview pass, then the final
one (`generate`) — and fits each mesh that comes back into its object's box, the right way round
(`fit`). Each batch of arrivals is written as a new version of the same scene (`build`), which the
room call swaps in live: same bodies, same masses, same colliders (the quick room's template parts,
so the chairs tucked under the table still fit), new looks.

    litereality realroom <scan> [--pass preview|final] [--fake]      # offline, into run/<scan>/objects
    litereality roomcall <scan> --realistic [--publish | --deploy]    # quick room now, objects as they land
"""

from __future__ import annotations

from litereality_agent.pipeline.realroom.build import (
    BASE_NAME,
    DEFAULT_DIR,
    REPORT_NAME,
    STAGE,
    RealRoom,
    generate_room,
    write_realistic_scene,
)
from litereality_agent.pipeline.realroom.generate import (
    PASSES,
    FakeGenerator,
    ModalTrellisGenerator,
    Result,
    fake_glb,
)

__all__ = ["BASE_NAME", "DEFAULT_DIR", "PASSES", "REPORT_NAME", "STAGE", "FakeGenerator",
           "ModalTrellisGenerator", "RealRoom", "Result", "fake_glb", "generate_room",
           "write_realistic_scene"]
