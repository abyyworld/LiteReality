"""The one Modal app for LiteReality's models.

    uv run modal deploy --env main deploy/modal/app.py   # or `uv run litereality setup`

Functions: ``trellis`` (TRELLIS.2, deploy/modal/trellis), ``dino`` (GroundingDINO + DINOv2,
deploy/modal/dino), and ``image_edit`` (Qwen-Image-Edit-2509 reference images for
``LR_IMAGE_PROVIDER=modal``, deploy/modal/image_edit). A deploy replaces every function on the
app, which is why all three are deployed together here and never one at a time.
"""

from __future__ import annotations

import modal
from dino.app import app as dino_app
from image_edit.app import app as image_edit_app
from trellis.app import app as trellis_app

APP_NAME = "litereality"

app = modal.App(APP_NAME)
app.include(trellis_app)
app.include(dino_app)
app.include(image_edit_app)
