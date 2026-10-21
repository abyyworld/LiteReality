"""The room call as a permanent public link (see room_ops/roomcall/hosted.py and doc/roomcall.md).

    uv run litereality roomcall run/tea_room --deploy           # upload a scene, deploy, print link
    uv run litereality roomcall run/tea_room --publish          # a new version, live, no redeploy
    uv run modal deploy --env main deploy/modal/roomcall/app.py  # by hand: serves the scene uploaded
                                                                  # last, or the demo room

Its own app, never included into `litereality` (deploy/modal/app.py): that app is the GPU models
the pipeline looks up and is always deployed whole, while this is a CPU web server with its own
lifecycle — redeploying one must not restart the other.
"""

from __future__ import annotations

from pathlib import Path

import modal

from litereality_agent.room_ops.roomcall import hosted

WEB = Path(hosted.__file__).resolve().parent / "web"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*hosted.IMAGE_PACKAGES)
    # `add_local_python_source` carries .py files only; the browser client is html/js/css.
    .add_local_dir(WEB, hosted.WEB_IN_CONTAINER)
    .add_local_python_source("litereality_agent")
)
scenes = modal.Volume.from_name(hosted.SCENES_VOLUME, create_if_missing=True)
app = modal.App(hosted.APP_NAME)


# ONE container on purpose: every room lives in this process's memory, so a second container would
# put half of a call's people in a room the other half cannot see. It takes every connection at
# once instead (each WebSocket is one Modal input, held for as long as the person stays — up to
# `timeout`). Two CPUs: the event loop and one room's physics each get a core. It scales to zero
# `scaledown_window` after the last person leaves, so a link nobody is using costs nothing.
# 4 GB: during a live swap the old and the new version are both in memory (a real room's model is
# ~0.7 GB), and a version published without prebuilt geometry is built in a child process.
@app.function(
    image=image,
    volumes={hosted.SCENES_MOUNT: scenes},
    cpu=2.0,
    memory=4096,
    max_containers=1,
    scaledown_window=15 * 60,
    timeout=6 * 60 * 60,
)
@modal.concurrent(max_inputs=200)
@modal.asgi_app()
def web():
    # `reload` lets the app see versions `--publish` uploads while it runs (`hosted.VolumeWatch`):
    # a mounted Volume otherwise shows only what was committed when the container started.
    return hosted.build_app(Path(hosted.SCENES_MOUNT), web_dir=Path(hosted.WEB_IN_CONTAINER),
                            reload=scenes.reload, poll_seconds=hosted.POLL_SECONDS, freeze=True)
