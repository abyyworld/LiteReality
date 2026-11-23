"""Meeting reminders for the hosted room call: every minute, a Web Push to everyone who asked to be
reminded of a meeting starting in 15 minutes (see room_ops/roomcall/meetings.py and the "Meeting
reminders" section of doc/roomcall.md).

    uv run --env-file .env python scripts/roomcall/vapid_secret.py --subject mailto:<you>  # once
    uv run --env-file .env modal deploy --env main deploy/modal/reminders/app.py

Its own app, not a second function of `litereality-roomcall` (deploy/modal/roomcall/app.py), on
purpose:
- that app is ONE always-the-same web container holding every call room in memory; deploying a
  function into it replaces its running container, so every change here would drop the calls in
  progress. Deployed apart, the reminders change without anyone leaving a room;
- only this function holds the VAPID private key (the `roomcall-vapid` secret), so the room call
  deploys, as before, with no secret at all;
- the two share nothing but the scenes Volume, under `.meetings/` (one writer per file: the room
  call writes meetings and subscriptions, this writes which were reminded and the public key).

Cost: 1440 runs a day of 0.125 CPU and 256 MB, each a container start plus a second or two of
work (`scaledown_window=2`, so nothing idles between runs) — roughly $0.5 a month on Modal's
per-second prices; see doc/roomcall.md.
"""

from __future__ import annotations

from pathlib import Path

import modal

from litereality_agent.room_ops.roomcall import hosted, meetings

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(*meetings.REMINDER_PACKAGES)
    .add_local_python_source("litereality_agent")
)
scenes = modal.Volume.from_name(hosted.SCENES_VOLUME, create_if_missing=True)
app = modal.App(meetings.REMINDERS_APP)


# Every minute, one container at a time (a slow round is never overlapped by the next, which would
# send its reminders twice), gone two seconds after each round. A round that fails is not retried:
# the next minute's covers it (a reminder is due for the whole last 15 minutes before the start).
# A round starts no new batch of pushes after `meetings.ROUND_SECONDS` (45 s), well inside the
# 120 s timeout, and commits the Volume after each batch, so a round cut short anyway loses no
# record of who was reminded and sends nobody the same reminder twice.
@app.function(
    image=image,
    volumes={hosted.SCENES_MOUNT: scenes},
    secrets=[modal.Secret.from_name(meetings.VAPID_SECRET, required_keys=list(meetings.VAPID_KEYS))],
    schedule=modal.Cron("* * * * *"),
    cpu=0.125,
    memory=256,
    timeout=120,
    max_containers=1,
    scaledown_window=2,
)
def remind() -> str:
    # A mounted Volume shows what was committed when the container started: reload for the
    # meetings and subscriptions the room call wrote since, commit what this round changed (after
    # each batch of pushes, and once more at the end for the pruning).
    scenes.reload()
    report = meetings.cron_round(Path(hosted.SCENES_MOUNT), commit=scenes.commit)
    scenes.commit()
    print(report)
    return str(report)
