"""Call the DEPLOYED gsplat trainer (`litereality-splat`.train, deploy/modal/splat) on a RealityScan
dataset tar from a `modal run` entrypoint (the Modal CLI's own login), for
scripts/realityscan/train_rs.py. Nothing here is deployed; research runs only.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file .env modal run deploy/modal/realityscan/train_splat.py \\
        --tar ~/LiteReality-scans/rs_stray/rs_w960.tar.gz --options '{"steps": 3000}' --out /tmp/x.pkl
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import modal

app = modal.App("litereality-realityscan-train")


@app.local_entrypoint()
def main(tar: str, options: str = "{}", out: str = "result.pkl") -> None:
    train = modal.Function.from_name("litereality-splat", "train")
    data = Path(tar).expanduser().read_bytes()
    events = []
    for ev in train.remote_gen(data, json.loads(options)):
        if not isinstance(ev, dict):
            continue
        events.append(ev)
        print({k: v for k, v in ev.items() if k not in ("spz", "log")}, flush=True)
    Path(out).write_bytes(pickle.dumps(events))
