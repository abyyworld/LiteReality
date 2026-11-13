"""The deployed splat trainer (`deploy/modal/splat/app.py`), from this machine."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from litereality_agent.runtimes.modal import ModalClient

APP_NAME = "litereality-splat"
FUNCTION_NAME = "train"
EVENT_KINDS = ("started", "snapshot", "done", "error")


class ModalSplatTrainer:
    """`stream(tar, options)` yields the trainer's events as they happen (see the app's docstring);
    it always ends with one "done" or "error" event, also when the call itself fails."""

    name = "gsplat-modal"

    def __init__(self, *, environment_name: str = "main", profile: str | None = None,
                 credentials: tuple[str, str] | None = None, function: Any = None) -> None:
        self.client = ModalClient(APP_NAME, FUNCTION_NAME, environment_name=environment_name,
                                  profile=profile, credentials=credentials, function=function)

    def warm(self) -> None:
        """Boot a GPU container now, in the background (``train(b"")`` returns at once and the
        container waits `scaledown_window` for the real call): its ~10 s cold boot then happens
        while the instant room is computed. Costs only those idle seconds; never raises."""
        import threading

        def run() -> None:
            try:
                for _event in self.client.function.remote_gen(b"", {}):
                    pass
            except Exception:  # noqa: BLE001 — a failed warm-up only costs the cold boot later
                pass
        threading.Thread(target=run, daemon=True, name="splat-warm-up").start()

    def stream(self, dataset_tar: bytes, options: dict | None = None) -> Iterator[dict]:
        ended = False
        try:
            for event in self.client.function.remote_gen(dataset_tar, options or {}):
                if not isinstance(event, dict) or event.get("kind") not in EVENT_KINDS:
                    continue
                ended = ended or event["kind"] in ("done", "error")
                yield event
        except Exception as exc:  # noqa: BLE001 — not deployed, no credit, network: one event
            if not ended:
                yield {"kind": "error", "error": f"{type(exc).__name__}: {exc}", "log": ""}
            return
        if not ended:
            yield {"kind": "error", "error": "the trainer stopped without a result", "log": ""}


SEGMENT_APP = "litereality-segment"
SEGMENT_FUNCTION = "segment"


def segment_call(*, environment_name: str = "main", profile: str | None = None,
                 credentials: tuple[str, str] | None = None):
    """The deployed SAM 2 function (`deploy/modal/segment`) as a plain ``request -> reply``
    callable, for `pipeline.splatroom.masks.request`."""
    client = ModalClient(SEGMENT_APP, SEGMENT_FUNCTION, environment_name=environment_name,
                         profile=profile, credentials=credentials)
    return client.function.remote
