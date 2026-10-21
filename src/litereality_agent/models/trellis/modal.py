"""TRELLIS hosted on Modal, using the canonical image-to-3D request contract.

Two ways to call it:

* `ModalTrellisService.reconstruct_many` — one ordered batch through ``Function.map``, every result
  at the end (the full pipeline's reconstruct stage);
* `ModalTrellisService.stream` — sequences of requests per key, every key in flight at once and
  each result yielded the moment it lands (`stream_sequences`; the objects stage of the room
  call, which installs each object as it arrives).

The request is ``{"image_b64", "seed", "simplify", "texture_size", "pipeline_type"}`` plus an
optional ``"id"`` the function echoes back, so a streamed result can be checked against the
request it answers; the response is ``{"glb_b64"}`` (and ``"id"``), or ``{"error"}``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
import queue
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litereality_agent.runtimes.modal import ModalClient

# Requests in flight at once for `stream`: twice the deployed function's `max_containers` (4), so
# every container has its next input queued the moment it finishes one, and no more — a request
# waiting in Modal's queue is a request that cannot be re-ordered or cancelled cheaply.
MAX_IN_FLIGHT = 8
# Seconds one streamed request may take, queueing included: a cold start (image pull and model
# load, minutes) plus the function's own 20-minute timeout.
CALL_TIMEOUT = 30 * 60
PIPELINE_TYPES = ("512", "1024", "1024_cascade", "1536_cascade")


@dataclass(slots=True)
class AssetReport:
    asset_id: str
    ok: bool
    wall_s: float
    glb_path: str = ""
    error: str = ""


@dataclass(slots=True)
class BatchReport:
    assets: list[AssetReport] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def ok_count(self) -> int:
        return sum(asset.ok for asset in self.assets)

    def render(self) -> str:
        lines = [
            f"  TRELLIS/Modal batch: {self.ok_count}/{len(self.assets)} ok  wall={self.wall_s:.1f}s"
        ]
        for asset in self.assets:
            status = "ok " if asset.ok else "ERR"
            suffix = "" if asset.ok else f"  {asset.error}"
            lines.append(f"    [{status}] {asset.asset_id:16s} wall={asset.wall_s:5.1f}s{suffix}")
        return "\n".join(lines)


class ModalTrellisService:
    """Generation3DService adapter for a deployed Modal TRELLIS function."""

    name = "trellis-modal"

    def __init__(
        self,
        *,
        app_name: str,
        function_name: str = "trellis",
        environment_name: str = "main",
        profile: str | None = None,
        credentials: tuple[str, str] | None = None,
        client: Any = None,
    ) -> None:
        self.client = client or ModalClient(
            app_name,
            function_name,
            environment_name=environment_name,
            profile=profile,
            credentials=credentials,
        )
        self.last_report: BatchReport | None = None

    def reconstruct(
        self, images: list[Any], *, out_dir: str, asset_id: str = "asset", **opts: Any
    ) -> str:
        image = images[0] if isinstance(images, (list, tuple)) else images
        return self.reconstruct_many({asset_id: image}, out_dir=out_dir, **opts)[asset_id]

    def reconstruct_many(
        self,
        items: dict[str, Any],
        *,
        out_dir: str,
        seed: int = 42,
        simplify: float = 0.95,
        texture_size: int = 1024,
        pipeline_type: str | None = None,
    ) -> dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        asset_ids = list(items)
        payloads = [
            {
                "image_b64": _image_b64(items[asset_id]),
                "seed": seed,
                "simplify": simplify,
                "texture_size": texture_size,
                "pipeline_type": pipeline_type,
            }
            for asset_id in asset_ids
        ]

        started = time.monotonic()
        outputs = self.client.map(payloads)
        elapsed = time.monotonic() - started
        if len(outputs) != len(asset_ids):
            raise RuntimeError(
                f"Modal TRELLIS returned {len(outputs)} results for {len(asset_ids)} inputs"
            )

        results: dict[str, str] = {}
        report = BatchReport(wall_s=elapsed)
        for asset_id, output in zip(asset_ids, outputs, strict=True):
            path = out / f"{asset_id}.glb"
            if isinstance(output, BaseException):
                results[asset_id] = ""
                report.assets.append(
                    AssetReport(
                        asset_id,
                        False,
                        elapsed,
                        error=f"{type(output).__name__}: {output}",
                    )
                )
                continue
            error = output.get("error")
            if error:
                results[asset_id] = ""
                report.assets.append(AssetReport(asset_id, False, elapsed, error=str(error)))
                continue
            try:
                path.write_bytes(base64.b64decode(output["glb_b64"], validate=True))
            except (KeyError, ValueError) as exc:
                results[asset_id] = ""
                report.assets.append(
                    AssetReport(asset_id, False, elapsed, error=f"invalid response: {exc}")
                )
                continue
            results[asset_id] = str(path)
            report.assets.append(AssetReport(asset_id, True, elapsed, glb_path=str(path)))

        self.last_report = report
        print(report.render(), flush=True)
        return results

    def stream(self, sequences: dict[str, list[dict]], *, max_in_flight: int = MAX_IN_FLIGHT,
               timeout: float = CALL_TIMEOUT) -> Iterator[StreamItem]:
        """`stream_sequences` on the deployed function, from synchronous code: the requests run
        on an event loop in a worker thread and each result is yielded here as it lands."""
        function = self.client.function
        remote = getattr(getattr(function, "remote", None), "aio", None)
        if remote is None:
            raise RuntimeError("streaming needs a Modal function with `.remote.aio`")
        hydrate = getattr(function, "hydrate", None)
        if callable(hydrate):
            # Look the function up once, here — a missing deployment is one clear error — rather
            # than in every one of the requests about to start at once.
            hydrate()
        yield from iterate_in_thread(
            stream_sequences(remote, sequences, max_in_flight=max_in_flight, timeout=timeout))


@dataclass(frozen=True, slots=True)
class StreamItem:
    """One finished request of `stream_sequences`: the ``index``-th request of ``key``'s
    sequence, its response (``output``) or why there is none (``error``), and how long it took
    from being sent (queueing on Modal included)."""

    key: str
    index: int
    output: dict | None
    seconds: float
    error: str = ""


async def stream_sequences(remote, sequences: dict[str, list[dict]], *,
                           max_in_flight: int = MAX_IN_FLIGHT, timeout: float = CALL_TIMEOUT
                           ) -> AsyncIterator[StreamItem]:
    """Run each key's requests in order and all keys at once; yield each result as it lands.

    ``remote`` is an async callable taking one request (``Function.remote.aio``). At most
    ``max_in_flight`` requests are out at a time, granted first come first served: every key's
    first request queues before any key's second one, so with sequences ``[preview, final]`` all
    previews go out before any final, and each final follows its own preview the moment it is
    back. A request that fails (an exception, a timeout, or an ``{"error"}`` response) is yielded
    with its error, and the key's next request is still sent — a final can succeed where its
    preview did not.
    """
    gate = asyncio.Semaphore(max(1, int(max_in_flight)))
    done: asyncio.Queue = asyncio.Queue()

    async def run(key: str, requests: list[dict]) -> None:
        for index, request in enumerate(requests):
            async with gate:
                started = time.monotonic()
                try:
                    output = await asyncio.wait_for(remote(request), timeout)
                except asyncio.TimeoutError:
                    item = StreamItem(key, index, None, time.monotonic() - started,
                                      f"no answer within {timeout:g} s")
                except Exception as exc:  # noqa: BLE001 — one object's failure is reported, not raised
                    item = StreamItem(key, index, None, time.monotonic() - started,
                                      f"{type(exc).__name__}: {exc}")
                else:
                    error = output.get("error") if isinstance(output, dict) else "not a dict"
                    item = StreamItem(key, index, output if not error else None,
                                      time.monotonic() - started, str(error or ""))
            await done.put(item)

    tasks = [asyncio.ensure_future(run(key, list(requests)))
             for key, requests in sequences.items() if requests]
    remaining = sum(len(requests) for requests in sequences.values())
    try:
        for _ in range(remaining):
            yield await done.get()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


_DONE = object()


def iterate_in_thread(agen: AsyncIterator[Any]) -> Iterator[Any]:
    """An async iterator consumed on its own event loop in a daemon thread, as a plain iterator.
    An exception that ends it is re-raised here; leaving the loop early stops the thread's loop
    (cancelling whatever is still in flight)."""
    out: queue.Queue = queue.Queue()
    stop = threading.Event()
    loop = asyncio.new_event_loop()

    async def pump() -> None:
        try:
            async for item in agen:
                out.put(item)
                if stop.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001 — handed to the consumer
            out.put(exc)
        finally:
            with contextlib.suppress(Exception):
                await agen.aclose()
            out.put(_DONE)

    def run() -> None:
        with contextlib.suppress(BaseException):  # pump hands every exception to the consumer
            loop.run_until_complete(pump())

    thread = threading.Thread(target=run, daemon=True, name="trellis-stream")
    thread.start()
    try:
        while True:
            item = out.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        if thread.is_alive():
            loop.call_soon_threadsafe(lambda: [t.cancel() for t in asyncio.all_tasks(loop)])
            thread.join(timeout=5)
        if not thread.is_alive():
            loop.close()


def request_payload(image: Any, *, seed: int = 42, simplify: float = 0.95,
                    texture_size: int = 1024, pipeline_type: str | None = None,
                    request_id: str | None = None) -> dict:
    """The function's request for one image (bytes, a path, or a PIL image; an RGBA image's alpha
    is used as the object's mask as-is — see deploy/modal/trellis/app.py)."""
    if pipeline_type is not None and pipeline_type not in PIPELINE_TYPES:
        raise ValueError(f"pipeline_type {pipeline_type!r}: expected one of {PIPELINE_TYPES}")
    payload = {"image_b64": _image_b64(image), "seed": int(seed), "simplify": float(simplify),
               "texture_size": int(texture_size), "pipeline_type": pipeline_type}
    if request_id is not None:
        payload["id"] = str(request_id)
    return payload


def decode_glb(output: Any, *, expect_id: str | None = None) -> bytes:
    """The GLB bytes of one response. ValueError for an error response, a response to another
    request (its echoed ``id`` differs; a function too old to echo one is accepted), or anything
    that is not a GLB."""
    if not isinstance(output, dict):
        raise ValueError(f"invalid response: {type(output).__name__}")
    if output.get("error"):
        raise ValueError(str(output["error"]))
    echoed = output.get("id")
    if expect_id is not None and echoed is not None and echoed != expect_id:
        raise ValueError(f"response for {echoed!r}, expected {expect_id!r}")
    try:
        data = base64.b64decode(output["glb_b64"], validate=True)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid response: {exc}") from None
    if data[:4] != b"glTF":
        raise ValueError("invalid response: not a GLB")
    return data


def _image_b64(image: Any) -> str:
    if isinstance(image, (bytes, bytearray)):
        data = bytes(image)
    elif isinstance(image, (str, os.PathLike)):
        data = Path(image).read_bytes()
    else:
        buffer = io.BytesIO()
        image.convert("RGBA").save(buffer, format="PNG")
        data = buffer.getvalue()
    return base64.b64encode(data).decode("ascii")
