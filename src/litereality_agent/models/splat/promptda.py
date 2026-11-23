"""Dense metric depth for every photo of a scan: Prompt Depth Anything on Modal, from this machine.

PromptDA (Lin et al., CVPR 2025, Apache-2.0) takes one photo and the iPhone's own 256 x 192 LiDAR
map as a "prompt" and returns metric depth at the photo's resolution: the LiDAR's scale and the
photo's edges. The deployed function (`deploy/modal/depth/app.py`, app `litereality-depth`)
returns it as uint16 millimetre PNGs at `width` x round(width x 3/4) for a 4:3 scan; here those
become float32 metres, keyed by the frame's index, and are cached beside the scan:

    <scan>/depth_dense/frame_00030.png      uint16 millimetres, 0 = none

so a scan is refined once: later runs (and the trainer's dataset, the instant splats, the cleanup)
read the cache instead of calling Modal. Each cached PNG carries the key of what it was made from
(a hash of the frame's LiDAR PNG and the photo's byte size, `source_key`) in a text chunk: a scan
exported again into the same folder misses the cache instead of reading the old capture's depth.
Consumers never trust the map blindly: `rgbd.dense_mask` drops dense samples that disagree with a
confident LiDAR sample under them.

`ModalDepthRefiner.refine` never raises: not deployed, no credit, a network drop, a bad reply —
it logs why and returns what it has (the cached frames, or {}), and the caller carries on with the
LiDAR alone.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from litereality_agent.models.splat.rgbd import Frame

APP_NAME = "litereality-depth"
FUNCTION_NAME = "refine"
DENSE_DIR = "depth_dense"
WIDTH = 960                   # the trainer's training width (dataset.TRAIN_WIDTH)
BATCH = 16                    # frames per call: ~4 MB up, ~10 MB back
PARALLEL_CALLS = 4            # the app's max_containers
# PromptDA's own input size (its README's `load_image`): the long side at most 1008 px, both sides
# a multiple of its ViT's 14 px patch. Photos may be shrunk to it here rather than on the GPU (the
# model does the same resize): fewer bytes up for a scan saved at high JPEG quality.
MODEL_SIDE, PATCH = 1008, 14
UPLOAD_QUALITY = 95

_log = logging.getLogger(__name__)


def model_size(width: int, height: int, max_side: int = MODEL_SIDE) -> tuple[int, int]:
    """The (width, height) PromptDA runs a photo at: its README's rule, scaled so the long side is
    at most `max_side`, each side rounded down to a multiple of 14."""
    max_side = max_side // PATCH * PATCH
    scale = min(1.0, max_side / max(width, height))
    return (max(PATCH, int(width * scale) // PATCH * PATCH),
            max(PATCH, int(height * scale) // PATCH * PATCH))


def output_height(frame: Frame, width: int) -> int:
    """The dense map's height at `width`: the photo's own aspect (720 for 960 on a 4:3 scan),
    matching the training images (`dataset.build`)."""
    return int(round(frame.size[1] * width / frame.size[0]))


def dense_path(frame: Frame) -> Path:
    """Where a frame's dense depth is cached: beside the scan, named like its training image."""
    return frame.image.parent / DENSE_DIR / f"frame_{frame.index:05d}.png"


SOURCE_KEY = "litereality-source"      # the PNG text chunk naming what a cached map came from


def source_key(frame: Frame) -> str:
    """What a frame's dense depth is made from, in 16 hex digits: a hash of its LiDAR PNG's bytes
    and its photo's size in bytes. A new capture written over the old one changes both."""
    h = hashlib.sha256(frame.depth.read_bytes())
    h.update(str(frame.image.stat().st_size).encode())
    return h.hexdigest()[:16]


def decode_depth(data: bytes) -> np.ndarray:
    """A uint16 millimetre PNG -> float32 metres (0 = none)."""
    return _decode(data)[0]


def _decode(data: bytes) -> tuple[np.ndarray, str | None]:
    """A uint16 millimetre PNG -> (float32 metres, its `SOURCE_KEY` text or None)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        mm = np.asarray(im)
        key = (getattr(im, "text", None) or {}).get(SOURCE_KEY)
    if mm.ndim != 2:
        raise ValueError(f"depth PNG has shape {mm.shape}, not one channel")
    return mm.astype(np.float32) / 1000.0, key


def encode_depth(depth: np.ndarray, key: str | None = None) -> bytes:
    """Float metres (0 = none) -> a uint16 millimetre PNG, with `key` as its `SOURCE_KEY`."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    mm = np.clip(np.round(np.nan_to_num(np.asarray(depth, dtype=np.float64)) * 1000.0), 0, 65535)
    info = None
    if key is not None:
        info = PngInfo()
        info.add_text(SOURCE_KEY, key)
    buf = io.BytesIO()
    Image.fromarray(mm.astype(np.uint16)).save(buf, format="PNG", compress_level=1, pnginfo=info)
    return buf.getvalue()


def load_dense(frame: Frame, width: int | None = None) -> np.ndarray | None:
    """The frame's cached dense depth in metres, or None when there is none: missing,
    unreadable, made from another capture (its key is not `source_key(frame)`, or it has none),
    or not `width` x `output_height` when `width` is given (made for another training size)."""
    path = dense_path(frame)
    if not path.is_file():
        return None
    try:
        depth, key = _decode(path.read_bytes())
        if key != source_key(frame):
            return None
    except (OSError, ValueError):
        return None
    if width is not None and depth.shape != (output_height(frame, width), width):
        return None
    return depth


def is_cached(frame: Frame, width: int | None = None) -> bool:
    """Whether `load_dense` would find this frame's depth, from the PNG's header alone (its size
    and key come before the pixels): quick enough to ask of every frame before deciding to wait."""
    from PIL import Image

    path = dense_path(frame)
    if not path.is_file():
        return False
    try:
        with Image.open(path) as im:
            size, key = im.size, im.info.get(SOURCE_KEY)
        if key != source_key(frame):
            return False
    except (OSError, ValueError):
        return False
    return width is None or size == (width, output_height(frame, width))


def save_dense(frame: Frame, depth: np.ndarray) -> Path:
    """Cache a frame's dense depth (metres) as `dense_path(frame)`, keyed to the LiDAR map and
    photo it came from (`source_key`); written whole or not at all."""
    path = dense_path(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(encode_depth(depth, source_key(frame)))
    tmp.replace(path)
    return path


def request_item(frame: Frame, upload_side: int | None = MODEL_SIDE) -> dict:
    """One frame as the deployed function takes it: its photo, its LiDAR PNG and confidence PNG
    as the scan wrote them. With `upload_side` the photo is also shrunk to PromptDA's input size
    and that JPEG sent instead when it is the smaller (a 450 KB 1920 x 1440 photo becomes ~120
    KB; a scan already saved at low quality is sent as it is)."""
    from PIL import Image

    image = frame.image.read_bytes()
    if upload_side:
        with Image.open(io.BytesIO(image)) as im:
            small = im.convert("RGB")
            size = model_size(*small.size, max_side=upload_side)
            if size != small.size:
                # BOX: the area average PromptDA's own cv2 INTER_AREA resize computes.
                small = small.resize(size, Image.BOX)
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=UPLOAD_QUALITY)
        if buf.tell() < len(image):
            image = buf.getvalue()
    # "size": the photo as captured, which sets the reply's aspect (`output_height`); the JPEG
    # sent may be shrunk to multiples of 14, which is not quite the same aspect.
    return {"name": f"frame_{frame.index:05d}", "image": image,
            "depth": frame.depth.read_bytes(),
            "conf": frame.conf.read_bytes() if frame.conf is not None else None,
            "size": [int(frame.size[0]), int(frame.size[1])]}


class ModalDepthRefiner:
    """`refine(frames, width)` -> {frame index: dense depth (float32 metres, H x width)}; cached
    frames are read from the scan's `depth_dense/`, the rest refined on Modal in batches of
    `batch` and cached there. Never raises (see the module docstring)."""

    name = "promptda-modal"

    def __init__(self, *, environment_name: str = "main", profile: str | None = None,
                 credentials: tuple[str, str] | None = None, function: Any = None,
                 batch: int = BATCH, upload_side: int | None = MODEL_SIDE, log=None) -> None:
        self.environment_name, self.profile, self.credentials = environment_name, profile, credentials
        self._function = function
        self.batch = max(1, int(batch))
        self.upload_side = upload_side
        self.log = log
        self.last_stats: dict = {}

    @property
    def function(self):
        """The deployed function, looked up on first use (so constructing never needs Modal)."""
        if self._function is None:
            from litereality_agent.runtimes.modal import ModalClient

            self._function = ModalClient(APP_NAME, FUNCTION_NAME,
                                         environment_name=self.environment_name,
                                         profile=self.profile,
                                         credentials=self.credentials).function
        return self._function

    def _say(self, message: str) -> None:
        _log.info(message)
        if self.log:
            self.log(message)

    def warm(self) -> None:
        """Boot a GPU container now, in the background (an empty call returns at once and the
        container waits for the real one), so its cold start overlaps other work. Never raises."""
        import threading

        def run() -> None:
            try:
                self.function.remote([], {})
            except Exception:  # noqa: BLE001 — a failed warm-up only costs the cold boot later
                pass
        threading.Thread(target=run, daemon=True, name="depth-warm-up").start()

    def refine(self, frames: list[Frame], width: int = WIDTH) -> dict[int, np.ndarray]:
        t0 = time.perf_counter()
        out: dict[int, np.ndarray] = {}
        stats = {"frames": len(frames), "cached": 0, "refined": 0, "failed": 0, "seconds": 0.0}
        self.last_stats = stats
        try:
            missing = []
            for frame in frames:
                depth = load_dense(frame, width)
                if depth is None:
                    missing.append(frame)
                else:
                    out[frame.index] = depth
            stats["cached"] = len(out)
            if missing:
                self._refine_missing(missing, width, out, stats)
        except Exception as exc:  # noqa: BLE001 — dense depth is an improvement, never a blocker
            self._say(f"dense depth: failed ({type(exc).__name__}: {exc}); "
                      f"using {len(out)} cached frames")
        stats["seconds"] = round(time.perf_counter() - t0, 2)
        return out

    def _refine_missing(self, frames: list[Frame], width: int, out: dict, stats: dict) -> None:
        t0 = time.perf_counter()
        function = self.function            # one lookup (and one credentials error), not per batch
        items: dict[str, tuple[Frame, dict]] = {}

        def prepare(frame: Frame):
            try:
                return frame, request_item(frame, self.upload_side)
            except (OSError, ValueError) as exc:
                self._say(f"dense depth: skipping frame {frame.index} ({exc})")
                return frame, None
        with ThreadPoolExecutor(max_workers=8) as pool:
            for frame, item in pool.map(prepare, frames):
                if item is not None:
                    items[item["name"]] = (frame, item)
        names = list(items)
        batches = [names[i:i + self.batch] for i in range(0, len(names), self.batch)]
        options = {"width": int(width)}
        upload = sum(len(items[n][1]["image"]) + len(items[n][1]["depth"]) for n in names)

        def call(batch: list[str]):
            try:
                return batch, function.remote([items[n][1] for n in batch], options), None
            except Exception as exc:  # noqa: BLE001 — one failed batch loses only its frames
                return batch, None, exc
        ms, errors = [], []
        with ThreadPoolExecutor(max_workers=PARALLEL_CALLS) as pool:
            for batch, reply, exc in pool.map(call, batches):
                if exc is not None:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                for result in reply if isinstance(reply, list) else []:
                    got = self._accept(result, items, width)
                    if got is not None:
                        frame, depth, took = got
                        out[frame.index] = depth
                        stats["refined"] += 1
                        if took is not None:
                            ms.append(took)
                        try:
                            save_dense(frame, depth)
                        except OSError as exc:
                            self._say(f"dense depth: could not cache frame {frame.index} ({exc})")
        stats["failed"] = len(frames) - stats["refined"]
        if ms:
            stats["gpu_ms_median"] = round(float(np.median(ms)), 1)
        if errors:
            stats["errors"] = errors[:3]
        self._say(f"dense depth: {stats['refined']}/{len(frames)} frames refined on Modal in "
                  f"{time.perf_counter() - t0:.1f} s ({upload / 1e6:.0f} MB up"
                  + (f", {stats['gpu_ms_median']:.0f} ms a frame on the GPU" if ms else "")
                  + f"), {stats['cached']} cached"
                  + (f"; {len(errors)} of {len(batches)} calls failed: {errors[0]}" if errors
                     else ""))

    def _accept(self, result: Any, items: dict, width: int):
        """One reply item -> (frame, depth in metres, GPU ms), or None when it is not a depth map
        of the asked size for a frame that was sent."""
        if not isinstance(result, dict) or result.get("name") not in items:
            return None
        frame = items[result["name"]][0]
        try:
            depth = decode_depth(result["depth"])
        except (KeyError, TypeError, OSError, ValueError):
            return None
        if depth.shape != (output_height(frame, width), width) or not (depth > 0).any():
            self._say(f"dense depth: frame {frame.index} came back {depth.shape}, unusable")
            return None
        took = result.get("ms")
        return frame, depth, float(took) if isinstance(took, (int, float)) else None
