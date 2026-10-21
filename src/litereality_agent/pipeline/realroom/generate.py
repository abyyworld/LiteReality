"""generate.py — each object's photo turned into a textured mesh, streamed back as it finishes.

The objects stage asks for every object at once and installs each result the moment it lands, so
the room on the call improves object by object instead of all at the end. Two passes, both TRELLIS:

    "preview"  pipeline '512', 512 px texture    ~3 s of H100 per object (upstream's figure)
    "final"    pipeline '1024_cascade', 1024 px  ~17 s (upstream), the one that stays

and per object the final is asked for only once its preview is back, so every object has SOMETHING
early and the GPUs are never idle in between (see `models.trellis.modal.stream_sequences`).

A `Generator` is anything with ``stream(images, passes) -> Iterator[Result]``:

* `ModalTrellisGenerator` — the deployed ``trellis`` function on your Modal workspace (up to 4 H100
  containers in parallel, `deploy/modal/trellis/app.py`), through `models.trellis.modal`.
* `FakeGenerator` — no GPU, no network: a textured box in TRELLIS's own conventions (+Y up, front
  +Z, inside the unit cube) carrying the object's photo, so the whole stage — fitting, the scene,
  the live swap — is exercised offline and in tests exactly as the real results would go through
  it.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Pass:
    """How hard one pass asks TRELLIS to work. ``simplify`` is the transport's legacy ratio: the
    function's face target is ``max(10k, 1e6 * (1 - simplify))``."""

    name: str
    pipeline_type: str
    texture_size: int
    simplify: float
    rank: int                   # a later pass replaces an earlier one, never the other way round


PASSES: dict[str, Pass] = {
    # 20k faces: a whole room of previews stays light on a phone.
    "preview": Pass("preview", "512", 512, 0.98, rank=1),
    # 50k faces, the pipeline's own decimation default.
    "final": Pass("final", "1024_cascade", 1024, 0.95, rank=2),
}
DEFAULT_SEED = 42

# ── what it costs (ESTIMATES: none of these was measured on this deployment) ─────────────────────
CONTAINERS = 4                    # deploy/modal/trellis/app.py MAX_CONTAINERS
SCALEDOWN_SECONDS = 120           # ...and SCALEDOWN_SECONDS: the idle tail every container bills
H100_PER_HOUR = 3.95              # Modal's H100 list price, USD
COLD_START_SECONDS = 120          # image + 16 GB of weights from the Volume + model to the GPU
# GPU seconds per object: upstream's H100 generation time (3 s at 512, 17 s at 1024) plus the GLB
# export (remesh, UV unwrap, texture bake), guessed at several times that.
PASS_SECONDS = {"preview": 20.0, "final": 60.0}


def estimate(objects: int, passes: Sequence[str], *, containers: int = CONTAINERS,
             price_per_hour: float | None = None) -> tuple[float, float]:
    """(minutes until the last result, USD of H100 time) for ``objects`` objects — a planning
    number built from the guesses above, not a measurement."""
    if objects <= 0:
        return 0.0, 0.0
    used = min(containers, objects)
    work = objects * sum(PASS_SECONDS.get(p, 60.0) for p in passes)
    wall = COLD_START_SECONDS + work / used
    gpu = used * (COLD_START_SECONDS + SCALEDOWN_SECONDS) + work
    return wall / 60.0, gpu / 3600.0 * (price_per_hour or H100_PER_HOUR)


@dataclass(frozen=True)
class Result:
    handle: str
    pass_name: str
    glb: bytes | None           # None: this pass failed for this object (see ``error``)
    seconds: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.glb is not None


class Generator(Protocol):
    name: str

    def stream(self, images: Mapping[str, bytes], passes: Sequence[str]) -> Iterator[Result]:
        """Every (object, pass) of ``images`` (handle -> RGBA PNG), yielded as each finishes;
        an object's passes in the order given. Never raises for one object's failure."""
        ...


def check_passes(passes: Sequence[str]) -> list[str]:
    unknown = [p for p in passes if p not in PASSES]
    if unknown or not passes:
        raise ValueError(f"unknown pass {', '.join(unknown) or '(none)'}: "
                         f"expected {' / '.join(PASSES)}")
    return list(passes)


# ── TRELLIS on Modal ─────────────────────────────────────────────────────────────────────────────


class ModalTrellisGenerator:
    """The deployed TRELLIS function, every object in flight at once (`stream_sequences`)."""

    name = "trellis-modal"

    def __init__(self, *, app_name: str = "litereality", function_name: str = "trellis",
                 environment_name: str = "main", profile: str | None = None,
                 credentials: tuple[str, str] | None = None, service: Any = None,
                 seed: int = DEFAULT_SEED, max_in_flight: int | None = None,
                 timeout: float | None = None) -> None:
        from litereality_agent.models.trellis import modal as trellis_modal

        self._transport = trellis_modal
        self.service = service or trellis_modal.ModalTrellisService(
            app_name=app_name, function_name=function_name, environment_name=environment_name,
            profile=profile, credentials=credentials)
        self.seed = seed
        self.max_in_flight = max_in_flight or trellis_modal.MAX_IN_FLIGHT
        self.timeout = timeout or trellis_modal.CALL_TIMEOUT

    def payload(self, image: bytes, spec: Pass, *, request_id: str) -> dict:
        return self._transport.request_payload(
            image, seed=self.seed, simplify=spec.simplify, texture_size=spec.texture_size,
            pipeline_type=spec.pipeline_type, request_id=request_id)

    def stream(self, images: Mapping[str, bytes], passes: Sequence[str]) -> Iterator[Result]:
        passes = check_passes(passes)
        sequences = {handle: [self.payload(png, PASSES[p], request_id=f"{handle}:{p}")
                              for p in passes]
                     for handle, png in images.items()}
        for item in self.service.stream(sequences, max_in_flight=self.max_in_flight,
                                        timeout=self.timeout):
            pass_name = passes[item.index]
            if item.error:
                yield Result(item.key, pass_name, None, item.seconds, item.error)
                continue
            try:
                glb = self._transport.decode_glb(item.output,
                                                 expect_id=f"{item.key}:{pass_name}")
            except ValueError as exc:
                yield Result(item.key, pass_name, None, item.seconds, str(exc))
                continue
            yield Result(item.key, pass_name, glb, item.seconds)


# ── the stand-in ─────────────────────────────────────────────────────────────────────────────────


class FakeGenerator:
    """Textured boxes instead of TRELLIS: offline, instant (or ``delay`` seconds per result, to
    watch versions stream in), deterministic. ``fail`` names objects whose every pass errors."""

    name = "fake"

    def __init__(self, *, delay: float = 0.0, fail: Sequence[str] = ()) -> None:
        self.delay = float(delay)
        self.fail = set(fail)
        self.calls: list[tuple[str, str]] = []

    def stream(self, images: Mapping[str, bytes], passes: Sequence[str]) -> Iterator[Result]:
        for pass_name in check_passes(passes):         # every preview first, as Modal returns them
            for handle, png in images.items():
                started = time.perf_counter()
                if self.delay:
                    time.sleep(self.delay)
                self.calls.append((handle, pass_name))
                if handle in self.fail:
                    yield Result(handle, pass_name, None, self.delay, "fake failure")
                    continue
                yield Result(handle, pass_name, fake_glb(png, pass_name=pass_name),
                             time.perf_counter() - started)


# (outward normal, u axis, v axis) of each face, so u x v = normal and the image is upright on
# every side face: the front is +Z with +X to the viewer's right, as TRELLIS's GLBs face.
_FACES = (((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
          ((0, 0, -1), (-1, 0, 0), (0, 1, 0)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
          ((0, 1, 0), (1, 0, 0), (0, 0, -1)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)))


def fake_glb(image: bytes | None = None, *, pass_name: str = "final",
             extents: Sequence[float] = (0.9, 0.9, 0.9)) -> bytes:
    """A GLB the way TRELLIS writes one — one mesh, +Y up, front +Z, inside [-0.5, 0.5]^3, a PBR
    base-colour texture — of a box wearing ``image`` (an RGBA photo crop) on every face. The
    preview is a smaller box with a blurred texture, so a swap from preview to final shows."""
    import trimesh
    from PIL import Image

    if image:
        with Image.open(io.BytesIO(image)) as im:
            rgba = im.convert("RGBA")
        arr = np.asarray(rgba)
        alpha = arr[..., 3] > 127
        mean = (arr[..., :3][alpha].mean(axis=0) if alpha.any()
                else arr[..., :3].reshape(-1, 3).mean(axis=0))
        flat = Image.new("RGB", rgba.size, tuple(int(c) for c in mean))
        flat.paste(rgba.convert("RGB"), mask=rgba.split()[3])
        texture = flat
    else:
        texture = Image.new("RGB", (64, 64), (150, 150, 150))
    preview = pass_name == "preview"
    side = 32 if preview else 512
    texture.thumbnail((side, side))
    half = np.asarray(extents, dtype=float) / 2.0 * (0.85 if preview else 1.0)
    verts, uvs, faces = [], [], []
    for normal, u, v in _FACES:
        n, u, v = (np.asarray(a, dtype=float) for a in (normal, u, v))
        c, du, dv = n * half, u * half, v * half
        base = len(verts)
        verts += [c - du - dv, c + du - dv, c + du + dv, c - du + dv]
        uvs += [(0, 0), (1, 0), (1, 1), (0, 1)]       # trimesh's (OBJ) convention: v up
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=texture, metallicFactor=0.0,
                                                   roughnessFactor=0.8)
    mesh = trimesh.Trimesh(np.array(verts), np.array(faces), process=False,
                           visual=trimesh.visual.TextureVisuals(uv=np.array(uvs, dtype=float),
                                                                material=material))
    return mesh.export(file_type="glb")
