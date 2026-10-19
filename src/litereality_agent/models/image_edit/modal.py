"""Qwen-Image-Edit-2509 hosted on the caller's own Modal workspace.

The reference step's third backend (``LR_IMAGE_PROVIDER=modal``): it needs no OpenAI or Gemini key,
and spends Modal credit instead. Every failure raises — the reference callers catch it and fall back
to the raw evidence sheet, which is only safe if a failed call can never come back looking like a
render.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

from litereality_agent.models.image_edit.inference import (
    DEFAULT_MODEL,
    DEFAULT_STEPS,
    DEFAULT_TRUE_CFG,
    parse_request,
)
from litereality_agent.runtimes.modal import ModalClient

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ModalImageEditService:
    """Image-edit adapter for the deployed Modal ``image_edit`` function."""

    name = "image-edit-modal"

    def __init__(
        self,
        *,
        app_name: str,
        function_name: str = "image_edit",
        environment_name: str = "main",
        profile: str | None = None,
        credentials: tuple[str, str] | None = None,
        client: Any = None,
    ) -> None:
        self.app_name = app_name
        self.function_name = function_name
        self.client = client or ModalClient(
            app_name,
            function_name,
            environment_name=environment_name,
            profile=profile,
            credentials=credentials,
        )

    def edit(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        steps: int = DEFAULT_STEPS,
        true_cfg_scale: float = DEFAULT_TRUE_CFG,
        seed: int = 0,
    ) -> tuple[bytes, dict]:
        """Edit one image remotely. Returns ``(png_bytes, response_meta)``; raises on any failure.

        There is deliberately no client-side timeout or retry here. The function runs one container
        (``max_containers=1``), so a burst of reference calls queues on Modal and the last one can
        wait many minutes behind the others plus a cold start; a timeout would abandon work that
        is still queued or running and has to be paid for anyway, and a retry would bill the GPU
        twice for the same reference.

        ``seed=0`` is only a library default: the reference step always passes its own, fresh per
        call, because the same sheet, prompt and seed render the same image again.
        """
        request = {
            "image_b64": base64.b64encode(bytes(image_bytes)).decode("ascii"),
            "prompt": prompt,
            "model": model,
            "steps": steps,
            "true_cfg_scale": true_cfg_scale,
            "seed": seed,
        }
        # The container validates too; doing it here first means a bad request costs nothing.
        parse_request(request)

        where = f"Modal {self.app_name}/{self.function_name}"
        output = self.client.map([request])[0]
        if isinstance(output, BaseException):
            raise RuntimeError(f"{where} failed: {type(output).__name__}: {output}") from output
        if not isinstance(output, dict):
            raise RuntimeError(f"{where} returned {type(output).__name__}, expected a dict")
        if output.get("error"):
            raise RuntimeError(f"{where} failed: {output['error']}")
        encoded = output.get("png_b64")
        if not encoded:
            raise RuntimeError(f"{where} returned no png_b64")
        try:
            png = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RuntimeError(f"{where} returned invalid base64: {exc}") from exc
        if not png.startswith(_PNG_SIGNATURE):
            raise RuntimeError(f"{where} returned bytes that are not a PNG")
        meta = {key: value for key, value in output.items() if key != "png_b64"}
        return png, meta
