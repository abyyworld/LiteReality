"""LR_IMAGE_PROVIDER=modal: Qwen-Image-Edit-2509 on the user's own Modal workspace.

Offline throughout — no torch, no diffusers, no network, no Modal. The contract that matters is the
one the reference callers rely on: a successful call writes a real PNG, and EVERY failure raises,
because `object_references` turns a raise into a raw-sheet fallback and would ship anything that
came back looking like success. The cost contract matters as much: a Modal attempt is billed H100
time, so nothing here may retry it.
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from litereality_agent.models.image_edit import inference
from litereality_agent.pipeline.scene_init.ingest.references import image_gen
from litereality_agent.settings import LiteRealitySettings

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "modal"
QWEN = "Qwen/Qwen-Image-Edit-2509"
ENV_KEYS = (
    "LR_IMAGE_PROVIDER",
    "LR_OPENAI_IMAGE_MODEL",
    "LR_GEMINI_IMAGE_MODEL",
    "LR_MODAL_IMAGE_MODEL",
    "LR_MODAL_IMAGE_STEPS",
    "LR_MODAL_IMAGE_CFG",
    "LR_MODAL_IMAGE_SEED",
    "LR_MODAL_GPU_PRICE_PER_HOUR",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    # A shell set up for hand deploys (deploy/modal/README.md) exports these. Left in, `_settings()`
    # would read them, and a test meant to fail on missing credentials would build a real
    # ModalClient and send a billed H100 job instead.
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_PROFILE",
    "MODAL_ENVIRONMENT",
    "MODAL_IMAGE_APP",
    "MODAL_IMAGE_FUNCTION",
)


def _png(size: tuple[int, int] = (4, 4), color=(0, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _settings(**overrides) -> LiteRealitySettings:
    """Real settings, isolated from the developer's own .env and models.env."""
    return LiteRealitySettings(_env_file=None, **overrides)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """No provider, model or key leaks in from the shell, and no cached service between tests."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    image_gen._build_modal_image_service.cache_clear()
    inference._PIPELINES.clear()
    yield
    image_gen._build_modal_image_service.cache_clear()
    inference._PIPELINES.clear()


class FakeClient:
    def __init__(self, outputs):
        self.outputs = outputs
        self.inputs = None

    def map(self, inputs):
        self.inputs = list(inputs)
        return self.outputs


# --- provider selection ------------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "explicit,model,expected",
    [
        ("modal", "gpt-image-2", "modal"),  # explicit wins over the CLI's default model name
        (" Modal ", None, "modal"),
        (None, QWEN, "modal"),  # --image-model Qwen/... alone is enough
        (None, "qwen/qwen-image-edit-2509", "modal"),
        (None, "gemini-2.5-flash-image", "gemini"),
        (None, "gpt-image-2", "openai"),
        (None, None, "openai"),
        ("openai", QWEN, "openai"),  # an explicit provider is never second-guessed
        ("bogus", QWEN, "modal"),  # an unknown value falls back to the model name
    ],
)
def test_resolve_provider(monkeypatch, explicit, model, expected):
    if explicit is not None:
        monkeypatch.setenv("LR_IMAGE_PROVIDER", explicit)
    assert image_gen.resolve_provider(model) == expected


@pytest.mark.parametrize(
    "env,model,expected",
    [
        ({"LR_IMAGE_PROVIDER": "modal"}, "gpt-image-2", ("modal", QWEN)),
        ({}, "Qwen/Other", ("modal", "Qwen/Other")),
        ({"LR_MODAL_IMAGE_MODEL": "Qwen/Pinned"}, QWEN, ("modal", "Qwen/Pinned")),
        ({}, "gemini-x", ("gemini", "gemini-x")),
        ({"LR_IMAGE_PROVIDER": "gemini"}, "gpt-image-2",
         ("gemini", image_gen.DEFAULT_GEMINI_IMAGE_MODEL)),
        ({"LR_OPENAI_IMAGE_MODEL": "gpt-image-2"}, None, ("openai", "gpt-image-2")),
        ({}, None, ("openai", image_gen.DEFAULT_IMAGE_MODEL)),
    ],
)
def test_resolve_model_is_what_the_reference_label_records(monkeypatch, env, model, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert image_gen.resolve_model(model) == expected


def test_models_env_leaves_the_provider_to_the_model_name():
    """A non-empty LR_IMAGE_PROVIDER default in models.env pinned openai for every run, so
    `--image-model Qwen/...` (or `gemini-*`) was silently sent to OpenAI."""
    settings = LiteRealitySettings(_env_file=(REPO / "models.env",))

    assert settings.image_provider is None
    assert "LR_IMAGE_PROVIDER" not in settings.as_environment()
    assert settings.modal_image_model == QWEN
    assert settings.modal_image_steps == inference.DEFAULT_STEPS
    assert settings.modal_image_cfg == inference.DEFAULT_TRUE_CFG


# --- request contract -------------------------------------------------------------------------- #
def test_parse_request_applies_defaults():
    parsed = inference.parse_request({"image_b64": _b64(b"sheet"), "prompt": "a chair"})

    assert parsed == inference.EditRequest(
        image=b"sheet",
        prompt="a chair",
        model=QWEN,
        steps=inference.DEFAULT_STEPS,
        true_cfg_scale=inference.DEFAULT_TRUE_CFG,
        seed=0,
    )


def test_parse_request_keeps_explicit_values():
    parsed = inference.parse_request(
        {
            "image_b64": _b64(b"sheet"),
            "prompt": "a chair",
            "model": "Qwen/Other",
            "steps": 20,
            "true_cfg_scale": 1,
            "seed": 7,
        }
    )

    assert (parsed.model, parsed.steps, parsed.true_cfg_scale, parsed.seed) == (
        "Qwen/Other",
        20,
        1.0,
        7,
    )
    assert isinstance(parsed.true_cfg_scale, float)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"image_b64": None}, "needs image_b64"),
        ({"image_b64": ""}, "needs image_b64"),
        ({"image_b64": 123}, "needs image_b64"),
        ({"image_b64": "not base64!!"}, "not valid base64"),
        ({"prompt": ""}, "non-empty prompt"),
        ({"prompt": "   "}, "non-empty prompt"),
        ({"prompt": None}, "non-empty prompt"),
        ({"model": ""}, "model"),
        ({"steps": 0}, "steps"),
        ({"steps": 101}, "steps"),
        ({"steps": 1.5}, "steps"),
        ({"steps": True}, "steps"),
        ({"true_cfg_scale": 0.5}, "true_cfg_scale"),
        ({"true_cfg_scale": 20.5}, "true_cfg_scale"),
        ({"true_cfg_scale": math.nan}, "true_cfg_scale"),
        ({"true_cfg_scale": "4"}, "true_cfg_scale"),
        ({"true_cfg_scale": True}, "true_cfg_scale"),
        ({"seed": -1}, "seed"),
        ({"seed": inference.MAX_SEED + 1}, "seed"),
        ({"seed": "0"}, "seed"),
    ],
)
def test_parse_request_rejects_bad_input(change, message):
    request = {"image_b64": _b64(b"sheet"), "prompt": "a chair", **change}
    with pytest.raises(ValueError, match=message):
        inference.parse_request(request)


def test_parse_request_rejects_a_non_dict():
    with pytest.raises(ValueError, match="must be a dict"):
        inference.parse_request(["image"])


def test_inference_and_client_import_without_torch_or_diffusers():
    """The client imports the request contract on machines that have neither."""
    code = (
        "import sys\n"
        "import litereality_agent.models.image_edit.inference\n"
        "import litereality_agent.models.image_edit.modal\n"
        "heavy = {'torch', 'diffusers', 'transformers', 'modal'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


# --- container-side inference (with fake torch/diffusers) --------------------------------------- #
class _Generator:
    def __init__(self, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


def _fake_torch(monkeypatch, calls: list | None = None) -> ModuleType:
    calls = [] if calls is None else calls
    torch = ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.Generator = _Generator
    torch.cuda = SimpleNamespace(empty_cache=lambda: calls.append(("empty_cache",)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_edit_calls_the_pipeline_with_the_model_card_settings(monkeypatch):
    _fake_torch(monkeypatch)
    seen = {}

    def pipeline(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), (10, 20, 30))])

    png = inference.edit(pipeline, _png((16, 16), (255, 0, 0)), "a chair", steps=12,
                         true_cfg_scale=4.0, seed=5)

    assert png.startswith(b"\x89PNG")
    assert Image.open(io.BytesIO(png)).size == (8, 8)
    assert [image.mode for image in seen["image"]] == ["RGB"]
    assert seen["prompt"] == "a chair"
    # A negative prompt is what switches true CFG on; " " is the model card's.
    assert seen["negative_prompt"] == " "
    assert (seen["true_cfg_scale"], seen["num_inference_steps"]) == (4.0, 12)
    assert (seen["guidance_scale"], seen["num_images_per_prompt"]) == (1.0, 1)
    assert (seen["generator"].device, seen["generator"].seed) == ("cuda", 5)


def test_edit_without_true_cfg_sends_no_negative_prompt(monkeypatch):
    _fake_torch(monkeypatch)
    seen = {}

    def pipeline(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (4, 4))])

    inference.edit(pipeline, _png(), "a chair", steps=4, true_cfg_scale=1.0, seed=0)

    assert "negative_prompt" not in seen


def test_load_pipeline_caches_and_evicts_another_model(monkeypatch):
    calls = []
    _fake_torch(monkeypatch, calls)

    class Pipeline:
        def __init__(self, model_id):
            self.model_id = model_id

        @classmethod
        def from_pretrained(cls, model_id, *, torch_dtype):
            calls.append(("load", model_id, torch_dtype))
            return cls(model_id)

        def to(self, device):
            calls.append(("to", device))
            return self

        def set_progress_bar_config(self, **options):
            calls.append(("progress", options))

    diffusers = ModuleType("diffusers")
    diffusers.QwenImageEditPlusPipeline = Pipeline
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    first = inference.load_pipeline(QWEN)
    assert inference.load_pipeline(QWEN) is first
    assert calls == [("load", QWEN, "bf16"), ("to", "cuda"), ("progress", {"disable": True})]

    other = inference.load_pipeline("Qwen/Other")
    assert other.model_id == "Qwen/Other"
    assert ("empty_cache",) in calls
    assert list(inference._PIPELINES) == ["Qwen/Other"]  # two ~58 GB pipelines never co-reside


def test_hosted_handler_reports_model_steps_and_gpu_time(monkeypatch):
    loaded, edits = [], []
    pipeline = object()

    def fake_edit(pipe, image, prompt, **options):
        edits.append((pipe, image, prompt, options))
        return _png()

    monkeypatch.setattr(inference, "edit", fake_edit)

    response = inference.hosted_handler(
        {"image_b64": _b64(b"sheet"), "prompt": "a chair", "steps": 8},
        loader=lambda model_id: loaded.append(model_id) or pipeline,
    )

    assert loaded == [QWEN]
    assert edits == [
        (pipeline, b"sheet", "a chair", {"steps": 8, "true_cfg_scale": 4.0, "seed": 0})
    ]
    assert base64.b64decode(response["png_b64"]) == _png()
    assert (response["model"], response["steps"], response["true_cfg_scale"]) == (QWEN, 8, 4.0)
    assert response["elapsed_s"] >= response["load_s"] >= 0


def test_hosted_handler_rejects_bad_input_before_loading_the_model():
    with pytest.raises(ValueError, match="prompt"):
        inference.hosted_handler(
            {"image_b64": _b64(b"sheet"), "prompt": ""},
            loader=lambda model_id: pytest.fail("a bad request must not load 58 GB of weights"),
        )


# --- client ------------------------------------------------------------------------------------ #
def test_service_sends_the_request_and_decodes_the_png():
    from litereality_agent.models.image_edit.modal import ModalImageEditService

    png = _png()
    client = FakeClient([{"png_b64": _b64(png), "model": QWEN, "steps": 20, "elapsed_s": 42.0}])
    service = ModalImageEditService(app_name="unused", client=client)

    out, meta = service.edit(b"sheet", "a chair", model=QWEN, steps=20, true_cfg_scale=3.0, seed=9)

    assert client.inputs == [
        {
            "image_b64": _b64(b"sheet"),
            "prompt": "a chair",
            "model": QWEN,
            "steps": 20,
            "true_cfg_scale": 3.0,
            "seed": 9,
        }
    ]
    assert out == png
    assert meta == {"model": QWEN, "steps": 20, "elapsed_s": 42.0}


def test_service_raises_a_remote_exception():
    from litereality_agent.models.image_edit.modal import ModalImageEditService

    remote = MemoryError("CUDA out of memory")
    service = ModalImageEditService(app_name="litereality", client=FakeClient([remote]))

    with pytest.raises(RuntimeError, match="CUDA out of memory") as caught:
        service.edit(b"sheet", "a chair")
    assert caught.value.__cause__ is remote


@pytest.mark.parametrize(
    "output,message",
    [
        ({"model": QWEN}, "no png_b64"),
        ({"png_b64": ""}, "no png_b64"),
        ({"png_b64": "%%%"}, "invalid base64"),
        ({"png_b64": _b64(b"GIF89a not a png")}, "not a PNG"),
        ({"error": "generation failed"}, "generation failed"),
        ("not a dict", "expected a dict"),
    ],
)
def test_service_never_returns_junk(output, message):
    from litereality_agent.models.image_edit.modal import ModalImageEditService

    service = ModalImageEditService(app_name="unused", client=FakeClient([output]))
    with pytest.raises(RuntimeError, match=message):
        service.edit(b"sheet", "a chair")


def test_service_validates_before_waking_a_gpu():
    from litereality_agent.models.image_edit.modal import ModalImageEditService

    client = FakeClient([])
    service = ModalImageEditService(app_name="unused", client=client)

    with pytest.raises(ValueError, match="steps"):
        service.edit(b"sheet", "a chair", steps=0)
    assert client.inputs is None


# --- registry and settings --------------------------------------------------------------------- #
def test_registry_refuses_without_modal_credentials():
    from litereality_agent.models import registry

    with pytest.raises(RuntimeError, match="MODAL_TOKEN_ID") as caught:
        registry.image_edit_from_settings(_settings())
    assert "litereality setup" in str(caught.value)


def test_registry_builds_the_modal_service_from_settings(monkeypatch):
    from litereality_agent.models import registry

    captured = {}

    class Service:
        def __init__(self, **options):
            captured.update(options)

    monkeypatch.setattr("litereality_agent.models.image_edit.modal.ModalImageEditService", Service)
    settings = _settings(modal_token_id="ak-1", modal_token_secret="as-2")

    assert isinstance(registry.image_edit_from_settings(settings), Service)
    assert captured == {
        "app_name": "litereality",
        "function_name": "image_edit",
        "environment_name": "main",
        "profile": None,
        "credentials": ("ak-1", "as-2"),
    }


def test_registry_accepts_a_profile_alone(monkeypatch):
    from litereality_agent.models import registry
    from litereality_agent.models.image_edit.modal import ModalImageEditService

    monkeypatch.setattr(
        "litereality_agent.models.image_edit.modal.ModalClient",
        lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
    )
    service = registry.image_edit_from_settings(_settings(modal_profile="team-a"))

    assert isinstance(service, ModalImageEditService)
    assert service.client.args == ("litereality", "image_edit")
    assert service.client.kwargs["profile"] == "team-a"


def test_settings_export_the_image_edit_names_and_knobs():
    env = _settings(modal_image_steps=20, modal_image_cfg=1.0).as_environment()

    assert env["MODAL_IMAGE_APP"] == "litereality"
    assert env["MODAL_IMAGE_FUNCTION"] == "image_edit"
    assert (env["LR_MODAL_IMAGE_STEPS"], env["LR_MODAL_IMAGE_CFG"]) == ("20", "1.0")
    assert "LR_MODAL_IMAGE_MODEL" not in env  # unset stays unset; image_gen supplies the default


@pytest.mark.parametrize(
    "field,value",
    [
        ("modal_image_steps", 0),
        ("modal_image_cfg", 0.5),
        ("modal_image_seed", -1),
        ("modal_image_seed", inference.MAX_SEED + 1),
        ("modal_gpu_price_per_hour", 0),
    ],
)
def test_settings_reject_out_of_range_knobs(field, value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _settings(**{field: value})


def test_settings_export_the_seed_and_gpu_price_to_the_subprocess():
    """Only fields reach the generating subprocess, so a `.env` line with none behind it is dead."""
    env = _settings(modal_image_seed=inference.MAX_SEED, modal_gpu_price_per_hour=2.0)
    exported = env.as_environment()

    assert exported["LR_MODAL_IMAGE_SEED"] == str(inference.MAX_SEED)
    assert exported["LR_MODAL_GPU_PRICE_PER_HOUR"] == "2.0"
    unset = _settings().as_environment()
    assert "LR_MODAL_IMAGE_SEED" not in unset  # unset draws a fresh seed per reference
    assert "LR_MODAL_GPU_PRICE_PER_HOUR" not in unset


def test_image_edit_function_is_deployed_to_bound_spend():
    """One container, no retries: the two settings that keep a free-tier bill bounded."""
    tree = ast.parse((DEPLOY / "image_edit" / "app.py").read_text(encoding="utf-8"))
    options = {}
    for node in tree.body:
        for decorator in getattr(node, "decorator_list", []):
            func = getattr(decorator, "func", None)
            if isinstance(decorator, ast.Call) and getattr(func, "attr", "") == "function":
                options = {keyword.arg: keyword.value for keyword in decorator.keywords}
    assert isinstance(options["max_containers"], ast.Constant)
    assert options["max_containers"].value == 1
    assert options["retries"].value == 0
    assert options["gpu"].value == "H100"
    assert (DEPLOY / "image_edit" / "__init__.py").exists()  # importable as `image_edit.app`


# --- reference generation end to end ------------------------------------------------------------ #
class FakeService:
    def __init__(self, png: bytes | None = None, error: Exception | None = None):
        self.png = png if png is not None else _png((8, 8))
        self.error = error
        self.calls = []

    def edit(self, image_bytes, prompt, *, model, steps, true_cfg_scale, seed=0):
        self.calls.append(
            {"image": image_bytes, "prompt": prompt, "model": model, "steps": steps,
             "true_cfg_scale": true_cfg_scale, "seed": seed}
        )
        if self.error is not None:
            raise self.error
        meta = {"model": model, "steps": steps, "true_cfg_scale": true_cfg_scale,
                "seed": seed, "load_s": 0.0, "elapsed_s": 60.0}
        return self.png, meta


@pytest.fixture
def sheet(tmp_path) -> Path:
    path = tmp_path / "input2imagegen.jpg"
    Image.new("RGB", (16, 16), (200, 100, 50)).save(path, format="JPEG")
    return path


def test_generate_reference_on_modal_needs_no_image_key(tmp_path, sheet, monkeypatch):
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "modal")
    service = FakeService()
    monkeypatch.setattr(image_gen, "_modal_image_service", lambda: service)
    out = tmp_path / "obj" / "clean_obj_reference.png"

    # The CLI always passes its default OpenAI model name; with provider modal it is ignored.
    meta = image_gen.generate_reference(
        sheet, "a chair", out, model="gpt-image-2", response_dir=tmp_path / "_responses"
    )

    assert meta["status"] == "ok"
    assert meta["model"] == "modal:Qwen/Qwen-Image-Edit-2509"
    assert meta["usage"] == {"gpu_seconds": 60.0, "steps": inference.DEFAULT_STEPS}
    assert meta["cost_usd"] == round(60.0 * image_gen._PRICE_MODAL_GPU_HOUR / 3600, 5)
    assert out.read_bytes() == service.png  # written as-is: the model renders onto black itself
    [call] = service.calls
    assert call["image"] == sheet.read_bytes()
    assert call["model"] == QWEN
    assert (call["steps"], call["true_cfg_scale"]) == (40, 4.0)
    assert call["prompt"].startswith("a chair")
    assert "SOLID PURE BLACK" in call["prompt"]
    response = json.loads((tmp_path / "_responses" / "modal_response_0.json").read_text())
    assert response["model"] == QWEN
    assert response["usage"]["gpu_seconds"] == 60.0


def test_generate_reference_infers_modal_from_a_qwen_model(tmp_path, sheet, monkeypatch):
    monkeypatch.setenv("LR_MODAL_IMAGE_STEPS", "20")
    monkeypatch.setenv("LR_MODAL_IMAGE_CFG", "1")
    service = FakeService()
    monkeypatch.setattr(image_gen, "_modal_image_service", lambda: service)

    meta = image_gen.generate_reference(sheet, "a chair", tmp_path / "out.png", model="Qwen/Other")

    assert meta["model"] == "modal:Qwen/Other"
    assert (service.calls[0]["steps"], service.calls[0]["true_cfg_scale"]) == (20, 1.0)


def test_generate_reference_prefers_the_configured_modal_model(tmp_path, sheet, monkeypatch):
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "modal")
    monkeypatch.setenv("LR_MODAL_IMAGE_MODEL", "Qwen/Pinned")
    service = FakeService()
    monkeypatch.setattr(image_gen, "_modal_image_service", lambda: service)

    meta = image_gen.generate_reference(sheet, "a chair", tmp_path / "out.png", model=QWEN)

    assert meta["model"] == "modal:Qwen/Pinned"


def test_a_failed_modal_call_raises_after_exactly_one_attempt(tmp_path, sheet, monkeypatch):
    """Every attempt is billed H100 time; a retry would re-bill the same OOM or timeout."""
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "modal")
    slept: list[float] = []
    monkeypatch.setattr(image_gen.time, "sleep", slept.append)
    service = FakeService(error=RuntimeError("Modal litereality/image_edit failed: timeout"))
    monkeypatch.setattr(image_gen, "_modal_image_service", lambda: service)
    out = tmp_path / "out.png"

    with pytest.raises(RuntimeError, match="modal image returned nothing after 1 tries"):
        image_gen.generate_reference(sheet, "a chair", out, retries=3)

    assert len(service.calls) == 1
    assert slept == []
    assert not out.exists()


def test_missing_modal_credentials_surface_as_a_raise(tmp_path, sheet, monkeypatch):
    """No silent success: the registry's explanation reaches the caller's fallback record."""
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "modal")
    monkeypatch.setattr("litereality_agent.models.registry.load_settings", _settings)
    # Belt and braces: however the shell is configured, this test must never reach Modal.
    monkeypatch.setattr(
        "litereality_agent.models.image_edit.modal.ModalClient",
        lambda *args, **kwargs: pytest.fail("a unit test built a real Modal client"),
    )

    with pytest.raises(RuntimeError, match="MODAL_TOKEN_ID"):
        image_gen.generate_reference(sheet, "a chair", tmp_path / "out.png")


def test_forced_regenerations_on_modal_draw_a_fresh_seed(tmp_path, sheet, monkeypatch):
    """Chair QC's repair loop and --force-image-generation regenerate to get a DIFFERENT image.
    With a fixed seed the same sheet and prompt render the same picture, so each attempt billed
    H100 time for the reference it was trying to replace."""
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "modal")
    draws = iter([11, 22])
    bounds = []
    monkeypatch.setattr(
        image_gen.secrets, "randbelow", lambda bound: bounds.append(bound) or next(draws)
    )
    service = FakeService()
    monkeypatch.setattr(image_gen, "_modal_image_service", lambda: service)
    out = tmp_path / "out.png"
    responses = tmp_path / "_responses"

    for _ in range(2):
        image_gen.generate_reference(sheet, "a chair", out, force=True, response_dir=responses)

    assert [call["seed"] for call in service.calls] == [11, 22]
    assert bounds == [inference.MAX_SEED + 1] * 2  # the whole range the container accepts
    # The seed of each render is kept, so a good one can be reproduced.
    assert json.loads((responses / "modal_response_0.json").read_text())["seed"] == 22


def test_unpatched_seeds_differ_between_calls(tmp_path, sheet, monkeypatch):
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "modal")
    service = FakeService()
    monkeypatch.setattr(image_gen, "_modal_image_service", lambda: service)

    for _ in range(2):
        image_gen.generate_reference(sheet, "a chair", tmp_path / "out.png", force=True)

    first, second = (call["seed"] for call in service.calls)
    assert first != second
    assert all(0 <= call["seed"] <= inference.MAX_SEED for call in service.calls)


def test_a_pinned_modal_seed_is_forwarded(tmp_path, sheet, monkeypatch):
    monkeypatch.setenv("LR_IMAGE_PROVIDER", "modal")
    monkeypatch.setenv("LR_MODAL_IMAGE_SEED", "7")
    monkeypatch.setattr(
        image_gen.secrets, "randbelow", lambda bound: pytest.fail("a pinned seed is not drawn")
    )
    service = FakeService()
    monkeypatch.setattr(image_gen, "_modal_image_service", lambda: service)
    responses = tmp_path / "_responses"

    image_gen.generate_reference(sheet, "a chair", tmp_path / "out.png", response_dir=responses)

    assert service.calls[0]["seed"] == 7
    assert json.loads((responses / "modal_response_0.json").read_text())["seed"] == 7


def test_the_modal_service_is_built_once_across_threads(monkeypatch):
    from litereality_agent.models import registry

    built = []
    gate = threading.Barrier(8)

    def build():
        built.append(1)
        return object()

    monkeypatch.setattr(registry, "image_edit_from_settings", build)
    results = []

    def worker():
        gate.wait()
        results.append(image_gen._modal_image_service())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(built) == 1
    assert len({id(result) for result in results}) == 1


def test_a_failed_service_build_is_not_cached(monkeypatch):
    from litereality_agent.models import registry

    outcomes = iter([RuntimeError("no credentials"), "service"])

    def build():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(registry, "image_edit_from_settings", build)

    with pytest.raises(RuntimeError, match="no credentials"):
        image_gen._modal_image_service()
    assert image_gen._modal_image_service() == "service"


# --- scene manifest ---------------------------------------------------------------------------- #
def _reference_meta(refroot: Path, kind: str, name: str, status: str, label: str | None = None):
    out = refroot / kind / "scan" / name
    out.mkdir(parents=True)
    record = {"status": status, **({"image_meta": {"model": label}} if label else {})}
    (out / "reference_meta.json").write_text(json.dumps(record), encoding="utf-8")


def test_manifest_records_the_image_provider_that_actually_ran(tmp_path, monkeypatch):
    """The manifest said openai/gpt-image-2 (always exported) for a scene made on Qwen."""
    from litereality_agent.pipeline.scene_init import artifacts

    monkeypatch.setenv("LR_OPENAI_IMAGE_MODEL", "gpt-image-2")
    _reference_meta(tmp_path, "object_refs", "Table0", "ok", f"modal:{QWEN}")
    _reference_meta(tmp_path, "opening_refs", "Wall0_Door_0", "ok", f"modal:{QWEN}")
    _reference_meta(tmp_path, "object_refs", "Lamp0", "error_fallback")

    assert artifacts._image_record(tmp_path, "scan") == ("modal", QWEN)


def test_manifest_lists_every_backend_a_run_mixed(tmp_path):
    from litereality_agent.pipeline.scene_init import artifacts

    _reference_meta(tmp_path, "object_refs", "Table0", "ok", f"modal:{QWEN}")
    _reference_meta(tmp_path, "object_refs", "Lamp0", "ok", "gemini:gemini-2.5-flash-image")

    assert artifacts._image_record(tmp_path, "scan") == (
        ["gemini", "modal"],
        ["Qwen/Qwen-Image-Edit-2509", "gemini-2.5-flash-image"],
    )


def test_manifest_falls_back_to_the_configured_provider(tmp_path, monkeypatch):
    """References reused from disk carry no label; the configured provider made them."""
    from litereality_agent.pipeline.scene_init import artifacts

    monkeypatch.setenv("LR_IMAGE_PROVIDER", "modal")
    assert artifacts._image_record(tmp_path, "scan") == ("modal", QWEN)  # nothing on disk at all

    _reference_meta(tmp_path, "object_refs", "Table0", "ok")
    assert artifacts._image_record(tmp_path, "scan") == ("modal", QWEN)


def test_manifest_credits_no_provider_when_nothing_was_generated(tmp_path, monkeypatch):
    from litereality_agent.pipeline.scene_init import artifacts

    monkeypatch.setenv("LR_IMAGE_PROVIDER", "modal")
    _reference_meta(tmp_path, "object_refs", "Table0", "error_fallback")
    _reference_meta(tmp_path, "opening_refs", "Wall0_Door_0", "skipped_image_generation")
    assert artifacts._image_record(tmp_path, "scan") == ("none", None)

    # A chair cluster that did get its image (chairs record no label) means one provider ran.
    clusters = tmp_path / "chair_clusters" / "scan"
    clusters.mkdir(parents=True)
    (clusters / "chair_clusters.json").write_text(
        json.dumps({"clusters": [{"reference_status": "ok"}]}), encoding="utf-8"
    )
    assert artifacts._image_record(tmp_path, "scan") == ("modal", QWEN)


# --- sanity ------------------------------------------------------------------------------------ #
@pytest.fixture
def sanity():
    """A fresh copy of sanity.py per test, so its FAILS/WARNS lists start empty."""
    spec = importlib.util.spec_from_file_location("lr_sanity_under_test", REPO / "sanity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROVIDER_CASES = [
    {"LR_IMAGE_PROVIDER": "modal"},
    {"LR_IMAGE_PROVIDER": "modal", "LR_MODAL_IMAGE_MODEL": "Qwen/Pinned"},
    {"LR_OPENAI_IMAGE_MODEL": QWEN},
    {"LR_OPENAI_IMAGE_MODEL": "gemini-2.5-flash-image"},
    {"LR_IMAGE_PROVIDER": "gemini", "LR_GEMINI_IMAGE_MODEL": "gemini-x"},
    {"LR_IMAGE_PROVIDER": "bogus"},
    {},
]


@pytest.mark.parametrize("env", PROVIDER_CASES)
@pytest.mark.parametrize("installed", [True, False])
def test_sanity_checks_the_provider_a_run_will_use(sanity, monkeypatch, env, installed):
    """Both sanity's call into image_gen and its inline half-installed fallback agree with it."""
    monkeypatch.setenv("LR_OPENAI_IMAGE_MODEL", "gpt-image-2")  # settings always export it
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    expected = image_gen.resolve_model(os.environ["LR_OPENAI_IMAGE_MODEL"])
    if not installed:
        monkeypatch.setitem(sys.modules, image_gen.__name__, None)  # the import now raises

    assert sanity._image_provider() == expected


def test_sanity_lets_an_empty_shell_var_fall_through_to_dotenv(sanity, monkeypatch):
    """`export LR_IMAGE_PROVIDER=` must not hide `.env`'s modal: the run would still use it."""
    import litereality_agent.settings as settings_module

    settings = _settings(image_provider="modal")
    monkeypatch.setattr(settings_module, "load_settings", lambda root=None: settings)
    environ = {"LR_IMAGE_PROVIDER": "", "LR_OPENAI_IMAGE_MODEL": "from-the-shell"}
    monkeypatch.setattr(os, "environ", environ)

    sanity._load_dotenv()

    assert environ["LR_IMAGE_PROVIDER"] == "modal"
    assert environ["LR_OPENAI_IMAGE_MODEL"] == "from-the-shell"  # a real shell value still wins


class NotFoundError(Exception):
    """Stands in for modal.exception.NotFoundError, which sanity matches by name."""


def _hosted_image_edit(monkeypatch, hydrate):
    import litereality_agent.settings as settings_module
    from litereality_agent.models import registry

    monkeypatch.setattr(
        settings_module, "load_settings", lambda root=None: _settings(modal_profile="team-a")
    )
    function = SimpleNamespace(hydrate=hydrate)
    service = SimpleNamespace(
        app_name="litereality",
        function_name="image_edit",
        client=SimpleNamespace(function=function),
    )
    monkeypatch.setattr(registry, "image_edit_from_settings", lambda settings: service)


def test_sanity_fails_an_image_edit_function_that_is_not_deployed(sanity, monkeypatch):
    """The workspace deployed before image_edit existed: every reference would fall back."""

    def hydrate():
        raise NotFoundError("Lookup failed for Function 'image_edit' from the 'litereality' app")

    _hosted_image_edit(monkeypatch, hydrate)
    sanity._check_hosted_image_edit(QWEN)

    [failure] = sanity.FAILS
    assert "litereality/image_edit is not deployed" in failure
    assert any("uv run litereality setup" in command for command, _ in sanity.CMD_FIXES)


def test_sanity_passes_a_deployed_image_edit_function(sanity, monkeypatch):
    _hosted_image_edit(monkeypatch, lambda: None)
    sanity._check_hosted_image_edit(QWEN)

    assert sanity.FAILS == [] and sanity.WARNS == []


def test_sanity_only_warns_when_modal_cannot_be_reached(sanity, monkeypatch):
    def hydrate():
        raise ConnectionError("Could not connect to the Modal server.")

    _hosted_image_edit(monkeypatch, hydrate)
    sanity._check_hosted_image_edit(QWEN)

    assert sanity.FAILS == []
    assert len(sanity.WARNS) == 1


def test_sanity_fails_modal_without_credentials(sanity, monkeypatch):
    import litereality_agent.settings as settings_module
    from litereality_agent.models import registry

    monkeypatch.setattr(settings_module, "load_settings", lambda root=None: _settings())
    monkeypatch.setattr(
        registry, "image_edit_from_settings", lambda settings: pytest.fail("no service to build")
    )
    sanity._check_hosted_image_edit(QWEN)

    [failure] = sanity.FAILS
    assert "no Modal credentials" in failure


def test_sanity_bounds_the_modal_lookup(sanity):
    """A dead network must cost a few seconds, not the Modal client's minute of retries."""
    release = threading.Event()
    function = SimpleNamespace(hydrate=lambda: release.wait(5))
    try:
        error = sanity._lookup_modal_function(function, timeout=0.05)
    finally:
        release.set()

    assert isinstance(error, TimeoutError)
