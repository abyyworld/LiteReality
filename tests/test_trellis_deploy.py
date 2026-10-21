"""TRELLIS on Modal, what can be checked without Modal: the deploy file's request handling, its
weights path (populated once on CPU, loaded offline, never written by a GPU container), its
autoscaling settings, and the streaming client the objects stage uses."""

from __future__ import annotations

import ast
import asyncio
import base64
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from litereality_agent.models.trellis import modal as trellis_modal

REPO = Path(__file__).resolve().parents[1]
APP_FILE = REPO / "deploy" / "modal" / "trellis" / "app.py"


@pytest.fixture()
def app():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("trellis_deploy_under_test", APP_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ── requests ─────────────────────────────────────────────────────────────────────────────────────


def test_requests_are_checked_before_any_gpu_work(app):
    req = app.parse_request({"image_b64": _b64(b"png"), "seed": "7", "simplify": 0.98,
                             "texture_size": 512, "pipeline_type": "512", "id": "Chair0:preview",
                             "something_newer": True})
    assert req == {"id": "Chair0:preview", "image": b"png", "seed": 7, "decimation_target": 20_000,
                   "texture_size": 512, "pipeline_type": "512"}
    assert app.parse_request({"image_b64": _b64(b"x")})["pipeline_type"] is None   # the default
    bad = [{}, {"image_b64": "not base64!"}, {"image_b64": _b64(b"x"), "pipeline_type": "2048"},
           {"image_b64": _b64(b"x"), "texture_size": 99999},
           {"image_b64": _b64(b"x"), "seed": "seven"},
           {"image_b64": _b64(b"x"), "id": "x" * 500}, "not a dict"]
    for request in bad:
        with pytest.raises(ValueError):
            app.parse_request(request)
    assert app._decimation_target(0.95) == 50_000 and app._decimation_target(1.5) == 10_000


def test_generate_echoes_the_id_and_never_commits_the_volume(app, monkeypatch, tmp_path):
    from litereality_agent.models.trellis import inference

    commits = []
    monkeypatch.setattr(app, "weights", SimpleNamespace(commit=lambda: commits.append(1),
                                                        reload=lambda: None))
    monkeypatch.setattr(app, "_load_pipeline", lambda: "pipeline")
    seen = {}

    def fake_generate(pipeline, image_path, output_path, **options):
        seen.update(options, pipeline=pipeline, image=Path(image_path).read_bytes())
        Path(output_path).write_bytes(b"glTF-fake")

    monkeypatch.setattr(inference, "generate", fake_generate)
    raw = app.generate.get_raw_f()
    out = raw({"image_b64": _b64(b"rgba"), "texture_size": 512, "pipeline_type": "512",
               "simplify": 0.98, "id": "Table0:preview"})
    assert base64.b64decode(out["glb_b64"]) == b"glTF-fake"
    assert out["id"] == "Table0:preview" and out["seconds"] >= 0
    assert seen == {"pipeline": "pipeline", "image": b"rgba", "seed": 42,
                    "decimation_target": 20_000, "texture_size": 512, "pipeline_type": "512"}
    legacy = raw({"image_b64": _b64(b"rgba")})                    # the full pipeline's request
    assert "id" not in legacy and "glb_b64" in legacy
    refused = raw({"image_b64": "%%%", "id": "Sofa0:final"})
    assert refused["id"] == "Sofa0:final" and "bad request" in refused["error"]
    assert commits == []


# ── weights ──────────────────────────────────────────────────────────────────────────────────────


def _fake_hub(monkeypatch, snapshot: Path, *, present: bool):
    """A huggingface_hub whose cache holds ``snapshot`` once ``present`` (a list's first item)."""
    calls = []
    state = {"present": present}
    hub = ModuleType("huggingface_hub")

    def snapshot_download(model, local_files_only=False):
        calls.append((model, local_files_only))
        if local_files_only and not state["present"]:
            raise FileNotFoundError("LocalEntryNotFoundError")
        state["present"] = True
        return str(snapshot)

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls, state


def _snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "hub" / "snapshots" / "abc123"
    (snap / "ckpts").mkdir(parents=True)
    (snap / "pipeline.json").write_text(json.dumps(
        {"args": {"image_cond_model": {"args": {"model_name": "facebook/dinov3-gated"}}}}))
    return snap


def test_a_gpu_container_loads_offline_into_its_own_folder(app, monkeypatch, tmp_path):
    snap = _snapshot(tmp_path)
    dino = tmp_path / "dinov3"
    dino.mkdir()
    monkeypatch.setenv("LITEREALITY_DINOV3", str(dino))
    monkeypatch.setenv("HF_HUB_OFFLINE", "")              # restored after: the code sets it
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "")
    monkeypatch.setattr(app, "LOCAL_MODEL_DIR", tmp_path / "container" / "model")
    monkeypatch.setattr(app, "FLEX_GEMM_CACHE", tmp_path / "container" / "flex" / "cache.json")
    volume_cache = tmp_path / "volume" / ".flex_gemm" / "autotune_cache.json"
    volume_cache.parent.mkdir(parents=True)
    volume_cache.write_text("{}")
    monkeypatch.setattr(app, "VOLUME_FLEX_GEMM_CACHE", volume_cache)
    events = []
    calls, state = _fake_hub(monkeypatch, snap, present=False)

    def populate():                                             # the CPU function's download
        events.append("populate")
        state["present"] = True

    monkeypatch.setattr(app, "populate", SimpleNamespace(remote=populate))
    monkeypatch.setattr(app, "weights", SimpleNamespace(reload=lambda: events.append("reload"),
                                                        commit=lambda: events.append("commit")))

    local = Path(app.offline_model_dir())
    assert events == ["populate", "reload"]                     # populated once, by the CPU fn
    assert all(only for _model, only in calls)                  # never online from a GPU box
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert local == tmp_path / "container" / "model"
    assert (local / "ckpts").resolve() == (snap / "ckpts").resolve()
    cfg = json.loads((local / "pipeline.json").read_text())
    assert cfg["args"]["image_cond_model"]["args"]["model_name"] == str(dino.resolve())
    assert app.FLEX_GEMM_CACHE.read_text() == "{}"              # seeded, read only

    events.clear()
    app.offline_model_dir()                                     # weights there: no populate
    assert events == []


def test_the_model_folder_is_rewritten_only_when_it_changes(monkeypatch, tmp_path):
    from litereality_agent.models.trellis import inference

    monkeypatch.delenv("LITEREALITY_DINOV3", raising=False)
    snap = _snapshot(tmp_path)
    local = inference.assemble_local_model(snap, tmp_path / "local")
    path = local / "pipeline.json"
    os.utime(path, (1_000_000, 1_000_000))
    inference.assemble_local_model(snap, tmp_path / "local")
    assert path.stat().st_mtime == 1_000_000                    # same content: not touched
    other = tmp_path / "other"
    (other / "ckpts").mkdir(parents=True)
    (other / "pipeline.json").write_text(json.dumps({"args": {"x": 1}}))
    inference.assemble_local_model(other, tmp_path / "local")
    assert path.stat().st_mtime != 1_000_000
    assert (local / "ckpts").resolve() == (other / "ckpts").resolve()


def test_prepare_local_model_can_stay_offline(monkeypatch, tmp_path):
    from litereality_agent.models.trellis import inference

    snap = _snapshot(tmp_path)
    calls, _ = _fake_hub(monkeypatch, snap, present=True)
    out = inference.prepare_local_model("microsoft/TRELLIS.2-4B", local_dir=tmp_path / "l",
                                        offline=True)
    assert Path(out) == tmp_path / "l" and calls == [("microsoft/TRELLIS.2-4B", True)]
    inference.prepare_local_model("microsoft/TRELLIS.2-4B", local_dir=tmp_path / "l")
    assert calls[-1] == ("microsoft/TRELLIS.2-4B", False)       # the local CLI, as before


def test_populate_downloads_once_and_commits(app, monkeypatch, tmp_path):
    snap = _snapshot(tmp_path)
    monkeypatch.delenv("LITEREALITY_DINOV3", raising=False)
    monkeypatch.setattr(app, "WEIGHTS_ROOT", tmp_path / "volume")
    monkeypatch.setattr(app, "POPULATED_MARKER", tmp_path / "volume" / ".populated")
    (tmp_path / "volume").mkdir()
    events = []
    monkeypatch.setattr(app, "weights", SimpleNamespace(reload=lambda: events.append("reload"),
                                                        commit=lambda: events.append("commit")))
    calls, _ = _fake_hub(monkeypatch, snap, present=False)
    raw = app.populate.get_raw_f()
    first = raw()
    assert first["status"] == "downloaded" and events == ["reload", "commit"]
    assert calls[-1] == (app.MODEL_ID, False)
    assert (tmp_path / "volume" / "trellis2_4b_local" / "pipeline.json").is_file()
    events.clear()
    second = raw()                                              # another container, later
    assert second["status"] == "present" and events == ["reload"]


def _decorator_kwargs(function_name: str) -> dict[str, ast.AST]:
    tree = ast.parse(APP_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for deco in node.decorator_list:
                if isinstance(deco, ast.Call) and getattr(deco.func, "attr", "") == "function":
                    return {kw.arg: kw.value for kw in deco.keywords}
    raise AssertionError(f"no @app.function on {function_name}")


def test_up_to_four_gpu_containers_and_a_single_cpu_populator(app):
    gpu = _decorator_kwargs("generate")
    assert ast.unparse(gpu["max_containers"]) == "MAX_CONTAINERS" and app.MAX_CONTAINERS == 4
    assert ast.unparse(gpu["scaledown_window"]) == "SCALEDOWN_SECONDS"
    assert 60 <= app.SCALEDOWN_SECONDS <= 600
    assert ast.unparse(gpu["gpu"]) == "'H100'"
    cpu = _decorator_kwargs("populate")
    assert "gpu" not in cpu and ast.unparse(cpu["max_containers"]) == "1"
    assert set(app.app.registered_functions) == {app.FUNCTION_NAME, app.POPULATE_FUNCTION}


def test_nothing_on_the_request_path_writes_the_volume():
    tree = ast.parse(APP_FILE.read_text())
    hot = {"generate", "_load_pipeline", "offline_model_dir", "_seed_flex_gemm_cache"}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in hot:
            text = ast.unparse(node)
            assert "commit(" not in text, node.name
            assert "WEIGHTS_ROOT /" not in text, node.name


def test_the_image_keeps_flex_gemm_tuning_per_container(app):
    assert str(app.FLEX_GEMM_CACHE).startswith("/tmp/")
    assert "FLEX_GEMM_AUTOTUNE_CACHE_PATH" in APP_FILE.read_text()


# ── the streaming client ─────────────────────────────────────────────────────────────────────────


def test_stream_sequences_runs_each_key_in_order_within_the_limit():
    live, peak, started = [0], [0], []

    async def remote(request):
        started.append(request["id"])
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        await asyncio.sleep(0.01 if request["id"] != "slow:0" else 0.05)
        live[0] -= 1
        if request["id"] == "bad:0":
            raise RuntimeError("container lost")
        if request["id"] == "err:1":
            return {"error": "OOM"}
        return {"glb_b64": "x", "id": request["id"]}

    sequences = {k: [{"id": f"{k}:0"}, {"id": f"{k}:1"}] for k in ("a", "slow", "bad", "err")}

    async def collect():
        return [item async for item in trellis_modal.stream_sequences(
            remote, sequences, max_in_flight=2, timeout=5)]

    items = asyncio.run(collect())
    assert len(items) == 8 and peak[0] <= 2
    firsts = [i for i, rid in enumerate(started) if rid.endswith(":0")]
    seconds = [i for i, rid in enumerate(started) if rid.endswith(":1")]
    assert max(firsts) < min(seconds)                    # every first pass goes out first
    by = {(i.key, i.index): i for i in items}
    assert "container lost" in by["bad", 0].error and by["bad", 1].output   # it carried on
    assert by["err", 1].error == "OOM" and by["err", 1].output is None
    assert by["a", 1].output["id"] == "a:1"


def test_stream_sequences_times_out_one_request_not_the_stream():
    async def remote(request):
        await asyncio.sleep(1.0 if request["id"] == "stuck" else 0.0)
        return {"glb_b64": "x"}

    async def collect():
        return [i async for i in trellis_modal.stream_sequences(
            remote, {"s": [{"id": "stuck"}], "f": [{"id": "fine"}]}, timeout=0.1)]

    items = {i.key: i for i in asyncio.run(collect())}
    assert "no answer" in items["s"].error and items["f"].output


def test_the_service_streams_from_synchronous_code():
    async def aio(request):
        await asyncio.sleep(0.01)
        return {"glb_b64": base64.b64encode(b"glTF....").decode(), "id": request["id"]}

    service = trellis_modal.ModalTrellisService(
        app_name="unused", client=SimpleNamespace(function=SimpleNamespace(
            remote=SimpleNamespace(aio=aio))))
    started = time.monotonic()
    items = list(service.stream({k: [{"id": k}] for k in "abcdef"}, max_in_flight=6))
    assert sorted(i.key for i in items) == list("abcdef")
    assert time.monotonic() - started < 1.0              # in parallel, not one after another
    assert trellis_modal.decode_glb(items[0].output, expect_id=items[0].key) == b"glTF...."


def test_payloads_and_responses_are_checked():
    png = io.BytesIO()
    from PIL import Image

    Image.new("RGBA", (4, 4)).save(png, format="PNG")
    payload = trellis_modal.request_payload(png.getvalue(), pipeline_type="512",
                                            texture_size=512, request_id="Chair0:preview")
    assert payload["id"] == "Chair0:preview" and payload["pipeline_type"] == "512"
    assert base64.b64decode(payload["image_b64"]) == png.getvalue()
    assert "id" not in trellis_modal.request_payload(b"x")
    with pytest.raises(ValueError):
        trellis_modal.request_payload(b"x", pipeline_type="huge")
    glb = {"glb_b64": base64.b64encode(b"glTFxyz").decode(), "id": "A:final"}
    assert trellis_modal.decode_glb(glb, expect_id="A:final") == b"glTFxyz"
    assert trellis_modal.decode_glb({"glb_b64": glb["glb_b64"]}, expect_id="A:final")  # old fn
    for bad, why in (({"error": "boom"}, "boom"), ({**glb, "id": "B:final"}, "expected"),
                     ({"glb_b64": base64.b64encode(b"PK").decode()}, "not a GLB"),
                     ({"glb_b64": "%%"}, "invalid"), ([], "invalid")):
        with pytest.raises(ValueError, match=why):
            trellis_modal.decode_glb(bad, expect_id="A:final")
