# TRELLIS on Modal

This is a deployment wrapper around
`litereality_agent.models.trellis.inference`; model inference remains in `src/` and the Modal
runtime only supplies a GPU, parallel execution, and a persistent weight cache.

Modal associates a deployment with the workspace belonging to the active profile; the workspace
cannot be selected by app source code. Confirm which one you are about to bill, then deploy into
its `main` environment:

```bash
uv run modal profile list
uv run modal profile activate <name>
uv run modal profile current

uv run modal deploy --env main deploy/modal/app.py  # trellis, dino and image_edit together
```

That deploys this model as the `trellis` function of the combined `litereality` app, which is also
what `uv run litereality setup` does. Never deploy this directory's `app.py` on its own: it would
create a separate `litereality-trellis` app that the pipeline never looks up. See
[Deploying by hand](../README.md#deploying-by-hand).

## No Hugging Face account is needed

TRELLIS.2's image encoder is the gated `facebook/dinov3-vitl16-pretrain-lvd1689m`, which is why
this app previously required a `huggingface` Modal secret. It no longer does. The DINOv3 licence
grants the right to "distribute, copy" the trained weights to third parties provided a copy of the
licence travels with them, so the image bakes in an ungated mirror instead:

- repo `camenduru/dinov3-vitl16-pretrain-lvd1689m`
- revision `3c276edd87d6f6e569ff0c4400e086807d0f3881`
- `model.safetensors` sha256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`

The build verifies that digest and fails if it does not match, so a mirror force-push cannot
silently swap the weights. Four independent uploaders publish byte-identical copies under that same
digest, which is how the weights were confirmed unmodified without holding gate access. Always pin
by revision. `resolve_dinov3()` finds the baked copy through `LITEREALITY_DINOV3` and repoints
`pipeline.json` at it, so loading never reaches the gated repo.

`microsoft/TRELLIS.2-4B` is MIT and ungated, so no token is needed for it either.

The base CUDA image, OS packages, PyTorch, and Python dependencies build on CPU. Modal attaches an
H100 only to the separately cached native-extension layer required by the upstream TRELLIS
installer; the deployed inference function also uses an H100. Source-only application changes do
not rebuild the extension layer. The function runs on up to four H100 containers at once (the
room call's objects stage sends a whole room together), performs zero application retries, and
scales a container down after two idle minutes — so one object's final pass, queued the moment its
preview is back, and a re-run right after both find a warm container. The idle tail is at most
4 x 120 s of H100 time per burst.

The weights are hosted on Hugging Face, but the model is not currently served by a Hugging Face
Inference Provider. They are downloaded into the `litereality-trellis-models` Volume ONCE, by the
CPU function `trellis_populate` (one container at a time, idempotent): the first GPU request that
finds them missing calls it and waits, and every later container reuses the cache. GPU containers
never write the Volume — they load with Hugging Face offline, assemble their model folder
(patched `pipeline.json` + a `ckpts` link) and flex_gemm's autotune cache under `/tmp`, and commit
nothing — which is what makes more than one container safe. To populate ahead of time:

```bash
uv run python -c "import modal; print(modal.Function.from_name('litereality', 'trellis_populate').remote())"
```

Requests may carry an `"id"`, echoed in the response, so a client streaming many requests at
once can match each GLB to its request; the response also says how many seconds it took.

Keeping a container warm for a call (no cold start), and putting it back afterwards — an override
lasts until the next deploy, and a warm H100 bills whether or not anyone uses it:

```python
import modal
f = modal.Function.from_name("litereality", "trellis")
f.update_autoscaler(min_containers=1)   # at call start
f.update_autoscaler(min_containers=0)   # when the call is over
```

Configure the application by setting `MODAL_PROFILE` to your profile name; the app name, function
name, and environment already default. Install client support with `uv sync --extra modal`.

Measured on a fresh workspace, 2026-08-06: the image built in 11.3 minutes for about $0.45, split
almost evenly in time between the CPU dependency layers (318.0s) and the H100 extension compile
(318.4s), with the GPU layer carrying roughly 78% of the cost. Redeploy after a source-only change
took 2.8 seconds. The first request returned a valid 512-texture GLB from `ChairCluster0` — 33,898
vertices, 47,793 faces — in 295.7s including the weights download; later requests skip it. Neither
building nor inference is part of the offline test suite.
