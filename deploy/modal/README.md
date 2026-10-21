# Modal setup

Modal hosts the GPU-backed model functions while the pipeline stays on the calling machine. They
live in one app, `litereality` ([`app.py`](app.py)), with three functions: `trellis`, `dino`, and
`image_edit`. The app remains deployed between calls, but its containers scale to zero when idle.

Modal is rented compute, not a model API: a Modal token gives you GPUs, not models, so nothing is
callable until you deploy this app into your own workspace.

## One-time setup

Join the intended Modal workspace, or use your own — nothing is pinned to a particular one, and no
workspace is hardcoded in source.

Install the client with `uv sync --extra modal`, then put your token pair from
[modal.com/settings/tokens](https://modal.com/settings/tokens) into `.env`:

```dotenv
MODAL_TOKEN_ID=ak-...
MODAL_TOKEN_SECRET=as-...
```

Both halves are required — Modal sends them as separate request headers, and the pipeline treats
half a pair as unconfigured rather than failing at the first model call.

**No Hugging Face account or token is required for any function.** DINO's weights are public.
TRELLIS.2-4B is MIT and ungated, and TRELLIS.2's gated DINOv3 image encoder is baked into the image
from a digest-verified ungated mirror — see [`trellis/README.md`](trellis/README.md).
Qwen-Image-Edit-2509 is Apache-2.0 and ungated.

## Deploy

```bash
uv run litereality setup
```

That is the whole step, and it runs unattended. It checks the client is installed, quotes the build
time, and runs the deploy below with the `.env` tokens injected into the subprocess — no browser
login, no profile to manage, and nothing to confirm.

Useful flags: `--skip-deploy` authenticates and records the profile only, `--env <name>` targets a
non-default Modal environment, and `--profile <name>` forces a specific `~/.modal.toml` profile
instead of the tokens.

### Deploying by hand

The one command `setup` wraps, if you would rather run it yourself:

```bash
uv run modal deploy --env main deploy/modal/app.py
```

Always deploy `app.py`, never the per-model files under `dino/`, `trellis/`, or `image_edit/`: a
deploy replaces every function on the app, and the per-model files on their own create separate
`litereality-*` apps the pipeline never looks up.

These need credentials in the environment first — either the same `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET` pair exported, or a profile:

```bash
uv run modal setup
uv run modal profile activate <profile>
export MODAL_PROFILE=<profile>
```

A profile works everywhere the token pair does; set `MODAL_PROFILE=<profile>` in `.env` instead of
the tokens. The app names, function names, and environment already default, so that one line is
enough. `models/registry.py` selects Modal whenever either form of credential is configured.

### What deploying gets you

The first deployment builds the images; the first *invocation* downloads model weights into
persistent Modal Volumes, so both are slow once and fast afterwards. DINO runs on an L4. TRELLIS
runs on one H100 with zero application retries and a ten-second idle scale-down window. A deployed
app with zero active containers does not consume GPU compute, though its stored image and Volume
data remain.

### Reference images without an OpenAI or Gemini key

The `image_edit` function runs [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
(Apache-2.0, ungated, no Hugging Face token) and serves `LR_IMAGE_PROVIDER=modal`: the
reference-image step renders on your own workspace and spends Modal credit instead of a paid
image API. It runs on one H100 with `max_containers=1`. Its image is a CPU-only pip install that
builds quickly; the cost is the first call, which downloads roughly 58 GB of weights into the
`litereality-image-edit-models` Volume, once per workspace. See
[`image_edit/README.md`](image_edit/README.md) for details.


See [`dino/README.md`](dino/README.md), [`trellis/README.md`](trellis/README.md), and
[`image_edit/README.md`](image_edit/README.md) for model-specific details.

## The room call (a separate app)

`roomcall/app.py` is the room call's web server (`doc/roomcall.md`), deployed as its own app,
`litereality-roomcall`, by `uv run litereality roomcall <scene> --deploy`. It is CPU-only and never
included in `app.py`: it has its own lifecycle, and redeploying it must not restart the model
functions, or the other way round. Its scene lives in the Volume `litereality-roomcall-scenes`.
