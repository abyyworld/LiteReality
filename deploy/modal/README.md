# Modal setup

Modal hosts the GPU-backed model functions while the pipeline stays on the calling machine. The
apps remain deployed between calls, but their containers scale to zero when idle.

Modal is rented compute, not a model API: a Modal token gives you GPUs, not models, so nothing is
callable until you deploy these apps into your own workspace.

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

**No Hugging Face account or token is required for either app.** DINO's weights are public.
TRELLIS.2-4B is MIT and ungated, and TRELLIS.2's gated DINOv3 image encoder is baked into the image
from a digest-verified ungated mirror — see [`trellis/README.md`](trellis/README.md).

## Deploy

```bash
uv run litereality setup
```

That is the whole step, and it runs unattended. It checks the client is installed, quotes the build
time, and runs both deploys below with the `.env` tokens injected into each subprocess — no browser
login, no profile to manage, and nothing to confirm.

Useful flags: `--skip-deploy` authenticates and records the profile only, `--env <name>` targets a
non-default Modal environment, and `--profile <name>` forces a specific `~/.modal.toml` profile
instead of the tokens.

### Deploying by hand

The two commands `setup` wraps, if you would rather run them yourself:

```bash
uv run modal deploy --env main deploy/modal/dino/app.py
uv run modal deploy --env main deploy/modal/trellis/app.py
```

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


See [`dino/README.md`](dino/README.md) and [`trellis/README.md`](trellis/README.md) for model-specific
details.
