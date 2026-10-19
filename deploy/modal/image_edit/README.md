# Qwen-Image-Edit on Modal

This is a deployment wrapper around `litereality_agent.models.image_edit.inference`; inference
stays in `src/` and the Modal runtime only supplies a GPU and a persistent weight cache. It is the
backend behind `LR_IMAGE_PROVIDER=modal`: the reference-image step sends each evidence sheet and
its prompt here instead of to OpenAI or Gemini, so **no image API key is needed** — the step spends
your Modal credit instead (Modal's Starter plan includes a monthly free credit, $30 at the time of
writing).

It deploys as the `image_edit` function of the combined `litereality` app, together with `trellis`
and `dino`:

```bash
uv run litereality setup          # or by hand: uv run modal deploy --env main deploy/modal/app.py
```

Never deploy this file on its own: it would create a separate `litereality-image-edit` app that the
pipeline never looks up.

## Selecting it

```dotenv
LR_IMAGE_PROVIDER=modal
```

in `.env`, with the same `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` (or `MODAL_PROFILE`) that TRELLIS
and DINO already use. While `LR_IMAGE_PROVIDER` is unset, naming a `Qwen/...` model (the scene-init
flow's `--image-model Qwen/Qwen-Image-Edit-2509`) selects it too. `uv run python sanity.py` reports
which provider a run will use and whether its credentials are present.

## Model

| | |
|---|---|
| model | [`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) via diffusers `QwenImageEditPlusPipeline` |
| licence | Apache-2.0, ungated — no Hugging Face account or token |
| weights | ~58 GB in bf16: the 20B MMDiT transformer plus the Qwen2.5-VL-7B text encoder |
| GPU | one H100 (80 GB); the weights alone rule out 40-48 GB cards |
| defaults | 40 steps, true CFG 4.0 with a `" "` negative prompt, the model card's own settings |

The first call in a workspace downloads the weights into the `litereality-image-edit-models`
Volume, inside the H100 function, so it takes several minutes and is billed at the H100 rate; the
function timeout is 60 minutes to leave room for it. Later cold starts read the weights from the
Volume, which still takes a minute or more, and a warm container (it idles for 60 seconds after
the last call) skips loading entirely. The image itself is a CPU-only pip install of prebuilt
wheels and builds in minutes.

## Cost

These are estimates, not measurements: 40 steps with true CFG is roughly a minute of H100 time per
reference at 1024x1024, i.e. about $0.05-0.07 per reference at Modal's H100 list price of $3.95/h,
plus the cold-start load whenever the container was not warm. Each reference logs its own
GPU-seconds and an estimated cost (`~$… (… GPU s, estimate)`); set `LR_MODAL_GPU_PRICE_PER_HOUR`
in `.env` (or `models.env`) if the rate changes.

To spend less:

- `LR_MODAL_IMAGE_STEPS` (default 40) — cost is linear in steps, so 20 roughly halves it, at some
  loss of detail.
- `LR_MODAL_IMAGE_CFG` (default 4.0) — `1` turns true CFG off, one transformer pass per step instead
  of two, roughly halving cost again at a cost in prompt adherence.

Both live in `models.env` beside the other model choices.

Each reference draws a fresh sampling seed, so `--force-image-generation` and the chair-QC repair
loop get a new image rather than paying for the one they rejected; the seed of every render is in
its `_responses/modal_response_*.json`. `LR_MODAL_IMAGE_SEED` pins one to reproduce a render, which
also makes every forced or chair-QC regeneration identical again.

## Throughput and failure behaviour

The function runs at most one container (`max_containers=1`). That serializes cold starts, so two
containers can never race to populate the Volume, and it caps spend on the free tier. The
reference step still submits up to `LR_IMAGE_WORKERS` (20) calls at once; they simply queue on
Modal, and queueing is not billed. The trade is wall time: references come back one after another,
about a minute each, rather than in one parallel wave as with the hosted APIs.

Nothing retries. The function is deployed with `retries=0`, and the client makes exactly one
attempt with no client-side timeout, because every attempt is billed GPU time and the failures it
can hit (an out-of-memory, the function timeout, a bad request, an undeployed app) do not clear on
a second try. A failed reference raises, and the pipeline falls back to the raw evidence sheet for
that object and records the error — the same as for the other providers.
