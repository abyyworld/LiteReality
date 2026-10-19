# DINO on Modal

This scale-to-zero app serves both GroundingDINO detection and DINOv2 embeddings through the
canonical model code. Its image is CPU-built from precompiled wheels; an L4 is attached only while
requests run. Public Hugging Face weights are cached in the `litereality-dino-models` Volume.

It deploys as the `dino` function of the combined `litereality` app, into whichever workspace the
active profile names (`uv run litereality setup` runs the same deploy for you):

```bash
MODAL_PROFILE=<profile> uv run modal deploy --env main deploy/modal/app.py  # trellis, dino and image_edit together
```

Never deploy this directory's `app.py` on its own: it would create a separate `litereality-dino`
app that the pipeline never looks up. See [Deploying by hand](../README.md#deploying-by-hand).

Configure the pipeline by setting `MODAL_PROFILE` to that same profile; `MODAL_DINO_APP`,
`MODAL_DINO_FUNCTION`, and `MODAL_ENVIRONMENT` already default. No Hugging Face token is needed —
both models are public.
