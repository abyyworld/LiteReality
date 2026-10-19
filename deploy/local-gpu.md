# Running the models on your own GPU

The alternative to [Modal](modal/README.md). TRELLIS and GroundingDINO run on your machine instead
of hosted, which needs Linux with an NVIDIA GPU of 24 GB or more.

Steps 1, 2 and 4 of [Install](../README.md#install) are unchanged. The only difference is that you
leave `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` empty — that is the switch.
`models/registry.py` selects Modal whenever credentials are configured and falls back to these
local runtimes only when they are not, so an empty pair is what makes the fallback reachable.

## TRELLIS.2

It needs `torch 2.6+cu124` and compiles native CUDA extensions, so it gets its own environment
rather than sharing the light app env:

```bash
git clone --recursive https://github.com/microsoft/TRELLIS.2.git backends/TRELLIS.2
cd backends/TRELLIS.2
./setup.sh --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec
```

`setup.sh` checks `nvidia-smi`, so the build itself wants a visible GPU.

### Checkpoints

`microsoft/TRELLIS.2-4B` (MIT, ungated) downloads itself on first use into `backends/weights/`, and
needs no Hugging Face account.

TRELLIS.2's DINOv3 image encoder *is* gated upstream. Put an ungated copy in
`backends/weights/dinov3` and inference finds it there instead of reaching for the gated repo —
`resolve_dinov3()` checks `$LITEREALITY_DINOV3` first, then that path.
[`modal/trellis/README.md`](modal/trellis/README.md) names a digest-verified mirror and its
revision.

## GroundingDINO

Its own environment too, since it pulls torch:

```bash
uv sync --extra detect     # in a separate environment
```

## Wiring it up

Point `.env` at both interpreters:

```dotenv
TRELLIS_PYTHON=/path/to/trellis-env/bin/python
GROUNDING_DINO_PYTHON=/path/to/detect-env/bin/python
```

Then `uv run python sanity.py` — it loads each model and reports anything that would silently fall
back to a weaker path.
