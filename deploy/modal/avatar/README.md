# Full-body avatars on Modal (LHM++)

The room call's full-body people: each person's turnaround frames or photos go to this app, which
runs [LHM++](https://github.com/aigc3d/LHM-plusplus) (`LHMPP-700M-SMPLX-FREE`) on one L4 and answers
with a 3D Gaussian body in the SMPL-X T-pose plus per-splat skinning weights (an LRAV v1 file,
`room_ops/roomcall/avatar_format.py`). Every browser animates it with Spark.

It is its own app, `litereality-avatar`, and is **optional**: without it, or whenever it fails or
times out, the room server builds its stand-in body instead. Inference is in
`src/litereality_agent/models/avatar/lhm.py`; this file only adds the image, the weights Volume and
the GPU.

## Deploy and try it

Credentials as for the other apps ([../README.md](../README.md#deploying-by-hand)): a Modal profile,
or `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` exported in the shell. Then, from the checkout:

```bash
uv run modal run --env main deploy/modal/avatar/app.py::populate    # weights into the Volume, once
uv run modal deploy --env main deploy/modal/avatar/app.py           # the `avatar` function
uv run modal run --env main deploy/modal/avatar/app.py \
    --images me_front.jpg,me_side.jpg,me_back.jpg --deployed --out me.lrav  # one body, to disk
uv run litereality roomcall --demo --avatars modal                  # the room call using it
```

- Use the same `--env` everywhere: the Volume belongs to one Modal environment.
- The first command builds the image. Nothing here has been built yet, so this is its first
  build: about 20-40 min on Modal's CPU builders, longer if pytorch3d's prebuilt wheel is missing
  and it compiles from source. No build step attaches a GPU.
- `--images` takes any phone photos (JPEG/PNG). They are turned upright, shrunk to 1600 px and
  re-encoded without EXIF on the laptop first, as the browser does. Use `--mode turnaround` for
  frames from a turn in front of a webcam. Without `--deployed` the file runs as a temporary app
  with the same image and Volume.
- The printed stats include `skin_err_mm`: in a test pose, how far linear blending with the file's
  8-bit top-4 weights lands from linear blending with LHM++'s full 55 weights. It is an estimate
  for comparing builds: browsers skin with Spark's dual quaternions, so it is not what people see,
  and there is no threshold to hold it to.

## Weights

`avatar_populate` (CPU, one container at a time, idempotent) downloads `3DAIGC/LHMPP-Prior` and
`3DAIGC/LHMPP-700M-SMPLX-FREE` from Hugging Face. If that fails it uses ModelScope
(`Damo_XR_Lab/...`). They go into the `litereality-avatar-models` Volume (a few GB).

- It checks that every prior file inference opens is present (`REQUIRED_PRIOR`, FLAME included)
  and that the checkpoint folder has `config.json` and `model.safetensors`, the only weights file
  `from_pretrained` opens from a folder. Then it writes a marker and commits.
- It logs the resolved commits. Pin them in `MODEL_REVISION` / `PRIOR_REVISION` after the first
  run.
- A GPU container finding no weights calls it and waits, but a first download can outlast the room
  server's 420 s timeout. Run it ahead of time, as above.

GPU containers read the weights offline. They link LHM++'s `pretrained_models` to the Volume on
their own disk and never write the Volume, so two can run at once (`MAX_CONTAINERS = 2`).

## When it fails

- Photos that cannot make a body (nobody in them, a file that does not decode) get their reason
  back. Every other failure is answered `internal: <type>`, with the traceback in the Modal logs.
- A fault with the model loaded (a CUDA error, say) retires the container: it takes no more inputs
  and the next request starts a fresh one, since a CUDA fault can break the process for good.
- A model that fails to load is not retried in that container for `LOAD_RETRY_SECONDS` (3 min):
  requests in that window get `internal: ModelLoadError` at once instead of each rebuilding the
  model on the GPU. The logs carry the first error.

## Licences

- LHM++ code: Apache-2.0.
- LHM++ weights: **CC-BY-NC-4.0**.
- The SMPL-X and FLAME body models in the prior bundle: **non-commercial research licences**.

Fine for research. Not for a product.

## Cost

These are Modal list prices at the time of writing; check [modal.com/pricing](https://modal.com/pricing).
An L4 costs about $0.80/h, and the function's 32 GB of memory adds about $0.26/h.

| Situation | Time | Cost |
|---|---|---|
| A cold start (image load plus model load) | about 1-2 min | about $0.03 |
| One body on a warm container | about 5-15 s | under $0.01 |
| The idle tail (`SCALEDOWN_SECONDS = 300`) | 5 min | about $0.09 |

So the first body in a call costs about $0.10-0.15, and each further body within 5 minutes costs
about $0.01. The room server caps GPU builds at 40 per UTC day, so the worst case is about $5 a
day. Populating costs a few cents of CPU. An idle deployed app costs nothing.

## Not yet verified

This could not be run from the machine it was written on, which has no Modal, Hugging Face or GPU
access. The tests (`tests/test_avatar_gpu.py`) cover everything that runs without them.

- The image build. Every version is pinned; pytorch3d has a pinned source fallback.
- The prior bundle's layout.
- `LHMRunner`, which was written from LHM++'s source at the pinned commit.
- The framing of cut-off views. A person cut off by the frame's bottom edge (the webcam turnaround
  asks for head to knees) is bottom-aligned and spans at most `CUT_FILL` (0.75) of the input
  height, as in LHM++'s training crops. Compare a body from cut views with one from uncut photos of
  the same person on the first run.
- Real timings.
