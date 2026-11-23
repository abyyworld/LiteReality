# Dense depth (PromptDA)

`app.py` deploys **`litereality-depth`**, one function, `refine`: Prompt Depth Anything
([DepthAnything/PromptDA](https://github.com/DepthAnything/PromptDA), CVPR 2025, Apache-2.0) turns
each photo plus the iPhone's own 256 x 192 LiDAR map into metric depth at the training resolution
(960 x 720 for a 4:3 scan). The splat room uses it to place more, sharper starting points and to
supervise the trainer with a full depth map instead of 4000 samples per photo.

It is its own app, like `splat/`: never included into `litereality` (`deploy/modal/app.py`), so
`litereality setup` neither deploys nor deletes it, and it imports nothing from this repo.

## Deploy

```bash
uv run modal deploy --env main deploy/modal/depth/app.py
```

The first build downloads torch and the 1.36 GB checkpoint and loads the model once on CPU to check
the pins (a few minutes). No Hugging Face token: the weights are ungated.

## Pins

| What | Pin | Checked |
|:-|:-|:-|
| Code | `DepthAnything/PromptDA@9161547` (main, 2026-01-29; inference code unchanged since `3f770c7`, 2025-02-03) | `git fetch` of that exact commit |
| Weights | `depth-anything/prompt-depth-anything-vitl@573ab70`, `model.ckpt` | SHA-256 `4b3a3e0a…f10f`, plus a strict state-dict load and a CPU forward pass at build |
| Runtime | torch 2.4.1 (cu124), numpy 1.26.4, opencv-python-headless 4.9.0.80 | |

The checkpoint's pickle holds only tensors and an `OrderedDict` (no Lightning classes), so it loads
without PromptDA's training requirements (Lightning, xFormers, gradio). PromptDA's `setup.py` looks
for packages under `src/`, which the repo does not have, so `pip install -e .` installs nothing: the
checkout is on `PYTHONPATH` instead, which is also where the model finds its vendored DINOv2.

## Call

```python
refine(frames, {"width": 960})
# frames:  [{"name": "frame_00030", "image": <JPEG>, "depth": <uint16 mm PNG>, "conf": <PNG> | None}]
# returns: [{"name": "frame_00030", "depth": <uint16 mm PNG at 960 x 720, 0 = invalid>, "ms": 140.2}]
```

Exactly the README's inference: RGB in 0..1 with the long side at most 1008 px and both sides a
multiple of 14 (cv2 `INTER_AREA`), the LiDAR map in metres as written (no resize, no masking: the
model normalises it by its own min and max), `model.predict(image, prompt_depth)` in fp32. The
1008 x 756 result is resized bilinearly to the asked width. `conf` is accepted and ignored by the
model; the client uses it to check the result against confident LiDAR (`rgbd.dense_mask`).

`refine([])` boots a container and returns `[]`: the client's `warm()` sends it so the cold start
overlaps other work.

The client is `litereality_agent.models.splat.promptda.ModalDepthRefiner`: 16 frames a call, up to
four calls in parallel, each photo sent either as the scan saved it or shrunk to 1008 x 756 (what
the model resizes it to anyway), whichever JPEG is smaller (office_room's ~465 KB photos go up as
120-310 KB, suite_3693's ~365 KB as 75-160 KB; tea_room's 97 KB ones as they are), results cached
in `<scan>/depth_dense/frame_XXXXX.png` so a scan is refined once. It never raises; a failed call only means those frames keep their LiDAR.

## Cost

One L4 (or A10G / L40S, whichever is free first), up to four containers. Not yet measured: the
measure stage records seconds per frame (`ms` in each reply, `gpu_ms_median` in the client's
`last_stats`). Plain fp32 attention (no xFormers) is the slow part at ~3900 tokens a photo; if it
proves too slow, xFormers 0.0.28.post1 (built for torch 2.4.1) is the first thing to try.
