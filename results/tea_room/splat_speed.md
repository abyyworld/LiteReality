# tea_room: the trained splat, sooner (2026-11-13)

**Result:** the final trained room now arrives **95 s** after pressing enter, down from 184 s.
With the warm-up call (not wired into the client yet, see below) it arrives at **74 s**. Quality
is the same or better: held-out PSNR **20.24 dB**, SSIM 0.783, LPIPS 0.364, against 19.17 / 0.778 /
0.359 for the old 7000-step run. Training itself takes **33 s** instead of 142 s. A room costs
**about $0.15** instead of about $0.30.

The change that matters is **4 photos per step** (`--batch-size 4`), 1250 steps. The H100 is
limited by per-step overhead, not by pixels: at batch 1 it ran 49 steps/s using 1.3 GB of its
80 GB. Four photos per step run 38 steps/s, which is 151 photos/s instead of 49. The averaged
gradient also trains better per photo seen.

Two kernel options add the rest of the speed:
- the fused bilateral grid, a CUDA kernel in place of PyTorch ops: 48 to 66 steps/s;
- visible-splat Adam (Taming 3DGS): 66 to 76 steps/s, with better LPIPS.

## How it was measured

Every run used gsplat 1.5.3 `simple_trainer.py mcmc` on one Modal H100, on the 62-photo
tea_room scan, with LiDAR depth loss, antialiasing, an 800k splat cap and SH degree 3 in training.
Every 8th photo was held out, which leaves 8 test photos. Configurations were run through the
deployed `train` function with explicit options (a small script that calls `remote_gen`). The
chosen default was then run end to end with `litereality splatroom ... --no-masks`.

Times in the sweep table are **container-side**, counted from the start of the `train` call
inside the container. They leave out the cold boot and the upload, because many sweep runs reused
a warm container. The end-to-end table below includes both.

Noise: two runs of the same baseline gave 19.17 and 19.01 dB. Three runs of the new default gave
20.22, 20.24 and 20.26 dB. Treat differences under about 0.2 dB as noise.

## Sweep (container-side seconds)

"Schedule scaled" means MCMC's growth schedule and the SH-degree schedule are compressed to the
shorter run: start, refine interval and SH interval are all multiplied by steps / 7000. Cost assumes a
standalone run: (busy time + 10 s cold boot + 60 s idle) at $0.00127/s for the H100, 8 CPUs and
32 GiB.

| Config | Steps/s | Training loop (s) | First snapshot (s) | Final (s) | PSNR | SSIM | LPIPS | Cost |
|---|---|---|---|---|---|---|---|---|
| baseline: 7000 steps, batch 1 (old defaults) | 49 | 142 | 51 (step 2000) | 157 | 19.01 | 0.777 | 0.363 | $0.29 |
| 3000 steps, schedule scaled | 48 | 62 | 34 (step 1000) | 76 | 18.51 | 0.725 | 0.491 | $0.19 |
| + fused bilateral grid | 66 | 45 | 27 (step 1000) | 57 | 18.66 | 0.724 | 0.493 | $0.16 |
| + fused grid + packed | 66 | 46 | 26 (step 1000) | 57 | 18.43 | 0.724 | 0.492 | $0.16 |
| + fused grid + visible Adam | 76 | 39 | 26 (step 1000) | 53 | 18.54 | 0.722 | 0.450 | $0.16 |
| + fused grid, cap 500k | 74 | 40 | 23 (step 1000) | 49 | 18.30 | 0.696 | 0.538 | $0.16 |
| + fused grid, SH degree 1 | 57 | 53 | 30 (step 1000) | 64 | 18.57 | 0.732 | 0.486 | $0.17 |
| 7000 steps, fused grid + visible Adam | 79 | 88 | 35 (step 2000) | 99 | 18.65 | 0.751 | 0.389 | $0.22 |
| 5000 steps, fused + vis, scaled | 77 | 65 | 29 (step 1500) | 75 | 18.40 | 0.739 | 0.417 | $0.19 |
| 4000 steps, fused + vis, scaled | 75 | 53 | 24 (step 1000) | 65 | 18.49 | 0.731 | 0.435 | $0.17 |
| 4000 steps, fused + vis, schedule NOT scaled | 82 | 49 | 21 (step 1000) | 60 | 18.38 | 0.737 | 0.436 | $0.17 |
| 4000 steps, fused + vis, scaled, 720 px photos | 80 | 50 | 23 (step 1000) | 61 | 18.59 | 0.696 | 0.412 | $0.17 |
| batch 2 x 2500 steps, fused + vis, scaled | 63 | 40 | 23 (step 800) | 50 | 19.40 | 0.763 | 0.381 | $0.15 |
| batch 2 x 2000 steps, fused + vis, scaled | 60 | 33 | 23 (step 600) | 47 | 19.46 | 0.761 | 0.393 | $0.15 |
| **batch 4 x 1250 steps, fused + vis, scaled (new default)** | 38 | 33 | 23 (step 400) | 46 | 20.22 | 0.783 | 0.364 | $0.15 |

What each knob did:
- **Fewer steps alone** (3000, 4000, 5000): 0.4–0.6 dB lost, and much worse LPIPS (0.42–0.49
  against 0.36). The schedule scaling made no measurable difference.
- **Fused bilateral grid**: +37% steps/s, at no cost to quality. Kept.
- **Visible Adam**: a further +15% steps/s, and better LPIPS at 3000 steps. At 7000 steps it
  scored 18.65 dB against 19.01, which is at the edge of the noise. Kept: with batch 4 the result is
  better than the baseline anyway.
- **Packed rasterisation**: no faster. Not used.
- **Cap 500k**: +12% steps/s, but −0.36 dB and LPIPS 0.54. Not used.
- **SH degree 1**: no faster (the measured 57 steps/s is probably noise between containers), and
  no better. Not used: training stays at degree 3, and the SPZ files are still degree 1.
- **720 px photos**: only +7% steps/s, which confirms the trainer is not limited by pixels. The
  metrics are at 720 px, so they are not directly comparable. Not used: `TRAIN_WIDTH` stays at 960.
- **Batch 2** (2500 or 2000 steps): 19.40–19.46 dB with a 33–40 s loop.
- **Batch 4 × 1250 steps**: 20.22 dB, 0.783, 0.364, with a 33 s loop. **This is the new default.**

Snapshots are now at steps 400, 800 and 1250. Step 400 already holds 597k splats, more than the
old first snapshot (step 2000, 491k splats), and it arrives about 22 s after the container
starts instead of 51 s.

## End to end (`litereality splatroom tea_room --no-masks`, seconds after pressing enter)

| | Old (7000 × 1) | New, cold | New + warm-up call |
|---|---|---|---|
| Instant room | 10.3 | 22.5 | 13.4 |
| Training set ready (MB uploaded) | 12.8 (28) | 28.6 (17) | 17.5 (17) |
| Trainer up on the H100 | 29.4 | 45.9 | **24.8** |
| First trained snapshot in the call | 81.3 (step 2000) | 71.8 (step 400) | **53.7** (step 400) |
| Final snapshot in the call | 184.3 | 95.3 | **74.1** |
| Held-out PSNR / SSIM / LPIPS | 19.17 / 0.778 / 0.359 | 20.24 / 0.783 / 0.364 | 20.26 / 0.783 / 0.362 |

During the "new, cold" run the laptop was busy with other sessions' work: the instant room took
22.5 s instead of the usual ~10 s. From "trainer up" to "final", the new trainer takes 49 s
against 155 s before. Each snapshot also takes 4–5 s on the laptop to cut against the physics
before it is in the call; this is included above.

## Getting the trainer up sooner

Where the time goes before the first training step, on a cold start:

| Step | Seconds | What changed |
|---|---|---|
| Uploading the dataset to Modal (the call's argument) | ~4–7 → ~3 | the tar is now gzipped (level 1): 28 → 17 MB, ~0.8 s to compress. 70% of it was the float64 COLMAP points, not the photos. Measured ~5–7 MB/s from this laptop. |
| GPU container cold boot | ~9–10 | **hidden by the warm-up call** (below), or by keeping one warm |
| Extracting the tar, launching the trainer | 2.0 | the trainer now starts before the parent imports torch |
| Trainer imports and data loading (torch, gsplat, torchmetrics, the COLMAP model) | 5.2 | |
| KNN scale init and the LPIPS network | 1.4 | |
| First step | 0.6 | |

The last three are measured by the app itself: the "done" event now carries
`timing: {launched, data_loaded, model_ready, training}`.

Options:

1. **Warm-up call (done in the app; the client change is still to do).** `train(b"")` now
   boots a container, initialises CUDA and returns at once, with
   `{"kind": "done", "warm": True}`. The container then stays up for `scaledown_window`
   (60 s). If the client sends it when the scan arrives, the ~10 s cold boot happens while the
   instant room is being built: trainer up at 24.8 s instead of 45.9 s, final at 74 s instead
   of 95 s. It costs only the extra idle seconds ($0.01–0.02 a room). To wire it in,
   `splatroom.build` (or `ModalSplatTrainer`) would start a thread that drains
   `function.remote_gen(b"", {})` before computing the instant room. That is
   `pipeline/splatroom/__init__.py` and `models/splat/remote.py`, which were outside this task.
2. **`min_containers=1`**: no cold boot ever, but one H100 kept warm costs about
   **$4.60/h (~$110 a day, ~$3,300 a month)**. Only worth it during a demo or study session, for
   example `train.update_autoscaler(min_containers=1)` at the start and 0 at the end. The
   warm-up call gets the same benefit for cents.
3. **`scaledown_window`**: each 60 s of idle after a run costs $0.076. That is now about 40%
   of a room's $0.15. Lowering it to 30 s would save about $0.04 a room, but the warm-up call
   needs the container to stay up until the real call arrives (~10–20 s later, more for a large
   scan), so it stays at 60 s.
4. Not done: keeping a pre-imported trainer process in the container (it would save the ~5 s of
   imports, but only on a container that is already warm), and Modal memory snapshots (the trainer
   runs as a subprocess, so its imports are not captured).

## What changed

- `deploy/modal/splat/app.py`:
  - New `DEFAULTS`: `steps` 1250, `batch_size` 4, `snapshots` [400, 800, 1250],
    `fused_bilagrid`, `visible_adam` and `scale_schedule` on.
  - New options: `batch_size`, `fused_bilagrid`, `packed`, `visible_adam`,
    `scale_schedule`, and `trainer_args` (extra trainer flags passed through).
  - `--tb-every 0`.
  - The trainer launches before the parent imports torch.
  - Polling every 0.2 s instead of 0.5 s.
  - `timing` in the "done" event.
  - The `train(b"")` warm-up call.
  - Deployed to `main`.
- `src/litereality_agent/models/splat/dataset.py`:
  - `Dataset.tar()` is gzipped.
  - Frames with fewer than 20 usable LiDAR points are left out of the training set, and the
    rest are numbered 1..n. ValueError when none is left. This fixes a gsplat `KeyError` in
    the depth loss; the test is in `tests/test_splat.py`.
- `litereality splatroom --steps N` still works. Its help text says "default 7000", which is
  now wrong (`cli.py`). Note that any `--steps` is now N steps of 4 photos.

## Cost of this study

17 trainer runs (15 in the sweep, 2 end to end) and a few small probes, about **$2.8–3.0** in all. Modal's billing report for
`litereality-splat` shows $2.24 in the 16:00 UTC hour (which may include other sessions' runs
of the app in that hour) and $0.84 after 17:00.
