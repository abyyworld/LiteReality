# Eyeful Tower office1b: is our splat pipeline the bottleneck?

## 1. The question

Our phone rooms look worse than published splats. Is that the capture, or our pipeline?

To find out, we trained our pipeline on a room captured by professionals, Meta's Eyeful Tower
(MIT licence), and compared the result with the published 3DGS score under the same protocol.

- Near the published number: the trainer is fine and capture is the limit.
- Far below: our pipeline loses quality.

## 2. Setup

**Scene.** office1b, an indoor office from Eyeful v2. It has 22 pinhole cameras, 71 rig
positions and 1562 photos.

**Protocol.** We matched the published protocol as closely as the papers allow.

- Split: the official `splits.json`. All 71 photos from camera 17 are held out for testing. The
  other 1491 photos are for training. No test photo is used for training (the log shows
  `train: 1491 / val: 71 (test_images.txt)`).
- Resolution: about 2K. We use the official undistorted COLMAP export (`images_4`, PINHOLE,
  about 1477x2250, test camera 1478x2232). We do not downscale it (data factor 1).
- Colour: plain sRGB JPEGs, as for the paper's 3DGS baseline. We report plain PSNR. The
  colour-corrected PSNR is only for reference.
- Metrics: PSNR, SSIM, and LPIPS (AlexNet). We use torchmetrics, as gsplat does. Each score is
  the mean over the 71 test photos, on a black background.

**Published reference.** HybridNeRF (CVPR 2024) Table 1 gives the pinhole-scene **mean**. No
per-scene 3DGS or Zip-NeRF numbers exist for Eyeful Tower. We searched the paper, its
supplement, the thesis and related papers.

**Arms.** Both arms use gsplat 1.5.3 on a Modal H100.

- **A: gsplat `default`.** This is the standard 3DGS recipe: 30k steps, batch 1, SH3, data
  factor 1.
- **B: our room settings.** MCMC, batch 4, cap 800k, fused bilateral grid, antialiased, visible
  Adam, scaled schedules, no world renormalisation, and depth off. We ran 1250 steps (the deployed
  default) and 5000 steps. The 10k version would not fit the budget.

**Browser.** We exported each arm to SPZ and rendered it with Spark 2.3.1 in headless Chromium,
using the Mac GPU (Metal). We rendered all 71 test cameras at 1:1. The page waited for each view
to settle before reading it.

- "Call" uses the room call's own Spark settings: maxStdDev sqrt(5), Spark defaults otherwise,
  near plane 0.05.
- "Matched" sets Spark as close as it can get to gsplat's renderer.

## 3. Results

| what | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Published 3DGS (mean of pinhole scenes) | 27.42 | 0.877 | 0.291 |
| Published Zip-NeRF (same mean) | 29.71 | 0.868 | 0.305 |
| Published HybridNeRF (same mean) | 29.07 | 0.880 | 0.268 |
| **A, gsplat default 30k, trainer render** | **27.57** | **0.884** | **0.341** |
| A, Spark, call settings | 27.03 | 0.873 | 0.359 |
| A, Spark, matched settings | 27.41 | 0.879 | 0.342 |
| **B_5000, room settings, trainer render** | **24.94** | **0.868** | **0.394** |
| B_5000, Spark, call settings | 22.74 | 0.825 | 0.390 |
| B_5000, Spark, matched settings | 24.89 | 0.859 | 0.372 |
| **B_1250, room settings (deployed length), trainer render** | **23.88** | **0.849** | **0.515** |
| B_1250, Spark, call settings (SH3 / SH1 as deployed) | 19.89 / 19.84 | 0.755 | 0.534 |
| B_1250, Spark, matched settings | 22.29 | 0.815 | 0.497 |

**How to read the Spark rows.** Each Spark gap is measured against the trainer's own saved JPEG
renders, scored with the same code. For A those score 27.53 / 0.880 / 0.335.

| Spark minus trainer, PSNR | call settings | matched settings |
|---|---|---|
| A_30000 | -0.50 dB | -0.12 dB |
| B_5000 | -2.21 dB | -0.06 dB |
| B_1250 | -3.98 dB | -1.58 dB |

**Numbers I re-checked.** I recomputed PSNR from the saved files with independent numpy code.

- A: trainer 27.53, Spark call 27.01. Step 3 reported 27.03; the 0.02 dB difference likely comes
  from the saved Spark JPEGs being recompressed.
- B_5000: trainer 24.95, Spark call 22.76.
- B_1250: trainer 23.87, Spark call 19.90.
- The ground truth inside the trainer's canvases matches the ground-truth files at 43.5 dB, which
  is JPEG noise only. So the views are aligned and in the same order.

## 4. Verdict

**Is our trainer fine?** Yes. With gsplat's default 3DGS recipe it scores 27.57 dB on office1b,
against the published 3DGS mean of 27.42 dB. SSIM matches too, at 0.884 vs 0.877. The gsplat
code, our data loading, and our cameras are not the problem.

**Do our room settings lose quality?** Yes, a lot. On the same data they score 24.94 dB at 5k
steps and 23.88 dB at the deployed 1250 steps. That is 2.6 to 3.7 dB below the default recipe.
Three things stand out. This is not a clean ablation, because B differs from A in several
settings at once.

- **The bilateral grid absorbs colour that the exported splats do not keep.** B's
  colour-corrected PSNR is close to A's at 7k steps. Its plain PSNR is 1.9 dB lower.
- **The splat count never grows.** MCMC starts from the 1.29M tie points, which is already above
  the 800k cap, so it never adds splats.
- **The deployed run is short.** 1250 steps at batch 4 is far less training than the standard
  30k.

**Does the browser lose quality?** For a well-trained splat (A), only a little: 0.5 dB with the
room call's settings and 0.1 dB at best. For our room-setting splats (B), it loses another 2.2 to
4 dB. Most of that loss comes from Spark's 512 px cap on splat size (maxPixelRadius). Lifting only
that cap takes B_5000 from -2.21 dB to -0.30 dB. The setting is in
`room_ops/roomcall/web/app.js`, and I have not changed it.

**Bottom line.** On professional data, the path as deployed (room settings, 1250 steps, SH1,
Spark call settings) gives **19.8 dB, against 27.4 dB published**. The trainer code is fine. The
loss comes from our training settings and our Spark settings. Capture may still limit the phone
rooms too, but this test cannot measure that, and the pipeline losses are big enough to fix
first.

## 5. Caveats

- **The comparison is one scene against a mean.** The published number is a mean over (probably)
  6 pinhole scenes, and the per-scene spread is unknown. Across A's own 71 test views, PSNR ranges
  from 19.8 to 32.0 dB. So "matches" means "no sign of a gap", not "identical".
- **LPIPS is about 0.05 worse than published** (0.341 vs 0.291), while PSNR and SSIM match. No
  metric code was released, so the gap may come from how LPIPS was computed: VGG vs AlexNet, input
  scaling, or image size. It may also be scene spread. Our LPIPS was still improving a lot late in
  training (0.455 at 7k, 0.341 at 30k).
- **The images may differ from the paper's.** We use the undistorted COLMAP export (about
  1477x2250). The paper says "2K" and does not say how 3DGS handled lens distortion. The raw 2K
  JPEGs are 1368x2048 and still distorted. The pixel counts are similar, but the effect is
  unknown.
- **The SfM points include the test photos.** The Metashape tie points and the test poses come
  from one reconstruction that included camera 17. This is the usual practice (Mip-NeRF 360 does
  the same), and it only affects initialisation. The test photos' pixels are never trained on.
- **The test camera has its own colour.** Camera 17 is never trained on. Even for A,
  colour-corrected PSNR is 1.9 dB higher than plain PSNR. Published methods face the same problem,
  so the comparison is still fair.
- **B was not trained as long as A.** B_5000 saw 20k photos and A saw 30k. B differs from A in
  MCMC, cap, batch, bilateral grid and antialiasing all at once.
- **The browser scores come from one machine.** Spark was scored on Metal, on this Mac. The older
  `score.py` path uses SwiftShader and reads each view after 6 frames. For large rooms, that can
  read a view before its depth sort is finished: A view 40 scored 24.3 dB stale and 28.0 dB
  settled. Earlier `score.py` numbers for big rooms may be off and are worth re-checking.
- **One MPS metric glitch was caught and fixed.** On the Mac GPU, torchmetrics SSIM once returned
  values above 1. Step 3 now computes PSNR and SSIM in float64 numpy and double-checks LPIPS. The
  early 7-view runs in `spark/explore/` predate this fix.
- **The coordinate frames differ.** A's splats are in gsplat's renormalised frame; the 4x4
  transform is in `A_30000/result.json`. B's splats are in the COLMAP frame. The A-to-Spark
  agreement of 32.3 dB PSNR shows that the cameras and intrinsics are right in both renderers.

## 6. Modal cost and the Volume

**Cost.** The Modal billing report, summed by app for today, gives **$1.865** for the whole
workflow, under the $2 cap.

| app | cost |
|---|---|
| eyeful-train | $1.840 |
| eyeful-verify | $0.017 |
| eyeful-prepare | $0.007 |
| eyeful-fetch-eval | $0.0003 |

This review step ran no Modal containers; it only read billing, app and Volume listings. No
eyeful app is running, and nothing was deployed.

**Volume `litereality-eyeful` (env main).** It holds about 5.5 to 6 GB.

| path | contents | size |
|---|---|---|
| `/office1b/raw/` | distorted 2K JPEGs, full-res COLMAP, cameras.json, splits.json | about 2 GB |
| `/office1b/gsplat_2k/` | the training dataset | about 2.5 GB |
| `/office1b/runs/{A_30000,B_5000,B_1250}/` | `point_cloud.ply` (A 396 MiB, each B about 290 MiB), SPZ, 71 renders, logs | about 1.2 GB |
| `/office1b/spark_eval/` | test cameras and 71 ground-truth JPEGs | about 90 MB |

**Should it be deleted?**

- Delete `/office1b/raw/` now. Nothing uses it.
- Keep `gsplat_2k/` and `runs/` only if a follow-up run is planned soon. One example is
  re-training B with a cap above the seed count and the grid colour baked in, then scoring it in
  Spark with the size cap lifted.
- Otherwise delete the whole Volume with `modal volume delete litereality-eyeful --env main`. The
  SPZ files, metrics and renders are already copied under `results/eyeful/`. Only the PLY files
  live solely on the Volume.

## Files

- Scripts in `scripts/eyeful/`:
  - `prepare_eyeful.py`: download and build.
  - `verify_eyeful.py`: the pycolmap check.
  - `train_eyeful.py`: the training app.
  - `patch_colmap_split.py`: the official split, the training crop and JPEG eval output.
  - `spark_score.py`, `render_splat_eyeful.html`, `run_spark*.sh` and `spark_summary.py`: browser
    scoring.
  - `fetch_eval.py` and `spz_variant.py`.
- Results in `results/eyeful/`:
  - `step1_prepare.json`, `step2_train.json`, `step3_spark.json` and `step3_spark.md`.
  - `compare.jpg`.
  - Per-run folders: `A_30000/`, `B_5000/` and `B_1250/`. Each holds `result.json`, the trainer
    log, stats, 4 renders and the SPZ files.
  - `spark/`: per-view Spark results.
- Local eval inputs and Spark renders, outside the repo: `~/LiteReality-scans/eyeful_eval/`.
- Size: `results/eyeful/` is about 122 MB, mostly SPZ files of 19 to 30 MB each. Decide before
  committing them.
- Lint: `prepare_eyeful.py` still has 2 old ruff warnings (F401, E731).

## Where the splats are

The exported SPZ files (19 to 30 MB each) are kept out of git, in `~/LiteReality-scans/eyeful_eval/spz/<run>/`.
