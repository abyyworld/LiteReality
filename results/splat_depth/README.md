# Dense depth for the photoreal room: PromptDA, depth and normal losses, cleanup

This tests whether the photoreal room (the trained Gaussian splat) gets better on views the scan barely saw if we use PromptDA's dense depth in place of the 256 x 192 LiDAR alone. It was run on three rooms: tea_room, office_room and suite_3693. Run on 2026-11-22. Every number comes from **one training run**, with no seed averaging. The noise estimate is the repeated runs marked "repeat" below.

## Recommendation (in short)

- **Make this the default trained configuration:** `--depth promptda --depth-loss-dense --normal-lambda 0.05`, with the default `--dense-points` (the 800k splat budget shared over the photos, about 13k points per photo).
  - Mean trainer held-out score: **PSNR +0.48 dB, LPIPS −0.027, SSIM +0.003**.
  - Spark render score (what users see): **PSNR +0.82 dB, SSIM +0.033**.
  - Size is unchanged: 800k splats, about 14 MB.
  - PSNR improves or stays level in every room, and LPIPS improves in every room. The repeated runs reproduce these numbers within 0.05 dB.
- **Do not use `--dense-points 20000`.** It puts about 1.2M splats past the 800k cap, makes the SPZ 21 MB, adds about 30 s, and lowers SSIM by about 0.05 on its own.
- **Do not turn `--clean` on for trained snapshots.** It costs 0.2–0.9 dB on held-out Spark renders in every room.
  - Splitting it by rule: carving (2 votes) is the main cost in both rooms tested, the outside rule costs 0.3 dB on tea_room, and the faint rule changes nothing.
- **Do clean the instant room.** Cleaning the dense instant room gives **+2.4 dB** in Spark (mean 15.38 → 17.80). That is better than the current *trained* base room (16.74).
  - This needs a small code change: clean only the instant stage. I did not change the default or the code.
- **Cost:** the trained room arrives about 35 s later (time to final snapshot 75 s → 109 s; first trained snapshot 48 s → 73 s). Details are under "Where the time goes".
  - The instant room is not delayed when depth has to be computed: PromptDA runs beside it. It *is* about 4 s later when depth is already cached, because the dense instant room takes longer to fuse.
  - Whether the trained-room delay counts as "users wait" is a product call. The room on screen during that time is the instant room, and with dense depth plus cleaning that instant room is better than today's trained room.

## Setup

- **Trainer:** gsplat 1.5.3 MCMC on one H100, 1250 steps, batch 4, held-out photo every 8th (`test_every 8`), cap 800k. Every run used `--no-masks`, which affects only the cut, never `raw.spz.bin`.
- **Rooms:** tea_room (62 photos), office_room (59), suite_3693 (63), against `run/<room>/objects/scene.xml`. Each variant was written to its own `run/<room>/splat_abl_<name>/`.
- **Depth:** PromptDA ViT-L on Modal (`litereality-depth`; it ran on an A10G, the first free GPU of L4/A10G/L40S), at 960 x 720, cached in `<scan>/depth_dense/`.
  - Every ablation run read the cache, so its "first room" is the *dense* instant room. In a fresh capture the LiDAR room is shown first, at the base time.
- **Trainer metrics:** gsplat's own held-out PSNR / SSIM / LPIPS on 8 held-out photos per room. These are computed before `clean()`, so they never reflect cleaning.
- **Spark metrics** (`pipeline/evaluate/fidelity.evaluate_splat_fidelity`): Spark 2.3.1 renders in headless Chromium, scored at 320 x 240 against the same held-out photos only. Each snapshot takes about 7 minutes in software WebGL. Spark does not score LPIPS.
  - "walls+floor" is the static-region PSNR.
  - **Known issue:** the first view (frame 0) renders blank grey in every snapshot, in all three rooms. That lowers every absolute Spark number by about 1 dB, but it is the same for every snapshot, so the deltas are unaffected. It looks like a first-render race in `SplatRenderer` / `render_splat.html` and is worth fixing before quoting absolute numbers.

## 1. Depth sanity check (PromptDA vs LiDAR at valid LiDAR pixels)

| room | median abs diff | p90 abs | median rel | median abs, conf=2 | within max(5 cm, 5%) | refine wall time (62 frames) | GPU per frame |
|:-|-:|-:|-:|-:|-:|-:|-:|
| tea_room | 1.0 cm | 4.9 cm | 0.7 % | 0.9 cm | 95 % | 32 s (cold) | 0.66 s |
| office_room | 1.8 cm | 7.6 cm | 0.9 % | 1.5 cm | 93 % | 17 s (warm) | 0.66 s |
| suite_3693 | 1.9 cm | 5.6 cm | 1.7 % | 1.8 cm | 94 % | 17 s (warm) | 0.66 s |

These are within the expected "few cm". Side-by-sides (photo | LiDAR | dense, same colour scale) are `<room>_photo_lidar_dense.jpg` in this folder.

- PromptDA keeps the LiDAR's scale and adds sharp object edges.
- It invents structure behind glass (tea_room's door panes). The dataset's agreement check exists to reject that.

## 2. Ablation: trainer held-out metrics (mean of the three rooms)

Times are seconds from the start of the run. "first room" is the first instant room on screen (the dense one for promptda runs, because depth was cached). "first trained" is the first trained snapshot (step 400), and "trained" is the final one (step 1250).

| variant | PSNR | SSIM | LPIPS | ΔPSNR | ΔSSIM | ΔLPIPS | first room s | first trained s | trained s | MB |
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
| base (LiDAR, 4000 pts/photo) | 19.20 | 0.732 | 0.412 | +0.00 | +0.000 | +0.000 | 9.0 | 47.5 | 75.2 | 14.1 |
| base, repeat | 19.27 | 0.731 | 0.411 | +0.07 | -0.001 | -0.001 | 10.8 | 50.5 | 78.1 | 14.0 |
| promptda (20000 pts/photo) | 19.13 | 0.681 | 0.438 | -0.07 | -0.051 | +0.026 | 12.8 | 78.2 | 105.6 | 21.1 |
| promptda + dense loss | 19.27 | 0.684 | 0.429 | +0.07 | -0.048 | +0.017 | 13.0 | 84.1 | 119.4 | 21.0 |
| promptda + dense loss + normals | 19.62 | 0.727 | 0.377 | +0.41 | -0.005 | -0.034 | 13.2 | 86.4 | 124.3 | 21.2 |
| promptda (budget pts) + normals | 19.56 | 0.735 | 0.386 | +0.35 | +0.003 | -0.026 | 15.6 | 77.4 | 113.4 | 13.9 |
| **promptda (budget pts) + dense loss + normals** | 19.68 | 0.735 | 0.385 | +0.48 | +0.003 | -0.027 | 13.1 | 72.9 | 108.8 | 13.9 |
| same, repeat | 19.68 | 0.735 | 0.385 | +0.48 | +0.003 | -0.027 | 15.8 | 76.9 | 109.0 | 13.9 |

Per room (Δ is against that room's first base run):

### tea_room

| variant | PSNR | SSIM | LPIPS | ΔPSNR | ΔLPIPS | first room s | first trained s | trained s | splats | MB | pts/photo |
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
| base (LiDAR, 4000 pts/photo) | 20.02 | 0.782 | 0.364 | +0.00 | +0.000 | 8.4 | 47.1 | 74.7 | 800,000 | 14.0 | 4,000 |
| base, repeat | 20.21 | 0.784 | 0.363 | +0.19 | -0.002 | 10.2 | 49.8 | 75.0 | 800,000 | 13.9 | 4,000 |
| promptda (20000 pts/photo) | 20.26 | 0.735 | 0.390 | +0.24 | +0.025 | 11.8 | 76.9 | 103.2 | 1,240,000 | 21.1 | 20,000 |
| promptda + dense loss | 20.25 | 0.733 | 0.383 | +0.23 | +0.019 | 11.4 | 81.1 | 112.9 | 1,240,000 | 21.1 | 20,000 |
| promptda + dense loss + normals | 20.62 | 0.771 | 0.341 | +0.59 | -0.024 | 11.6 | 81.7 | 118.5 | 1,240,000 | 21.3 | 20,000 |
| promptda (budget pts) + normals | 20.68 | 0.781 | 0.353 | +0.66 | -0.012 | 15.8 | 77.5 | 111.0 | 800,000 | 13.8 | 12,903 |
| **promptda (budget pts) + dense loss + normals** | 20.67 | 0.779 | 0.352 | +0.64 | -0.013 | 11.6 | 69.2 | 103.6 | 800,000 | 13.8 | 12,903 |
| same, repeat | 20.63 | 0.778 | 0.353 | +0.60 | -0.012 | 15.8 | 76.9 | 110.0 | 800,000 | 13.8 | 12,903 |

### office_room

| variant | PSNR | SSIM | LPIPS | ΔPSNR | ΔLPIPS | first room s | first trained s | trained s | splats | MB | pts/photo |
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
| base (LiDAR, 4000 pts/photo) | 19.22 | 0.661 | 0.375 | +0.00 | +0.000 | 7.9 | 46.5 | 75.5 | 800,000 | 14.3 | 4,000 |
| base, repeat | 19.23 | 0.661 | 0.374 | +0.01 | -0.001 | 10.2 | 49.8 | 79.3 | 800,000 | 14.3 | 4,000 |
| promptda (20000 pts/photo) | 18.84 | 0.622 | 0.369 | -0.38 | -0.006 | 12.7 | 76.9 | 105.0 | 1,180,000 | 20.7 | 20,000 |
| promptda + dense loss | 18.85 | 0.625 | 0.360 | -0.37 | -0.015 | 12.9 | 82.5 | 121.6 | 1,180,000 | 20.6 | 20,000 |
| promptda + dense loss + normals | 19.13 | 0.653 | 0.330 | -0.08 | -0.045 | 13.0 | 86.9 | 127.8 | 1,180,000 | 20.9 | 20,000 |
| promptda (budget pts) + normals | 19.20 | 0.658 | 0.346 | -0.01 | -0.030 | 16.6 | 82.3 | 119.8 | 800,000 | 14.3 | 13,559 |
| **promptda (budget pts) + dense loss + normals** | 19.24 | 0.659 | 0.343 | +0.02 | -0.032 | 13.6 | 75.8 | 115.0 | 800,000 | 14.2 | 13,559 |
| same, repeat | 19.23 | 0.658 | 0.343 | +0.01 | -0.032 | 16.7 | 84.0 | 121.0 | 800,000 | 14.2 | 13,559 |

### suite_3693

| variant | PSNR | SSIM | LPIPS | ΔPSNR | ΔLPIPS | first room s | first trained s | trained s | splats | MB | pts/photo |
|:-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
| base (LiDAR, 4000 pts/photo) | 18.37 | 0.752 | 0.496 | +0.00 | +0.000 | 10.6 | 48.9 | 75.6 | 796,303 | 13.9 | 4,000 |
| base, repeat | 18.37 | 0.749 | 0.496 | +0.01 | -0.000 | 11.9 | 52.0 | 80.0 | 796,288 | 13.9 | 4,000 |
| promptda (20000 pts/photo) | 18.28 | 0.687 | 0.556 | -0.09 | +0.059 | 13.9 | 80.8 | 108.7 | 1,260,000 | 21.3 | 20,000 |
| promptda + dense loss | 18.71 | 0.694 | 0.544 | +0.35 | +0.048 | 14.6 | 88.7 | 123.8 | 1,260,000 | 21.3 | 20,000 |
| promptda + dense loss + normals | 19.10 | 0.756 | 0.462 | +0.73 | -0.034 | 15.2 | 90.7 | 126.6 | 1,260,000 | 21.5 | 20,000 |
| promptda (budget pts) + normals | 18.78 | 0.767 | 0.460 | +0.42 | -0.036 | 14.3 | 72.4 | 109.3 | 800,000 | 13.7 | 12,698 |
| **promptda (budget pts) + dense loss + normals** | 19.13 | 0.768 | 0.459 | +0.76 | -0.037 | 14.0 | 73.6 | 107.6 | 800,000 | 13.7 | 12,698 |
| same, repeat | 19.18 | 0.768 | 0.460 | +0.81 | -0.036 | 14.8 | 69.7 | 95.8 | 800,000 | 13.7 | 12,698 |

What the variants show:

- **Dense points alone hurt** (20000/photo): about 1.2M splats, past the cap, so MCMC cannot grow or relocate freely. SSIM drops by 0.04–0.07 in every room, PSNR ±0.4 dB.
- **The dense L1 depth loss alone** does almost nothing on tea and office, and gives +0.4 dB on suite.
- **The normal loss is the main gain:** +0.3 to +0.4 dB and LPIPS −0.03 to −0.08 on top of the dense L1, in all three rooms.
- **Budget-matched points** (the default `dense_points_for`, about 13k/photo) with dense L1 and normals beat the same losses at 20000 points on SSIM and size, with equal or better PSNR. This is the recommended setting.
- **Normals without the dense L1** (`auto_n`) equals the recommended setting on tea and office, but is 0.35 dB worse on suite. Keep both losses.
- **Noise:** base repeats differ by 0.19 dB (tea) and 0.01 dB (office, suite). Repeats of the recommended setting differ by ≤ 0.05 dB. The office gain in PSNR (+0.02) is within noise; its LPIPS gain (−0.032, twice) is not.

## 3. What users see: Spark renders of held-out photos

Instant rooms are `rgbd.instant_splats` at the pipeline's voxel (LiDAR, or PromptDA from the cache), each scored raw and after `clean.clean` (room bounds from RoomPlan, carving with training photos only). Trained rooms are each run's `raw.spz.bin` and `raw.clean.spz.bin`.

| snapshot | tea_room PSNR / SSIM | office_room PSNR / SSIM | suite_3693 PSNR / SSIM | mean PSNR | mean SSIM | mean walls+floor PSNR | Δ mean PSNR vs base |
|:-|-:|-:|-:|-:|-:|-:|-:|
| instant_lidar | 17.35 / 0.700 | 14.67 / 0.606 | 13.64 / 0.625 | 15.22 | 0.644 | 15.97 | -1.52 |
| instant_lidar_clean | 17.34 / 0.700 | 14.82 / 0.604 | 15.81 / 0.657 | 15.99 | 0.654 | 17.31 | -0.75 |
| instant_promptda | 18.44 / 0.694 | 14.86 / 0.573 | 12.84 / 0.604 | 15.38 | 0.624 | 15.70 | -1.36 |
| instant_promptda_clean | 18.93 / 0.700 | 16.93 / 0.607 | 17.54 / 0.695 | 17.80 | 0.667 | 18.46 | +1.06 |
| base | 17.86 / 0.701 | 16.40 / 0.591 | 15.95 / 0.620 | 16.74 | 0.637 | 17.50 | +0.00 |
| base_clean | 16.97 / 0.682 | 16.18 / 0.571 | 15.60 / 0.589 | 16.25 | 0.614 | 17.12 | -0.49 |
| auto_dl_n | 18.54 / 0.718 | 17.16 / 0.626 | 16.98 / 0.667 | 17.56 | 0.670 | 18.15 | +0.82 |
| auto_dl_n_clean | 17.88 / 0.705 | 16.49 / 0.596 | 16.67 / 0.642 | 17.02 | 0.647 | 17.77 | +0.28 |

- **Trained:** the recommended setting is better than base in every room, by +0.68 / +0.76 / +1.03 dB, a larger gain than the trainer's own metric shows. Visually, the contact sheets (`spark/<room>_<snapshot>/contact_sheet.jpg`) show less floor smear and fewer floaters around the chairs in office_room.
- **Instant, dense vs LiDAR, uncleaned:** mixed, +1.1 / +0.2 / −0.8 dB. The dense instant room has about 35 % more surfels, including PromptDA's mistakes beyond the walls and through windows.
- **Instant, cleaned:** cleaning removes those surfels (suite: 108k outside, 101k carved) and the dense instant room jumps to 17.80 mean, above the trained base room. Cleaning also helps the LiDAR instant room: +2.2 dB on suite, +0.15 dB on office, and neutral on tea.
- **Trained, cleaned:** cleaning hurts every trained room (−0.2 to −0.9 dB).

## 4. Why cleaning hurts trained rooms (recommended snapshot, one rule at a time)

| snapshot | tea_room PSNR / SSIM | office_room PSNR / SSIM | mean PSNR | mean SSIM | mean walls+floor PSNR | Δ mean PSNR vs base |
|:-|-:|-:|-:|-:|-:|-:|
| auto_dl_n | 18.54 / 0.718 | 17.16 / 0.626 | 17.85 | 0.672 | 18.11 | +0.72 |
| rule_faint_only | 18.54 / 0.718 | 17.16 / 0.626 | 17.85 | 0.672 | 18.11 | +0.72 |
| rule_outside_only | 18.20 / 0.714 | 17.14 / 0.625 | 17.67 | 0.669 | 18.11 | +0.54 |
| rule_carve_only | 18.27 / 0.707 | 16.52 / 0.596 | 17.39 | 0.652 | 17.70 | +0.26 |
| rule_carve3_outside | 18.03 / 0.709 | 17.68 / 0.597 | 17.86 | 0.653 | 18.27 | +0.72 |
| auto_dl_n_clean | 17.88 / 0.705 | 16.49 / 0.596 | 17.19 | 0.650 | 17.70 | +0.06 |

- **Faint** (opacity < 0.02; 137k–170k splats) changes nothing. It is safe, but it only saves size: about 17–21 % fewer splats.
- **Outside** (beyond the RoomPlan walls + 15 cm) costs 0.34 dB on tea_room and nothing on office_room (−0.02). Trained splats past the walls can be real content that the held-out photos see: window views, a doorway, the far side of open doors.
- **Carving** (2 votes) costs 0.27 dB (tea) and 0.64 dB (office), with SSIM lower in both. Trained splats in front of the measured depth are often legitimate: thin chair legs and edges where LiDAR or PromptDA depth is blurred, and semi-transparent splats the optimiser uses for view-dependent colour.
- **Carving with 3 votes plus outside** disagrees between the rooms: −0.51 dB on tea, and +0.52 dB on office but with SSIM −0.029. With 7 scoring views per room, this is not a result to act on.
- **Suggested change (not made):** keep `clean()` for the instant room, and for trained snapshots at most the faint rule, for size. To bring carving back for trained rooms, it needs a stricter test, for example several votes *and* low opacity, or votes counted only where PromptDA and LiDAR agree.

## Where the time goes (office_room, base → recommended)

| step | base | recommended | Δ |
|:-|-:|-:|-:|
| instant room fused | 7.9 s | 13.6 s | +5.7 s (dense surfels, 960 x 720 instead of 256 x 192) |
| dataset built | 3.2 s | 7.9 s | +4.7 s |
| dataset tar | 21 MB | 66 MB | +45 MB (uint16 depth PNGs at 960 x 720) |
| trainer started | 15.3 s | 29.3 s | +14 s |
| trainer: step 400 / 1250 after the call | 26 s / 54 s | 41 s / 79 s | +15 s upload and setup, then about 35 % slower per step |
| final trained snapshot | 75 s | 115 s | +40 s |

Likely ways to recover most of this (not tried):

- Ship the depth maps at half resolution, or as 8-bit disparity, because the upload dominates on a slow link.
- Fuse the dense instant room off the critical path, showing the LiDAR room first even when depth is cached.
- Compute the normal loss on every second pixel.

## Costs actually spent (Modal billing, 17:00 UTC onward)

| item | cost |
|:-|-:|
| PromptDA image build + refining 3 scans (A10G) | $0.25 |
| splat image rebuild (patch layers only; the GPU-compiled layers were cached) + 24 H100 training runs | $3.56 |
| **total** | **$3.81** of the $6 budget |

The workspace's billing also shows $1.51 of `litereality-splat` H100 time in the 16:00 UTC hour, before this experiment's first Modal call. It is not counted above and is not from these runs.

## Files

- `ablation.json`: every run's numbers (from `run/<room>/splat_abl_*/report.json`)
- `spark/<room>_<snapshot>.json` and `spark/<room>_<snapshot>/contact_sheet.jpg`: Spark scores and render sheets (held-out views)
- `instant/`: instant-room SPZs (LiDAR / PromptDA, raw and cleaned) and `instant.json` (sizes, fuse times, clean stats)
- `clean_rules/`: single-rule cleaned snapshots and their stats
- `smoke_*.json`, `<room>_photo_lidar_dense.jpg`: the depth check
- `logs/`: every splatroom and evaluation log

The `splat_abl_*_clean` folders are copies of their trained run, recut with `--recut --clean`, with no new training. Their trainer metrics are therefore those of the uncleaned run.

## Side effects to know about

- **New cache folders** were written to each scan: `<scan>/depth_dense/`, including `~/LiteReality-scans/suite_3693/depth_dense/`.
- **`scene_splat.xml` was rewritten:** `splatroom` writes `run/<room>/objects/scene_splat.xml` (the moved-bodies scene) on every run, so the ablation runs rewrote it.
- **Nothing published:** nothing was published to the room link, and no defaults were changed.
