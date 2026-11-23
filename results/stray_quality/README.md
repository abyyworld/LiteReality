# Atrium (Stray recording) splat quality: v0..v3 on one fixed held-out set

Nothing here was published or deployed. All four variants were trained on Modal (task 2) and scored locally (task 3, CPU only).

## How it was judged (fairness)

- **Eval set.** `eval_frames.txt` holds 29 frames spread over the 58 s recording, picked from the pool import (`~/LiteReality-scans/stray_atrium_pool`).
- **No variant trained on them.** `check_fair.py` writes `fairness.json`:
  - Checked by recording row: 0 eval rows in `stray_atrium_q206` (v0, 188 frames) and 0 in `stray_atrium_q600` (v1/v2/v3, 637 frames).
  - Checked by time: the nearest training frame is at least 7 rows (0.2 s) away.
  - Checked by pose: the nearest training camera is a median 4.3 cm (q206) and 3.8 cm (q600) from the eval camera.
  - Some eval views are still close to a training view because the walk paused there. 8 of 29 (q206) and 11 of 29 (q600) have a training camera within 5 cm and 5°.
  - `subset.py` therefore also scores the 18 "far" frames that have no such neighbour in either scan (`subset.md`). The ranking does not change.
- **Same renderer and resolution for every variant.** Spark 2.3.1 / three.js 0.180 in headless Chromium (SwiftShader), which is the room call's own renderer, through `fidelity.SplatRenderer`.
  - Each eval frame is rendered at 960×720 and box-filtered to 480×360.
  - Each render uses the exported `raw.spz.bin` at SH degree 1, which is what viewers see.
  - Eval poses are mapped from the pool's alignment into each scan's alignment: 3.4 mm for q206 and 0.7 mm for q600.
- **Wander views.** 3 fixed room-frame cameras, rendered at 960×720 from the **served** room: the room splat plus the body splats placed by the corrected physics.

## Results (`table.md`, `subset.md`)

| variant | frames | train width | steps | SH (train) | PSNR | PSNR cc | SSIM | LPIPS ↓ | PSNR win/door (cc) | PSNR chairs/table (cc) | far-18 PSNR cc | far-18 LPIPS | trainer GPU s | wall s | GPU $ (list est. / billed upper bound) | splats | raw MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | 188 | 960 | 5000 | 3 | 19.15 | 21.65 | 0.738 | 0.286 | 20.76 | 20.17 | 22.34 | 0.257 | 152 | 305 | 0.19 / 0.38 | 800,000 | 14.5 |
| v1 | 637 | 960 | 7000 | 3 | 19.97 | 22.23 | 0.759 | 0.267 | 21.20 | 20.92 | 22.68 | 0.241 | 262 | 589 | 0.34 / 0.54 | 2,228,000 | 39.5 |
| **v2** | 637 | **1440** | 10000 | 3 | **20.19** | **22.53** | **0.767** | 0.262 | **21.46** | 21.01 | **23.01** | 0.237 | 588 | 1231 | 0.65 / 1.02 | 2,228,000 | 40.7 |
| v3 | 637 | 960 | 7000 | **1** | 20.05 | 22.31 | 0.761 | **0.258** | 21.25 | **21.39** | 22.72 | **0.234** | 271 | 569 | 0.30 / 0.55 | 2,228,000 | 40.0 |

Paired per-view differences, as mean ± standard error over the 29 views:

- **v1 − v0** (more frames):
  - PSNR cc +0.58 ± 0.16 dB, better on 22 of 29 views.
  - LPIPS −0.020 ± 0.005, better on 25 of 29.
  - This is the only big step.
- **v2 − v1** (1440 px, 10k steps):
  - PSNR cc +0.30 ± 0.06 dB, better on 26 of 29.
  - Windows/door +0.26 dB.
  - LPIPS −0.005 ± 0.003.
- **v3 − v1** (SH1):
  - PSNR cc +0.08 ± 0.06 dB, which is noise.
  - LPIPS −0.009 ± 0.001, better on 27 of 29.
  - Chairs/table +0.48 ± 0.13 dB.
- **v2 vs v3:**
  - v2 is +0.22 dB on PSNR cc and +0.22 dB on windows/door.
  - v3 is +0.38 dB on chairs/table (better on 13 of 15 views) and slightly better on LPIPS.

The trainer's own held-out PSNR (21.7 / 22.7 / 22.8 / 22.5) is not comparable across variants: each was scored on different frames, at a different resolution, and v0–v2 were scored at SH3.

## Chairs (physics from the same Stray capture)

Every variant moved the bodies using photo masks. The corrected scene is `run/atrium_stray_q/<v>/physics/scene_splat.xml`. Moves are dx, dy in metres:

| body | v0 | v1 | v2 | v3 |
|---|---|---|---|---|
| Chair0 | −0.044, −0.131 | −0.003, −0.180 | −0.003, −0.181 | −0.003, −0.181 |
| Chair1 (scaled 0.5) | −0.148, +0.051 | −0.174, +0.005 | −0.177, +0.005 | −0.176, +0.005 |
| Chair2 | +0.182, +0.124 | +0.049, +0.069 | +0.047, +0.068 | +0.049, +0.070 |

- The 637-frame fits agree to within 3 mm.
- **The cut is now in the right place.** The served room (bodies cut out and re-placed) differs from the uncut snapshot by only 0.6 grey levels per pixel in the chair view (0.4 % of pixels by more than 30): compare `wander/v2_chairs.jpg` with `wander/v2raw_chairs.jpg`.
- So the doubled chairs caused by cutting at the old capture's positions are gone. The smear that remains is in the trained splat itself (see below).
- Every correction has `dyaw_deg = 0`. The fit only translates, so a chair that was turned would still be cut at the wrong angle.

## Recommendation

- **Of these four, v2 is the one to publish.** It is best on PSNR, PSNR cc, SSIM, the windows/door and the stricter far subset. It is also consistent: better than v1 on 26 of 29 views.
- **v3 is the cost-effective alternative.** It costs half as much, has slightly better LPIPS and +0.4 dB on chairs/table, and is −0.2 dB elsewhere.
- **On screen, v1, v2 and v3 look almost the same** (`compare.jpg`). The real gain is v0 → 637 frames. Resolution and SH degree are each worth a few tenths of a dB.
- **What to actually publish:**
  - Retrain once with v2's settings plus `sh_degree=1` (`--train-width 1440 --steps 10000 --trainer-option sh_degree=1`) on **all** frames. The variants here leave out the eval frames and the 0.2 s around each one, about 20 % of the walk.
  - Expected cost is about $0.7–1.0.
  - Then publish that run's bundle and corrected physics. This needs the user's go-ahead.

## What is still bad (honestly: "better but still shit" still holds)

- **Under the desk and around the chair legs: grey/white blobs.** These are in the uncut snapshot too, so they are not a cutting error. That region is barely seen in the walk (low, occluded), and no variant fixes it. Only recapturing it (low passes under and around the desk) will.
- **Windows: static-like.** The glass has a patterned/frosted film and the photos look like that too (see the eval sheets). The splat reproduces it as noisy streaks with horizontal banding on the left window.
  - Windows/door PSNR is the lowest region score, 21.5 at best.
  - SH1 did not help: v3 scored the same as v1 on windows/door.
- **Carpet: speckled and smeared.** The texture is real, but at 2.2 M splats it becomes blotchy. The splat budget scales with photo count, not resolution, so v2 had no extra capacity for 1440 px.
  - Raising `cap_max` is the untested lever here.
- **Close to the camera: smears.** Expected: no training view stood that close.
- **Rainbow fringes on the door edge: not reproduced** in any wander render at the desk → door view (all variants export SH1). If they still show in the viewer, they come from another viewpoint or from the viewer, not from SH3.
- **Absolute fidelity is modest.** PSNR is about 20 (22.5 after colour fit) and LPIPS about 0.26 on unseen views.

## Files

- `table.md`, `subset.md`, `fairness.json`, `<v>_scores.json` (per-view rows, corrections, costs).
- `compare.jpg`: rows are the desk → door, windows and chairs views; columns are v0, v1, v2, v3.
- `<v>_wander.jpg`, `wander/<v>_<view>.jpg`, `wander/v2raw_*` (uncut v2), `<v>_eval_sheet.jpg`.
- Scripts: `score.py`, `pick_eval.py`, `check_fair.py`, `subset.py`, `dataset_size.py`. Scoring logs are in `logs/`.
