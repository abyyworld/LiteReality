# NVIDIA Difix3D on the atrium: no gain on real views, mixed off the walk

**Question:** does NVIDIA's diffusion clean-up (Difix3D's progressive update, `nvidia/difix_ref`)
make the live atrium room better, especially from angles the phone never filmed?

**Answer:** not as tried here.
- **Held-out real frames:** no better, and slightly softer.
- **Off the walk:**
  - Better: the blinds, the window and the door close-up are clearly cleaner. The pink and green
    blotches are gone.
  - Worse: a new beige stain appears on the white wall beside the door.
- **Verdict:** not shipped. Difix is research-only under NVIDIA's licence anyway; see
  `../any_video/PLAN.md` for the service-safe route (NVIDIA Fixer).

## Setup

**Same data as the live room.** Both arms use the live v2 room's training set, options and
10,000 steps, run on H100:
- 637 photos, 557 PromptDA depth maps, 2.23M seed points.
- Rebuilt by `scripts/difix/prepare.py`.

**The two arms:**

| arm | what |
|---|---|
| C (control) | the live room's training, again |
| B (Difix) | C plus 4 Difix rounds at steps 6000, 7000, 8000 and 9000 |

Each round of arm B works like this:
1. Render 70 novel views (see "The views" below).
2. Repair each with difix_ref, using the nearest training photo as reference, at native 1440 x 1080.
3. Train on the repaired views in 30% of steps, at loss weight 0.2.
   - The loss only counts where the render already had splats (alpha > 0.5).
   - These steps skip the per-photo colour grid, the LiDAR loss and the depth/normal losses.

The loop is our own code (`scripts/difix/patch_difix.py`, `litereality_difix.py`). The image
clones Difix3D at build time (commit c76edc59), and none of NVIDIA's code is in this repo.

**The views** (`scripts/difix/poses.py`, `novel_poses.png`):
- 2 of the door, 16 orbiting the chairs, and 52 standing spots at least 0.4 m from the walk.
- Each moves 0.5 m a round from its nearest training camera toward its target.
- They are rolled like the photos: the atrium's frames lie on their side, with the image x axis
  pointing at the floor.
- No held-out frame is a target.

**Cost:** about 30 H100-minutes in all, roughly $2.
- C trained in 575 s.
- B trained in 1061 s, of which about 115 s per round went to Difix (70 views, 1.6 s each, 27 GB
  peak).
- The smoke test (`smoke.json`) timed one view at 0.73 s at 1024 px and 1.6 s at native.

## Numbers

**The trainer's own held-out frames** (every 8th, 80 frames; gsplat's renders):

| | PSNR | SSIM | LPIPS | cc PSNR | cc LPIPS |
|---|---|---|---|---|---|
| live v2 (report.json) | 22.82 | 0.878 | 0.209 | 27.57 | 0.195 |
| C | 22.82 | 0.877 | 0.209 | 27.54 | 0.195 |
| B | 22.83 | 0.877 | 0.213 | 27.39 | 0.200 |

**Fixed sharp held-out set in Spark** (`scores/`):
- 29 frames that no version trained on, rendered at full 1920 x 1440 by the room call's renderer.
- Produced by `results/stray_quality/score.py run ... --full --settle --gl metal`.

| | PSNR | PSNR cc | SSIM | LPIPS | sharpness (lap) | windows/door | objects |
|---|---|---|---|---|---|---|---|
| live v2 | 19.91 | 22.13 | 0.803 | 0.306 | 0.283 | 21.03 | 20.64 |
| C | 19.91 | 22.13 | 0.803 | 0.305 | 0.281 | 20.99 | 20.70 |
| B | 19.87 | 22.09 | 0.804 | 0.311 | 0.270 | 21.07 | 20.61 |

C reproduces the live room to within noise, so the comparison is fair. B is within 0.05 dB of C
on real views, a little worse on LPIPS, and slightly softer.

## Pictures

- **`offpath_live_C_B.jpg`:** 12 off-walk views; columns are live, C and B.
- **`offpath_zoom.jpg`:** four of them at full size:
  - door (row 1);
  - blinds (row 2): B is clearly cleaner;
  - the wall beside the door (row 3): B's new beige stain;
  - door close-up (row 4): B has no pink or green.
- **`difix_rounds.jpg`:** reference, render and Difix output in rounds 1 and 4.
  - Difix's outputs are clean: no stain.
  - Difix invents hard panel edges in the blinds.
- **`smoke_sheet.jpg`:** Difix alone on renders of the live room, at 1024 px and at native size.

## Why the stain, and what would be tried next (not done)

- **The likely cause is the colour model.** The repaired views train the splats with no per-photo
  colour grid, but the real photos keep theirs. In a wall region seen mostly at an angle, the two
  disagree, and colour drifts. Untested: give novel steps the identity grid explicitly, or a
  learned "novel" grid.
- **Smaller fixes to try:**
  - Fewer, closer novel views per round.
  - A lower weight than 0.2.
  - A colour-matched reference: match the Difix output's colour to the render before training.
- **Whatever is tried must win a blind A/B** on off-walk views and stay level on the sharp held-out
  set. This run does neither.

## Licence

Difix (code and weights) is licensed for research and evaluation only. Everything here lives in an
ephemeral research app (`litereality-difix-research`) and the `litereality-difix` volume. No room
trained with it is published, and none may be.
