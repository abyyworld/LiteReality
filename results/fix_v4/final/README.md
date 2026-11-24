# v2 (published) vs v4 (10000-step retrain): final before/after score

Both bundles scored by `results/stray_quality/score.py eval` on `raw.spz.bin` (the uncut trained
snapshot), with the room call's NEW Spark settings (`SPARK_OPTIONS` in roomcall web/app.js:
maxStdDev sqrt 5, maxPixelRadius 4096), on the Mac GPU (`--gl metal`, settle 1000 ms). Same view
lists for both; every view of both sets was scored. Driver: `run_scores.sh`; log:
`scores/score_all.log`; per-view numbers and eval sheets in `scores/`.

| set | size | bundle | views (unseen) | PSNR | PSNR_cc | SSIM | LPIPS | hf_ratio | lap_ratio | coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| sharp | 480 | v2 | 29 (29) | 20.18 | 22.53 | 0.768 | 0.261 | 0.545 | 0.288 | 0.978 |
| sharp | 480 | v4 | 29 (29) | 20.18 | 21.70 | 0.710 | 0.377 | 0.858 | 0.509 | 0.985 |
| random | 480 | v2 | 30 (2) | 20.83 | 23.12 | 0.811 | 0.227 | 0.706 | 0.589 | 0.979 |
| random | 480 | v4 | 30 (30) | 20.29 | 21.92 | 0.738 | 0.357 | 1.503 | 1.073 | 0.988 |
| sharp | 1920 | v2 | 29 (29) | 19.91 | 22.13 | 0.803 | 0.306 | 0.184 | 0.283 | 0.978 |
| sharp | 1920 | v4 | 29 (29) | 19.87 | 21.31 | 0.780 | 0.376 | 0.367 | 0.551 | 0.984 |
| random | 1920 | v2 | 30 (2) | 20.68 | 22.90 | 0.858 | 0.259 | 0.602 | 0.948 | 0.979 |
| random | 1920 | v4 | 30 (30) | 20.10 | 21.68 | 0.823 | 0.349 | 1.573 | 1.644 | 0.988 |

- The sharp set is the like-for-like one: held out of both trainings (29/29 unseen each). There v4
  ties v2 on PSNR (19.87 vs 19.91 full) but is worse on PSNR_cc (-0.82), SSIM (-0.023) and LPIPS
  (+0.070). v4 beats v2 on PSNR_cc in 7 of 29 views (same 7/29 on the random set).
- The random set is not like for like: v2 trained on 28 of its 30 views. On the 2 views unseen by
  both (rows 821, 1097, full res) v2 scores PSNR 16.81 / cc 19.03 / LPIPS 0.329, v4 18.35 / 22.44 /
  0.335.
- hf_ratio / lap_ratio rise for v4, but above 1 (random set) they mean more high-frequency energy
  than the photo: the side-by-sides show needle-shaped splat streaks, not recovered detail.
- The 4096 cap is not what hurts v4: v4 sharp full with maxPixelRadius=512 scores PSNR 19.88,
  cc 21.31, LPIPS 0.376 (`scores/v4_r512_sharp_full_scores.json`), the same as at 4096.
- Bundles: v2 2,228,000 splats, SH 1, raw.spz.bin 40.7 MB, served room spz 37.3 MB; v4 2,400,000
  splats, SH 3, raw.spz.bin 53.1 MB, room spz 49.4 MB (2,249,156 splats).

Side by side (photo | v2 | v4), 5760 x 1440, held-out views, `side_by_side.py`:

- `side_sharp_row1321_chairs.jpg` - teal chair close up: v4 cc 21.46 vs v2 20.65.
- `side_sharp_row1078_chairs_worst.jpg` - blue chair under the desk: v4 breaks up into streaks,
  cc 13.64 vs v2 19.23 (v4's worst view).
- `side_random_row821.jpg` - window sill, unseen by both: v4 cc 25.56 vs v2 22.28, but LPIPS
  0.215 vs 0.118 (v4 mottled).
- `side_random_row1097.jpg` - under the desk, unseen by both, motion-blurred photo: both fail
  (v2 cc 15.79, v4 19.33).
