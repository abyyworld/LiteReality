# Spark's splat-size cap in the room call (maxPixelRadius 512 -> 4096)

Spark 2.3.1 shrinks any splat whose on-screen radius (maxStdDev x sigma) is over
`maxPixelRadius`, 512 px by default. On the Eyeful room this cost short-trained splats 2-4 dB
(results/eyeful/README.md). This step lifts it in the room call and makes the scorer draw with
the call's own settings. No Modal was used.

## What changed

- `room_ops/roomcall/web/app.js`: one JSON line, `const SPARK_OPTIONS = {"maxStdDev": sqrt(5),
  "maxPixelRadius": 4096}`, and the call's only SparkRenderer is built from it. It was
  `{maxStdDev: sqrt(5)}`, so the cap was Spark's 512.
- `pipeline/evaluate/fidelity.py`: `room_call_spark_options()` reads that line. `SplatRenderer`
  draws with it by default and takes `gl="swiftshader"|"metal"` and `settle_ms`.
- `pipeline/evaluate/render_splat.html`: applies each job's `spark` options and can draw a view
  until three frames match (`settleMs`).
- `results/stray_quality/score.py`: `--gl metal` (the default now), `--settle-ms` (1000 on Metal)
  and `--spark KEY=VALUE` to override one option. On Metal a 1920x1440 view takes about 2 s,
  against 60-110 s in SwiftShader. Metal matches SwiftShader to 0.03 dB on the same views (rows
  68 and 2343 of the random set: 21.35 vs 21.33 and 19.50 vs 19.47 dB).

## Scores: atrium v2 (run/atrium_stray_q/v2, raw.spz.bin), full 1920x1440, Metal

All views of each set: 30 random, 29 sharp. Only maxPixelRadius differs.

| set | cap | PSNR | PSNR_cc | SSIM | LPIPS |
|---|---|---|---|---|---|
| random | 512 (before) | 20.6796 | 22.8996 | 0.8579 | 0.2588 |
| random | 2048 | 20.6805 | 22.9012 | 0.8579 | 0.2588 |
| random | 4096 (after) | 20.6805 | 22.9012 | 0.8579 | 0.2588 |
| random | 1e5 | 20.6805 | 22.9012 | 0.8579 | 0.2588 |
| sharp | 512 (before) | 19.9101 | 22.1314 | 0.8031 | 0.3060 |
| sharp | 4096 (after) | 19.9098 | 22.1318 | 0.8031 | 0.3059 |

Delta after - before: random +0.0009 PSNR, +0.0016 PSNR_cc, 0 SSIM, 0 LPIPS. Sharp -0.0003
PSNR, +0.0004 PSNR_cc, 0 SSIM, -0.0001 LPIPS. The biggest single view moves 0.0135 dB.

The 1250-step snapshot run/atrium_stray_q/v4_a1_nogrid, 512 -> 1e5:
- random: 19.102 -> 19.099 PSNR;
- sharp: 18.781 -> 18.797 PSNR;
- per view, within +-0.5 dB.

So on the phone atrium, few splats reach 512 px at the photo poses, and the cap was not what
made it blurry. The full-set numbers (30 views) are lower than results/fix_v4/scores, which used
4 views (21.70 random).

## Scores: Eyeful office1b (Spark "call" settings, 71 views, 1478x2232, Metal)

| run | 512 | 2048 | 4096 | 1e5 | trainer |
|---|---|---|---|---|---|
| B_5000 | 22.74 | 24.56 | 24.65 | 24.65 | 24.95 |
| B_1250 | 19.89 | 21.92 | - | 21.93 | 23.87 |

4096 recovers the whole cap loss on B_5000, the same as no cap. 2048 leaves 0.09 dB. A screen at
device pixel ratio 2 doubles every splat's pixel radius, so 4096 there is like 2048 at ratio 1.
Files: `results/eyeful/spark/B_*_splats_sh3_metal_call_r{2048,4096,1e5}.json`.

## Frame rate: the room call itself

Setup:
- Headless Chromium on ANGLE Metal (Apple M4), 1920x1080, device pixel ratio 1, with the call's
  own page.
- The call is served locally with atrium v2 (stage trained, 2.04M room splats plus 4 bodies).
- After joining, `?debug=1` `__roomcall.look()` places the camera.
- The cap is switched at runtime on the call's SparkRenderer. The caps are interleaved per view,
  so they share drift.
- FPS is the rAF rate over 4 s (vsync-capped at 60, never reached).

Six room views (`fps_views_main.json`), quiet machine (`fps_main.json`): 512 / 2048 / 1e5 all
mean 18.1 fps, min 13.8-13.9. The per-view numbers agree to 0.1 fps. Screenshots at 512 and 1e5
differ in no pixel by more than 8/255 on these views (PSNR 69-138 dB between them).

Four close-up views, about 20-50 cm from a chair, the door, the desk and the floor
(`fps_views_close.json`):

| run | 512 | 2048 | 4096 | 1e5 |
|---|---|---|---|---|
| quiet machine (`fps_close.json`): mean / min | 20.0 / 13.9 | 19.8 / 13.6 | - | 19.2 / 12.9 |
| loaded machine, 3 reps (`fps_close_4096.json`): mean / min | 21.0 / 15.5 | - | 20.9 / 15.6 | 20.1 / 14.7 |

With no cap, the worst close-up (the door from about 20 cm) drops 24.9 -> 22.6 fps (-9%), and 3.5% of
its pixels change. 4096 costs 0-3% per view. With the new default on the six room views on the
loaded machine (`fps_main_4096.json`): 512 mean 16.5 / min 13.5, 4096 mean 16.8 / min 13.6.

## Choice

The new cap is 4096. It gives the same quality as no cap on every scored set. The frame rate is
the same as at 512 on room views and within 3% on close-ups. A finite cap still bounds the
fill cost of a splat right in front of the camera.

Reproduce with `fps_ab.py`, `cap_shot.py` and `join.py` here. Their usage lines show the local
server command. `score.log`, `score_v4p1250.log` and `eyeful_r2048.log` hold the scorer output.
