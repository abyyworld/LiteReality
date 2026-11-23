# Whole room: our videos -> RealityScan -> LichtFeld (atrium)

> Internal research only (RealityScan EULA). RealityScan ran as ephemeral `modal run` jobs of
> `deploy/modal/realityscan/app.py`; its .deb and all inputs/outputs stay on the private Volume
> `litereality-tools`. Not wired into any service.

**Result: STOPPED at the alignment gate. No LichtFeld arm was trained or scored.**

The test was the user's pipeline: two videos of the same atrium (the Stray Scanner recording
`901a305fdf` and the normal iPhone Camera video `atrium_video.MOV`) -> RealityScan camera poses ->
LichtFeld training, scored on the 29 Stray eval frames against our current best (v2). The gate before
training was: RealityScan must reconstruct the walk as **one component, or with >= 90 % of frames in
the largest one**. Neither video passed it, so there is no score table and no
`whole_room_compare.jpg`. Training on what RealityScan produced would have meant training on poses
that are either a small fraction of the room or inconsistent with each other (see below). That would
measure RealityScan's failure, not LichtFeld.

## Was it tried properly?

The first study (`README.md` in this folder) was **not** a proper test. It used only 400 sparse pool
frames, and its ARKit prior runs had no focal prior or calibration in the XMP sidecars. It also never
checked how RealityScan uses the priors. The retest (`dense/README.md`) fixed all of that:

- **Dense frames.**
  - Stray: 1095 frames, every 2nd odometry row, 1920x1440. The blurriest 10 % were dropped, with
    blur measured relative to the ±1 s neighbourhood so plain walls are kept.
  - Phone video: 1248 frames, 1080x1920 portrait.
- **One shared lens group** and a focal prior: 25.04 mm from ARKit fx for Stray, 29 mm for the video.
- **Settings sweep:** detector sensitivity Default, High and Ultra; preselector and feature counts
  up to 60k per image; image overlap Low, Medium and High; `-mergeComponents`; a second `-align`;
  distortion models 2 and 0.
- **ARKit priors, convention measured.**
  - The XMP convention was taken from RealityScan's own `-exportXMP` against its COLMAP export
    (residual 1e-12).
  - The priors were Z-up, with the calibration in every sidecar.
  - Accuracies were ±3 cm/3° and ±5 cm/5°, in both "initial" and "draft" modes, with merging of
    georeferenced components, a second align and a forced rematch.

## What RealityScan produced

| Video | Best image-only largest component | With ARKit priors | Never in the largest / merged piece |
|---|---|---|---|
| Stray (1095 frames) | 171 frames (16 %, Ultra/Low/2nd align, 3222 s); defaults 145 (13 %) | 551–616 "largest", but made of 11–16 separately solved pieces | 8.0–8.6 s, 18.7–22 s, 34–45 s, 46–49 s, 52–56 s |
| Phone video (1248 frames) | 388 frames (31 %, High + merge, 684 s) | n/a (no ARKit) | 0–16 s, 34–45 s |

The last column lists the stretches missing from the largest piece (image only) or the merged prior
set. It is not "registered in no component":
- Phone video: 34–45 s registers in no component in either run (0 of 276 frames). 0–16 s does
  register, but only in smaller pieces (245 of 457 frames in the best run).
- Stray, image only: the defaults run registers 0 of 44 frames in 18.7–22 s and 0 of 77 in 52–56 s.
  The Ultra run puts 52 of those 77 and 44 of the 219 frames in 34–45 s into small pieces.
- Stray, with priors: 1081–1083 of 1095 frames get an exported pose, but in separately solved pieces
  (see below).
- Only 4 of the 29 eval frames are in any image-only largest piece (15–17 with priors, 0 for the
  phone video), so neither arm could even be scored on the eval set as one consistent model.

- **Stray, image only.**
  - The largest piece always ends at 8 s of the 58 s walk.
  - Where frames do register (0–8 s, defaults run, 145 frames), RealityScan matches ARKit to 8.9 mm
    median / 13.0 mm p90 in centre, and 0.39° / 0.46° in rotation with an orientation-only fit
    (1.09° median with the centre Sim(3), which is weakly constrained over this 0.26 m stretch). Its
    fx is within 0.65 % of ARKit's. The best image-only run (Ultra, 171 frames) gives 8.6 / 14.1 mm
    and 0.43° / 0.73°.
  - RealityScan is accurate where it works; it just does not work on most of this room.
- **Stray, with priors.**
  - RealityScan 2.2 only *places* each separately solved piece by a fit to its camera positions.
    There is no joint bundle adjustment.
  - Every piece keeps its own fx: 11–16 different lenses despite one calibration group.
  - Each piece keeps a constant rotation offset. Re-checked by grouping cameras by fx (one fx per
    piece) against the global Sim(3) to ARKit:
    - ±3 cm/3° run: offsets of 0.5–23° per piece, plus 162° and 56° for two 3-frame pieces.
    - ±5 cm/5° rematch run: offsets of 0.5–10.8°.
    - Within each piece, an own rotation fit leaves only 0.06–0.5° median. So each piece is
      internally consistent with ARKit and the error is the offset between pieces, not an axis bug.
  - The best merged set (±5 cm/5°, forced rematch, 551 frames) is still 1.45° median / 5.15° p90 off
    ARKit in rotation.
- **Phone video.**
  - Even the best piece cannot be placed in the room reliably. Scaled ICP to the Stray LiDAR cloud
    gives 30 % inliers and a 1.6 cm residual, but the runner-up solution is 103° away in yaw and
    2.3 m away.
  - That solution is 19° off the MapAnything poses of the same frames, used only as a check.
- **Cause.** The atrium's white walls and ceiling, and its dark mesh blinds, give features that do
  not chain between views. Blind "features" are fabric noise and light through the mesh. The walk
  breaks at the same time spans in every run, even for frames with 3000–7000 detected features.
  Those frames either register in no component or only in small pieces that do not connect.

## Reference (context only, no new spend)

Our current best pipeline on this room: v2, ARKit poses + our gsplat, 637 frames at 1440 px, 10k
steps, scored on the same 29 eval frames (`../stray_quality/v2_scores.json`).

| Arm | Poses | Trainer | PSNR | PSNR_cc | SSIM | LPIPS | Train time | GPU $ |
|---|---|---|---|---|---|---|---|---|
| v2 (reference) | ARKit | our gsplat | 20.19 | 22.53 | 0.767 | 0.262 | 588 s | ~0.65 |
| A: Stray -> RealityScan -> LichtFeld | RealityScan | LichtFeld | not run: alignment gate failed | | | | | |
| B: phone video -> RealityScan -> LichtFeld | RealityScan | LichtFeld | not run: alignment gate failed | | | | | |

The earlier LichtFeld test on ARKit/COLMAP poses (188 frames, mrnf 30k, `--no-calibration-correction`)
scored 21.80 PSNR_cc / 0.288 LPIPS, equal to our gsplat on the same set (21.65 / 0.286; plain PSNR
18.85 vs 19.15). LichtFeld itself is not the
bottleneck here; the RealityScan poses are.

## Verdict for this room

**For this atrium, videos -> RealityScan -> LichtFeld does not work, for either video.**

- **Stray video.** RealityScan recovers one consistent piece covering only the first 8 s, at most 16 %
  of frames. With ARKit priors it covers about half the frames, but as a patchwork of pieces with
  different lenses and rotation offsets of several degrees. A splat trained on that would be worse than
  v2 by construction. v2 already uses ARKit poses for all 637 frames, and on the part both cover,
  RealityScan agrees with ARKit to about 1 cm / 0.4°. So it offers no pose-accuracy gain to trade
  against the coverage loss.
- **Phone video.** RealityScan covers at most 31 % of the frames, and that piece cannot be placed in
  the room unambiguously. It could not be scored against the Stray eval frames in any meaningful way.

This is a property of this room's texture-poor surfaces, not of the setup. A textured room might pass.
For this room, ARKit poses (v2) remain the better pose source.

## Time and cost

| Step | Time | Modal cost |
|---|---|---|
| RealityScan dense sweep, Stray (8 runs incl. a stopped distortion-model-0 run) | 431–3222 s per run | |
| RealityScan dense sweep, phone video (2 runs) | 684 s and 751 s | |
| RealityScan total (CPU, 8 cores / 16 GB, incl. two preempted restarts) | | ~$2.2 (ephemeral $4.63 -> $6.86) |
| LichtFeld training, scoring, wander renders | not run | $0 |
| **Total for this job** | | **~$2.2** of the $5 budget; metered month total $16.53 |

**Cleanup.**
- No ephemeral app is running.
- `litereality-realityscan` apps are all stopped.
- The only task listed is the deployed `litereality-roomcall-reminders` service, which this job did
  not touch.
- No commits, no deploys.

## Files

- `dense/README.md`, `dense/runs_table.json`, `dense/<tag>_report.json`: per-run details.
- `video/registration_dense_v2hm.json`: video piece -> room registration.
- The best-effort COLMAP exports fail the gate and were not used:
  - `~/LiteReality-scans/rs_dense_stray/colmap/sparse/0` (551 frames)
  - `~/LiteReality-scans/rs_dense_video/colmap/sparse/0` (388 frames)
- Scripts:
  - `scripts/realityscan/prep_dense.py`
  - `scripts/realityscan/write_priors_dense.py`
  - `scripts/realityscan/dense_report.py`
  - `scripts/realityscan/dense_to_room.py` (written, not run)
