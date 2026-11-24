# Atrium diagnosis: chairs that double, chairs that move together, and blur

Room: the atrium, Stray v2 bundle (`run/atrium_stray_q/v2`, the one in the room picker).
Physics: RoomPlan boxes from the LiteReality Scanner capture `atrium_1059`.
Every cause below was checked a second time by a separate reviewer. Only causes the reviewer
confirmed or partly confirmed are listed as causes, with the reviewer's corrected impact.
Refuted and unproven claims are at the end.

All paths are relative to `~/LiteReality/` unless they start with `~`.
Diagnosis scripts and outputs: `results/diagnosis/` (folders `chairs`, `review_chairs`, `blur`,
`review_blur`, `export_path`, `review_render_path`).

Spend and safety: four short ephemeral Modal runs on one L4 GPU (`export_path/gsplat_render_modal.py`,
app `diag-gsplat-render`, nothing deployed). Two of them crashed at startup. Estimated under $0.10; the
bill was not read back. Everything else ran on the Mac CPU. Nothing was committed, pushed or deployed.
The live app was not touched.

## The short version

- **The chair bugs come from where the physics boxes are, not from the splat.** RoomPlan output three
  identical chairs in a perfectly even row. One of them (Chair1) has no real chair under it. One real
  chair (the dark office chair) got no body. The SAM mask step then gave pieces of chairs to the wrong
  bodies. So a grab moves part of a chair and leaves the rest behind, or moves pieces of two chairs.
- **The blur is in the trained splat itself, not in the browser.** The browser costs about 0.5-0.7 dB.
  The biggest visible cause is that people stand where the camera never was: the walk covered about
  1.8 x 1.6 m of a 4.2 x 4.1 m room. Second, a large, patchy colour error stays in the exported splat.
  Motion blur in the video is real but small.
- **Method:** keep the room method (one phone video, one room-wide splat, Spark viewer). Change the
  object method: find objects in the capture's own geometry and stop trusting RoomPlan boxes for
  object count and position.

---

## 1) Why a grabbed chair doubles ("one copy moves, one stays")

A grab moves the splats of one physics body. If a real chair's splats are spread over two bodies, or
over a body and the static room, only part of the chair moves. The rest stays behind and looks like a
second chair.

Who owns each real chair's splats in the served bundle (`chairs/fractions.json`, picture
`chairs/true_chairs_ownership.png`):

| Real chair | Owners of its splats |
|---|---|
| Blue chair | Chair0 62%, Chair1 30%, room 7.5% |
| Teal chair | room 83%, Chair2 17% |
| Dark office chair | room 93%, Chair2 5.6%, Chair1 1.2% |

A clean result would be about 90% or more in one body. None of the three chairs reaches that.

### Cause 1 (major, root cause): RoomPlan's chairs are not measurements

- All three RoomPlan chairs have the same size (0.505 x 0.585 x 0.8861 m) and the same yaw
  (-141.87 / -141.87 / -141.84 deg).
- They are evenly spaced to 0.1 mm: Chair0 to Chair1 is (-0.8580, -0.6736) and Chair1 to Chair2 is
  (-0.8579, -0.6735). Source: `review_chairs/r3_scan_coverage.json` (parse of
  `~/LiteReality-scans/atrium_1059/roomplan/room.usdz`).
- RoomPlan gave a tidy row, not three real positions.
- **Chair1 is a phantom, even in its own scan.** At Chair1's centre the scan's own LiDAR sees floor
  (6,741 points) and the desk top (6,728), but only ~280 points between 0.15 and 0.70 m and none at
  backrest height (0.80-1.05 m). A real chair there would show thousands: the dark chair's column has
  5,339 (`review_chairs/r4_chair1_column.json`).
- **The dark chair has no body.** The splat holds exactly three chair clusters (blue, teal, dark).
  Teal and dark touch and form one connected group of 17,173 opaque splats
  (`chairs/clusters_topdown.png`, `chairs/s5_clusters.py`).
- **The chairs did not move between the two captures.** The dark chair's backrest is 4.7 cm apart
  and the teal seat 7.8 cm apart between the scan's LiDAR and the Stray splat
  (`review_chairs/r6_same_place.json`, picture `review_chairs/r2_scan_vs_stray.png`).
- The automatic corrections only slide a box a short way and never add or remove a body. Chair1's
  vote correction was -0.354 m, cut to -0.177 m ("scaled 0.5", `run/atrium_stray_q/v2/report.json`).
  Snap limit is 0.40 m / 10 deg (`split.py`), vote limit 0.5 m.
- Even the "matched" Chair2 is off: the teal chair's centre is about 0.28 m from the corrected
  Chair2 origin (-1.145, 0.265).

What this means: recording RoomPlan and the Stray video in one session would **not** fix this. RoomPlan
was already wrong inside its own capture. Body count and position must come from the geometry.

### Cause 2 (major): the SAM mask vote hands chair pieces to the wrong body

- Masks are prompted with each body's RoomPlan box grown 30% on every side (`GROW = 0.30`,
  `src/litereality_agent/pipeline/splatroom/masks.py:33`, used at `:117-125`). Since the boxes sit at
  the wrong places, the prompts hold other chairs. Chair1's prompt holds the dark chair in 24 of 24
  views, the blue chair in 7 and the teal chair in 6 (`chairs/sam_prompts.json`, picture
  `chairs/sam_prompt_view00068.jpg`). Chair1's prompt is centred on empty floor.
- The candidate splats are the union over **all** bodies (`masks.py:239-243`). Each body then votes on
  every candidate with only its 2D prompt and mask. Nothing in 3D keeps a body's splats near its own
  box. Result: Chair1 owns 581 opaque splats within 0.30 m of the blue chair, 1.18 m from Chair1's
  origin. 1,270 of its splats sit inside Chair0's box and only 180 inside its own
  (`review_chairs/r5_ownership_simple.json`).
- A splat that photos saw but that fell in no mask goes back to the room, even when it lies inside
  the body's box (`split.py:540-549`). Inside the corrected Chair2 box, 7,167 splats went to the room
  (1,357 opaque) and only 3,235 to Chair2. That is how most of the teal chair was stripped.
- Proof the masks caused it: the same trained splat re-cut without masks (`chairs/s7_recut.py`,
  `chairs/recut_compare.json`) gives the blue chair 96% to Chair0, against 62% served.
- The masks themselves were never saved, so the exact SAM output could not be seen.

### Cause 3 (moderate): the box is off-centre and too small for the chair

- Template box 0.505 x 0.585 x 0.886 m, yaw shared by all three chairs. The vote never turns a box
  (`dyaw_deg 0.0`, `split.py:354`); the snap turns it at most 10 deg (`split.py:44`). Cut margin is
  2 cm (`MARGIN`, `split.py:39`).
- Share of each chair inside its corrected box + 2 cm (`chairs/outside_box.json`): teal 39.6%,
  dark 19% (nearest box), blue 95.8%.
- The reviewer's correction: the main problem is the box centre (about 0.28 m off for teal), not
  the box size. The blue chair's box is well centred and the same size, and holds 95.8%.

### Why this was missed (minor, process gap)

The earlier check (`results/stray_quality/README.md:56-58`) compared the scene at rest. A split is
lossless at rest by construction: bodies are stored relative to their corrected pose
(`split.py:552-561`). Room + bodies equals the raw splat exactly: 2,044,126 + 450 floor-fill +
183,874 = 2,228,450 (`chairs/s1_consistency.py`). Any ownership, right or wrong, passes that test.
A **move test** is needed: move each body and check what stays behind.

---

## 2) Why grabbing one chair grabs its neighbour

### Main reason (follows from section 1): one body holds pieces of two chairs

Chair2 (the teal chair's body) owns 1,046 opaque splats of the teal chair and 360 of the dark chair
(`review_chairs/r5_ownership_simple.json`). Chair1 owns 30% of the blue chair from across the room.
So grabbing Chair2 moves bits of both the teal and the dark chair. That is the "grabs BOTH" look.
A wrong pick is not needed to explain it.

### Contributing reason (moderate): picking hits hidden boxes, and the highlight is invisible

- Picking raycasts the physics meshes, including hidden ones (`src/litereality_agent/room_ops/roomcall/web/app.js:728-734`, `:731`).
- A body's mesh is hidden once it has a splat (`app.js:1279-1284`).
- The highlight only changes the colour of that hidden mesh (`app.js:613-619`), so a person cannot
  see what they took.
- The dark chair has no body, so a click on it grabs whichever hidden box the ray hits. Simulated
  clicks from 8 real camera spots (`chairs/s12_pick.py`, `chairs/pick_what_a_click_grabs.json`):
  click on the dark chair: 14.5% grab Chair2, 14.3% grab Chair1, 70% nothing. Click on the teal
  chair: 40% grab Chair2.
- The reviewer notes the simulation stops a ray at the chair. In the browser the ray keeps going and
  can hit a box behind it, so wrong grabs are probably **more** common than this.

### Not the reason: physics

No welds (`neq = 0`), every chair is its own free body under the world, no contacts at rest
(`ncon = 0`). Chair1 to Chair2 is 0.88 m. Dragging Chair2 0.6 m toward Chair1 moves Chair1 only 3 cm,
an ordinary push (`chairs/s8_physics.py`, `chairs/physics_drag.json`).

---

## 3) Why everything is blurry

Bottom line: the blur and smears are already in the trained splat. gsplat (the trainer's own renderer)
and Spark (the browser) look equally smeared at the trainer's own held-out cameras
(`export_path/gsplat_vs_spark.jpg`: photo | Spark | gsplat).

Ranked causes, as the reviewers left them:

| # | Cause | Corrected impact | Key number | Evidence |
|---|---|---|---|---|
| 1 | People stand where the camera never was (view extrapolation) | Major for what users see; not measured in dB | Walk covered 1.8 x 1.6 m of a ~4.2 x 4.1 m room. 47% of the floor is >1 m from any capture spot, 76% is >0.5 m. Eval frames are a median 3.8 cm from a training camera, so the score never sees this | `run/atrium_stray_q/v2/splat.json` (200 capture spots), walls in `physics/scene_splat.xml`; walk zone off unless `?zone=1` (`app.js:153`); picture `results/stray_quality/wander/v2_desk_to_door.jpg` |
| 2 | A large, patchy colour error stays in the exported splat | Major for PSNR (3-5 dB); also looks patchy and smeared | After the global colour fit, a per-tile fit gains another 3.1-5.3 dB on all 6 full-res views (train_234 19.97 -> 25.08). On a merely blurred photo the same fit gains 0.6 dB | `review_blur/local_cc.json`, `review_blur/local_cc.py`. Likely source: the per-photo bilateral grid absorbs local colour in training and is dropped at export (`deploy/modal/splat/app.py:89-90`). Trainer's own raw vs cc gap: 22.82 vs 27.57 (`report.json`) |
| 3 | The error is at room scale: floaters, wrong colour fields, ghosted parts | Major (same family as 1-2) | cc-PSNR barely rises when the images are shrunk: train_234 19.97 -> 20.35 -> 21.77 at 1920 / 480 / 60 px wide. A photo blurred by sigma 8 px would score 27.0 -> 29.5 -> 42.7 | `review_blur/scale_check.json`; pictures `review_blur/side_train_234.jpg`, `review_blur/crop_train_234.jpg` (white blobs on carpet, ghosted chair base, at a training view) |
| 4 | Motion-blurred, heavily compressed video frames, and a frame picker that cannot avoid them | Real but secondary: fine-scale softness only | Training frames: median blur 2.5 px box length at 1920 (about sigma 0.7 px), 28% at 4 px or more. Picker has 2.3 candidates per window, so close to random. Video is 0.082 bits/pixel | `blur/blur_stats.json`, `blur/blur_hist.png`, picture `blur/crop_blurred_training_frame_2002_vs_eval_2010.png`; picker `src/litereality_agent/models/capture/stray.py:607-683`, scored at 480 px (`stray.py:102`) |
| 5 | The score could not see blur | Moderate for method: we were measuring the wrong thing | All scores at 480x360 (`results/stray_quality/score.py:46-47`). Full resolution lowers renders only 0.4-0.8 dB. PSNR is dominated by causes 2-3, so a separate sharpness metric is needed | `blur/render_sharpness.json`, `review_blur/scale_check.json` |
| 6 | Splat count can never grow (config bug) | Real bug; effect on sharpness not measured; likely minor | 557 depth maps x 4000 seeds = 2,228,000 points, but `cap_max` is 800,000. gsplat MCMC grows to `min(cap_max, 1.05 n)`, so it adds nothing. Final count = seed count | `pipeline/splatroom/__init__.py:90-94`, `deploy/modal/splat/app.py:84, :186`, `report.json`, gsplat `strategy/mcmc.py:176` (v1.5.3) |
| 7 | SH3 -> SH1 cut at export | Big at training views, small at new views | Proxy test on LichtFeld SH3 splats of this room: training views lose 3-10 dB (frame_00048: 42.65 -> 31.87). Held-out views lose 0.4-2.3 dB. On our own runs, v3 (trained SH1) vs v1 (SH3 cut to SH1) is +0.08 dB | `review_blur/sh_trunc.py`, picture `review_blur/sh_lfs_mrnf30k_nocal_frame_00048_720_540.png`; `app.py:85-86`; `results/stray_quality/table.md` |
| 8 | The browser draws a bit worse than the trainer | Minor: 0.5-0.7 dB, worst in close-ups | See section 4 | `review_render_path/recheck.json` |

Notes on the table:
- Coverage per surface is not the problem: every surface type is seen by a median of 32-95 training
  views (`blur/coverage.json`). The problem is that all views come from a small patch of floor, so
  most places a person can stand are new viewpoints.
- Pose optimisation was never turned on (no `pose_opt` in `deploy/modal/splat/*.py` or
  `pipeline/splatroom/*.py`; gsplat 1.5.3 defaults it to False). How much it would help is **not
  measured**; the pose-error estimate behind it was unreliable (see the end).
- Compared to published work: on the ScanNet++ iPhone benchmark, vanilla 3DGS scores 19.19 dB cc /
  LPIPS 0.370 at 1920x1440 against DSLR ground truth. Our 22.53 dB cc is on an easier protocol
  (480x360, sharp frames from the same video). Different protocols, so this is only a rough guide.

---

## 4) What is lost in the browser vs the trainer

Measured on `run/atrium_stray_q/v2/raw.spz.bin` (stage "trained", step 10000) at 9 of the trainer's
own held-out cameras, 960x720.

| Step | Score | Cost | Evidence |
|---|---|---|---|
| Trainer's own hold-out (gsplat, float ply, SH3, 1440 px, 80 frames) | PSNR 22.82, cc 27.57 | - | `results/stray_quality/v2_scores.json` |
| gsplat drawing the served SPZ (SH1), 9 frames | 21.10 dB | 1.7 dB below the trainer, **not explained** (see below) | `review_render_path/recheck.json` (corrected for a double-alpha bug in `export_path/compare_gsplat.py`) |
| Spark drawing the same SPZ | 20.44 dB, SSIM 0.787 | 0.66 +/- 0.20 dB mean, **0.55 median** below gsplat; gsplat better on 8 of 9 | `review_render_path/recheck.json` |
| ...in a close-up (frame 320) | - | 2.05 dB; Spark draws white streaks on the near chair | picture `review_render_path/frame320_photo_spark_gsplat_diff.jpg` (photo / Spark / gsplat / 4x difference) |
| Spark cut-off `maxStdDev: sqrt(5)` (gsplat uses 3, Spark's default is sqrt(8)) | - | 0.16 dB; better on 8 of 9 with 3 | `app.js:1179`, `src/litereality_agent/pipeline/evaluate/render_splat.html:30`, `export_path/variants_9views.json` |
| Spark's 16-byte repack (float16 centres, 8-bit scale, colour clipped to 0-1) | - | 0.05 dB or less | `export_path/pack_check.json`, `export_path/gsplat_vs_spark.json` |
| SH1 -> SH0 | - | 0.07 dB | `export_path/gsplat_vs_spark.json` |
| Room is never fully opaque | 2.2% mean transparency in Spark (1.6% gsplat) | Light background shows through as haze on ~11% of pixels; 1.95% with maxStdDev 3 | reviewer's measurement on the 9 renders |
| Tone mapping, colour space, MSAA, SPZ antialias flag, LoD, bundle filtering | - | none | `export_path/variants_3views.json`; all 2,228,000 splats load |

About the headline numbers:
- The "~26.7 dB" was **not our trainer**. It is LichtFeld MRNF's own hold-out (26.66,
  `results/trainers/README.md`).
- Part of "27.6 vs 22.5" is just different colour-fit formulas: on the same Spark renders, a gsplat-style
  fit scores 24.01 and our scorer's 3x4 fit 23.18 (+0.8 dB from the definition alone).
- The unexplained 1.7 dB (22.82 -> 21.10) could be frame sampling (9 of 80 frames, about 1.5 standard
  errors), 960 vs 1440 px, SH3 -> SH1, or SPZ quantisation. The float ply that would settle this is
  deleted after export (`deploy/modal/splat/app.py:300`). The trainer-minus-Spark gap is a steady
  2.4-2.7 dB across v0-v3 (`results/stray_quality/table.md`), which looks systematic.

Straight answer: the browser is responsible for about half a dB, mostly in close-ups. It is not why the
room looks blurry.

---

## 5) Should we change method?

**Recommendation: keep the room method, change the object method.**

- Keep: one phone capture -> one room-wide 3DGS with ARKit poses and LiDAR depth -> Spark in the
  browser. The blur is fixed by capture and trainer settings, not by a new reconstruction method.
- Change: RoomPlan boxes no longer decide which objects exist or where they are. Objects come from the
  capture's own geometry (splat or LiDAR clusters). RoomPlan supplies only walls, doors, windows and
  object categories. Every bundle must pass a move test.

Alternatives (licences checked with `gh api` and LICENSE files; "Inria NC" = non-commercial, cannot be
used in a hosted service). Gains are estimates unless marked measured.

**Objects**

| Option | Fixes | Licence | Effort | Cost | Expected gain |
|---|---|---|---|---|---|
| Physics and splat from one session | Almost nothing; RoomPlan was wrong inside its own capture | n/a | Capture only | $0 | Small (alignment is already 1-3 cm) |
| **Instances from splat/LiDAR clusters, matched to RoomPlan categories, add/remove bodies, oriented box per cluster, cut by cluster** (recommended) | Phantom body, missing body, wrong pose, off-centre box | Our code (numpy, scipy) | 1-2 days | $0, local re-cut | Large. Clustering already found exactly the three real chairs. Weak spot: teal and dark chairs touch |
| SAM 2 point prompts at the matched cluster, exclusive masks, vote limited to own box | Boundary between touching chairs | Apache-2.0 | ~1 day | ~$0.1 Modal to verify | Moderate on top of the above |
| Per-Gaussian instance features (Gaussian Grouping, SAGA, OpenGaussian, Feature 3DGS, GARField, ObjectGS) | Clean object boundaries | Mostly Inria NC; only GARField (MIT, NeRF) and ObjectGS (MIT, replaces our trainer) can be hosted | A week or more | ~$1-3 retrain per room (estimate) | Similar to the two rows above, much costlier. Not now |
| Separate object scan per movable object (Apple Object Capture) | Objects fully, incl. back and underside | Apple SDK on device | Medium, +1-2 min capture per object | $0 | Cleanest objects, but needs the original chair removed from the room. Optional for key objects |
| Mesh objects in a splat room | Exact colliders | as above | Medium | $0 | Style mismatch; works against "photoreal". No |
| Inpaint the hole behind a moved object (Splat Repair Fast Fill, or our `fill_floor`) | What shows once an object moves | Apache-2.0 | Hours | $0 | Moderate, only after the cut is right |

**Blur**

| Option | Fixes | Licence | Effort | Cost | Expected gain |
|---|---|---|---|---|---|
| gsplat built-in `pose_opt` | Pose inconsistency | Apache (gsplat) | Low: flag plus wiring | ~$0.65-1 per retrain | Unmeasured; guess +0.5-1.5 dB on held-out views |
| 3dgs-deblur ("Gaussian Splatting on the Move"), BAD-Gaussians | Motion blur, rolling shutter | Apache-2.0 | Days to port | ~$1 per retrain | Paper: +1.7-2.35 dB on phone data; less certain here because our input blur is small |
| Mip-Splatting / Analytic-Splatting | Aliasing when zooming | Inria NC / MIT | - | - | Small; we already use gsplat `--antialiased`. Skip |
| Depth/normal priors (DN-Splatter, 2DGS, PGSR) | Geometry, floaters on walls | Apache (already used) / Inria NC / non-profit only | - | - | Small for sharpness. Keep what we have |
| Generative cleanup (NVIDIA Fixer; Difix3D+; 3DGS-Enhancer) | Off-path smears (cause 1) | Fixer: Apache + NVIDIA Open Model License (from earlier sweep, not re-verified). Difix3D+: non-commercial. 3DGS-Enhancer: no licence | Medium | ~$1 (estimate) | Better-looking off-path views; PSNR can drop. Later, A/B tested |
| Feed-forward (AnySplat, VGGT) | Speed only | MIT / only the VGGT-1B-Commercial checkpoint | Medium | low | Blurrier (~448 px). Our trainer already takes minutes. Preview only |

---

## 6) Fix plan, ranked by impact per effort

Nothing below has been run end to end. Only the no-mask re-cut was simulated (`chairs/s7_recut.py`).

### Pipeline (local, no GPU)

| Rank | Fix | Fixes | Effort | Cost |
|---|---|---|---|---|
| 1 | Re-cut v2 with snap and **without masks**. Simulated: blue 96% Chair0, dark 63% Chair1 (Chair1 snaps onto the dark chair), teal 63.5% Chair2 (`chairs/recut_compare.json`) | Most of the doubling, today | Minutes | $0 |
| 2 | Limit each body's vote to its own box + REACH (`masks.py:239-243`). Drop the "seen, in no mask -> room" rule unless another body's mask claims the splat (`split.py:540-549`) | Chairs stolen across the room | Hours | $0 (verifying with SAM ~$0.1) |
| 3 | Add the move test to every bundle: each real object >= 90% in one body, <= 5% in any other (`chairs/s6_fractions.py` is most of it) | Stops this from shipping again | Hours | $0 |
| 4 | Object method change: clusters -> match by category -> add/remove bodies -> oriented box per cluster (free yaw, height from cluster) -> cut by cluster membership | Both chair bugs at the root | 1-2 days | $0 |
| 5 | Fix `cap_max` (1.5-3x the seed count, or seed ~800k) | Lets MCMC grow detail | Config | in the retrain below |
| 6 | Frame picker: blur measured at full res, 8+ candidates per window, eval frames at random times | Fine-scale softness | Hours, re-import ~20 s | in the retrain below |
| 7 | Score at 1920 (or 960) and add a sharpness metric; keep the float ply (`app.py:300`) | Measures what users complain about | Hours | $0 |
| 8 | **One retrain** with 5-7 plus `pose_opt` on, `sh_degree=1`, a regularised bilateral grid (or gsplat's PPISP) | Blur causes 2, 4, 6 | 1 Modal run | **~$0.65-1** (v2: 588 s on an H100, $0.65) |
| 9 | SAM 2 point prompts, exclusive masks, for touching chairs | Teal/dark boundary | ~1 day | ~$0.1 |
| 10 | Hole fill behind moved objects (`fill_floor` or Splat Repair) | Holes after a move | Hours | $0 |

### Capture (no code, one recapture)

- Walk a full loop around the edges of the room at chest height, not a 1.8 x 1.6 m patch. Add low and
  high passes, and close passes on the chairs and the door. This is the most likely big visible gain
  for "blurry everywhere" (cause 1). Untested.
- Walk slowly. Lock exposure and white balance if Stray allows it.
- Retrain cost after recapture: ~$0.65-1.

### Client (`app.js`; touches the live app, so only with your go-ahead)

| Fix | Effect | Effort |
|---|---|---|
| Highlight the grabbed body's **splat** (tint or outline the SplatMesh), not the hidden box (`app.js:613-619`) | People see what they took | A few hours |
| Do not offer a body as grabbable if its splat holds too little of a real object (phantoms) | No "grab nothing / grab ghost" | Hours |
| `maxStdDev: 3` in `app.js:1179` and `render_splat.html:30` | +0.16 dB, less haze | Two lines + phone frame-rate check |
| Walk zone on by default (`app.js:153`) | Keeps people near where the room was filmed | One line |
| Write the physics file's hash into `splat.json` and refuse a mismatch | Guards against serving `scene.xml` with this bundle | Small |

Later, only if blur is still visible after the capture and retrain: NVIDIA Fixer distillation for
off-path views (~$1), then a port of 3dgs-deblur (days).

---

## 7) What we could not check

- **SAM masks:** never saved. The mask conclusion rests on what the prompts contained and on the final
  ownership. Reproducing them needs a Modal call (~$0.1), not run.
- **Live app:** which bundle it serves, and whether it serves `scene_splat.xml` or `scene.xml`. Rules
  said not to touch it. Everything was measured on the local v2 bundle.
- **Other bundles:** only v2 was analysed. v0, v1, v3 and the tea_room bundles were not measured.
- **The browser client:** not run with real clicks. Picking was simulated with MuJoCo rays from capture
  spots. All Spark renders were headless Chromium with software rendering (SwiftShader), not a real GPU
  or phone, so part of the Spark-vs-gsplat gap could be SwiftShader.
- **"True chair" splat sets** are approximate (3 cm voxel groups; teal and dark split by a 2-means
  rule). Fractions are roughly +/- 5 points (a judgement).
- **The real yaw of each chair** was not measured. The scan LiDAR barely covers the blue chair.
- **Float ply vs SPZ:** the ply is deleted at export, so SPZ quantisation and SH3 -> SH1 on v2 itself
  were not measured. The SH loss is inferred from a LichtFeld proxy and from v3 vs v1.
- **The 1.7 dB** between the trainer's hold-out and gsplat-on-SPZ is unexplained.
- **Bilateral grid behaviour:** no grids saved. That it causes the patchy colour (blur cause 2) is a
  likely explanation, not a proven one.
- **MCMC run log:** not saved. "No growth" is inferred from code plus final count = seed count.
- **Pose error size:** the estimator was unreliable (see below); no COLMAP or `pose_opt` run was made.
- **Exposure drift and coverage** were not recomputed by the reviewer. Full-resolution LPIPS was not
  computed (no torch locally). The user's screen size and pixel ratio are unknown.
- **No fix was tested end to end.** All "expected gain" figures for alternatives are estimates from
  papers or diagnoses, except the simulated no-mask re-cut.
- The actual Modal bill for the four L4 runs (estimated under $0.10) was not read back.

---

## Refuted or unproven (do not use in the paper)

- **"The chairs were moved between the two captures."** Refuted: same places within 4.7 / 7.8 cm
  (`review_chairs/r6_same_place.json`). The cause is RoomPlan's tidy chair row.
- **"Record RoomPlan and Stray in one session to fix the chairs."** Refuted: Chair1 is a phantom in
  its own scan.
- **"The dark chair is inside Chair1's box"** (from `chairs/scan1059_Chair1_f00003.jpg`): a 2D overlap
  in the photo, not true in 3D.
- **"Teal chair is 5-8 cm from Chair2."** Wrong: that was the size of the correction; the remaining
  offset is about 0.28 m.
- **Alignment error between splat and physics:** ruled out (1.05 cm fit residual; walls within a few
  cm; `alignment_check.json`).
- **The client draws the uncut room or physics meshes over the cut splats:** not seen in the code or
  `splat.json`; the live server was not checked.
- **"Motion blur in the video is the main cause of blur."** Refuted as main cause: input blur is about
  sigma 0.7 px, while the error sits at room scale (`review_blur/scale_check.json`).
- **"ARKit poses disagree by 2-5 px and, with blur, explain most of the gap."** Unproven: most pairs
  had 15-44 matches, some disagreed by 43-1155 px with no jump in the ARKit poses
  (`blur/pose_consistency.json`). Only one solid pair: about 2.4 px.
- **"The render matches a 1.3-8 px blur."** Misreported: it matches sigma 10 px or more (the largest
  tested) in 4 of 6 views (`blur/render_sharpness.json`).
- **"Training resolution is not the limit."** Unsupported: the +0.30 dB from 960 to 1440 px was scored
  at 480 px, where extra detail cannot show.
- **"Spark is 0.87 dB worse than gsplat."** Corrected to 0.66 mean / 0.55 median (double-alpha bug).
- **"SPZ and Spark compression cost about 0.1 dB at most."** Spark's repack yes (<= 0.05 dB); SPZ's own
  quantisation is unmeasured.
- **"Viewer magnification (retina) is a main cause."** Minor next to view extrapolation; the 1185 px/rad
  figure was an estimate.
