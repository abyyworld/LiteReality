# Any video in, best room out: plan (2026-11-25)

Written by a planning pass over an audit of this repo (ingest; train, export and cut) and a survey of robust casual-capture methods and shipping apps. File:line references come from the audits and were not all re-checked. A skeptical review follows the plan. The raw audits and surveys are in `inputs.md`.

PLAN: "any video in, best room out" for ~/LiteReality

Rule for every stage: it either improves the room or hands the previous stage's output forward with a plain-language note. A crash is a bug. A paid run on hopeless input is a bug. Publishing a worse room over a better one is a bug. File:line references come from the ingest and train audits. I did not re-verify them.

## 1. Pipeline stages, in order

**S0 Intake.** One upload takes .mov/.mp4, a Stray zip, a Scanner zip or photos, and detects the type itself. Today the upload refuses video (scans.py:98 suffixes, scans.py:89 200 MB cap), and import-video and import-stray are CLI-only (cli.py:1357-1460).
- Licence: our own code.
- Fallback: unknown type gives "we couldn't read this file; send the original video from Photos".

**S1 Normalise (CPU).**
- Tone map HDR (HLG/Dolby Vision, BT.2020) to sRGB with ffmpeg zscale+tonemap. Today it decodes straight to rgb24 (video.py:232) and detects HDR only as a log line (video.py:117-118, 721).
- Make every frame upright using gravity from the poses (ARKit, or MapAnything's up), not the rotation tag. Rotate the pixels and K together. This fixes sideways Stray frames (stray.py:19-21, capture/__init__.py:232-246) and mid-clip turns.
- Delete the hard-coded ROTATE_270 calls (opening_references.py:340, chair_type_judge.py:54, object_references.py:111, chair_clusters.py:105, refine_objects.py:317, render_room_cameras.py:328-339). Replace them with the pose-derived rule in realroom/crops.py:436-442.
- Fallback: if gravity is ambiguous, run a horizon/vanishing-line check per frame. If that also disagrees, report "orientation uncertain".

**S2 Pre-flight gate (CPU, free).** Measure duration, usable frames, translation baseline (stand-and-pan vs walk), walk extent vs room extent, blur histogram at full resolution, brightness/noise, and lens switches (focal jumps).
- Hard floor (no paid run): under about 5 s, under about 30 sharp frames, or zero translation. The answer names the reason and gives one re-film tip, using Apple RoomPlan/ObjectCapture wording: slow down, turn on a light, walk along the walls facing in.
- Above the floor the build always runs, and the weak points go into the report.

**S3 Poses.**
- ARKit/Stray: use the poses as they are, plus gsplat `pose_opt` (in 1.5.3, never enabled).
- Split at tracking jumps and treat each piece as its own segment, then register the segments. Today stray.py:600-615 keeps both sides of a relocalisation, which doubles walls.
- Plain video: run MapAnything-apache with real intrinsics from the QuickTime lens/focal metadata (cli.py:1573 passes only max_views; the fx is about 33% short, selfcal.py:3-4). In parallel run VGGT-1B-Commercial (+BA). Keep whichever has the better reprojection/photometric consistency. COLMAP 4 global mapper is the last resort.
- Split at lens switches instead of forcing one median K (app.py:407-414).
- Not allowed: MASt3R, VGGT-X, Pi3, DA3-Giant/Large (all NC).
- Fallback: the largest consistent segment, plus "only part of your video could be placed (0:12-0:41)".

**S4 Frame selection and masks.**
- Make the full-resolution blur picker the default on both paths (stray.py:119-135, 640-707; cli.py:1397). Add a blur picker to video, whose picker scores at 480 px with no blur score (video.py:77).
- Use soft per-frame weights instead of hard drops, under a coverage constraint so plain walls are not thinned out (fixes video.py:293-298).
- Mask people and pets with SAM 2 (Apache-2.0, already used) seeded by a permissively licensed detector. Avoid AGPL YOLO.
- Fallback: if masking fails, train unmasked and rely on S7's robust loss.

**S5 Metric frame without RoomPlan.**
- Fit the floor (RANSAC) and wall planes on the fused cloud, set Z up, take scale from LiDAR (or from MapAnything, flagged as ±10%).
- Generate a MuJoCo scene_xml from those planes so room.usdz is optional. Today quickroom/build.py:1145-1148, cli.py:745-755 and splatroom/__init__.py:276 all need it.
- If a RoomPlan scan exists, align to it as now.
- Fallback: if the floor was never seen, use the camera-height prior, and the report says "the floor was barely filmed".

**S6 Instant room.**
- Fuse incrementally so the 2.5 GB hosted cap holds (rgbd.py:339-358 concatenates every frame; scans.py:159). Serve it within about a minute. This is the guaranteed floor.

**S7 Train (gsplat on H100).**
- PPISP replaces the bilateral grid (app.py:102). It is Apache-2.0 and in gsplat main. Port it into patch_trainer.py on 1.5.3, or bump the pin.
- `--global-scale` from the seed-cloud extent (today scene_scale follows the walk: app.py:225).
- Seeds from the instant cloud, voxel-thinned evenly (fixes 4000 to 718: __init__.py:105-110, 648-656).
- Steps from the photo count, cap from the fused area (app.py:91-95, presets.py:33-38).
- dense_depth (PromptDA, Apache) and normal priors on.
- Raise the depth caps for confident depth (dataset.py:68-70 drops photos with under 20 points).
- SpotLessSplats-style robust loss, used only when residuals show transients.
- Production uses no hold-out. QA uses segment hold-out.

**S8 Repair (optional).** A progressive NVIDIA Fixer loop that we write ourselves, toward standing-height viewpoints over the floor. Code is Apache-2.0; weights are under the NVIDIA Open Model License.
- Generated views are down-weighted, and LiDAR depth constrains real views only.
- Do not ship Difix3D or its simple_trainer_difix3d.py. Its licence is research-only.
- It only switches on if it beats no-repair on our indoor test set.

**S9 Export.**
- Fold PPISP's exposure and 3x3 colour at the median frame into all SH coefficients, minding the DC offset. Apply the CRF as a Spark post-pass and drop vignetting.
- Ship SH3 if size allows (the hosted path ships SH1).
- Drop splats past ±2 km instead of raising (spz.py:265-266).

**S10 Publish gate.**
- Run each snapshot inside try/except (__init__.py:574-617).
- Publish a snapshot only if its NaN share is about 0, its splat count is sane, and it beats the instant room on segment-held-out frames. Otherwise keep the last good room.
- Kill the trainer process when the stream is abandoned (app.py:371-375).
- Add retries for upload and stream (remote.py:40-53, BundleUploader __init__.py:846-864).

**S11 Cut.**
- Without RoomPlan: generic objects from splat instances and SAM 2, with box colliders. This lifts the category, size and on-floor limits (instances.py:168-206, 60, 191).
- With RoomPlan: let instances move bodies, which `--keep-physics` turns off today (__init__.py:341, split.py:1159).
- Fallback: if the cut fails, serve a static room with "objects couldn't be separated; the room is view-only".

**S12 Report and viewer.**
- A per-position confidence map from camera distance and view overlap. It sets the spawn point and a soft zone in the call.
- The plain report says things like: "you walked 1.8 x 1.6 m of a 4.2 x 4.1 m room; the far corner was only seen from 3 m away"; "12 s were too blurry and were skipped"; "a person walked through and was removed".

## 2. Ranked work list (most value per unit of effort first)

| # | Item | Files | Effort | Modal cost to validate | Proof |
|---|---|---|---|---|---|
| 1 | Never-crash publish (S10) | splatroom/__init__.py:574-617, spz.py:265, splat/app.py:351, 371-375 | 0.5 d | ~$0.5 | Inject NaN, a 3 km splat and a MuJoCo failure: the instant room stays served and report.json is written every time |
| 2 | Honest eval harness | eval code, dataset.py:225-230 | 1-2 d | $0 (reuses runs) | Contiguous-segment and extrapolation hold-out, rendered from the exported SPZ in Spark (not the trainer), plus a sharpness ratio. Every later item is measured with this |
| 3 | Pre-flight gate + quality report (S2, S12) | capture/__init__.py:172-178, 304-312; video.py:927-944 | 2 d | $0 | Every bad clip in §3 gets the right sentence, and nothing under the hard floor starts an H100 |
| 4 | Upright frames from gravity (S1) | stray.py, capture/__init__.py:232-249, the six ROTATE_270 sites | 1-2 d | ~$1 | Horizon test passes on 100% of frames for portrait, landscape and mid-turn clips; SAM masks improve; PSNR no worse |
| 5 | Hosted settings, global-scale, even seeding, step scaling | scans.py:1462-1465; app.py:91-103, 225; __init__.py:105-110, 648-656; dataset.py:161-177; presets.py | 2 d | ~$10 | Seed density back to the 4000-per-photo level; segment-hold-out PSNR up; stand-and-pan and long-walk clips show no frozen splats or floaters |
| 6 | PPISP + SH bake | patch_trainer.py, app.py:102, spz export | 3-4 d | ~$5 | Held-out sharp frames rendered in Spark close most of the 3-5 dB gap; train-render vs export gap under 0.5 dB |
| 7 | Incremental fusion | rgbd.py:339-358 | 1 d | $0 | 637-frame atrium peaks under 1.5 GB |
| 8 | RoomPlan-free scene (S5) | quickroom/build.py:1145, cli.py:376-395, 745-755, splatroom/__init__.py:276 | 3-5 d | ~$3 | A plain .mov with no usdz gives a served room with movable objects; floor tilt under 1° |
| 9 | Video upload + auto importer | scans.py:89, 98; cli.py:1357-1460 | 1-2 d | ~$1 | Drag in a .mov from Photos and get a room |
| 10 | Blur picker default, soft weights | stray.py:119-135, 640-707; video.py:77, 293-298; cli.py:1397 | 1-2 d | ~$2 | Median blur down from 2.5 px; white-wall frame count kept; sharpness ratio up |
| 11 | Video intrinsics, lens split, HDR | video.py:117-118, 232; cli.py:1573; app.py:407-414 | 2 d | ~$3 | fx within 3% of the true iPhone value; colour-chart ΔE under 5 on HDR clips |
| 12 | Pose chain, pose_opt, jump segmentation | stray.py:600-615, geometry app | 1-2 wk | ~$20 | Every §3 clip gets a pose; no doubled walls on the relocalisation clip |
| 13 | Transient masking + robust loss | splatroom/masks.py, trainer | 3-5 d | ~$5 | People-walking clip has no ghost; clean clips change by under 0.1 dB |
| 14 | Confidence viewer | Spark viewer | 2-3 d | $0 | Spawn always lands in a high-confidence cell |
| 15 | Fixer loop + indoor fine-tune | new trainer stage | 2-3 wk | $50-150 | Off-path views win a blind A/B and extrapolation LPIPS drops; otherwise it stays off |

## 3. Robustness test suite

Record in two or three real rooms: a small bedroom, the atrium, and a large open-plan space over 8 m. Film each clip with both Stray and the iPhone Camera app (HDR on). Store the clips in results/robustness/ with a manifest of the expected report sentence and the pass floor.

Clips:
- Normal good walk (the control).
- Phone held portrait, held landscape, and turned mid-clip.
- Recording started with the phone flat (wrong tag).
- Fast pans and whip turns.
- Too short: 3 s and 8 s.
- Stand in one spot and pan.
- Small coverage: stay in a 1.5 m patch.
- Huge room, and a walk through two rooms.
- Dark room (one lamp), and a window-to-dark-corner exposure swing.
- A person walking through, and a pet.
- A large mirror, and a glass door or window.
- 0.5x/1x/zoom switch mid-clip.
- A 640-px messaging-app re-encode, and an Android or GoPro fisheye clip.
- An HDR clip.
- An ARKit relocalisation: cover the lens for 2 s.
- Furniture moved during the clip.

Pass, for every clip:
- No crash, and the report names the right cause.
- Something is served.
- No paid run for clips under the hard floor.
- Good clips stay above their PSNR floor.

Run the CPU gates on every commit and do a weekly paid run of about $15.

## 4. What stays impossible, and what the product says

- **Surfaces never filmed** (behind the sofa, a corner never in frame). They can only be invented. Shade them in the confidence map: "This area wasn't filmed; we filled it in."
- **Mirrors and glass** show a fake room behind the wall. "Mirrors can show a phantom room; we cut at the wall where we could."
- **Very dark or blown-out areas:** "Too dark to see here."
- **Scale without LiDAR** is an estimate: "Sizes may be off by about 10%."
- **Things that moved during filming:** "This chair moved while filming and may look doubled."
- **Viewing far from where the camera went** blurs, even with Fixer. The call keeps the viewer near the filmed space and says so.
- **A clip under the floor** cannot become a room. Say it at once, with one re-film tip, instead of after an H100 run.

No shipping product claims it "never fails" (Polycam, Scaniverse, Teleport). "Never fails" here means three things: something is always delivered, nothing worse ever replaces something better, and the user is always told the reason in plain words.

Unverified:
- Fixer quality indoors.
- Whether the Apache MapAnything checkpoint is much worse than the NC one.
- PPISP on gsplat 1.5.3 with batch size above 1.
- The Modal costs, which are estimates.


---

## Review

CRITIQUE (I spot-checked the repo read-only; I edited and ran nothing)

Confirmed against the repo:
- scans.py:89 and :98 (200 MB cap, no video suffixes).
- scans.py:1462-1465 (hosted path passes only --scene/--out/--keep-physics/--label).
- cli.py:1573 (only max_views is passed).
- deploy/modal/geometry/app.py:407-414 (one median K).
- splat app.py:225 (--no-normalize-world-space) and :88-104 (the defaults as stated).
- splat app.py:371-375 (the trainer process is never killed; the finally rmtree's the work dir).
- spz.py:265-266 (raises on a splat past 2 km).
- quickroom/build.py:1145-1148 (raises without room.usdz).
- `pose_opt` appears nowhere in src/ or deploy/.

The ROTATE_270 list is incomplete. The plan misses:
- room_render/annotate_views.py:150
- agent/tools/render/source/compose.py:161
- shared/image_selection/surface_views.py:202

Do the removal by grep, with a test, not from a hand-made list.

What is wrong or missing:

1. **The ordering is circular for plain video.** S1 makes frames upright from pose gravity, but poses only exist after S3. Camera-app .mov files carry no IMU track, so "gravity" there is MapAnything's own guess, made from frames that may be sideways. S2 has the same problem: translation baseline and walk extent can't be measured "free on CPU" without poses. Fix:
   - Run tag-rotate, then poses, then a gravity/horizon check, then re-rotate and re-pose if it was wrong.
   - Give S2 a cheap CPU proxy (optical-flow parallax) or accept a small GPU pose pass before the gate.
   - Pay for the VGGT+MapAnything ensemble only after the gate.

2. **The "guaranteed floor" isn't guaranteed for plain video.** S6's instant room needs depth. Without LiDAR, that depth comes from a paid GPU MapAnything call. Say so. Also define the floor for when that call fails: poses plus a sparse cloud, or a refusal with a reason.

3. **The primary input is under-ranked.** Most users will have no Stray app and often no LiDAR. Their input is Camera-app video: HDR, stabilised, auto-switching lenses. Items 8 (RoomPlan-free scene) and 9 (video upload) are CRIT prerequisites: without them a plain video gives nothing at all. Move them to ranks 2-3. Item 11 (intrinsics/HDR) belongs right after them.

4. **Turning the phone mid-clip gives mixed frame sizes.** That crashes batched training at dataset.py:339 (`size = kept[0]`) with gsplat collate (train audit #12). The plan doesn't cover it. Pad or crop to one size, or bucket by aspect.

5. **The publish gate contradicts the production rule.** S10 needs "segment-held-out frames", but S7 says production uses no hold-out. Reserve a few segment frames for gating and fold them back in for a final short pass, or gate on NaN share, splat count and training-view PSNR against the instant room.

6. **Intrinsics from metadata are unverified.** Stabilisation crops 10-20%, and the 35 mm-equivalent tag doesn't account for that. Validate fx on a checkerboard clip before trusting it, and keep self-calibration as the cross-check.

7. **Pinhole only.** The suite includes fisheye, stabilisation and rolling-shutter clips, but no stage handles them. Either undistort (or use 3DGUT with_ut) or make the gate say "unsupported lens" explicitly.

8. **Smaller items the plan drops from the audits:**
   - Video depth confidence is relative, so level 2 is trusted like LiDAR (ingest #15).
   - A photo under 20 points is dropped entirely, colour included. Keep it for colour with no depth loss; that needs a trainer patch.
   - Far-wall seeds are missing (train #8).
   - test_every is never validated, and fewer than 2 photos crashes (train #16).

9. **Cost and latency have no ceiling.** Limits today: MAX_CONTAINERS 2, the hosted child's 1200 s, Modal's 1800 s. The pose ensemble roughly doubles the GPU work, and Fixer adds 3-6x the training time. Set a per-room $ and wall-time cap, plus queue feedback ("third room waits").

10. **Upload.** 4K60 HDR is about 400 MB/min. Raising the cap isn't enough: it needs a resumable, chunked upload or an on-device transcode.

11. **Licence and ops.** VGGT-1B-Commercial is gated, and its licence excludes military use, so it needs legal sign-off first. Measure MapAnything-apache against the NC checkpoint before building the ensemble. It may be enough, or it may be useless.

12. **PPISP export.** Baking at the median frame looks wrong on window/dark swings. Use the controller's prediction for a canonical view. PPISP has a batch-1 limit (simple_trainer.py:472), so plan the 1.5.3 port as a real risk, not 3-4 days.

13. **The scale figure is unsupported.** The "±10%" has no source; the sim3 search grid of 0.7-1.4x suggests the error is larger. Add sanity checks against known sizes (door height, ceiling 2.3-3 m) and report what they find.

14. **The test suite is biased.** The developer filming "bad" clips on purpose isn't how users fail. Add 5+ clips from people told only "film your room". The suite is also about 96 clips, which $15 a week can't cover: tier it, with CPU gates on every clip and a paid run on a rotating subset.

15. **Fixer.** Add a blind A/B budget cap. Also check that the NVIDIA Open Model License terms (attribution, guardrail clauses) suit a hosted service.

Files checked: /Users/abyyworld/LiteReality/src/litereality_agent/room_ops/roomcall/scans.py, /Users/abyyworld/LiteReality/src/litereality_agent/cli.py, /Users/abyyworld/LiteReality/deploy/modal/splat/app.py, /Users/abyyworld/LiteReality/deploy/modal/geometry/app.py, /Users/abyyworld/LiteReality/src/litereality_agent/models/splat/spz.py, /Users/abyyworld/LiteReality/src/litereality_agent/pipeline/splatroom/__init__.py, /Users/abyyworld/LiteReality/src/litereality_agent/pipeline/quickroom/build.py, /Users/abyyworld/LiteReality/src/litereality_agent/models/capture/video.py
