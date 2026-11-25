# Audit and research inputs to PLAN.md

## audit: ingest
## INGEST audit: ranked list of the assumptions that break on a capture filmed any way

Repo: /Users/abyyworld/LiteReality. I only read the code; nothing was edited or run. Severity scale: CRIT means the product cannot work for "people just film". HIGH means a silent bad result on common footage. MED means a silent bad result on some footage. LOW means it is handled, or it is rare.

### 1. CRIT: A capture that comes with no RoomPlan scan of the same room cannot become a room
- **Where:**
  - `src/litereality_agent/pipeline/quickroom/build.py:1145-1148` raises FileNotFoundError when there is no room.usdz.
  - `src/litereality_agent/pipeline/splatroom/__init__.py:276` needs `scene_xml`, which is the MuJoCo scene built from that usdz.
  - Both importers get room.usdz only by copying it from `--align-to`, and only warn when it is missing: `models/capture/video.py:889-894` and `pipeline/capture/__init__.py:264-270`.
- **What breaks it:** a plain phone video, or a Stray recording, with no LiteReality Scanner scan of the room.
- **What happens today:** the import succeeds with a warning. Then quickroom crashes, so splatroom never runs. Every "just film" path also needs a second, special capture (LiDAR iPhone, RoomPlan app).

### 2. CRIT: The room-call upload cannot take a video at all
- **Where:**
  - `src/litereality_agent/room_ops/roomcall/scans.py:98` sets `ALLOWED_SUFFIXES = (.usdz .usda .jpg .jpeg .png .json .pcd .txt)`: no .mov or .mp4.
  - `scans.py:3-5`: the upload must be a LiteReality Scanner zip.
  - `scans.py:89`: `MAX_UPLOAD_BYTES` is 200 MB, below a minute of iPhone 4K.
- **What happens today:** a video or a Stray zip is refused, or its rgb.mp4 is never unpacked. `import-video` and `import-stray` exist only as CLI commands (`cli.py:1357-1460`).
- **More:** there is no automatic choice between the two importers. Each must be picked by hand, and neither handles the other's input.

### 3. HIGH: How much of the room the walk covered is never measured
- **Where:** the import stats and report (`pipeline/capture/__init__.py:172-178, 304-312`) and the video report (`video.py:927-944`) have no trajectory extent, no floor-reach figure and no view-direction spread.
- **What breaks it:** a walk confined to a small patch, as in the atrium: 1.8 x 1.6 m of a 4.2 x 4.1 m room. This is item 1 in `results/diagnosis/README.md:157`.
- **What happens today:** silent. Training goes ahead, and the view extrapolation it causes is never flagged.

### 4. HIGH: The default frame picker cannot see 2-4 px motion blur
- **Stray path:**
  - `pipeline/capture/__init__.py:97` defaults to `picker="laplacian"`, scored at `SCORE_WIDTH=480` (`stray.py:102`). `cli.py:1397` leaves it at that default.
  - The full-resolution blur picker exists (`stray.py:119-135, 640-684, 703-707`) but is opt-in.
  - On the atrium this gave a median blur of 2.5 px, with 28% of frames at 4 px or more.
- **Video path:**
  - `video.py:77` (`SCORE_LONG=480`). There is no blur picker at all.
  - `video.py:293-298` drops any pick below 0.3 x the 90th percentile of the frames within ±1.5 s, with no motion gate. That is the failure `stray.py:109-113` warns about: sharp views of white walls and ceilings are dropped next to textured ones.
- **What happens today:** silent. Training frames are blurry, and plain surfaces are under-sampled.

### 5. HIGH: Exposure and white balance per frame are never recorded or evened out at ingest
- **Where:**
  - `stray.frame_json` (`stray.py:949-958`) has no exposure, ISO or white-balance fields. Stray does not record them (`stray.py:19-55`), and the video path has none either.
  - The only compensation is the trainer's bilateral grid per photo (`deploy/modal/splat/app.py:102`), which the export drops.
  - The instant room's averaging is a mean colour across views (`rgbd.py:7`), not a correction.
- **What breaks it:** any auto-exposure change, such as turning from a window to a dark corner.
- **What happens today:** silent. Held-out colour error is 3-5 dB.
- **Fix direction:** estimate a gain and white balance per frame from overlapping pixels at ingest, and apply it to the photos before training.

### 6. HIGH: Phone-video focal length is known to be wrong and is never corrected
- **Focal length:**
  - `selfcal.py:3-4` notes that MapAnything put an iPhone fx about 33% short.
  - `video.py:629-635` only warns about it.
  - The model's `intrinsics` option exists (`deploy/modal/geometry/app.py:37-41`), but `cli.py:1573` passes only `max_views`. Nothing reads the lens or focal-length metadata from the file.
- **Lens switches:**
  - `app.py:407-414` forces one median K over the whole video.
  - A lens switch (auto-macro to the ultra-wide, 0.5x, pinch zoom) breaks that. It only shows as a spread warning (`video.py:603-606`).
- **What happens today:** silent. Cameras sit nearer or farther than they were, and depth scales with them, so splats come out blurry.

### 7. HIGH: Phone-video poses are never refined
- **Where:**
  - Poses are feed-forward at 518 px (`app.py:95` `RESOLUTION_SET`), with no bundle adjustment.
  - Passes beyond 320 views are chained by a Sim(3) fitted on anchor views (`app.py:214-225, 350-371`).
  - The trainer defaults have no pose optimisation (`app.py:95-128`).
- **What happens today:** silent. 1920-px photos are trained against poses that are only accurate to about the 518-px scale, which gives ghosting and softness.

### 8. HIGH: A phone video whose rotation tag is wrong, or changes, comes out sideways
- **Where:**
  - Orientation comes only from the container's display matrix (`video.py:148-155`).
  - The gravity guess is the cameras' mean image-up (`video.py:775-777`).
  - `estimate_up` searches only within 30°, 15° and 8° windows (`sim3.py:46, 156-169`).
- **What breaks it:**
  - Starting the recording with the phone flat, which tags the wrong orientation.
  - Turning the phone mid-clip: iOS keeps the starting orientation, so the rest of the clip is sideways.
  - A file whose tag was stripped.
- **What happens today:**
  - If image-up is about 90° off gravity, walls fall in the "horizontal" set, so the estimate can lock onto a wall normal and level the room on its side.
  - With `--align-to`, the yaw-only alignment then fails the residual check: a crash with a message.
  - Without `--align-to`: a silent sideways room.

### 9. HIGH: Portrait-held Stray frames are stored sideways and passed on as they are
- **Where:**
  - `stray.py:19-21` (landscape sensor frames, no rotation tag). The importer writes them unrotated: `pipeline/capture/__init__.py:232-246` and `_save_jpeg` (`pipeline/capture/__init__.py:344-349`).
  - The poses are right, but every image model gets the room on its side: SAM 2 (`splatroom/masks.py:210-216`), PromptDA (`models/splat/promptda.py`, no rotation), and the Difix/Fixer repair, whose priors are trained on upright images.
- **Hard-coded rotations elsewhere:** other code always applies `ROTATE_270`, assuming a portrait-held capture:
  - `scene_init/ingest/references/opening_references.py:340`
  - `chair_type_judge.py:54`
  - `object_references.py:111`
  - `chair_clusters.py:105`
  - `realism_authoring/author/refine_objects.py:317`
  - `room_render/render_room_cameras.py:328-339`
- **What happens today:**
  - Model-based steps get worse results, silently.
  - A landscape-held capture, or any phone-video import (already upright through ffmpeg's autorotate), is turned the wrong way by those hard-coded rotations.
  - Only `realroom/crops.py:436-442` (`upright_turns`) works the rotation out from the pose, which is the correct way.

### 10. MED-HIGH: HDR video (the iPhone default: HLG / Dolby Vision, 10-bit BT.2020) is not tone mapped or converted to sRGB colours
- **Where:**
  - `video.py:11-15` documents this as a choice.
  - `video.py:232` decodes straight to `-pix_fmt rgb24`.
  - `video.py:117-118` detects HDR, but it only appears in a log line (`video.py:721`), not in the warnings.
- **What happens today:** silent. BT.2020 colours are read as sRGB/BT.709 (only the matrix conversion happens), and the HLG curve is read as SDR gamma. Colours come out washed out and flat compared with the room.

### 11. MED-HIGH: Moving people and pets are not masked anywhere
- **Where:**
  - A grep across `models/capture`, `models/splat`, `pipeline/splatroom` and `deploy/modal/splat` finds no masking of people or anything moving.
  - LiDAR or model depth on a person becomes seed points (`dataset.py:121-139`).
  - Frames are chosen on sharpness only.
  - In the video picker, a person walking through the frame counts as camera motion (`video.py:283-287`), so frames cluster where the person was.
- **What happens today:** ghosts and floaters. The splat instance clustering could plausibly turn one into a phantom body; I did not verify that.

### 12. MED-HIGH: Tracking jumps and walks that never close the loop
- **Where:** `stray.tracking_ok` (`stray.py:600-615`; limits at `stray.py:141-143`) drops only ±0.25 s around a step faster than 3 m/s or 6 rad/s.
- **What happens today:**
  - The stretches before and after an ARKit relocalisation are both kept, although they are in inconsistent frames (drift before the jump is never corrected). The result is doubled walls.
  - Steps under about 5 cm at 60 fps (about 10 cm at 30 fps) are not detected at all.
  - Slow drift on a walk that never closes the loop is invisible.
  - All of this is silent.

### 13. MED: Too-short or too-shaky clips have no minimum
- **Where:**
  - The Stray import refuses only when 0 frames are left (`pipeline/capture/__init__.py:179-180`).
  - The video import refuses only under 3 frames (`video.py:726, 844`).
  - The dataset needs just one frame with 20 points (`dataset.py:70, 281-290`).
  - The blur picker with `max_blur_px` leaves windows empty with no floor (`stray.py:704-707`).
- **What happens today:** a 3-second or very shaky clip goes straight to a paid H100 run, and the result is silently bad. There is no minimum duration, frame count, baseline or coverage.

### 14. MED: Video metric scale is unchecked without a target scan
- **Where:**
  - The scale is MapAnything's own (`video.py:818, 913-916`).
  - The search grid when aligning is 0.7-1.4x (`sim3.py:42`), which says errors that large are expected.
- **What happens today:** silent. Physics objects come out at the wrong size.

### 15. MED: Video depth confidence is relative to the video, not absolute
- **Where:** `video.py:404-422`, with `CONF_LOW=0.10` and `CONF_MID=0.30` at `video.py:83`. About 70% of valid pixels are always level 2, however poor the video.
- **Why it matters:** downstream, level 2 is trusted as LiDAR-grade (`rgbd.py:44`, `dataset.py:69`).
- **What happens today:** silent. Glass, mirrors, dark areas and plain walls get "high"-confidence depth.

### 16. MED: Mirrors, glass and windows
- **Where:**
  - Depth behind a mirror, or through glass, becomes seed points.
  - Only the video path checks frames against the target (`video.py:639-667`, `is_bad` at `video.py:663-667`). That needs `--align-to` and drops whole frames, not pixels.
  - Stray has no check at all.
  - The RoomPlan walls rule runs only with `clean`, which is off by default (`splatroom/__init__.py:23, 36-37`).
- **What happens today:** silent floaters, or a mirrored room behind the wall.

### 17. MED: Depth range caps
- **Where:**
  - `rgbd.MAX_DEPTH=4.5` (`rgbd.py:42`)
  - `dataset.MAX_DEPTH=5.0` (`dataset.py:68`)
  - `CLOUD_MAX_DEPTH=6.0` (`video.py:84`)
  - Frames with fewer than 20 points are dropped (`dataset.py:70, 281`).
- **What breaks it:** large or open-plan rooms, or a walk that stays on one side.
- **What happens today:** silent. Far walls get no seed points.

### 18. MED: Very dark rooms
- **Where:**
  - Laplacian variance (`stray.py:432-440`, used by both pickers) rises with sensor noise, so the noisiest frames rank as "sharpest".
  - There is no brightness or noise check.
  - `BLUR_TURN_RATE` assumes a 1/60 s exposure (`stray.py:117`), while dim video runs at 1/30 s.
- **What happens today:** silent.

### 19. MED: Lens model, stabilisation and rolling shutter
- **Where:** everything assumes a pinhole camera: `dataset.py:314` (COLMAP model 1 = PINHOLE), `app.py:96-100`, and `video.py:328-343`.
- **What breaks it:** the iPhone Camera app's stabilisation re-warps every frame, rolling shutter on fast pans, leftover ultra-wide distortion, and Android or GoPro fisheye lenses.
- **What happens today:** silent.

### 20. LOW-MED: Low resolution and heavy compression
- **Where:**
  - `scaled_size` never scales up (`video.py:166-170`), but `dataset.py:268-271` resizes every photo to 960 wide, so a 640-px messaging-app video is upsampled.
  - There is no resolution or bitrate floor.
  - Stray video is about 0.082 bits/pixel. It is then re-encoded to JPEG at quality 92 (`stray.py:144`) and again at 95 (`dataset.py:72`).
- **What happens today:** silent loss of detail.

### 21. LOW-MED: Fixed frame targets whatever the length or area
- **Where:**
  - 200 frames for video (`cli.py:1432`, `video.py:76`) and 300 for Stray (`cli.py:1370`).
  - `seed_points_for` divides the splat budget over the photos (`splatroom/__init__.py:105-110`): 557 photos gave 718 points per photo.
  - MapAnything is capped at 1000 frames per call (`app.py:91`), with a 1200 s timeout (`app.py:90`).
- **What happens today:** a long walk of a big home is under-sampled; a short one may be over-seeded.

### 22. LOW (handled): Stray format problems
- A frame count that does not match the odometry rows is refused with a clear message (`stray.py:492-571`); `--allow-frame-mismatch` overrides.
- Headerless or old CSVs, .npy depth, and zips are all read (`stray.py:230-325`).
- A recording with no depth (no LiDAR, or depth turned off) ends in "no usable frame" (`pipeline/capture/__init__.py:159, 179-180`): a crash with a message.
- The principal-point sanity check is only a warning (`stray.py:359-362`).

### 23. LOW (handled): Portrait vs landscape when the rotation tag is right
- ffmpeg's autorotate and the size swap handle it (`video.py:148-155, 173-179`).
- If ffmpeg reports no per-frame times, the frame times fall back to index / fps (`video.py:218-221`).

### 24. LOW (handled): Duplicate frames
- Both pickers drop a frame that repeats the last kept one: by pose for Stray (`stray.py:759-773`) and by thumbnail for video (`video.py:300-308`).
- A long pause collapses to a single frame. That costs some frames but does no harm.

**What the ingest is missing for "film any way":** one upload that takes video (items 1-2); a RoomPlan-free route from capture to physics (1); orientation worked out per frame from gravity, not from tags or hard-coded rotations (8, 9); the full-resolution blur picker on by default for both paths (4); photometric normalisation and HDR→sRGB conversion before training (5, 10); a fixed focal, plus pose refinement or bundle adjustment, for video (6, 7); masking of moving things (11); coverage and minimum-quality gates that tell the person what to re-film instead of training anyway (3, 13).


## audit: train
AUDIT: TRAIN + EXPORT + CUT. I read the code only and ran nothing on Modal. The gsplat v1.5.3 sources I checked are a local copy at /private/tmp/claude-501/-Users-abyyworld-Desktop-llms-working-here-litereality/3dba2112-c4f5-423b-8511-99fb562955c0/scratchpad/gsplat_src. Repo paths below are relative to ~/LiteReality. Findings are ranked most severe first.

1. CRITICAL: no RoomPlan scene means no splat room at all
- Where: src/litereality_agent/cli.py:376-395 (`_quick_target` returns None without room.usdz) and cli.py:745-755 (SystemExit "no RoomPlan room.usdz found" or "no physics scene"). Also pipeline/splatroom/__init__.py:276 (`MjModel.from_xml_path(scene_xml)` is required) and split() on every bundle. The importers (models/capture/video.py:890, pipeline/capture/__init__.py:266) only warn.
- Breaking input: a Stray or plain phone video with no RoomPlan target, or any capture RoomPlan did not run on.
- Current behaviour: the command exits. You get not even the instant room, although training needs only frames and depth.
- Note: once this is lifted, every cut constant becomes dependent on MapAnything's own metric scale and gravity estimate (video.py:33-37) when there is no target (see item 18).

2. HIGH: the hosted product path runs the weakest settings
- Where: room_ops/roomcall/scans.py:1462-1465 passes only `--scene --out --keep-physics --label`. It passes no `--trainer-preset`, `--instances`, `--depth`, `--no-holdout` or `--clean`.
- What users therefore get (deploy/modal/splat/app.py:91-103):
  - `export_sh_degree` 1, which drops the SH2-3 texture.
  - The fused 16x16x8 bilateral grid, which the export drops. This is the known 3-5 dB colour error.
  - `test_every` 8: one photo in eight is never trained on and never seeds.
  - cap 800k and 1250 steps.
- The q_* presets also need the ephemeral trainer, because the deployed app predates the patch (app.py:42-44).
- Current behaviour: every real user gets the known-bad colour path, and their chairs are cut by unmoved RoomPlan boxes (see item 15).

3. HIGH: any exception on a trained snapshot kills the whole build
- Where: pipeline/splatroom/__init__.py:574-617. Inside the trainer event loop, `cleaned`, `publish`, `split`, `fit_scene` and `write_bundle` run with no try.
- Sources of exceptions:
  - instances.py:691-702 re-raises a MuJoCo load failure.
  - models/splat/spz.py:265-266 raises "a splat lies more than 2 km" instead of dropping that splat.
  - OOMs in split or instances (see item 14).
- On Modal, the same spz raise at app.py:351-352 turns into an "error" event, and every later snapshot is lost.
- Current behaviour: the build raises, report.json is not written, and the trained room is never served.
- Side effect: the stream is abandoned while the GPU keeps training. app.py:371-375 never kills `proc`, and its finally rmtree's the working directory under the running trainer.

4. HIGH: no quality gate before a trained snapshot replaces the instant room
- Where: __init__.py:593-605 publishes every snapshot unconditionally. spz.py:257-261 silently drops non-finite splats.
- Breaking input: a NaN run. This has been seen before: dataset.py:74-76 records MCMC going NaN within 90 steps.
- Current behaviour: an empty or garbage "trained" room is published over the good instant room. There is no splat-count, NaN-share or PSNR check against the instant room, and the held-out stats only arrive at "done".

5. HIGH: learning rates scale with how far the camera walked, not with the room
- Where: app.py:225 passes `--no-normalize-world-space`. In gsplat:
  - examples/datasets/colmap.py:345-348 sets scene_scale = the largest camera distance from the cameras' mean.
  - examples/simple_trainer.py:259 sets means_lr x scene_scale, and :705 multiplies the depth loss by scene_scale.
  - MCMC's position noise follows means_lr.
- Breaking inputs:
  - Stand-and-pan (about 0.1 m spread): position learning rate and depth loss drop about 10x, so splats freeze near their seeds and depth is ignored.
  - A long walk through a flat (about 5-10 m spread): 5-10x the learning rate and noise, which means floaters.
  - The atrium's 1.8 x 1.6 m walk sits near 1, which is why the test rooms work.
- Fix: pass `--global-scale` so that scene_scale x global_scale equals a room-size constant taken from the seed cloud.

6. HIGH (estimated, not measured): the hosted 2.5 GB memory cap against instant fusion
- Where: scans.py:159 (REALISTIC_MEMORY = 2560 MiB) and :1466-1467 (the splat child runs under it).
- Cause: models/splat/rgbd.py:339-358 concatenates every frame's surfels before the voxel reduce. That is about 44 B per sample, about 200k samples per frame at 512x384, and roughly 140 B per sample at peak with the sort temporaries.
- Breaking input: my rough estimate is that anything over about 100-150 LiDAR frames goes over the cap (atrium has 637 frames; the Stray importer targets 300).
- Current behaviour: the child is killed, so there is no photoreal room.
- Fix: voxel-reduce incrementally, frame batch by frame batch.

7. HIGH: seeds are random per photo, not even across the room (the 4000 to 718 regression)
- Where: __init__.py:105-110 and 648-656 (per_photo = min(4000, cap*0.5 / seeding photos)), and dataset.py:161-177.
- The arithmetic: 400000 / 557 seeding photos = 718.
- Problem: areas seen by few photos get few seeds. MCMC only adds splats by sampling existing ones by opacity (gsplat strategy/ops.py `_multinomial_sample`), so an area seeded thinly, or not at all, never recovers.
- Fix: seed from the instant room's voxel-fused cloud, thinned to cap*0.5 evenly across the room. Make cap_max scale with the fused surface area.

8. HIGH for big rooms or outdoor-ish scenes: hard depth caps and no far seeds
- Where:
  - dataset.py:68 (MAX_DEPTH 5.0), rgbd.py:42 (4.5).
  - dataset.py:70 and 281-284: a photo with fewer than 20 usable points is dropped from training entirely, colour included. gsplat's depth loss (colmap.py Dataset `point_indices[image_name]`) needs points for every training photo.
  - There are no background or sky seeds.
- Breaking inputs: a hall over 5 m, windows, outdoors, or MapAnything or PromptDA depth that does see 10 m or more.
- Current behaviour: far walls are missing or smeared into floaters, the photos aimed at them are discarded, and a mostly-far scene raises "none of the N frames has 20 usable LiDAR points".

9. MEDIUM-HIGH: steps and splat budget are fixed whatever the capture
- Where: app.py:91-95 (1250 steps, cap 800k) and models/splat/presets.py:33-38 (10000 steps x 4, cap 2.4M).
- Breaking inputs: a 50-photo capture gets about 800 views per photo (the known overfitting); a 2000-photo capture gets about 20. A closet and a four-room flat get the same budget.
- The preset's snapshots also become {400, 800, 1250, 10000}, which leaves minutes with no update on screen.
- Fix: scale steps with the photo count (a fixed number of epochs) and cap with the fused area.

10. MEDIUM: the hold-out costs coverage in production
- Where: dataset.py:225-230, 303 and 319; app.py:100; __init__.py:94 and 649-651.
- On a thin capture, one photo in eight never trains and gives no seeds. Only paper metrics need this.
- `--no-holdout` exists (it sets 100000, so only frame 0 is held out), but the hosted path does not pass it.

11. MEDIUM: objects are found only for categories RoomPlan boxed, and only on the floor
- Where:
  - instances.py:168-185: categories come only from the scene's free bodies.
  - instances.py:188-206 (`_classify`): an instance must fit an existing category's size, within 0.5-2x footprint and 0.6-1.6x height.
  - instances.py:60 and 191: STAND_MAX 0.5 m, so objects must stand on the floor.
  - instances.py:662-664: a clone needs a donor body of the same category.
  - instances.py:680-686: a body with under 300 opaque splats in its box is removed as a phantom.
- Breaking inputs: RoomPlan missed the chairs, or the object is a stool, box, plant or lamp, or it sits on a table.
- Current behaviour: nothing is grabbable, and move_test "pass" is False whenever no object is found (instances.py:954, `bool(rows)`). A real but thinly captured chair can be deleted as a phantom.

12. MEDIUM: photos of different sizes crash batched training
- Where: dataset.py:268-269 (height per frame) and :339 (`size = kept[0]`). gsplat simple_trainer.py:589-594 uses a DataLoader with batch_size 4, whose default collate stacks the images.
- Breaking input: a photo capture mixing portrait and landscape, or a scan merged from two videos of different resolution or aspect.
- Current behaviour: the trainer crashes at step 0 and sends an error event. A single phone video is not affected.
- Checked and fine: the sideways atrium frames train correctly, because K and c2w are self-consistent.

13. MEDIUM: networks, timeouts and limits
- Where and what:
  - models/splat/remote.py:40-53 makes one attempt with no retry or resume. A dropped tar upload or stream leaves only the instant room.
  - BundleUploader (__init__.py:846-864) never retries a failed final bundle, so the call can stay on an older room.
  - Time limits: Modal 1800 s (app.py:83), the hosted child 1200 s (scans.py:803).
  - MAX_CONTAINERS 2 (app.py:82): a third simultaneous room waits with no feedback and can hit the child's timeout.
  - MAX_DATASET_BYTES 400 MB (app.py:84 and 309): a very long capture (my estimate is over about 2500 photos) is refused, with no fallback to smaller or lower-quality images.
- Unverified: whether Modal cancels the container when the client goes away (see item 3 for the unkilled trainer process).

14. MEDIUM: the cut's cost grows with room size
- `_grow_legs` (split.py:301-357):
  - Its grid spans every free body's zone, and above 50M voxels it silently skips leg growth (306-307).
  - It holds a bodies x voxels boolean array, and each round dilates every body over the whole grid.
  - It runs per snapshot in split (1272-1283) and again in left_behind (402-411).
  - My estimate for a 20x20 m hall with 30 chairs: about 18M voxels, 0.5 GB or more, and minutes per snapshot.
- `instances.find` (instances.py:375-380): its voxel grid covers every opaque splat in the height band. It is limited to the room only when `_room_xy` finds walls (224-235). Without walls, far floaters can blow it up (100 x 100 m is about 800M int32 voxels, an OOM).
- `split` always calls `objects.find`, even without `--instances` (split.py:1197-1199).

15. MEDIUM: the hosted cut assumes RoomPlan's boxes are right
- Where: `--keep-physics` means move_bodies=False (__init__.py:341), so split.py:1159 turns off snapping and instances.
- The SAM masks still reassign splats (split.py:1285-1297), but `from_votes` is False, so the bodies never move.
- Breaking input: a chair 20-30 cm off its RoomPlan box, which is common.
- Current behaviour: its splat is cut into a body whose collider is elsewhere, so a grabbed chair floats beside its collider.

16. LOW-MEDIUM: very few frames or missing files
- Fewer than 2 kept photos: gsplat's train split (indices % test_every != 0) is empty, and scene_scale is 0. Result: a crash, or a position learning rate of 0.
- rgbd.py:79-81 silently drops frames that have no depth png, so those photos never train.
- `_options` (app.py:176-213) does not validate test_every. With test_every 0, dataset.py holds out nothing while gsplat computes `% 0`.

17. LOW: SPZ export limits
- Positions beyond ±2048 m raise an error instead of being dropped (spz.py:265-266).
- Higher SH coefficients are clipped to ±1 (`_quantize_sh`), and scales are limited to e^-10 to e^5.9 m.
- The per-photo colour correction is never folded into the splats, so dropping it at export is what causes item 2's colour error.

18. LOW, until item 1 is lifted: metric and Z-up assumptions
- Every cut and fusion constant is in metres and assumes Z is up: split.py:57-81, instances.py:47-70, INSTANT_VOXEL 0.016, MAX_DEPTH.
- This holds for ARKit. For MapAnything without a target, scale and up come from the model.
- If the capture never looks at the floor, floor_height falls back to its guess (instances.py:212-221). Without free bodies, that guess is the opaque splats' 2nd height percentile (instances.py:355-357), which floaters under the floor can pull down.

19. LOW: motion blur is not handled at the training stage
- dataset.build weights every kept photo equally (dataset.py:267-287). There is no per-photo blur weight or blur model in the loss.
- Blur is only screened at import (stray.py:132, MAX_BLUR_PX 2.0).

Cheapest fixes for the most "never fails" effect:
- Item 3: wrap each snapshot's processing in try and keep the previous bundle; drop splats past the 2 km limit instead of raising.
- Item 4: gate a snapshot on NaN share and splat count, and on beating the instant room.
- Item 5: set `--global-scale` from the seed cloud's extent.
- Item 2: send the hosted path a q_* preset with no hold-out, and redeploy the trainer.
- Item 7: seed by voxel from the instant cloud.
- Item 6: fuse frames incrementally.
- Item 1: make the RoomPlan scene optional for training and serving.

## research: robust
RESEARCH REPORT: making casual, unguided phone video of a room produce a good Gaussian-splat room that degrades gracefully instead of failing

Scope: I read the papers, READMEs and LICENSE files, and the repo read-only. I edited nothing, committed nothing, and ran nothing on Modal or anywhere that costs money. LICENSE files were fetched with curl and grepped.

Current setup as found in the repo (for context):
- ARKit poses are used directly, with no COLMAP run.
- gsplat 1.5.3 simple_trainer with MCMC, a LiDAR depth loss and a fused bilateral grid.
- Short runs: 1250 steps at batch 4.
- `pose_opt` is never enabled. `dense_depth` (PromptDA) and `normal_lambda` are off by default.
- The SH3 to SH1 cut at export is already in place.
- Frame picking is by sharpness in stray.py.
- results/diagnosis/README.md ranks the causes:
  1. View extrapolation.
  2. Bilateral-grid colour error, about 3-5 dB.
  3. Room-scale floaters.
  4. Blur, which is secondary.

======================================================================
RANKED RECOMMENDATIONS (value for a "never fails, best possible result" product)
======================================================================

1. Replace the bilateral grid with PPISP and bake a canonical ISP into the export. Highest value, near zero cost.
- What it is: NVIDIA, CVPR 2026 oral. A low-capacity, physically based camera model: per-frame exposure, vignetting, a 3x3 chromaticity homography, and a camera response curve (CRF).
  - A small "controller" learns to predict exposure and white balance for novel views, like a camera's auto exposure.
  - It can take metadata such as relative exposure as an input.
- Why it fits our biggest measured problem: on Tanks and Temples the paper measures training-view vs novel-view PSNR (Table 5):
  - Bilateral grid (BilaRF): 26.87 / 19.78.
  - PPISP: 25.85 / 24.62.
  - That is +4.8 dB on novel views. The grid's capacity overfits: it absorbs geometry and colour error, which is then lost at export. This is exactly our 3-5 dB "patchy colour" gap.
  - Hybrid PPISP + BilaRF: 23.52 on novel views.
  - Table 1, held-out PSNR (BilaRF vs PPISP with controller): Mip-NeRF 360 3DGS 25.46 vs 27.97; T&T 3DGS 20.58 vs 24.36.
  - Their own dataset includes an iPhone 13 Pro.
  - Compute overhead: 3% without the controller, 26% with it. The grid's is 36%.
- Licence: Apache-2.0 (github.com/nv-tlabs/ppisp, LICENSE checked). Commercial use is fine.
- Maturity: in gsplat main since Jan 2026 (`--post_processing ppisp`, `ppisp_use_controller`, `ppisp_controller_distillation`). Works with MCMC since Jul 2026.
  - It is not in a tagged release: PyPI is still v1.5.3 from Jul 2025, and the unreleased v1.6.0 needs PyTorch 2.7+.
  - The stock trainer refuses batch_size != 1 with PPISP (simple_trainer.py line 472). Either run batch 1, which costs about 2x wall time on a 33 s job, or patch it.
  - The `ppisp` package is a generic post-process (`rgb_out = ppisp(rgb_raw, camera_idx, frame_idx)`), so it could also be patched into our 1.5.3 trainer.
- Export (our design, not from the paper):
  - Exposure scale and the 3x3 colour transform are linear in radiance, so they fold exactly into all SH coefficients at a chosen canonical setting, such as the median frame. Mind the 3DGS SH DC 0.5 offset.
  - The CRF is nonlinear and is applied after alpha compositing. Either do a short colour-only distillation, training SH colours so the raw render matches canonicalISP(render), or apply the CRF as a fullscreen post-pass in Spark/three.js. The CRF is a few parameters.
  - Drop vignetting at export.
- Capture-side bonus: ARKit exposes per-frame `ARCamera.exposureDuration` and `exposureOffset` (EV). Our own scanner app could log them and feed them to PPISP's metadata path. Stray does not record them.
  - https://developer.apple.com/documentation/arkit/arcamera/exposureoffset
  - https://developer.apple.com/documentation/arkit/arcamera/exposureduration
- Cheap stopgap until then: `bilateral_mean_lambda` > 0 and a coarser grid, which the trainer already supports.
- Expected gain for us: a large share of the 3-5 dB colour gap, plus fewer floaters, because geometry can no longer hide behind the grid. Cost: about $0 extra.
- Sources:
  - https://arxiv.org/abs/2601.18336
  - https://research.nvidia.com/labs/sil/projects/ppisp
  - https://github.com/nv-tlabs/ppisp
  - https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/README.md
  - https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py

2. Generative repair of under-covered views with NVIDIA Fixer, using a progressive loop we write ourselves. Highest value for "people stand where the camera never was".
- What it is: a single-step diffusion transformer, 0.6B, based on Cosmos-Predict-0.6B. Images in and out at 576x1024. About 26.5 ms per frame on H100 (50.6 ms A100, 87.8 ms L40).
  - Offline mode: clean up pseudo views and distil them back into 3D.
  - Online mode: post-render enhancer. That needs a GPU, so it does not fit a browser viewer.
- Licence:
  - Code: Apache-2.0 (LICENSE checked).
  - Weights: NVIDIA Open Model License. The model card says it is ready for commercial use.
  - Difix / Difix3D (the test running now) is NOT service-safe. Its LICENSE.txt §3.3 reads "only may be used ... non-commercially ... research or evaluation purposes only", and it also carries the sd-turbo licence.
  - So do not reuse Difix3D's `examples/gsplat/simple_trainer_difix3d.py` in the product. Re-implement the loop. Fixer's repo ships inference and training scripts but no progressive-reconstruction script.
- The method to copy, from the Difix3D+ paper (CVPR 2025):
  - Every 1.5k iterations, nudge training poses toward the target views, render them, fix them, add them to the training set, and repeat.
  - Reported gains: 3DGS on DL3DV 17.18 to 17.99 dB, FID 107 to 41. Nerfbusters 17.66 to 18.51 dB, FID 114 to 42. LPIPS about -0.1.
  - The gain is mostly perceptual and on extrapolated views.
  - For us, the targets should be standing-height viewpoints on a grid over the floor (from LiDAR or RoomPlan), expanding outward from the walk path.
  - Down-weight the generated views and keep LiDAR depth on real views only, so the hallucinations do not override measured geometry.
- Caveat: Fixer was trained on an internal NVIDIA driving dataset, so indoor quality is unproven, and the card says it is "not guaranteed to fix 100%" of artifacts.
  - Fix: fine-tune it with the Apache training script on pairs made from our own captures, using the Difix recipes. "Sparse reconstruction" trains on every nth frame and pairs renders with held-out frames; "underfitting" trains for 25-75% of the schedule.
  - Avoid DL3DV and ScanNet++ for this training; their terms are non-commercial/research.
- Cost: inference is cents per room. Training gets about 3-6x longer than today's 33 s, so still well under $1 per room on an H100 (my estimate).
- Sources:
  - https://huggingface.co/nvidia/Fixer
  - https://github.com/nv-tlabs/Fixer
  - https://arxiv.org/abs/2503.01774
  - https://github.com/nv-tlabs/Difix3D (LICENSE.txt)

3. Graceful degradation in the product itself. Cheap, and it removes failure modes rather than improving dB.
a) Coverage-aware viewer:
  - Compute a per-position confidence: distance to the nearest capture camera plus viewing-angle overlap.
  - Spawn and default the call camera inside high-confidence space, with soft limits there (the walk zone exists but is off unless `?zone=1`).
  - Use Fixer-repaired zones as the "medium" ring.
b) Honest evaluation:
  - Hold out contiguous trajectory segments, not every 8th frame. The diagnosis found eval frames a median 3.8 cm from a training camera, so the score never sees extrapolation.
  - Add a sharpness metric.
  - Gate releases on this score.
c) Orientation-agnostic ingest:
  - Derive image "up" from ARKit gravity or pose, or from MapAnything's poses for plain video, and rotate frames before training. This fixes the sideways-stored Stray frames automatically.
  - Also check for iPhone HDR (HLG / Dolby Vision) camera-app video being decoded to SDR without tone mapping. This is my own engineering note, not from a cited source.
d) Redundant pose solutions with automatic selection. See item 4.
e) Restore the dense LiDAR seeding, which fell from 4000 to 718 points per photo, and fix the cap_max / seed-count interplay the diagnosis found (item 6 there). Indoor textureless walls depend on dense init.
- Licence: our own code. Cost: $0.

4. Poses: keep ARKit, refine without triangulation, and use permissive fallbacks for video without ARKit.
- ARKit VIO is the most blur- and low-texture-robust source we have, and it is metric. Turn on gsplat's built-in `pose_opt` (already in 1.5.3; never enabled here). Cost: $0. Gain: unmeasured, likely small to moderate.
- Do NOT run COLMAP triangulation plus prior-seeded bundle adjustment on ARKit poses.
  - Kyriazis (arXiv 2608.21008, Aug 2026): on 15 ScanNet++ iPhone rooms it "degraded an accurate ARKit prior in every case".
  - His triangulation-free BA reached 0.57 deg from about 0.55-0.74 deg priors with 0 failures in 330 perturbed runs, recovered 425/425 runs up to 16 deg / 80 mm, and takes about 10 s on a CPU.
  - Code is "will be released under Apache-2.0" and not yet available. Worth watching.
- Fallback chain for plain video (no ARKit), all commercial-safe:
  - (a) MapAnything `facebook/map-anything-apache` (Apache-2.0, already used in video.py). It accepts intrinsics, poses and depth as optional inputs and reaches up to 2000 views on 140 GB in memory-efficient mode. The CC-BY-NC checkpoint is said to perform better, but no numbers are given.
  - (b) VGGT-1B-Commercial. Commercial use is allowed except military, under the VGGT License v1 of Jul 29, 2025, with gated approval. `demo_colmap.py --use_ba` adds BA. A May 2026 memory fix gives about 2-3x more frames.
  - (c) VGGT-SLAM (BSD-2 code, plus the VGGT weights) for long walks.
  - (d) COLMAP 4.0's integrated global mapper (GLOMAP) as the classical cross-check.
  - Run two of them and keep the one with the best photometric/reprojection consistency. Redundancy is what makes this "never fail".
- Not service-safe:
  - MASt3R, MASt3R-SfM and MASt3R-SLAM: CC BY-NC-SA.
  - VGGT-X: CC BY-NC, LICENSE checked.
  - Pi3 / Pi3X weights: CC BY-NC (code is BSD).
  - Depth Anything 3 Giant/Large: CC BY-NC. DA3-Base/Small and DA3-Metric/Mono-Large are Apache-2.0.
- Sources:
  - https://arxiv.org/abs/2608.21008
  - https://github.com/facebookresearch/map-anything
  - https://huggingface.co/facebook/map-anything-apache
  - https://github.com/facebookresearch/vggt
  - https://huggingface.co/facebook/VGGT-1B-Commercial
  - https://github.com/MIT-SPARK/VGGT-SLAM
  - https://colmap.github.io/changelog.html
  - https://github.com/naver/mast3r
  - https://github.com/rmurai0610/MASt3R-SLAM
  - https://github.com/Linketic/VGGT-X
  - https://github.com/yyfz/Pi3
  - https://raw.githubusercontent.com/ByteDance-Seed/Depth-Anything-3/main/README.md

5. Regularise geometry with priors already in the repo, by default (counters "training longer overfits").
- Enable `dense_depth` (PromptDA) and DN-Splatter `normal_lambda` by default for LiDAR captures. Both exist but are off.
  - PromptDA code and checkpoints are tagged Apache-2.0 (https://huggingface.co/depth-anything/prompt-depth-anything-vits, https://promptda.github.io).
- For phones without LiDAR: MapAnything depth or DA3METRIC-LARGE (Apache).
- Cost: $0. Expected gain: fewer floaters and less ghosting (cause 3 in the diagnosis). Not measured here.

6. Motion blur and rolling shutter, modelled in the trainer. Moderate value; blur is secondary per the diagnosis.
- Frame picking already exists. Two upgrades:
  - (a) Soft per-frame weights instead of hard drops, which avoids coverage holes. ClearGS, arXiv 2609.31509 (2026): no code, gains given only qualitatively.
  - (b) A blur image-formation model: render K sharp images along poses interpolated across the exposure and average them, as in BAD-Gaussians (Apache-2.0, nerfstudio) and "Gaussian Splatting on the Move" (Apache-2.0, built on old forks of gsplat 0.1.11 / nerfstudio 1.1). Re-implement it in our patched trainer using ARKit poses at 60 Hz; the exposure time can come from our app's ARCamera.exposureDuration.
- Rolling shutter is already supported in gsplat 1.5.3 through 3DGUT (`with_ut`; `rasterization(rolling_shutter=..., viewmats_rs=...)`), with the second viewmat taken from the neighbouring ARKit pose.
- Cost: about K x render per step, still minutes and under $1. Gain: fine-scale sharpness; small in dB at walking speed, larger on fast pans.
- Sources:
  - https://github.com/WU-CVGL/BAD-Gaussians
  - https://github.com/SpectacularAI/3dgs-deblur
  - https://arxiv.org/abs/2403.13327
  - https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/docs/3dgut.md
  - https://arxiv.org/abs/2609.31509

7. Transients (people walking through, pets, the person filming showing up in mirrors).
- SpotLessSplats: Apache-2.0 and built on gsplat. Its RobustFilter (`--loss_type robust --no-semantics`) is cheap; the semantic version needs Stable Diffusion features.
- Gate it on residual-based transient detection, because a robust loss can also discard under-observed real content.
- WildGaussians' own code is MIT but builds on 3DGS and Mip-Splatting (Inria non-commercial), so it cannot be hosted.
- Gain: zero on clean captures, large on busy ones. Cost: about $0.
- Sources:
  - https://github.com/lilygoli/SpotLessSplats
  - https://arxiv.org/abs/2406.20055
  - https://github.com/jkulhanek/wild-gaussians

8. Heavier video-diffusion completion. Low priority: costly and risky for consistency.
- GEN3C: Apache-2.0 code, NVIDIA Open Model License weights. A 7B Cosmos-based model with about 43 GB peak memory when offloaded; tested on H100/A100. It accepts multi-view input with depth and poses, and can generate whole camera paths into unseen space. My estimate is about $1-3 per room.
- Lyra (from the same group) distils GEN3C into 3DGS.
- Not usable:
  - GSFixer (CogVideoX-5B based): its repo has no LICENSE file, so all rights are reserved.
  - Stable Virtual Camera: Stability non-commercial licence.
- Sources:
  - https://github.com/nv-tlabs/GEN3C
  - https://research.nvidia.com/labs/toronto-ai/lyra
  - https://github.com/gvclab/GSFixer
  - https://huggingface.co/stabilityai/stable-virtual-camera

======================================================================
SUGGESTED "UNFAILABLE" PIPELINE (in order)
======================================================================
1. Ingest any video:
   - Auto-orient by gravity.
   - Fix the colour space (HDR to SDR).
   - Score sharpness at full resolution and weight frames softly.
2. Poses:
   - ARKit plus in-trainer `pose_opt`; later, triangulation-free BA once the code is released.
   - Without ARKit: MapAnything-apache and VGGT-Commercial in parallel, keeping the more consistent result.
   - COLMAP 4 global mapper as a last resort.
3. Train:
   - gsplat with PPISP instead of the bilateral grid.
   - LiDAR / dense-depth / normal priors and dense seeding.
   - Robust loss when transients are detected.
   - Optional blur and rolling-shutter models.
4. Repair: a progressive Fixer loop (fine-tuned on our own indoor pairs) toward a standing-height viewpoint grid.
5. Export:
   - Bake canonical exposure and colour into SH; CRF by distillation or a Spark post-pass; no vignetting.
6. QA gate and viewer:
   - Segment-held-out PSNR plus a sharpness metric.
   - A confidence map drives the spawn point and soft zone in the call.
   - If a stage fails, the previous stage's output still ships (instant LiDAR room, then splat, then splat with repair) instead of an error.

Caveats:
- Cost figures marked as estimates are mine and were not measured.
- Fixer's indoor quality and the Apache MapAnything checkpoint's quality gap are unverified.
- PPISP in gsplat is on main, not on PyPI.


## research: products
RESEARCH REPORT: how shipping phone-video-to-splat/3D products deal with "people film however they want" (web research, 2026-11-25). No repo files were edited and nothing paid was run.

BOTTOM LINE
- None of these products claims it "never fails". Each one uses some mix of: (a) live coaching while filming, (b) automatic frame selection on the server or device, (c) hard input caps, and (d) a generic "Failed" status, often with a credit refund or advice to rescan. No product repairs bad coverage after the fact, except with generative fill, which is limited by how good the first reconstruction is.
- The strongest products control the capture itself rather than accepting arbitrary video. Apple Object Capture and RoomPlan, Luma Guided Capture, Scaniverse and Teleport all use their own capture app. Products that accept arbitrary uploaded video (Polycam Web, KIRI, Postshot, Teleport 2.0, vid2scene) rely on keyframe selection plus SfM, and simply fail or give poor quality on bad input.
- Realistic quality for casual iPhone indoor video, scored against sharp held-out views (ScanNet++ iPhone track): about 14-19 dB PSNR raw and about 19-22.7 dB after colour correction. Training on DSLR images reaches about 24-25 dB. Your 3-5 dB held-out colour error and per-frame exposure problem match the gap between the raw and colour-corrected columns.

PER PRODUCT (concrete facts)

Apple Object Capture (iOS 17+, RealityKit ObjectCaptureSession)
- Automatic: "our system will automatically select image shots with good sharpness, clarity, and exposure, and collect LiDAR points from various view angles." Capture is automatic (haptic tap and tone per shot). A "capture dial" plus a point cloud show where coverage is still missing. There are three recommended passes. Capture stops at the device's reconstruction limit. LiDAR is used to reconstruct low-texture objects. https://developer.apple.com/videos/play/wwdc2023/10191/
- Feedback enum (the exact failure and warning set): environmentLowLight ("Auto-capture still proceeds but reconstruction quality may suffer"), environmentTooDark ("Auto-capture will stop"), movingTooFast ("capturing may be paused to ensure quality"), objectTooClose, objectTooFar, outOfFieldOfView (auto-capture will not operate), objectNotDetected (falls back to a manual box), objectNotFlippable (low-texture), overCapturing (extra shots are unused on device). https://developer.apple.com/documentation/realitykit/objectcapturesession/feedback-swift.enum.md
- Area mode (WWDC24): no bounding box. The reticle acts like a brush and disappears when too close or too far. There is a camera-pose visualization for review. Areas over 6 ft may lose quality unless processed on a Mac. https://developer.apple.com/videos/play/wwdc2024/10107/

Apple RoomPlan (iOS 16+)
- Live coaching via RoomCaptureSession.Instruction: normal, moveCloseToWall, moveAwayFromWall, turnOnLight, slowDown, lowTexture ("doesn't detect distinguishable room features"). The app shows these as text during the scan. https://developer.apple.com/documentation/roomplan/roomcapturesession/instruction.md

Polycam (cloud, uses gsplat)
- Automatic: "Smart Keyframing" automatically picks the best frames from uploaded footage and "filters out blurry or redundant frames". It is described as especially useful for video. Processing "will work to minimize the presence of floaters".
- Still required from the user: footage with minimal motion blur and no shallow depth of field ("can cause floaters"), and mp4 format. Image sets are limited to 20-200 images on the basic path and up to 1,000 splat images on Pro and above. Video length caps depend on plan (Free 3 min, Basic 5 min, Business 30 min).
- Failure handling is generic: check connectivity, keep battery above 20%, "duplicate the capture" to restart processing past corruption, and use the FAST preset or simplification after memory crashes. No splat-specific error taxonomy is published.
- https://poly.cam/splat ; https://learn.poly.cam/hc/en-us/articles/30549121659412 (403 to fetchers; contents via search snippets) ; https://learn.poly.cam/hc/en-us/articles/37030074942612 ; https://learn.poly.cam/hc/en-us/articles/27426630160148

Luma AI
- Guided Capture shows a path line and a reticle that turns yellow with "move away" when too close, and a coverage progress score toward 100. Video upload also exists. https://www.tomsguide.com/how-to/how-to-take-3d-captures-smartphone-luma-ai
- Status: capture uploads were reported "temporarily paused" in Nov 2024, after Dream Machine load pushed queues from about 15 minutes to a week. Flythroughs and Genie shut down on 1 Jan 2026. The iOS app is still listed (v1.3.14, Jan 2026), but the Android app has been removed. In practice Luma is no longer a live reference for casual capture. https://radiancefields.com/radiance-field-uploads-suspended-on-luma-ai ; https://radiancefields.com/luma-ai-to-sunset-flythroughs-on-january-1-2026

Scaniverse (Niantic Spatial)
- On-device splat training in about 1 minute. Business accounts got cloud processing in iOS 5.2.0 (Feb 2026).
- Live feedback is the live mesh itself: if it is "patchy", "wobbly or disappearing", the user should slow down, point back at already-scanned features and increase overlap.
- v5.1.0 warns when a splat scan passes 180 s ("risk of processing failure"). The docs say up to 5 minutes. v5.1.0 also added sky segmentation and better initial pose.
- Indoor guidance: walk the perimeter facing inward, in 3 passes (level, tilted up, tilted down), then detail passes. Avoid blank or repetitive walls, mirrors and moving people.
- Failures: users report a generic "Failed" status with no error code. The fix offered is to chunk or shorten scans; one 8K 360 video only processed after being trimmed to 16 s. Raw data stays on device, so support cannot reprocess it.
- https://nianticspatial.com/docs/scaniverse/techniques/ ; https://www.nianticspatial.com/capture/scaniverse-release-notes ; https://community.nianticspatial.com/t/processing-fails-on-a-couple-of-my-splats/5734 ; https://community.nianticspatial.com/t/scan-render-failure-generic-failed-status-for-8k-360-hevc-video-gopro-max-2/5758 ; https://nianticspatial.com/docs/scaniverse/360camera/

KIRI Engine (cloud)
- Uploaded video must be at most 1920x1080 and at most 3 minutes (it was 2 minutes before v3.10; footage beyond the cap is cut silently). Video and photo sets both work since v3.13. There is a toggle for 3DGS-to-mesh.
- User requirements: a static subject, an inward orbit at 3 heights, moving "slower than feels natural". Motion blur is named as the most common cause of bad splats. No documented failure codes or auto-reject.
- https://docs.kiriengine.app/category/3dgs-scan/ ; https://www.kiriengine.app/blog/how-to-capture-3d-gaussian-splats-kiri-engine ; https://radiancefields.com/kiri-engine-now-allows-local-upload-for-3dgs

Teleport by Varjo (cloud, aimed at rooms and spaces; closest to your use case)
- The capture app uses the 0.5x ultrawide lens by default (for overlap and sharpness) and has a Pause button for relocating or when something moves.
- Recommended 250-1,000 frames (up to 2,000 for large spaces), and "more frames are not always better". The most common cause of bad results is the capture path. Rotating in place is named as giving weak 3D.
- Room recipe: walk sideways along the walls with your back near the wall, camera aimed at the centre, at chest-to-shoulder height. Then detail passes, staying at least 30 cm from surfaces. Capture both sides of doorways.
- Processing takes 30 min to 24 h. If reconstruction fails, the scan allowance is refunded (+1). Failed uploads are kept on the phone (5 GB free needed) for manual recovery by email.
- Teleport 2.0 (Apr 2025) accepts any camera (drone, DSLR). It needs a light source; dark-to-bright transitions are named as problems. It originally required iPhone 12 Pro or later and iOS 17+.
- https://get.teleport.varjo.com/blog/how-to-create-best-quality-3d-scans-of-your-environment-with-teleport ; https://teleport.varjo.com/help/en_US/capturing ; https://teleport.varjo.com/help/en_US/capturing-faq- ; https://teleport.varjo.com/help/en_US/capture-recovery- ; https://www.uploadvr.com/varjo-teleport-2-0-best-in-class/ ; https://www.digitalcameraworld.com/news/could-ai-spell-an-early-death-for-vr-cameras-new-iphone-app-tries-to-do-just-that

Postshot (Jawset, desktop)
- When given video or images, it runs automatic "Image Selection" plus camera tracking. When poses are imported it switches to "Use All Images" and skips tracking.
- v0.2 lowered the recommended maximum to 400 images to avoid long tracking jobs. Distorted images with Bundler poses give "likely unusable" results. No public detail on how frames are selected.
- https://jawset.com/docs/d/Postshot+User+Guide/Importing+Images ; https://jawset.com/docs/d/Postshot+User+Guide/Release+Notes/v0.2

Splatcatcher: I found nothing public on how it works.

Gracia: a processing service for multi-camera 4D capture through partner studios (3-5 day turnaround, $800-2,500 per minute). It is not a casual phone pipeline; for static scenes it takes stills made in Luma-type apps. https://radiancefields.com/platforms/gracia ; https://www.uploadvr.com/gracia-ai-6dof-truly-volumetric-platform/

Spline: import only (.ply, 300-480 MB). It points users to Polycam or Luma for capture. https://docs.spline.design/e17b7c105ef0433f8c5d2b39d512614e

Open-source comparison point: vid2scene (video -> GLOMAP SfM, or VGGT opt-in, which it says is "noticeably worse" -> gsplat -> web viewer). Its README documents no frame selection, fallback or quality-check logic. https://github.com/samuelm2/vid2scene/

Generic guidance shared across vendors (what they still ask of the user): slow, steady movement; 70-80% overlap; several heights; locked exposure; avoid mirrors and blank walls. PlayCanvas recommends 4K at 60 fps or more, extracting every 2nd-5th frame, and the "Sharp Frames" tool for automatic selection. https://developer.playcanvas.com/user-manual/gaussian-splatting/creating/taking-photos/ ; https://help.sketchup.com/en/gaussian-splats-best-practices

QUALITY ON CASUAL INDOOR PHONE VIDEO (hard numbers)
- ScanNet++ iPhone NVS benchmark: trained on iPhone video at 1920x1440 and 60 fps, tested against DSLR images, 12 scenes. Scores are PSNR raw / PSNR colour-corrected:
  - Vanilla 3DGS: 14.3 / 19.2
  - Splatfacto: 14.1 / 19.4
  - 3DGS with exposure compensation: 15.2 / 20.0
  - Best entry, DVSM-480p: 19.2 / 22.7
  - The paper says iPhone data is harder "due to the motion blur, varying brightness, and less field-of-view". The DSLR-trained Nerfacto reaches 25.0 dB.
  - https://scannetpp.mlsg.cit.tum.de/scannetpp/benchmark/nvs_iphone ; https://arxiv.org/pdf/2308.11417
- Difix3D+: average gain of more than 1 dB PSNR and about 2x better FID. The authors say performance is "inherently limited by the quality of the initial 3D reconstruction". https://arxiv.org/html/2503.01774v1
- NVIDIA Fixer (the service-safe option):
  - Works at 576x1024, 0.6B parameters, based on Cosmos. Takes 26.5 ms per frame on an H100. Licence is the NVIDIA Open Model License (commercial use allowed).
  - The model card publishes no metric numbers. It was trained on an internal autonomous-driving dataset, so it is unproven on indoor rooms; benchmark it on your held-out frames before relying on it.
  - It runs offline (fix pseudo-views and distil them back into the splat) or online (enhance renders at view time).
  - https://huggingface.co/nvidia/Fixer/blob/main/README.md

WHAT THIS MEANS FOR AN "ALWAYS WORKS" PIPELINE (my synthesis, mapped to your known issues)
1. Sideways frames: no product asks users about orientation. Derive it from the video rotation metadata or the ARKit/IMU gravity vector, then check it with a horizon or gravity test before SfM. Never assume a layout.
2. Blur (median 2.5 px, 28% of frames at 4 px or more): this is what Polycam Smart Keyframing and Apple shot selection do. Pick the sharpest frame in each time or pose window under a coverage constraint, rather than a fixed stride. For the frames you must keep, Spectacular AI's blur and rolling-shutter-aware 3DGS uses VIO velocities, which Stray/ARKit provides. https://arxiv.org/abs/2403.13327 (code: github.com/SpectacularAI/3dgs-deblur)
3. Exposure (3-5 dB held-out error, bilateral grid dropped at export): NVIDIA PPISP (Apache-2.0) models per-frame exposure and white balance plus per-camera vignetting and response curve. Its controller predicts these for novel views ("without requiring per-frame parameters at inference time", frame_idx=-1). That is exactly what the dropped bilateral grid lacks. https://arxiv.org/abs/2601.18336 ; https://github.com/nv-tlabs/ppisp . Colour correction alone is worth about 4-6 dB on ScanNet++.
4. Coverage (1.8 x 1.6 m walked in a 4.2 x 4.1 m room): every serious product fixes this while filming, not afterwards: Apple's capture dial, RoomPlan instructions, Luma's progress score, Scaniverse's live mesh, Teleport's walk-the-walls recipe. After the fact, the only option is generative repair (Fixer/Difix), which is bounded by the base reconstruction.
   - "Never fails" in practice therefore means always returning something, plus a per-region confidence or coverage map, plus honest masking or fill of unseen areas.
   - Add optional live coaching in your capture flow, using the RoomPlan or ObjectCapture message set as the template: slowDown, turnOnLight, lowTexture, move closer or further, and "point back at what you already scanned".
5. Length and size caps are used everywhere as a failure guard: Scaniverse 180 s, KIRI 3 min at 1080p, Postshot 400 images, Teleport 250-1,000 frames. Cap or subsample internally instead of failing.
6. Failure UX norm: a generic "Failed" status plus refunding the credit (Teleport), "duplicate to reprocess" (Polycam), and "split into chunks" (Scaniverse). No vendor publishes a detailed reason taxonomy. Apple's Feedback enum is the best public model for user-facing reasons.
