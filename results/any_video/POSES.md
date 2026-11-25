# Camera poses for every frame of a casual iPhone video: what is best (2026-11-25)

Research pass (feed-forward models, casual-video SLAM, classical and hybrid SfM), a recommendation, and a skeptical review. Nothing here has been run yet. The raw research is in `poses_inputs.md`.

## Getting accurate poses and focal from a casual iPhone room video

**Short answer:** no single model does this well enough. What leads the published phone-video results is a pipeline: fix the focal length from a calibration, get dense first poses that cope with plain walls, run one global bundle adjustment (BA) with loop closures, then take metric scale and gravity from separate models. VidMap is the best published example of this. Its code is Apache, but its default depth model is non-commercial and has to be swapped out. The MapAnything one-pass result (focal 35% short, cameras up to 0.8 m off) is about what the benchmarks predict, so don't try to tune it.

### 1. Recommended pipeline

1. **Pick frames.**
   - Convert HLG to SDR first.
   - Score sharpness (Laplacian) and use parallax keyframing to choose 600–1,200 frames.
   - Detect lens switches from metadata or a jump in focal length. Treat each lens as its own camera.
2. **Fix the focal length per lens; don't estimate it.**
   - For each iPhone model, lens and stabilisation mode, calibrate once against the Stray capture. The stabilisation crop is the likely cause of the 35% focal shortfall.
   - Read `CameraFocalLength35mmEquivalent` from the file's metadata and use it to check the calibration.
   - Use GeoCalib (Apache code, CC BY weights) or MoGe-2 (MIT) only as a fallback and cross-check.
   - Why: every report shows uncalibrated focal as the weakest link. VidMap's calibrated runs beat its uncalibrated ones, and MegaSaM's focal error on TUM is 10–12°.
3. **Get first poses with a dense method.** Use ViPE `pose_only` (DROID-style dense BA, MoGe-2 depth, our intrinsics passed in), or VGGT-1B-Commercial run in chunks.
   - Why: dense flow and feed-forward models don't need keypoints, so they survive white walls, where RealityScan fragmented.
4. **Run one global BA with loop closures.**
   - Use COLMAP 4.2.1 or newer (BSD), started from step 3's poses.
   - Features: ALIKED+LightGlue or LoMa. Matching: sequential, plus MegaLoc retrieval for loops.
   - Focal: one shared camera per lens, held near the calibrated value.
   - Why: BA is what takes errors from tens of cm down to cm. VidMap is this recipe and scores 88.5 on LaMAR iPhone videos, against 76.6 for ViPE alone.
5. **Metric scale:** take the robust median ratio against MoGe-2 or DA3METRIC-LARGE (Apache) depth.
6. **Gravity:** average GeoCalib's per-frame gravity, then check it against the floor-plane normal.
7. **Refine in gsplat:** turn on `pose_opt` as a final polish only. It fixes centimetres, not decimetres.
8. **Fail loudly.** Stop the job and flag it when any of these is true:
   - fewer than about 95% of frames registered, or more than one model component;
   - high reprojection error;
   - focal more than 3–5% from the calibration;
   - scale estimates disagree by more than 5%;
   - gravity estimates disagree by more than 3°;
   - ViPE and the BA result disagree by more than a few cm in ATE after Sim3 alignment.

### 2. Pipelines to benchmark, ranked

Cost assumes Modal's H100 list price of about $4/h; check current pricing. All runtimes are my estimates for 1,000 frames; none are measured.

**A. VidMap with licence-safe swaps**
- Swaps needed:
  - Replace the default depth model (DA3-Large or DA3NESTED-GIANT, both CC BY-NC) with DA3METRIC-LARGE or MoGe-2. This is a code change and its effect on accuracy is untested.
  - Pass our calibrated focal.
  - FlexGlue and RoMaV2 embed DINOv3, whose licence bars military use and adds trade terms; legal must review it. Otherwise replace them with LoMa or ALIKED+LightGlue.
- Runtime: about 5–15 min (the paper only says "GLOMAP speed"). About $0.30–1.00 per video.

**B. ViPE `pose_only` (MoGe-2, our intrinsics), then COLMAP 4.2 BA**
- Avoid: the `default` pipeline (UniDepth is CC BY-NC), Depth Pro 2, the MetricTrail weights, and the vendored Segment-and-Track-Anything module (AGPL).
- Runtime: about 3–6 min for ViPE plus 3–5 min for COLMAP. About $0.40–0.80 per video.
- This is the easiest to build and gives us the cross-check for free.

**C. VGGT-1B-Commercial in chunks of about 100 frames with 50% overlap, joined by a Sim3 pose graph, then COLMAP BA**
- Use the VGGT-Long or VGGT-SLAM code with the commercial weights only.
- Avoid: the original VGGT-1B weights, the SALAD checkpoint (GPL), and the VGGSfM tracker and SuperPoint in VGGT's own BA demo.
- The VGGT-SLAM 2.0 authors report VGGT failing on white walls.
- Runtime: about 2 + 4 min. About $0.40 per video.

**D. COLMAP 4.2 global mapper on its own (ALIKED+LightGlue, known focal)**
- Fully BSD/Apache. This is the textured-scene baseline; expect it to break into pieces on white walls.
- Runtime: about 4–6 min. About $0.30 per video, mostly CPU.

**Controls**
- Today's MapAnything-apache pipeline, plus one rerun with our calibrated focal fed in. That rerun is cheap and shows how much of the error was focal alone.
- I'd skip research-only references (MASt3R-SLAM, DA3-Giant streaming). Even internal use is arguably commercial.

### 3. Benchmark plan using the LiDAR truth

1. **Ground truth.** If the Camera-app frames are not themselves posed in the Stray frame, build pseudo-ground-truth first:
   - Localise about 150 sharp frames against the Stray RGB-D using ALIKED+LightGlue and PnP-RANSAC on the LiDAR depth, refining one shared focal. That refined focal is the "LiDAR-fit focal".
   - Keep only frames with a high inlier count and under 1 px reprojection error, and report how noisy this ground truth is.
   - Take ground-truth gravity from ARKit.
2. **Metrics for each pipeline:**
   - ATE RMSE after Sim3 alignment, and after SE3 alignment with the pipeline's own metric scale (this tests metric accuracy);
   - scale error (%);
   - RPE (translation and rotation per metre);
   - focal error against the metadata and against the LiDAR-fit focal;
   - gravity error (°);
   - % of frames registered and number of components;
   - wall time, peak VRAM and $.
3. **Splat check:** PSNR, SSIM and LPIPS on the 29 shared eval frames. Same gsplat settings for every pipeline. Register the eval frames but keep them out of training. Report PSNR both raw and after test-time pose alignment of the eval frames, so blurry eval poses don't hide good geometry.
4. **Robustness:** run all pipelines on 2–3 more videos (portrait, a lens switch, fast pans). There's no ground truth for these, so judge them on the fail-loud checks and on cross-method agreement. Choosing on one video risks overfitting.
5. **Cost:** about 6 pipelines × 4 videos × (about 10 min poses + training), plus building the ground truth. That comes to roughly $30–60 of H100. The real cost is engineering time, mainly the VidMap depth swap.

### 4. Still uncertain

- **Licence wording:**
  - VGGT's code licence is a "Research Materials" grant (wording is ambiguous).
  - The DINOv3 licence is unreviewed.
  - Not yet confirmed that ViPE's AGPL module can be switched off fully.
- **The two research reports conflict:**
  - Pi3 weights: one says CC BY-NC, the other says BSD-2. Check the Hugging Face card before trying GLUEMAP+Pi3.
  - DA3-LARGE-1.1: the Hugging Face tag says Apache but the README says non-commercial.
- **Unpublished and unmeasured:**
  - No results exist on HLG, stabilised, 60 fps iPhone video.
  - Every H100 runtime above is an estimate.
  - We don't know how much VidMap loses with an Apache depth model.
- **Not modelled by any of these:**
  - Electronic stabilisation warps each frame, so it isn't a true pinhole camera.
  - None of the pipelines model rolling shutter, which can limit cm accuracy on fast pans.
- **Splat quality:** good poses may not close the whole gap between 9.9 dB and the LiDAR capture's 20.2 dB. Exposure/HDR and blur also cost dB, so measure them separately.

---

## Review

## Verdict: the overall direction is sound, but the plan has six concrete problems

**1. "VidMap is this recipe" is wrong.**
- VidMap is dense tracks plus FlexGlue/MegaLoc loops plus COLMAP global positioning. It is not "ViPE init, then COLMAP BA".
- Its 88.5 vs 76.6 on LaMAR was measured with its non-commercial DA3 depth. It does not support pipeline B.
- Fix: cite it only as evidence for A, and call B unproven.

**2. Step 4 won't work as written.**
- COLMAP's global mapper does its own rotation averaging and positioning. It cannot be started from ViPE poses.
- Fix: write the ViPE poses into a sparse model, then run `point_triangulator` and `bundle_adjuster`. Or use pyceres BA with the poses as priors.
- Also, ALIKED+LightGlue BA puts back the keypoint weakness on white walls. Frames with too few matches should keep their ViPE pose and be flagged, not silently dropped.

**3. Per-device calibration doesn't scale.**
- There is one Stray phone. Customers' iPhone models and stabilisation modes will mostly have no calibration.
- Fix: run every pipeline twice, once with the calibrated focal and once with metadata/estimated focal. The second run is what ships.
- The "focal within 3–5% of calibration" gate does nothing when focal is held at the calibrated value. Instead, gate on the gap in reprojection error between focal-free and focal-fixed runs.

**4. Strong candidates are missing.**
- DROID-SLAM or DPV-SLAM++ with known K: BSD-3/MIT, under 1 min per 1,000 frames, a cheap independent cross-check.
- GLUEMAP with the VGGT-1B-Commercial or MapAnything-apache backbone: the best indoor multi-room result in the research (SMERF AUC@5 54.9 vs 18.0). Needs `skip_doppelgangers` and the GPL/NC trackers removed.
- Map-Long with known intrinsics.

**5. The VidMap swap can't be measured as planned.** Without one run of VidMap's default NC depth model as an upper bound, "how much the Apache swap costs" stays unknown. Fix: get legal sign-off for one internal eval run, or drop this as a stated question.

**6. Ground-truth bias.**
- PnP on low-res Stray LiDAR depth will fail on the white-wall frames, which are the frames that matter. Pseudo-GT will cover mostly textured frames, where every method looks good.
- Fix: report GT coverage separately for textured and plain-wall segments. Add dense depth-to-LiDAR-mesh refinement for frames with few inliers. State the ARKit drift floor (several cm) so cm-level ATE differences below it are not over-read.
- Stabilisation changes the crop per frame, so check whether one shared focal fits all frames by looking at PnP residuals against the frame index.

**Minor fixes**
- MegaSaM's focal error is 10.5–12.6°, not "10–12°".
- VidMap's first run downloads about 9 GB. Cache it in a Modal volume or it inflates runtime and cost.
- Budget gsplat training at about 15–30 min per run. With 6 pipelines × 4 videos, $30–60 is plausible only at the low end. Only 1 video has GT, so the other 3 add cost without accuracy data.
- The Pi3 and DA3-LARGE-1.1 licence conflicts are correctly flagged. Check the Hugging Face cards before including GLUEMAP+Pi3.
