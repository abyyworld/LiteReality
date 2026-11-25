# Research inputs to POSES.md

## feedforward
## Feed-forward multi-view 3D models for long iPhone room videos: research report (checked November 2026)

**Short answer:** for a commercial service, the best setup today is **VGGT-1B-Commercial run in overlapping chunks**. Join the chunks with a Sim(3) pose graph, then run one global bundle adjustment with COLMAP or GLOMAP. Neither VGGT nor any other model here gives metric scale or gravity in a commercial-safe way, so get those from separate Apache or MIT models.

MapAnything is the wrong model for poses. On the Depth Anything 3 (DA3) paper's benchmark its pose accuracy is far below VGGT. The Apache checkpoint you use is also weaker than the non-commercial one, and 192 frames in one pass is well beyond what it was trained on. Each point below is cited.

### 1. Model by model

**VGGT (Meta)**
- **Licence:**
  - Code: the README says (May 2025) all code moved to a "commercial-use-friendly license", excluding military use.
  - LICENSE.txt itself is the "VGGT License v1" (29 Jul 2025). It is a royalty-free, non-transferable "Research Materials" grant with an acceptable-use policy (AUP); it neither allows nor forbids commercial use explicitly. Have a lawyer read it.
  - Original `facebook/VGGT-1B` weights: non-commercial.
  - `facebook/VGGT-1B-Commercial`: gated (automatic approval form), tagged `vggt-aup-license`, "licensed for commercial use, with the exception of military applications".
  - Accuracy: the README reports Co3D AUC@30 of 90.37 for the commercial checkpoint against 89.98 for the original.
  - Links: https://github.com/facebookresearch/vggt, https://huggingface.co/facebook/VGGT-1B-Commercial
- **Frames per pass:** no hard limit. The README says "hundreds of views". A memory fix on 15 May 2026 runs "roughly 2-3x more input frames" in the same GPU memory.
  - Earlier README versions had an H100 table of about 40 GB at 200 frames. That table is gone now; I did not re-verify it.
  - Global attention is quadratic. VGG-T3 (NVIDIA) measured VGGT at more than 11 min for 1,000 images and about 27 min for 2,000 (https://arxiv.org/html/2602.23361v1), so chunk long videos.
- **Accuracy:**
  - DA3 paper Table 2 (AUC3 / AUC30), about 100 frames per scene (https://arxiv.org/html/2511.10647):
    - ScanNet++: 62.6 / 95.1
    - 7-Scenes: 23.9 / 85.0
    - ETH3D: 26.3 / 80.8
    - HiRoom: 49.1 / 88.0
  - Pi3 paper (https://arxiv.org/html/2507.13347), ATE / RPE-t / RPE-r:
    - ScanNet: 0.035 / 0.015 / 0.382
    - TUM-dynamics: 0.012 / 0.010 / 0.311
- **Outputs:** poses plus per-frame intrinsics (OpenCV convention). Scale is normalised, not metric. No gravity.
- **Commercial:** yes, with the commercial checkpoint.

**VGGT-Long (Deng et al., ICRA 2026)** — https://github.com/DengKaiCQ/VGGT-Long
- **Method:** splits the video into chunks, aligns them with Sim(3) (an SE(3) option was added 8 Oct 2025 for metric models), and closes loops.
  - Loop detection uses DBoW, a CPU-only place-recognition model, plus image retrieval through DPRetrieval.
  - Loop correction uses a Sim(3) pose-graph solver (Python or C++).
- **Back-ends:** refactored 12 Dec 2025 to support any model: VGGT, Pi3 ("Pi-Long"), MapAnything ("Map-Long").
- **Evaluation:** KITTI, Waymo and Virtual KITTI, plus a phone demo. No indoor numbers.
- **Disk:** about 50 GiB for 4,500 frames.
- **Licence:** code follows the VGGT licence. The README says to use VGGT-1B-Commercial for commercial use.

**DA3-Streaming (ByteDance, 11 Dec 2025)** — https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md
- Built on VGGT-Long's code, with GPU acceleration. Chunks of 30–120 frames, overlap of half a chunk, loop closure at a 0.85 similarity threshold.
- **Peak VRAM** on TUM at 504×378: 28.3 GB with 120-frame chunks, 18.7 GB with 30-frame chunks.
- **Speed:** about 8.5–10 FPS on an A100. On KITTI (11,373 frames) it took 22 min against 65 min for VGGT-Long.
- **TUM RGB-D ATE:** DA3-Streaming 0.087 m, Pi-Long 0.094, VGGT-Long 0.110. For reference, MASt3R-SLAM scores 0.060.
- **Licence problems:**
  - Its weight script downloads `DA3NESTED-GIANT-LARGE-1.1`, which is CC BY-NC 4.0.
  - It also downloads the SALAD checkpoint for loop closure; the SALAD repo is GPL-3.0 (https://github.com/serizba/salad).
  - **As shipped, it is not commercial-safe.**

**MapAnything (Meta/CMU, 3DV 2026)** — https://github.com/facebookresearch/map-anything, https://huggingface.co/facebook/map-anything-apache
- **Licence:** code Apache-2.0. `map-anything-apache` weights Apache-2.0 (latest release 20 Jan 2026). `map-anything` weights CC BY-NC 4.0.
- **Apache checkpoint is weaker:**
  - It is trained on 6 datasets: BlendedMVS, Mapillary, ScanNet++ v2, Spring, TartanAirV2, UnrealStereo4K.
  - The non-commercial model adds 7 more: ASE, DL3DV, Dynamic Replica, MegaDepth, MVS-Synth, PD4D, SAIL-VOS3D.
  - The paper supplement says Apache performance "drops", though it is "still competitive with the VGGT baseline" (https://arxiv.org/html/2509.13414v3).
- **Frames:**
  - Training used 2–24 views; evaluation went up to 100 views.
  - `memory_efficient_inference` allows "up to 2000 views on 140 GB".
  - There is no official chunking. Map-Long (through VGGT-Long) is the chunked route.
- **Accuracy:**
  - DA3 Table 2 (AUC3 / AUC30): ScanNet++ 20.2 / 84.1, 7-Scenes 12.6 / 79.7, ETH3D 19.2 / 77.4. That is far below VGGT, Pi3 and DA3 at the tight threshold. This matches your 0.8 m camera errors.
  - The MapAnything paper's own 2-view table beats VGGT, but those are 2-view results, not long multi-view ones.
- **Outputs:** metric scale (`metric_scaling_factor`). It accepts intrinsics and poses as inputs, so you can feed it known intrinsics. No gravity.

**Pi3 / Pi3X** — https://github.com/yyfz/Pi3
- **Licence:** code BSD-3. Weights for both Pi3 and Pi3X are **CC BY-NC 4.0, "Strictly Non-Commercial"**.
- **Accuracy:**
  - DA3 Table 2: ScanNet++ 50.7 / 92.1, 7-Scenes 25.5 / 86.3, ETH3D 35.2 / 87.3.
  - Pi3 paper ScanNet ATE 0.031, the best in that table.
  - Effectively insensitive to input frame order (Table 6).
  - 57 FPS on an A800.
- **Outputs:** Pi3X (28 Dec 2025) gives approximate metric scale and accepts pose, intrinsics and depth as conditioning.
- **Commercial:** not usable.

**Depth Anything 3 (ByteDance, Nov 2025)** — https://github.com/ByteDance-Seed/Depth-Anything-3, https://arxiv.org/abs/2511.10647
- **Licence:** code Apache-2.0.
  - CC BY-NC 4.0: GIANT, LARGE, NESTED-GIANT-LARGE (including the 1.1 versions).
  - Apache-2.0: BASE (0.12B), SMALL, METRIC-LARGE, MONO-LARGE.
- **Accuracy:** claims +44.3% pose accuracy over VGGT.
  - DA3-Giant: ScanNet++ 85.0 / 98.1, 7-Scenes 28.5 / 86.8, ETH3D 48.4 / 91.2. This is the strongest model listed, but non-commercial.
  - The Apache DA3-Base is much weaker: ScanNet++ 25.1 / 83.4, 7-Scenes 20.1 / 82.9. That is roughly MapAnything level and below VGGT.
- **Metric scale:**
  - DA3METRIC-LARGE (Apache) gives metric depth from a focal length: depth = focal × output / 300.
  - The Nested model outputs metres, but it is non-commercial.

**MoGe-2 (Microsoft)** — https://github.com/microsoft/MoGe, https://huggingface.co/Ruicheng/moge-2-vitl-normal
- **Licence:** code MIT (the bundled DINOv2 code is Apache-2.0). The `moge-2-vitl-normal` HF card is tagged MIT.
- **What it gives:** single-image metric point maps plus field of view.
- **Accuracy:** no FOV accuracy figures in the README.
- MoGe-3 was released 18 Aug 2026; I did not check its licence.
- **Commercial:** OK. Useful as a per-frame focal and metric-scale prior.

**CUT3R (CVPR 2025)** — https://github.com/CUT3R/CUT3R
- **Licence:** CC BY-NC-SA 4.0, and it is built on DUSt3R, which is also CC BY-NC-SA (https://github.com/naver/dust3r).
- **Method:** recurrent streaming. Memory grows linearly; checkpoint trained on 4–64 views.
- **Accuracy:** ScanNet ATE 0.094 (Pi3 paper).
- **Commercial:** no.

**TTT3R (ICLR 2026)** — https://github.com/Inception3D/TTT3R
- A state-update rule that helps CUT3R handle long sequences; uses CUT3R weights.
- The LICENSE file contains both CC BY-NC-SA and MIT text, and the weights are CUT3R's.
- **Commercial:** no.

**Fast3R (Meta)** — https://github.com/facebookresearch/fast3r
- **Licence:** "FAIR NC Research License" for code and models.
- "1000+ images in one forward pass", but poses are weak: ScanNet ATE 0.155; DA3 table ScanNet++ 17.9 / 72.5.
- **Commercial:** no.

**Spann3R, Point3R, StreamVGGT** — https://github.com/HengyiWang/spann3r, https://github.com/YkiWu/Point3R, https://github.com/wzzheng/StreamVGGT
- All three LICENSE files are **CC BY-NC-SA 4.0**. Spann3R and Point3R start from DUSt3R weights; StreamVGGT distils from VGGT-1B.
- **Commercial:** no.

**VGGT-Ω (Meta, CVPR 2026 oral)** — https://github.com/facebookresearch/vggt-omega, https://arxiv.org/abs/2605.15195
- **Licence:** LICENSE is the **FAIR Noncommercial Research License v1**. Weights are gated, and there is no commercial checkpoint.
- **Memory:** 1B-512 model on an A100 at 624×416: 6 GB for 1 frame, 43 GB for 500 frames.
- **Outputs:** scene-normalised scale, not metric.
- A "reproduction" checkpoint was released on 8 Sep 2026 over a "potential concern" with the original checkpoint. A secondary source reports possible benchmark contamination (https://learnopencv.com/vggt-vs-vggt-%CF%89-vggt-omega-a-complete-guide-to-feed-forward-3d-reconstruction/).
- **Commercial:** no.

**Other newer work**
- **VGGT-X** (https://github.com/Linketic/VGGT-X): scales VGGT to dense views, aligns globally with XFeat matches, does joint pose and 3DGS training, and exports COLMAP. LICENSE is **CC BY-NC 4.0**. Reuse the idea, not the code.
- **VGGT-SLAM 2.0** (MIT-SPARK, RSS 2026; https://github.com/MIT-SPARK/VGGT-SLAM): code BSD-2. It runs on whichever VGGT weights you supply (the README does not name a checkpoint), so use the commercial ones. SL(4) submap alignment and loop closure. No indoor numbers in the README.
- **VGG-T3** (NVIDIA; https://arxiv.org/html/2602.23361v1): scales linearly; 2,000 images in 48.5 s on 4 GPUs. Pose is worse than VGGT on ScanNet (ATE 0.070 vs 0.035). I found no code licence.
- **HiP3R** (https://arxiv.org/abs/2610.31635): a training-free hierarchical Sim(3) then SE(3) pose graph that sits on top of any chunked or streaming back-end. Code is announced at kkakkkka.github.io/HiP3R; licence not checked. Note the CC BY 4.0 on arXiv covers the paper only, not code.
- **HunyuanWorld-Mirror** (https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror): Tencent community licence that excludes the EU, UK and South Korea and needs a separate licence above 1M monthly active users. Risky for a hosted service.
- **GeoCalib** (https://github.com/cvg/GeoCalib): single-image gravity and focal estimation. Code Apache-2.0, weights CC BY 4.0, so commercial use is OK. Roll AUC@5° is about 90 on most of its benchmarks. FOV accuracy is weak (AUC@10° about 48–60).

**Gravity and robustness, across all models**
- **Gravity:** none of the models above outputs a gravity direction.
- **Blur and textureless walls:** no paper reports a dedicated blur or white-wall benchmark.
  - Feed-forward models do not depend on keypoints, so they do better than classical SfM on plain walls. Your RealityScan run registering 26% of frames is consistent with that.
  - Motion blur still hurts, so pick sharp frames before running anything.

### 2. Commercially usable options

- **Usable:**
  - VGGT-1B-Commercial (AUP; no military use)
  - MapAnything-apache (weaker variant)
  - DA3-BASE and DA3-SMALL (weak poses)
  - DA3METRIC-LARGE, DA3MONO-LARGE
  - MoGe-2 (MIT)
  - GeoCalib (Apache code, CC BY weights)
  - VGGT-Long and VGGT-SLAM code, used with the commercial VGGT weights
- **Not usable:**
  - Pi3 and Pi3X
  - DA3 Giant, Large and Nested
  - VGGT-Ω
  - CUT3R and TTT3R
  - Fast3R
  - Spann3R, Point3R, StreamVGGT
  - VGGT-X code
  - DUSt3R and MASt3R
  - SALAD checkpoint (GPL code)
  - HunyuanWorld-Mirror (territory exclusions)
- **Internal benchmarking with non-commercial weights:** even inside a company this is arguably commercial use. Get legal sign-off first.

### 3. Recommended pipeline (best accuracy that stays commercial-safe)

1. **Decode and select frames.**
   - Convert HLG to SDR first; the models were trained on SDR.
   - Score sharpness (Laplacian variance) and choose 600–1,200 frames by overlap.
   - Detect lens switches (metadata or a focal jump) and treat each lens as its own camera.
2. **Initial intrinsics.**
   - Run MoGe-2 or GeoCalib on all selected frames and take the median per lens.
   - Better still, calibrate each iPhone model's lenses once. The Stray/ARKit recording already contains wide-lens intrinsics.
   - Stabilisation crops the image, so allow focal refinement later.
3. **Poses.**
   - Run VGGT-1B-Commercial in chunks of about 100–200 frames with about 50% overlap. On an H100 this is seconds per chunk; budget about 1–2 min per video.
   - Feed the known intrinsics if possible, or replace VGGT's focal with yours.
   - Join chunks with a Sim(3) pose graph: VGGT-Long's solver, or HiP3R's two-stage graph.
   - Replace the SALAD loop detector (GPL) with your own DINOv2 (Apache) retrieval, or with DBoW.
4. **Global refinement.**
   - Match features between nearby and loop frames (XFeat is VGGT-X's choice; its licence is not verified here).
   - Run COLMAP or GLOMAP bundle adjustment (BSD) starting from the VGGT poses, with shared intrinsics per lens.
   - Then let gsplat refine poses while training.
5. **Metric scale.**
   - Compute a robust median ratio between the bundle-adjusted depth and metric depth from DA3METRIC-LARGE, MoGe-2 and MapAnything-apache's metric factor.
   - Require the estimators to agree; if they do not, flag the job.
6. **Gravity.**
   - Average GeoCalib's per-frame gravity after rotating it into the world frame.
   - Cross-check with the floor normal (the largest horizontal plane) and with vertical vanishing directions.
7. **Never fail silently.** Fail loudly on any of these:
   - registration below about 95%
   - high reprojection error after bundle adjustment
   - large chunk-alignment residuals
   - focal disagreement between sources above 5%
   - metric scale disagreement above 5%
   - gravity disagreement above 3°
8. **Benchmark against your Stray LiDAR capture.**
   - Measure ATE after Sim(3) alignment, the error in the estimated scale factor, and focal error.
   - Measure PSNR of the finished splat.
   - Compare at least: (a) MapAnything-apache alone, today's baseline; (b) chunked VGGT-1B-Commercial with Sim(3); (c) (b) plus COLMAP/GLOMAP bundle adjustment; (d) Map-Long with known intrinsics.

### 4. Caveats
- The VGGT licence wording ("Research Materials") is ambiguous; have a lawyer review it.
- I found no published numbers on iPhone HDR or stabilised video, or on blur. Your Stray benchmark is the deciding test.
- The DA3 Table 2 comparison uses the non-commercial MapAnything checkpoint; the Apache one is likely somewhat worse.
- The VGGT H100 memory table is from memory of an earlier README and was not re-verified.


## slam
# How to get camera poses from casual iPhone room videos: SLAM and casual-video methods, 2024–2026 (checked 2026-11-25)

## Bottom line
1. **First choice: ViPE (NVIDIA) with a commercially licensed configuration.** It also fits the "few minutes on one GPU" budget.
   - **What it does:** it starts the focal from GeoCalib and refines it, runs DROID-style dense bundle adjustment, and anchors metric scale with a monocular metric-depth prior.
   - **TUM accuracy (beats the alternatives):**
     | Method | ATE F1 / F3 | Focal error F1 / F3 |
     |---|---|---|
     | ViPE | 3.6 / 1.5 cm | 1.8° / 0.6° |
     | MegaSaM | 7.0 / 1.5 cm | 10.5° / 12.6° |
     | VGGT | 8.4 / 12.9 cm | 11.1° / 10.1° |
     | MASt3R-SLAM | 6.8 / 7.6 cm | none |
   - **Licence trap:** the default pipeline is not commercial-safe. It uses UniDepth-L, which is CC BY-NC.
2. **Second candidate: VidMap (ETH/CVG, ECCV 2026, Apache code).** It is offline video SfM built on COLMAP, with loop closures and metric depth.
   - It leads every table on phone-like data. On CroCoDL phone (uncalibrated) it scores 95.5 W-AUC, against 88.9 for DA3-Long and 76.4 for GLOMAP. On LaMAR it scores 88.5, against 76.6 for ViPE.
   - **Licence trap:** its default depth and camera prior is DA3-Large, which is CC BY-NC. It would need DA3-Base or DA3-Metric-Large (both Apache). That swap is untested and accuracy is unknown.
3. **Fallback or cross-check: DROID-SLAM or DPV-SLAM++.** Both have permissive code and weights (BSD-3 / MIT), but both need intrinsics. Supply them from GeoCalib or the iPhone's own metadata.
4. **Benchmark only, never ship:** MASt3R-SLAM (CC BY-NC-SA), CUT3R (CC BY-NC-SA), VGGT-X (CC BY-NC), π3 weights (CC BY-NC).
5. **MegaSaM:** the code is Apache, but it needs UniDepth (NC) and supports at most 1024 frames. Below 1024 frames it is fine only after replacing UniDepth.
6. **VGGT-SLAM 2.0:** the authors themselves report that VGGT fails on plain white walls. It loads the non-commercial VGGT-1B checkpoint by default.

## Method by method

### MegaSaM (Google/Cornell, CVPR 2025)
- **Code and weights:** https://github.com/mega-sam/mega-sam · paper https://arxiv.org/abs/2412.04463
- **Accuracy (paper, ATE / RTE / RRE, uncalibrated):**
  - Sintel: 0.023 / 0.008 / 0.06
  - DyCheck: 0.020 / 0.005 / 0.06
  - In-the-wild: 0.004
  - Baselines it beats: CasualSAM, ACE-Zero, Particle-SfM, MonST3R.
- **Accuracy (ViPE paper, TUM):** ATE 7.0 cm on F1 and 1.5 cm on F3, but focal error 10.5° to 12.6°. On the RDS driving set its focal error was 47.7°, and its scale is "indefinite" (no metric scale).
- **Focal:** starts from the median of per-frame UniDepth estimates, then refines it in bundle adjustment. Focal refinement switches off when the focal Hessian entry falls below τ_f = 50, i.e. when focal is unobservable.
- **Runtime:**
  - The ViPE paper measured about 15 minutes per TUM sequence.
  - The authors say the maximum supported is 1024 frames, and issue #20 reports out-of-memory errors on long clips: https://github.com/mega-sam/mega-sam/issues/20
- **Failure modes (paper):**
  - Moving objects dominating the view.
  - Too little trackable content (white walls).
  - Camera motion collinear with object motion.
  - It cannot handle a focal length that changes during the video (so iPhone lens switches break it) or strong radial distortion.
  - It does handle near-pure rotation, using mono-depth regularisation.
- **Loop closure:** none explicit; it relies on DROID-style global bundle adjustment.
- **Licence:**
  - Code: Apache-2.0.
  - Depth Anything v1: Apache.
  - RAFT and DROID: BSD.
  - **UniDepth: CC BY-NC 4.0 (blocker)**, https://github.com/lpiccinelli-eth/UniDepth
  - A commercial build would replace UniDepth with MoGe-2 (MIT, https://github.com/microsoft/MoGe) and drop the focal prior.

### MASt3R-SLAM (Imperial, CVPR 2025)
- **Code:** https://github.com/rmurai0610/MASt3R-SLAM · paper https://arxiv.org/abs/2412.12392
- **Accuracy (ATE):**
  | Benchmark | Calibrated | Uncalibrated |
  |---|---|---|
  | TUM | 0.030 m | 0.060 m |
  | 7-Scenes | 0.047 m | 0.066 m |
  | EuRoC | 0.041 m | 0.164 m |
- **Focal:** treated as a generic central camera with rays only. No explicit focal is output (the ViPE paper lists its focal as "N/A"), so focal would need a separate fit. Because each frame effectively has its own camera model, it tolerates zoom and lens changes.
- **Runtime:** about 15 FPS on an RTX 4090, roughly 1–1.5 minutes per 1000 frames.
- **Loop closure:** yes, retrieval plus MASt3R decoder checks. It takes TUM uncalibrated from 0.090 to 0.060 m.
- **Failure modes:** pointmaps are not globally refined; trained on pinhole images only; drift on KITTI.
- **Licence:** code CC BY-NC-SA 4.0 (LICENSE.md), and the MASt3R checkpoints are also non-commercial (Naver). **Research only.**

### DROID-SLAM, DPVO, DPV-SLAM (Princeton)
- **Code:** https://github.com/princeton-vl/DROID-SLAM (BSD-3) and https://github.com/princeton-vl/DPVO (MIT). The released weights ship in the same repos.
- **Accuracy:**
  - DROID-SLAM, calibrated: TUM 0.038, EuRoC 0.022.
  - DROID-SLAM, uncalibrated with GeoCalib intrinsics: TUM 0.158, with 0.918 on fr1/room.
  - ViPE paper's DROID run on TUM: ATE 4.4 / 2.7 cm, focal error 4.1° / 4.3°.
  - DPV-SLAM / DPV-SLAM++: TUM 0.076 / 0.054, EuRoC 0.024 / 0.023 (paper https://arxiv.org/abs/2408.01654).
- **Runtime (RTX 3090):** DPV-SLAM runs at 30–50 FPS in 4–7 GB; DROID-SLAM at about 20 FPS in about 20 GB. That is under a minute per 1000 frames.
- **Intrinsics:** required by all three, with no estimation.
- **Loop closure:**
  - DROID-SLAM: global bundle adjustment.
  - DPV-SLAM: proximity loop closure.
  - DPV-SLAM++: adds dBoW2 / ORB retrieval with Sim(3) correction.
- **Failure modes:**
  - Scale drift.
  - Pure rotation is degenerate for depth.
  - Weak on textureless walls.
  - DPV-SLAM's global bundle adjustment is capped at 1000 keyframes.
- **Licence:** commercial OK.

### VGGT-SLAM 1.0 (NeurIPS 2025) and 2.0 (RSS 2026)
- **Code:** https://github.com/MIT-SPARK/VGGT-SLAM (BSD-2) · papers https://arxiv.org/abs/2505.12549 and https://arxiv.org/abs/2601.19887
- **Accuracy (uncalibrated ATE):**
  - v1, SL(4) backend: TUM 0.053, 7-Scenes 0.036.
  - v2: TUM 0.041, the best uncalibrated result there.
- **Focal:** comes from VGGT's camera head. In v2 the true calibration is not recovered; the method only enforces a shared calibration across overlapping frames.
- **Runtime (v2):** 8.4 FPS on a 3090 with 16-frame submaps. VGGT inference is about 1.25 s per submap, and VGGT fits about 60 frames per pass on 24 GB.
- **Failure modes (v2 paper):**
  - **Plain white walls made VGGT fail, and the system diverged in an empty house.**
  - No lost-tracking recovery.
  - The first frame of each submap dominates.
  - Diverged on KITTI.
  - v1 additionally suffered planar degeneracy and trouble with pure rotation.
- **Loop closure:** SALAD retrieval plus VGGT attention-based verification.
- **Licence:**
  - Code: BSD-2.
  - It downloads `facebook/VGGT-1B` by default (main.py), which is **non-commercial**. It must be swapped for VGGT-1B-Commercial: commercial use allowed except military, gated (an access application on Hugging Face) — https://huggingface.co/facebook/VGGT-1B-Commercial
  - SALAD retrieval is GPL-3.0.

### GLORIE-SLAM (2024)
- **Code:** https://github.com/zhangganlin/GlORIE-SLAM (Apache-2.0) · paper https://arxiv.org/abs/2403.19549
- **Accuracy (ATE):** Replica 0.35 cm, TUM 1.1–2.1 cm, ScanNet 7.5–7.6 cm.
- **Runtime:** 0.23 FPS with 15 GB on a 3090, which is over an hour per 1000 frames. **Too slow for the budget.**
- **Intrinsics:** a fixed, known K is required.
- **Loop closure:** yes.
- **Depth prior:** Omnidata, licence not checked.
- **Verdict:** not suitable.

### ViPE (NVIDIA, 2025; v1.2 released 2026)
- **Code:** https://github.com/nv-tlabs/vipe · paper https://arxiv.org/abs/2508.10934
- **Accuracy (TUM):** see the table under "Bottom line". On KITTI its ATE is 9.2 m, against 25.4 m for MegaSaM.
- **Runtime:**
  - Paper: 3–5 FPS on an RTX 5090 at 640×480, about 3 minutes per TUM sequence.
  - v1.2 claims a 2.7× speed-up.
  - My unmeasured estimate: about 1.5–4 minutes per 1000 frames on an H100.
- **Pipeline:** GeoCalib on 4 frames for intrinsics, DROID flow plus bundle adjustment with focal optimisation, and a keyframe metric-depth prior.
- **Long videos:** `pose_only_long` keeps memory bounded.
- **Camera models:** pinhole and wide-angle.
- **Loop closure:** in the metrictrail pipeline, using PyPose plus external SALAD.
- **Licence:**
  - ViPE's own code: Apache.
  - Defaults: `default` uses UniDepth-L (NC). `pose_only` uses `moge2-l`, which is MIT and commercial OK.
  - Third-party code vendored in the repo: UniDepth and UniK3D (NC), and **Segment-and-Track-Anything (AGPL-3.0)**, which is used for the instance masks.
  - Research-only weights: Depth Pro 2 (Apple research-only) and the MetricTrail DINO-DPVO weights (NVIDIA non-commercial).
  - GeoCalib: Apache.
- **Commercial-safe recipe:** `pipeline=pose_only` with MoGe-2 and GeoCalib, instance segmentation disabled (the room is static), and no UniDepth, Depth Pro 2 or MetricTrail. I have not confirmed that a config option fully disables the AGPL instance-tracking module. The open-source pipeline in the paper also used NVIDIA cuVSLAM sparse tracks; the shipped config uses `sparse_tracks: dummy`.

### VidMap (ETH/CVG, ECCV 2026)
- **Code:** https://github.com/cvg/vidmap (Apache-2.0) · paper https://arxiv.org/abs/2607.27194
- **Accuracy:**
  - LaMAR, uncalibrated W-AUC: 88.5, against ViPE 76.6, DA3-Long 76.5, and MegaSaM or MASt3R-SLAM at about 52–58.
  - ETH3D-SLAM, uncalibrated AUC@5cm: 51.4, against ViPE 38.7 and MegaSaM 38.4.
  - EuRoC, uncalibrated AUC@5cm: 11.5, against VGGT-SLAM 2.0 at 1.3.
  - Calibrated and uncalibrated results are nearly identical, which suggests the focal is estimated accurately.
- **Focal:** GeoCalib or DA3 priors plus view-graph calibration, refined in bundle adjustment. It supports distortion models.
- **Runtime:** no per-1000-frame figure; the paper says it is about GLOMAP speed and linear in keyframes. The first run downloads about 9 GB of models.
- **Loop closure:** yes, via MegaLoc retrieval and FlexGlue matching.
- **Failure modes (paper):**
  - Forward-only motion.
  - Repeated, look-alike structure causing false loops.
  - Severe blur or low texture breaking tracks.
  - Unreliable priors on out-of-distribution data.
- **Licence:**
  - FlexGlue, MegaLoc, RoMaV2: MIT.
  - COLMAP: BSD.
  - FlexGlue uses a DINOv3 backbone; the DINOv3 licence terms are not checked.
  - **The default `da3-large` is CC BY-NC.**

### Others
- **DROID-W** (CVPR 2026, DROID-SLAM for dynamic in-the-wild video): code Apache (https://github.com/MoyangLi00/DROID-W). Depth comes from Metric3D v2, whose code is BSD-2; the authors refer commercial enquiries to themselves, so the weights' terms are unclear. Whether it estimates intrinsics is unconfirmed.
- **ViSTA-SLAM** (3DV 2026): TUM uncalibrated 0.052. Mixed licence: Apache plus files derived from DUSt3R under CC BY-NC-SA.
- **VGGT-Long:** follows the VGGT licence, so it needs VGGT-1B-Commercial.
- **Depth Anything 3:** code Apache. DA3-Giant, DA3-Large and the Nested model are CC BY-NC. DA3-Base, DA3-Small, DA3-Metric-Large and DA3-Mono-Large are Apache. DA3-Streaming runs long video in under 12 GB.
- **π3 / π3X:** code BSD-3, **weights CC BY-NC**.
- **CUT3R:** CC BY-NC-SA.
- **VGGT-X:** CC BY-NC.
- **MapAnything:** Apache, as you already use.

## Recommendations for this product
1. **Benchmark against your Stray ground truth.** Run these on 1–2 room videos and record ATE after Sim(3) alignment, metric scale error, focal error, and the splat's PSNR:
   - ViPE `pose_only` with MoGe-2, on about 10 fps subsampled frames (600–1200 frames).
   - VidMap uncalibrated, with DA3-Large to measure accuracy and with the Apache DA3-Base or DA3-Metric-Large to see the cost of a licence-safe swap.
   - DPV-SLAM++ or DROID-SLAM given GeoCalib intrinsics.
   - MASt3R-SLAM and MegaSaM as research references only.
2. **Give the focal a strong prior from the iPhone itself.** The Camera app's MOV files can carry Keys:CameraLensModel and Keys:CameraFocalLength35mmEquivalent (see the exiftool forum, https://exiftool.org/forum/index.php?msg=90937). Check with `exiftool -a -G1 -ee` on your own file. Two caveats:
   - Stabilisation crop and lens switches change the effective focal, so treat the metadata as a prior, not a fixed value.
   - Detect lens switches, then split or flag the video, because MegaSaM and DROID-style methods assume a single fixed focal.
3. **Gravity:** none of these methods output it. GeoCalib predicts gravity per frame (Apache, https://github.com/cvg/GeoCalib); take a robust average over keyframes.
4. **Never fail silently. Gate each run on:**
   - fraction of frames registered or tracked;
   - focal agreement with the metadata prior, within about 5%;
   - reprojection or flow residuals;
   - loop-closure count;
   - Sim(3) agreement between two independent methods, e.g. ViPE against VidMap or DPV-SLAM.

   Then select frames by sharpness and coverage.
5. **Treat these as unverified:**
   - ViPE's runtime on an H100 (only RTX 5090 numbers exist);
   - whether ViPE's AGPL module can be fully disabled;
   - DINOv3's licence terms;
   - Metric3D's weight licence;
   - VidMap's accuracy after swapping in an Apache DA3 model.

## Sources
- MegaSaM: https://github.com/mega-sam/mega-sam and https://arxiv.org/abs/2412.04463
- MASt3R-SLAM: https://github.com/rmurai0610/MASt3R-SLAM and https://arxiv.org/abs/2412.12392
- VGGT-SLAM: https://github.com/MIT-SPARK/VGGT-SLAM, https://arxiv.org/abs/2505.12549 and https://arxiv.org/abs/2601.19887
- VGGT licence: https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt
- ViPE: https://github.com/nv-tlabs/vipe and https://arxiv.org/abs/2508.10934
- VidMap: https://github.com/cvg/vidmap and https://arxiv.org/abs/2607.27194
- DPV-SLAM: https://arxiv.org/abs/2408.01654
- GLORIE-SLAM: https://arxiv.org/abs/2403.19549
- DROID-W: https://github.com/MoyangLi00/DROID-W
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- π3: https://github.com/yyfz/Pi3
- Depth Anything V2 licences: https://github.com/DepthAnything/Depth-Anything-V2
- CUT3R: https://github.com/CUT3R/CUT3R
- ViSTA-SLAM: https://github.com/zhangganlin/vista-slam
- VGGT-Long: https://github.com/DengKaiCQ/VGGT-Long
- Depth Pro 2 model licence: https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL
- MetricTrail weights: https://huggingface.co/nvidia/metrictrail
- SALAD: https://github.com/serizba/salad

## classical
## Short answer
The most reliable commercial-safe setup is a classical global SfM back end with learned features. Use **COLMAP 4.2 global mapper (GLOMAP, now built into COLMAP; BSD)** with **ALIKED+LightGlue or LoMa features**, **sequential + loop-closure matching**, a **focal prior from the iPhone metadata**, and **metric scale from a commercial-OK monocular depth model** (MoGe-2 MIT, or DA3METRIC-LARGE Apache).

Benchmark two video-specific systems against it on the Stray ground truth:
- **VidMap** (ETH/CVG, ECCV 2026, Apache code). It is built on COLMAP's global mapper and has the best published phone-video numbers. Its default depth checkpoint is non-commercial and must be swapped.
- **NVIDIA ViPE `pose_only`** (Apache, DROID-SLAM BSD + MoGe-2 MIT + GeoCalib Apache). It is fast, dense, metric, and copes well with white walls.

Run ViPE and COLMAP side by side and flag any video where their trajectories disagree. That gives the "never fail silently" check.

Do not count on splat-time pose refinement (gsplat `pose_opt`) to fix errors of 0.8 m or a wrong focal. It only corrects small residuals.

## Licence check (verified from LICENSE files / HF tags)
**Commercial OK**
- COLMAP 4.x + GLOMAP: new BSD (colmap/colmap LICENSE). GLOMAP has been maintained inside COLMAP since 4.0.0 (14 Mar 2026).
- hloc: Apache-2.0.
- LightGlue code and weights: Apache-2.0. The LightGlue README states that SuperPoint has a different, restrictive licence, including the `superpoint.py` inference file.
- ALIKED: BSD-3. DISK: Apache-2.0. LoMa: MIT, with the matcher Apache-2.0 (COLMAP models-loma release notes).
- RoMa: MIT. RoMaV2 and FlexGlue: MIT, but both embed a DINOv3 backbone under the DINOv3 License. That licence permits commercial use but bars ITAR and military uses and adds sanctions and trade-control terms. Have someone read it.
- MegaLoc: MIT. gsplat: Apache-2.0.
- ViPE core: Apache-2.0. DROID-SLAM: BSD-3. MoGe/MoGe-2 code and weights: MIT. GeoCalib: Apache-2.0.
- Depth Anything 3: code Apache. Weights DA3-SMALL, DA3-BASE and DA3METRIC-LARGE are Apache.
- Pi3 weights (yyfz233/Pi3): BSD-2. GLUEMAP code: BSD.
- VGGT code (VGGT License, July 2025) allows commercial use except military. Only the gated `facebook/VGGT-1B-Commercial` checkpoint is covered.
- MapAnything: `facebook/map-anything-apache` is Apache. Note that `facebook/map-anything` is CC BY-NC.

**Research-only / non-commercial (do not ship)**
- VGGT-X code: CC BY-NC 4.0.
- InstantSfM: CC BY-NC 4.0.
- ACE0 (ACE Zero): Niantic non-commercial licence.
- MASt3R (and so MASt3R-SfM / MASt3R-SLAM): CC BY-NC-SA. MUSt3R: NAVER licence, the header suggests non-commercial (not read in full).
- InstantSplat: code is Apache, but it downloads MASt3R weights, so it is non-commercial in practice.
- CF-3DGS: NVIDIA Source Code License (non-commercial).
- VGGT-1B (original): CC BY-NC. VGGSfM tracker (`facebook/VGGSfM`): CC BY-NC.
- SuperPoint / SuperGlue: Magic Leap non-commercial.
- UniDepth: CC BY-NC. UniK3D: CC BY-NC-SA.
- Apple Depth Pro 2: research-only model licence.
- ViPE MetricTrail DINO-DPVO weights: NVIDIA one-way non-commercial.
- Pi3X: CC BY-NC.
- DA3-GIANT, DA3NESTED-GIANT-LARGE(-1.1) and DA3-LARGE: CC BY-NC. For DA3-LARGE-1.1, the HF tag says Apache but the official README table says CC BY-NC, so treat it as non-commercial.

**Hidden non-commercial dependencies to watch**
- VGGT's own BA demo (`demo_colmap.py --use_ba`) uses `aliked+sp` (SuperPoint) plus the CC BY-NC VGGSfM tracker.
- ViPE's `default` pipeline uses UniDepth-L. The `pose_only` pipeline uses MoGe-2, which is fine.
- VidMap's default depth model is `depth-anything/DA3NESTED-GIANT-LARGE-1.1`, which is CC BY-NC. I found this in `vidmap/frontend/models/depth/da3_video.py`.
- GLUEMAP's defaults use Doppelgangers++ (MASt3R-based, non-commercial), the VGGSfM tracker (CC BY-NC) and SALAD (GPL-3 code).

## Candidates

**1. COLMAP 4.2.x (GLOMAP global mapper + learned features). BSD. Primary recommendation.**
- Relevant features, all from the official CHANGELOG:
  - **4.0:** GLOMAP integrated; ALIKED (ONNX) + LightGlue for SIFT and ALIKED; EXIF orientation auto-rotation; structure-less registration fallback; view-graph calibration.
  - **4.1:** Caspar GPU bundle adjustment (rig-aware), described as 1–2 orders of magnitude faster than Ceres-CUDA on medium and large problems; gravity priors from EXIF orientation.
  - **4.2:** global mapper returns every disconnected component as its own model; LoMa learned features; a 6-point shared-focal relative-pose solver; DEGENSAC for plane-dominated scenes (white walls and floors); `SequentialMatching.loop_detection_min_index_distance`.
  - **4.2.1:** fixes a global-mapper BA regression. Use 4.2.1 or newer.
- **Weak spot:** the global mapper estimates intrinsics through view-graph calibration and needs a good focal prior. InstantSfM reports GLOMAP failing on ScanNet because Ceres diverges in ViewGraphCalibration. Pass known intrinsics so that step is avoided.
- **Accuracy and robustness:** on the LaMAR phone videos, GLOMAP+ALIKED+LightGlue gets AUC 10.9/30.1/46.0 versus GLUEMAP 26.4/59.1/80.6. Sparse keypoints lose repeatability on textureless regions (VidMap App. B.2). Expect it to be accurate where texture exists and to break into pieces on white walls, which 4.2 at least reports explicitly.
- **Runtime:** no official figure for 1,000 video frames. My estimate is a few minutes for GPU feature extraction + sequential matching + global mapping on an H100.

**2. VidMap (cvg/vidmap, Apache code, ECCV 2026). Best published video SfM. Top benchmark candidate.**
- **How it works:** keyframing, dense tracks, loop closures and metric depth priors, then COLMAP global positioning + BA. Matcher is FlexGlue by default, RoMaV2 optional. Retrieval is MegaLoc. Focal comes from GeoCalib + view-graph calibration, or is supplied (a calibrated mode exists). It handles rigs.
- **Results (paper):**
  - LaMAR iPhone videos (LiDAR-aligned ground truth, mixed indoor/outdoor), full-trajectory AUC: 88.5 uncalibrated / 89.7 calibrated. The best baseline is ViPE at about 76.6. Baselines included MegaSaM, MASt3R-SLAM, DA3-Long, GLOMAP and COLMAP.
  - ETH3D-SLAM AUC@5cm: 51.4 uncalibrated (ViPE 38.7) and 69.4 calibrated.
- **Runtime:** "comparable to GLOMAP, much faster than COLMAP". No absolute numbers in the text. The first run downloads about 9 GB of checkpoints.
- **Stated failure modes:** severe blur, low texture, forward-only motion, out-of-distribution depth priors.
- **Licence work needed:** swap DA3NESTED-GIANT for an Apache depth model (code change, because it uses DA3's nested any-view + metric branches), and clear the DINOv3 terms.

**3. NVIDIA ViPE (nv-tlabs/vipe), `pose_only` / `pose_only_long` pipelines. Apache chain if configured right.**
- **How it works:** DROID-style dense-flow BA with metric keyframe depth (MoGe-2) and GeoCalib intrinsics. It accepts `intrinsics: gt`, so the iPhone focal can be fed in.
- **Speed:** 3–5 FPS on one GPU (paper), so about 3–6 min for 1,000 frames.
- **Accuracy:** reports +18% over uncalibrated baselines on TUM. Dense flow tolerates low texture better than keypoints, but it is SLAM-grade, not BA-grade (VidMap beats it clearly).
- **Use:** as an initialiser or independent cross-check, and as a fallback when COLMAP fragments.

**4. GLUEMAP (colmap/gluemap, BSD code, CVPR 2026 Highlight). Feed-forward local stars inside global SfM.**
- **Backbones:** Pi3 (BSD-2 weights, the default), VGGT, MapAnything.
- **Strong where classical fails:** SMERF indoor multi-room AUC@5 54.9 versus Pi3 18.0. On ETH3D it beats SIFT-GLOMAP.
- **Slow:** augmented BA median 95 s on ETH3D, and about 5.4 h on LaMAR (6–9k images) on a GH200.
- **Licence work needed:** default deps include non-commercial parts (Doppelgangers++, VGGSfM tracker, SALAD GPL). Set `skip_doppelgangers` and check the tracker.

**5. Feed-forward + BA**
- **VGGT-X:** AUC@30 0.992 on MipNeRF360 and 0.971 on Tanks & Temples. It uses XFeat epipolar global alignment (1.78 min / 11 GB on MipNeRF360). NVS PSNR is still 1 dB below COLMAP-initialised 3DGS. The authors note overfitting, and say joint pose optimisation cannot fix large deviations. The code is CC BY-NC, so the idea can be reimplemented (VGGT-1B-Commercial + XFeat Apache) but not the repo.
- **VGGT + BA (vggt repo):** the BA path pulls non-commercial SuperPoint and VGGSfM.
- **MASt3R-SfM:** non-commercial and weaker (AUC@30 0.718 on MipNeRF360 in the VGGT-X table).
- **InstantSfM:** GPU global SfM, 2–20x faster than COLMAP on large scenes. On ScanNet COLMAP/GLOMAP often fail; InstantSfM gets Chamfer 0.842 without depth and 0.658 with a depth prior. Non-commercial.
- **ACE0:** non-commercial. I did not verify its numbers.
- **Takeaway:** your MapAnything result (35% focal error, 0.8 m off) is consistent with feed-forward models degrading on many views and large rooms (GLUEMAP finding). Use them only as local initialisers inside global BA.

**6. Pose refinement during splat training**
- **gsplat `pose_opt`** (Apache): per-camera SE(3) delta, lr 1e-5, off by default, no intrinsics refinement. It fixes cm/sub-degree errors only.
- **VGGT-X Table 4:** joint pose optimisation raised PSNR from 22.4 to 26.3 from a rough init, but residual large errors persist.
- **CF-3DGS** (NVIDIA non-commercial) and **InstantSplat** (MASt3R weights, non-commercial) are not shippable.
- Recommendation: enable `pose_opt` as a polish only. Poses must arrive about cm-accurate, with the focal fixed from the prior.

## What the splat apps use (public info is thin)
- **Polycam:** says an SfM step runs first and offers "Smart Keyframing" (drops blurred frames). Engine not disclosed. A third party (PlayCanvas) says it exports COLMAP-format poses.
- **Postshot (Jawset):** has its own camera-tracking stage (algorithm not disclosed). It imports COLMAP bin/txt, BlocksExchange XML and Pix4D OPF and then skips tracking.
- **Scaniverse:** 100% on-device processing, with an option to disable LiDAR. The pose source is not disclosed; ARKit is likely but unverified.
- **Luma, KIRI:** I found no public description.

## Practical notes for iPhone Camera-app video
- **Focal prior:** pull the lens model and 35 mm-equivalent focal from QuickTime metadata to fix or prior the focal. Stabilisation crops and warps frames, so the effective focal is longer than the stills value. Calibrate per lens/mode against the Stray capture. This may explain the 35% shortfall.
- **Lens switches:** split into per-lens camera segments, i.e. separate COLMAP cameras.
- **HDR:** tone-map HLG to 8-bit SDR before feature extraction.
- **Frame selection:** pick about 500–1,500 frames by sharpness (Laplacian) plus parallax keyframing.
- **Metric scale:** take a robust median ratio of a metric depth prior to SfM depth.
- **Gravity:** dominant floor and wall normals, or the depth/normal model. COLMAP's EXIF gravity is coarse.
- **"Never fail silently" checks:** registered fraction; number of components (4.2 multi-model); mean reprojection error; focal versus prior; scale consistency across frames; ViPE-vs-COLMAP Sim(3) ATE.

## Benchmark plan on the Stray ground truth
Run COLMAP4.2+ALIKED/LightGlue (sequential + loop), COLMAP4.2+LoMa, VidMap (Apache depth swap), ViPE pose_only, and GLUEMAP+Pi3. For each, report:
- ATE after Sim(3) alignment, rotation error, and focal error.
- Registration rate and wall time on the H100.
- Splat PSNR on held-out frames, compared with the 20.2 dB LiDAR reference.

## Sources
- COLMAP changelog: https://github.com/colmap/colmap/blob/main/CHANGELOG.rst ; COLMAP LICENSE: https://github.com/colmap/colmap/blob/main/LICENSE ; LoMa/FlexGlue model releases: https://github.com/colmap/colmap/releases
- COLMAP 4.2 article: https://radiancefields.com/colmap-4.2.0-adds-multi-component-global-mapping-and-loma-learned-matching
- VidMap: https://github.com/cvg/vidmap , https://arxiv.org/abs/2607.27194
- GLUEMAP: https://github.com/colmap/gluemap , https://arxiv.org/abs/2605.26103
- ViPE: https://github.com/nv-tlabs/vipe , https://arxiv.org/abs/2508.10934
- VGGT-X: https://arxiv.org/html/2509.25191v2 , https://github.com/Linketic/VGGT-X (LICENSE.txt = CC BY-NC 4.0)
- InstantSfM: https://arxiv.org/html/2510.13310v2 , https://github.com/cre185/InstantSfM (LICENSE = CC BY-NC 4.0)
- VGGT licence: https://github.com/facebookresearch/vggt , https://huggingface.co/facebook/VGGT-1B-Commercial
- Depth Anything 3 model licences: https://github.com/ByteDance-Seed/Depth-Anything-3
- MoGe: https://github.com/microsoft/MoGe ; GeoCalib: https://github.com/cvg/GeoCalib ; Pi3: https://github.com/yyfz/Pi3
- LightGlue licence note: https://github.com/cvg/LightGlue ; hloc: https://github.com/cvg/Hierarchical-Localization
- ACE0 LICENSE: https://github.com/nianticlabs/acezero ; MASt3R LICENSE: https://github.com/naver/mast3r
- InstantSplat: https://github.com/NVlabs/InstantSplat ; CF-3DGS LICENSE: https://github.com/NVlabs/CF-3DGS
- DINOv3 licence: https://github.com/facebookresearch/dinov3 ; RoMaV2: https://github.com/Parskatt/RoMaV2
- gsplat simple_trainer (pose_opt): https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
- Apple Depth Pro 2 model licence: https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE_MODEL
- Polycam: https://learn.poly.cam/hc/en-us/articles/30549121659412 , https://poly.cam/gaussian-splatting , https://developer.playcanvas.com/user-manual/gaussian-splatting/creating/recommended-tools/
- Postshot import docs: https://www.jawset.com/docs/d/Postshot+User+Guide/Importing+Images
- Scaniverse: https://scaniverse.com/news/scaniverse-introduces-support-for-3d-gaussian-splatting , https://apps.apple.com/us/app/id1541433223

Scratch clones used for checking (safe to delete): /private/tmp/claude-501/-Users-abyyworld-Desktop-llms-working-here-litereality/3dba2112-c4f5-423b-8511-99fb562955c0/scratchpad/vidmap_src and /private/tmp/claude-501/-Users-abyyworld-Desktop-llms-working-here-litereality/3dba2112-c4f5-423b-8511-99fb562955c0/scratchpad/vipe_src
