# Open-source parts LiteReality can reuse (checked 2026-11-22)

**How to read this.** "Verified" means a second agent re-checked that repo or weight file in detail: the code, the issue tracker and the license. "Searcher-confirmed" means only the first searcher checked it, by confirming the repo or weights exist through the GitHub CLI or the Hugging Face (HF) API, without that deep re-check. "Unconfirmed / no code" means we could not find runnable code. None of these were run by us. Every speed is a number the authors reported, unless it says otherwise.

**The main lesson.** The repairs that help our score most are changes to *how we train the splat*. They use depth we already have and never invent pixels. The AI "fixer" models (diffusion models that repaint a bad render) make pictures look nicer. But they can *lower* PSNR and SSIM (pixel-match scores against the real phone clip), because they invent detail. So we use them second, and always compare before and after.

---

## 1) Rung 1: a 1-second video along a new camera path that looks real and has the right geometry

Try these in this order.

**A. Better training using the LiDAR depth we already have.** No AI guessing, cheap, and the licenses are clean.

**Prompt Depth Anything (PromptDA).** Searcher-confirmed.
- **Repo:** https://github.com/DepthAnything/PromptDA
- **Weights:** https://huggingface.co/depth-anything/prompt-depth-anything-vitl (also -vits)
- **License:** Apache-2.0
- **What it does:** takes one RGB photo plus the iPhone's 256x192 LiDAR depth map. It returns sharp, metric (real-meters) depth at full photo size. The README example uses exactly our ARKit depth format.
- **What it needs:** one RGB frame and its LiDAR depth. No camera poses.
- **Speed:** one model pass per frame (ms figure not published). The 340M model fits any GPU we have.
- **Use:** denser and sharper "instant" surfels (the small flat discs we build the instant splat from), plus a full-resolution depth target for training.

**DN-Splatter / AGS-Mesh.** Searcher-confirmed.
- **Repo:** https://github.com/maturk/dn-splatter
- **License:** Apache-2.0. It is built on gsplat, so no restricted Inria code.
- **What it does:** extra training rules ("losses") that make the splat match the LiDAR depth and the surface direction ("normals"), plus filtering of bad depth. It was made for iPhone indoor scans.
- **What we copy:** the depth loss, the normal loss and the confidence filtering, ported into our own gsplat 1.5.3 trainer. We don't adopt nerfstudio.
- **Speed:** the authors' version takes minutes. Inside our 1250-step run it adds only a small cost (normals add about 20%).
- **Targets:** the smeared floor and floaters (stray blobs in mid-air) seen from angles the scan barely covered.

**splat-transform** (PlayCanvas). Searcher-confirmed.
- **Repo:** https://github.com/playcanvas/splat-transform
- **License:** MIT
- **Use:** a command-line step on Modal that deletes Gaussians outside the RoomPlan room box, and very faint ones. It takes seconds and removes many floaters before the browser ever sees them.

**B. Single-frame AI fixer (cleans each rendered frame).**

**NVIDIA Fixer.** Verified. Best license of all the fixers, and the fastest.
- **Repo:** https://github.com/nv-tlabs/Fixer
- **Weights:** https://huggingface.co/nvidia/Fixer (ungated)
- **License:** code is Apache-2.0. Weights use the NVIDIA Open Model License, which allows commercial use.
- **What it needs:** one RGB frame and nothing else. It only accepts 16:9 sizes (1024x576 by default), so our 4:3 frames must be cropped or padded.
- **Speed:** 26.5 ms per frame on an H100, 50.6 ms on an A100, 87.8 ms on an L40. That is the model alone: it excludes rendering and video encoding.
- **Catches:**
  - It was trained on driving scenes, so it will probably need fine-tuning on pairs of (our render, real ARKit frame). The training script is included.
  - It cannot take a real reference photo (issue #4), so expect flicker from frame to frame.
  - The Docker image needs a free NVIDIA NGC account and API key, which the user must create.
- **Best use:** clean extra views from new angles and feed them back into gsplat training ("Difix3D+-style": fixed images become extra training targets). The fix then ends up inside the splat, and Spark (our three.js splat viewer) can show it.

**Difix-ref.** Verified. Use only as a quick comparison.
- **Repo:** https://github.com/nv-tlabs/Difix3D
- **Weights:** https://huggingface.co/nvidia/difix_ref
- **License:** non-commercial.
- **What it does:** cleans a render, guided by one real photo (the nearest scan frame).
- **Ease:** three lines of code through the diffusers library, and there is a ready gsplat example.
- **Speed:** really about 0.8 s per 1080p frame on an A100 in fp16 (issue #21). The paper's 76 ms could not be reproduced.
- **Status:** NVIDIA has deprecated it in favour of Fixer, and its old library versions need a separate Modal image.

**C. Offline video fixers and generators.** These make a 1 s clip in seconds to minutes. Use them as baselines and "best possible" references.

**ArtiFixer, 1.3B model.** Verified.
- **Repo:** https://github.com/nv-tlabs/ArtiFixer
- **Weights:** https://huggingface.co/nvidia/ArtiFixer
- **License:** code is Apache-2.0. Weights are non-commercial (research only).
- **What it does:** a video model that fixes a whole rendered path at once, guided by up to 12 real photos. It accepts a custom camera path.
- **What it needs:**
  - our render and its alpha (opacity) map
  - poses and intrinsics (camera settings), which ARKit gives us
  - real reference frames
  - a caption: use the small 4B captioner, because the default 30B one needs about 60 GB
  - an adapter we write ourselves, because the scripts assume COLMAP
- **Speed:** reported 34 fps on one GPU once the model is loaded. Per-scene setup takes longer. It only runs as an offline script.
- **Setup:** use the ready-made image `ghcr.io/nv-tlabs/artifixer:cuda12-ffmpeg-free-20261021-5b1bb939`. Building it yourself takes 1-1.5 h.
- **Risk:** it was trained on *sparse* splats. One user found it "barely works" on dense, LiDAR-started splats (issue #10), and ours are dense.

**GEN3C.** Verified. The "best possible" offline renderer.
- **Repo:** https://github.com/nv-tlabs/GEN3C
- **Weights:** https://huggingface.co/nvidia/GEN3C-Cosmos-7B (ungated)
- **License:** code is Apache-2.0. Weights use the NVIDIA Open Model License, which allows commercial use.
- **What it does:** turns posed photos plus depth into a 3D point cloud, renders it along any path, and a 7B video model makes it look real. Our ARKit poses and LiDAR depth plug straight in, and the custom path goes in as `w2cs_all`.
- **Speed:** about 15 minutes per 121-frame clip on an A100. Our guess is about 10 minutes on an H100.
- **Memory and storage:** about 43 GB with offloading. Downloads are about 90 GB.
- **Catches:**
  - It uses only 2 scan views per 121-frame chunk.
  - The first output frame is pinned to the first keyframe's photo.
  - It only outputs 1280x704.
  - ARKit gives poses in OpenGL convention, and GEN3C needs them converted to OpenCV world-to-camera.

**Other options, lower priority:**
- **ArtifactWorld.** Verified.
  - Repo: https://github.com/fyting/ArtifactWorld
  - License: non-commercial weights, gated with manual approval that takes days to weeks.
  - Its config files only stopped producing garbage in Oct 2026.
- **GSFixer.** Searcher-confirmed.
  - Repo: https://github.com/GVCLab/GSFixer (weights: flow666/GSFixer)
  - License: none.
- **WorldForge V2.** Searcher-confirmed, newer than mid-2026.
  - Repo: https://github.com/Westlake-AGI-Lab/WorldForge
  - Takes our depth-warped frames plus hole masks.
  - Needs 80 GB.
  - Uses the MiniMax-H3 community license.
- **Stable Virtual Camera (SEVA).** Searcher-confirmed.
  - Repo: https://github.com/Stability-AI/stable-virtual-camera
  - License: non-commercial.
  - It ignores our depth.

---

## 2) Fast capture: seconds, with no long per-room training

The honest picture:
- No open model yet gives a finished, high-quality room splat in seconds.
- Splats made in one model pass ("feed-forward") score about 20 PSNR on new views.
- A realistic target is an instant splat in about 10 s, then a 20-40 s refine.

What to copy:

**1. Improve our own instant splat.** Use PromptDA's dense depth, LiDAR confidence filtering, and the RoomPlan room-box crop. This is free and keeps metric scale.

**2. Free-space carving (idea only).** "Carving" means anything the LiDAR sees *through* must be empty air, so floaters can't form there.
- GSFusion: https://github.com/goldoak/GSFusion. Searcher-confirmed. Its Gaussian part is under the non-commercial Inria license. It takes known poses, which suits ARKit.
- GPS-SLAM: https://github.com/MisEty/GPS-SLAM. AGPL license, so read it, don't embed it.

**3. A fast short refine.** We write this ourselves on gsplat, and we need it anyway:
- **FastGS's rule.** Only add or keep Gaussians that several views agree on. The core is about 60-100 lines (`compute_gaussian_score_fastgs`). Copy the *idea*, not the repo.
  - Repo: https://github.com/fastgs/FastGS. Verified.
  - Its rasterizer is under the non-commercial Inria license and crashes on new CUDA versions (issue #7).
  - Pick comparison cameras whose views overlap, using the ARKit poses, not at random (issue #57).
- **Taming 3DGS's Gaussian budget.**
  - Repo: https://github.com/humansensinglab/taming-3dgs. MIT, searcher-confirmed.
  - A fixed Gaussian budget gives predictable training time and file size.
- **Upgrade gsplat from 1.5.3 to main.** Main has faster 2026 kernels, about 30% on some paths, but is not yet on PyPI.
- **Recipe to follow (read, don't run as-is):** the SIGGRAPH Asia 2025 winning "room in 1 minute" pipeline.
  - Repo: https://github.com/will-zzy/siggraph_asia. Searcher-confirmed, research-only.
  - Swap their Metric3D depth estimate for our LiDAR depth.

**4. Feed-forward starting splat** (optional A/B test against our instant splat):
- **AnySplat.** Searcher-confirmed.
  - Repo: https://github.com/InternRobotics/AnySplat (weights: https://huggingface.co/lhjiang/anysplat)
  - License: MIT.
  - Speed: 1.4 s for 32 views, 4.1 s for 64.
  - It estimates its own poses, so we must fit a scale-and-rotation match (a "Sim(3) alignment") to ARKit. Training resolution is only 448 px.
- **WorldMirror 2.0.** Verified.
  - Repo: https://github.com/Tencent-Hunyuan/HY-World-2.0
  - It accepts our poses and depth, but rescales them, so the output is *not* metric and not in our world coordinates.
  - Its exported opacity looks wrong in standard viewers, so we would write our own export.
  - The Tencent license excludes the EU, UK and South Korea.
  - A 1-day test at most.
- **SplatAnything / InfiniSplat.** Searcher-confirmed, Nov 2026, untested by anyone.
  - Repo: https://github.com/zju3dv/InfiniSplat
  - Organizations must register before using it, and the weights bundle Tencent's model.

---

## 3) Later rungs

**Rung 2: an object has moved**
- **Cut the object out of the splat.** Take the Gaussians inside its RoomPlan box and make them a separate SplatMesh (one splat object) in Spark. Move it by changing its position, with no retraining. Each frame, set its position from its MuJoCo body.
- **Spark:** https://github.com/sparkjsdev/spark. Verified, MIT. Upgrade to v2.4.0.
  - Moved objects lag one frame behind the camera (issue #205). Use the `preUpdate` option.
  - Mouse picking ignores edits (issue #497).
- **Cleaner object edges:** SAGA. Repo: https://github.com/Jumpat/SegAnyGAussians. Apache-2.0, searcher-confirmed. Needs a few minutes of training per scene.
- **Filling the hole the object leaves:** Inpaint360GS. Repo: https://github.com/dfki-av/Inpaint360GS. Apache-2.0, searcher-confirmed. Takes minutes, so run it once at capture time as a precompute step.
- **Offline video of the moved scene:** edit GEN3C's depth/photo inputs before rendering.

**Rung 3: live at 30 fps while walking**
- **Keep rendering in the browser.** Spark already draws a 1-2M-Gaussian room in real time. Spark's own guide says iPhones handle 1-3M splats.
- **Put the AI fixes into the splat, not into a video stream.** Use Fixer-cleaned training views. Streaming a server-fixed video breaks "the browser stays the viewer", and it would need one H100 per viewer.
- **Faster first load:** Spark's `build-lod` tool makes a .RAD file the browser can stream in pieces.
- **Splat that grows while scanning:**
  - **2DGS-SLAM.** Repo: https://github.com/PRBonn/2DGS-SLAM. MIT. Its loop-closure part uses MASt3R, which is non-commercial. Searcher-confirmed, very active (Nov 2026).
    - It already reads iPhone LiDAR recordings from the Stray Scanner app.
    - It already streams to a Spark web viewer.
    - Its speed has not been reported.
  - **GS-ICP SLAM.** Repo: https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM. MIT, but its rasterizer is under the Inria license. Searcher-confirmed.
    - Reported about 100 fps on Replica, a standard synthetic benchmark.
    - We would keep its mapping thread and use ARKit poses instead of its tracking.

**Rung 4: several people at once**
- Each person loads the same splat file and renders it locally. Spark supports several viewpoints, but its level-of-detail streaming handles only one camera (issue #369).
- We sync only camera poses and object positions, over a WebRTC data channel or a sync library such as Yjs. We write that part ourselves.
- No open-source multi-user browser splat system was found (searcher's finding, not re-checked).

---

## 4) Traps: things that look good but aren't

- **AI fixers as an accuracy boost.** They invent detail.
  - Difix3D+ lowered gsplat PSNR from 28.5 to 26.7 for one user (issue #40).
  - ArtiFixer 1.3B scores only about 19.3 PSNR on the DL3DV benchmark.
  - Always measure before and after against the held-out clip.
- **ArtiFixer on our splats.** Our dense LiDAR splats are not what it was trained on (issue #10). It is offline only, with no streaming code.
- **Difix speed claims.** The 76 ms cannot be reproduced. It is deprecated, and the 3D distillation takes minutes and can make results worse.
- **GEN3C as a fast tool.** It takes 10-15 minutes per clip and uses only 2 scan views per chunk. It is a "best possible" reference only.
- **WorldMirror 2.0 as an exact metric splat.** It is not metric, its PLY opacity export is probably wrong, and its license bans using its outputs to train other models.
- **Running the FastGS repo directly.** Its rasterizer is non-commercial and crashes on new CUDA. It was never tested on walk-through scans, and it ignores depth.
- **ArtifactWorld.**
  - The weights are gated, with waits of weeks.
  - Its configs were broken until Oct 2026.
  - It needs real photos at the *start and end* of the path. If those come from the held-out test clip, the test is spoiled.
- **HY-WorldPlay** (https://github.com/Tencent-Hunyuan/HY-WorldPlay).
  - Its "24 fps" needs 8 GPUs.
  - It starts from one image, not our scan, so the geometry won't match our room.
  - **Lyra 2.0** (https://github.com/nv-tlabs/lyra) has the same problem: it guesses scale from one image and invents content.
- **Scoring with gsplat renders.** Spark draws Gaussians slightly differently from gsplat, so the scores won't match what users see. Render the test frames in Spark too, or report both.
- **Leaking the test clip.** Never use frames from the held-out phone clip as references, anchors or training data.
- **No license or copyleft.**
  - No license at all (all rights reserved by default): GSFixer, 3DGS-Enhancer, PhysGaussian.
  - GPS-SLAM is AGPL (copyleft): using it inside a web service forces releasing our source.
- **No code exists.** GeoStream (a good design for rung 3), FreeSplat++, DensiTok, TokenSplat. Keep watching them.

---

## 5) Licenses

**Fine for a research paper *and* for a commercial product:**
- Spark (MIT)
- NVIDIA Fixer (Apache code + NVIDIA Open Model License weights)
- GEN3C (Apache code + NVIDIA Open Model License weights; its T5 text encoder is Apache; the Cosmos tokenizer uses a custom license)
- PromptDA (Apache)
- DN-Splatter (Apache)
- AnySplat (MIT; check VGGT's license if we ship VGGT)
- Taming 3DGS (MIT)
- SAGA, Inpaint360GS, Brush (Apache)
- splat-transform and SuperSplat (MIT)
- map-anything-apache weights (Apache)
- Depth Anything 3 Small and Base (Apache)
- GenFusion (MIT/Apache)
- 2DGS-SLAM and GS-ICP SLAM, *except* their MASt3R and Inria parts

**Allowed, but with conditions:**
- WorldMirror 2.0 and HY-WorldPlay: Tencent license. No EU, UK or South Korea; over 1M monthly users needs a deal; no training other models on its outputs.
- WorldForge V2: MiniMax-H3 community license.
- SplatAnything: organizations must register first.
- LTX-Video base model: custom license.

**Research only (fine for the paper, blocks commercial use):**
- ArtiFixer weights
- Difix and Difix-ref
- ArtifactWorld
- Stable Virtual Camera
- Lyra 2.0
- Depth Anything 3 Giant and Large
- Pi3X
- Long-LRM (Adobe Research license)
- Anything containing the Inria rasterizer: FastGS, Speedy-Splat, GSFusion's Gaussian part
- The SIGGRAPH Asia pipeline

**Ask the authors first:** GSFixer, 3DGS-Enhancer (no license file).

**Avoid embedding:** GPS-SLAM (AGPL).

This is not legal advice. Check with the university before anything ships.

---

## Plan for next week

1. **Day 1: build the test harness.** No change to the splat yet.
   - Record a held-out phone clip along a new path, in the same ARKit session or re-aligned to it.
   - Render our current splats along that path in both gsplat and headless Spark.
   - Compute PSNR, SSIM and LPIPS on matching crops.
   - This gives the baseline every later step is compared against.

2. **Day 2: depth fixes.**
   - Run PromptDA (vitl) on Modal to turn every frame's LiDAR into full-resolution metric depth.
   - Rebuild the instant splat from that depth.
   - Add DN-Splatter-style depth and normal losses, plus confidence masking, to our gsplat trainer.
   - Crop to the RoomPlan box with splat-transform.
   - Try gsplat main.
   - Measure again.

3. **Day 3: faster training.**
   - Re-implement FastGS's multi-view-agreement rule on gsplat, choosing comparison cameras by pose overlap.
   - Add a Taming-style Gaussian budget.
   - Aim for 20-40 s on an H100 with equal or better scores.
   - Also try AnySplat as a starting splat for 1 hour, after Sim(3)-aligning it to ARKit.

4. **Day 4: fixers.**
   - The user creates an NGC key and stores it as a Modal secret.
   - Run NVIDIA Fixer on 16:9 crops of new-path renders. Then feed Fixer-cleaned extra views back into training for 1-2 rounds.
   - Run Difix-ref (nearest scan frame as reference) as a comparison.
   - Report all three: raw, post-fixed, and fixed-into-the-splat.

5. **Day 5: offline video baselines.**
   - Cache weights on a Modal Volume.
   - Run ArtiFixer 1.3B from the ghcr image on one path (we write the split.json/caption adapter).
   - Start one GEN3C clip on an H100 as the "best possible" reference.
   - The user requests ArtifactWorld HF access, as a long-shot backup.
   - Write a single results table for the professor: scores, seconds from capture, and license for each method.