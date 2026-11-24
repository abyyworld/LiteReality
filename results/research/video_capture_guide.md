# Indoor room capture by iPhone video for RealityScan poses and LichtFeld training

Labels used below:
- **[official]**: written by the tool's maker: Epic help or docs, the LichtFeld README, wiki or source, or the vendor's own guide.
- **[community]**: forum posts, third-party sites, or replies without a staff badge.
- **[ours]**: our own inference or calculation. No source states it.

---

## 1) What RealityScan requires

**Capture rules (desktop help, [official])**, from https://rshelp.capturingreality.com/en-US/tutorials/takingpictures.htm
- "Use the highest resolution possible."
- "Do not limit image count, RealityScan can handle any."
- Every surface point must be clearly visible in at least 2 high-quality images.
- "Do not change a view point more than 30 degrees" between neighbouring images.
- "Standing at one point produces just a panorama and it does not contribute to a 3D model creation." You must move the camera, not just turn it.
- Work coarse to fine. Move closer gradually, never in one jump.
- This page has nothing on ISO, exposure, lighting, white walls or interiors.

**Overlap [official]**
- About 70% overlap in all directions, and "Avoid hasty movements". This is from the RealityScan *mobile* docs: https://dev.epicgames.com/documentation/realityscan-mobile/Photogrammetry-Camera-Movement
- Neighbouring images should overlap by more than 60% for best quality. "A smaller overlap may cause disconnected components." Source: https://rshelp.capturingreality.com/en-US/appbasics/alignsettings.htm

**Video import [official]**, from https://rshelp.capturingreality.com/en-US/tools/videoimport.htm
- Use Workflow > Video, or drag and drop. Accepted formats: .mov, .mp4, .avi, .asf, .mpg and .wmv. Frames are saved as PNG.
- The only setting is "Jumps length", the gap between extracted frames. On the CLI it is in seconds: https://dev.epicgames.com/documentation/en-us/realityscan/project-and-image-commands
- RealityScan "extracts purely key frames" and ignores codec-interpolated frames. It gives no recommended fps, resolution or Jumps value.
- **Not official:** video import does not check for blur. "Every frame it picks goes into the project, sharp or not." The advice is to use a short Jumps length, then delete blurry frames by hand or with a sharpness tool. This is a reply by Jakub Vanko, with no staff badge: https://forums.unrealengine.com/t/video-frames-import-2026/2845444
- **[community]** iPhone HEVC .MOV files can fail to import on Windows because of codec problems. Staff said "In mostly cases it is related to the codecs", and no fix is posted: https://forums.unrealengine.com/t/knowledge-base-unable-to-import-heif-and-hevc-files-in-realitycapture/682558

**Why alignment splits into components [official]**
- Causes: weak texture, too few images, too large a change of view, and small overlap. https://rshelp.capturingreality.com/en-US/tutorials/mergecomponents.htm
- This matches what happened last time: only 16% of frames joined one component, and it broke on white walls and blinds [ours].

**Tools for joining components [official]**
- Add images where coverage is thin, then Align again (F6). Alignment continues from where it stopped. https://rshelp.capturingreality.com/en-US/tutorials/mergecomponents_images.htm
- Control points: "Connect two components with any 6 control points". Each point must be in more than 1 image of every component. Then Align (F6). https://rshelp.capturingreality.com/en-US/tutorials/mergecomponents_cp.htm
- Printed markers are detected automatically and become tie points. See section 5. https://rshelp.capturingreality.com/en-US/tools/detectmarkers.htm
- RealityScan 2.0 has new default settings that "improve image alignment, especially on surfaces with minimal features". It also has Quality Analysis heatmaps that show where more data is needed. https://www.realityscan.com/news/realityscan-20-new-release-brings-powerful-new-features-to-a-rebranded-realitycapture

**Exporting for splats [official]**
- 2.1: COLMAP export can match the original distorted images, and OpenCV export was added. https://dev.epicgames.com/documentation/realityscan/realityscan-2-1
- 2.1.1 added:
  - binary or ASCII output and flat or standard folder layout
  - a fix for quality loss when exporting with the original images
  - a fix for masks missing from CLI exports

  Source: https://dev.epicgames.com/documentation/en-us/realityscan/realityscan-2-1-1
- The default lens model is Brown3 (for lenses under 180 degrees). https://rshelp.capturingreality.com/en-US/appbasics/settings_distortion_models.htm

**Licence [official]:** free for individuals and companies under $1M USD revenue, with all features. https://www.realityscan.com/download

---

## 2) What LichtFeld Studio requires

LichtFeld does **not** compute camera poses. Poses must come from RealityScan or COLMAP. A built-in video-to-splat pipeline is only an open roadmap issue: https://github.com/MrNeRF/LichtFeld-Studio/issues/1828

LichtFeld publishes **no capture guide**. It says nothing on motion, overlap, fps, shutter, white walls or iPhone.

**Input format [official]**
- "The preferred way… is to import your data (undistorted images + pointcloud + camera locations) in COLMAP format." The wiki also links a RealityScan-to-LichtFeld video tutorial (YouTube JWmkhTlbDvg). We did not watch it. https://github.com/MrNeRF/LichtFeld-Studio/wiki
- The loader looks for the COLMAP files in `sparse/0`, then `sparse`, then the folder root. The code calls the root layout the "Reality Capture / flat structure". https://github.com/MrNeRF/LichtFeld-Studio/blob/master/src/io/include/io/filesystem_utils.hpp
- Images go in `images/` (change with `--images`). Point-cloud priority is points3D.ply, then .bin, then .txt. With no point cloud, LichtFeld uses random initialisation and logs a warning.
- Do **not** use RealityScan's transforms.json. Its SIMPLE_RADIAL camera falls back to PINHOLE with only a warning. MrNeRF: "Why don't you export the colmap format?" https://github.com/MrNeRF/LichtFeld-Studio/issues/4031

**Undistortion [official]**
- Training refuses distorted cameras unless Undistort or GUT is enabled. https://github.com/MrNeRF/LichtFeld-Studio/blob/master/src/core/optimization_properties.cpp
- `--undistort` resamples each image (and its mask) once into a pinhole camera.
- The GUT tooltip warns that "most viewers do not support rendering splats trained with this mode." Prefer pinhole or `--undistort`.
- Warning in the docs: RealityScan can export images that are already undistorted while the metadata still names a distorted camera model. This can give wrong 3DGUT results. https://github.com/MrNeRF/LichtFeld-Studio/blob/master/docs/docs/features/gut.md
- **[community, Epic staff reply]** RealityScan writes PINHOLE or SIMPLE_PINHOLE when the prior is Fixed and the camera model is "No lens distortion". There was a bug where FULL_OPENCV was written with zero distortion, and a fix was promised. https://forums.unrealengine.com/t/colmap-text-export-pinhole-option/2657130
- FAQ: do not rescale, sharpen or denoise images after undistortion. Distorted images and bad alignment make scenes "explode". https://github.com/MrNeRF/LichtFeld-Studio/wiki/Frequently-Asked-Questions

**Size and count [official]**
- `--max-width` defaults to 3840 px. The FAQ gives a hard maximum of 16,777,216 px (4096x4096). Export at the size you will train at. https://github.com/MrNeRF/LichtFeld-Studio/blob/master/src/core/argument_parser.cpp
- There is no minimum or maximum image count. Auto mode runs about 100 training steps per image, with at least 30,000. Resolution drives VRAM; image count does not directly. (FAQ link above.)

**Helpful training options [official]**

Check every flag with `--help`, which MrNeRF calls "the ground truth": https://github.com/MrNeRF/LichtFeld-Studio/issues/1009
- `--exposure-correction`: for images that vary in brightness or colour. Off by default. (optimization_properties.cpp)
- Calibration correction: on by default. It absorbs "rolling shutter, stabilization, focus breathing, a slightly wrong pose". https://github.com/MrNeRF/LichtFeld-Studio/blob/master/docs/docs/cli.md
- `--pose-opt`: refines poses during training. It needs pinhole cameras or `--undistort`, and cannot be combined with `--gut`. (cli.md)
- `--enable-mip`: for "fine textures, repeated patterns, or shimmer". Using it for blinds is [ours]. (argument_parser.cpp)
- Masks: put them in `masks/` next to the images. 0 means exclude and 255 means keep. Use `--mask-mode` (default none). There is **no** `--mask-threshold` CLI flag; that value is only a GUI or config setting. https://github.com/MrNeRF/LichtFeld-Studio/blob/master/src/core/include/core/parameters.hpp
- The default strategy is MRNF ("Use MRNF as the default").

**Hardware [official]:** NVIDIA GPU with compute capability 7.5 or newer, driver 570+ (CUDA 12.8+), at least 8 GB VRAM recommended. AMD and Intel GPUs are not supported. https://github.com/MrNeRF/LichtFeld-Studio#installation

---

## 3) Exact iPhone video settings

Neither Epic nor LichtFeld gives iPhone settings. The settings below combine their rules (sharp frames, highest resolution, fixed camera) with what the app documents.

**App: Blackmagic Camera (free) [official specs]**

It offers manual shutter and ISO, each lockable, plus manual focus, white-balance lock and stabilization Off. https://www.blackmagicdesign.com/products/blackmagiccamera/techspecs/W-APP-01

| Setting | Value | Basis |
|---|---|---|
| Resolution | 4K (3840x2160) | RealityScan says highest resolution. 3840 is also LichtFeld's default max width, so no rescaling [ours] |
| Frame rate | 30 fps | No official number. We subsample later anyway [ours]. See section 7 |
| Shutter | Speed mode, **1/250**, locked. Use 1/500 in bright light. Never slower than 1/125 | The app offers a fixed list (…1/125, 1/250, 1/500…). PlayCanvas minimum is 1/125 handheld [official for PlayCanvas only]. 1/250 is [ours] |
| ISO | Manual, locked. Raise it as needed | Postshot: radiance fields "tolerate noise better than blur" [official Postshot] https://jawset.com/docs/d/Postshot+User+Guide/Capturing+Guidelines |
| Focus | Auto focus OFF, manual, set once for room distance | App spec. RealityScan wants sharp frames |
| White balance and tint | Locked | App spec |
| Stabilization | **Off** | Research: "IS must be disabled when accuracy matters" https://isprs-archives.copernicus.org/articles/XLVI-2-W1-2022/395/2022/ [community/research] |
| Lens | **1x main camera** only, never switch or zoom | See notes below |
| Codec and colour | HEVC or H.264 in SDR (Rec.709), not HDR or Apple Log | [ours] See notes below |

Notes:
- **Shutter and ISO:** Blackmagic's auto exposure "adjusts ISO and shutter angle" (https://www.blackmagicdesign.com/products/blackmagiccamera), so set both by hand. If the picture is very grainy, add room light rather than going slower than 1/125 [ours].
- **Lens:** the 1x choice is [ours]. Postshot's official guide says about 24 mm full-frame equivalent is a good handheld start. The iPhone 1x lens is about 24–26 mm [ours]. PlayCanvas warns that ultra-wide lenses under 24 mm "confuse reconstruction" (https://developer.playcanvas.com/user-manual/gaussian-splatting/creating/taking-photos/). See section 7 for Teleport's opposite advice.
- **Codec and colour:** LichtFeld's extractor has an HDR-to-SDR converter, which shows that HDR needs extra handling [official]. ProRes (Pro models only) avoids keyframe questions but makes files up to 30 times larger. 4K60 ProRes needs external storage: https://support.apple.com/en-us/109041
- **Orientation:** landscape [ours]. The sources disagree; see section 7.

**If you must use the stock Camera app [official Apple]**
- Settings > Camera > Record Video:
  - turn **Enhanced Stabilization OFF** (it is on by default and "zooms in")
  - turn **Lock Camera ON** (it stops automatic lens switching)
  - turn **Lock White Balance ON**
  - turn **Auto Low Light FPS OFF** (it can drop to 24 fps)

  Source: https://support.apple.com/guide/iphone/change-video-recording-settings-iphc1827d32f/ios
- Touch and hold until you see "AE/AF Lock". The stock app has **no manual shutter or ISO**: https://support.apple.com/en-afri/guide/iphone/-iph3dc593597/16.0/ios/16.0
- Do not use Action mode (maximum 2.8K, heavy crop) or Cinematic mode (1080p by default): https://support.apple.com/guide/iphone/record-videos-iph61f49e4bb/ios

---

## 4) How to move and what to film

**Core rules**
- **Never turn on the spot.** Always move the camera sideways or forward while you aim.
  - RealityScan's panorama rule (takingpictures.htm) [official]
  - Postshot: "Don't pan, tilt or roll without moving the camera at the same time" [official Postshot]
  - Teleport: no spinning in place, https://teleport.varjo.com/help/en_US/capturing/movement [official Teleport]
  - COLMAP: https://colmap.github.io/tutorial.html [official COLMAP]
- **No more than 30 degrees of view change between frames you keep** (RealityScan) [official]. Turn corners slowly.

**Room pattern** [official Postshot and Teleport]
- Stand with your **back to the wall** and point the camera into the room, along the longest line of sight.
- Walk **sideways** along the walls for a full 360-degree loop. https://teleport.varjo.com/help/en_US/capturing/capture-paths-
- Do **3 loops** (Postshot says 2 to 5 rings at different heights and tilts):
  1. eye height, looking straight
  2. hip height, tilted up to catch the ceiling and see under desks and chairs
  3. above your head, tilted down to catch the floor
- Include the floor and the ceiling (Postshot).
- Optional: an inner ring of 1–2 m in the room centre, facing outward, then detail passes. Shoot "visual bridges" back toward the ring positions. https://poly.cam/blog/high-resolution-interior-photogrammetry-full-walkthrough-polycam-tutorial [official Polycam tutorial]

**Speed and distance**
- Move slowly and evenly, with no jerks (Teleport, Epic mobile) [official].
- Work coarse to fine and move closer gradually (RealityScan) [official].
- Do not go close to plain, featureless areas. Do not film blank or shiny surfaces straight on. "Use oblique angles and include nearby structure." Do not go closer than about 30 cm unless the object has fine detail. https://get.teleport.varjo.com/blog/how-to-create-best-quality-3d-scans-of-your-environment-with-teleport [official Teleport]

**Scene**
- Keep people and objects still.
- No flash, no moving lights or shadows (Postshot) [official].
- Do not film into bright windows or through doors (COLMAP) [official COLMAP].

**Length**
- No source for RealityScan or LichtFeld gives a length.
- Plan **about 3–6 minutes total** (3 slow loops) instead of 58 s [ours]. Teleport accepts videos of 30 s to 15 min.
- One continuous take is simplest. If you split it into clips, make each clip overlap the previous one [ours].

---

## 5) Fixing white walls and blinds

Official cause: weak texture leads to split components (mergecomponents.htm). No official Epic page gives white-wall techniques. Epic's FAQ only says "Avoid transparent, shiny, or textureless objects". That FAQ is from an anonymised account: https://forums.unrealengine.com/t/faq-about-realityscan-application/712415

**A. Printed RealityScan markers (best official tool)**

Rules from https://rshelp.capturingreality.com/en-US/tools/detectmarkers.htm [official]:
- Generate the markers in RealityScan and print them.
- Each marker should be **at least 100 px wide** in the frame. Large scenes need large markers.
- The camera-to-marker angle should be above 45 degrees; straight on is best.
- Never cover or crop a marker, including its white border.
- **Use each marker ID only once.**
  - AprilTag 16h5 has only 30 IDs. It works better in hard light or at distance.
  - Single ring 12-bit has 161 IDs. Dual ring 12-bit has 512 IDs.
- Set Required measurements greater than 1.
- Detected markers become tie points shared across images. This is how they link frames and components.

Size and placement [ours, calculated]:
- With the 1x lens at 4K, 100 px is about 12 cm at 3 m and about 20 cm at 5 m. An A4 sheet with one marker works up to about 5 m.
- If frames are downsized to 1920 px, double these sizes.
- Spread the markers over the white walls, ceiling edges and floor so every frame sees a few.

Trade-off [ours]: markers will appear in the splat. Either accept that, or mask them in LichtFeld with `--mask-mode ignore`. Masking is untested.

**B. Add real texture**
- COLMAP [official COLMAP]: avoid a "white wall or empty desk". Add posters or other objects.
- RealityScan forum [community]: project a pattern, or use "lots of small stickers or larger things like newspaper". https://forums.unrealengine.com/t/flat-featureless-surfaces-an-alternative-workflow/708773
- SketchUp [official SketchUp]: add temporary texture markers to plain surfaces. https://help.sketchup.com/en/gaussian-splats-best-practices
- Use even, stable lighting (Teleport blog) [official]. Use diffuse or overcast light [community]. https://forums.unrealengine.com/t/white-facades-not-showing-in-pointcloud/2256218

**C. Blinds**
- **Close them fully.** SketchUp: "For interiors, close blinds to prevent bright patches" [official SketchUp].
- Do not film through windows (COLMAP). Avoid blown-out windows (Polycam tutorial).
- No source covers dark mesh blinds or repeating patterns for alignment. For training, `--enable-mip` targets repeated patterns [official flag; using it for blinds is ours].

**D. Inside RealityScan, after capture [official]**

Alignment settings are documented at alignsettings.htm.
- Raise Max features per mpx and per image ("can result in less components").
- Set Preselector features to 1/4–1/2 of the detected features.
- Try Detector sensitivity Ultra. It finds features on weak texture but may add points from noise. See section 7.
- Keep Image downscale factor at 1.
- If components remain, set Force component rematch = True and Align again.
- Use Quality Analysis or Inspect to find thin areas, then add bridging frames and Align (F6).
- Last resort: link components with **6+ manual control points**, each in more than 1 image of every component.

---

## 6) How many frames to extract, and how

**What the sources say**
- RealityScan: do not limit image count.
- LichtFeld: no minimum or maximum. Training time grows with image count (about 100 steps per image).
- Teleport [official Teleport]: **250–1000 frames** for most captures, up to about 2000 for big spaces. Under about 100 often fails. https://teleport.varjo.com/help/en_US/capturing/frame-count-
- Postshot uses only 2–3 fps of video [official Postshot].
- COLMAP: "consider down-sampling the frame rate" [official COLMAP].

**Our plan [ours]**
- Extract about **3 fps**, keeping the sharpest frame in each interval.
- A 4–5 minute capture gives about 700–900 frames. Delete any that are still blurry.
- Aim for **500–1000 sharp frames**.
- Keep full 3840 px width.

**How to extract (best first)**
1. **LichtFeld Video Extractor** [official], File > Extract Frames from Video. https://github.com/MrNeRF/LichtFeld-Studio/blob/master/docs/docs/features/video-masking.md
   - Mode: FPS (3).
   - Sharpness filter: "keep sharpest of N per interval".
   - Format: JPG or PNG.
   - Turn on HDR-to-SDR only if the clip is HDR.
   - It can also mask the camera operator with SAM2.1.
2. **ffmpeg** [ours, not from a source]: `ffmpeg -i clip.mov -vf fps=3 -q:v 1 frames/f_%05d.jpg`. Then remove blurry frames by hand.
3. **RealityScan video import** with a short Jumps length (in seconds) [official]. It extracts key frames only and does **no blur check** [community]. This is the least control.

**Then**
1. Import the frames into RealityScan as images.
2. Align.
3. Export COLMAP in flat layout, with undistorted images and a pinhole camera model (2.1.1+), at training size. Do not edit the images afterwards.
4. Train in LichtFeld. Use `--undistort` if the camera model is not pinhole. Consider `--exposure-correction`.

---

## 7) Where sources disagree, and what we could not confirm

**Disagreements**
- **Overlap:**
  - Epic mobile: about 70%
  - RealityScan desktop: more than 60%
  - Postshot and Polycam: 30–50%
  - SketchUp: 70–80% between consecutive frames

  Follow RealityScan. Video easily gives more overlap.
- **Frame rate:** Postshot says use the lowest frame rate. PlayCanvas says 60 fps or more for smooth motion. Epic and LichtFeld give no number.
- **Lens:** Teleport recommends the 0.5x ultra-wide (for its own app), and its blog says 3 fixed loops are not strictly needed with 0.5x. PlayCanvas warns that ultra-wide lenses under 24 mm cause distortion. Postshot says about 24 mm.
- **Hip-height tilt:** the Teleport Capture Paths page says tilt **up**. Its Movement page says angle **down** 30–45 degrees.
- **Phone orientation:** the Teleport Movement page says portrait. The Teleport blog says landscape for most interiors.
- **Detector sensitivity Ultra:** the current RealityScan help describes it as useful on weak texture. A 2017 knowledge-base post (anonymised author) says avoid Ultra because of false points from noise. https://forums.unrealengine.com/t/how-to-put-together-more-components/766678
- **Control points:** the same RealityScan page says both "6" and "at least 4". Use 6 or more.
- **LichtFeld maximum width:** the FAQ says 4096 in one place and 3840 in another. The code default is 3840.
- **KIRI** says 3DGS "handles featureless surfaces well". That is about rendering, not pose alignment, so it does not help RealityScan.

**Could not confirm**
- **No official values from Epic or LichtFeld for:** fps, shutter speed, rolling shutter, iPhone HDR, stabilization or lens. The Epic knowledge-base article "Video import and possible issues" would not render.
- **Unsourced numbers:** 1/250 s and the marker sizes are ours.
- **Blackmagic Camera behaviour:**
  - Whether locking the shutter freezes ISO, i.e. whether there is no shutter-priority or auto-ISO mode. Test this in the app.
  - Whether "Off" stabilization also disables the iPhone's optical stabilization. One SAE study, on still photos, says iPhone OIS "cannot be deactivated". LichtFeld's default calibration correction partly absorbs it.
- **What RealityScan "key frames only" means for iPhone HEVC files.**
- **The LichtFeld RealityScan tutorial video:** not watched, so its export settings are unknown.
- **Unverifiable third-party tips:**
  - Postshot's "1–2 minutes of video"
  - Luma's "disable HDR"
  - Polycam Smart Keyframing
  - Scaniverse's tips
- **Corrected claim:** Blackmagic 3.1.2 added a 1920x1440 Open Gate mode. The claim that full-sensor 4:3 is "only via ProRes RAW on iPhone 17 Pro" was wrong. The mode may not be in every colour space and may turn off stabilisation.
- **Light flicker:** no source covers flicker or banding under LED or fluorescent lights at fast shutter speeds. Check a test clip [ours].

---

## Top 5 changes vs the last capture (58 s, default camera, walking loop with turning, white walls, blinds half-open)

1. **Lock the camera.** Use Blackmagic Camera at 4K with the 1x lens only. Set shutter to 1/250 (locked), ISO manual, focus manual, white balance locked and stabilization Off. Last time the default app ran with auto exposure, auto focus, auto white balance and a cropping stabilizer.
2. **Move along the walls, don't turn.** Keep your back to the wall and walk sideways in 3 slow loops (eye, hip tilted up, overhead tilted down), with no more than 30 degrees between kept frames. Last time, turning in a loop gave panorama-like rotation that RealityScan cannot use.
3. **Give the white walls texture.** Add printed RealityScan markers (at least 100 px in frame, unique IDs), plus posters or objects. Film blank areas at an angle, never close and straight on.
4. **Close the blinds fully, and keep light even.** Do not film into windows.
5. **Film longer and choose the frames yourself.** Shoot about 3–6 minutes instead of 58 s. Extract about 3 fps with a sharpness filter (LichtFeld extractor) to get 500–1000 sharp frames, and import them as images. Do not rely on RealityScan's video import, which has no blur check.
