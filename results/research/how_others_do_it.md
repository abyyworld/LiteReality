# How others do it: shared scanned rooms, sharp splats, movable objects

Research date: 24 Nov 2026. Five sweeps (products, research papers, practitioners, shared-room systems, open-source repos). A second pass checked each one against its source.
Only items that held up are kept. Corrections are applied. Anything I could not check is marked **(unverified)**. Items that turned out wrong were fixed or dropped (Zillow SkyTour, Spline, Hyperfy, the Matterport patent and Polycam "Smart Keyframing" are gone).

"Official" means a product page, docs, a paper or the authors' own repo. "Community" means a third party, a hobby project or a press write-up.

---

## The short answer

- **Many people do parts of our job. Nobody does the whole thing.** No product or open repo combines a normal video call, a browser, a phone-scanned photoreal room, and shared server physics on objects cut from that same scan. Since this is a negative claim, word it carefully in the paper.
- **No shipping product lets people pick up a real scanned object.** Products keep the room splat fixed and put invisible boxes or meshes on top for collisions (Meta Hyperscape, HTC VIVERSE, World Labs demos, NVIDIA NuRec). Matterport and IKEA Kreativ can *erase* furniture from a scan, but they cannot move it.
- **Research does cut real objects out and move them.** It uses three steps we skip: (1) masks from the *same* capture, (2) filling the hole behind the object, (3) giving each object its own collider. GaussianNexus (UIST 2025) and VR-GS (SIGGRAPH 2024) are the closest.
- **Our blur is mostly a capture problem, not a trainer problem.** Every sharp-room recipe starts with blur-free frames: fast shutter, locked exposure, sharpest frames only, slow close passes.
- **Our 22.5 dB is roughly normal for phone rooms.** On held-out views of iPhone room captures, plain splatting gets 21.99 dB and DN-Splatter 21.89 dB (MuSHRoom, DN-Splatter paper, Table 3). One open-source author reports about 23.9 dB on his own iPhone room. So do not judge only by PSNR. Judge by eye too, and compare against a commercial result on the same data.
- **The white-room failure is expected.** Photo-only posing (RealityScan, COLMAP) fails on plain walls. Everyone who succeeds there uses device tracking (ARKit/LiDAR), LiDAR hardware, or tape on the walls. We already have the right approach (ARKit poses). Keep it.

---

## 1) Who already does this

| Name | What it does | Capture | Objects movable? | Multi-user? | Official? | Link |
|---|---|---|---|---|---|---|
| Meta Horizon Hyperscape Capture | Scan your room, get a photoreal splat world | Quest 3/3S headset: ~30 s coarse pass, then 5-10 min close-up walk. Cloud processing 1-8 h | No | Was up to 8 people (Quest + phone guests) from Nov 2025. **Sharing ended in 2026.** Now view-only | Official (Meta) + UploadVR | https://uploadvr.com/horizon-hyperscape-worlds-invite-friends-visit-meta-avatars , https://communityforums.atmeta.com/blog/announcementsblog/updates-to-your-meta-quest-experience-in-2026/1369435 |
| Varjo Teleport | Phone scan to cloud splat, share by link, PC VR | iPhone app, 5-10 min walk, ~500 photos for a 5x5 m room (cap 2000) | No | No (shared viewing only) | Official | https://get.teleport.varjo.com/blog/how-to-create-best-quality-3d-scans-of-your-environment-with-teleport |
| Niantic Scaniverse (+ SPZ format) | Splat trained on the phone in ~1 min | Any iPhone A12+ or Android; warns past 180 s | No | No | Official | https://github.com/nianticlabs/spz |
| Polycam | Cloud splats from photos or video | Phone photos/video | No (crop only) | No | Official | https://learn.poly.cam |
| Matterport Defurnish | Erases furniture from a scanned home, fills the hole | Matterport camera or phone, mesh not splat | Removal only | No | Official | https://matterport.com/blog/all-we-want-is-an-empty-room-how-emptying-your-digital-twin-unlocks-design |
| IKEA Kreativ | Erase real furniture from an iPhone room scan, place IKEA items | iPhone LiDAR room scan | Removal only | No | Official | https://www.ikea.com/us/en/customer-service/knowledge/articles/e86d70g6-3673-4306-b373-20f7cg7fd5ed.html |
| HTC VIVERSE + PlayCanvas guide | Room splat becomes a multiplayer web world | Room video via Luma or Polycam | No (static colliders, click triggers) | Yes (world URL, avatars) | Official | https://news.viverse.com/post/transform-your-room-into-a-playable-virtual-space-with-gaussian-splatting-and-playcanvas-in-viverse |
| World Labs Robot Roommates | Browser multiplayer splat world (Spark + Rapier + Socket.IO) | None. The world is AI-generated (Marble) | Shared objects are ordinary meshes, not cut from a splat | Yes | Official showcase | https://www.worldlabs.ai/labs/showcase/robot-roommates |
| MuJoCo-GS-Web | MuJoCo (WASM) + Spark in the browser; drag any physics body; bodies hide the splat behind them | None. Uses a splat file plus a hand-written MuJoCo XML | Only separately defined bodies | No (single browser) | Official showcase (individual author) | https://www.worldlabs.ai/labs/showcase/mujoco-gs-web , https://github.com/Vector-Wangel/MuJoCo-GS-Web |
| Arrival.Space | Social browser/VR platform for splat worlds | Upload a splat, or train in the browser (Splat.js) | No | Yes (invites, avatars, spatial audio) | Community source (Radiance Fields page) | https://radiancefields.com/platforms/arrival-space |
| GaussianNexus (UIST 2025) | 1 AR user + 1 VR user share a pre-scanned splat room. Real objects being moved are tracked and their splats follow | ~3 min video, 300-400 frames, COLMAP, standard 3DGS | Yes, but they follow **real** objects being moved, not simulated ones | 2 users (HoloLens 2 + Quest 3) | Official (paper PDF) | https://dl.acm.org/doi/10.1145/3746059.3747693 , https://xincheng.me/files/7-gaussiannexus.pdf |
| VR-GS (SIGGRAPH 2024) | Grab and physically play with objects cut from a real splat in VR | Consumer-camera video, COLMAP | Yes (cut out, hole inpainted, XPBD physics) | No | Official | https://arxiv.org/abs/2401.16663 |
| DecoupledGaussian (CVPR 2025) | Separates an object from the floor it sits on, repairs both, then simulates | In-the-wild video, COLMAP | Yes (MPM physics) | No | Official | https://arxiv.org/abs/2503.05484 |
| HoloScene (NeurIPS 2025) | Scanned room becomes a physics scene with completed furniture | One video + poses | Yes (furniture rearranged in an Unreal demo) | No | Official | https://github.com/xiahongchi/HoloScene |
| Video2Game (CVPR 2024) | Video to a browser game where real objects can be pushed | One video, COLMAP | Yes (mesh + Cannon.js colliders) | No | Official | https://github.com/video2game/video2game |
| DISCOVERSE (IROS 2025) | MuJoCo + splats for robot learning | Room splat (LiDAR-regularised) + separately scanned objects | Yes (objects are separate assets posed by MuJoCo) | No | Official | https://github.com/discoverse-dev/DISCOVERSE |
| auto-splat-seg | Splits a finished room splat into ~35 object splats + background | Input: one room .ply | Splits only, no physics, no hole fill | No | Community | https://github.com/alexmoed/auto-splat-seg |
| Morpheus (hackathon) | Room splat to MuJoCo box colliders; objects masked and swapped for catalogue meshes; Spark viewer | Room splat + collider mesh | Swaps in catalogue meshes, does not move the real splat | No | Community | https://github.com/bede-lau/Morpheus |
| LiteReality-Agent | iPhone RGB-D + RoomPlan scan to MuJoCo bodies with mass, colliders and joints | LiteReality Scanner app, one walk | Yes, but as generated meshes, not photoreal splats | No | Official | https://github.com/LiteReality/LiteReality-Agent |

---

## 2) How the people with SHARP rooms capture

### The common recipe (sources agree)

1. **No blurry frames.** This matters more than which trainer you use.
   - Use stills, interval stills, or video with a forced fast shutter. Raise ISO instead of letting the shutter slow down. Postshot's guide: noise hurts less than blur.
   - Then throw away the blurry frames. Score each frame with a sharpness measure and keep the sharpest few hundred.
2. **Lock exposure, white balance and focus. Turn off stabilisation.** Every guide says this. On the trainer side, per-image colour correction (bilateral grid or PPISP) cleans up what is left.
3. **Good poses.** On plain white rooms, phone tracking (ARKit + LiDAR) or LiDAR hardware is what works. A 2026 study found that re-running COLMAP bundle adjustment on ARKit room poses made them *worse* in all 15 ScanNet++ rooms (0.55 to 0.74 degrees mean error; arXiv 2608.21008). Keep ARKit poses and let the trainer nudge them.
4. **A planned path, not a free walk.**
   - Start with a wide loop around the room edges at chest-to-shoulder height, aimed at the centre.
   - Add lower and higher loops only where they help: under furniture, shelves, ceiling.
   - Then orbit the key objects. Keep at least 30 cm away.
   - Shoot plain surfaces and glass at an angle, with some nearby structure in frame.
   - Walk slowly. Close the loop where you started.
5. **Many frames, full resolution.** Teleport: ~500 photos for a 5x5 m room. Magnopus: 500-900 is "medium", 1000-2000 is "high". A film crew used ~4000 interval stills for one dim bedroom.
6. **Keep the scene still.** A door that moves during capture *will* smear. Hyperscape reviewers saw the same thing: an object moved during the scan leaves a halo.

### The 2-3 best concrete recipes

**Recipe A: phone + LiDAR + ARKit poses with a fast shutter. Closest to ours, smallest change.**
- **SplatKing** (iPhone app; made by the Radiance Fields founder; community source): manual ISO, shutter, white balance and focus lock. Its "Minimum Shutter Speed" raises ISO to keep frames sharp. LiDAR mode exports a COLMAP text model on the phone, with no SfM step, ready for LichtFeld, Postshot or Brush. https://radiancefields.com/splatking-1.0.2-native-colmap-export-from-lidar-a-redesigned-capture-interface-and-expanded-manual-controls
- **Splatcatcher** (Warpgate, free, iPhone 12 Pro+, iOS 26+; official App Store listing): ARKit poses, live coloured LiDAR point cloud, blur rejection while capturing, loop-closure guidance, optional on-phone LiDAR pose refinement, COLMAP export. https://apps.apple.com/app/id6759800588
- Then run **Sharp Frames** (`pip install sharp-frames`; Laplacian + Tenengrad scoring, default keeps the best 300) https://github.com/reflct/sharp-frames-python . Train in LichtFeld or gsplat with pose refinement and bilateral grid turned on.
- Reference number: the Splatify-Anything author's own iPhone room (715 train / 143 held-out photos, PPISP) scores about 23.9 dB, or 25.35 dB after colour alignment. https://github.com/KuoFengYuan/splatify-anything

**Recipe B: the professional stills recipe (Magnopus, Allee Willis House; official blog).**
- Mirrorless camera, one prime lens per room, still photos (not video). Faster shutter or a monopod, narrow aperture, RAW, polarizer for reflections, 70-80% overlap.
- Domes or circles at several heights. Half-shell arcs where furniture blocks a full circle.
- **Temporary black dots or tape on flat white walls**, so posing has something to lock onto.
- RealityScan for poses, then Postshot, Nerfstudio or Polycam to train. Flat untextured walls still come out "cloudy".
- https://www.magnopus.com/blog/lessons-in-gaussian-splats-from-the-allee-willis-house

**Recipe C: the guided product path (Varjo Teleport + Meta Hyperscape).**
- Teleport's official guide: wide loop at chest-to-shoulder height, extra loops only where useful, close-ups at least 30 cm away, oblique shots of plain surfaces, 0.5x wide lens by default, 250-1000 frames.
- Hyperscape: a 30 s coarse pass, then 5-10 min holding the headset close to every surface. A reviewer reported dim areas, plain ceilings and mirrors as weak spots (reviewer's experience, not Meta's numbers).
- Teleport pricing (official help centre): 5 free scans with no PLY export; EUR 29.99/mo for 15 scans **with PLY export**. That makes it usable as a quality ceiling on our own room.

### What does NOT make rooms sharper
- **360 cameras.** They cover a room in 1-2 min, but experienced users say the result is lower quality. VFX Soup's verdict: "significantly more difficult ... with a lower quality result" (YouTube, Jun 2026). Use one only for coverage, never as the main source.
- **Depth/normal priors alone.** DN-Splatter (Apache-2.0) makes geometry, walls and meshes much cleaner. It does **not** raise PSNR (MuSHRoom: 24.58 vs 24.67 within-trajectory). It helps floaters and fog on white walls, not crispness.
- **Wrong intrinsics.** 360 Splat Pro found that a wrong lens centre gives a "soft splat" even when the alignment report looks clean. Check that our Stray/ARKit intrinsics are scaled to the RGB resolution we actually train on.

---

## 3) How others make objects movable without leaving a copy behind or grabbing the neighbour

### Why we get a copy behind
We cut chairs with RoomPlan boxes from a *different* capture. Boxes don't line up with the splat, and the floor under the chair was never seen. Everyone who does this well does three things:

**Step 1. Masks from the SAME capture, lifted onto each Gaussian.** This fixes the copy and the neighbour.
- **GaussianNexus** uses SAGA, prompted by YOLOv11 boxes on the same capture, with some hand cleanup. Each object's Gaussians are pulled out of the room. They do **not** fill the hole, and they report leftover floaters that need manual cleanup. Each object gets a rough collision box from its splat points.
- **SAGA** (Apache-2.0, but ships the Inria non-commercial rasterizer): click a chair to select it. A *scale* knob separates "one chair" from "two chairs". It works on an already-trained splat. https://github.com/Jumpat/SegAnyGAussians
- **Gaussian Grouping** (Apache-2.0 + Inria rasterizer): SAM masks tracked through the video, then a 16-number ID on every Gaussian. https://github.com/lkeab/gaussian-grouping
- **ObjectGS** (MIT, built on a gsplat fork; the cleanest licence): trains the room already split by object, so nothing has to be cut afterwards. https://github.com/RuijieZhu94/ObjectGS
- **InstaScene** (no licence): its rules for neighbours are worth copying. Throw away any mask that covers two tracked objects, merge masks only when views agree above 0.9, and clean each object with DBSCAN. 85.6 mIoU vs 58.1 for Gaussian Grouping on LERF-Mask. https://github.com/zju3dv/InstaScene
- **COB-GS** (Inria rasterizer): splits Gaussians that straddle a mask edge. Big Gaussians spanning two objects are exactly what makes neighbours stick together. https://github.com/ZestfulJX/COB-GS
- **auto-splat-seg** (MIT, needs a 48 GB GPU): renders views of our own splat, runs SAM3 on them, votes in 3D, drops floor points, and subtracts objects from the background. https://github.com/alexmoed/auto-splat-seg
- **FlashSplat** (ECCV 2024): assigns 2D masks to an already-trained splat in one fast solve, with no retraining. https://github.com/florinshen/FlashSplat
- **Manual fallback:** **SuperSplat** (MIT). Select a chair with box, sphere or brush, use "Separate" to make it its own splat, and export. Minutes per chair. https://github.com/playcanvas/supersplat
- Warning: *semantic* or *language* features (LangSplat, Feature 3DGS) do **not** separate two identical chairs. You need *instance* features.
- Warning: GARField (MIT) needs its own trained NeRF checkpoint first. It does not drop straight onto our splat.

**Step 2. Fill the hole where the chair stood.**
- **Cheap:** Gaussian Grouping and VR-GS. Render the hole mask in every view, inpaint each view with LaMa, then optimise a small patch of new Gaussians there.
- **Cheapest for our case:** **Splat Repair** (Apache-2.0). Its "Fast Fill" copies nearby splats in 0.3-1.5 s on CPU and works best on flat walls and floors, which is exactly our plain room. "Deep Fill" uses LaMa + gsplat (~20 s on an RTX 4090). https://github.com/NicolasDrum/splat-repair
- **Better:** **DecoupledGaussian** fills the *geometry* of the floor and the chair's underside first (Poisson fields), then fixes the colour with 2D inpainting. Scene restoration 27.32 dB vs 25.13 for VR-GS. It has no licence, uses MPM physics, and bundles Inria code, so **copy the idea only**. **AuraFusion360** fills depth first, then inpaints one reference view and spreads it, to avoid flicker.
- **Runtime fallback:** Spark's **SplatEdit SDF** and its `examples/interactive-holes` demo can hide leftover splats in the old footprint in the browser. https://github.com/sparkjsdev/spark

**Step 3. Give each object its own collider, and move its Gaussians with the physics pose.**
- Nobody uses raw splats for collisions. They pair each object's splats with a simple shape:
  - DISCOVERSE: convex decomposition in MuJoCo.
  - Video2Game: box or convex shapes in Cannon.js.
  - DRAWER (MIT): an SDF mesh with Gaussians bound to its faces.
  - Splatting Physical Scenes: Gaussians on mesh faces, with MuJoCo MJX.
- Tools: **chitin** (MIT; CoACD convex hulls from Gaussians, runs in the browser) https://github.com/Autarkis/chitin , and **collider-forge** (MIT; Poisson collider from splats) https://github.com/hh-hang/collider-forge . For the room: PlayCanvas **splat-transform `--collision-mesh`** (open source) https://github.com/playcanvas/splat-transform , or a TSDF mesh from our LiDAR depth.
- Moving the Gaussians: SplatSim applies the physics pose to each object's Gaussians (position, plus a rotated covariance) https://github.com/qureshinomaan/SplatSim . **Our own engineering note** (not SplatSim's claim): also rotate the colour (SH) coefficients, or lighting will look wrong when a chair turns. Spark already lets each `SplatMesh` take its own transform.

**The other way: never bake movable objects into the room.**
- DISCOVERSE and RoboGSim scan each object *separately* and place it into a room that has no copy of it.
- For us: capture each chair with a short orbit (including the underside), and capture or inpaint the empty floor.
- VFX Soup's SuperSplat snap-alignment tool merges two separately trained splats. Using it for "room + each chair" is our idea, not something the video shows. https://github.com/VFX-Soup/supersplat-snap

**Complete the hidden back of a moved chair** (optional, bigger job):
- InstaScene (MVDFusion completion).
- HoloScene (Wonder3D completion + a stability check in the simulator; Apache-2.0).
- Meta **SAM 3D Objects** (one image + mask gives a full object as a splat .ply; gated weights, custom SAM licence) https://github.com/facebookresearch/sam-3d-objects

### Licence warning (matters if we host this)
- The original Inria gaussian-splatting rasterizer is **non-commercial**. Many "Apache/MIT" research repos still ship it: Gaussian Grouping, SAGA, COB-GS, DecoupledGaussian, PhysGaussian, SplatSim. InstaScene and AuraFusion360 ship the 2DGS rasterizer, which uses the same licence.
- Clean bases: gsplat (Apache-2.0), Brush (Apache-2.0), Spark (MIT), SuperSplat (MIT), ObjectGS (MIT), DN-Splatter (Apache-2.0), DRAWER (MIT), Video2Game (MIT), HoloScene (Apache-2.0), auto-splat-seg (MIT), Splat Repair (Apache-2.0).
- LichtFeld is GPL-3: fine as a separate offline tool, risky to link into a service. OpenSplat is AGPL-3: avoid it in a hosted pipeline.
- VGGT, MASt3R and InstantSplat (SfM-free posing) carry restrictive licences.

---

## 4) The closest systems to ours and how we differ (for the paper)

| System | What it shares with us | How we differ |
|---|---|---|
| **GaussianNexus** (UIST 2025) | Shared real scanned splat room. Objects cut from the *same* scan (SAGA) and moved as rigid Gaussian sets. 0.95 s end-to-end latency, n=9 study | Their objects follow **real** objects tracked by a 360 camera at ~2.33 fps. Ours are moved by **remote people through server physics**. They support 2 headset users (AR + VR); we run in a browser on a normal video call with any number of users. They do not fill holes. Note: "objects cut from the same scan" alone is **not** new. |
| **Meta Hyperscape** (social version, 2025-26) | Friends hang out in a photoreal scanned room; phone guests allowed | Static only. No movable objects. Closed. Headset capture. **Multi-user sharing was discontinued in 2026**, so this is an open gap we can point to. |
| **VR-GS** (SIGGRAPH 2024) | The best published recipe for "grab a real object out of a splat with physics": per-Gaussian IDs, LaMa hole fill, collision mesh, XPBD. Physics beat plain dragging on realism (6.1 vs 4.8, n=10) | Single user, Unity, headset. We are multi-user, browser-based, with MuJoCo on a server. |
| **World Labs Robot Roommates / STARSPEED / MuJoCo-GS-Web** | Same web stack (Spark + physics + socket sync). MuJoCo-GS-Web even uses MuJoCo with Spark | Worlds are AI-generated, not real scans. Shared objects are ordinary meshes. MuJoCo-GS-Web runs in one browser with no multi-user. Whether Robot Roommates' server is authoritative is not stated. |
| **DISCOVERSE / SplatSim / Splatting Physical Scenes** | The physics engine drives each object's splat pose, the same loop as ours (DISCOVERSE uses MuJoCo) | One robot or policy, not people. Objects are scanned separately or come from datasets, not cut from a phone room scan. |
| **Video2Game** (CVPR 2024) | Video to a browser world with per-object physics | Single user. Baked meshes, so less photoreal. Objects split by semantic labels or user boxes. |
| **HoloScene / DRAWER** | Scanned rooms with movable furniture and physically checked layouts | Offline. They run in Unreal or Isaac. HoloScene's game uses textured meshes, not splats. No multi-user. |
| **Teleop papers** (Reality Fusion IROS 2024, n=24; Locomanipulation 2025, n=15) | Splat rooms improve remote spatial understanding | Both splats are static. The locomanipulation authors list "you can't see the grabbed object move" as a limitation. This is a ready-made motivation for our design. |
| **CrossGaussian** (ISMAR 2025 poster) | Remote collaboration in a splat room, with live 360 video blended in | No object moving. Their study of update delays is unverified. |
| **IKEA Kreativ / Matterport Defurnish** | Take a real object out of a scanned room and fill the hole | Removal only, no moving, no physics, no shared session. |

**Novelty sentence we can defend:** *a normal video call* + *browser* + *phone-scanned photoreal room* + *many users* + *server-authoritative physics on real objects cut from that same scan*. No system found does all of these. Also cite that no controlled study of multi-user co-presence in splat rooms was found. That is a second gap.

---

## 5) What we should copy, ranked by impact per effort

Problems: **P1** blurry room and door smear, **P2** copy left behind, **P3** neighbouring chairs grabbed together, **P4** RealityScan cannot pose the white room.

| # | Copy this | From | Fixes | Effort | Changes our method? |
|---|---|---|---|---|---|
| 1 | **Sharp capture.** Lock exposure, focus and white balance. Force a fast shutter (raise ISO). Walk slowly. Keep the door completely still (or mask it). Wide chest-height loop, then low and high loops, then close orbits (≥30 cm) of each chair and the door. Stray Scanner is open source (MIT, https://github.com/strayrobots/scanner ), so we can patch it, or switch to SplatKing or Splatcatcher, which keep ARKit/LiDAR poses and export COLMAP | Teleport guide, Postshot guide, Magnopus, SplatKing, Splatcatcher | P1 (main fix), P4 | Low | Capture protocol only. Keeps our ARKit-pose method. |
| 2 | **Keep only the sharpest frames** (Sharp Frames, best ~300-800 frames, full resolution) and **train on all of them** (do not hold out frames by accident: nerfstudio `--eval-mode fraction --train-split-fraction 1`) | Sharp Frames, Postshot "Use Best Images", nerfstudio issue #3421 | P1 | Low | No |
| 3 | **Turn on trainer options we already have.** In gsplat: `pose_opt` (nudges ARKit poses), `post_processing=bilateral_grid` or `ppisp` (phone auto-exposure), MCMC with a higher splat cap, `antialiased`, `depth_loss` from LiDAR (cleaner walls, fewer floaters). In LichtFeld: `--pose-opt`, bilateral grid, MRNF/MCMC. Note: gsplat's `with_ut` does **not** fix rolling shutter; that needs custom code using `rasterization(rolling_shutter=...)` | gsplat, LichtFeld, DN-Splatter, Google Immersive View (per-image appearance) | P1, P4 | Low | No. Same pipeline, different settings. |
| 4 | **Get a quality ceiling on our own data.** Train the same frames in Postshot (imports COLMAP poses; images must be undistorted) or Teleport paid (PLY export). If they look much sharper, the gap is our trainer settings. If not, it is our capture | Postshot User Guide, Teleport | P1 (diagnosis) | Low | No |
| 5 | **Cut chairs with masks from the SAME capture, never with RoomPlan boxes.** Now: SuperSplat "Separate" by hand. Next: SAGA-style click-to-select (as GaussianNexus does), ObjectGS (MIT) or the auto-splat-seg method. Use RoomPlan boxes only as prompts | GaussianNexus, SAGA, ObjectGS, auto-splat-seg, SuperSplat | P2, P3 | Low (manual) to Medium (auto) | **Yes.** RoomPlan stops being the cut. |
| 6 | **Neighbour rules.** Discard masks that cover two objects. Split Gaussians that straddle a mask edge. Clean each object with DBSCAN. Use a scale knob for "one chair" | InstaScene, COB-GS, SAGA | P3 | Medium | Adds a step |
| 7 | **Fill the floor and wall behind each chair.** Splat Repair Fast Fill (flat floors, ~1 s) or a LaMa-patch pass. Hide leftovers in the browser with Spark SplatEdit SDF | Splat Repair, Gaussian Grouping, VR-GS, DecoupledGaussian (idea), Spark | P2 | Medium | Adds a step |
| 8 | **Per-chair collider + pose loop.** CoACD convex hulls (chitin) per chair as MuJoCo geoms. Room collider from splat-transform `--collision-mesh` or a LiDAR TSDF mesh. Send only each body's 6-DoF pose; Spark moves that chair's `SplatMesh`. Rotate SH colours with the chair | DISCOVERSE, chitin, splat-transform, SplatSim, Streaming Gaussian Proxies paper | P2 (no drift between look and physics) | Low-Medium | Mostly matches ours already |
| 9 | **Scan chairs separately** (short orbit incl. underside) and compose them into a room scanned without them, or with the floor inpainted | DISCOVERSE, RoboGSim, Apple Object Capture (its WWDC23 example is a textureless chair) | P2 (cleanest result) | Medium (more capture time) | **Yes.** Changes capture. |
| 10 | **Ship lighter files.** SPZ (MIT, ~10x smaller than PLY by README claim) or Spark 2.0 level-of-detail streaming, so phone guests on a call load fast | Niantic SPZ, Spark 2.0, PlayCanvas SOG | Call experience | Low | No |
| 11 | **Model motion blur instead of only dropping it.** Spectacular AI 3dgs-deblur (Apache-2.0) uses phone velocity; BAD-Gaussians (Apache-2.0) | Gaussian Splatting on the Move (ECCV 2024), BAD-Gaussians | P1 (leftover blur) | Medium-High | Trainer change |
| 12 | **Complete the chair's unseen back** with a generative model, and check it rests stably in MuJoCo before shipping | InstaScene, HoloScene, SAM 3D Objects | P2 (looks complete when lifted) | High | Adds a step |

**Bottom line for the user:**
- Items 1-4 are cheap. They keep our method and target the blur.
- Items 5-8 are the real fix for the ghost copy and the double grab. They mean one method change: stop cutting with RoomPlan boxes, and cut with masks from the same capture.
- Item 9 is the cleanest fix for the copy, at the cost of a slightly longer capture.

---

### Unverified or caveated (kept out of the ranking)
- Hyperscape: "72/90 fps streaming", "only the chunk for the predicted head pose", and semantic per-chair colliders shipping in the 2025 beta are **unverified**. The proxy-mesh idea came from a 2024 interview about an early demo.
- Polycam frame-selection and price details, KIRI version history, Scaniverse 360-video limits, 3DVF capture numbers (20 cm above head, 1 m/s, 1 m from walls), ScanNet++ terms, CrossGaussian's delay conditions, NuRec details: **unverified**.
- Marble's Free plan **cannot** export splats. Marble plan prices are secondary-source only.
- Luma: the Flythroughs app shut on 1 Jan 2026. The Interactive Scenes page now redirects, and the luma-web-library repo was archived in Apr 2025. Do not depend on Luma.
- XGRIDS PortalCam (LiDAR + 4 cameras, the pro fix for white rooms): official store price USD 4,999.
- Arcturus Vision's official script (`avcam_to_3dgs.py`) uses COLMAP + LichtFeld, the same trainer we use. It is a reference script, not a product we need.
