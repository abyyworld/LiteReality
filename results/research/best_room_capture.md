# Best room capture: what actually gives the sharpest splats

Date: 2026-11-24. Built only from checked sources. Each number has its source URL.
Tags: **[official]** = paper, dataset page, maker's own page or docs. **[community]** = reseller, press site, marketplace or forum.
Where a check corrected a claim, the corrected version is used. Claims that could not be checked are marked **(unverified)**.

---

## 1. The honest answer

Nothing is guaranteed. No paper captures the same room with a phone, a mirrorless camera, a 360 camera and a scanner, and scores them all on the same test. No paper measures how much white walls hurt splats. The two same-room device comparisons are ScanNet++ (DSLR vs iPhone) and MuSHRoom (Kinect vs iPhone). But one recipe comes closest to guaranteed, because it has been repeated on hundreds of real rooms. It is the ScanNet++ recipe: **still photos (not video) from a full-frame mirrorless camera with a very wide or fisheye lens, exposure, white balance and focus locked, shot static, plus a laser scan that pins the camera poses so blank walls cannot break them.** ScanNet++ ran this on 460 scenes (about 280k DSLR images) at release, and v2 has 1000+ scenes [official, https://arxiv.org/html/2308.11417 and https://scannetpp.mlsg.cit.tum.de/scannetpp/]. On the same rooms, training on these stills beats training on iPhone video by about 2.3 to 7.3 dB, depending on the method. Our phone result is already in the range of the best published phone results. More training tricks on phone video are worth about 1 to 2 dB. So the upgrade is a change of capture, not a better app. It does not have to be an app.

---

## 2. Ranked options

PSNR numbers from different datasets use different test protocols. Do not compare them across rows as if they were one scale. The cleanest same-room comparison is ScanNet++ paper Table 3 vs Table 4.

| # | Setup | Quality evidence (exact numbers + source) | White walls? | Cost (buy / rent) | Time per room | Fits our pipeline? | Fits "anyone with a phone"? |
|---|---|---|---|---|---|---|---|
| 1 | **Full-frame mirrorless + fisheye/very wide lens, static stills, locked exposure, plus terrestrial laser scan to pin poses** (ScanNet++ recipe) | Same rooms, DSLR test images. Nerfacto: 25.02 / 0.858 / 0.180 (DSLR-trained) vs 17.70 / 0.755 / 0.300 (iPhone-trained) (PSNR / SSIM / LPIPS) [official, paper Tables 3-4, https://arxiv.org/html/2308.11417]. Leaderboard, 50 scenes: plain 3DGS 23.893, Zip-NeRF 25.041, best 25.361 [official, https://scannetpp.mlsg.cit.tum.de/scannetpp/benchmark/nvs]. Test views are off the capture path, which is a hard test. | Yes, the best documented fix. The paper picks fisheye because "indoor scenes can have large textureless regions (e.g., walls, cupboards)" and adds laser-scan pseudo images to COLMAP. | Rent (US, Lensrentals list price, before coverage and shipping): Sony a7 IV $156 / 7 days or $113 / 3 days; Sigma 15mm f/1.4 fisheye $113 / 7 days or $79 / 3 days; Laowa 8-15mm fisheye $50 / 7 days [official listings]. Leica BLK360 G2 scanner $250/day or $1,000/week plus $315 Register360 7-day licence [community, Kwipped]. | ScanNet++: "around 30 minutes on average" per scene for the whole multi-device capture, "upwards of 2 hours" for large scenes [official]. | Yes. COLMAP poses + images, gsplat trains directly. gsplat has 3DGUT for fisheye [official, gsplat README]. Laser scan also gives metric geometry for MuJoCo. | No. It is the gold standard and ground truth, not the shipping path. |
| 2 | **Same camera, photos only**, with temporary coded targets or tape on blank walls, poses from COLMAP / Metashape / RealityScan | No splat PSNR for the targets themselves. The camera side is the same as row 1. Mip-NeRF 360 indoor scenes (Fujifilm X100V stills, locked settings, 100-330 photos): 3DGS room 30.632, counter 28.700, kitchen 30.317, bonsai 31.980 [official, https://ar5iv.labs.arxiv.org/html/2308.04079]. These are textured, inward-facing scenes, so they are easier than our atrium. | Partly. A wide FOV gives "larger overlap between images" (ScanNet++). Targets act as matches (Agisoft [official, https://agisoft.freshdesk.com/support/solutions/articles/31000148855]). Magnopus uses "black dots or tape" [official studio blog]. Not guaranteed: ScanNet++ still needed the laser scan. Targets get baked into the splat unless masked or removed. | Camera + lens about $192 for 3 days or $269 for a week (Lensrentals, before coverage and shipping). Coded targets need Metashape Pro (about $3,499 per reseller listings, unverified). RealityScan has no coded-target cost. | Shooting about 30-90 min, depending on image count. | Yes. | No. |
| 3 | **Multi-camera HDR rig** (Meta Eyeful Tower: 22 Sony a1, 12 mm, f/8, ISO 500, 9-shot brackets) | Overall (pinhole + fisheye scenes): VR-NeRF 31.01, Zip-NeRF 31.75, HybridNeRF 31.57. Pinhole scenes only: 3DGS 27.42 vs Zip-NeRF 29.71 [official, https://arxiv.org/html/2312.03160]. Held-out views along a dense rig path, so an easier test than ScanNet++. | Yes, by brute force: rigid rig calibration, stops about 30 cm apart, scale bars, Metashape poses [official, https://arxiv.org/html/2311.02542]. | Not for sale or rent. Very expensive (estimate, unverified). The dataset is free and MIT-licensed since March 2026 [official, https://github.com/facebookresearch/EyefulTower]. | "about one minute per m²" [official]. | The dataset ships COLMAP exports, so we can benchmark our trainer on it. | No. |
| 4 | **LiDAR-SLAM splat scanner** (XGRIDS PortalCam) | **No published PSNR anywhere.** One reseller comparison, qualitative only ("higher resolution, finer detail, and fewer artifacts" than the Lixel L2 Pro) [community, Heliguy]. Cameras: 4000x3000 (12 MP), 1/2-inch CMOS, rolling shutter [official, https://store.xgrids.com/products/portalcam]. Phone-class sensors. | Probably yes for poses, since LiDAR SLAM does not need wall texture. No plain-wall test exists. | Buy: $4,999, sold out [official store]. Rent: $1,009/week (Kwipped, US), £355/week inc VAT (Heliguy, UK), $300/day (Radiance Fields, NYC, from a search snippet) [all community]. | Heliguy: "up to 200 m² in 15 minutes" [community]. Processing about 5 h (Kwipped listing) [community]. | Weak. LCC Studio exports .lcc, .lcc2, .ply, .usd, 3D Tiles [community]. No documented COLMAP or image+pose export, so we cannot retrain or score held-out frames. The .ply could load in Spark as a demo. | No. |
| 5 | **360 camera** (Insta360 One RS 1-inch on tripod, fixed settings) | FIORD, vanilla 3DGS on indoor scenes: 23.33, 25.47, 26.28, 27.41, 27.15 [official, https://arxiv.org/html/2504.01732v2]. Own protocol, so not comparable to row 1. No same-room test against mirrorless. Paper warns of "ghosting artifacts along stitching lines". | Partly: very wide FOV, but poses still come from image features. | Insta360 X5 rent $60 / 7 days (Lensrentals) [official listing]. | Not reported. | Yes, via fisheye or cubemap + COLMAP. | Close to it (cheap consumer device). |
| 6 | **iPhone LiDAR video** (our current setup) | ScanNet++ iPhone leaderboard (train on iPhone, test on DSLR frames, colour-corrected): 3DGS 19.186; 3DGS + exposure compensation 20.027; best splat entry NeGS 21.572; best overall DVSM-480p 22.716 (not a splat) [official, https://scannetpp.mlsg.cit.tum.de/scannetpp/benchmark/nvs_iphone]. Meta, 49 ScanNet++ iPhone scenes scored on held-out iPhone frames: no compensation 21.71, 3DGS + colour + pose fitting 23.87, with blur modelling 24.08 [official, https://arxiv.org/html/2404.04211v1]. Blur + rolling-shutter compensation: +1.68 dB on average, in-sequence test [official, https://arxiv.org/html/2403.13327]. Ours: about 22.5 dB, colour-corrected, on our own held-out frames, so not directly comparable to any of these. | ARKit tracked our white room. ScanNet++ says "blurriness and a small field-of-view can cause misalignments" in SfM poses. | Already owned. | About 10-20 min. | Yes (this is the pipeline). | **Yes.** This is the shipping path. |
| 7 | **Matterport Pro3** | No splat feature or export [official, https://matterport.com/en-gb/cameras]. | LiDAR geometry is fine, but there are too few viewpoints for sharp splats. | Rentals "as low as $149" [official]. | "<20 seconds per sweep" [official]. | Poor. E57 point cloud only. | No. |

Notes on the table:
- The DSLR-vs-iPhone gap depends on the method. Nerfacto (paper): 7.3 dB. Vanilla 3DGS (leaderboard): 23.389 vs 19.186 colour-corrected, 4.2 dB. Best entries: 24.975 vs 22.716, 2.3 dB. Only the paper's Table 3 vs Table 4 is a clean same-scene comparison. The leaderboard gap mixes capture quality with colour and intrinsics mismatch, so treat it as an upper bound. It is also unconfirmed that the two 12-scene leaderboard sets are the same rooms.
- Two checks disagree on whether the paper's Table 4 iPhone numbers are colour-corrected. The paper text says "We perform color correction to compensate for color differences between the iPhone and DSLR captures". The Table 4 caption does not say so. Treat the 7.3 dB as possibly including colour mismatch.
- Correction: the Azure Kinect colour camera is rolling-shutter, not global-shutter [official, https://learn.microsoft.com/en-us/azure/kinect-dk/hardware-specification]. MuSHRoom's Kinect-over-iPhone win (different-sequence Splatfacto: 23.28 vs 21.39) is best explained by fixed exposure and white balance, not shutter type [official, https://arxiv.org/html/2311.02778].
- Correction: 3DGUT's "about 6 dB from fisheye-native training" (29.11 vs 22.76) is not a same-pixel gain. The 3DGS baseline was scored on fisheye periphery it never trained on [official, https://arxiv.org/html/2412.12507]. What it does show: cropping fisheye images to pinhole throws away coverage.
- A fisheye is not proven to be required. Mip-NeRF 360 reached about 30 dB with a roughly 35 mm-equivalent lens on textured scenes. Eyeful's v2 rig used a rectilinear 12 mm. What the evidence supports: a large field of view for plain rooms, plus static, locked-exposure stills.

---

## 3. The recommended plan

**(a) Gold standard capture of the atrium (upper bound and ground truth).**
- Rent a full-frame mirrorless body and a fisheye or very wide lens for 3 days. Shoot static stills with locked exposure, white balance and focus. Follow the recipe in section 4.
- On the same day, rent a tripod laser scanner (Leica BLK360 G2 is the cheapest one found). Scan the atrium from 4+ positions. This gives (1) poses that do not depend on wall texture, and (2) a metric point cloud as ground-truth geometry for accuracy numbers and MuJoCo collision.
- Shoot 15-25 extra **test stills at new positions and heights**, off the training path. Never train on them.

**(b) The everyday phone capture we ship.**
- Keep iPhone + ARKit poses + gsplat. It is the only option that fits "anyone captures with a phone".
- Apply the cheap fixes the evidence supports: lock exposure and white balance; slow, steady movement; per-image exposure compensation (leaderboard 3DGS 19.186 to 20.027 with it); blur-aware frame selection or blur compensation (about +1 to +2 dB, in-sequence tests). Expect about 1-2 dB from these, not a jump to gold-standard quality.

**How the paper uses both.**
- Score **both** captures against the **same mirrorless test stills**. This copies the ScanNet++ iPhone track protocol, so our numbers can be set next to theirs with care.
- Report: gold standard = upper bound; phone = what ships; the gap between them = the honest cost of "anyone with a phone".
- Use the laser scan as ground truth for geometry accuracy (object sizes, collision boxes) for both captures.
- Optionally ship the gold-standard atrium as the hero demo room and the phone capture as the user-study condition.

**Before renting anything (free, do first):**
- Run our gsplat to Spark pipeline on a ScanNet++ DSLR scene (undistorted DSLR images released 30 Apr 2025) and on an Eyeful Tower scene (1K JPEG subset is 14.1 GB). Target: plain 3DGS should land near 23.9 dB on ScanNet++. This tests our trainer before we spend money.

---

## 4. Exact recipe for the gold-standard capture

**Gear (rent):**
- Body: Sony a7 IV (33 MP full-frame; the ScanNet++ camera). Lensrentals $113 / 3 days.
- Lens, pick one:
  - Laowa 8-15mm f/2.8 fisheye at 8 mm: circular 180°, closest to the wide overlap ScanNet++ wanted. $35 / 3 days, $50 / 7 days.
  - Sigma 15mm f/1.4 DG DN fisheye: 180° diagonal only, so less overlap. $79 / 3 days.
  - The ScanNet++ paper does not name its lens. Both are our choices.
- Tripod (needed if we bracket; strongly advised anyway). Eyeful used a rigid base with a 1.8 m pole.
- Scanner: Leica BLK360 G2, $250/day + $315 Register360 7-day licence (Kwipped listing, community; prices change). Fallback: FARO Focus S 70, $800/day (Kwipped).
- Printed coded targets (paper), only if we try photo-only poses.

**Camera settings:**
- Manual everything. Lock white balance, exposure and focus for the whole room. Never change them mid-capture.
- Shutter 1/100 s or faster ("to avoid flickering effects from indoor lights and minimize motion blur", ScanNet++).
- f/8 (Eyeful). ISO as high as needed to keep the shutter; noise hurts less than blur (Postshot). Eyeful used ISO 500.
- RAW. Image stabilisation off on a tripod (Magnopus).
- If the windows or blinds clip: bracket 3-5 shots. Eyeful used 9 shots, -4 to +4 EV. The long frames are up to 16x the centre exposure, so brackets **only work on a tripod**.

**White-wall prep:**
- Lights on, steady, no flicker. Close or fix the blinds in one position for the whole capture.
- Put a few textured objects in view where walls are blank, or place coded targets / small tape marks on blank walls and the ceiling. Size targets so the centre circle is about 4-30 px radius in the image at your shooting distance (Agisoft). Mask or remove them before training.
- Never shoot a frame that is only blank wall. Use oblique angles that include nearby structure (Varjo Teleport). Stay at least 30 cm from walls (Eyeful).
- Optional: put a ColorChecker in one corner shot for colour reference (Eyeful).

**Path:**
- Move the camera between shots. Never rotate in place (Postshot, Teleport).
- Stops about 30 cm apart across the floor (Eyeful). At each stop, shoot outward in several directions.
- 2-3 heights: chest, overhead (pole), low. Include floor and ceiling.
- Back to the wall, shoot along the longest sightline (Postshot).
- Extra close passes around grabbable objects (desks, chairs, door) for sharp detail.
- Overlap at least 70% between neighbours (Magnopus says 70-80%).

**Image count:**
- ScanNet++: about 200 for a medium room, 200 to 2k for bigger ones. Magnopus: 1000-2000 for "high" fidelity.
- For the atrium: plan **600-1,000 training stills** (an estimate; scale with floor area), plus **15-25 held-out test stills** at new positions and heights.

**Laser scan:**
- BLK360: about 20 s per full-dome scan with photos [official, https://leica-geosystems.com/products/laser-scanners/scanners/leica-blk360].
- At least 4 positions (ScanNet++ averaged 4 for a medium room). More for a large atrium. Export E57.

**Poses:**
- Route A (recommended): RealityScan. Import the E57 laser scan, align the photos to it, then use the COLMAP export (RealityScan 2.1.1: standard COLMAP layout, and a fix for COLMAP export quality from distorted originals) [official, https://dev.epicgames.com/documentation/en-us/realityscan/realityscan-2-1-1]. The Seti I tomb case study used FARO E57 scans "as a scaled base" [official, https://www.realityscan.com/en-US/spotlights/seti-i-preserving-a-3000-year-old-tomb-with-realityscan]. **(Unverified):** the Epic knowledge-base page on laser-scan import did not load. A community note says stationary terrestrial scans are supported and SLAM scans are not. The BLK360 is a stationary scanner, so it should be fine, but test it.
- Route B: Metashape Pro with coded targets and scale bars (Eyeful's method), then export to COLMAP.
- Route C: COLMAP alone on the fisheye stills. Try it first. It is free. Fall back to A if it fails on the walls.
- **(Unverified):** whether ScanNet++'s own laser-to-photo registration code is public. Do not plan on it.
- Check: the poses should put the photos on the laser geometry with no drift.

**Training:**
- gsplat with 3DGUT, trained on the fisheye images directly. This keeps the wide periphery that helps plain rooms. Or undistort to pinhole, accepting the crop.
- Seed the Gaussians from the laser point cloud. FIORD saw +0.93 dB from LiDAR seeding on one scene, though the paper's own tables are inconsistent on this.
- Per-image exposure / appearance compensation on.
- Score on the held-out test stills: PSNR / SSIM / LPIPS, and colour-corrected PSNR for the phone run.
- Export to Spark as now.

**Rough cost (US list prices, before coverage, shipping and tax):**
- Camera + Laowa lens, 3 days: about $148. With the Sigma: about $192.
- BLK360 G2, 1 day + licence: about $565.
- Total: about $710-760 for a one-day shoot. A week of everything: about $1,500-1,600.
- Prices are from marketplace and rental listings and change. Re-check before booking.

---

## 5. What we still need from the user

1. **Budget:** is about $750 for a one-day gold-standard shoot OK? Or start photos-only, about $150-200?
2. **Country / city:** the prices above are US (Lensrentals, Kwipped). UK and other options differ.
3. **What can be borrowed for free:** ask the university's geomatics, civil, architecture, robotics or film departments for a full-frame mirrorless camera, a fisheye or 12-16 mm lens, a tripod, and above all a laser scanner (BLK360, FARO Focus, RTC360). Library lending searches found none.
4. **Atrium access:** a 2-3 hour slot with no people moving through; permission to stick small removable targets on the walls; control over the lights and blinds.
5. **Atrium size** (rough floor area and ceiling height), to set image count and scan positions.
6. **Software:** do we have, or can we get, a Metashape Pro licence? If not, use RealityScan or COLMAP.
7. **Go-ahead** to first run the free rehearsal on ScanNet++ and Eyeful Tower data with our pipeline.

### Not verified / open
- Whether ScanNet++'s pose registration code is public.
- RealityScan laser-scan import details (the Epic knowledge-base page did not load).
- PortalCam splat quality and any image+pose export.
- Metashape Pro price on agisoft.com.
- Rental stock and lead times (Lensrentals needs dates).
- Any measured effect of white walls on splat quality: no paper found.
