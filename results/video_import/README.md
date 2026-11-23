# Phone video -> scan: first real run (atrium)

Input: `~/LiteReality-scans/atrium_video.MOV`, shot in the iPhone Camera app on an iPhone 16 Pro Max.
It is 3840x2160 HEVC Main10 HLG at 59.94 fps, 46 s long, with rotation -90 (portrait).
Model: MapAnything, Apache-2.0 weights `facebook/map-anything-apache@00f9c245`, code `map-anything@3d10cf7a`.
It runs as the Modal app `litereality-geometry.reconstruct` on an H100.
Output: `~/LiteReality-scans/atrium_phone`, aligned to `atrium_1059` and checked against the Stray LiDAR recording `stray_atrium`, which is aligned to `atrium_1059` to about 1 cm.

```
MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file .env modal deploy deploy/modal/geometry/app.py
uv run --env-file .env litereality import-video ~/LiteReality-scans/atrium_video.MOV \
    --out ~/LiteReality-scans/atrium_phone --align-to ~/LiteReality-scans/atrium_1059 --frames 200 \
    --reference ~/LiteReality-scans/stray_atrium --viz-dir results/video_import --overwrite
# the floor column below: the same evaluation of the LiDAR scan atrium_1059 against the same truth
uv run python -m litereality_agent.models.capture.video_eval ~/LiteReality-scans/atrium_1059 \
    ~/LiteReality-scans/stray_atrium --json floor.json
```

Every number below comes from those two commands: `import_report.json` (`alignment`, `intrinsics_check`, `frame_check`, `reference_eval`) and the floor JSON.
The model's reply is cached in `atrium_phone/video_geometry/`, so a re-import of the same picks costs no GPU.

## Numbers (final run)

| stage | result |
|---|---|
| frames | 192 picked of 2773 (81 blurry and 23 near-duplicates dropped); **179 written**: 13 dropped by the frame check (below). Photos 1080x1920, depth and conf maps 144x256 |
| upload | 16.4 MB (582x1036 JPEGs) |
| model | H100 80GB: model load 17.8 s, calibration pass of 48 views 6.3 s, one pass of 192 views 24.3 s. Total 50.4 s, peak 50.2 GB, 76 s round trip |
| local (cached reply) | scoring 19 s, decode 34 s, alignment 33 s, checks 31 s (self-calibration + frame check), evaluation about 3 min |
| gravity | 3.2° from the cameras' mean up |
| alignment to atrium_1059 | scale 0.934, yaw -53.4°, residual 1.4 cm, 76% of the video on the scan's surface, heading not ambiguous (margin 0.24 over the runner-up at 128°). **Scale spread 2.2%** (below) |
| frame check vs atrium_1059 | 157 ok, 2 degenerate, 33 not registered (kept: the target may not cover what they see), 13 needing more than 20 cm / 10° to fit (dropped) |
| pointcloud.pcd | 246,206 points |

### Accuracy against the Stray LiDAR ground truth

| measure | video (MapAnything), 179 frames | floor: atrium_1059 LiDAR vs the same truth, 51 frames |
|---|---|---|
| **camera position vs PnP on LiDAR-lifted matches** (at the best focal; frames with >= 20 inliers, median < 3 px) | **0.81 m median, p90 1.01 m, 3.9°** (32 frames) | 6.8 cm, 1.8° (11 frames) |
| **focal length the LiDAR matches support** (PnP over a focal grid) | **1579 px vs the scan's 1032 (+53%)**, clear minimum | 1422 vs 1342 (+6%), flat curve: no real preference |
| reprojection with the scan's pose and K (PnP-verified matches, per-frame median) | 56 px median, p90 238 px (61 frames) | 27 px, p90 41 px (21 frames) |
| geometry registration (each frame's depth ICP'd onto the LiDAR; pose, depth and K mixed) | 8.1 cm / 3.8° over 159 well-constrained frames, p90 17 cm; **all 179 frames: 8.7 cm, p90 >= 30 cm** (17 did not register, 3 degenerate) | 3.0 cm / 1.7°, p90 7.2 cm, 0 failed |
| depth after the scale fit to atrium_1059 | 5.3 cm median, 4.9% relative (129 frames) | 2.8 cm, 1.9% |
| depth as MapAnything gives it (the fit's scale undone: what an import without `--align-to` gets) | **12.7 cm, 9.7% relative**; its metric scale is 7.1% too large | - |
| whole fused cloud vs the LiDAR (yaw + translation + scale ICP) | 0.13% scale, 0.9°, 2.0 cm centroid shift, 0.9 cm residual | 0.09%, 0.0°, 0.6 cm |
| nearest-viewpoint pairs (indicative only: two different walks, paired by the estimated pose) | 36 cm / 11.4°, 42 pairs, 137 unpaired | 23 cm / 16°, 25 pairs, 26 unpaired |

Summary:
- **The room as a whole is right; the cameras are not.** The fused cloud lands within 0.2% scale and 2 cm of the LiDAR, and each frame's depth fits the surface to about 8 cm. But the cameras sit about **0.8 m** from where the LiDAR matches put them.
- The cause is the focal length. MapAnything's K is about 35% too short (fx 1032 px where the LiDAR says 1579 and the photos alone say 1604). A wider field of view puts each camera closer to what it sees, and its depth shrinks to match, like a dolly zoom. So the frame's geometry still fits the room, and depth and ICP checks cannot see the error. Only reprojection and PnP checks can.
- The earlier "8.2 cm / 3.8° per-frame pose" was this geometry registration. It is not the pose error. It also left out the frames that failed to register.

Side-by-side images `depth_*.png` (photo | video depth | LiDAR depth on the same colour scale | |error| from 0 to 0.3 m) are in this folder.

## The focal length: the main open issue

- **LiDAR PnP** (`reference_eval.focal_vs_lidar`): 1579 px; the same check on the LiDAR scan gives +6% with a flat curve, so it is not biased toward long focals.
- **Self-calibration from the video alone** (`intrinsics_check.self_calibration`, `models/capture/selfcal.py`): Mendonça-Cipolla over 19 frame pairs gives 1604 px (per-pair median 1553, IQR 1396-1618). This needs no LiDAR and no metadata. It runs on every import, which now warns:
  > self-calibration from the photos puts the focal at 1604 px, the model's K says 1032 px (+55%, 19 frame pairs) ...
- **The model's own views** disagree by 13% on fx (`model_fx_relative_spread` 0.133), which also produces a warning now.
- **iPhone metadata**: `focal_length.35mm_equivalent = 24`, which gives 1331 px for the full sensor width. Stabilisation crops the video, so this is a lower bound.

Earlier experiments (one-off GPU runs, not reproducible from the repo, kept only as notes):
- Conditioning the model on a given K (the `intrinsics` option) moved its returned K only part of the way.
- Full runs with fx fixed at 1550 or 1331 did not change the geometry-registration and depth metrics. That is expected now: those metrics cannot see a focal error.
- Re-solving poses by PnP onto the model's own (distorted) cloud gave 20-30 px residuals.
- Landscape input was worse.

Likely fix: re-pose the frames at the self-calibrated focal, against something that does not share the model's distortion. Options:
- a feature bundle adjustment with the focal free, started from the MapAnything poses;
- with `--align-to`, PnP of each frame onto the target scan's LiDAR, which needs the target's photos for matches.

The new `focal_vs_lidar.pose_vs_pnp` and `reprojection` measures will show whether a fix works.

## Scale: the alignment pins it down to about 2%

The ICP runs converge to 0.934, 0.950, 0.950 and 0.954 from different starts. The wall-inlier score that picks among them plateaus (0.761 against 0.745), so the import now reports `scale_spread` (2.2%) and warns above 1%.
Here the pick (0.934) is the closest of the four to the truth: the whole-cloud check against the LiDAR wants +0.13%. Picking by residual or by all-point inliers would have chosen 0.950, which is 1.7% off. So the objective is unchanged; only the uncertainty is reported.

## Fixes in this round

- `sim3.align_sim3`: the heading verdict could describe the re-run's heading while the returned transform was another. Now, when the re-run at the found scale prefers a different heading, the kept transform is judged against that heading with `align`'s own rule. `scale_spread` is new.
- `video_eval` (all measures above):
  - frames that fail to register are counted (at the 30 cm capture radius, a lower bound), not dropped;
  - degenerate ICP fits (constraint ratio < 0.02) are reported separately;
  - new measures: PnP-verified reprojection, the LiDAR focal search with PnP poses, whole-cloud Sim(3), and raw metric depth;
  - "nearest-frame pairs" is renamed and no longer called an upper bound;
  - it refuses scans that are not in one world;
  - it is runnable as `python -m litereality_agent.models.capture.video_eval` for the floor.
- `import-video`:
  - `--reference` needs `--align-to`;
  - self-calibration and fx-spread warnings;
  - a per-frame registration check against the target (`videoCheck` in each frame JSON; frames needing more than 20 cm / 10° are dropped unless `--keep-bad-frames`);
  - a scale-spread warning.
- The app's `CONVENTION` now says K uses OpenCV integer pixel centres (MapAnything's `get_rays_in_camera_frame` uses `arange(width)`). The app is redeployed; nothing else changed.

Tests: `tests/test_capture_video.py` has 24 tests (8 new), all passing.

## Spend (from Modal billing, not an estimate)

`modal.billing` for the `litereality-geometry` app: **$1.17** in total. That is $0.22 in the 15:00 UTC hour and $0.95 in the 16:00 UTC hour. It covers the image build and the six GPU calls of the first round.
The earlier "$0.50" counted only GPU seconds. It left out:
- each call's cold start (image load plus 18 s model load);
- the 60 s idle before scale-down, which is billed;
- the 8 CPUs and 64 GiB reserved next to the H100.

This round made no GPU calls: every re-import reused the cached reply, and the redeploy built nothing.

On running two imports at once: each GPU call takes the same time as it would alone, so the wall time is about the same as for one. But each concurrent container pays its own cold start and its own 60 s idle tail. Two imports run back to back within 60 s share one warm container. So running in parallel is no slower, but it costs somewhat more than two sequential runs, not exactly the same.

## Next: train the photoreal room

```
uv run litereality splatroom ~/LiteReality-scans/atrium_phone --scene run/atrium_1059/quick/scene.xml \
    --label atrium_phone --out run/atrium_phone/splat --steps 5000
```

Risks for that step:
- **The cameras are about 0.8 m off with a 35% short focal**, as above. Expect blur and floaters away from the main surfaces. Fixing K and the poses first is the likely way to get a sharp room.
- The photos are portrait 1080x1920 with 144x256 depth. The splat room has only been run on 4:3 landscape scans.
- `rgbd.MAX_DEPTH` is 4.5 m.
