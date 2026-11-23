# RealityScan on DENSE frames: can it reconstruct the whole atrium walk as one piece?

> Internal research only (RealityScan EULA). Ephemeral `modal run` of
> `deploy/modal/realityscan/app.py::run_dir` (CPU, 8 cores / 16 GB); inputs and outputs on the
> private Volume `litereality-tools` (`inputs/`, `runs/<tag>/`). Not wired into any service.

**Answer: no.** Neither video reaches one component, or 90 % of frames in the largest. With dense
frames and every setting tried, the Stray walk always breaks at about 8 s. The largest true
component is 13–16 % of the frames. With ARKit priors, RealityScan only *places* its pieces by their
camera positions. The result is still 11–16 independent reconstructions, each with its own lens and
a rotation offset of about 0.5–23° (see `../whole_room.md` for the per-piece re-check). The phone video's best largest piece is 31 %.

## Inputs

`scripts/realityscan/prep_dense.py` writes both. It uses the repo's own decoders and adds a blur
measure relative to the ±1 s neighbourhood, so plain walls are not dropped as "blurry".

- **Stray** (`~/LiteReality-scans/rs_dense_stray`)
  - Every 2nd odometry row: 1201 frames.
  - No ARKit tracking jumps were found.
  - The blurriest 10 % were dropped, plus the 29 eval rows always kept: **1095 frames** at 1920×1440.
  - Names: `r<row>.jpg`, with eval frames as `r<row>_eval.jpg`.
- **Phone video** (`~/LiteReality-scans/rs_dense_video`)
  - Every 2nd of the 2773 frames, with the blurriest 10 % dropped: **1248 frames**, `v<frame>.jpg`.
  - Size 1080×1920, portrait (display rotation applied, HLG read as SDR).

## Runs (Stray: 1095 frames; video: 1248)

| Tag | Settings | Components | Largest | Share | Pose vs ARKit, largest (Sim(3)): centre mm med/p90, rot ° med/p90 (orientation fit) | Lenses in largest | RS time |
|---|---|---|---|---|---|---|---|
| s2_def_dm2 | Defaults, focal prior 25.04 mm, Brown3 | 15 | 145 (0–8 s) | 13.2 % | 8.9 / 13.0, 0.39 / 0.46 | 1 (fx −0.65 %) | 431 s |
| s2_high_med_merge | High detector, 20k preselector, 20k/Mpx, 60k/image, then `-mergeComponents` | 17 → 14 | 148 | 13.5 % | 8.7 / 12.9, 0.38 / 0.49 | 1 | 627 s |
| s2_high_hiov | Same, overlap **High**, then merge | 16 → 12 | 144 | 13.2 % | 8.1 / 11.5, 0.39 / 0.46 | 1 | 538 s |
| s2_ultra_low_align2 | **Ultra** detector, overlap **Low**, 2nd `-align` | 15 → 18 | 171 | 15.6 % | 8.6 / 14.1, 0.43 / 0.73 | 1 | 3222 s |
| s2_def_dm0 | Defaults, **no distortion (model 0)** | not finished: preempted once, then still aligning after 55 min (dm2 took 7 min), so stopped to stay in budget | | | | | >3300 s |
| s2_prior3_align2 | ARKit priors "initial" ±3 cm / ±3°, merge georeferenced, 2nd align | 12 → 13 | 611 | 55.8 % | 7.7 / 17.5, **3.18 / 14.1** | **13** | 733 s |
| s2_prior5_rematch | Priors ±5 cm / ±5°, 2nd align with force component rematch | 12 | 551 | 50.3 % | 4.9 / 10.8, **1.45 / 5.15** | **11** | 852 s |
| s2_draft3 | Priors as PosePrior **"draft"**, ±3 cm / ±3° | 17 | 616 | 56.3 % | 4.5 / 12.2, **3.38 / 5.99** | **16** | 493 s |
| v2_high_med_merge | Video, High, focal prior 29 mm, merge | 7 → 5 | 388 | 31.1 % | (no ARKit) fx 1547 px | 1 | 684 s |
| v2_high_low_merge | Video, High, overlap Low, merge | 8 → 8 | 190 | 15.2 % | – | 1 | 751 s |

`runs_table.json` holds these rows. `<tag>_report.json` adds the per-run details: missing stretches,
camera parameters, and 1 s relative errors.

### Why it fails

- **Image content.**
  - Every image-only setting breaks at the same places:
    - 8.0–8.6 s: a white ceiling and wall, ~460 features per frame.
    - 18.7–22 s.
    - 34–45 s: white walls, then the dark mesh blinds. Their "features" are fabric noise and light through the mesh, which does not repeat between views.
    - 46–49 s.
    - 52–56 s.
  - Even frames with 3000–7000 detected features in those stretches register in no component or only in small pieces that do not connect (Ultra puts some 34–56 s frames in small pieces).
  - Dense frames do not help. Consecutive frames 1/30 s apart add almost no baseline, so a turn in
    front of a plain surface triangulates nothing.
  - Ultra/Low gets more frames into small pieces (702 vs 497 registered anywhere), at 7× the time.
    The largest piece still ends at 8 s.
- **Priors do not join pieces.**
  - RealityScan 2.2's pose priors, "initial" and "draft" alike, georeference each component
    separately. It places each piece by a similarity fitted to its camera *positions*, then merges
    the placed pieces.
  - There is no joint bundle adjustment. Every piece keeps its own lens (11–16 different fx in the
    "largest component"), even with one calibration group set in the CLI *and* in every XMP.
  - Each piece carries a constant rotation offset. In s2_prior3_align2 these are per piece:
    3.0°, 4.7°, 4.6°, 2.6°, 1.9°, 2.6°, 5.7°, 14.2°, 21.2°, and 161° / 56° for two 3-frame pieces.
    Positions agree to millimetres.
  - This is the "90° turned / 85° off" result of the first study. It is not an axis-convention bug.
- **The XMP convention was measured, not assumed.** A no-prior run's own `-exportXMP` was compared
  with its COLMAP export (145 cameras, residual 1e-12):
  - `xcr:Rotation` is world→camera with an OpenCV camera (COLMAP's R).
  - `xcr:Position` is the centre.
  - XMP world = A · COLMAP world, with A = [[1,0,0],[0,0,1],[0,−1,0]]: Z up in the XMP world.
  - The earlier `write_priors.py` convention was right. The priors here were written Z-up (gravity),
    with the shared calibration in every sidecar (`write_priors_dense.py`).

Where RealityScan does register frames as one true component (0–8 s, 145 frames), it agrees with
ARKit to 9 mm median / 13 mm p90, and 0.39° median / 0.46° p90 with the orientation fit.

### Video → room

The best video piece (v2_high_med_merge, 388 frames) was registered to the Stray LiDAR cloud with
`video_register.py` (levelling, coarse yaw/scale sweep, scaled ICP). The result is **ambiguous**:

- 30 % inliers, residual 1.6 cm, margin 0.055.
- The runner-up is 103° away in yaw and 2.3 m away.
- Against the MapAnything import of the same frames it is 19° off in rotation after its own Sim(3).

See `../video/registration_dense_v2hm.json`. Even if the registration were right, 69 % of the video
is outside the piece.

## Exports (best-effort, criterion NOT met)

- `~/LiteReality-scans/rs_dense_stray/colmap/{sparse/0,images}`: s2_prior5_rematch, the prior-merged
  piece set with the lowest rotation error, 551 frames. FULL_OPENCV on the original frames.
- `~/LiteReality-scans/rs_dense_video/colmap/{sparse/0,images}`: v2_high_med_merge, 388 frames.

## Spend and cleanup

- **Modal ephemeral:** $4.63 → $6.86, about **$2.2** for this job, all on CPU containers. This
  includes two preempted-and-restarted runs and the stopped dm0 run.
- **Metered total:** $16.53.
- **Running apps:** none.
- **Volume:** the inputs stay on the private Volume `litereality-tools` under `inputs/rs_dense_*`,
  and the runs under `runs/s2_*` and `runs/v2_*`.
