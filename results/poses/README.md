# Camera poses for a casual iPhone video: commercial-safe shootout on the atrium

**Question:** which method that may be used commercially gives the best camera poses and focal
length for every frame of an unguided phone video?

**Answer: NVIDIA ViPE (pose-only, with MoGe-2 depth).**
- It posed all 693 frames.
- Median camera-position error is 1.8 cm.
- It estimates its own focal length, 1.3% off the truth.
- Its scale is metric to within 1.1%.
- Cost: $0.11 on an A10, in 6 minutes.

DROID-SLAM was a little tighter, but only when it was given the right focal length. It cannot
find the focal itself, and its scale is arbitrary.

How it was run: `scripts/poses/` (README there). The inputs and every method's output are in
`atrium/`.

## The test

**Video:** `IMG_5657.MOV`, a 46 s handheld walk of the atrium. Shot in 4K HLG at 60 fps on an
iPhone 16 Pro Max, with nothing special about the filming.
- Converted to SDR with Apple's `avconvert`, at 1080 x 1920.
- 693 frames kept: the sharpest of every 4.

**Reference:** `scripts/poses/ground_truth/ground_truth.py` builds the true poses from the same
room scanned with LiDAR (Stray).
- Each video frame is matched to the LiDAR photos with SIFT.
- Matched points are lifted to 3D with LiDAR depth.
- PnP then fixes the frame's pose.
- The focal length is a grid search over all frames: **fx = 1556 px**. The tag's 24 mm
  equivalent (1222 px) is 27% short, because video stabilisation crops the sensor.
- Accuracy where it exists:
  - reprojection error median 1.5 px;
  - neighbouring frames agree to about 3 mm;
  - the two LiDAR alignments agree to 4.5 mm.
- **Coverage: only 143 frames** (0–95, 255–315, 356–408). SIFT could not match the low,
  close views of desks and floor to the LiDAR walk's standing-height photos. Nothing after
  frame 408 is scored.

**Scoring:** `scripts/poses/score.py`.
1. A similarity (scale, rotation, translation) aligns each method to the reference.
2. It then reports position error, rotation error, scale and focal error.

## Results (143 reference frames)

| method | licence (code / weights) | frames posed | position error median / p90 | rotation error* | fx error | scale | time | cost |
|---|---|---|---|---|---|---|---|---|
| **ViPE, own focal** | Apache-2.0 / DROID BSD-3, GeoCalib CC BY 4.0, MoGe-2 MIT | 693/693 | **1.8 / 6.7 cm** | 0.9° | **−1.3%** (estimated) | 0.989 (metric) | 349 s | $0.11 |
| DROID-SLAM, given fx 1604 | BSD-3 / BSD-3 repo | 693/693 | 1.6 / 5.5 cm | 0.5° | +3.1% (given) | 1.13 (not metric) | 107 s | $0.03 |
| ViPE, focal fixed at the tag | as above | 693/693 | 8.9 / 24.9 cm | 8.3° | −21.5% | 1.04 | 221 s | $0.07 |
| COLMAP 4 global, ALIKED + LightGlue | BSD-3 / BSD-3 + Apache-2.0 | 684/693 | 11.2 / 20.9 cm | 9.2° | −2.0% (estimated) | not metric | 2238 s | $0.77 |
| DROID-SLAM, fx from the tag | BSD-3 | 693/693 | 13.5 / 30.5 cm | 20.5° | −21.5% | not metric | 119 s | $0.03 |
| old MapAnything import (control) | Apache-2.0 | 181 (43 compared) | 13.7 / 32.9 cm | 3.2° | −33.7% | 0.83 | – | – |
| VGGT-1B-Commercial | VGGT License v1 | not run: weights gated behind a manual Meta approval | | | | | | |

\* Rotation error after aligning on camera orientations. `score.py`'s own figure aligns on
camera centres only. With the reference's small spread that adds 2–3° for good methods, and
gives nonsense (50–100°) for broken ones.

**Local accuracy is the same for the three good runs.** On frames 0–95, with its own alignment,
each is off by about 0.7 cm: ViPE, DROID-f1604 and COLMAP alike. The differences above come
from drift between the stretches of video.

**The old import was far worse than it looked.** Its earlier "~8 cm" figure was a depth-cloud
registration measure. Measured against these reference poses in the room's own frame:
- its cameras are off by a median 0.80 m, about 1 m too far forward along the view;
- that fits its focal length being 34% short.

## What this means for "any video"

- **The focal length is the whole game.**
  - Every method given the tag's 24 mm focal fails, at 9–14 cm and up to 20°.
  - Every method that finds the focal itself, or is handed the right one, lands within
    1–2 cm.
  - The pipeline must therefore estimate the focal, never trust the tag.
- **COLMAP is unsafe on its own here.** About 34 frames (mostly 563–634, a fast blurred pan)
  land thousands of units away. Re-running did not fix it, and a stricter matching threshold
  only reduced it.
- **ViPE is the pick.** It is cheap and complete, finds its own focal, and gives metric scale.

## Open problems

1. **Frames 530–692 are unchecked, and the methods disagree there.** ViPE and DROID-f1604
   differ by a median of 31–54 cm. Against COLMAP's non-wild frames there, ViPE is 40 cm off
   and DROID 16 cm off, a hint that ViPE drifts at the end. The reference should be extended
   past frame 408, for example with a learned matcher, ALIKED + LightGlue, against the LiDAR
   photos.
2. **ViPE and DROID share a network (`droid.pth`),** so their agreement is weaker evidence than
   two independent methods. They share the same pose jumps at frames 286, 341–346, 434 and
   500.
3. **`droid.pth` has no explicit weights licence.** DROID-SLAM's README links it, and the repo
   is BSD-3, but the file itself has no terms. Both ViPE and DROID depend on it, which is a
   grey area for commercial use. ViPE's repo also ships non-commercial and AGPL code (Unik3D,
   Segment-and-Track). None of it runs in pose-only mode, but it must be stripped before
   anything is redistributed.
4. **Reference frame 00255 is suspect.** It disagrees with all three good methods by 5.5–6.7
   cm, and it sits at the end of a run.

## Next

1. Train splat rooms from ViPE's poses and from DROID-f1604's.
2. Score both renders on the 29 sharp held-out LiDAR frames, as for every room.
3. Use the render-quality ranking to settle the unchecked last third of the video.
