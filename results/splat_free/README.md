# splat_free: does the agreement filter make thin-scan splat rooms look better?

Measured 2026-11-23. Everything ran locally on an Apple M4 laptop: no Modal, no GPU training, no
SAM masks.

**Recommendation: ship `base`, and leave `--support-views` off.** Do not ship `support3` or
`clean_support3`.

- **More holes, not less speckle.** At 3 photos the agreement filter cuts large holes in walls,
  ceilings, floors, doors and furniture. Uncovered screen area in the off-path views goes from
  about 6% to about 38%.
- **Worse on held-out photos.** It costs about 5 dB PSNR on the held-out photos in every room.
- **No better where splats remain.** On pixels every variant still covers, the filtered rooms
  score the same or worse.
- **Speckle hardly changes.** It is low in the base renders already. The real faults (smeared
  surfaces, a few large floaters next to the camera) are only partly removed.

## What was compared

For each room the existing trained snapshot was re-cut three ways. Each copy of `raw.spz.bin`
went into its own `run/<room>/splat_free_<variant>` and ran through `litereality splatroom --recut
--no-masks --keep-physics`.

| variant | flags | what it keeps |
|---|---|---|
| `base` | none | the trained snapshot as is (800,000 splats) |
| `support3` | `--support-views 3` | only splats that ≥3 training photos put on their measured surface (LiDAR, `max(5 cm, 4%)`) |
| `clean_support3` | `--clean --support-views 3` | the above plus the faint, outside-the-room and carving rules |

### How the re-cuts ran

- **No Modal or GPU.** The re-cut path does not touch Modal: `SavedTrainer` replays the saved
  snapshot, `--no-masks` turns SAM off, and `--publish` and `--depth promptda` were not used.
- **Physics untouched.** `--scene` pointed at the served physics: `objects/scene_splat.xml` for
  tea_room and office_room, `objects/scene.xml` for suite_3693, `quick/scene_splat.xml` for
  atrium_1059. With `--keep-physics` nothing moved, and every scene XML's checksum was the same
  before and after.

### What was scored

- **Uncut trained snapshot.** Scoring used the snapshot before any cutting: `raw.spz.bin` for
  `base`, `raw.clean.spz.bin` for the others. This is the same snapshot `litereality evaluate`
  scores in its splat stage.
- **Same renderer as the evaluation.** `fidelity.SplatRenderer`: Spark 2.3.1 with three.js
  0.180 in headless Chromium, using software WebGL (SwiftShader). Scored with
  `fidelity.compare` at 320×240 after a 2× render.
- **Views.** Every held-out photo the trainer kept out (every 8th in name order: 8 per room, 7
  for atrium_1059) plus 2 training photos.
- **Wander views.** 3 cameras that are not capture views. They are at least 0.3 m inside the
  walls, not inside tall furniture, and 0.6–1.0 m from the nearest capture spot, so they fall
  inside the client's 1 m walk zone. Camera height is the nearest photo's height (1.1–1.7 m
  above the floor), looking at the far corner of the room and 8° down.
- **Background.** The wander renders are drawn on the call's dark-theme background (#1d2227), so
  holes look the way they do in the call without the backdrop.

## Held-out photos (the trainer never saw these)

| room | variant | splats kept | PSNR | SSIM | PSNR gain-matched | PSNR walls/floor | PSNR objects | coverage | training-view PSNR | training-view coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| tea_room | base | 800,000 | **19.31** | **0.702** | **20.28** | **19.80** | **18.59** | 94.6% | **20.73** | 96.9% |
| tea_room | support3 | 544,956 (68%) | 13.88 | 0.660 | 15.74 | 15.00 | 12.82 | 64.3% | 15.42 | 73.0% |
| tea_room | clean_support3 | 448,385 (56%) | 13.78 | 0.657 | 15.68 | 14.93 | 12.74 | 63.9% | 15.28 | 72.4% |
| office_room | base | 800,000 | **17.58** | **0.591** | **18.39** | **17.90** | **16.00** | 92.1% | **22.31** | 97.2% |
| office_room | support3 | 612,063 (77%) | 12.09 | 0.521 | 13.97 | 13.59 | 8.31 | 71.2% | 15.84 | 86.7% |
| office_room | clean_support3 | 494,298 (62%) | 12.01 | 0.509 | 13.92 | 13.49 | 8.23 | 70.3% | 15.46 | 85.5% |
| suite_3693 | base | 800,000 | **17.52** | **0.643** | **18.33** | **18.68** | **16.16** | 92.9% | **16.43** | 95.1% |
| suite_3693 | support3 | 498,894 (62%) | 12.02 | 0.606 | 14.53 | 13.37 | 8.24 | 61.4% | 9.30 | 59.7% |
| suite_3693 | clean_support3 | 396,787 (50%) | 11.99 | 0.602 | 14.51 | 13.32 | 8.22 | 61.1% | 9.27 | 59.5% |
| atrium_1059 | base | 800,000 | **15.69** | **0.516** | **16.60** | **16.44** | **12.08** | 83.8% | **16.13** | 90.2% |
| atrium_1059 | support3 | 465,130 (58%) | 11.32 | 0.494 | 13.56 | 12.29 | 8.37 | 52.6% | 11.80 | 69.4% |
| atrium_1059 | clean_support3 | 394,005 (49%) | 11.29 | 0.494 | 13.54 | 12.28 | 8.29 | 52.5% | 11.75 | 69.2% |
| **mean of 4** | base | | **17.53** | **0.613** | | | | 90.9% | | |
| **mean of 4** | support3 | | 12.33 | 0.570 | | | | 62.4% | | |
| **mean of 4** | clean_support3 | | 12.27 | 0.566 | | | | 62.0% | | |

How to read the columns:

- **coverage** is how much of the photo the splats cover.
- **PSNR gain-matched** applies the single brightness gain that best fits the photo first.
- **PSNR objects** uses RoomPlan's object boxes, masked by LiDAR depth.
- **Where uncovered pixels count:** they show the evaluation's light-grey background, which
  counts against them. In the call with the new static-mesh backdrop (contract C1), those pixels
  would show the textured room mesh instead. That would score better than grey, but it would not
  be photoreal.

**LPIPS was not measured.** This checkout has no LPIPS: `fidelity.compare` computes PSNR and
SSIM only, and torch is not installed. Adding it means installing torch and lpips, plus the
AlexNet weights, about 1 GB. I did not install it. The trainer's own LPIPS (`report.json`
`stats`, gsplat's held-out pass) is only for the uncleaned snapshot: tea_room 0.366,
office_room 0.377, atrium_1059 0.454, suite_3693 not recorded.

### Only the pixels every variant covers (holes taken out)

PSNR on pixels where all three variants have alpha ≥ 0.99 (`measure.py --shared`). "Shared" is
the share of the image that qualifies.

| room | shared (held-out) | base | support3 | clean_support3 | training views: base / support3 / clean_support3 |
|---|---|---|---|---|---|
| tea_room | 23% | **23.72** | 23.34 | 23.29 | **24.28** / 22.96 / 22.82 |
| office_room | 26% | **23.19** | 21.70 | 21.38 | **25.81** / 22.77 / 22.30 |
| suite_3693 | 24% | **20.91** | 20.76 | 20.72 | **18.10** / 17.96 / 17.95 |
| atrium_1059 | 11% | 18.48 | 18.48 | 18.48 | 20.94 / 20.94 / 20.93 |

Even where the filter leaves the surface in place, it never improves the image. The splats it
removes also contributed colour to pixels that stay covered, so taking them out makes those
pixels worse. The worst is office_room with `clean_support3`: 1.8 dB on held-out photos (23.19
to 21.38) and 3.5 dB on training views (25.81 to 22.30); `support3` alone loses 1.5 and 3.0 dB
there.

## Wander views (off the capture path)

Side by side, one row per camera, columns `base | support3 | clean_support3`:
`tea_room_wander.jpg`, `office_room_wander.jpg`, `suite_3693_wander.jpg`,
`atrium_1059_wander.jpg`.

| room | variant | holes per spot (alpha < 0.5) | mean holes | speckle per spot |
|---|---|---|---|---|
| tea_room | base | 2.6% / 1.4% / 2.3% | **2.1%** | 0.05% / 0.03% / 0.02% |
| tea_room | support3 | 36.8% / 21.7% / 32.0% | 30.1% | 0.06% / 0.06% / 0.07% |
| tea_room | clean_support3 | 37.6% / 21.9% / 32.2% | 30.6% | 0.06% / 0.07% / 0.09% |
| office_room | base | 8.5% / 3.5% / 11.1% | **7.7%** | 0.17% / 0.13% / 0.23% |
| office_room | support3 | 45.6% / 16.0% / 39.1% | 33.5% | 0.24% / 0.16% / 0.19% |
| office_room | clean_support3 | 46.9% / 16.5% / 40.0% | 34.5% | 0.31% / 0.17% / 0.25% |
| suite_3693 | base | 1.5% / 7.6% / 6.9% | **5.3%** | 0.01% / 0.05% / 0.06% |
| suite_3693 | support3 | 20.2% / 35.5% / 39.8% | 31.8% | 0.03% / 0.05% / 0.06% |
| suite_3693 | clean_support3 | 20.7% / 35.9% / 40.1% | 32.2% | 0.04% / 0.05% / 0.07% |
| atrium_1059 | base | 7.4% / 1.9% / 12.7% | **7.3%** | 0.08% / 0.11% / 0.19% |
| atrium_1059 | support3 | 77.5% / 26.1% / 65.0% | 56.2% | 0.10% / 0.20% / 0.22% |
| atrium_1059 | clean_support3 | 77.5% / 26.2% / 65.1% | 56.3% | 0.10% / 0.20% / 0.23% |

"Speckle" is the share of covered pixels whose brightness is more than 0.12 away from the median
of their 3×3 neighbourhood. It is a crude measure: it barely moves, and the filter does not lower
it. Edges around the new holes even push it up a little.

What the renders show:

- **Holes:** the filter makes many more. In every room it removes large areas of real, flat,
  well-lit surface. These include upper walls and ceilings, floor under and around tables, the
  open door and doorway in suite_3693, the left wall and shelving in office_room, and most of
  the walls near the camera in atrium_1059.
- **It does not target the holes the task describes.** In the base renders the bright windows
  are mostly filled. The filter cuts *new* holes beside and into them (office_room spots 1 and 3,
  atrium_1059 spot 3).
- **Speckle:** hardly changes. What it does remove is a few **large floaters near the camera**,
  which is a real improvement:
  - the dark spiky blob in office_room spot 3 and the orange smear in spot 2;
  - some of the blurred clutter in front of the camera in tea_room spot 3.

  But it removes them together with everything else that fewer than 3 photos saw.
- **Smeared and blotchy surfaces stay.** Examples are the left wall in office_room spot 2 and
  the floor in atrium_1059 spot 2. Their splats are on the surface, so the filter keeps them.
- **`clean_support3` looks the same as `support3`.** The faint, outside-the-room and carving
  rules remove another ~100k splats, but they are nearly invisible from these viewpoints.

## Why the filter removes so much

The diagnostic counted, per splat, the training photos that see it on valid depth and the
training photos that put it on the surface. Training photos are capped at 96, as in `clean.py`.

| room | removed at 3 (as built) | of those, on valid depth in < 3 photos at all | median removed splat: on valid depth / on the surface (photos) | conf ≥ 1 instead of 2 | 15 cm / 8% instead of 5 cm / 4% | both |
|---|---|---|---|---|---|---|
| tea_room | 31.9% (38.7% of opacity) | 60% | 2 / 1 | 29.0% | 24.9% | 21.5% |
| office_room | 23.5% (29.1%) | 73% | 2 / 1 | 16.3% | 19.7% | 12.4% |
| suite_3693 | 37.6% (36.9%) | 40% | 3 / 1 | 30.7% | 30.4% | 22.7% |
| atrium_1059 | 41.9% (44.4%) | 71% | 2 / 1 | 35.0% | 36.1% | 27.3% |

- **In three of the four rooms it is mostly a coverage test.** In tea_room, office_room and
  atrium_1059, 60–73% of the removed splats had valid depth (confident, edge-free LiDAR) in
  fewer than 3 training photos at all, so they could never reach the bar; the median removed
  splat there is on valid depth in 2 photos and on the surface in 1. The other 27–40% were
  measured by 3 or more photos and sat on the surface in only 1–2 of them: those are
  disagreement, not coverage.
- **suite_3693 is the exception.** Only 40% of its removed splats had valid depth in fewer than
  3 photos. Most (60%) were measured by 3 or more photos but sat on the surface in only 1–2;
  the median removed splat is on valid depth in 3 photos and on the surface in 1. There the
  rule removes mostly splats that the depth disagrees with, which is what the narrower rule
  below is built on.
- **Thin coverage is exactly the problem a thin scan has.** So the rule throws away the areas
  the thin scan covered poorly, together with their speckle.
- **Loosening the tolerance or the confidence does not rescue it.** Both together still remove
  12–27% of each room.

A narrower rule targets floaters without punishing thin coverage. It drops a splat only when
**≥ 3 photos covered it with valid depth and none of them put it on the surface**. That would
remove 3.4% (tea_room), 0.9% (office_room), 6.7% (suite_3693) and 5.1% (atrium_1059) of
splats. A ratio rule (`on < 25% of seen`, with `seen ≥ 3`) would remove 3–13%. I have not
rendered either one, so treat them as hypotheses. If the speckle work continues, the next
experiment is one of them, with `support3` replaced.

## Timing

| step | time |
|---|---|
| re-cut one variant (`splatroom --recut --no-masks`, instant room plus trained room, split, bundle) | 10.6–15.6 s per run (`report.json` `total_s`), 11–16 s wall |
| agreement filter alone, 800k splats × 44–55 training frames | 0.93–1.37 s |
| all cleaning rules together | 0.97–1.19 s |
| one headless Spark render in SwiftShader (800k splats, 640×480) | ≈40–90 s per view: about 7 s per `renderer.render`, and the page draws each view 7 times |
| scoring one room (3 variants × 13 views) | 22–34 min; 2.5 h for all four |

Running several renders at once is slower in total, because SwiftShader already uses every
core.

Fewer warm-up draws (2 or 3 instead of 6) are 1.8–2.3× faster. But 2.4% of one view's pixels
changed by more than 8/255, while the original page is bit-for-bit repeatable. So the original
page was kept.

## Files

- `measure.py`: scoring, wander cameras and contact sheets. Run it with
  `uv run --frozen --with playwright python results/splat_free/measure.py [room ...]`, and add
  `--shared` for the shared-coverage table.
- `<room>.json`: per-view metrics, wander spot positions and hole/speckle numbers per variant.
- `<room>_wander.jpg`: the side-by-side wander renders.
- `cache/` (ignored by git, 65 MB): every render. A second run reuses it.
- `run/<room>/splat_free_<variant>/`: the re-cut bundles. Each one can be served with
  `uv run litereality roomcall <scene> --splat run/<room>/splat_free_<variant>`.

## Caveats

- **Few views.** 7–8 held-out photos per room. The gaps between variants (4–6 dB) are far
  larger than the spread between views, but small differences in this table mean little.
- **No backdrop in these renders.** They show raw holes. With contract C1's static-mesh backdrop
  the holes become textured mesh, which is better than the dark background but not photoreal.
  It does not change which variant wins.
- **Unchecked held-out lists.** tea_room, office_room and suite_3693 have no `images` list in
  `report.json`. Their held-out photos are taken as every 8th by file name, as the trainer and
  `clean.py` assume. This was not checked against the trainer's actual tar.
