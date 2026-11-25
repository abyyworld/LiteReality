# Pose benchmark: camera poses for every frame of a casual phone video

**Question:** which commercially usable method gives the best camera poses and focal length for
the frames of an unguided iPhone video?

**Reference answer:** the same atrium recorded with iPhone LiDAR (Stray), whose poses are good to
about 1 cm in the room's frame.

## Shared input

`~/LiteReality-scans/poses_bench/atrium/` holds the input. The Modal volume `litereality-poses`
has a copy: `/atrium/frames.tar` and `/atrium/frames.json`.

- **`frames/NNNNN.jpg`:** 693 frames, 1080 x 1920 portrait.
  - Source: `IMG_5657.MOV` (iPhone 16 Pro Max, 4K HLG HDR, 60 fps, 46 s).
  - Converted to SDR, upright, by `avconvert -p Preset1920x1080` (Apple's own tone mapping).
  - The sharpest frame of every 4 (`frames.py`).
- **`frames.json`:**
  - each frame's name, source frame index, time and sharpness;
  - `focal_35mm` 24 mm, from the file's tag;
  - `fx_from_35mm_px` 1221.9, which ignores the stabilisation crop. The 2026-11-23 import
    self-calibrated about 1604 px; MapAnything said 1032.

## What every method writes

Each method writes `~/LiteReality-scans/poses_bench/atrium/poses_<method>.json`:

```json
{"method": "<name>", "licence": "<code licence> / <weights licence>",
 "metric": true,
 "fx": 1500.0,
 "frames": {"00000.jpg": {"c2w": [[...4x4...]]}, "...": {}},
 "seconds": 0.0, "gpu_seconds": 0.0, "cost_usd": 0.0, "notes": "..."}
```

- **`metric`:** true if the poses claim to be in metres.
- **`fx`:** one shared focal length in pixels at 1080 x 1920. Use the median if the method gives
  one per frame, and say so in `notes`.
- **`c2w`:** camera-to-world, with OpenCV axes (x right, y down, z forward), in any world frame.
- **Frames:** leave out any frame the method could not pose. Never invent a pose.
- **`seconds`:** wall time of the pose run.
- **`gpu_seconds` and `cost_usd`:** from Modal.

The reference is `gt.json` in the same format, in the room's frame, made by `ground_truth.py`.

## Scoring

```
uv run python scripts/poses/score.py GT.json POSES.json
```

The scorer aligns each method's world to the reference with a similarity (scale, rotation,
translation). It then reports:
- the share of frames registered;
- camera-position error: ATE RMSE, median and p90;
- camera-orientation error;
- scale;
- focal error.

## Rules

- **Commercial use only.** A method's code AND weights must allow commercial use. Check each
  licence file yourself, record it, and skip anything research-only or non-commercial.
- **Modal:** ephemeral `modal run` apps only (no deploys), `--env-file ~/LiteReality/.env`,
  `MODAL_IMAGE_BUILDER_VERSION=2025.06`. Runs longer than 5 minutes go detached, writing their
  results to a volume.
