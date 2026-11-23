"""The ordinary phone video's frames RealityScan aligns: ~400 sharp frames at a 1920 px long side,
display rotation applied (the iPhone clip is landscape-coded 3840 x 2160 with a -90 display
matrix, so the frames are portrait 1080 x 1920).

Reuses models/capture/video.py's own steps (as `import-video` does): `probe` (size, rotation,
HDR: the iPhone's HLG is read as its SDR-compatible signal), `scan_video` (sharpness, PTS and a
thumbnail per frame, one small decode), `select_frames` (windows of half elapsed time / half
appearance change, the sharpest of each, blurry and near-duplicate picks dropped), then
`decode_frames` at the photo size.

    uv run --frozen python scripts/realityscan/prep_video.py [--n 400] [--out ~/LiteReality-scans/rs_video]

Writes <out>/rs_input/images/frame_XXXXX.jpg (pick order) and <out>/rs_input/selection.json
({name: {"video_frame", "time_s", "sharpness"}}, the video's probe and the selection stats).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
VIDEO = Path("~/LiteReality-scans/atrium_video.MOV").expanduser()


def main() -> None:
    from PIL import Image

    from litereality_agent.models.capture import video as V

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--long", type=int, default=1920)
    ap.add_argument("--video", type=Path, default=VIDEO)
    ap.add_argument("--out", type=Path, default=Path("~/LiteReality-scans/rs_video"))
    a = ap.parse_args()
    info = V.probe(a.video)
    print(info.summary())
    scan = V.scan_video(info)
    sel = V.select_frames(scan.times, scan.scores, scan.thumbs, target=a.n)
    print(sel.stats)
    size = V.scaled_size(info.size, a.long)
    out = a.out.expanduser() / "rs_input"
    img = out / "images"
    if img.exists():
        shutil.rmtree(img)
    img.mkdir(parents=True)
    frames = {}
    order = {int(k): j for j, k in enumerate(sel.indices)}
    for k, rgb in V.decode_frames(info, sel.indices, size):
        name = f"frame_{order[k]:05d}.jpg"
        Image.fromarray(rgb).save(img / name, quality=95)
        frames[name] = {"video_frame": int(k), "time_s": round(float(scan.times[k]), 4),
                        "sharpness": round(float(scan.scores[k]), 2)}
    (out / "selection.json").write_text(json.dumps(
        {"video": str(info.path), "info": info.summary(), "photo_size": list(size),
         "selection": sel.stats, "frames": dict(sorted(frames.items()))}, indent=1))
    print(f"{len(frames)} frames at {size[0]}x{size[1]} -> {img}")


if __name__ == "__main__":
    main()
