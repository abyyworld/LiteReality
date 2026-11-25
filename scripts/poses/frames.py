"""The pose benchmark's shared frame set from one phone video: every method gets the same frames.

    uv run python scripts/poses/frames.py VIDEO OUT_DIR [--window 4]

VIDEO should already be SDR and upright (an iPhone HDR clip: `avconvert -p Preset1920x1080`, Apple's
own HDR -> SDR tone mapping, rotation applied). Of every `window` consecutive frames the sharpest
(variance of the Laplacian of the grey image at half size) is kept, written as OUT_DIR/frames/
NNNNN.jpg (quality 95), and OUT_DIR/frames.json lists them: name, source frame index, time,
sharpness, plus the clip's size, fps and the focal from the file's 35 mm-equivalent tag when present.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": num / den,
            "frames": int(s.get("nb_frames") or 0)}


def focal_35mm(video: Path) -> float | None:
    """The clip's 35 mm-equivalent focal length from its QuickTime tags (iPhone:
    com.apple.quicktime.camera.focal_length.35mm_equivalent), or None."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json",
                          str(video)], capture_output=True, text=True)
    try:
        tags = json.loads(out.stdout).get("format", {}).get("tags", {})
    except ValueError:
        return None
    for key, value in tags.items():
        if "35mm" in key.lower():
            try:
                return float(str(value).split()[0])
            except ValueError:
                pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--source-35mm", default=None, help="file to read the 35 mm focal tag from")
    args = ap.parse_args()
    video, out = Path(args.video), Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    info = probe(video)
    w, h = info["width"], info["height"]
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    kept, best, i = [], None, 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        grey = cv2.cvtColor(cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_RGB2GRAY)
        sharp = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        if best is None or sharp > best[1]:
            best = (i, sharp, rgb.copy())
        i += 1
        if i % args.window == 0:
            kept.append(best)
            name = f"{len(kept) - 1:05d}.jpg"
            cv2.imwrite(str(out / "frames" / name), cv2.cvtColor(best[2], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            best = None
    proc.wait()
    f35 = focal_35mm(Path(args.source_35mm or args.video))
    # 35 mm equivalent: 36 mm across the long side of a 3:2 frame, i.e. a 43.3 mm diagonal.
    diag = float(np.hypot(w, h))
    fx35 = f35 / 43.27 * diag if f35 else None
    sharp = np.array([k[1] for k in kept])
    manifest = {"video": str(video), "size": [w, h], "fps": info["fps"], "decoded": i,
                "window": args.window, "focal_35mm": f35, "fx_from_35mm_px": fx35,
                "frames": [{"name": f"{n:05d}.jpg", "source_index": k[0],
                            "time_s": round(k[0] / info["fps"], 4), "sharpness": round(k[1], 2)}
                           for n, k in enumerate(kept)],
                "sharpness_quantiles": np.percentile(sharp, [5, 25, 50, 75, 95]).round(1).tolist()}
    (out / "frames.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "frames"}))


if __name__ == "__main__":
    main()
