"""Per-frame sharpness of EVERY frame of the Stray atrium recording, at full resolution.

Streams rgb.mp4 once through ffmpeg (grey, 1920x1440, one frame in memory at a time) and writes
frames_blur.csv with, per video frame:
  lap_full     variance of the 3x3 Laplacian at 1920x1440
  lap_960      same after an area downscale to 960x720
  lap_480      same at 480x360 (what the import's frame picker uses, stray.SCORE_WIDTH)
  hf_ratio     lap_full / (4 * lap_960): energy in the finest octave relative to the next one.
               Texture-normalised: ~1 for a sharp natural image at this scale, falls with blur.
  gx, gy       mean |d/dx|, |d/dy| at full res (motion blur is directional)
  pkt_size     HEVC packet bytes, pict_type I/P/B (from ffprobe, separate pass)
Also writes the gyro rate (imu.csv) interpolated at each odometry timestamp.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REC = Path.home() / "LiteReality-scans/stray_atrium_raw/901a305fdf"
OUT = Path(__file__).resolve().parent
W, H = 1920, 1440


def lapvar(g: np.ndarray) -> float:
    return float(cv2.Laplacian(g, cv2.CV_32F, ksize=1).var())


def main() -> None:
    cmd = ["ffmpeg", "-v", "error", "-ignore_editlist", "1", "-i", str(REC / "rgb.mp4"),
           "-vf", "format=gray", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=W * H * 2)
    rows = []
    n = 0
    while True:
        buf = p.stdout.read(W * H)
        if len(buf) < W * H:
            break
        g = np.frombuffer(buf, np.uint8).reshape(H, W).astype(np.float32)
        g960 = cv2.resize(g, (960, 720), interpolation=cv2.INTER_AREA)
        g480 = cv2.resize(g, (480, 360), interpolation=cv2.INTER_AREA)
        lf, l9, l4 = lapvar(g), lapvar(g960), lapvar(g480)
        gx = float(np.abs(np.diff(g, axis=1)).mean())
        gy = float(np.abs(np.diff(g, axis=0)).mean())
        rows.append((n, lf, l9, l4, lf / (4 * l9 + 1e-9), gx, gy, float(g.mean())))
        n += 1
        if n % 300 == 0:
            print(n, file=sys.stderr, flush=True)
    p.wait()
    # packet sizes and picture types
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                         "frame=pkt_size,pict_type", "-of", "csv=p=0", str(REC / "rgb.mp4")],
                        capture_output=True, text=True).stdout.strip().splitlines()
    pk = [ln.split(",") for ln in pr]
    with open(OUT / "frames_blur.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_frame", "lap_full", "lap_960", "lap_480", "hf_ratio", "gx", "gy",
                    "mean_grey", "pkt_size", "pict_type"])
        for r in rows:
            k = r[0]
            ps, pt = (pk[k] + ["", ""])[:2] if k < len(pk) else ("", "")
            w.writerow([*(round(x, 4) if isinstance(x, float) else x for x in r), ps, pt])
    print("frames", n, "packets", len(pk))


if __name__ == "__main__":
    main()
