"""Stray Scanner recordings as LiteReality scans: thousands of sharp frames instead of ~51.

Stray Scanner (free on the App Store, Stray Robots) records the phone's whole ARKit session: the
camera video at 60 fps with a pose, intrinsics, a LiDAR depth map and its confidence for every
frame. This reads one, picks the training set (the sharpest frame of each stretch of the walk,
near-duplicate poses dropped) and writes it as our own scan folder, so `rgbd.load_frames`, the
splat room and everything else that reads a LiteReality Scanner scan read it unchanged.

The format, confirmed from the app's own source and docs (strayrobots/scanner @ ec3e1dc9d3,
2026-04-04; read 2026-11-23), not guessed:

  docs      https://github.com/strayrobots/scanner/blob/main/docs/format.md
  writer    https://github.com/strayrobots/scanner/blob/main/StrayScanner/Helpers/
            {DatasetEncoder,OdometryEncoder,VideoEncoder,DepthEncoder,ConfidenceEncoder}.swift
  reader    https://github.com/kekeblom/StrayVisualizer/blob/main/stray_visualize.py (the
            reference reader the app's README links: T_WC from the quaternion, Open3D extrinsic
            inv(T_WC), i.e. an OpenCV camera)

  <hash>/rgb.mp4          HEVC, `ARWorldTrackingConfiguration.videoFormat.imageResolution` (1920 x
                          1440 on the iPhones we use), landscape sensor orientation, no rotation
                          tag. One video frame per saved ARFrame, appended in the same serial
                          block as its odometry row (DatasetEncoder.add), so the k-th frame of the
                          stream is odometry row k (the row's ``frame`` column, the saved-frame
                          number). Its presentation time is NOT k / 60 but the ARKit frame counter
                          before the increment / 60 (``totalFrames``, VideoEncoder timescale 60):
                          -1/60 for the first saved frame, and later ones skip where the fps
                          divider or the 3-deep encoding semaphore dropped an ARFrame. The writer's
                          session starts at 0, so the first frame sits before the session start
                          (an edit list trims it): every decode here reads with
                          ``-ignore_editlist 1`` to keep it. Since that is the fragile spot, the
                          sharpness pass also records every decoded frame's PTS (``showinfo``) and
                          `pair_video` pairs video frames with rows by it when the counts differ,
                          refusing what it cannot explain. (History, `git log` of DatasetEncoder:
                          the -1 start and counter PTS since 9e470db, 2022-02; 4341d22, 2021-08,
                          wrote PTS = saved-frame number from 0; earlier versions used wall time.)
  depth/000000.png        uint16 millimetres, 256 x 192, the same field of view as the photo
                          (old versions: depth/000000.npy, millimetres too).
  confidence/000000.png   uint8 0..2 (ARKit's low / medium / high), 256 x 192.
  odometry.csv            header ``timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy,
                          distortion_center_x, distortion_center_y`` (older versions: the first 9
                          only). timestamp = ARFrame.timestamp (s, device uptime); frame = the
                          %06d name of the depth/confidence files; x y z = the translation of
                          ARKit's camera.transform (camera -> world, ARKit world: gravity-aligned,
                          Y up, metres, origin where the session started); q = q_WA * q_AC with
                          q_AC = (1, 0, 0, 0) (x, y, z, w): ARKit's camera rotation turned half a
                          turn about the camera's X, i.e. the SAME camera-to-world pose in the
                          OpenCV camera convention (x right, y down, z ahead), which is how
                          StrayVisualizer uses it. fx fy cx cy: ARFrame.camera.intrinsics at the
                          video's resolution, per frame (ARKit refocuses).
  camera_matrix.csv       3 x 3 K, row-major, of the LAST frame (kept for old readers; the
                          per-frame columns win when present).
  imu.csv                 timestamp, a_x a_y a_z (m/s^2), alpha_x alpha_y alpha_z (rad/s): unused.
  distortion/*.bin        optional float32 lens-distortion lookup tables: unused (ARKit's
                          captured image is what its intrinsics describe, as in our own scans).

  Not recorded: ARKit's tracking state. Relocalisation shows as a jump in the poses, so frames
  around a jump no handheld phone can make are dropped instead (`tracking_ok`).

Our scan format (what `rgbd.load_frames`, `evidence_kit.read_scan` and the LiteReality Scanner
write; checked against ~/LiteReality-scans/atrium_1059 and example-scans/tea_room):

  frame_00000.jpg         the photo, landscape 1920 x 1440 (written at JPEG quality 92)
  frame_00000.json        "cameraPoseARFrame": 16 numbers, the 4 x 4 camera -> ARKit world ROW-major
                          (translation at [3], [7], [11]), OpenGL camera (x right, y up, looking
                          down -z) = ARKit's camera.transform; "intrinsics": 9 numbers, K ROW-major
                          ([fx, 0, cx, 0, fy, cy, 0, 0, 1], cx ~ width / 2); "frame_index",
                          "time", "averageVelocity", "averageAngularVelocity". (The scanner's
                          "projectionMatrix" depends on its display orientation and viewport and
                          nothing here reads it, so it is not invented; "motionQuality" neither.)
                          Imports add "stray": where the frame came from and its sharpness.
  depth_00000.png         uint16 millimetres;  conf_00000.png  uint8 0..2
  pointcloud.pcd          ARKit-world points (ASCII PCD, x y z)
  scan_label.json         displayName / sceneName / scannerName / createdAt / uploadFolderName
  scene_alignment.json    which world the poses and the point cloud are in
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from litereality_agent.models.splat.rgbd import MAX_DEPTH, MIN_DEPTH, off_silhouettes

# Stray's q_AC: half a turn about the camera's X. Its own inverse: OpenCV <-> OpenGL camera.
CV_TO_GL = np.diag([1.0, -1.0, -1.0])

SCORE_WIDTH = 480          # sharpness is measured on the frame scaled to this width (pixels)
TARGET_FRAMES = 300
MIN_MOVE = 0.05            # metres: a frame closer than this to the last kept one ...
MIN_TURN_DEG = 3.0         # ... and turned less than this is a near-duplicate
# A pick is motion blur in a stretch with nothing sharper (a fast pan), dropped rather than
# trained on, when BOTH its sharpness is below this fraction of the sharp end (90th percentile)
# of the frames within +-BLUR_SPAN_S of it AND the camera was moving fast then (BLUR_TURN_RATE
# or BLUR_SPEED, from the poses). Never against the whole recording: the Laplacian measures
# texture as much as blur, and a steady look at a ceiling or a white wall scores a tenth of a
# bookshelf's while being perfectly sharp (on atrium_1059, all stills, a fifth of the frames
# score below 0.3 x the median). The motion gate keeps the edges of such a stretch too, whose
# time window still holds the textured frames around it.
BLUR_FLOOR = 0.3
BLUR_SPAN_S = 1.5
BLUR_PERCENTILE = 90
BLUR_TURN_RATE = 0.6       # rad/s (~35 deg/s: ~10 px of blur at 1920 wide in a 1/60 s exposure)
BLUR_SPEED = 0.5           # m/s
# The blur picker (`select_frames` with `blur_px`, import-stray --max-blur-px). The Laplacian at
# 480 px wide cannot see 2-4 px of motion blur at 1920 (results/diagnosis/blur: 28% of the q600
# training frames were blurred by 4+ px), so it scores the frame at full resolution and RELATIVE
# to the sharpest frame within +-BLUR_REL_SPAN_S (the same view a few frames on: texture cancels),
# and turns that ratio into a blur length in pixels at BLUR_REF_WIDTH by the curve measured on
# this phone's own still frames box-blurred by a known length (blur_stats.json "blur_calibration":
# Laplacian variance ratio vs box length; its 1 px point, still 1.0, is left out so the curve is
# strictly monotone).
FULL_SCORE_WIDTH = 1 << 16  # "score at the video's own width" (scan_video clamps to it)
BLUR_REL_SPAN_S = 0.25
BLUR_REF_WIDTH = 1920
BLUR_CURVE_PX = (0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
BLUR_CURVE_RATIO = (1.0, 0.6932, 0.5161, 0.4237, 0.2931, 0.2215, 0.1551, 0.122, 0.0828)
MAX_BLUR_PX = 2.0          # px at BLUR_REF_WIDTH: picks blurrier than this are not trained on
# Each window's pick may come from this many windows' width either side of it as well (never a
# frame another window took), so a window that falls on a fast pan takes a sharp frame next to it.
BLUR_WINDOW_GROW = 0.5
VIDEO_RATE = 60            # the writer's timescale: a frame's PTS x 60 = its ARKit frame counter
# Rows at the end of the odometry with no video frame that are accepted without --allow-frame-
# mismatch (the writer finished before the last append landed).
MAX_TAIL_MISSING = 2
# Poses no handheld phone makes between two 60 fps frames: ARKit relocalised (tracking was lost).
MAX_SPEED = 3.0            # m/s
MAX_TURN_RATE = 6.0        # rad/s (~340 deg/s)
JUMP_MARGIN = 0.25         # s of frames dropped either side of a jump
JPEG_QUALITY = 92
PCD_VOXEL = 0.02           # metres
MIN_CONFIDENCE = 2


@dataclass
class StrayFrame:
    row: int                 # saved-frame number (odometry ``frame``): its place in rgb.mp4
    name: str                # "000123": depth/<name>.png, confidence/<name>.png
    timestamp: float         # seconds (ARFrame.timestamp)
    c2w: np.ndarray          # 4x4 camera -> ARKit world, OpenGL camera (ARKit's camera.transform)
    K: np.ndarray            # 3x3 at the video's resolution
    depth: Path | None
    conf: Path | None


@dataclass
class StrayRecording:
    root: Path
    frames: list[StrayFrame]
    video: Path
    size: tuple[int, int]    # video (width, height)
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.root.name


# --------------------------------------------------------------------------------------------
# Poses
# --------------------------------------------------------------------------------------------

def quat_to_matrix(q) -> np.ndarray:
    """(x, y, z, w) -> 3x3 rotation."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (x, y, z, w), w >= 0."""
    m = np.asarray(R, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def pose_from_odometry(xyz, q) -> np.ndarray:
    """An odometry row's position and quaternion (OpenCV camera) -> ARKit's camera.transform
    (camera -> world, OpenGL camera), the pose our scans store."""
    c2w = np.eye(4)
    c2w[:3, :3] = quat_to_matrix(q) @ CV_TO_GL
    c2w[:3, 3] = xyz
    return c2w


def odometry_from_pose(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of `pose_from_odometry`: what Stray writes for an ARKit camera.transform."""
    return np.asarray(c2w[:3, 3], float), matrix_to_quat(c2w[:3, :3] @ CV_TO_GL)


def _angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Rotation angle (radians) between rotations, batched over the leading axis."""
    tr = np.einsum("...ij,...ij->...", Ra, Rb)        # trace(Ra^T Rb)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


# --------------------------------------------------------------------------------------------
# Reading a recording
# --------------------------------------------------------------------------------------------

def find_root(path: Path) -> Path:
    """The folder holding odometry.csv: `path` itself or one folder down (an exported zip's)."""
    path = Path(path)
    if (path / "odometry.csv").is_file():
        return path
    found = sorted(p.parent for p in path.glob("*/odometry.csv"))
    found = [p for p in found if not p.name.startswith(("__MACOSX", "."))]
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"{path}: not a Stray Scanner recording (no odometry.csv)")
    raise ValueError(f"{path}: holds {len(found)} recordings; name one of "
                     + ", ".join(p.name for p in found))


@contextlib.contextmanager
def open_recording(path: Path | str) -> Iterator[StrayRecording]:
    """A recording folder, or a .zip of one (unpacked to a temporary folder for the duration)."""
    path = Path(path).expanduser()
    if path.is_file() and zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory(prefix="stray-") as tmp:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmp)          # extractall drops absolute and ".." member paths
            root = find_root(Path(tmp))
            if root == Path(tmp):           # a zip of the folder's contents: name it after the zip
                named = Path(tmp) / path.stem
                named.mkdir()
                for child in list(Path(tmp).iterdir()):
                    if child != named:
                        child.rename(named / child.name)
                root = named
            yield read_recording(root)
        return
    if not path.is_dir():
        raise FileNotFoundError(f"{path}: no such recording folder or zip")
    yield read_recording(find_root(path))


def _read_camera_matrix(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        K = np.loadtxt(path, delimiter=",", dtype=float).reshape(3, 3)
    except (OSError, ValueError):
        return None
    return K if np.isfinite(K).all() and K[0, 0] > 0 else None


def read_odometry(path: Path) -> list[dict]:
    """odometry.csv as one dict per row: timestamp, frame (str), xyz, q and, when the file has
    them, fx fy cx cy. Header names are matched, so column order and newer extra columns do
    not matter; headerless legacy files (x y z qx qy qz qw, or with timestamp and frame first)
    are read by position."""
    with open(path, newline="") as fh:
        rows = [[c.strip() for c in r] for r in csv.reader(fh) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path}: empty")
    try:
        float(rows[0][0])
        header = None
    except ValueError:
        header = [h.lower() for h in rows[0]]
        rows = rows[1:]
    if header is None:
        width = len(rows[0])
        header = (["x", "y", "z", "qx", "qy", "qz", "qw"] if width == 7 else
                  ["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"])
    col = {h: i for i, h in enumerate(header)}
    missing = [k for k in ("x", "y", "z", "qx", "qy", "qz", "qw") if k not in col]
    if missing:
        raise ValueError(f"{path}: no {', '.join(missing)} column")

    def num(r, key):
        i = col.get(key)
        if i is None or i >= len(r) or r[i] == "":
            return None
        return float(r[i])

    out = []
    for n, r in enumerate(rows):
        try:
            ts = num(r, "timestamp")
            frame = r[col["frame"]] if "frame" in col and col["frame"] < len(r) else str(n)
            pose = [num(r, k) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")]
            if any(v is None for v in pose):
                continue      # a row cut short when the app stopped: no pose
            item = {"timestamp": n / 60.0 if ts is None else ts,
                    "frame": f"{int(float(frame)):06d}",
                    "xyz": np.array(pose[:3], float), "q": np.array(pose[3:], float)}
            intr = [num(r, k) for k in ("fx", "fy", "cx", "cy")]
        except (TypeError, ValueError, IndexError):
            continue
        if all(v is not None for v in intr):
            item["fx"], item["fy"], item["cx"], item["cy"] = intr
        out.append(item)
    return out


def read_recording(root: Path | str) -> StrayRecording:
    root = Path(root)
    video = root / "rgb.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"{root}: no rgb.mp4 (not a Stray Scanner recording?)")
    size = video_size(video)
    K_file = _read_camera_matrix(root / "camera_matrix.csv")
    warnings: list[str] = []
    frames = []
    for item in read_odometry(root / "odometry.csv"):
        row = int(item["frame"])        # the saved-frame number: the frame's place in rgb.mp4
        if "fx" in item:
            K = np.array([[item["fx"], 0.0, item["cx"]], [0.0, item["fy"], item["cy"]],
                          [0.0, 0.0, 1.0]])
        elif K_file is not None:
            K = K_file
        else:
            raise ValueError(f"{root}: no intrinsics (odometry has no fx..cy and no "
                             "camera_matrix.csv)")
        if not (np.isfinite(item["xyz"]).all() and np.isfinite(item["q"]).all()
                and np.linalg.norm(item["q"]) > 0.5):
            continue
        name = item["frame"]
        depth = next((p for p in (root / "depth" / f"{name}.png", root / "depth" / f"{name}.npy")
                      if p.is_file()), None)
        conf = root / "confidence" / f"{name}.png"
        frames.append(StrayFrame(row, name, item["timestamp"],
                                 pose_from_odometry(item["xyz"], item["q"]), K, depth,
                                 conf if conf.is_file() else None))
    if not frames:
        raise ValueError(f"{root}: odometry.csv has no usable poses")
    cx = float(np.median([f.K[0, 2] for f in frames]))
    if abs(cx - size[0] / 2) > 0.1 * size[0]:
        warnings.append(f"intrinsics cx {cx:.0f} is far from the video's centre "
                        f"{size[0] / 2:.0f}: are the video and odometry from one recording?")
    return StrayRecording(root, frames, video, size, warnings)


# --------------------------------------------------------------------------------------------
# Video (ffmpeg in a subprocess)
# --------------------------------------------------------------------------------------------

def ffmpeg_binary(name: str = "ffmpeg") -> str:
    found = shutil.which(name) or next(
        (p for p in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}", f"/usr/bin/{name}")
         if os.path.isfile(p)), None)
    if not found:
        raise RuntimeError(f"{name} not found: install ffmpeg (brew install ffmpeg / "
                           "apt install ffmpeg)")
    return found


@lru_cache(maxsize=None)
def _passthrough() -> tuple[str, ...]:
    """Every decoded frame out exactly once (no duplicates or drops to a frame rate)."""
    try:
        head = subprocess.run([ffmpeg_binary(), "-version"], capture_output=True, text=True,
                              timeout=30).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        head = ""
    m = re.search(r"version\s+n?(\d+)\.(\d+)", head)
    if m and (int(m.group(1)), int(m.group(2))) < (5, 1):
        return ("-vsync", "passthrough")
    return ("-fps_mode", "passthrough")


def _input(video: Path, loglevel: str = "error") -> list[str]:
    hw = ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []
    return [ffmpeg_binary(), "-hide_banner", "-loglevel", loglevel, "-nostdin", *hw,
            "-ignore_editlist", "1", "-i", str(video)]


def video_size(video: Path) -> tuple[int, int]:
    out = subprocess.run([ffmpeg_binary("ffprobe"), "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True, timeout=120)
    try:
        w, h = (int(v) for v in out.stdout.strip().split(",")[:2])
    except ValueError:
        raise ValueError(f"{video}: ffprobe found no video stream ({out.stderr.strip()})") from None
    return w, h


def _raw_frames(cmd: list[str], frame_bytes: int,
                stderr_sink: list[str] | None = None) -> Iterator[bytes]:
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                yield buf
        finally:
            proc.stdout.close()
            code = proc.wait()
        err.seek(0)
        text = err.read().decode(errors="replace")
        if code != 0:
            raise RuntimeError(f"ffmpeg failed ({code}): {text[-800:]}")
        if stderr_sink is not None:
            stderr_sink.append(text)


def _laplacian_variance(gray: np.ndarray) -> float:
    try:
        import cv2

        return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var())
    except ImportError:
        g = gray.astype(np.float32)
        lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
        return float(lap.var())


_SHOWINFO = re.compile(r"Parsed_showinfo\S*\s.*?\bn:\s*(\d+)\s.*?\bpts_time:\s*(\S+)")


@dataclass
class VideoScan:
    scores: np.ndarray            # sharpness of every decoded frame, in stream order
    pts: np.ndarray | None        # its presentation time (s), None if ffmpeg did not report it


def scan_video(video: Path, size: tuple[int, int], *, width: int = SCORE_WIDTH,
               progress: Callable[[int], None] | None = None) -> VideoScan:
    """Variance of the Laplacian of every frame, scaled to `width` (grey), and each frame's PTS
    (ffmpeg's ``showinfo`` on the small grey frame): one decode of the whole video. Higher is
    sharper; motion blur and defocus both lower it."""
    w = min(width, size[0])
    h = max(2, int(round(size[1] * w / size[0] / 2)) * 2)
    cmd = [*_input(video, "info"), "-vf", f"scale={w}:{h}:flags=area,format=gray,showinfo",
           *_passthrough(), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    scores: list[float] = []
    log: list[str] = []
    for buf in _raw_frames(cmd, w * h, stderr_sink=log):
        scores.append(_laplacian_variance(np.frombuffer(buf, np.uint8).reshape(h, w)))
        if progress and len(scores) % 600 == 0:
            progress(len(scores))
    pts: dict[int, float] = {}
    for m in _SHOWINFO.finditer(log[0] if log else ""):
        try:
            pts[int(m.group(1))] = float(m.group(2))
        except ValueError:
            continue
    times = (np.array([pts[i] for i in range(len(scores))])
             if len(pts) == len(scores) and all(i in pts for i in range(len(scores))) else None)
    return VideoScan(np.asarray(scores, dtype=np.float64), times)


def sharpness_scores(video: Path, size: tuple[int, int], *, width: int = SCORE_WIDTH,
                     progress: Callable[[int], None] | None = None) -> np.ndarray:
    """`scan_video`'s scores alone."""
    return scan_video(video, size, width=width, progress=progress).scores


@dataclass
class VideoPairing:
    video_of_row: dict[int, int]  # odometry row (saved-frame number) -> place in the stream
    rows_before: int              # rows whose frame is missing before the video's first frame
    rows_after: int               # rows whose frame is missing after its last
    note: str | None              # what was not one-to-one, for the report (None when it was)


def pair_video(pts: np.ndarray | None, rows, times, *, n_video_hint: int = 0,
               allow_mismatch: bool = False) -> VideoPairing:
    """Which decoded video frame belongs to which odometry row.

    One frame per row is the normal case: frame k is row k. When the counts differ the stream's
    own PTS decides, never the order alone (a frame lost at the start would put every photo one
    frame off its pose): the writer stamps the first saved frame -1/60, so a stream whose first
    PTS is >= 0 lost the rows whose ARKit counter (estimated from the timestamps, near the start
    only) is below it; the rest must then be missing at the end. Where the fps divider or the
    encoding semaphore skipped ARFrames the PTS gaps and the timestamp gaps must agree under the
    pairing, which catches a frame lost in the middle. Anything else raises ValueError unless
    `allow_mismatch` (then frames are paired from the start, as the counts allow).
    `n_video_hint` is the stream's frame count when `pts` is None.

    Assumes the writer of 2022-02 on (first PTS -1/60). A 2021-08..2022-02 recording (PTS =
    saved-frame number from 0) with a frame count that differs would be read as having lost its
    first frame; those are paired one-to-one whenever the counts agree, as they should."""
    rows = np.asarray(rows, int)
    times = np.asarray(times, float)
    order = np.argsort(rows)
    rows, times = rows[order], times[order]
    n_video = len(pts) if pts is not None else n_video_hint
    n_rows = int(rows[-1]) + 1 if len(rows) else 0
    if n_video == n_rows:
        return VideoPairing({int(r): int(r) for r in rows}, 0, 0, None)

    problem = None
    before = 0
    if pts is None:
        problem = "ffmpeg reported no per-frame PTS to pair them by"
        counter = np.zeros(0, int)
    else:
        counter = np.round(np.asarray(pts, float) * VIDEO_RATE).astype(int)
    if problem is None and n_video and counter[0] >= 0 and n_rows > n_video:
        if len(rows) and rows[0] == 0:
            # The writer's counter of the first rows: -1, then one per 1/60 s of ARKit time.
            est = -1 + np.round((times - times[0]) * VIDEO_RATE).astype(int)
            head = rows <= (n_rows - n_video)
            before = int(np.sum(est[head] < counter[0]))
        else:
            problem = "odometry row 0 is unreadable, so the stream's start cannot be anchored"
    after = n_rows - n_video - before
    if problem is None and after < 0:
        problem = f"the stream has {-after} frame(s) more than the odometry has rows"
    elif problem is None and after > MAX_TAIL_MISSING:
        problem = (f"{after} rows have no frame at the end of the stream: frames were lost "
                   "where the PTS cannot place them")
    mapping = {int(r): int(r) - before for r in rows if 0 <= int(r) - before < n_video}

    if problem is None:
        # Skipped ARFrames show in both streams; under the right pairing they line up.
        t_of = dict(zip(rows.tolist(), times.tolist()))
        events = agree = 0
        for k in range(n_video - 1):
            r = k + before
            if r not in t_of or r + 1 not in t_of:
                continue
            gap_v = int(counter[k + 1] - counter[k])
            gap_t = int(round((t_of[r + 1] - t_of[r]) * VIDEO_RATE))
            if gap_v > 1 or gap_t > 1:
                events += 1
                agree += gap_v == gap_t
        if events >= 3 and agree < 0.75 * events:
            problem = (f"the stream's skipped frames and the odometry's do not line up "
                       f"({agree} of {events}): a frame is missing in the middle")

    if problem is not None:
        msg = (f"rgb.mp4 has {n_video} frames but odometry.csv {n_rows} rows and {problem}. "
               "Pairing them by order would put photos off their poses; nothing was written "
               "(--allow-frame-mismatch pairs them from the first anyway)")
        if not allow_mismatch:
            raise ValueError(msg)
        mapping = {int(r): int(r) for r in rows if int(r) < n_video}
        return VideoPairing(mapping, 0, max(n_rows - n_video, 0),
                            f"{msg.split('. Pairing')[0]}: paired from the first frame "
                            "(--allow-frame-mismatch)")
    note = (f"rgb.mp4 has {n_video} frames for {n_rows} odometry rows: by its PTS (first "
            f"{int(counter[0]) if len(counter) else '?'}/60 s) the first {before} and the last "
            f"{after} rows have no frame and are skipped")
    return VideoPairing(mapping, before, after, note)


def decode_frames(video: Path, indices, size: tuple[int, int]) -> Iterator[tuple[int, np.ndarray]]:
    """The frames at `indices` (places in the stream, as `sharpness_scores` counts them), full
    resolution RGB, in stream order."""
    want = sorted({int(i) for i in indices})
    if not want:
        return
    w, h = size
    # Every frame is decoded and the unwanted ones dropped here: a `select` filter naming a
    # few hundred frames (eq(n,..)+eq(n,..)+...) makes ffmpeg fail ("Cannot allocate memory")
    # on a real 2400-frame recording, and one sequential decode is fast anyway. The stream is
    # read to its end (stopping early would close the pipe and fail ffmpeg).
    cmd = [*_input(video), *_passthrough(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    wanted = set(want)
    got = 0
    for n, buf in enumerate(_raw_frames(cmd, w * h * 3)):
        if n in wanted:
            yield n, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            got += 1
    if got < len(want):
        raise RuntimeError(f"{video}: decoded {got} of the {len(want)} chosen frames")


# --------------------------------------------------------------------------------------------
# Choosing the training set
# --------------------------------------------------------------------------------------------

def tracking_ok(times: np.ndarray, poses: np.ndarray, *, max_speed: float = MAX_SPEED,
                max_turn_rate: float = MAX_TURN_RATE, margin: float = JUMP_MARGIN) -> np.ndarray:
    """False around pose jumps no handheld phone makes (ARKit lost tracking and relocalised;
    Stray does not record ARKit's tracking state, so this is how a bad stretch shows)."""
    n = len(times)
    ok = np.ones(n, bool)
    if n < 2:
        return ok
    dt = np.maximum(np.diff(times), 1.0 / 120.0)
    speed = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) / dt
    turn = _angle(poses[:-1, :3, :3], poses[1:, :3, :3]) / dt
    for k in np.flatnonzero((speed > max_speed) | (turn > max_turn_rate)):
        t_jump = times[k + 1]
        ok &= np.abs(times - t_jump) > margin
        ok[k:k + 2] = False
    return ok


@dataclass
class Selection:
    rows: np.ndarray             # chosen positions in the recording's frame list, time order
    stats: dict


def relative_sharpness(times: np.ndarray, scores: np.ndarray,
                       span: float = BLUR_REL_SPAN_S) -> np.ndarray:
    """Each frame's sharpness over the sharpest within +-`span` seconds of it (1 = the sharpest
    of its moment). Every frame counts as a reference, usable or not: they show the same view."""
    t = np.asarray(times, float)
    s = np.asarray(scores, float)
    order = np.argsort(t, kind="stable")
    ts, ss = t[order], np.where(np.isfinite(s[order]), s[order], 0.0)
    lo = np.searchsorted(ts, ts - span, side="left")
    hi = np.searchsorted(ts, ts + span, side="right")
    best = np.array([ss[a:b].max() for a, b in zip(lo, hi)]) if len(ts) else np.zeros(0)
    out = np.empty(len(t))
    out[order] = np.where(best > 0, ss / np.maximum(best, 1e-12), 0.0)
    return out


def blur_px_from_ratio(ratio) -> np.ndarray:
    """A relative Laplacian-variance ratio as a motion-blur length in pixels at `BLUR_REF_WIDTH`
    (`BLUR_CURVE_RATIO` / `BLUR_CURVE_PX`, interpolated in log ratio; past the curve's end it
    stays at its last length)."""
    r = np.clip(np.asarray(ratio, float), 1e-6, 1.0)
    return np.interp(-np.log(r), -np.log(np.asarray(BLUR_CURVE_RATIO)), np.asarray(BLUR_CURVE_PX))


def blur_estimate(times: np.ndarray, scores: np.ndarray, *, scored_width: int,
                  span: float = BLUR_REL_SPAN_S) -> np.ndarray:
    """Motion blur of every frame, in pixels at `BLUR_REF_WIDTH`, from `scan_video` scores taken
    at `scored_width` (`relative_sharpness`, `blur_px_from_ratio`): the curve gives pixels at the
    width scored, scaled to `BLUR_REF_WIDTH`. It was measured at full resolution (1920): scores
    of a downscaled frame see less of the blur, and the scaling is then only a rough correction.
    Score at full resolution."""
    px = blur_px_from_ratio(relative_sharpness(times, scores, span))
    return px * (BLUR_REF_WIDTH / float(scored_width))


def _blur_picks(window: np.ndarray, progress: np.ndarray, target: int, blur: np.ndarray,
                sharp: np.ndarray, grow: float) -> np.ndarray:
    """One pick per window (positions into the candidate arrays): the least blurred candidate
    within the window widened by `grow` windows either side, never one already taken; ties go to
    the sharper. Windows choose cleanest first, so a sharp frame goes to the window it sits in
    before a neighbour can borrow it."""
    pos = np.asarray(progress, float) * target         # in window units
    by_pos = np.argsort(pos, kind="stable")
    sorted_pos = pos[by_pos]
    taken = np.zeros(len(pos), bool)
    best_in = {}
    for w in np.unique(window):
        inside = window == w
        best_in[int(w)] = float(blur[inside].min())
    picks = []
    for w in sorted(best_in, key=lambda k: (best_in[k], k)):
        a = np.searchsorted(sorted_pos, w - grow, side="left")
        b = np.searchsorted(sorted_pos, w + 1 + grow, side="left")
        near = by_pos[a:b]
        near = near[~taken[near]]
        if not len(near):
            continue
        k = int(near[np.lexsort((-sharp[near], blur[near]))[0]])
        taken[k] = True
        picks.append(k)
    return np.sort(np.asarray(picks, int))


def select_frames(times: np.ndarray, poses: np.ndarray, sharpness: np.ndarray,
                  usable: np.ndarray, target: int = TARGET_FRAMES, *,
                  min_move: float = MIN_MOVE, min_turn_deg: float = MIN_TURN_DEG,
                  blur_floor: float = BLUR_FLOOR, blur_px: np.ndarray | None = None,
                  max_blur_px: float | None = None,
                  window_grow: float = BLUR_WINDOW_GROW) -> Selection:
    """The training set: the recording cut into `target` windows of equal progress, where
    progress is half elapsed time and half camera motion (metres walked + angle turned, 10 cm ~
    10 degrees), so a slow look around a corner gets as many frames as a quick walk past a
    wall; the sharpest usable frame of each window; then, in time order, a frame within
    `min_move` AND `min_turn_deg` of the last one kept is dropped (the phone stood still), as
    is a pick taken while the camera moved fast (`BLUR_TURN_RATE`, `BLUR_SPEED`) and blurrier
    than `blur_floor` x the sharp end of the frames within `BLUR_SPAN_S` of it (a fast pan with
    no sharp frame; a steady look at a plain surface keeps its sharpest frames however low the
    Laplacian scores texture-less views).

    With `blur_px` (each frame's motion blur in pixels at `BLUR_REF_WIDTH`, `blur_estimate`) the
    picks are the LEAST BLURRED frames instead: each window takes the least blurred usable frame
    within it or `window_grow` windows either side (`_blur_picks`), and with `max_blur_px` no
    frame blurrier than that is a candidate at all (a window with none stays empty: fewer, sharp
    frames rather than smeared ones). The `blur_floor` rule is then not used, and a near-duplicate
    pair keeps its less blurred frame."""
    times = np.asarray(times, float)
    sharp = np.asarray(sharpness, float)
    ok = np.asarray(usable, bool) & np.isfinite(sharp)
    blur = None
    over = np.zeros(len(times), bool)
    if blur_px is not None:
        blur = np.asarray(blur_px, float)
        ok &= np.isfinite(blur)
        if max_blur_px is not None:
            over = ok & (blur > max_blur_px)
            ok &= ~over
    idx = np.flatnonzero(ok)
    stats = {"candidates": int(len(idx)), "recording_frames": int(len(times))}
    if blur is not None:
        stats.update(picker="blur", max_blur_px=max_blur_px, over_max_blur=int(over.sum()),
                     window_grow=window_grow)
    if len(idx) == 0 or target <= 0:
        return Selection(np.zeros(0, int), {**stats, "kept": 0})
    P = poses[idx]
    step_m = np.r_[0.0, np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1)]
    step_r = np.r_[0.0, _angle(P[:-1, :3, :3], P[1:, :3, :3])]
    motion = np.cumsum(step_m / 0.10 + step_r / np.deg2rad(10.0))
    t = times[idx] - times[idx][0]
    t_norm = t / t[-1] if t[-1] > 0 else np.linspace(0, 1, len(idx))
    progress = (0.5 * t_norm + 0.5 * motion / motion[-1]) if motion[-1] > 0 else t_norm
    window = np.minimum((progress * target).astype(int), target - 1)

    if blur is not None:
        firsts = np.unique(window)
        picks = idx[_blur_picks(window, progress, target, blur[idx], sharp[idx], window_grow)]
        blurry = np.zeros(0, bool)
        better = -blur                     # a near-duplicate pair keeps the less blurred
    else:
        # The sharpest candidate of each window: sort by (window, -sharpness), first of each run.
        order = np.lexsort((-sharp[idx], window))
        firsts = order[np.r_[True, window[order][1:] != window[order][:-1]]]
        picks = np.sort(idx[firsts])

        t_usable = times[idx]
        s_usable = sharp[idx]
        lo = np.searchsorted(t_usable, times[picks] - BLUR_SPAN_S, side="left")
        hi = np.searchsorted(t_usable, times[picks] + BLUR_SPAN_S, side="right")
        floor = np.array([blur_floor * np.percentile(s_usable[a:b], BLUR_PERCENTILE)
                          for a, b in zip(lo, hi)])
        speed, turn = velocities(times, poses)
        moving = (turn[picks] > BLUR_TURN_RATE) | (speed[picks] > BLUR_SPEED)
        blurry = (sharp[picks] < floor) & moving
        picks = picks[~blurry]
        better = sharp

    min_turn = np.deg2rad(min_turn_deg)

    def near(a: int, b: int) -> bool:
        return bool(np.linalg.norm(poses[a, :3, 3] - poses[b, :3, 3]) < min_move
                    and _angle(poses[a, :3, :3], poses[b, :3, :3]) < min_turn)

    kept: list[int] = []
    for i in picks:
        if kept and near(int(i), kept[-1]):
            # One view of the spot: the sharper of the two (a window edge can split a pause, its
            # blurry start in one window and its sharp middle in the next).
            if better[i] > better[kept[-1]] and not (len(kept) > 1 and near(int(i), kept[-2])):
                kept[-1] = int(i)
            continue
        kept.append(int(i))
    rows = np.asarray(kept, int)
    s_all, s_kept = sharp[idx], sharp[rows] if len(rows) else np.zeros(1)
    stats.update({
        "windows": int(len(firsts)), "blurry_dropped": int(blurry.sum()),
        "duplicates_dropped": int(len(picks) - len(rows)), "kept": int(len(rows)),
        "sharpness": {"all_median": round(float(np.median(s_all)), 2),
                      "all_p10": round(float(np.percentile(s_all, 10)), 2),
                      "kept_median": round(float(np.median(s_kept)), 2),
                      "kept_min": round(float(np.min(s_kept)), 2),
                      "kept_percentile_of_all": round(float(
                          100.0 * np.mean(s_all <= np.median(s_kept))), 1)},
    })
    if blur is not None:
        b_all, b_kept = blur[idx], blur[rows] if len(rows) else np.zeros(1)
        stats["blur_px"] = {"candidates_median": round(float(np.median(b_all)), 2),
                            "kept_median": round(float(np.median(b_kept)), 2),
                            "kept_p90": round(float(np.percentile(b_kept, 90)), 2),
                            "kept_max": round(float(np.max(b_kept)), 2),
                            "recording_median": round(float(np.nanmedian(blur)), 2),
                            "recording_frac_le_max": (round(float(np.mean(blur <= max_blur_px)), 3)
                                                      if max_blur_px is not None else None)}
    return Selection(rows, stats)


# --------------------------------------------------------------------------------------------
# Depth -> points
# --------------------------------------------------------------------------------------------

def read_depth(frame: StrayFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Depth in metres (float32) and confidence (None if absent or another size)."""
    from PIL import Image

    if frame.depth is None:
        raise FileNotFoundError(f"frame {frame.name}: no depth map")
    if frame.depth.suffix == ".npy":
        depth = np.load(frame.depth).astype(np.float32) / 1000.0
    else:
        with Image.open(frame.depth) as im:
            depth = np.asarray(im, dtype=np.float32) / 1000.0
    conf = None
    if frame.conf is not None:
        with Image.open(frame.conf) as im:
            c = np.asarray(im)
        conf = c if c.shape == depth.shape else None
    return depth, conf


def frame_points(frame: StrayFrame, size: tuple[int, int], *, stride: int = 2,
                 min_confidence: int = MIN_CONFIDENCE, max_depth: float = MAX_DEPTH,
                 c2w: np.ndarray | None = None) -> np.ndarray:
    """The frame's confident LiDAR samples (every `stride`-th, off silhouettes) in its world:
    ARKit's, or `c2w`'s if given. Same back-projection as `rgbd` (OpenGL camera, pixel centres)."""
    depth, conf = read_depth(frame)
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > MIN_DEPTH) & (depth < max_depth)
    if conf is not None:
        valid &= conf >= min_confidence
    valid &= off_silhouettes(np.where(np.isfinite(depth), depth, 0.0))
    sub = np.zeros_like(valid)
    sub[::stride, ::stride] = True
    vv, uu = np.nonzero(valid & sub)
    if len(vv) == 0:
        return np.zeros((0, 3), np.float32)
    sx, sy = w / size[0], h / size[1]
    fx, fy = frame.K[0, 0] * sx, frame.K[1, 1] * sy
    cx, cy = (frame.K[0, 2] + 0.5) * sx - 0.5, (frame.K[1, 2] + 0.5) * sy - 0.5
    z = depth[vv, uu]
    cam = np.stack([(uu - cx) / fx * z, -(vv - cy) / fy * z, -z], axis=1)
    pose = frame.c2w if c2w is None else c2w
    with np.errstate(all="ignore"):     # Accelerate's matmul flags spurious overflows on macOS
        return (cam @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """One point per occupied `voxel`-sized cell: the mean of the points in it."""
    if len(points) == 0:
        return points.reshape(0, 3).astype(np.float32)
    key = np.floor(points / voxel).astype(np.int64)
    key -= key.min(0)
    span = key.max(0) + 1
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    out = np.zeros((len(counts), 3), np.float64)
    for d in range(3):
        out[:, d] = np.bincount(inverse, weights=points[:, d], minlength=len(counts))
    return (out / counts[:, None]).astype(np.float32)


def fuse_points(frames: list[StrayFrame], size: tuple[int, int], *, voxel: float = PCD_VOXEL,
                transform: np.ndarray | None = None, stride: int = 2) -> np.ndarray:
    """Every frame's confident depth, fused on a `voxel` grid, in ARKit's world (or moved by the
    4x4 `transform`)."""
    parts, batch = [], []
    for f in frames:
        if f.depth is None:
            continue
        c2w = f.c2w if transform is None else transform @ f.c2w
        batch.append(frame_points(f, size, stride=stride, c2w=c2w))
        if len(batch) >= 40:
            parts.append(voxel_downsample(np.concatenate(batch), voxel))
            batch = []
    if batch:
        parts.append(voxel_downsample(np.concatenate(batch), voxel))
    if not parts:
        return np.zeros((0, 3), np.float32)
    return voxel_downsample(np.concatenate(parts), voxel)


def write_pcd(path: Path, points: np.ndarray, generator: str = "litereality import-stray") -> None:
    pts = np.asarray(points, np.float32).reshape(-1, 3)
    header = (f"# .PCD v0.7 - Point Cloud Data file generated by {generator}\n"
              "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
              f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\n"
              "DATA ascii\n")
    with open(path, "w", encoding="ascii") as fh:
        fh.write(header)
        np.savetxt(fh, pts, fmt="%.5f")


def read_pcd(path: Path) -> np.ndarray:
    """x y z of an ASCII or binary (uncompressed) PCD."""
    with open(path, "rb") as fh:
        meta: dict[str, list[str]] = {}
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: no DATA line")
            text = line.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            key, *vals = text.split()
            meta[key.upper()] = vals
            if key.upper() == "DATA":
                break
        fields = meta.get("FIELDS", [])
        kind = meta["DATA"][0].lower()
        n = int(meta.get("POINTS", meta.get("WIDTH", ["0"]))[0])
        if kind == "ascii":
            arr = np.loadtxt(fh, dtype=np.float64, ndmin=2)
        elif kind == "binary":
            sizes = [int(s) for s in meta.get("SIZE", [])]
            types = meta.get("TYPE", [])
            counts = [int(c) for c in meta.get("COUNT", ["1"] * len(fields))]
            dt = np.dtype([(f"{name}{k}" if c > 1 else name,
                            {"F": "f", "I": "i", "U": "u"}[t] + str(s))
                           for name, s, t, c in zip(fields, sizes, types, counts)
                           for k in range(c)])
            raw = np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
            arr = np.stack([raw[name].astype(np.float64) for name in ("x", "y", "z")], axis=1)
            fields = ["x", "y", "z"]
        else:
            raise ValueError(f"{path}: PCD DATA {kind} is not supported")
    ix = [fields.index(a) for a in ("x", "y", "z")]
    pts = arr[:, ix].astype(np.float32)
    return pts[np.isfinite(pts).all(1)]


# --------------------------------------------------------------------------------------------
# Writing our scan
# --------------------------------------------------------------------------------------------

def velocities(times: np.ndarray, poses: np.ndarray, span: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Speed (m/s) and turn rate (rad/s) of each frame over +-`span` frames."""
    n = len(times)
    lo = np.clip(np.arange(n) - span, 0, n - 1)
    hi = np.clip(np.arange(n) + span, 0, n - 1)
    dt = np.maximum(times[hi] - times[lo], 1e-6)
    speed = np.linalg.norm(poses[hi, :3, 3] - poses[lo, :3, 3], axis=1) / dt
    turn = _angle(poses[lo, :3, :3], poses[hi, :3, :3]) / dt
    same = hi == lo
    speed[same], turn[same] = 0.0, 0.0
    return speed, turn


def frame_json(index: int, c2w: np.ndarray, K: np.ndarray, time: float, speed: float,
               turn: float, source: dict) -> dict:
    """One frame_XXXXX.json, laid out as the LiteReality Scanner writes it."""
    return {"averageAngularVelocity": round(float(turn), 6),
            "averageVelocity": round(float(speed), 6),
            "cameraPoseARFrame": [float(v) for v in np.asarray(c2w, float).reshape(-1)],
            "frame_index": int(index),
            "intrinsics": [float(v) for v in np.asarray(K, float).reshape(-1)],
            "time": float(time),
            "stray": source}


def write_depth(frame: StrayFrame, path: Path) -> None:
    """depth_XXXXX.png, uint16 millimetres: the recording's own PNG copied byte for byte."""
    if frame.depth is None:
        raise FileNotFoundError(f"frame {frame.name}: no depth map")
    if frame.depth.suffix == ".png":
        shutil.copyfile(frame.depth, path)
        return
    from PIL import Image

    mm = np.clip(np.nan_to_num(np.load(frame.depth)), 0, 65535).astype(np.uint16)
    Image.fromarray(mm).save(path)


def scan_label(name: str, recording: str) -> dict:
    return {"createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "displayName": f"Stray Scanner - {name}",
            "scannerName": "Stray Scanner",
            "sceneName": name,
            "uploadFolderName": recording}


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
