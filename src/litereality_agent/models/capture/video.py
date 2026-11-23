"""Any phone video as a LiteReality scan: no LiDAR, no special app, poses and depth from a model.

Someone films their room with the ordinary Camera app. This picks the training frames from the
video, has MapAnything (`deploy/modal/geometry`, app `litereality-geometry`: open Apache-2.0
weights, a feed-forward multi-view model) estimate every frame's camera pose, intrinsics and
metric depth on one Modal GPU, turns the result into our convention, levels it (gravity), and,
with ``align_to``, moves it into the room's LiteReality Scanner ARKit world by a similarity
(yaw + translation + scale, `models/capture/sim3`). It writes the same scan folder the Stray
importer writes, so `rgbd.load_frames`, the splat room and the rest read it unchanged.

    decode    ffprobe for the size, rotation and colour; ffmpeg decodes (VideoToolbox on macOS,
              HEVC included), applies the rotation the phone recorded (a portrait iPhone video is
              landscape-coded with a -90 display matrix) and scales. HDR (an iPhone's HLG / Dolby
              Vision 8.4) is read as its SDR-compatible signal, not tone mapped: HLG is built to
              look right that way, and the geometry model needs no more.
    score     one decode of every frame at `SCORE_LONG` px (grey): the variance of the Laplacian
              (sharpness, `stray._laplacian_variance`), the PTS, and a 32 px thumbnail per frame.
    select    no poses yet, so progress through the video is half elapsed time and half
              appearance change (the summed thumbnail differences between consecutive frames,
              the stand-in for camera motion); the sharpest frame of each of `frames` windows;
              a pick blurrier than `stray.BLUR_FLOOR` x the sharp end of its +-1.5 s is dropped,
              and so is one whose thumbnail is within `DUP_DIFF` of the last kept one (the
              camera stood still; the sharper one is kept).
    model     the picks at `PHOTO_LONG` px (1920 x 1080 / 1080 x 1920) are written as the scan's
              photos; copies at `UPLOAD_LONG` px go to `reconstruct` (the model works at 518 px).
              It returns OpenCV camera -> world poses in its own world (view 0's camera), K at its
              own 518-px crop (`prep` says which crop; OpenCV pinhole, pixel centres at integer
              coordinates) and z-depth + confidence there.
    convert   K onto the photo: shift by the crop, then scale by photo / resized per axis about
              the image's corner (c -> (c + 0.5) a - 0.5, which keeps integer pixel centres).
              Poses: OpenCV -> OpenGL camera (x right, y up, -z ahead: ARKit's camera.transform)
              by flipping the camera's y and z axes.
    gravity   `sim3.estimate_up` on the model's cloud, then `sim3.level_transform`: +Y up, the
              first camera at the origin looking down -Z (as ARKit starts a session).
    align     `sim3.align_sim3` onto the target scan's cloud; the import is refused when the
              residual, the inlier share or the heading margin is not convincing (as the Stray
              importer's). The scale found there multiplies every depth and camera position.
    check     the focal length: the model's calibration spread across views and, from the photos
              alone, `selfcal.estimate_focal` (Mendonca-Cipolla on feature matches): a warning
              when they disagree with the model's K. With a target, every frame's depth is
              registered onto the target's surface (`video_eval.frame_check`): frames that
              need a large correction to fit it (a mirror, a window, a lost pose) are dropped
              unless `keep_bad_frames`; frames that do not register at all are kept (the target
              may not cover what they see); every frame's result is in its JSON (`videoCheck`). It cannot see a focal error: a too-short focal moves a camera
              toward what it sees and shrinks its depth with it (a dolly zoom), and the frame's
              geometry still fits the surface.
    write     frame_XXXXX.jpg/.json, depth_XXXXX.png (uint16 mm at `DEPTH_LONG` px on the
              photo's field of view, 0 = no estimate), conf_XXXXX.png (0..2 from the model's
              confidence: its bottom `CONF_LOW` -> 0, next to `CONF_MID` -> 1, the rest 2),
              pointcloud.pcd (the confident depth fused on a 2 cm grid), scan_label.json,
              scene_alignment.json, room.usdz + roomplan/ from the target, import_report.json,
              and video_geometry/ (the model's raw reply, reused by a re-import of the same
              picks so changing the alignment costs no GPU).
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from litereality_agent.models.capture import sim3, stray

APP_NAME = "litereality-geometry"
FUNCTION_NAME = "reconstruct"
TARGET_FRAMES = 200
SCORE_LONG = 480             # sharpness / thumbnails are measured at this long side (px)
THUMB_LONG = 32
PHOTO_LONG = 1920            # the scan's photos
UPLOAD_LONG = 1036           # what the model gets (2 x its 518-px input: it resizes itself)
DEPTH_LONG = 256             # depth_*.png / conf_*.png long side (256 x 144 for 16:9, as LiDAR)
DUP_DIFF = 0.12              # thumbnails closer than this (mean |a - b|, unit-variance) repeat
CONF_LOW, CONF_MID = 0.10, 0.30
CLOUD_MAX_DEPTH = 6.0        # metres: farther model depth is left out of gravity / alignment
CLOUD_STRIDE = 3
JPEG_QUALITY = stray.JPEG_QUALITY
UPLOAD_QUALITY = 92
RAW_DIR = "video_geometry"
REPORT = "import_report.json"
SCAN_FILES = ("frame_*.jpg", "frame_*.json", "depth_*.png", "conf_*.png")
USDZ_CANDIDATES = ("room.usdz", "roomplan/room.usdz")
CV_TO_GL4 = np.diag([1.0, -1.0, -1.0, 1.0])
LOW_INLIERS = 0.7
FX_SPREAD_WARN = 0.05        # the model's per-view fx disagree by more than this (std / median)
SELFCAL_WARN = 0.10          # self-calibration and the model's fx disagree by more than this
DROP_POSITION, DROP_DEG = 0.20, 10.0   # a frame's registration correction above these: dropped


# --------------------------------------------------------------------------------------------
# Probe / decode
# --------------------------------------------------------------------------------------------

@dataclass
class VideoInfo:
    path: Path
    coded: tuple[int, int]           # as stored
    size: tuple[int, int]            # as displayed (rotation applied)
    rotation: int                    # degrees the display matrix turns it
    fps: float
    frames: int | None
    duration: float | None
    codec: str
    color_transfer: str | None
    pix_fmt: str | None

    @property
    def hdr(self) -> bool:
        return self.color_transfer in ("arib-std-b67", "smpte2084")

    def summary(self) -> dict:
        return {"coded": list(self.coded), "size": list(self.size), "rotation": self.rotation,
                "fps": round(self.fps, 3), "frames": self.frames, "duration_s": self.duration,
                "codec": self.codec, "color_transfer": self.color_transfer,
                "pix_fmt": self.pix_fmt, "hdr": self.hdr}


def _rate(text: str | None) -> float:
    try:
        num, _, den = (text or "0/1").partition("/")
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(video: Path | str) -> VideoInfo:
    path = Path(video).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such video")
    out = subprocess.run([stray.ffmpeg_binary("ffprobe"), "-v", "error", "-select_streams", "v:0",
                          "-show_streams", "-show_format", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=120)
    try:
        doc = json.loads(out.stdout)
        s = doc["streams"][0]
    except (ValueError, KeyError, IndexError):
        raise ValueError(f"{path}: ffprobe found no video stream ({out.stderr.strip()[-300:]})"
                         ) from None
    rotation = 0
    for side in s.get("side_data_list") or []:
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    if not rotation and (s.get("tags") or {}).get("rotate"):
        rotation = -int(float(s["tags"]["rotate"]))   # the old tag turns the other way
    coded = (int(s["width"]), int(s["height"]))
    size = coded[::-1] if abs(rotation) % 180 == 90 else coded
    try:
        n = int(s.get("nb_frames")) if s.get("nb_frames") else None
    except ValueError:
        n = None
    dur = s.get("duration") or (doc.get("format") or {}).get("duration")
    return VideoInfo(path, coded, size, rotation, _rate(s.get("avg_frame_rate"))
                     or _rate(s.get("r_frame_rate")), n, float(dur) if dur else None,
                     str(s.get("codec_name")), s.get("color_transfer"), s.get("pix_fmt"))


def scaled_size(size: tuple[int, int], long_side: int) -> tuple[int, int]:
    """`size` with its long side at most `long_side`, both sides even."""
    w, h = size
    k = min(1.0, long_side / max(w, h))
    return max(2, int(round(w * k / 2)) * 2), max(2, int(round(h * k / 2)) * 2)


def _input(video: Path, loglevel: str = "error") -> list[str]:
    """ffmpeg reading `video` with the display rotation applied (ffmpeg's default autorotate)
    and every frame out once. Unlike Stray's rgb.mp4 the edit list is honoured: a phone video's
    edit list is the clip the user saw."""
    hw = ["-hwaccel", "videotoolbox"] if sys.platform == "darwin" else []
    return [stray.ffmpeg_binary(), "-hide_banner", "-loglevel", loglevel, "-nostdin", *hw,
            "-i", str(video)]


@dataclass
class VideoScan:
    scores: np.ndarray        # sharpness per decoded frame
    times: np.ndarray         # PTS seconds per frame (index / fps when ffmpeg did not say)
    thumbs: np.ndarray        # (n, th, tw) float32, each frame normalised to mean 0, std 1


def scan_video(info: VideoInfo, *, long_side: int = SCORE_LONG,
               progress: Callable[[int], None] | None = None) -> VideoScan:
    """One decode of every frame (small, grey): sharpness, PTS and a thumbnail each."""
    w, h = scaled_size(info.size, long_side)
    tw, th = scaled_size(info.size, THUMB_LONG)
    cmd = [*_input(info.path, "info"), "-vf",
           f"scale={w}:{h}:flags=area,format=gray,showinfo", *stray._passthrough(),
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    scores, thumbs = [], []
    log: list[str] = []
    ry, rx = h // th, w // tw
    for buf in stray._raw_frames(cmd, w * h, stderr_sink=log):
        g = np.frombuffer(buf, np.uint8).reshape(h, w)
        scores.append(stray._laplacian_variance(g))
        small = g[:th * ry, :tw * rx].reshape(th, ry, tw, rx).mean((1, 3)).astype(np.float32)
        small -= small.mean()
        small /= max(float(small.std()), 1e-3)
        thumbs.append(small)
        if progress and len(scores) % 600 == 0:
            progress(len(scores))
    if not scores:
        raise ValueError(f"{info.path}: ffmpeg decoded no frames")
    pts: dict[int, float] = {}
    for m in stray._SHOWINFO.finditer(log[0] if log else ""):
        try:
            pts[int(m.group(1))] = float(m.group(2))
        except ValueError:
            continue
    n = len(scores)
    if len(pts) == n and all(i in pts for i in range(n)):
        times = np.array([pts[i] for i in range(n)])
    else:
        times = np.arange(n) / (info.fps or 30.0)
    return VideoScan(np.asarray(scores, np.float64), times, np.stack(thumbs))


def decode_frames(info: VideoInfo, indices, size: tuple[int, int]):
    """(index, RGB uint8 H x W x 3 at `size`) for the frames at `indices` (places in the stream,
    counted as `scan_video` counts them), in stream order. One sequential decode."""
    want = set(int(i) for i in indices)
    if not want:
        return
    w, h = size
    cmd = [*_input(info.path), "-vf", f"scale={w}:{h}:flags=lanczos", *stray._passthrough(),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    got = 0
    for n, buf in enumerate(stray._raw_frames(cmd, w * h * 3)):
        if n in want:
            yield n, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            got += 1
    if got < len(want):
        raise RuntimeError(f"{info.path}: decoded {got} of the {len(want)} chosen frames")


# --------------------------------------------------------------------------------------------
# Choosing frames without poses
# --------------------------------------------------------------------------------------------

def thumb_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


@dataclass
class Selection:
    indices: np.ndarray
    stats: dict


def select_frames(times: np.ndarray, sharpness: np.ndarray, thumbs: np.ndarray,
                  target: int = TARGET_FRAMES, *, dup_diff: float = DUP_DIFF,
                  blur_floor: float = stray.BLUR_FLOOR, max_rounds: int = 4) -> Selection:
    """The training frames of a video with no poses (see the module docstring). Blurry and
    repeated picks are dropped, so the windows are made finer (up to 3 x `target`) until about
    `target` frames survive."""
    times = np.asarray(times, float)
    sharp = np.asarray(sharpness, float)
    n = len(times)
    if n == 0 or target <= 0:
        return Selection(np.zeros(0, int), {"video_frames": int(n), "kept": 0})
    windows = min(target, n)
    sel = _select(times, sharp, thumbs, windows, dup_diff, blur_floor)
    for _ in range(max_rounds):
        kept = sel.stats["kept"]
        if kept >= 0.95 * target or windows >= min(3 * target, n) or kept == 0:
            break
        windows = min(int(np.ceil(windows * target / kept)), 3 * target, n)
        sel = _select(times, sharp, thumbs, windows, dup_diff, blur_floor)
    sel.stats["target"] = int(target)
    return sel


def _select(times, sharp, thumbs, target, dup_diff, blur_floor) -> Selection:
    n = len(times)
    stats = {"video_frames": int(n)}
    change = np.r_[0.0, np.mean(np.abs(np.diff(thumbs, axis=0)), axis=(1, 2))]
    motion = np.cumsum(change)
    t = times - times[0]
    t_norm = t / t[-1] if t[-1] > 0 else np.linspace(0, 1, n)
    progress = (0.5 * t_norm + 0.5 * motion / motion[-1]) if motion[-1] > 0 else t_norm
    window = np.minimum((progress * target).astype(int), target - 1)
    order = np.lexsort((-sharp, window))
    firsts = order[np.r_[True, window[order][1:] != window[order][:-1]]]
    picks = np.sort(firsts)

    lo = np.searchsorted(times, times[picks] - stray.BLUR_SPAN_S, side="left")
    hi = np.searchsorted(times, times[picks] + stray.BLUR_SPAN_S, side="right")
    floor = np.array([blur_floor * np.percentile(sharp[a:b], stray.BLUR_PERCENTILE)
                      for a, b in zip(lo, hi)])
    blurry = sharp[picks] < floor
    picks = picks[~blurry]

    kept: list[int] = []
    for i in picks:
        i = int(i)
        if kept and thumb_diff(thumbs[i], thumbs[kept[-1]]) < dup_diff:
            if sharp[i] > sharp[kept[-1]] and not (
                    len(kept) > 1 and thumb_diff(thumbs[i], thumbs[kept[-2]]) < dup_diff):
                kept[-1] = i
            continue
        kept.append(i)
    idx = np.asarray(kept, int)
    s_kept = sharp[idx] if len(idx) else np.zeros(1)
    stats.update({
        "windows": int(len(firsts)), "blurry_dropped": int(blurry.sum()),
        "duplicates_dropped": int(len(picks) - len(idx)), "kept": int(len(idx)),
        "appearance_change_total": round(float(motion[-1]), 2),
        "sharpness": {"all_median": round(float(np.median(sharp)), 2),
                      "kept_median": round(float(np.median(s_kept)), 2),
                      "kept_min": round(float(np.min(s_kept)), 2),
                      "kept_percentile_of_all": round(float(
                          100.0 * np.mean(sharp <= np.median(s_kept))), 1)},
    })
    return Selection(idx, stats)


# --------------------------------------------------------------------------------------------
# Model output -> our convention
# --------------------------------------------------------------------------------------------

def model_K_to_photo(K: np.ndarray, prep: dict, photo_size: tuple[int, int]) -> np.ndarray:
    """Intrinsics at the model's crop -> at the photo. `prep` = {"size": the image the model was
    sent, "resized": what it was resized to, "crop": [left, top] of the crop}; the image sent is
    the photo resized (same field of view), so photo = resized x (W / rw, H / rh) per axis."""
    rw, rh = prep["resized"]
    left, top = prep["crop"]
    W, H = photo_size
    ax, ay = W / rw, H / rh
    out = np.array(K, dtype=float).copy()
    cx, cy = out[0, 2] + left, out[1, 2] + top
    out[0, 0] *= ax
    out[1, 1] *= ay
    out[0, 1] *= ax
    out[0, 2] = (cx + 0.5) * ax - 0.5
    out[1, 2] = (cy + 0.5) * ay - 0.5
    return out


def photo_K_to_model(K: np.ndarray, prep: dict, photo_size: tuple[int, int]) -> np.ndarray:
    """The inverse of `model_K_to_photo`."""
    rw, rh = prep["resized"]
    left, top = prep["crop"]
    W, H = photo_size
    ax, ay = W / rw, H / rh
    out = np.array(K, dtype=float).copy()
    out[0, 0] /= ax
    out[1, 1] /= ay
    out[0, 1] /= ax
    out[0, 2] = (out[0, 2] + 0.5) / ax - 0.5 - left
    out[1, 2] = (out[1, 2] + 0.5) / ay - 0.5 - top
    return out


def depth_grid_size(photo_size: tuple[int, int], long_side: int = DEPTH_LONG) -> tuple[int, int]:
    W, H = photo_size
    k = long_side / max(W, H)
    return max(1, int(round(W * k))), max(1, int(round(H * k)))


def model_map_to_photo_grid(grid: np.ndarray, prep: dict, out_size: tuple[int, int],
                            fill: float = 0.0) -> np.ndarray:
    """A per-pixel map at the model's crop resampled (nearest) onto an `out_size` grid covering
    the photo's whole field of view; cells outside the crop get `fill`."""
    rw, rh = prep["resized"]
    left, top = prep["crop"]
    h, w = grid.shape[:2]
    dw, dh = out_size
    xm = (np.arange(dw) + 0.5) * rw / dw - 0.5 - left
    ym = (np.arange(dh) + 0.5) * rh / dh - 0.5 - top
    xi, yi = np.round(xm).astype(int), np.round(ym).astype(int)
    okx, oky = (xi >= 0) & (xi < w), (yi >= 0) & (yi < h)
    out = np.full((dh, dw) + grid.shape[2:], fill, dtype=grid.dtype)
    sub = grid[np.clip(yi, 0, h - 1)][:, np.clip(xi, 0, w - 1)]
    mask = oky[:, None] & okx[None, :]
    out[mask] = sub[mask]
    return out


def cv_to_gl(c2w_cv: np.ndarray) -> np.ndarray:
    """OpenCV camera -> world (x right, y down, z ahead) -> OpenGL / ARKit camera -> world (x
    right, y up, -z ahead): the camera's y and z axes flipped. Its own inverse."""
    return np.asarray(c2w_cv, float) @ CV_TO_GL4


def backproject(depth: np.ndarray, K: np.ndarray, c2w_cv: np.ndarray, mask: np.ndarray,
                stride: int = 1) -> np.ndarray:
    """World points of the `mask`ed pixels of a z-depth map (OpenCV camera)."""
    sub = np.zeros_like(mask)
    sub[::stride, ::stride] = True
    vv, uu = np.nonzero(mask & sub)
    z = depth[vv, uu].astype(np.float64)
    cam = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], 1)
    with np.errstate(all="ignore"):     # Accelerate's matmul flags spurious FP errors on macOS
        return cam @ c2w_cv[:3, :3].T + c2w_cv[:3, 3]


def confidence_levels(conf: np.ndarray, valid: np.ndarray, *, low: float = CONF_LOW,
                      mid: float = CONF_MID) -> tuple[np.ndarray, dict]:
    """The model's confidence (any scale) -> ARKit's 0 / 1 / 2 by its quantiles over every valid
    pixel of the whole video: one bad frame gets more zeros, not the same share. A quantile that
    falls on the lowest value (many pixels tied at the confidence floor) is taken strictly, so
    the tied pixels stay below it instead of all becoming level 2."""
    values = conf[valid].astype(np.float64)
    if values.size == 0:
        return np.zeros(conf.shape, np.uint8), {"low": None, "mid": None}
    q_low, q_mid = np.quantile(values, [low, mid])
    floor, top = float(values.min()), float(values.max())

    def above(q):
        # all equal: no order to use, every valid pixel is level 2
        return conf > q if floor < top and q <= floor else conf >= q

    out = np.where(above(q_mid), 2, np.where(above(q_low), 1, 0)).astype(np.uint8)
    out[~valid] = 0
    return out, {"low": float(q_low), "mid": float(q_mid)}


@dataclass
class Geometry:
    """The model's reply, unpacked."""

    c2w_cv: np.ndarray          # (n, 4, 4) model world
    K_model: np.ndarray         # (n, 3, 3)
    depth: np.ndarray           # (n, h, w) float32 metres, 0 = masked
    conf: np.ndarray            # (n, h, w) float32
    prep: list[dict]
    meta: dict = field(default_factory=dict)


def unpack(result: dict) -> Geometry:
    with np.load(io.BytesIO(result["arrays"])) as z:
        arrays = {k: z[k] for k in z.files}
    meta = {k: v for k, v in result.items() if k != "arrays"}
    n = len(arrays["c2w"])
    # conf_log (log(conf - 1)) keeps the order that float16 `conf` loses near its floor of 1;
    # either works here: only its quantiles are used.
    conf = arrays["conf_log"] if "conf_log" in arrays else arrays["conf"]
    if not (len(arrays["K"]) == len(arrays["depth"]) == len(conf) == n
            == len(result["prep"])):
        raise ValueError("model reply: per-frame arrays of different lengths")
    return Geometry(arrays["c2w"].astype(np.float64), arrays["K"].astype(np.float64),
                    arrays["depth"].astype(np.float32), conf.astype(np.float32),
                    list(result["prep"]), meta)


def model_cloud(geo: Geometry, levels: np.ndarray, *, stride: int = CLOUD_STRIDE,
                max_depth: float = CLOUD_MAX_DEPTH) -> np.ndarray:
    parts = []
    for i in range(len(geo.depth)):
        d = geo.depth[i]
        ok = (levels[i] >= 2) & (d > 0.05) & (d < max_depth) & np.isfinite(d)
        parts.append(backproject(d, geo.K_model[i], geo.c2w_cv[i], ok, stride))
    pts = np.concatenate(parts) if parts else np.zeros((0, 3))
    return stray.voxel_downsample(pts, 0.02).astype(np.float64)


# --------------------------------------------------------------------------------------------
# The target scan
# --------------------------------------------------------------------------------------------

def find_room_usdz(scan_dir: Path) -> Path | None:
    return next((scan_dir / n for n in USDZ_CANDIDATES if (scan_dir / n).is_file()), None)


def scan_cloud(scan_dir: Path, voxel: float = stray.PCD_VOXEL) -> np.ndarray:
    """A scan's surface in its ARKit world: pointcloud.pcd and its frames' confident depth."""
    from litereality_agent.models.splat import rgbd

    scan_dir = Path(scan_dir)
    parts = []
    pcd = scan_dir / "pointcloud.pcd"
    if pcd.is_file():
        parts.append(stray.read_pcd(pcd))
    to_arkit = np.linalg.inv(rgbd.ARKIT_TO_Z_UP)
    for f in rgbd.load_frames(scan_dir):
        frame = stray.StrayFrame(f.index, f"{f.index:05d}", 0.0, to_arkit @ f.c2w, f.K, f.depth,
                                 f.conf)
        parts.append(stray.frame_points(frame, f.size))
    if not parts:
        raise FileNotFoundError(f"{scan_dir}: no pointcloud.pcd and no frames with depth")
    return stray.voxel_downsample(np.concatenate(parts).astype(np.float64), voxel)


# --------------------------------------------------------------------------------------------
# Calling the model
# --------------------------------------------------------------------------------------------

def jpeg_bytes(rgb: np.ndarray, quality: int = UPLOAD_QUALITY) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _resize(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if (rgb.shape[1], rgb.shape[0]) == tuple(size):
        return rgb
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS))


def modal_function(*, environment_name: str = "main", profile: str | None = None,
                   credentials: tuple[str, str] | None = None):
    from litereality_agent.runtimes.modal import ModalClient

    return ModalClient(APP_NAME, FUNCTION_NAME, environment_name=environment_name,
                       profile=profile, credentials=credentials).function


def _cache_key(info: VideoInfo, indices, options: dict, upload: tuple[int, int]) -> str:
    st = info.path.stat()
    doc = json.dumps({"video": str(info.path.resolve()), "bytes": st.st_size,
                      "mtime": int(st.st_mtime), "frames": [int(i) for i in indices],
                      "options": options, "upload": list(upload)}, sort_keys=True)
    return hashlib.sha256(doc.encode()).hexdigest()[:16]


def _read_cache(out: Path, key: str) -> dict | None:
    meta, arrays = out / RAW_DIR / "meta.json", out / RAW_DIR / "arrays.npz"
    try:
        doc = json.loads(meta.read_text(encoding="utf-8"))
        if doc.get("key") != key:
            return None
        return {**doc["result"], "arrays": arrays.read_bytes()}
    except (OSError, ValueError, KeyError):
        return None


def _write_cache(out: Path, key: str, result: dict) -> None:
    d = out / RAW_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "arrays.npz").write_bytes(result["arrays"])
    stray.write_json(d / "meta.json", {"key": key, "result": {
        k: v for k, v in result.items() if k != "arrays"}})


# --------------------------------------------------------------------------------------------
# The import
# --------------------------------------------------------------------------------------------

def _clear(out_dir: Path) -> None:
    for pattern in SCAN_FILES:
        for p in out_dir.glob(pattern):
            p.unlink()
    for name in ("pointcloud.pcd", "scene_alignment.json", "scan_label.json", REPORT,
                 "room.usdz"):
        (out_dir / name).unlink(missing_ok=True)
    shutil.rmtree(out_dir / "roomplan", ignore_errors=True)
    shutil.rmtree(out_dir / "depth_dense", ignore_errors=True)


def _bounds(points: np.ndarray) -> dict:
    if len(points) == 0:
        return {}
    lo, hi = points.min(0), points.max(0)
    return {"max": [round(float(v), 5) for v in hi], "min": [round(float(v), 5) for v in lo],
            "size": [round(float(v), 5) for v in hi - lo]}


def _save_jpeg(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image

    tmp = path.with_suffix(".tmp.jpg")
    Image.fromarray(rgb).save(tmp, quality=JPEG_QUALITY)
    tmp.replace(path)


def _save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image

    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(arr).save(tmp)
    tmp.replace(path)


def frame_json(index: int, c2w: np.ndarray, K: np.ndarray, time_s: float, speed: float,
               turn: float, source: dict) -> dict:
    doc = stray.frame_json(index, c2w, K, time_s, speed, turn, source)
    doc["video"] = doc.pop("stray")
    return doc


def check_intrinsics(geo: Geometry, K_photo: np.ndarray, photo_dir: Path, warnings: list[str],
                     *, self_calibrate: bool = True, log=print) -> dict:
    """The focal length the scan will carry vs what the model's views and the photos say: the
    calibration pass's per-view fx spread and, with `self_calibrate`, the photos' own focal
    (`selfcal.estimate_focal` on the staged frame_XXXXX.jpg). Appends to `warnings`."""
    fx = float(np.median(K_photo[:, 0, 0]))
    doc: dict = {"fx_photo": round(fx, 2)}
    calib = next((p for p in geo.meta.get("passes", []) if p.get("kind") == "calibration"), None)
    if calib and calib.get("fx_relative_spread") is not None:
        spread = float(calib["fx_relative_spread"])
        doc["model_fx_relative_spread"] = spread
        if spread > FX_SPREAD_WARN:
            warnings.append(f"the model's views disagree on the focal length by {spread:.0%} "
                            f"(std / median over {calib.get('views')} views): its K "
                            f"(fx {fx:.0f} px) is uncertain")
    if self_calibrate:
        from litereality_agent.models.capture import selfcal

        try:
            import cv2  # noqa: F401
        except ImportError:
            doc["self_calibration"] = {"focal": None, "note": "opencv not installed"}
            return doc

        class _Photos:
            def __getitem__(self, i):
                from PIL import Image

                with Image.open(photo_dir / f"frame_{i:05d}.jpg") as im:
                    return np.asarray(im.convert("RGB"))

        pairs = selfcal.pairs_by_rotation(geo.c2w_cv)
        sc = selfcal.estimate_focal(_Photos(), pairs, reference_focal=fx, log=log)
        doc["self_calibration"] = {k: v for k, v in sc.items() if k != "curve"}
        if sc.get("focal"):
            ratio = sc["focal"] / fx
            doc["self_calibration"]["ratio_to_model"] = round(ratio, 4)
            if abs(ratio - 1.0) > SELFCAL_WARN:
                warnings.append(
                    f"self-calibration from the photos puts the focal at {sc['focal']:.0f} px, "
                    f"the model's K says {fx:.0f} px ({ratio - 1:+.0%}, {sc['pairs']} frame "
                    "pairs): with a focal that far off the cameras sit nearer or farther than "
                    "they were, which depth and alignment checks cannot see; expect blur in a "
                    "photoreal room trained on it")
    return doc


def check_frames(geo: Geometry, levels: np.ndarray, transform: np.ndarray, poses: np.ndarray,
                 target_cloud: np.ndarray, *, stride: int = 2) -> list[dict]:
    """Every frame's confident model depth, in the target's world, registered onto the target's
    surface (`video_eval.frame_check`)."""
    from scipy.spatial import cKDTree

    from litereality_agent.models.capture import video_eval
    from litereality_agent.models.capture.align import estimate_normals

    tgt = np.asarray(target_cloud, np.float64)
    tree = cKDTree(tgt)
    with np.errstate(all="ignore"):
        normals = estimate_normals(tgt, tree)
    out = []
    for j in range(len(geo.depth)):
        d = geo.depth[j]
        ok = (levels[j] >= 2) & (d > 0.05) & (d < CLOUD_MAX_DEPTH) & np.isfinite(d)
        with np.errstate(all="ignore"):
            pts = sim3.apply(transform, backproject(d, geo.K_model[j], geo.c2w_cv[j], ok, stride))
        pts = stray.voxel_downsample(pts, 0.02).astype(np.float64)
        out.append(video_eval.frame_check(pts, poses[j], tgt, tree, normals))
    return out


def is_bad(check: dict) -> bool:
    """A frame that registered onto the target only after a large correction. One that did not
    register at all is not dropped: the target may just not cover what it sees."""
    return check["status"] != "failed" and (check.get("position_m", 0.0) > DROP_POSITION
                                            or check.get("rotation_deg", 0.0) > DROP_DEG)


def summarise_checks(checks: list[dict | None], dropped: list[int]) -> dict:
    rows = [c for c in checks if c is not None]
    measured = [c for c in rows if "position_m" in c]
    large = [c for c in rows if is_bad(c)]
    return {"frames": len(rows), "ok": sum(c["status"] == "ok" for c in rows),
            "degenerate": sum(c["status"] == "degenerate" for c in rows),
            "failed": sum(c["status"] == "failed" for c in rows), "large": len(large),
            "bad": sum(is_bad(c) for c in rows), "dropped": len(dropped),
            "dropped_picks": dropped,
            "position_median_m": round(float(np.median([c["position_m"] for c in measured])), 4)
            if measured else None,
            "limits": {"position_m": DROP_POSITION, "rotation_deg": DROP_DEG},
            "note": "registration of each frame's depth onto the --align-to surface: catches "
                    "frames whose geometry is off, not a wrong focal length"}


def _round_check(c: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}


def import_video(video: Path | str, out_dir: Path | str, *, align_to: Path | str | None = None,
                 frames: int = TARGET_FRAMES, function=None, options: dict | None = None,
                 max_residual: float = 0.05, min_inliers: float = 0.4,
                 accept_ambiguous: bool = False, overwrite: bool = False,
                 scales=None, upload_long: int = UPLOAD_LONG,
                 photo_long: int = PHOTO_LONG, keep_bad_frames: bool = False,
                 self_calibrate: bool = True, log: Callable[[str], None] = print) -> dict:
    """Import a phone video into `out_dir` (see the module docstring); returns the report (also
    written to `REPORT`). `function` is the deployed `reconstruct` (anything with ``.remote``;
    looked up on Modal when None). Raises ValueError before writing the scan when the alignment
    is not trustworthy (residual above `max_residual` m, fewer than `min_inliers` of the video's
    points on the target's surface, or another heading nearly as good unless
    `accept_ambiguous`). Frames failing the per-frame check against the target are dropped
    unless `keep_bad_frames`."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    out = Path(out_dir).expanduser()
    target = Path(align_to).expanduser() if align_to else None
    warnings: list[str] = []
    if out.exists() and any(out.glob("frame_*.json")) and not overwrite:
        raise ValueError(f"{out} already holds a scan (frame_*.json); pick another --out or "
                         "pass --overwrite")
    if target is not None:
        if not target.is_dir():
            raise FileNotFoundError(f"--align-to {target}: no such scan folder")
        if target.resolve() == out.resolve():
            raise ValueError("--out must not be the scan it is aligned to")

    info = probe(video)
    log(f"video {info.path.name}: {info.coded[0]}x{info.coded[1]} {info.codec}, rotation "
        f"{info.rotation}, shown {info.size[0]}x{info.size[1]}, {info.fps:.2f} fps"
        + (f", HDR ({info.color_transfer}) read as SDR" if info.hdr else ""))
    t = time.perf_counter()
    vscan = scan_video(info, progress=lambda n: log(f"  scored {n} frames ..."))
    timings["score_s"] = time.perf_counter() - t
    sel = select_frames(vscan.times, vscan.scores, vscan.thumbs, target=frames)
    if len(sel.indices) < 3:
        raise ValueError(f"{info.path}: only {len(sel.indices)} usable frames")
    s = sel.stats["sharpness"]
    log(f"  {len(vscan.scores)} frames scored in {timings['score_s']:.1f} s; kept "
        f"{len(sel.indices)} (target {frames}; {sel.stats['duplicates_dropped']} near-"
        f"duplicates, {sel.stats['blurry_dropped']} blurry); sharpness median "
        f"{s['kept_median']} kept vs {s['all_median']} all")

    photo_size = scaled_size(info.size, photo_long)
    upload_size = scaled_size(info.size, upload_long)
    out.mkdir(parents=True, exist_ok=True)
    stage = out / ".video_import_tmp"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    t = time.perf_counter()
    uploads: list[bytes] = []
    for j, (k, rgb) in enumerate(decode_frames(info, sel.indices, photo_size)):
        _save_jpeg(rgb, stage / f"frame_{j:05d}.jpg")
        uploads.append(jpeg_bytes(_resize(rgb, upload_size)))
    timings["decode_s"] = time.perf_counter() - t
    log(f"  decoded {len(uploads)} frames at {photo_size[0]}x{photo_size[1]} in "
        f"{timings['decode_s']:.1f} s ({sum(map(len, uploads)) / 1e6:.1f} MB to upload at "
        f"{upload_size[0]}x{upload_size[1]})")

    opts = dict(options or {})
    key = _cache_key(info, sel.indices, opts, upload_size)
    result = _read_cache(out, key)
    t = time.perf_counter()
    if result is not None:
        log(f"  model reply reused from {out / RAW_DIR} (same frames and options)")
        cached = True
    else:
        if function is None:
            function = modal_function()
        log(f"  {APP_NAME}.{FUNCTION_NAME} on {len(uploads)} frames ...")
        result = function.remote(uploads, opts)
        _write_cache(out, key, result)
        cached = False
    timings["model_s"] = time.perf_counter() - t
    geo = unpack(result)
    if len(geo.c2w_cv) != len(uploads):
        raise ValueError(f"model returned {len(geo.c2w_cv)} frames for {len(uploads)}")
    log(f"  model: {geo.meta.get('gpu')}, {geo.meta.get('seconds')} s, passes "
        f"{[p.get('views') for p in geo.meta.get('passes', [])]}"
        + (" (cached)" if cached else f", {timings['model_s']:.1f} s round trip"))

    valid = (geo.depth > 0) & np.isfinite(geo.depth)
    levels, conf_q = confidence_levels(geo.conf, valid)
    cloud_model = model_cloud(geo, levels)
    ups = -geo.c2w_cv[:, :3, 1]                      # an OpenCV camera's up is -y
    guess = ups.mean(0)
    up, up_stats = sim3.estimate_up(cloud_model, guess)
    level = sim3.level_transform(up, origin=geo.c2w_cv[0, :3, 3], forward=geo.c2w_cv[0, :3, 2])
    log(f"  gravity: {up_stats.get('from_guess_deg')} deg from the cameras' mean up, "
        f"{len(cloud_model)} cloud points")

    transform = level
    alignment = None
    if target is not None:
        t = time.perf_counter()
        log(f"aligning to {target} (yaw + translation + scale) ...")
        src = sim3.apply(level, cloud_model)
        target_cloud = scan_cloud(target)
        alignment = sim3.align_sim3(src, target_cloud, scales=scales, log=log)
        timings["align_s"] = time.perf_counter() - t
        log(f"  scale {alignment.scale:.4f}, yaw {alignment.yaw_deg:.2f} deg, translation "
            f"{np.round(alignment.translation, 3).tolist()} m, residual "
            f"{alignment.residual * 100:.1f} cm, {alignment.inliers:.0%} of the video on the "
            f"scan's surface")
        if not (alignment.residual <= max_residual and alignment.inliers >= min_inliers):
            shutil.rmtree(stage, ignore_errors=True)
            raise ValueError(
                f"alignment to {target} is not trustworthy: residual "
                f"{alignment.residual * 100:.1f} cm (limit {max_residual * 100:.1f}), "
                f"{alignment.inliers:.0%} on its surface (need {min_inliers:.0%}). The model's "
                f"reply is kept in {out / RAW_DIR}; --max-residual / --min-inliers relax this")
        if alignment.ambiguous:
            msg = (f"alignment to {target} is ambiguous (heading margin {alignment.margin:.2f}, "
                   f"runner-up {alignment.runner_up})")
            if not accept_ambiguous:
                shutil.rmtree(stage, ignore_errors=True)
                raise ValueError(msg + "; nothing was written (--accept-ambiguous takes it)")
            warnings.append(msg + ": taken anyway (--accept-ambiguous)")
        if alignment.scale_spread > sim3.SCALE_SPREAD_WARN:
            warnings.append(f"the alignment pins the scale down only to about "
                            f"{alignment.scale_spread:.1%} (near-equal fits at scales "
                            f"{sorted({t['scale'] for t in alignment.tries if 'scale' in t})}): "
                            "every depth and camera position carries that uncertainty")
        if alignment.inliers < LOW_INLIERS:
            warnings.append(f"only {alignment.inliers:.0%} of the video lies on {target.name}'s "
                            "surface: check the room in the viewer")
        transform = alignment.transform @ level
    scale = sim3.sim3_scale(transform)
    poses_all = np.stack([sim3.apply_sim3_to_pose(transform, cv_to_gl(c)) for c in geo.c2w_cv])
    K_all = np.stack([model_K_to_photo(geo.K_model[j], geo.prep[j], photo_size)
                      for j in range(len(geo.K_model))])

    # ---- checks ----
    t = time.perf_counter()
    intrinsics_check = check_intrinsics(geo, K_all, stage, warnings,
                                        self_calibrate=self_calibrate, log=log)
    checks: list[dict | None] = [None] * len(sel.indices)
    if target is not None:
        checks = check_frames(geo, levels, transform, poses_all, target_cloud)
    keep = [j for j, c in enumerate(checks)
            if keep_bad_frames or c is None or not is_bad(c)]
    dropped = sorted(set(range(len(checks))) - set(keep))
    timings["check_s"] = time.perf_counter() - t
    frame_check_doc = summarise_checks(checks, dropped) if target is not None else None
    if frame_check_doc:
        log(f"  frame check vs {target.name}: {frame_check_doc['ok']} ok, "
            f"{frame_check_doc['degenerate']} degenerate, {frame_check_doc['failed']} not "
            f"registered, {frame_check_doc['large']} with a correction over "
            f"{DROP_POSITION * 100:.0f} cm / {DROP_DEG:.0f} deg; "
            + (f"{len(dropped)} dropped" if dropped else "none dropped"))
        if keep_bad_frames and frame_check_doc["bad"]:
            warnings.append(f"{frame_check_doc['bad']} frames failed the check against "
                            f"{target.name} and were kept (--keep-bad-frames)")
    if len(keep) < 3:
        shutil.rmtree(stage, ignore_errors=True)
        raise ValueError(f"only {len(keep)} frames passed the check against {target}")

    # ---- write the scan ----
    if overwrite:
        _clear(out)
    t = time.perf_counter()
    times = vscan.times[sel.indices][keep]
    poses = poses_all[keep]
    speed, turn = stray.velocities(times, poses)
    dsize = depth_grid_size(photo_size)
    written: list[stray.StrayFrame] = []
    for n, j in enumerate(keep):
        k = sel.indices[j]
        stem = f"{n:05d}"
        (stage / f"frame_{j:05d}.jpg").replace(out / f"frame_{stem}.jpg")
        K_photo = K_all[j]
        depth = model_map_to_photo_grid(geo.depth[j] * scale, geo.prep[j], dsize)
        mm = np.clip(np.round(np.nan_to_num(depth) * 1000.0), 0, 65535).astype(np.uint16)
        conf = model_map_to_photo_grid(levels[j], geo.prep[j], dsize)
        conf[mm == 0] = 0
        _save_png(mm, out / f"depth_{stem}.png")
        _save_png(conf, out / f"conf_{stem}.png")
        doc = frame_json(n, poses[n], K_photo, float(times[n]), speed[n], turn[n],
                         {"video": info.path.name, "video_frame": int(k),
                          "sharpness": round(float(vscan.scores[k]), 3),
                          "model": "map-anything-apache", "pick": int(j)})
        if checks[j] is not None:
            doc["videoCheck"] = _round_check(checks[j])
        stray.write_json(out / f"frame_{stem}.json", doc)
        written.append(stray.StrayFrame(n, stem, float(times[n]), poses[n], K_photo,
                                        out / f"depth_{stem}.png", out / f"conf_{stem}.png"))
    shutil.rmtree(stage, ignore_errors=True)
    timings["write_s"] = time.perf_counter() - t
    cloud = stray.fuse_points(written, photo_size)
    stray.write_pcd(out / "pointcloud.pcd", cloud, generator="litereality import-video")

    name = info.path.stem
    label = {"createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "displayName": f"Phone video - {name}", "scannerName": "Phone video (MapAnything)",
             "sceneName": name, "uploadFolderName": name}
    alignment_doc: dict = {"coordinateSystem": "ARKit world coordinates shared by camera poses "
                                               "and pointcloud.pcd", "unit": "meters"}
    if target is not None:
        if find_room_usdz(target) is None:
            warnings.append(f"{target} has no RoomPlan room.usdz: quickroom / splatroom need one")
        if (target / "room.usdz").is_file():
            shutil.copyfile(target / "room.usdz", out / "room.usdz")
        if (target / "roomplan").is_dir():
            shutil.copytree(target / "roomplan", out / "roomplan", dirs_exist_ok=True)
        for fname, merge in (("scan_label.json", "label"), ("scene_alignment.json", "align")):
            try:
                theirs = json.loads((target / fname).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(theirs, dict):
                continue
            if merge == "label" and theirs.get("roomType"):
                label["roomType"] = theirs["roomType"]
            elif merge == "align":
                alignment_doc = theirs
        alignment_doc["note"] = (
            f"Imported from the phone video {info.path.name}: poses and depth by MapAnything, "
            f"levelled by its horizontal / vertical surfaces, then moved into {target.name}'s "
            "ARKit world by a similarity (yaw about Y, translation, scale; "
            "`videoImport.transform`), so the copied RoomPlan room.usdz, the poses and "
            "pointcloud.pcd share one frame.")
    else:
        alignment_doc["note"] = (
            f"The phone video {info.path.name}'s own world: MapAnything's metric scale, levelled "
            "(+Y up), the first camera at the origin looking down -Z. No RoomPlan scan to line up "
            "with: --align-to adds one.")
    alignment_doc["pointCloudBounds"] = _bounds(cloud)
    alignment_doc["videoImport"] = {
        "video": str(info.path), "alignedTo": str(target) if target else None,
        "transform": [[round(float(v), 6) for v in r] for r in transform],
        "scale": round(scale, 6),
        "residual_m": round(alignment.residual, 4) if alignment else None}
    stray.write_json(out / "scan_label.json", label)
    stray.write_json(out / "scene_alignment.json", alignment_doc)

    timings["total_s"] = time.perf_counter() - t0
    report = {
        "video": str(info.path), "out": str(out), "info": info.summary(),
        "selection": sel.stats, "frames": int(len(keep)), "picked": int(len(sel.indices)),
        "photo_size": list(photo_size), "depth_size": list(dsize),
        "upload": {"size": list(upload_size), "megabytes": round(sum(map(len, uploads)) / 1e6, 2)},
        "model": {k: geo.meta.get(k) for k in ("model", "code", "gpu", "model_size", "passes",
                                               "intrinsics_shared", "metric_scaling",
                                               "peak_memory_gb", "seconds", "options")},
        "model_cached": cached,
        "confidence_quantiles": conf_q,
        "gravity": {"up_model": [round(float(v), 5) for v in up], **up_stats},
        "alignment": alignment.summary() if alignment else None,
        "intrinsics_check": intrinsics_check,
        "frame_check": frame_check_doc,
        "scale": round(scale, 6),
        "pointcloud_points": int(len(cloud)), "warnings": warnings,
        "seconds": {k: round(v, 2) for k, v in timings.items()},
    }
    stray.write_json(out / REPORT, report)
    for w in warnings:
        log(f"  warning: {w}")
    return report

