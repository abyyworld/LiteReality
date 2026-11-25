"""DROID-SLAM on a folder of frames, inside the Modal container (copied into /opt/DROID-SLAM).

Same as DROID-SLAM's demo.py (sync mode, one GPU), with stride 1 so every frame is fed, no
viewer, and the result saved: the final trajectory (all frames, filled by DROID's own
PoseTrajectoryFiller after the global bundle adjustment in Droid.terminate) plus the keyframe
count and keyframe timestamps.

    python droid_driver.py IMAGEDIR FX FY CX CY OUT.npz [--buffer 1024]
"""

import argparse
import os
import sys
import time

sys.path.append("/opt/DROID-SLAM/droid_slam")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from droid import Droid  # noqa: E402


def image_stream(imagedir, fx, fy, cx, cy):
    """demo.py's image_stream (no distortion, stride 1)."""
    for t, imfile in enumerate(sorted(os.listdir(imagedir))):
        image = cv2.imread(os.path.join(imagedir, imfile))
        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0
        yield t, image[None], intrinsics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imagedir")
    ap.add_argument("fx", type=float)
    ap.add_argument("fy", type=float)
    ap.add_argument("cx", type=float)
    ap.add_argument("cy", type=float)
    ap.add_argument("out")
    ap.add_argument("--buffer", type=int, default=1024)
    a = ap.parse_args()

    # demo.py defaults, plus stride 1, a larger buffer and no viewer
    args = argparse.Namespace(
        weights="/opt/DROID-SLAM/droid.pth", buffer=a.buffer, image_size=None, disable_vis=True,
        beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0, frontend_thresh=16.0,
        frontend_window=25, frontend_radius=2, frontend_nms=1, backend_thresh=22.0,
        backend_radius=2, backend_nms=3, upsample=False, asynchronous=False,
        frontend_device="cuda", backend_device="cuda", stereo=False,
    )
    names = sorted(os.listdir(a.imagedir))
    t0 = time.time()
    droid = None
    for t, image, intrinsics in image_stream(a.imagedir, a.fx, a.fy, a.cx, a.cy):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            print("network input size", args.image_size, "intrinsics", intrinsics.tolist(), flush=True)
            droid = Droid(args)
        droid.track(t, image, intrinsics=intrinsics)
        if t % 100 == 0:
            print(f"tracked {t}/{len(names)} keyframes {droid.video.counter.value}", flush=True)
    t_track = time.time() - t0
    n_kf = int(droid.video.counter.value)
    kf_tstamps = droid.video.tstamp[:n_kf].cpu().numpy()
    print(f"tracking done in {t_track:.1f}s, keyframes {n_kf}", flush=True)
    # global BA (backend 7 + 12 iterations over proximity edges) and the non-keyframe filler
    traj = droid.terminate(image_stream(a.imagedir, a.fx, a.fy, a.cx, a.cy))
    secs = time.time() - t0
    print(f"done in {secs:.1f}s, trajectory {traj.shape}", flush=True)
    np.savez(a.out, traj=traj, names=np.array(names), keyframes=n_kf, kf_tstamps=kf_tstamps,
             seconds=secs, track_seconds=t_track, image_size=np.array(args.image_size))


if __name__ == "__main__":
    main()
