"""(1) motion blur of the Stray frames and of the TRAINING set, (4) exposure drift.

Inputs: frames_blur.csv (frames_blur.py), the recording's odometry.csv and imu.csv, the training
set's rows (stray_atrium_q600/frame_*.json "stray.video_frame"), the eval rows.
Outputs: blur_stats.json, plots blur_*.png.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REC = Path.home() / "LiteReality-scans/stray_atrium_raw/901a305fdf"
Q600 = Path.home() / "LiteReality-scans/stray_atrium_q600"
Q206 = Path.home() / "LiteReality-scans/stray_atrium_q206"
EVAL = HERE.parents[1] / "stray_quality/eval_frames.txt"
F = 1335.4  # fx at 1920 wide


def load_csv(p):
    with open(p) as f:
        r = list(csv.DictReader(f, skipinitialspace=True))
    return r


def main():
    fb = load_csv(HERE / "frames_blur.csv")
    lap = np.array([float(r["lap_full"]) for r in fb])
    lap9 = np.array([float(r["lap_960"]) for r in fb])
    lap4 = np.array([float(r["lap_480"]) for r in fb])
    hfr = np.array([float(r["hf_ratio"]) for r in fb])
    gx = np.array([float(r["gx"]) for r in fb]); gy = np.array([float(r["gy"]) for r in fb])
    grey = np.array([float(r["mean_grey"]) for r in fb])
    pk = np.array([float(r["pkt_size"] or 0) for r in fb])
    ptype = np.array([r["pict_type"] for r in fb])
    od = load_csv(REC / "odometry.csv")
    t = np.array([float(r["timestamp"]) for r in od]); t -= t[0]
    n = min(len(t), len(lap))
    t, lap, lap9, lap4, hfr, gx, gy, grey, pk, ptype = (a[:n] for a in (t, lap, lap9, lap4, hfr, gx, gy, grey, pk, ptype))
    xyz = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in od])[:n]
    q = np.array([[float(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in od])[:n]

    # gyro (rad/s, device frame) interpolated at frame times
    imu = np.loadtxt(REC / "imu.csv", delimiter=",", skiprows=1)
    t0 = float(od[0]["timestamp"])
    ti = imu[:, 0] - t0
    w = np.stack([np.interp(t, ti, imu[:, k]) for k in (4, 5, 6)], 1)
    wmag = np.linalg.norm(w, axis=1)
    # Image-plane angular rate: rotation about the optical axis blurs less at the centre; use the
    # two axes perpendicular to the optical axis. Device landscape: optical axis = device z.
    wperp = np.linalg.norm(w[:, :2], axis=1)
    speed = np.r_[0, np.linalg.norm(np.diff(xyz, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)]
    speed = np.convolve(speed, np.ones(5) / 5, mode="same")

    # Local relative sharpness: lap_full / max over the frames within +-0.25 s (same content).
    rel = np.zeros(n)
    for i in range(n):
        a, b = np.searchsorted(t, [t[i] - 0.25, t[i] + 0.25])
        rel[i] = lap[i] / lap[a:b + 1].max()
    rel480 = np.zeros(n)
    for i in range(n):
        a, b = np.searchsorted(t, [t[i] - 0.25, t[i] + 0.25])
        rel480[i] = lap4[i] / lap4[a:b + 1].max()

    # Calibrate "relative Laplacian" -> blur length: box-blur the 40 locally sharpest frames.
    sharp_idx = np.argsort(-(rel * (wperp < 0.15)))[:12]
    Ls = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24]
    curves = []
    cap = cv2.VideoCapture(str(REC / "rgb.mp4"))
    want = set(int(i) for i in sharp_idx)
    k = 0
    while want:
        ok, fr = cap.read()
        if not ok:
            break
        if k in want:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            base = cv2.Laplacian(g, cv2.CV_32F, ksize=1).var()
            c = []
            for L in Ls:
                if L == 0:
                    c.append(1.0); continue
                kx = np.ones((1, L), np.float32) / L
                gb = cv2.filter2D(g, -1, kx)
                c.append(float(cv2.Laplacian(gb, cv2.CV_32F, ksize=1).var() / base))
            curves.append(c)
            want.discard(k)
        k += 1
    cap.release()
    curve = np.median(np.array(curves), axis=0)

    def blur_len(r):
        # monotone decreasing curve: interpolate log
        return np.interp(-np.log(np.clip(r, 1e-4, 1)), -np.log(curve), Ls)

    Lest = blur_len(rel)
    # exposure fit: L = wperp * F * t_exp  (frames where Lest > 1 and motion > 0.2 rad/s)
    m = (wperp > 0.2) & (rel < 0.95)
    texp = float(np.median(Lest[m] / (wperp[m] * F)))
    Lpred = wperp * F * texp

    # training rows
    def rows_of(scan):
        out = []
        for j in sorted(scan.glob("frame_*.json")):
            mm = json.loads(j.read_text())
            out.append(int(mm["stray"]["video_frame"]))
        return np.array(out)
    tr = rows_of(Q600)
    tr206 = rows_of(Q206)
    ev = np.array([int(l.split()[0]) for l in EVAL.read_text().splitlines() if l and not l.startswith("#")])

    def summ(idx):
        return {"n": int(len(idx)),
                "lap_full_median": round(float(np.median(lap[idx])), 1),
                "rel_sharp_median": round(float(np.median(rel[idx])), 3),
                "frac_rel_below_0.5": round(float(np.mean(rel[idx] < 0.5)), 3),
                "frac_rel_below_0.7": round(float(np.mean(rel[idx] < 0.7)), 3),
                "blur_px1920_median": round(float(np.median(Lest[idx])), 2),
                "blur_px1920_p75": round(float(np.percentile(Lest[idx], 75)), 2),
                "blur_px1920_p90": round(float(np.percentile(Lest[idx], 90)), 2),
                "frac_blur_ge_2px1920": round(float(np.mean(Lest[idx] >= 2)), 3),
                "frac_blur_ge_4px1920": round(float(np.mean(Lest[idx] >= 4)), 3),
                "gyro_perp_median_deg_s": round(float(np.degrees(np.median(wperp[idx]))), 1),
                "gyro_perp_p90_deg_s": round(float(np.degrees(np.percentile(wperp[idx], 90))), 1),
                "pred_blur_px1920_median(gyro*f*texp)": round(float(np.median(Lpred[idx])), 2),
                "frac_I_frames": round(float(np.mean(ptype[idx] == "I")), 3)}

    allidx = np.arange(n)
    stats = {
        "frames": n, "duration_s": round(float(t[-1]), 2), "fps_mean": round(n / float(t[-1]), 1),
        "video_bitrate_bps": 9412839, "bits_per_pixel": round(9412839 / (1920 * 1440 * n / t[-1]), 4),
        "blur_calibration": {"box_len_px": Ls, "lap_ratio": [round(x, 4) for x in curve],
                             "frames_used": [int(i) for i in sharp_idx]},
        "exposure_estimate_s": round(texp, 4),
        "all": summ(allidx), "train_q600_637": summ(tr), "train_q206_188": summ(tr206), "eval_29": summ(ev),
        "corr_rel_vs_wperp_spearman": None,
        "packet_bytes": {"I_median": float(np.median(pk[ptype == "I"])) if (ptype == "I").any() else None,
                         "P_median": float(np.median(pk[ptype == "P"])) if (ptype == "P").any() else None,
                         "B_median": float(np.median(pk[ptype == "B"])) if (ptype == "B").any() else None,
                         "counts": {k: int((ptype == k).sum()) for k in ("I", "P", "B")}},
        "exposure_drift": {"mean_grey_p5_p95": [round(float(np.percentile(grey, 5)), 1), round(float(np.percentile(grey, 95)), 1)]},
        "eval_rows": {int(r): {"t": round(float(t[r]), 2), "rel_sharp": round(float(rel[r]), 3), "blur_px1920": round(float(Lest[r]), 2), "lap_full": round(float(lap[r]), 1)} for r in ev},
    }
    from scipy.stats import spearmanr
    stats["corr_rel_vs_wperp_spearman"] = round(float(spearmanr(rel, wperp).correlation), 3)
    stats["picker_480_vs_full"] = {
        "spearman_lap480_lapfull": round(float(spearmanr(lap4, lap).correlation), 3),
        "spearman_rel480_relfull": round(float(spearmanr(rel480, rel).correlation), 3)}
    np.savez(HERE / "blur_arrays.npz", t=t, lap=lap, rel=rel, Lest=Lest, wperp=wperp, speed=speed, grey=grey, tr=tr, ev=ev, tr206=tr206)
    (HERE / "blur_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "eval_rows"}, indent=1))

    # plots
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t, Lest, lw=0.5, color="#888", label="all frames")
    ax[0].scatter(t[tr], Lest[tr], s=6, color="#1f6feb", label="trained on (q600, 637)")
    ax[0].scatter(t[ev], Lest[ev], s=18, color="#d1242f", marker="x", label="eval (29)")
    ax[0].set_ylabel("est. motion blur (px @1920)"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 25)
    ax[1].plot(t, np.degrees(wperp), lw=0.6, color="#555"); ax[1].set_ylabel("gyro, perp. to optical axis (deg/s)")
    ax[2].plot(t, grey, lw=0.6, color="#555"); ax[2].set_ylabel("mean grey (0-255)"); ax[2].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(HERE / "blur_timeline.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.linspace(0, 20, 41)
    ax[0].hist(Lest, bins, alpha=.5, density=True, label="all 2402", color="#888")
    ax[0].hist(Lest[tr], bins, alpha=.6, density=True, label="trained 637", color="#1f6feb")
    ax[0].set_xlabel("est. blur length (px @1920)"); ax[0].legend()
    ax[1].scatter(np.degrees(wperp), Lest, s=3, alpha=.4, color="#555")
    xx = np.linspace(0, np.degrees(wperp).max(), 50)
    ax[1].plot(xx, np.radians(xx) * F * texp, color="#d1242f", label=f"gyro*f*t, t={texp*1000:.1f} ms")
    ax[1].set_xlabel("gyro perp (deg/s)"); ax[1].set_ylabel("est. blur (px @1920)"); ax[1].legend(); ax[1].set_ylim(0, 25)
    ax[2].plot(Ls, curve, "o-"); ax[2].set_xlabel("box blur length (px)"); ax[2].set_ylabel("Laplacian var ratio"); ax[2].set_yscale("log")
    ax[2].set_title("calibration on sharp still frames")
    fig.tight_layout(); fig.savefig(HERE / "blur_hist.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
