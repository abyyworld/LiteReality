"""(6) ARKit pose self-consistency, in pixels. For pairs of TRAINING frames (q600) that look at the
same surfaces: SIFT matches (ratio 0.75) between the photos; each match in frame A is lifted to 3D
with A's own LiDAR depth (confidence 2, smooth depth only) and A's ARKit pose, projected into B with
B's ARKit pose, and compared with the matched keypoint in B. Median residual over matches, per pair.
  short: 0.2-1.0 s apart (local tracking + depth + matching noise floor)
  revisit: >= 8 s apart, cameras within 40 cm and 20 deg (loop: drift shows here)
Residuals reported at 1920 px and scaled to the 1440 px training width.
Writes pose_consistency.json, pose_consistency.png."""
import json, random
from pathlib import Path
import cv2, numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from litereality_agent.pipeline.evaluate import fidelity as fd
HERE = Path(__file__).resolve().parent; Q = Path.home() / "LiteReality-scans/stray_atrium_q600"
fr = []
for j in sorted(Q.glob("frame_*.json")):
    m = json.loads(j.read_text())
    fr.append({"stem": j.stem, "n": j.stem[6:], "t": m["time"], "pose": fd.camera_pose(m["cameraPoseARFrame"]), "K": np.array(m["intrinsics"]).reshape(3, 3), "vf": m["stray"]["video_frame"]})
blur = np.load(HERE / "blur_arrays.npz")["Lest"]; W = np.load(HERE / "blur_arrays.npz")["wperp"]
T = np.array([f["t"] for f in fr]); C = np.array([f["pose"][:3, 3] for f in fr]); Fw = np.array([-f["pose"][:3, 2] for f in fr])
sift = cv2.SIFT_create(6000, contrastThreshold=0.01)
cache = {}
def feats(i):
    if i not in cache:
        g = cv2.cvtColor(cv2.imread(str(Q / (fr[i]["stem"] + ".jpg"))), cv2.COLOR_BGR2GRAY)
        cache[i] = sift.detectAndCompute(g, None)
        if len(cache) > 40: cache.pop(next(iter(cache)))
    return cache[i]
def lift(i, uv):
    f = fr[i]; D = np.asarray(Image.open(Q / f"depth_{f['n']}.png"), np.float32) / 1000
    Cf = np.asarray(Image.open(Q / f"conf_{f['n']}.png")); h, w = D.shape
    iu = np.clip((uv[:, 0] * w / 1920).astype(int), 1, w - 2); iv = np.clip((uv[:, 1] * h / 1440).astype(int), 1, h - 2)
    d = D[iv, iu]; nb = np.stack([D[iv + a, iu + b] for a in (-1, 0, 1) for b in (-1, 0, 1)], 1)
    ok = (Cf[iv, iu] >= 2) & (d > 0.2) & (np.ptp(nb, 1) < 0.05)
    K = f["K"]; x = (uv[:, 0] - K[0, 2]) / K[0, 0] * d; y = (uv[:, 1] - K[1, 2]) / K[1, 1] * d
    pc = np.c_[x, -y, -d]  # OpenGL camera
    return pc @ f["pose"][:3, :3].T + f["pose"][:3, 3], ok
def project(i, P):
    f = fr[i]; pc = (P - f["pose"][:3, 3]) @ f["pose"][:3, :3]; z = -pc[:, 2]
    K = f["K"]; return np.c_[K[0, 0] * pc[:, 0] / z + K[0, 2], K[1, 1] * (-pc[:, 1]) / z + K[1, 2]], z
bf = cv2.BFMatcher()
def pair(i, j):
    ka, da = feats(i); kb, db = feats(j)
    if da is None or db is None or len(ka) < 50 or len(kb) < 50: return None
    m = [a for a, b in bf.knnMatch(da, db, k=2) if a.distance < 0.75 * b.distance]
    if len(m) < 40: return None
    ua = np.array([ka[x.queryIdx].pt for x in m]); ub = np.array([kb[x.trainIdx].pt for x in m])
    # geometric outlier rejection independent of ARKit: fundamental matrix RANSAC
    Fm, inl = cv2.findFundamentalMat(ua, ub, cv2.FM_RANSAC, 2.0, 0.999)
    if inl is None: return None
    inl = inl.ravel().astype(bool); ua, ub = ua[inl], ub[inl]
    P, ok = lift(i, ua)
    if ok.sum() < 15: return None
    pb, z = project(j, P[ok]); r = np.linalg.norm(pb - ub[ok], axis=1); r = r[z > 0]
    # depth-free check: distance of B's keypoint from the epipolar line ARKit's relative pose gives
    Ta, Tb = fr[i]["pose"], fr[j]["pose"]; G = np.diag([1, -1, -1])
    Ra, Rb = Ta[:3, :3] @ G, Tb[:3, :3] @ G  # OpenCV camera -> world
    Rab = Rb.T @ Ra; tab = Rb.T @ (Ta[:3, 3] - Tb[:3, 3])
    tx = np.array([[0, -tab[2], tab[1]], [tab[2], 0, -tab[0]], [-tab[1], tab[0], 0]])
    Fk = np.linalg.inv(fr[j]["K"]).T @ tx @ Rab @ np.linalg.inv(fr[i]["K"])
    ha = np.c_[ua, np.ones(len(ua))]; hb = np.c_[ub, np.ones(len(ub))]
    l = ha @ Fk.T; epi = np.abs(np.sum(l * hb, 1)) / np.hypot(l[:, 0], l[:, 1])
    lr = ha @ Fm.T; epi_ransac = np.abs(np.sum(lr * hb, 1)) / np.hypot(lr[:, 0], lr[:, 1])
    return {"a": fr[i]["stem"], "b": fr[j]["stem"], "dt": round(abs(fr[j]["t"] - fr[i]["t"]), 2), "n": int(len(r)),
            "median_px1920": round(float(np.median(r)), 2), "p75_px1920": round(float(np.percentile(r, 75)), 2),
            "baseline_cm": round(float(np.linalg.norm(C[i] - C[j]) * 100), 1),
            "epipolar_median_px1920": round(float(np.median(epi)), 2), "epipolar_own_F_median_px1920": round(float(np.median(epi_ransac)), 2),
            "gyro_deg_s": [round(float(np.degrees(W[fr[i]["vf"]])), 1), round(float(np.degrees(W[fr[j]["vf"]])), 1)],
            "blur_a": round(float(blur[fr[i]["vf"]]), 1), "blur_b": round(float(blur[fr[j]["vf"]]), 1)}
random.seed(0)
short, revisit = [], []
idx = list(range(len(fr)))
for i in random.sample(idx, 120):
    js = [j for j in idx if 0.2 <= abs(T[j] - T[i]) <= 1.0]
    if js:
        r = pair(i, random.choice(js))
        if r: short.append(r)
    if len(short) >= 60: break
cand = [(i, j) for i in idx for j in idx if j > i and T[j] - T[i] >= 8 and np.linalg.norm(C[i] - C[j]) < 0.4 and np.degrees(np.arccos(np.clip(Fw[i] @ Fw[j], -1, 1))) < 20]
print("revisit candidates", len(cand))
for i, j in random.sample(cand, min(150, len(cand))):
    r = pair(i, j)
    if r: revisit.append(r)
    if len(revisit) >= 60: break
print("pairs", len(short), len(revisit))
def summ(rs):
    a = np.array([r["median_px1920"] for r in rs])
    return {"pairs": len(rs), "median_of_pair_medians_px1920": round(float(np.median(a)), 2), "p75_px1920": round(float(np.percentile(a, 75)), 2), "p90_px1920": round(float(np.percentile(a, 90)), 2),
            "median_px_at_1440": round(float(np.median(a) * 0.75), 2), "p90_px_at_1440": round(float(np.percentile(a, 90) * 0.75), 2),
            "median_dt_s": round(float(np.median([r["dt"] for r in rs])), 1),
            "epipolar_vs_ARKit_median_px1920": round(float(np.median([r["epipolar_median_px1920"] for r in rs])), 2),
            "epipolar_vs_ARKit_p90_px1920": round(float(np.percentile([r["epipolar_median_px1920"] for r in rs], 90)), 2),
            "epipolar_vs_own_F_median_px1920": round(float(np.median([r["epipolar_own_F_median_px1920"] for r in rs])), 2),
            "spearman_epipolar_vs_max_gyro": round(float(__import__("scipy.stats").stats.spearmanr([r["epipolar_median_px1920"] for r in rs], [max(r["gyro_deg_s"]) for r in rs]).correlation), 3)}
out = {"short_0.2-1s": summ(short), "revisit_ge_8s": summ(revisit), "short": short, "revisit": revisit}
(HERE / "pose_consistency.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("short", "revisit")}, indent=1))
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter([r["dt"] for r in short], [r["median_px1920"] * 0.75 for r in short], label="0.2-1 s apart", color="#1f6feb")
ax.scatter([r["dt"] for r in revisit], [r["median_px1920"] * 0.75 for r in revisit], label="revisits >= 8 s apart", color="#d1242f")
ax.set_xscale("log"); ax.set_xlabel("time between the two frames (s)"); ax.set_ylabel("median reprojection residual (px @1440)"); ax.legend()
ax.set_title("ARKit pose + LiDAR depth self-consistency (SIFT matches)")
fig.tight_layout(); fig.savefig(HERE / "pose_consistency.png", dpi=110)
