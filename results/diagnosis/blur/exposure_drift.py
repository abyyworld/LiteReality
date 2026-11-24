"""(4) Exposure / white-balance drift: the same surface points (SIFT matches, F-RANSAC) in pairs of
training frames (q600); per pair, the median log2 ratio of RGB at matched points (5x5 mean)
-> gain (stops) and R/G, B/G white-balance shift. Pairs: every frame vs one ~1 s later, and the
revisit pairs (>= 8 s apart, cameras within 40 cm / 20 deg). Writes exposure_drift.json/.png."""
import json, random
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from litereality_agent.pipeline.evaluate import fidelity as fd
HERE = Path(__file__).resolve().parent; Q = Path.home() / "LiteReality-scans/stray_atrium_q600"
fr = []
for j in sorted(Q.glob("frame_*.json")):
    m = json.loads(j.read_text()); p = fd.camera_pose(m["cameraPoseARFrame"])
    fr.append({"stem": j.stem, "t": m["time"], "c": p[:3, 3], "f": -p[:3, 2]})
T = np.array([f["t"] for f in fr]); T -= T[0]; C = np.array([f["c"] for f in fr]); Fw = np.array([f["f"] for f in fr])
sift = cv2.SIFT_create(3000, contrastThreshold=0.01); bf = cv2.BFMatcher()
def img(i): return cv2.imread(str(Q / (fr[i]["stem"] + ".jpg")))
def gain(i, j):
    a, b = img(i), img(j)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None); kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None: return None
    m = [x for x, y in bf.knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
    if len(m) < 30: return None
    ua = np.array([ka[x.queryIdx].pt for x in m]); ub = np.array([kb[x.trainIdx].pt for x in m])
    _, inl = cv2.findFundamentalMat(ua, ub, cv2.FM_RANSAC, 2.0, 0.999)
    if inl is None or inl.sum() < 30: return None
    ua, ub = ua[inl.ravel() > 0], ub[inl.ravel() > 0]
    def samp(im, uv):
        ba = cv2.blur(im.astype(np.float32), (5, 5)); x = np.clip(uv[:, 0].astype(int), 0, im.shape[1] - 1); y = np.clip(uv[:, 1].astype(int), 0, im.shape[0] - 1)
        return ba[y, x][:, ::-1]  # RGB
    A, B = samp(a, ua), samp(b, ub)
    ok = (A.min(1) > 15) & (B.min(1) > 15) & (A.max(1) < 245) & (B.max(1) < 245)
    if ok.sum() < 20: return None
    lr = np.log2(B[ok] / A[ok]); med = np.median(lr, 0)
    return {"a": fr[i]["stem"], "b": fr[j]["stem"], "t_a": round(float(T[i]), 2), "dt": round(float(T[j] - T[i]), 2), "n": int(ok.sum()),
            "gain_stops": round(float(med[1]), 3), "wb_R_over_G_stops": round(float(med[0] - med[1]), 3), "wb_B_over_G_stops": round(float(med[2] - med[1]), 3)}
random.seed(1)
consec = []
for i in range(0, len(fr) - 1, 6):
    js = [j for j in range(i + 1, len(fr)) if 0.8 <= T[j] - T[i] <= 1.5]
    if js:
        r = gain(i, js[0])
        if r: consec.append(r)
cand = [(i, j) for i in range(len(fr)) for j in range(i + 1, len(fr)) if T[j] - T[i] >= 8 and np.linalg.norm(C[i] - C[j]) < 0.4 and Fw[i] @ Fw[j] > np.cos(np.radians(20))]
rev = [r for r in (gain(i, j) for i, j in random.sample(cand, min(80, len(cand)))) if r]
def s(rs, k): a = np.abs([r[k] for r in rs]); return {"median_abs": round(float(np.median(a)), 3), "p90_abs": round(float(np.percentile(a, 90)), 3), "max_abs": round(float(a.max()), 3)}
cum = np.cumsum([r["gain_stops"] for r in consec])
out = {"consecutive_~1s": {"pairs": len(consec), **{k: s(consec, k) for k in ("gain_stops", "wb_R_over_G_stops", "wb_B_over_G_stops")}},
       "revisit_>=8s": {"pairs": len(rev), **{k: s(rev, k) for k in ("gain_stops", "wb_R_over_G_stops", "wb_B_over_G_stops")}},
       "note": "1 stop = 2x brightness; 0.1 stop = 7 %. Training uses a per-photo bilateral grid that absorbs this; the exported splat has no per-view correction.",
       "consec": consec, "revisit": rev}
(HERE / "exposure_drift.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ("consec", "revisit")}, indent=1))
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot([r["t_a"] for r in consec], cum, label="cumulative gain, chained ~1 s pairs (stops)")
ax.plot([r["t_a"] for r in consec], np.cumsum([r["wb_B_over_G_stops"] for r in consec]), label="cumulative B/G shift (stops)")
ax.set_xlabel("time (s)"); ax.legend(); fig.tight_layout(); fig.savefig(HERE / "exposure_drift.png", dpi=110)
