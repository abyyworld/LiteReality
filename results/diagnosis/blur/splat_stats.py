"""(7) Gaussian statistics of the v2 snapshot (raw.spz.bin, what viewers get, uncut) and of v0/v1/v3.
Writes splat_stats.json, splat_scales.png."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from litereality_agent.models.splat import spz
HERE = Path(__file__).resolve().parent; RUN = Path.home() / "LiteReality/run/atrium_stray_q"
F_SCREEN = 1185.0  # px/rad: 70 deg vertical FOV on a 1660 device-px-tall canvas (13" laptop, DPR 2)
out = {}
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for v in ("v0", "v1", "v2", "v3"):
    s = spz.decode((RUN / v / "raw.spz.bin").read_bytes())
    sc = np.sort(s.scales, 1)  # ascending: thickness, mid, major
    major, mid, minor = sc[:, 2], sc[:, 1], sc[:, 0]
    op = s.opacities
    area = major * mid * op  # how much surface a splat paints (opacity-weighted footprint)
    w = area / area.sum()
    def wq(x, q):
        o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, q)])
    r = {"count": int(len(s)), "sh_degree": int(s.sh_degree),
         "major_sigma_mm": {"p10": round(float(np.percentile(major, 10) * 1000), 2), "p50": round(float(np.median(major) * 1000), 2), "p90": round(float(np.percentile(major, 90) * 1000), 2), "p99": round(float(np.percentile(major, 99) * 1000), 1)},
         "major_sigma_mm_area_weighted": {"p50": round(wq(major, 0.5) * 1000, 2), "p90": round(wq(major, 0.9) * 1000, 2)},
         "frac_major_gt_1cm": round(float(np.mean(major > 0.01)), 4), "frac_major_gt_3cm": round(float(np.mean(major > 0.03)), 4), "frac_major_gt_10cm": round(float(np.mean(major > 0.10)), 4),
         "area_share_major_gt_1cm": round(float(w[major > 0.01].sum()), 3), "area_share_major_gt_3cm": round(float(w[major > 0.03].sum()), 3),
         "anisotropy_major_over_mid_p50": round(float(np.median(major / np.maximum(mid, 1e-9))), 2),
         "opacity": {"p10": round(float(np.percentile(op, 10)), 3), "p50": round(float(np.median(op)), 3), "frac_lt_0.05": round(float(np.mean(op < 0.05)), 3), "frac_gt_0.9": round(float(np.mean(op > 0.9)), 3)},
         "screen_sigma_px_at_1.5m_area_weighted_p50": round(wq(major, 0.5) * F_SCREEN / 1.5, 2)}
    if v == "v2":
        idx = np.random.default_rng(0).choice(len(s), 400000, replace=False)
        tree = cKDTree(s.positions)
        d, _ = tree.query(s.positions[idx], k=2)
        r["nn_spacing_mm_p50"] = round(float(np.median(d[:, 1]) * 1000), 2)
        lo, hi = np.percentile(s.positions, 1, 0), np.percentile(s.positions, 99, 0)
        r["extent_m_p1_p99"] = [round(float(x), 2) for x in (hi - lo)]
        del tree
        ax[0].hist(np.log10(major * 1000), 100, color="#1f6feb", alpha=.7, label="major sigma"); ax[0].hist(np.log10(minor * 1000), 100, color="#888", alpha=.5, label="minor sigma")
        ax[0].set_xlabel("log10 sigma (mm)"); ax[0].legend(); ax[0].set_title(f"v2 raw.spz.bin, {len(s):,} splats")
        ax[1].hist(op, 50, color="#1f6feb"); ax[1].set_xlabel("opacity")
    out[v] = r
    print(v, json.dumps(r))
    del s
fig.tight_layout(); fig.savefig(HERE / "splat_scales.png", dpi=110)
(HERE / "splat_stats.json").write_text(json.dumps(out, indent=1))
