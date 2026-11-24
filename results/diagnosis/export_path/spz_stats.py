"""Static look at what the SPZ export keeps of the trained v2 room (no GPU, no browser).

    uv run --frozen python results/diagnosis/export_path/spz_stats.py run/atrium_stray_q/v2/raw.spz.bin
"""
import gzip, json, struct, sys
from pathlib import Path
import numpy as np
from litereality_agent.models.splat import spz

path = Path(sys.argv[1])
data = path.read_bytes()
raw = gzip.decompress(data)
magic, version, n, degree, bits, flags, _ = struct.unpack_from("<IIIBBBB", raw)
s = spz.decode(data)
out = {"file": str(path), "n": n, "version": version, "sh_degree": degree, "fractional_bits": bits,
       "flag_antialiased": bool(flags & 1)}
# colour bytes (DC): saturation at 0/255 means the trained colour was outside SPZ's range
off = 16 + n * 9 + n
col = np.frombuffer(raw, np.uint8, n * 3, off)
out["dc_byte_saturated_frac"] = float(((col == 0) | (col == 255)).mean())
off += n * 3
sc = np.frombuffer(raw, np.uint8, n * 3, off)
out["scale_byte_saturated_frac"] = float(((sc == 0) | (sc == 255)).mean())
off += n * 3 + n * 4
if degree:
    shb = np.frombuffer(raw, np.uint8, n * spz.sh_dim(degree) * 3, off)
    out["sh_byte_saturated_frac"] = float(((shb == 0) | (shb >= 248)).mean())
    vals = (shb.astype(np.float32) - 128) / 128
    out["sh_abs_mean"] = float(np.abs(vals).mean())
    out["sh_zero_frac"] = float((shb == 128).mean())
sig = s.scales.max(1)
out["scale_max_sigma_m_percentiles"] = {p: float(np.percentile(sig, p)) for p in (5, 25, 50, 75, 95, 99)}
smin = s.scales.min(1)
out["scale_min_sigma_m_median"] = float(np.median(smin))
out["opacity_percentiles"] = {p: float(np.percentile(s.opacities, p)) for p in (5, 25, 50, 75, 95)}
# quantisation steps, in the units a viewer sees
out["quant_steps"] = {
    "position_m": 1 / (1 << bits),
    "colour_grey_levels_per_step": float(spz.SH_C0 / (spz.COLOR_SCALE * 255) * 255),
    "scale_rel_per_step": float(np.exp(1 / 16) - 1),
    "opacity_per_step": 1 / 255,
    "sh1_step": 8 / 128, "sh23_step": 16 / 128,
}
# how far splats are from the capture path (distance from camera matters for pixel footprint)
bundle = json.loads((path.parent / "splat.json").read_text())
cap = np.asarray(bundle.get("capture") or [[0, 0, 0]], float)
d = np.full(n, np.inf, np.float32)
for i in range(0, n, 200_000):
    p = s.positions[i:i + 200_000]
    d[i:i + 200_000] = np.sqrt(((p[:, None, :] - cap[None, :, :]) ** 2).sum(-1)).min(1)
out["dist_to_nearest_capture_m_percentiles"] = {p: float(np.percentile(d, p)) for p in (5, 25, 50, 75, 95)}
# footprint in pixels at the nearest capture spot for a 1440-wide photo (fx ~1001) and a
# 1920-wide browser at 70 deg vertical FOV on 16:9 (fy = 540/tan(35deg) ~ 771 px)
for name, f in (("train_1440", 1335.5 * 1440 / 1920), ("browser_1080p_fov70", 540 / np.tan(np.radians(35)))):
    px = sig / np.maximum(d, 0.05) * f
    out[f"max_sigma_px_at_capture_{name}_percentiles"] = {p: float(np.percentile(px, p)) for p in (25, 50, 75, 95)}
print(json.dumps(out, indent=1))
