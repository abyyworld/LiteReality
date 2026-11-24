"""Collect the STEP 3 browser scores (results/eyeful/spark/*.json) into results/eyeful/step3_spark.json
and a markdown table (results/eyeful/step3_spark.md).

    uv run --frozen python scripts/eyeful/spark_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "eyeful"
SPARK = RESULTS / "spark"
RUNS = ("A_30000", "B_5000", "B_1250")
ORDER = ("call", "std3", "blur", "limits", "matched")


def load(stem: str) -> dict | None:
    p = SPARK / f"{stem}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def main() -> int:
    out: dict = {"views": None, "runs": {}}
    lines = ["| run | SPZ | Spark settings | views | trainer PSNR (JPEG) | Spark PSNR | gap dB "
             "| Spark vs trainer PSNR | SSIM trainer / Spark | LPIPS trainer / Spark "
             "| coverage |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in RUNS:
        check = json.loads((SPARK / f"{run}_trainer_check.json").read_text())
        entry = {"trainer_reported_all71": {k: round(v, 4) for k, v in
                                            check["trainer_reported"].items()
                                            if k in ("psnr", "ssim", "lpips")},
                 "trainer_jpeg_recomputed_all71": check["trainer_jpeg_recomputed"],
                 "spark": {}}
        stems = [(f"{sh}_{preset}", f"{run}_splats_{sh}_metal_{preset}", preset)
                 for sh in ("sh3", "sh1") for preset in ORDER]
        # Single-factor and renderer checks: the call with only maxPixelRadius lifted, and the
        # call in SwiftShader (software WebGL, every earlier score's renderer) on a few views.
        stems += [("sh3_call_maxPixelRadius1e5", f"{run}_splats_sh3_metal_call_r1e5",
                   "call + maxPixelRadius 1e5 only"),
                  ("sh3_call_swiftshader", f"{run}_splats_sh3_swiftshader_call",
                   "call, SwiftShader")]
        for key, stem, preset in stems:
            doc = load(stem)
            if doc is None:
                continue
            s = doc["summary"]
            sh = key.split("_")[0]
            entry["spark"][key] = {
                "views": doc["views"], "spark_options": doc["spark_options"],
                "spark": {k: s[f"spark_{k}"] for k in ("psnr", "ssim", "lpips")},
                "trainer_same_views": {k: s[f"trainer_{k}"] for k in ("psnr", "ssim", "lpips")},
                "gap_psnr_db": s["gap_psnr_db"], "gap_ssim": s["gap_ssim"],
                "gap_lpips": s["gap_lpips"], "spark_vs_trainer_psnr": s["spark_vs_trainer_psnr"],
                "coverage": s["coverage"], "render_s": doc["render_s"],
                "settle_ms_mean": round(sum(f["ms"] for f in doc["settle_frames"])
                                        / max(1, len(doc["settle_frames"]))),
                "renderer": doc["renderer"], "spz_mb": doc["spz_mb"]}
            out["views"] = max(out["views"] or 0, doc["views"])
            lines.append(
                f"| {run} | {sh} | {preset} | {doc['views']} | {s['trainer_psnr']:.2f} "
                f"| {s['spark_psnr']:.2f} | {s['gap_psnr_db']:+.2f} "
                f"| {s['spark_vs_trainer_psnr']:.2f} | {s['trainer_ssim']:.3f} / "
                f"{s['spark_ssim']:.3f} | {s['trainer_lpips']:.3f} / {s['spark_lpips']:.3f} "
                f"| {s['coverage']:.3f} |")
        out["runs"][run] = entry
    (RESULTS / "step3_spark.json").write_text(json.dumps(out, indent=1) + "\n")
    text = "\n".join(lines) + "\n"
    (RESULTS / "step3_spark.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
