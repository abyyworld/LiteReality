"""Download a run_colmap.py result from the volume and write poses_colmap.json (largest model).

    uv run --env-file ~/LiteReality/.env python scripts/poses/colmap/fetch.py --run aliked_lg \
        --gpu-rate 0.80 --cpu 8 --mem-gib 32

Cost estimate: container seconds x (GPU $/h + Modal CPU/memory list prices), printed in notes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

BENCH = Path.home() / "LiteReality-scans" / "poses_bench" / "atrium"
CPU_PER_CORE_H = 0.0473   # Modal list price, $/physical-core-hour (cpu= counts physical cores)
MEM_PER_GIB_H = 0.0080    # $/GiB-hour


def get(remote: str, local: Path) -> None:
    env = {**os.environ, "MODAL_IMAGE_BUILDER_VERSION": "2025.06"}
    for attempt in range(6):
        local.unlink(missing_ok=True)
        r = subprocess.run(["modal", "volume", "get", "--force", "--env", "main", "litereality-poses",
                            remote, str(local)], env=env, capture_output=True, text=True)
        if r.returncode == 0 and local.exists():
            return
        print("retry", attempt, r.stderr[-300:])
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not download {remote}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="aliked_lg")
    ap.add_argument("--gpu-rate", type=float, default=0.80)
    ap.add_argument("--cpu", type=float, default=8)
    ap.add_argument("--mem-gib", type=float, default=32)
    ap.add_argument("--out", default=str(BENCH / "poses_colmap.json"))
    ap.add_argument("--note", default="", help="appended to notes")
    a = ap.parse_args()

    cache = Path(__file__).resolve().parent / ".cache" / a.run
    cache.mkdir(parents=True, exist_ok=True)
    raw_p = cache / "poses_raw.json"
    get(f"/colmap/{a.run}/poses_raw.json", raw_p)
    raw = json.loads(raw_p.read_text())
    models = raw["models"]
    best = models[0]
    cam = next(iter(best["cameras"].values()))
    secs = raw["container_seconds"]
    rate = a.gpu_rate + a.cpu * CPU_PER_CORE_H + a.mem_gib * MEM_PER_GIB_H
    cost = secs / 3600 * rate
    aliked = raw["features"] == "aliked"
    comp = "; ".join(f"model {m['id']}: {m['registered']} frames ({m['first']}..{m['last']}), "
                     f"{m['points3D']} points" for m in models)
    out = {
        "method": "colmap4_global_aliked_lightglue" if aliked else "colmap4_global_sift",
        "licence": ("COLMAP BSD-3-Clause / ALIKED BSD-3-Clause + LightGlue Apache-2.0 "
                    "(ONNX models from the COLMAP 3.13.0 release)" if aliked
                    else "COLMAP BSD-3-Clause / none (SIFT)"),
        "metric": False,
        "fx": float(cam["params"][0]),
        "frames": {n: {"c2w": p["c2w"]} for n, p in sorted(best["poses"].items())},
        "seconds": raw["seconds"],
        "gpu_seconds": secs,
        "cost_usd": round(cost, 3),
        "notes": (f"COLMAP {raw['colmap_image']} global_mapper on {raw['gpu']}; "
                  f"{'ALIKED n16rot (max ' + str(raw['max_features']) + ') + LightGlue' if aliked else 'SIFT'}; "
                  f"sequential overlap {raw['overlap']} quadratic + vocab-tree loop detection; one "
                  f"SIMPLE_RADIAL camera initialised f=1221.9, refined: params {cam['params']}. "
                  f"Largest of {len(models)} model(s): {comp}. Timing {raw['timing']}. "
                  f"gpu_seconds = whole container time; cost = that x ${rate:.3f}/h "
                  f"(GPU {a.gpu_rate} + {a.cpu} cores + {a.mem_gib} GiB, list prices). {a.note}").strip(),
    }
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "frames"}, indent=1))
    print("frames posed", len(out["frames"]), "->", a.out)


if __name__ == "__main__":
    main()
