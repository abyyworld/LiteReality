"""Fairness check for the trainer comparison: no fixed eval frame (results/stray_quality/eval_frames.txt)
is in ANY training set that was scored, read from the dataset tars themselves (the exact photos the
trainers saw), not from the scan folders.

* full walk (gsplat_v0, gsplat_nodepth5k, every LichtFeld full-walk run): run/atrium_trainers/dataset_q206.tar.gz
  (photos named after ~/LiteReality-scans/stray_atrium_q206/frame_XXXXX.json -> stray.row)
* RealityScan piece (gsplat and LichtFeld, ARKit and RS poses): ~/LiteReality-scans/rs_stray/*_w*.tar.gz
  (photos named after rs_stray/rs_input/selection.json -> row)

Times come from the raw recording's odometry.csv, so the gap is in seconds of the walk.

    uv run --frozen python scripts/compare_trainers/check_fairness.py   # -> results/trainers/fairness.json
"""

from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCANS = Path("~/LiteReality-scans").expanduser()
EVAL = REPO / "results/stray_quality/eval_frames.txt"
ODOM = SCANS / "stray_atrium_raw/901a305fdf/odometry.csv"


def eval_rows() -> list[int]:
    out = []
    for line in EVAL.read_text().splitlines():
        tok = line.split("#", 1)[0].split()
        if tok:
            out.append(int(tok[0]))
    return out


def row_times() -> dict[int, float]:
    with ODOM.open() as f:
        rd = csv.reader(f)
        next(rd)
        rows = [(int(r[1]), float(r[0])) for r in rd if r]
    return dict(rows)


def tar_images(path: Path) -> list[str]:
    with tarfile.open(path) as t:
        return sorted(Path(m.name).name for m in t.getmembers()
                      if m.isfile() and "images/" in m.name and m.name.endswith(".jpg"))


def check(name: str, tar: Path, name_to_row: dict[str, int], ev: list[int], t: dict[int, float]):
    imgs = tar_images(tar)
    missing = [n for n in imgs if n not in name_to_row]
    rows = sorted(name_to_row[n] for n in imgs if n in name_to_row)
    ev_set = set(ev)
    overlap = sorted(set(rows) & ev_set)
    gaps = [min(abs(t[r] - t[e]) for r in rows) for e in ev]
    rgaps = [min(abs(r - e) for r in rows) for e in ev]
    return {"dataset": str(tar).replace(str(Path.home()), "~"), "train_photos": len(imgs),
            "unmapped_photos": missing, "eval_rows_in_train": overlap,
            "min_row_gap": min(rgaps), "min_time_gap_s": round(min(gaps), 3),
            "median_time_gap_s": round(sorted(gaps)[len(gaps) // 2], 3), "used_by": name}


def main() -> int:
    ev = eval_rows()
    t = row_times()
    res = {"eval_frames": len(ev), "eval_source": str(EVAL.relative_to(REPO)), "sets": {}}

    q206 = SCANS / "stray_atrium_q206"
    m = {f"{p.stem}.jpg": int(json.loads(p.read_text())["stray"]["row"])
         for p in q206.glob("frame_*.json")}
    res["sets"]["full_walk_q206"] = check(
        "gsplat_v0, gsplat_nodepth5k, lfs_* (full walk)",
        REPO / "run/atrium_trainers/dataset_q206.tar.gz", m, ev, t)

    sel = json.loads((SCANS / "rs_stray/rs_input/selection.json").read_text())["frames"]
    m = {n: int(v["row"]) for n, v in sel.items()}
    for v in ("rs_w960", "arkit_w960", "rs_w1920", "arkit_w1920"):
        res["sets"][f"rs_piece_{v}"] = check("RealityScan-piece runs", SCANS / f"rs_stray/{v}.tar.gz",
                                             m, ev, t)
    # Disclosure: the eval frames were part of RealityScan's SfM input (alignment only).
    in_sfm = sorted(int(v["row"]) for v in sel.values() if v.get("eval"))
    res["eval_rows_in_realityscan_sfm_input"] = {"count": len(in_sfm), "rows": in_sfm}

    ok = all(not s["eval_rows_in_train"] and not s["unmapped_photos"] for s in res["sets"].values())
    res["fair"] = ok
    out = REPO / "results/trainers/fairness.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
