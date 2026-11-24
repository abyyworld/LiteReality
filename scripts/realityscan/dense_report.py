"""One dense RealityScan run (deploy/modal/realityscan/app.py::align_dir, fetched from the Volume's
runs/<tag>) -> what it reconstructed and, for the Stray walk, how its poses agree with ARKit.

    uv run --frozen python scripts/realityscan/dense_report.py <run dir> <frames dir> [--arkit]

<run dir> holds colmap.tar.gz, xmp.tar.gz, rs_log.txt, run.json (unpacked here on first use);
<frames dir> the prep_dense.py output (selection.json). Components come from RealityScan's own
XMP export (xcr:ComponentId per image) and the log's "Finalizing N components". With --arkit
(Stray only) the largest component's cameras are compared with the raw ARKit odometry of the
same rows after a Sim(3) fit of the centres (compare_poses.py's measures: centre error, rotation
error with the centre fit and with the orientation fit, 1 s relative errors, fx vs ARKit).
Writes <run dir>/report.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_poses import umeyama  # noqa: E402
from rs_common import FLIP, angle_deg, read_model, stats  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
RAW = Path("~/LiteReality-scans/stray_atrium_raw/901a305fdf").expanduser()


def unpack(run: Path) -> Path:
    for name in ("colmap.tar.gz", "xmp.tar.gz"):
        tgt = run / name.split(".")[0]
        if (run / name).exists() and not tgt.exists():
            with tarfile.open(run / name) as tf:
                tf.extractall(run)
    sp = next((p.parent for p in run.rglob("images.txt")), None)
    if sp is None:
        raise SystemExit(f"{run}: no images.txt")
    return sp


def xmp_components(run: Path) -> dict[str, str]:
    out = {}
    for p in sorted((run / "xmp").glob("*.xmp")):
        m = re.search(r'ComponentId="([^"]+)"', p.read_text(errors="replace"))
        if m is None:
            m = re.search(r"<xcr:ComponentId>([^<]+)<", p.read_text(errors="replace"))
        out[p.stem] = m.group(1) if m else ""
    return out


def arkit_rows() -> dict[int, tuple[np.ndarray, np.ndarray, float, float]]:
    from litereality_agent.models.capture import stray as S
    rec = S.read_recording(RAW)
    return {f.row: (f.c2w[:3, :3] @ FLIP, f.c2w[:3, 3], float(f.K[0, 0]), f.timestamp)
            for f in rec.frames}


def compare_arkit(imgs, cams, sel) -> dict:
    ar = arkit_rows()
    names = [n for n in sorted(imgs) if n in sel]
    rows = [sel[n]["row"] for n in names]
    src = np.array([imgs[n].C for n in names])
    dst = np.array([ar[r][1] for r in rows])
    s, R, t = umeyama(src, dst)
    Rs = [imgs[n].R for n in names]
    Ra = [ar[r][0] for r in rows]
    pos = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
    rot_c = [angle_deg(R @ a, b) for a, b in zip(Rs, Ra)]
    U, _, Vt = np.linalg.svd(sum(b @ a.T for a, b in zip(Rs, Ra)))
    Ro = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    rot_o = [angle_deg(Ro @ a, b) for a, b in zip(Rs, Ra)]
    # 1 s relative motion
    ts = np.array([ar[r][3] for r in rows])
    rpe_r, rpe_p = [], []
    for i in range(len(names)):
        j = int(np.searchsorted(ts, ts[i] + 1.0))
        if j >= len(names) or abs(ts[j] - ts[i] - 1.0) > 0.1:
            continue
        dr = Rs[i].T @ Rs[j]
        da = Ra[i].T @ Ra[j]
        rpe_r.append(angle_deg(dr, da))
        rpe_p.append(float(np.linalg.norm(s * R @ (src[j] - src[i]) - (dst[j] - dst[i]))))
    fx_ar = float(np.median([ar[r][2] for r in rows]))
    cam_fx = {k: float(c.K[0, 0]) for k, c in cams.items()}
    worst = sorted(zip(pos, names), reverse=True)[:5]
    return {"compared": len(names), "scale_rs_to_m": float(s),
            "arkit_extent_m": float(np.linalg.norm(dst - dst.mean(0), axis=1).max()),
            "pos_err_mm": {k: (v * 1000 if k != "n" else v) for k, v in stats(pos).items()},
            "rot_err_deg_centrefit": stats(rot_c), "rot_err_deg_rotfit": stats(rot_o),
            "centre_vs_orientation_frame_deg": angle_deg(Ro, R),
            "rpe_1s_rot_deg": stats(rpe_r) if rpe_r else None,
            "rpe_1s_pos_mm": ({k: (v * 1000 if k != "n" else v) for k, v in stats(rpe_p).items()}
                              if rpe_p else None),
            "fx_arkit_median": fx_ar,
            "fx_rs": cam_fx, "fx_rel_diff": {k: v / fx_ar - 1 for k, v in cam_fx.items()},
            "worst_pos": [[n, round(float(p) * 1000, 1)] for p, n in worst],
            "T_rs_to_arkit_raw": {"s": float(s), "R": R.tolist(), "t": t.tolist()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("frames", type=Path)
    ap.add_argument("--arkit", action="store_true")
    a = ap.parse_args()
    run = a.run.expanduser()
    sp = unpack(run)
    sel = json.loads((a.frames.expanduser() / "selection.json").read_text())["frames"]
    imgs, cams, pts = read_model(sp)
    meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    log = (run / "rs_log.txt").read_text(errors="replace") if (run / "rs_log.txt").exists() else ""
    m = re.findall(r"Finalizing (\d+) components", log)
    recon_s = re.findall(r"Reconstruction completed in ([\d.]+) seconds", log)
    comps = xmp_components(run) if (run / "xmp").exists() else {}
    sizes = Counter(v for v in comps.values() if v)
    n_in = len(sel)
    rep = {"run": str(run), "tag": meta.get("tag"), "input_frames": n_in,
           "registered_largest": len(imgs), "largest_share": round(len(imgs) / n_in, 4),
           "components_log": [int(v) for v in m],
           "xmp_files": len(comps), "xmp_components": len(sizes),
           "xmp_component_sizes": sorted(sizes.values(), reverse=True)[:15],
           "xmp_registered_total": sum(sizes.values()),
           "eval_in_largest": sum(1 for n in imgs if sel.get(n, {}).get("eval")),
           "eval_total": sum(1 for v in sel.values() if v.get("eval")),
           "n_cameras": len(cams), "points3D": int(len(pts["xyz"])),
           "cameras": {k: {"model": c.model, "w": c.w, "h": c.h,
                           "params": [round(float(v), 6) for v in c.params]}
                       for k, c in list(cams.items())[:3]},
           "seconds": meta.get("seconds"), "reconstruction_s": [float(v) for v in recon_s],
           "settings": {k: meta.get(k) for k in ("focal35", "distortion_model", "extra", "post")}}
    # Which stretches of the walk are missing from the largest component (time order).
    order = sorted(sel, key=lambda n: sel[n]["time_s"])
    inside = np.array([n in imgs for n in order])
    gaps, start = [], None
    for i, v in enumerate(inside):
        if not v and start is None:
            start = i
        if (v or i == len(order) - 1) and start is not None:
            end = i if v else i + 1
            gaps.append([sel[order[start]]["time_s"], sel[order[end - 1]]["time_s"], end - start])
            start = None
    rep["missing_stretches_s"] = [g for g in gaps if g[2] >= 3][:30]
    if a.arkit:
        rep["vs_arkit"] = compare_arkit(imgs, cams, sel)
    (run / "report.json").write_text(json.dumps(rep, indent=1))
    show = {k: v for k, v in rep.items() if k not in ("missing_stretches_s",)}
    if "vs_arkit" in show:
        va = dict(show["vs_arkit"])
        va.pop("T_rs_to_arkit_raw")
        show["vs_arkit"] = va
    print(json.dumps(show, indent=1))
    print("missing stretches (t0, t1, frames):", rep["missing_stretches_s"][:12])


if __name__ == "__main__":
    main()
